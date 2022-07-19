#include <ice-cream.hpp>

#include "Canvas/CodeGen/PyTorchCodeGen.hpp"
#include "Canvas/Core/Variable.hpp"
#include "Canvas/Primitives/Factory.hpp"

// #define CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SHAPES_IN_PYTHON


namespace canvas {

/// Translate into PyTorch-style variable, return "1" if the variable is empty.
static std::string TorchStyleVariable(const Variable& var) {
    if (not var.SatisfyAssumption()) {
        std::stringstream ss;
        ss << "Find a variable which does not satisfy the requirement: " << var;
        throw CanNotApplyPyTorchCodeGen(ss.str());
    }
    static_assert(Variable::kStaticVarCount == 4);
    static const char* info[Variable::kStaticVarCount] =
            {"self.g", "self.c", "self.h", "self.w"};
    assert(var.IsStatic());
    return var.Format(info, " * ", " // ", "None[", "]");
}

/// Translate into PyTorch-style shape, skip if the variable inside is empty.
static std::string TorchStyleShape(const Shape& shape) {
    bool displayed = false;
    std::stringstream ss;
    for (const auto& dim: shape.Continuous()) {
        assert(not dim.Empty());
        ss << (displayed ? ", " : "") << TorchStyleVariable(dim), displayed = true;
    }
    return ss.str();
}

static constexpr const char* TorchStyleActivationFunctionName(ActivationType type) {
    switch (type) {
        case GeLU: return "F.gelu";
        case ReLU: return "torch.relu";
        case Sigmoid: return "torch.sigmoid";
        case TanH: return "torch.tanh";
    }
    return "";
}

static constexpr const char* TorchStyleFoldName(FoldType type) {
    switch (type) {
        case FoldAvg: return "mean";
        case FoldMax: return "max";
    }
    return "";
}

static constexpr const char* TorchStyleFoldSuffix(FoldType type) {
    switch (type) {
        case FoldAvg: return "";
        case FoldMax: return "[0]";
    }
    return "";
}

static constexpr const char* TorchStyleElementWiseFunctionPrefix(ElementWiseType type) {
    switch (type) {
        case Abs:  return "torch.abs(";
        case Exp:  return "torch.exp(";
        case Neg:  return "-";
        case Sin:  return "torch.sin(";
        case Sqrt: return "torch.sqrt(torch.abs(";
        case Sqr:  return "torch.pow(";
    }
    return "";
}

static constexpr const char* TorchStyleElementWiseFunctionSuffix(ElementWiseType type) {
    switch (type) {
        case Abs:  return ")"; // NOLINT(bugprone-branch-clone)
        case Exp:  return ")";
        case Neg:  return "";
        case Sin:  return " * 1.5707963)";
        case Sqrt: return "))";
        case Sqr:  return ", 2)";
    }
    return "";
}

static std::string TorchStyleBroadcasting(const std::vector<Variable>& prefix,
                                          const Variable& multiplier,
                                          const Variable& pi,
                                          const std::vector<Variable>& suffix) {
    bool displayed = false;
    std::stringstream ss;
    auto Display = [&ss, &displayed](const std::string& info) {
        ss << (displayed ? ", " : "") << info, displayed = true;
    };
    for (const auto& var: prefix)
        Display(TorchStyleVariable(var));
    Display(TorchStyleVariable(multiplier));
    if (not pi.Empty())
        Display(TorchStyleVariable(pi));
    for (const auto& var: suffix)
        Display(TorchStyleVariable(var));
    return ss.str();
}

static std::string PyTorchReshapeToNCHW(CodeGen* gen, VarMap& var_map, const TensorSP& in, const TensorSP& out) {
    assert(in->shape.IsChannelSpatial());
    auto channel = in->shape.Channel();
    auto spatial = in->shape.Spatial();
    int gckk_count = 0;
    gckk_count += not channel->G().Empty();
    gckk_count += not channel->C().Empty();
    gckk_count += not channel->KH().Empty();
    gckk_count += not channel->KW().Empty();
    if (gckk_count != 1 or spatial->H().Empty() or spatial->W().Empty()) {
        gen->Write() << var_map[out]
                     << " = " << var_map[in]
                     << ".view(self.n, "
                     << TorchStyleVariable(channel->Pi()) << ", "
                     << TorchStyleVariable(spatial->H()) << ", " << TorchStyleVariable(spatial->W())
                     << ")"
                     << std::endl;
        return var_map[out];
    } else {
        return var_map[in];
    }
}

static std::string PyTorchBroadcastExpr(const std::string& lhs, const std::string& rhs, const PrimitiveSP& p) {
    auto broadcast = DynamicCast<BroadcastPrimitive>(p);
    assert(broadcast);
    switch (broadcast->type) {
        case BMax:
            return "torch.maximum(" + lhs + ", " + rhs + ")";
        default:
            return lhs + " " + broadcast->TypeToSign() + " " + rhs;
    }
}

void PyTorchInitTranslator::operator () (CodeGen* gen, const PrimitiveSP& p) {
    // Create variables' mapping.
    for (const auto& t: p->ins)
        assert(var_map.Count(t));
    for (const auto& t: p->outs) {
        assert(not var_map.Count(t));
        var_map[t] = "t_" + std::to_string(var_map.TensorSize());
    }

    // Handled different operators.
    auto primitive_var = (var_map[p] = "p_" + std::to_string(var_map.PrimitiveSize()));
    gen->Write() << "# " << p->name << ": " << primitive_var << std::endl;
    if (auto conv = DynamicCast<ConvolutionPrimitive>(p)) {
        auto& in_shape = p->ins[0]->shape;
        auto& out_shape = p->outs[0]->shape;
        assert(in_shape.IsChannelSpatial() and out_shape.IsChannelSpatial());
        int ph = conv->dh * (conv->kh - 1) / 2;
        int pw = conv->dw * (conv->kw - 1) / 2;
        auto groups = conv->depth_wise ? in_shape.Channel()->C() : in_shape.Channel()->G();
        gen->Write() << "self." << primitive_var
                     << " = nn.Conv2d("
                     << TorchStyleVariable(in_shape.Channel()->Pi()) << ", "
                     << TorchStyleVariable(out_shape.Channel()->Pi()) << ", "
                     << "(" << conv->kh << ", " << conv->kw << "), "
                     << "dilation=(" << conv->dh << ", " << conv->dw << "), "
                     << "padding=(" << ph << ", " << pw << "), "
                     << "groups=" << TorchStyleVariable(groups) << ", "
                     << "bias=False)"
                     << std::endl;
    } else if (auto fc = DynamicCast<FCPrimitive>(p)) {
        auto& in_shape = p->ins[0]->shape;
        auto& out_shape = p->outs[0]->shape;
        assert(in_shape.IsChannelSpatial() and out_shape.IsChannelSpatial());
        gen->Write() << "self." << primitive_var
                     << " = nn.Conv2d("
                     << TorchStyleVariable(in_shape.Channel()->Pi()) << ", " // Input channels.
                     << TorchStyleVariable(out_shape.Channel()->Pi()) << ", " // Output channels.
                     << "1, padding=0, "
                     << "groups=" << TorchStyleVariable(in_shape.Channel()->G()) << ", "
                     << "bias=False)"
                     << std::endl;
    } else if (auto mix = DynamicCast<MixPrimitive>(p)) {
        std::stringstream shape_ss;
        auto fan_in = Variable();
        for (const auto& index: mix->indices) {
            shape_ss << TorchStyleVariable(mix->ins[0]->shape[index]) << ", ";
            fan_in = fan_in * mix->ins[0]->shape[index];
        }
        for (const auto& index: mix->indices)
            shape_ss << TorchStyleVariable(mix->outs[0]->shape[index]) << ", ";
        gen->Write() << "self." << primitive_var << "_w"
                     << " = nn.Parameter(torch.ones("
                     << "(" << shape_ss.str() << ")"
                     << "), requires_grad=True)"
                     << std::endl;
        gen->Write() << "bound = math.sqrt(3.0 / ("
                     << TorchStyleVariable(fan_in)
                     << "))"
                     << std::endl;
        gen->Write() << "nn.init.uniform_("
                     << "self." << primitive_var << "_w"
                     << ", a=-bound, b=bound)" << std::endl;
    } else if (auto shift = DynamicCast<ShiftPrimitive>(p)) {
        int k = shift->k;
        for (const auto& index: shift->indices)
            gen->Write() << "self." << primitive_var << "_" << index.d << "_" << index.k
                         << " = random.randint(-" << k << ", " << k << ")" << std::endl;
    } else if (auto scale = DynamicCast<ScalePrimitive>(p)) {
        auto sorted = scale->indices;
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& lhs, const auto& rhs) -> bool {
                      return lhs.d == rhs.d ? lhs.k < rhs.k : lhs.d < rhs.d;
                  });
        auto t_shape = p->ins[0]->shape;
        std::stringstream shape_str;
        int next = 0;
        for (int d = 0; d < 2; ++ d) {
            for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
                auto index = Shape::Index(d, k);
                if (t_shape[index].Empty())
                    continue;
                if (next < sorted.size() and index == sorted[next])
                    shape_str << ", " << TorchStyleVariable(t_shape[index]), next ++;
                else
                    shape_str << ", 1";
            }
        }
        gen->Write() << "self." << primitive_var << "_w"
                     << " = nn.Parameter(torch.ones("
                     << "(1" << shape_str.str() << ",)"
                     << "), requires_grad=True)"
                     << std::endl;
        gen->Write() << "nn.init.trunc_normal_("
                     << "self." << primitive_var << "_w"
                     << ", std=.02)" << std::endl;
    } else if (DynamicCast<InputPrimitive>(p) or
            DynamicCast<ActivationPrimitive>(p) or
            DynamicCast<BroadcastPrimitive>(p) or
            DynamicCast<ElementWisePrimitive>(p) or
            DynamicCast<FoldPrimitive>(p) or
            DynamicCast<GroupPrimitive>(p) or
            DynamicCast<MatrixMultiplicationPrimitive>(p) or
            DynamicCast<SoftmaxPrimitive>(p) or
            DynamicCast<OutputPrimitive>(p) or
            DynamicCast<UnfoldPrimitive>(p)) {
        gen->Write() << "pass" << std::endl;
    } else {
        CriticalError("Unknown or unimplemented primitive " +
                      p->name + " while generating initializers");
    }
}

void PyTorchForwardTranslator::operator () (CodeGen* gen, const PrimitiveSP& p) {
    for (const auto& t: br::join(p->ins, p->outs))
        assert(var_map.Count(t));

    // Handle different operators.
    assert(var_map.Count(p));
    auto primitive_var = var_map[p];
    gen->Write() << "# " << p->name << ": " << primitive_var << std::endl;

    if (auto activation = DynamicCast<ActivationPrimitive>(p)) {
        gen->Write() << var_map[activation->outs[0]]
                     << " = "
                     << TorchStyleActivationFunctionName(activation->type)
                     << "(" << var_map[activation->ins[0]] << ")"
                     << std::endl;
    } else if (auto broadcast = DynamicCast<BroadcastPrimitive>(p)) {
        if (broadcast->aligned) { // Not really broadcasting.
            gen->Write() << var_map[broadcast->outs[0]]
                         << " = "
                         << PyTorchBroadcastExpr(var_map[broadcast->ins[0]], var_map[broadcast->ins[1]], broadcast)
                         << std::endl;
        } else { // Broadcasting.
            gen->Write() << var_map[broadcast->outs[0]] << "_lhs"
                         << " = " << var_map[broadcast->ins[0]]
                         << ".view(self.n, "
                         << TorchStyleBroadcasting(broadcast->prefix, {}, broadcast->lhs_pi, broadcast->suffix)
                         << ")"
                         << std::endl;
            gen->Write() << var_map[broadcast->outs[0]] << "_rhs"
                         << " = " << var_map[broadcast->ins[1]]
                         << ".view(self.n, "
                         << TorchStyleBroadcasting(broadcast->prefix, broadcast->multiplier, broadcast->lhs_pi, broadcast->suffix)
                         << ")"
                         << std::endl;
            auto lhs = var_map[broadcast->outs[0]] + "_lhs", rhs = var_map[broadcast->outs[0]] + "_rhs";
            gen->Write() << var_map[broadcast->outs[0]]
                         << " = "
                         << PyTorchBroadcastExpr(lhs, rhs, broadcast)
                         << std::endl;
            gen->Write() << var_map[broadcast->outs[0]]
                         << " = " << var_map[broadcast->outs[0]]
                         << ".view(self.n, "
                         << TorchStyleShape(broadcast->outs[0]->shape)
                         << ")"
                         << std::endl;
        }
    } else if (auto conv = DynamicCast<ConvolutionPrimitive>(p)) {
        auto reference = PyTorchReshapeToNCHW(gen, var_map, conv->ins[0], conv->outs[0]);
        gen->Write() << var_map[conv->outs[0]]
                     << " = self." << primitive_var
                     << "(" << reference << ")"
                     << std::endl;
        gen->Write() << var_map[conv->outs[0]]
                     << " = " << var_map[conv->outs[0]]
                     << ".view(self.n, "
                     << TorchStyleShape(conv->outs[0]->shape)
                     << ")"
                     << std::endl;
    } else if (auto element_wise = DynamicCast<ElementWisePrimitive>(p)) {
        gen->Write() << var_map[element_wise->outs[0]]
                     << " = "
                     << TorchStyleElementWiseFunctionPrefix(element_wise->type)
                     << var_map[element_wise->ins[0]]
                     << TorchStyleElementWiseFunctionSuffix(element_wise->type)
                     << std::endl;
    } else if (auto fc = DynamicCast<FCPrimitive>(p)) {
        auto reference = PyTorchReshapeToNCHW(gen, var_map, fc->ins[0], fc->outs[0]);
        gen->Write() << var_map[fc->outs[0]]
                     << " = self." << primitive_var
                     << "(" << reference << ")"
                     << std::endl;
        gen->Write() << var_map[fc->outs[0]]
                     << " = " << var_map[fc->outs[0]]
                     << ".view(self.n, "
                     << TorchStyleShape(fc->outs[0]->shape)
                     << ")"
                     << std::endl;
    } else if (auto fold = DynamicCast<FoldPrimitive>(p)) {
        auto reference_shape = fold->ins[0]->shape;
        gen->Write() << var_map[fold->outs[0]]
                     << " = " << var_map[fold->ins[0]];
        assert(not fold->indices.empty());
        for (const auto& index: fold->indices) {
            if (reference_shape[index].Empty())
                continue;
            int dim = reference_shape.GetRelativeIndex(index) + 1;
            gen->Write(false) << "." << TorchStyleFoldName(fold->type)
                              << "(" << dim << ")"
                              << TorchStyleFoldSuffix(fold->type);
            reference_shape[index].Reset();
        }
        gen->Write(false) << std::endl;
    } else if (auto group = DynamicCast<GroupPrimitive>(p)) {
        auto old_shape_str = TorchStyleShape(group->ins[0]->shape);
        auto new_shape_str = TorchStyleShape(group->outs[0]->shape);
        if (old_shape_str != new_shape_str) {
            gen->Write() << var_map[group->outs[0]]
                         << " = " << var_map[group->ins[0]]
                         << ".view(self.n, "
                         << new_shape_str
                         << ")"
                         << std::endl;
        } else {
            gen->Write() << var_map[group->outs[0]]
                         << " = " << var_map[group->ins[0]]
                         << std::endl;
        }
    } else if (auto input = DynamicCast<InputPrimitive>(p)) {
        // The input `x` must be in the shape of [N, C, H, W].
        gen->Write() << var_map[input->outs[0]]
                     << " = x"
                     << std::endl;
        gen->Write() << "self.n = "
                     << var_map[input->outs[0]] << ".size(0)"
                     << std::endl;
        gen->Write() << "assert "
                     << "(self.n, self.c, self.h, self.w)"
                     << " == "
                     << "tuple(" << var_map[input->outs[0]] << ".size())"
                     << std::endl;
    } else if (auto bmm = DynamicCast<MatrixMultiplicationPrimitive>(p)) {
        auto ReshapeAndTranspose = [&](const char* suffix, const TensorSP& t, bool transpose) {
            gen->Write() << var_map[t] << "_" << suffix
                         << " = "
                         << var_map[t]
                         << ".view(self.n, "
                         << TorchStyleVariable(t->shape.dims[0]->Pi()) << ", "
                         << TorchStyleVariable(t->shape.dims[1]->Pi())
                         << ")";
            if (transpose)
                gen->Write(false) << ".transpose(1, 2)";
            gen->Write() << std::endl;
        };
        ReshapeAndTranspose("lhs", bmm->ins[0], bmm->transpose_lhs);
        ReshapeAndTranspose("rhs", bmm->ins[1], bmm->transpose_rhs);
        gen->Write() << var_map[bmm->outs[0]]
                     << " = "
                     << "torch.bmm("
                     << var_map[bmm->ins[0]] << "_lhs, "
                     << var_map[bmm->ins[1]] << "_rhs)"
                     << ".view(self.n, "
                     << TorchStyleShape(bmm->outs[0]->shape)
                     << ")";
        auto scale = bmm->ins[0]->shape.dims[not bmm->transpose_lhs]->Pi();
        if (not scale.Empty())
            gen->Write(false) << " / math.sqrt(" << TorchStyleVariable(scale) << ")";
        gen->Write(false) << std::endl;
    } else if (auto mix = DynamicCast<MixPrimitive>(p)) {
        std::set<Shape::Index> mapping_indices;
        for (const auto& index: mix->indices)
            mapping_indices.insert(index);
        char next_unused_iterator = 'b';
        // Pattern of the input tensor.
        std::stringstream pattern_in;
        pattern_in << "a";
        std::map<Shape::Index, char> input_iterator_mapping, reused_iterator_mapping;
        for (int d = 0; d < 2; ++ d) {
            for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
                auto index = Shape::Index(d, k);
                if (mix->ins[0]->shape[index].Empty() and not mapping_indices.count(index))
                    continue;
                assert(next_unused_iterator <= 'z');
                pattern_in << next_unused_iterator;
                if (mapping_indices.count(index))
                    input_iterator_mapping[index] = next_unused_iterator;
                else
                    reused_iterator_mapping[index] = next_unused_iterator;
                next_unused_iterator ++;
            }
        }
        // Pattern of the output tensor.
        std::stringstream pattern_out;
        pattern_out << "a";
        std::map<Shape::Index, char> output_iterator_mapping;
        for (int d = 0; d < 2; ++ d) {
            for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
                auto index = Shape::Index(d, k);
                if (mix->outs[0]->shape[index].Empty() and not mapping_indices.count(index))
                    continue;
                if (mapping_indices.count(index)) {
                    assert(next_unused_iterator <= 'z');
                    pattern_out << next_unused_iterator;
                    output_iterator_mapping[index] = next_unused_iterator;
                    next_unused_iterator ++;
                } else {
                    assert(reused_iterator_mapping.count(index));
                    pattern_out << reused_iterator_mapping[index];
                }
            }
        }
        // Pattern of the weight tensor.
        std::stringstream pattern_weight;
        for (const auto& index: mix->indices) {
            assert(input_iterator_mapping.count(index));
            pattern_weight << input_iterator_mapping[index];
        }
        for (const auto& index: mix->indices) {
            assert(output_iterator_mapping.count(index));
            pattern_weight << output_iterator_mapping[index];
        }
        // Code generation, may reshape before calculation.
        std::stringstream reference_ss;
        reference_ss << var_map[mix->ins[0]];
        std::vector<Variable> in_shape;
        for (int d = 0; d < 2; ++ d) {
            for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
                auto index = Shape::Index(d, k);
                if (not mix->ins[0]->shape[index].Empty() or mapping_indices.count(index))
                    in_shape.push_back(mix->ins[0]->shape[index]);
            }
        }
        if (in_shape.size() != mix->ins[0]->shape.Continuous().size()) {
            reference_ss << ".view(self.n, ";
            for (const auto& var: in_shape)
                reference_ss << TorchStyleVariable(var) << ", ";
            reference_ss << ")";
        }
        gen->Write() << var_map[mix->outs[0]]
                     << " = "
                     << "torch.einsum(\'"
                     << pattern_in.str() << "," << pattern_weight.str() << "->" << pattern_out.str()
                     << "\', "
                     << "[" << reference_ss.str() << ", self." << primitive_var << "_w])"
                     << ".view(self.n, "
                     << TorchStyleShape(mix->outs[0]->shape)
                     << ").contiguous()" << std::endl;
    } else if (auto output = DynamicCast<OutputPrimitive>(p)) {
        // Reshape (may permute) and return the output variable.
        auto continuous = output->ins[0]->shape.Continuous();
        std::vector<int> permuted(continuous.size());
        assert(not permuted.empty());
        for (int i = 0; i < permuted.size(); ++ i)
            permuted[i] = i;
        // Move C into the front.
        bool swapped = false;
        if (continuous[permuted[0]].static_power[StaticVarPos::VC] == 0) {
            for (int i = 1; i < permuted.size(); ++ i) {
                if (continuous[permuted[i]].static_power[StaticVarPos::VC] > 0) {
                    swapped = true, std::swap(permuted[0], permuted[i]);
                    break;
                }
            }
        }
        // Move W into the end.
        if (continuous[permuted.back()].static_power[StaticVarPos::VW] == 0) {
            for (int i = 1; i < permuted.size() - 1; ++ i) {
                if (continuous[permuted[i]].static_power[StaticVarPos::VW] > 0) {
                    swapped = true, std::swap(permuted[permuted.size() - 1], permuted[i]);
                    break;
                }
            }
        }
        std::stringstream ss;
        if (swapped) {
            ss << ".permute(0";
            for (int index: permuted)
                ss << ", " << index + 1;
            ss << ").contiguous()";
        }
        gen->Write() << "return "
                     << var_map[output->ins[0]]
                     << ss.str()
                     << ".view(self.n, self.c, self.h, self.w)"
                     << std::endl;
    } else if (auto scale = DynamicCast<ScalePrimitive>(p)) {
        gen->Write() << var_map[scale->outs[0]]
                     << " = "
                     << "self." << primitive_var << "_w"
                     << " * "
                     << var_map[scale->ins[0]]
                     << std::endl;
    } else if (auto shift = DynamicCast<ShiftPrimitive>(p)) {
        auto reference = var_map[shift->ins[0]];
        bool shifted = false;
        auto shape = shift->ins[0]->shape;
        for (const auto& index: shift->indices) {
            if (shape[index].Empty())
                continue;
            shifted = true;
            int dim = shape.GetRelativeIndex(index) + 1;
            gen->Write() << var_map[shift->outs[0]]
                         << " = torch.roll("
                         << reference << ", "
                         << "self." << primitive_var << "_" << index.d << "_" << index.k << ", "
                         << dim << ")"
                         << std::endl;
            reference = var_map[shift->outs[0]];
        }
        if (not shifted)
            gen->Write() << var_map[shift->outs[0]]
                         << " = "
                         << reference
                         << std::endl;
    } else if (auto softmax = DynamicCast<SoftmaxPrimitive>(p)) {
        if (softmax->ins[0]->shape[softmax->index].Empty())
            gen->Write() << var_map[softmax->outs[0]]
                         << " = "
                         << var_map[softmax->ins[0]]
                         << std::endl;
        else
            gen->Write() << var_map[softmax->outs[0]]
                         << " = F.softmax("
                         << var_map[softmax->ins[0]] << ", "
                         << "dim=" << softmax->ins[0]->shape.GetRelativeIndex(softmax->index) + 1
                         << ")"
                         << std::endl;
    } else if (auto unfold = DynamicCast<UnfoldPrimitive>(p)) {
        auto reference = PyTorchReshapeToNCHW(gen, var_map, unfold->ins[0], unfold->outs[0]);
        int dilation = unfold->d, kernel_size = unfold->k;
        int padding = dilation * (kernel_size - 1) / 2;
        gen->Write() << var_map[unfold->outs[0]]
                     << " = F.unfold("
                     << reference << ", ("
                     << ((unfold->type == UnfoldH or unfold->type == UnfoldHW) ? kernel_size : 1) << ", "
                     << ((unfold->type == UnfoldW or unfold->type == UnfoldHW) ? kernel_size : 1)
                     << "), dilation=("
                     << ((unfold->type == UnfoldH or unfold->type == UnfoldHW) ? dilation : 1) << ", "
                     << ((unfold->type == UnfoldW or unfold->type == UnfoldHW) ? dilation : 1)
                     << "), padding=("
                     << ((unfold->type == UnfoldH or unfold->type == UnfoldHW) ? padding : 0) << ", "
                     << ((unfold->type == UnfoldW or unfold->type == UnfoldHW) ? padding : 0)
                     << "))"
                     << std::endl;
        gen->Write() << var_map[unfold->outs[0]]
                     << " = " << var_map[unfold->outs[0]]
                     << ".view(self.n, "
                     << TorchStyleShape(unfold->outs[0]->shape)
                     << ")"
                     << std::endl;
    } else {
        CriticalError("Unknown or unimplemented primitive " +
                      p->name + " while generating forward passes");
    }

#ifdef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SHAPES_IN_PYTHON
    // Print sizes (for debugging)
    for (const auto& t: p->outs) {
        gen->Write() << "print('"
                     << var_map[t]
                     << ": ', end='')"
                     << std::endl;
        gen->Write() << "print("
                     << var_map[t] << ".shape)"
                     << std::endl;
    }
#endif
}

Code PyTorchCodeGen::GenImpl(const Solution& solution, std::string name) {
    // TODO: add inplace operations for saving memory.
    try {
        auto global_specs = solution.global_specs;
        auto net_specs = solution.net_specs;
        auto graph = solution.graph;

        // Rename if with an empty name.
        if (name.empty())
            name = "Kernel_" + std::to_string(solution.Hash());

        // Imports
        Write() << "import math" << std::endl;
        Write() << "import random" << std::endl;
        Write() << "import torch" << std::endl;
        Write() << "import torch.nn as nn" << std::endl;
        Write() << "import torch.nn.functional as F" << std::endl;
        Write() << std::endl << std::endl;

        // Class definition and configurations.
        Write() << "class " << name << "(nn.Module):" << std::endl;
        BeginScope();
        Write() << "def __init__(self, c: int, h: int, w: int):" << std::endl;
        BeginScope();
        Write() << "# Configurations" << std::endl;
        Write() << "super(" << name << ", self).__init__()" << std::endl;
        Write() << "self.g = " << global_specs.g << std::endl;
        Write() << "self.n, self.c, self.h, self.w = None, c, h, w" << std::endl;
        Write() << std::endl;

        // Define kernels.
        Write() << "# Kernels" << std::endl;

        // Create variable map.
        VarMap var_map;

        // Travel the graph with the `InitTranslator`.
        PyTorchInitTranslator init_translator(var_map);
        Travel(graph, init_translator);

        // End the `__init__` scope.
        EndScope();
        Write() << std::endl;

        // Write the `forward` function.
        Write() << "def forward(self, x: torch.Tensor):" << std::endl;
        BeginScope();

        // Travel the graph the `ForwardTranslator`.
        PyTorchForwardTranslator forward_translator(var_map);
        Travel(graph, forward_translator);

        // End the `forward` scope.
        EndScope();

        // End the class scope.
        EndScope();
        Write() << std::endl;

        return Dump();
    } catch (CanNotApplyPyTorchCodeGen& ex) {
        CriticalError("Failed to run PyTorch code generation on solution " + std::to_string(solution.Hash()) + " (error: " + ex.info + ")");
    }
}

} // namespace canvas

#ifdef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SHAPES_IN_PYTHON
#undef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SHAPES_IN_PYTHON
#endif

#ifdef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SPECS_IN_PYTHON
#undef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SPECS_IN_PYTHON
#endif
