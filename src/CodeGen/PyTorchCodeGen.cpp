#include <ice-cream.hpp>

#include "Canvas/CodeGen/PyTorchCodeGen.hpp"
#include "Canvas/Core/Variable.hpp"
#include "Canvas/Primitives/Factory.hpp"

// #define CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SHAPES_IN_PYTHON
// #define CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SPECS_IN_PYTHON


namespace canvas {

/// Translate into PyTorch-style variable, return "1" if the variable is empty
static std::string TorchStyleVariable(const Variable& var) {
    if (not var.SatisfyAssumption()) {
        std::stringstream ss;
        ss << "Find a variable which does not satisfy the requirement: " << var;
        throw CanNotApplyPyTorchCodeGen(ss.str());
    }
    static_assert(Variable::kStaticVarCount == 8);
    static const char* info[Variable::kStaticVarCount] =
            {"self.g", "self.a", "self.c", "self.k", "self.k", "self.h", "self.w", "self.r"};
    return var.Format(info, " * ", " // ", "self.x[", "]");
}

/// Translate into PyTorch-style variable, return "1" if the variable is empty
static std::string TorchStyleGCKK(const Variable& g, const Variable& c,
                                  const Variable& kh, const Variable& kw) {
    return TorchStyleVariable(g * c * kh * kw);
}

/// Translate into PyTorch-style shape, skip if the variable inside is empty
static std::string TorchStyleShape(const Shape& shape) {
    bool displayed = false;
    std::stringstream ss;
    for (const auto& dim: shape.dims)
        if (not dim.Empty())
            ss << (displayed ? ", " : "") << TorchStyleVariable(dim), displayed = true;
    return ss.str();
}

/// Translate into PyTorch-style shape, skip if the variable inside is empty
static std::string TorchStyleShapeWithoutHW(const Shape& shape) {
    bool displayed = false;
    std::stringstream ss;
    for (int i = 0; i < Shape::kShapeMaxDim; ++ i)
        if (i != DimPos::PH and i != DimPos::PW)
            if (not shape.dims[i].Empty())
                ss << (displayed ? ", " : "") << TorchStyleVariable(shape.dims[i]), displayed = true;
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

static constexpr const char* TorchStyleFoldName(FoldArithmeticType type) {
    switch (type) {
        case FoldAvg: return "mean";
        case FoldMax: return "max";
    }
    return "";
}

static constexpr const char* TorchStyleFoldSuffix(FoldArithmeticType type) {
    switch (type) {
        case FoldAvg: return "";
        case FoldMax: return "[0]";
    }
    return "";
}

static constexpr const char* TorchStyleElementWiseFunctionPrefix(ElementWiseType type) {
    switch (type) {
        case Abs: return "torch.abs(";
        case Exp: return "torch.exp(";
        case Neg: return "-";
        case Sin: return "torch.sin(";
    }
    return "";
}

static constexpr const char* TorchStyleElementWiseFunctionSuffix(ElementWiseType type) {
    switch (type) {
        case Abs: return ")"; // NOLINT(bugprone-branch-clone)
        case Exp: return ")";
        case Neg: return "";
        case Sin: return " * 1.5707963)";
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

PyTorchNCHWRecorder::PyTorchNCHWRecorder(CodeGen* gen, VarMap& var_map, const TensorSP& in, const TensorSP& out,
                                         bool record_os, bool force_record_os):
        gen(gen), var_map(var_map), out(out), force_record_os(force_record_os) {
    auto& in_shape = in->shape;
    int gckk_count = 0;
    gckk_count += not in_shape.G().Empty();
    gckk_count += not in_shape.C().Empty();
    gckk_count += not in_shape.KH().Empty();
    gckk_count += not in_shape.KW().Empty();
    need_view = (gckk_count != 1 or in_shape.H().Empty() or in_shape.W().Empty());
    if (force_record_os)
        assert(record_os);
    if ((need_view and record_os) or force_record_os) {
        gen->Write() << var_map[out] << "_os"
                     << " = " << var_map[in]
                     << ".shape"
                     << std::endl;
    }
    if (need_view) {
        gen->Write() << var_map[out]
                     << " = " << var_map[in]
                     << ".view(self.n, "
                     << TorchStyleGCKK(in_shape.G(), in_shape.C(), in_shape.KH(), in_shape.KW()) << ", "
                     << TorchStyleVariable(in_shape.H()) << ", " << TorchStyleVariable(in_shape.W())
                     << ")"
                     << std::endl;
        reference = var_map[out];
    } else {
        reference = var_map[in];
    }
}

void PyTorchNCHWRecorder::GenCopyShapeCode() {
    if (need_view or force_record_os) {
        gen->Write() << var_map[out]
                     << " = " << var_map[out]
                     << ".view("
                     << var_map[out] << "_os)"
                     << std::endl;
    }
}

void PyTorchInitTranslator::operator () (CodeGen* gen, const PrimitiveSP& p) {
    // Create variables' mapping
    for (const auto& t: p->ins)
        assert(var_map.Count(t));
    for (const auto& t: p->outs) {
        assert(not var_map.Count(t));
        var_map[t] = "t_" + std::to_string(var_map.TensorSize());
    }

    // Handled different operators
    auto primitive_var = (var_map[p] = "p_" + std::to_string(var_map.PrimitiveSize()));
    gen->Write() << "# " << p->name << ": " << primitive_var << std::endl;

    if (DynamicCast<DotPrimitive>(p)) {
        auto& in_shape = p->ins[0]->shape;
        auto& out_shape = p->outs[0]->shape;
        gen->Write() << "self." << primitive_var
                     << " = nn.Conv2d("
                     << TorchStyleGCKK(in_shape.G(), in_shape.C(), in_shape.KH(), in_shape.KW()) // Input channels
                     << ", "
                     << TorchStyleVariable(out_shape.GCKK()) // Output channels
                     << ", 1, padding=0"
                     << ", groups=" << TorchStyleVariable(in_shape.GCKK())
                     << ", bias=False)"
                     << std::endl;
    } else if (DynamicCast<DropoutPrimitive>(p)) {
        gen->Write() << "self." << primitive_var
                     << " = nn.Dropout(p=0.4)"
                     << std::endl;
    } else if (auto fc = DynamicCast<FCPrimitive>(p)) {
        auto& in_shape = p->ins[0]->shape;
        auto& out_shape = p->outs[0]->shape;
        gen->Write() << "self." << primitive_var
                     << " = nn.Conv2d("
                     << TorchStyleGCKK(in_shape.G(), in_shape.C(), in_shape.KH(), in_shape.KW()) // Input channels
                     << ", "
                     << TorchStyleVariable(out_shape.GCKK()) // Output channels
                     << ", 1, padding=0"
                     << ", groups=" << TorchStyleVariable(in_shape.G())
                     << ", bias=False)"
                     << std::endl;
        if (fc->with_norm)
            gen->Write() << "self." << primitive_var << "_bn"
                         << " = nn.BatchNorm2d("
                         << TorchStyleVariable(out_shape.C())
                         << ")"
                         << std::endl;
        if (fc->with_relu)
            gen->Write() << "self." << primitive_var << "_relu"
                         << " = nn.ReLU(inplace=True)"
                         << std::endl;
    } else if (DynamicCast<InputPrimitive>(p)) {
        gen->Write() << "self." << primitive_var
                     << " = nn.AvgPool2d(self.s, self.s) if self.s > 1 else nn.Identity()"
                     << std::endl;
    } else if (DynamicCast<NormPrimitive>(p)) {
        auto& in_shape = p->ins[0]->shape;
        gen->Write() << "self." << primitive_var
                     << " = nn.BatchNorm2d("
                     << TorchStyleGCKK(in_shape.G(), in_shape.C(), in_shape.KH(), in_shape.KW())
                     << ")"
                     << std::endl;
    } else if (DynamicCast<ShiftPrimitive>(p)) {
        gen->Write() << "self." << primitive_var << "_sh"
                     << " = random.randint(-self.p, self.p)" << std::endl;
        gen->Write() << "self." << primitive_var << "_sw"
                     << " = random.randint(-self.p, self.p)" << std::endl;
    } else if (auto softmax = DynamicCast<SoftmaxPrimitive>(p)) {
        if (softmax->type == SoftmaxHW) {
            gen->Write() << "self." << primitive_var
                         << " = nn.Softmax2d()"
                         << std::endl;
        } else {
            int index = -1;
            if (softmax->type == SoftmaxC)
                index = 1;
            else if (softmax->type == SoftmaxH)
                index = 2;
            else if (softmax->type == SoftmaxW)
                index = 3;
            assert(index != -1);
            gen->Write() << "self." << primitive_var
                         << " = nn.Softmax(dim="
                         << index
                         << ")"
                         << std::endl;
        }
    } else if (DynamicCast<ActivationPrimitive>(p) or
            DynamicCast<BroadcastPrimitive>(p) or
            DynamicCast<ChannelShufflePrimitive>(p) or
            DynamicCast<ElementWisePrimitive>(p) or
            DynamicCast<FoldPrimitive>(p) or
            DynamicCast<GroupPrimitive>(p) or
            DynamicCast<OutputPrimitive>(p) or
            DynamicCast<PoolPrimitive>(p) or
            DynamicCast<ReorderPrimitive>(p) or
            DynamicCast<ReshapePrimitive>(p) or
            DynamicCast<TransposePrimitive>(p) or
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

    // Handle different operators
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
        if (broadcast->aligned) { // Not really broadcasting
            gen->Write() << var_map[broadcast->outs[0]]
                         << " = "
                         << var_map[broadcast->ins[0]]
                         << " " << broadcast->sign << " "
                         << var_map[broadcast->ins[1]]
                         << std::endl;
        } else { // Broadcasting
            gen->Write() << var_map[broadcast->outs[0]] << "_f"
                         << " = " << var_map[broadcast->ins[0]]
                         << ".view(self.n, "
                         << TorchStyleBroadcasting(broadcast->prefix, {}, broadcast->lhs_pi, broadcast->suffix)
                         << ")"
                         << std::endl;
            gen->Write() << var_map[broadcast->outs[0]] << "_t"
                         << " = " << var_map[broadcast->ins[1]]
                         << ".view(self.n, "
                         << TorchStyleBroadcasting(broadcast->prefix, broadcast->multiplier, broadcast->lhs_pi, broadcast->suffix)
                         << ")"
                         << std::endl;
            gen->Write() << var_map[broadcast->outs[0]]
                         << " = "
                         << var_map[broadcast->outs[0]] << "_f"
                         << " " << broadcast->sign << " "
                         << var_map[broadcast->outs[0]] << "_t"
                         << std::endl;
            gen->Write() << var_map[broadcast->outs[0]]
                         << " = " << var_map[broadcast->outs[0]]
                         << ".view(self.n, "
                         << TorchStyleShape(broadcast->outs[0]->shape)
                         << ")"
                         << std::endl;
        }
    } else if (auto channel_shuffle = DynamicCast<ChannelShufflePrimitive>(p)) {
        PyTorchNCHWRecorder recorder(gen, var_map, channel_shuffle->ins[0], channel_shuffle->outs[0],
                                     true, true);
        auto& in_shape = channel_shuffle->ins[0]->shape;
        gen->Write() << var_map[channel_shuffle->outs[0]]
                     << " = "
                     << recorder.reference
                     << ".view(self.n, "
                     << TorchStyleVariable(Variable(StaticVar::VG)) << ", "
                     << TorchStyleVariable(in_shape.GCKK() / StaticVar::VG) << ", "
                     << TorchStyleVariable(in_shape.H()) << ", "
                     << TorchStyleVariable(in_shape.W()) << ")"
                     << std::endl;
        gen->Write() << var_map[channel_shuffle->outs[0]]
                     << " = "
                     << var_map[channel_shuffle->outs[0]]
                     << ".permute(0, 2, 1, 3, 4).contiguous()"
                     << std::endl;
        recorder.GenCopyShapeCode();
    } else if (auto dot = DynamicCast<DotPrimitive>(p)) {
        PyTorchNCHWRecorder recorder(gen, var_map, dot->ins[0], dot->outs[0], false);
        gen->Write() << var_map[dot->outs[0]]
                     << " = self." << primitive_var
                     << "(" << recorder.reference << ")"
                     << std::endl;
        gen->Write() << var_map[dot->outs[0]]
                     << " = " << var_map[dot->outs[0]]
                     << ".view(self.n, "
                     << TorchStyleShape(dot->outs[0]->shape)
                     << ")"
                     << std::endl;
    } else if (auto dropout = DynamicCast<DropoutPrimitive>(p)) {
        gen->Write() << var_map[dropout->outs[0]]
                     << " = self." << primitive_var
                     << "(" << var_map[dropout->ins[0]] << ")"
                     << std::endl;
    } else if (auto element_wise = DynamicCast<ElementWisePrimitive>(p)) {
        gen->Write() << var_map[element_wise->outs[0]]
                     << " = "
                     << TorchStyleElementWiseFunctionPrefix(element_wise->type)
                     << var_map[element_wise->ins[0]]
                     << TorchStyleElementWiseFunctionSuffix(element_wise->type)
                     << std::endl;
    } else if (auto fc = DynamicCast<FCPrimitive>(p)) {
        PyTorchNCHWRecorder recorder(gen, var_map, fc->ins[0], fc->outs[0], false);
        gen->Write() << var_map[fc->outs[0]]
                     << " = self." << primitive_var
                     << "(" << recorder.reference << ")"
                     << std::endl;
        if (fc->with_norm) {
            gen->Write() << var_map[fc->outs[0]]
                         << " = self." << primitive_var << "_bn"
                         << "(" << var_map[fc->outs[0]] << ")"
                         << std::endl;
        }
        if (fc->with_relu) {
            gen->Write() << var_map[fc->outs[0]]
                         << " = self." << primitive_var << "_relu"
                         << "(" << var_map[fc->outs[0]] << ")"
                         << std::endl;
        }
        gen->Write() << var_map[fc->outs[0]]
                     << " = " << var_map[fc->outs[0]]
                     << ".view(self.n, "
                     << TorchStyleShape(fc->outs[0]->shape)
                     << ")"
                     << std::endl;
    } else if (auto fold = DynamicCast<FoldPrimitive>(p)) {
        // Shape must be [..., KH?, KW?, H?, W?]
        int n_hw = 0;
        n_hw += static_cast<int>(not fold->ins[0]->shape.H().Empty());
        n_hw += static_cast<int>(not fold->ins[0]->shape.W().Empty());
        int kh_index = -1 - n_hw, kw_index = -1 - n_hw;
        kh_index -= static_cast<int>(not fold->ins[0]->shape.KW().Empty());
        auto reference = var_map[fold->ins[0]];
        // Reduce KW
        if (fold->type == FoldW or fold->type == FoldHW) {
            assert(kw_index <= -1);
            gen->Write() << var_map[fold->outs[0]]
                         << " = " << reference
                         << "." << TorchStyleFoldName(fold->arith_type) << "(" << kw_index << ")"
                         << TorchStyleFoldSuffix(fold->arith_type)
                         << std::endl;
            kh_index += 1;
            reference = var_map[fold->outs[0]];
        }
        // Reduce KH
        if (fold->type == FoldH or fold->type == FoldHW) {
            assert(kh_index <= -1);
            gen->Write() << var_map[fold->outs[0]]
                         << " = " << reference
                         << "." << TorchStyleFoldName(fold->arith_type) << "(" << kh_index << ")"
                         << TorchStyleFoldSuffix(fold->arith_type)
                         << std::endl;
        }
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
        // The input `x` must be in the shape of [N, C, H, W]
        gen->Write() << var_map[input->outs[0]]
                     << " = self." << primitive_var << "(x)"
                     << std::endl;
        gen->Write() << "self.n, self.h, self.w = "
                     << var_map[input->outs[0]] << ".size(0), "
                     << var_map[input->outs[0]] << ".size(-2), "
                     << var_map[input->outs[0]] << ".size(-1)"
                     << std::endl;

        gen->Write() << "if self.ic >= self.oc:" << std::endl;
        gen->BeginScope();
        gen->Write() << var_map[input->outs[0]]
                     << " = " << var_map[input->outs[0]]
                     << ".view(self.n, self.a, self.c, self.h, self.w)"
                     << std::endl;
        gen->EndScope();

        gen->Write() << "else:" << std::endl;
        gen->BeginScope();
        gen->Write() << var_map[input->outs[0]]
                     << " = " << var_map[input->outs[0]]
                     << ".unsqueeze(1).repeat(1, self.a, 1, 1, 1)"
                     << std::endl;
        gen->EndScope();
    } else if (auto norm = DynamicCast<NormPrimitive>(p)) {
        PyTorchNCHWRecorder recorder(gen, var_map, norm->ins[0], norm->outs[0], true);
        gen->Write() << var_map[norm->outs[0]]
                     << " = self." << primitive_var
                     << "(" << recorder.reference << ")"
                     << std::endl;
        recorder.GenCopyShapeCode();
    } else if (auto output = DynamicCast<OutputPrimitive>(p)) {
        // Reshape and return the output variable
        // Not need to sum and get average
        gen->Write() << "if self.ic <= self.oc:" << std::endl;
        gen->BeginScope();
        gen->Write() << "return "
                     << var_map[output->ins[0]]
                     << ".view(self.n, self.oc, self.h, self.w)"
                     << std::endl;
        gen->EndScope();
        // Gather to get average
        gen->Write() << "return "
                     << var_map[output->ins[0]]
                     << ".view(self.n, self.a, self.c, self.h, self.w).mean(dim=1)"
                     << std::endl;
    } else if (auto pool = DynamicCast<PoolPrimitive>(p)) {
        PyTorchNCHWRecorder recorder(gen, var_map, pool->ins[0], pool->outs[0], false);
        gen->Write() << var_map[pool->outs[0]]
                     << " = F.adaptive_avg_pool2d("
                     << recorder.reference << ", ("
                     << (pool->type == PoolH or pool->type == PoolHW or pool->ins[0]->shape.H().Empty() ? "1" : "self.h")
                     << ", "
                     << (pool->type == PoolW or pool->type == PoolHW or pool->ins[0]->shape.W().Empty() ? "1" : "self.w")
                     << "))"
                     << std::endl;
        gen->Write() << var_map[pool->outs[0]]
                     << " = "
                     << var_map[pool->outs[0]]
                     << ".view(self.n, "
                     << TorchStyleShape(pool->outs[0]->shape)
                     << ")"
                     << std::endl;
    } else if (auto reorder = DynamicCast<ReorderPrimitive>(p)) {
        PyTorchNCHWRecorder recorder(gen, var_map, reorder->ins[0], reorder->outs[0], true);
        if (not reorder->inverse) {
            auto reference = recorder.reference;
            if (reorder->type == ReorderH or reorder->type == ReorderHW) {
                gen->Write() << var_map[reorder->outs[0]] << "_0"
                             << " = " << reference
                             << "[::, ::, 0::2, ::]"
                             << std::endl;
                gen->Write() << var_map[reorder->outs[0]] << "_1"
                             << " = " << reference
                             << "[::, ::, 1::2, ::]"
                             << std::endl;
                gen->Write() << var_map[reorder->outs[0]]
                             << " = torch.cat(("
                             << var_map[reorder->outs[0]] << "_0, "
                             << var_map[reorder->outs[0]] << "_1), "
                             << "-2)"
                             << std::endl;
                reference = var_map[reorder->outs[0]];
            }
            if (reorder->type == ReorderW or reorder->type == ReorderHW) {
                gen->Write() << var_map[reorder->outs[0]] << "_0"
                             << " = " << reference
                             << "[::, ::, ::, 0::2]"
                             << std::endl;
                gen->Write() << var_map[reorder->outs[0]] << "_1"
                             << " = " << reference
                             << "[::, ::, ::, 1::2]"
                             << std::endl;
                gen->Write() << var_map[reorder->outs[0]]
                             << " = torch.cat(("
                             << var_map[reorder->outs[0]] << "_0, "
                             << var_map[reorder->outs[0]] << "_1), "
                             << "-1)"
                             << std::endl;
                reference = var_map[reorder->outs[0]];
            }
        } else {
            auto reference = recorder.reference;
            gen->Write() << var_map[reorder->outs[0]]
                         << " = "
                         << reference << ".clone()"
                         << std::endl;
            if (reorder->type == ReorderH or reorder->type == ReorderHW) {
                gen->Write() << "half"
                             << " = "
                             << "(" << reference << ".size(-2) + 1) // 2"
                             << std::endl;
                gen->Write() << var_map[reorder->outs[0]]
                             << "[::, ::, 0::2, ::]"
                             << " = "
                             << reference
                             << "[::, ::, 0:half, ::]"
                             << std::endl;
                gen->Write() << var_map[reorder->outs[0]]
                             << "[::, ::, 1::2, ::]"
                             << " = "
                             << reference
                             << "[::, ::, half:, ::]"
                             << std::endl;
            }
            if (reorder->type == ReorderW or reorder->type == ReorderHW) {
                gen->Write() << "half"
                             << " = "
                             << "(" << reference << ".size(-1) + 1) // 2"
                             << std::endl;
                gen->Write() << var_map[reorder->outs[0]]
                             << "[::, ::, ::, 0::2]"
                             << " = "
                             << reference
                             << "[::, ::, ::, 0:half]"
                             << std::endl;
                gen->Write() << var_map[reorder->outs[0]]
                             << "[::, ::, ::, 1::2]"
                             << " = "
                             << reference
                             << "[::, ::, ::, half:]"
                             << std::endl;
            }
        }
        recorder.GenCopyShapeCode();
    } else if (auto reshape = DynamicCast<ReshapePrimitive>(p)) {
        gen->Write() << var_map[reshape->outs[0]]
                     << " = "
                     << var_map[reshape->ins[0]]
                     << ".view(self.n, "
                     << TorchStyleShape(reshape->outs[0]->shape)
                     << ")"
                     << std::endl;
    } else if (auto shift = DynamicCast<ShiftPrimitive>(p)) {
        int h_index = -1, w_index = -1;
        h_index -= static_cast<int>(not shift->ins[0]->shape.W().Empty());
        auto reference = var_map[shift->ins[0]];
        if (shift->type == ShiftH or shift->type == ShiftHW) {
            gen->Write() << var_map[shift->outs[0]]
                         << " = torch.roll("
                         << reference << ", "
                         << "self." << primitive_var << "_sh, "
                         << h_index << ")"
                         << std::endl;
            reference = var_map[shift->outs[0]];
        }
        if (shift->type == ShiftW or shift->type == ShiftHW) {
            gen->Write() << var_map[shift->outs[0]]
                         << " = torch.roll("
                         << reference << ", "
                         << "self." << primitive_var << "_sw, "
                         << w_index << ")"
                         << std::endl;
        }
    } else if (auto softmax = DynamicCast<SoftmaxPrimitive>(p)) {
        PyTorchNCHWRecorder recorder(gen, var_map, softmax->ins[0], softmax->outs[0], true);
        gen->Write() << var_map[softmax->outs[0]]
                     << " = self." << primitive_var
                     << "(" << recorder.reference << ")"
                     << std::endl;
        recorder.GenCopyShapeCode();
    } else if (auto transpose = DynamicCast<TransposePrimitive>(p)) {
        gen->Write() << var_map[transpose->outs[0]] << "_nd"
                     << " = len("
                     << var_map[transpose->ins[0]]
                     << ".shape)"
                     << std::endl;
        gen->Write() << "assert " << var_map[transpose->outs[0]] << "_nd >= 2"
                     << std::endl;
        gen->Write() << var_map[transpose->outs[0]]
                     << " = torch.transpose("
                     << var_map[transpose->ins[0]] << ", "
                     << var_map[transpose->outs[0]] << "_nd - 1, "
                     << var_map[transpose->outs[0]] << "_nd - 2)"
                     << ".contiguous()" // Deep copy to change the memory layout
                     << std::endl;
    } else if (auto unfold = DynamicCast<UnfoldPrimitive>(p)) {
        // PyTorch's `Unfold` only support [N, C, ...] format (C may be empty, 1)
        PyTorchNCHWRecorder recorder(gen, var_map, unfold->ins[0], unfold->outs[0], false);
        gen->Write() << var_map[unfold->outs[0]]
                     << " = F.unfold("
                     << recorder.reference << ", ("
                     << ((unfold->type == UnfoldH or unfold->type == UnfoldHW) ? "self.k" : "1") << ", "
                     << ((unfold->type == UnfoldW or unfold->type == UnfoldHW) ? "self.k" : "1")
                     << "), padding=("
                     << ((unfold->type == UnfoldH or unfold->type == UnfoldHW) ? "self.p" : "0") << ", "
                     << ((unfold->type == UnfoldW or unfold->type == UnfoldHW) ? "self.p" : "0")
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
    try {
        auto net_specs = solution.specs;
        auto graph = solution.graph;
        auto preferences = solution.preferences;
        auto net_fills = solution.fills;

        // Rename if with an empty name
        if (name.empty())
            name = "Kernel_" + std::to_string(solution.Hash());

        // Imports
        Write() << "import math" << std::endl;
        Write() << "import random" << std::endl;
        Write() << "import torch" << std::endl;
        Write() << "import torch.nn as nn" << std::endl;
        Write() << "import torch.nn.functional as F" << std::endl;
        Write() << std::endl << std::endl;

        // Class definition and configurations
        Write() << "class " << name << "(nn.Module):" << std::endl;
        BeginScope();
        Write() << "def __init__(self, ic: int, oc: int, k: int, s: int, h: int, w: int, x: [int]):" << std::endl;
        BeginScope();
        Write() << "# Configurations" << std::endl;
        Write() << "super(" << name << ", self).__init__()" << std::endl;
        Write() << "assert k % 2 == 1" << std::endl;
        Write() << "assert math.gcd(ic, oc) == min(ic, oc)" << std::endl;
        Write() << "self.ic, self.oc = ic, oc" << std::endl;
        Write() << "self.a = max(ic, oc) // min(ic, oc)" << std::endl;
        Write() << "assert self.a >= 1" << std::endl;
        Write() << "self.c, self.k, self.s, self.p = min(ic, oc), k, s, (k - 1) // 2" << std::endl;
        Write() << "self.g, self.r = " << preferences.g << ", " << preferences.r << std::endl;
        Write() << "self.h, self.w = h // s, w // s" << std::endl;
        Write() << "self.x = x" << std::endl;
        Write() << std::endl;

#ifdef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SPECS_IN_PYTHON
        Write() << "# Print specs" << std::endl;
        Write() << "print(self.x)" << std::endl;
        Write() << std::endl;
#endif

        // Define kernels
        Write() << "# Kernels" << std::endl;

        // Create variable map
        VarMap var_map;

        // Travel the graph with the `InitTranslator`
        PyTorchInitTranslator init_translator(var_map);
        Travel(graph, init_translator);

        // End the `__init__` scope
        EndScope();
        Write() << std::endl;

        // Write the `forward` function
        Write() << "def forward(self, x: torch.Tensor):" << std::endl;
        BeginScope();

        // Travel the graph the `ForwardTranslator`
        PyTorchForwardTranslator forward_translator(var_map);
        Travel(graph, forward_translator);

        // End the `forward` scope
        EndScope();

        // End the class scope
        EndScope();
        Write() << std::endl;

        return Dump();
    } catch (CanNotApplyPyTorchCodeGen& ex) {
        CriticalError("Failed to run PyTorch code generation on solution " + std::to_string(solution.Hash()) + " (error: " + ex.info + ")");
    }
}

} // End namespace canvas

#ifdef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SHAPES_IN_PYTHON
#undef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SHAPES_IN_PYTHON
#endif

#ifdef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SPECS_IN_PYTHON
#undef CANVAS_DEBUG_PYTORCH_CODEGEN_PRINT_SPECS_IN_PYTHON
#endif
