#include <boost/algorithm/string.hpp>

#include "Canvas/Core/Variable.hpp"
#include "Canvas/CodeGen/TVMCodeGen.hpp"

// #define CANVAS_DEBUG_TVM_CODEGEN_PRINT_SPECS_IN_PYTHON


namespace bal = boost::algorithm;

namespace canvas {

/// Translate into TVM-style variable
static std::string TVMStyleVariable(const Variable& var) {
    static_assert(Variable::kStaticVarCount == 8);
    static const char* info[Variable::kStaticVarCount] =
        {"g", "a", "c", "k", "k", "h", "w", "r"};
    return var.Format(info, " * ", " // ", "x[", "]");
}

static int TVMStyleCKKCount(const Shape& shape) {
    int count = 0;
    count += not shape.dims[Shape::PC].Empty();
    count += not shape.dims[Shape::PKH].Empty();
    count += not shape.dims[Shape::PKW].Empty();
    return count;
}

static std::string TVMStyleShapeWithoutHW(const Shape& shape) {
    bool displayed = false;
    std::stringstream ss;
    for (int i = 0; i < Shape::kShapeMaxDim; ++ i)
        if (i != DimPos::PH and i != DimPos::PW)
            if (not shape.dims[i].Empty())
                ss << (displayed ? ", " : "") << TVMStyleVariable(shape.dims[i]), displayed = true;
    return ss.str();
}

static std::string TVMStyleCKKShape(const Shape& shape, bool with_comma=false) {
    bool displayed = false;
    std::stringstream ss;
    if (not shape.dims[Shape::PC].Empty())
        ss << (displayed ? ", " : "") << TVMStyleVariable(shape.dims[Shape::PC]), displayed = true;
    if (not shape.dims[Shape::PKH].Empty())
        ss << (displayed ? ", " : "") << TVMStyleVariable(shape.dims[Shape::PKH]), displayed = true;
    if (not shape.dims[Shape::PKW].Empty())
        ss << (displayed ? ", " : "") << TVMStyleVariable(shape.dims[Shape::PKW]), displayed = true;
    if (with_comma and displayed)
        ss << ", ";
    return ss.str();
}

static std::string TVMStyleBroadcasting(const std::vector<Variable>& prefix,
                                        const Variable& multiplier,
                                        const Variable& pi,
                                        const std::vector<Variable>& suffix) {
    bool displayed = false;
    std::stringstream ss;
    auto Display = [&ss, &displayed](const std::string& info) {
        ss << (displayed ? ", " : "") << info, displayed = true;
    };
    for (const auto& var: prefix)
        Display(TVMStyleVariable(var));
    Display(TVMStyleVariable(multiplier));
    if (not pi.Empty())
        Display(TVMStyleVariable(pi));
    for (const auto& var: suffix)
        Display(TVMStyleVariable(var));
    return ss.str();
}

static std::string TVMStyleShape(const Shape& shape) {
    bool displayed = false;
    std::stringstream ss;
    for (const auto& dim: shape.dims)
        if (not dim.Empty())
            ss << (displayed ? ", " : "") << TVMStyleVariable(dim), displayed = true;
    return ss.str();
}

void TVMReshapeOptimizationTranslator::operator ()(CodeGen* gen, const PrimitiveSP& p) {
    // Get storage
    assert(kOptIndex < var_map.optimizations.size());
    auto storage = DynamicCast<TVMReshapeOptimizationStorage>(var_map.optimizations[kOptIndex]);
    assert(storage != nullptr);

    // Now we have all the status of the output tensors, infer the inputs (note that this translator is reversed)
    if (not (DynamicCast<GroupPrimitive>(p) or DynamicCast<BroadcastPrimitive>(p) or DynamicCast<OutputPrimitive>(p))) {
        if (DynamicCast<ActivationPrimitive>(p) or DynamicCast<ElementWisePrimitive>(p) or
            DynamicCast<ShiftPrimitive>(p) or DynamicCast<DotPrimitive>(p)) {
            // No need to restrict the shape, the same as the single output
            assert(p->ins.size() == 1 and p->outs.size() == 1);
            if (storage->has_non_reshape_usage.count(p->outs[0]))
                storage->has_non_reshape_usage.insert(p->ins[0]);
        } else {
            // Must be aligned with the shape stored in the `Tensor` structure
            for (const auto& t: p->ins)
                storage->has_non_reshape_usage.insert(t);
        }
    }
}

void TVMActualShapeInferenceTranslator::operator ()(CodeGen* gen, const PrimitiveSP& p) {
    // Get storage
    assert(kOptIndex < var_map.optimizations.size());
    auto storage = DynamicCast<TVMReshapeOptimizationStorage>(var_map.optimizations[kOptIndex]);
    assert(storage != nullptr);

    if (auto group = DynamicCast<GroupPrimitive>(p)) {
        // Optimized reshape, the actual shape is not `group->outs[0].shape` but `group->ins[0]->shape`
        if (not storage->has_non_reshape_usage.count(group->outs[0]))
            storage->actual_shape[group->outs[0]] = group->ins[0]->shape;
    } else if (DynamicCast<ActivationPrimitive>(p) or DynamicCast<ElementWisePrimitive>(p) or
               DynamicCast<ShiftPrimitive>(p) or DynamicCast<DotPrimitive>(p)) {
        assert(p->ins.size() == 1 and p->outs.size() == 1);
        if (storage->actual_shape.count(p->ins[0]))
            storage->actual_shape[p->outs[0]] = storage->actual_shape[p->ins[0]];
    }
}

void TVMPlaceholderTranslator::operator ()(CodeGen* gen, const PrimitiveSP& p) {
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

    // Reshape optimization
    assert(TVMReshapeOptimizationTranslator::kOptIndex < var_map.optimizations.size());
    auto storage = DynamicCast<TVMReshapeOptimizationStorage>(
            var_map.optimizations[TVMReshapeOptimizationTranslator::kOptIndex]);
    assert(storage != nullptr);

    if (auto input = DynamicCast<InputPrimitive>(p)) {
        // Input placeholder
        gen->Write() << "I = te.placeholder((n, ic, h, w))" << std::endl;

        // Pool
        gen->Write() << "X = tlib.avg_pool(I, s, name=\'" << primitive_var << "_ap\')" << std::endl;
        gen->Write() << "h, w = h // s, w // s" << std::endl;

        // Repeat `a` times
        gen->Write() << "if ic <= oc:" << std::endl;
        gen->BeginScope();
        gen->Write() << var_map[input->outs[0]]
                     << " = tlib.repeat_to_group(X, 1, a, name=\'" << primitive_var << "_repeat\')"
                     << std::endl;
        gen->EndScope();
        gen->Write() << "else:" << std::endl;
        gen->BeginScope();
        gen->Write() << var_map[input->outs[0]]
                     << " = tlib.reshape(X, (n, a, c, h, w))"
                     << std::endl;
        gen->EndScope();
    } else if (auto dot = DynamicCast<DotPrimitive>(p)) {
        // Get the actual shape
        auto shape = dot->ins[0]->shape;
        if (storage->actual_shape.count(dot->ins[0]))
            shape = storage->actual_shape[dot->ins[0]];

        // Code generation
        if (shape.GCKK().Empty()) {
            // Scalar
            gen->Write() << primitive_var << "_w = 3.14" << std::endl;
        } else {
            gen->Write() << primitive_var << "_w = te.placeholder(("
                         << "1, " << TVMStyleShapeWithoutHW(shape)
                         << (shape.H().Empty() ? "" : ", 1")
                         << (shape.W().Empty() ? "" : ", 1")
                         << "))" << std::endl;
        }
    } else if (DynamicCast<FCPrimitive>(p)) {
        if (p->outs[0]->shape.GCKK().Empty() and p->ins[0]->shape.GCKK().Empty()) {
            // Scalar
            gen->Write() << primitive_var << "_w = 3.14" << std::endl;
        } else {
            gen->Write() << primitive_var << "_w = te.placeholder(("
                         << (p->outs[0]->shape.GCKK().Empty() ? "" : (TVMStyleVariable(p->outs[0]->shape.GCKK()) + ", "))
                         << (p->ins[0]->shape.CKK().Empty() ? "" : TVMStyleCKKShape(p->ins[0]->shape, true))
                         << "))" << std::endl;
        }
    } else if (DynamicCast<ActivationPrimitive>(p) or
              DynamicCast<BroadcastPrimitive>(p) or
              DynamicCast<ElementWisePrimitive>(p) or
              DynamicCast<FoldPrimitive>(p) or
              DynamicCast<GroupPrimitive>(p) or
              DynamicCast<PoolPrimitive>(p) or
              DynamicCast<ShiftPrimitive>(p) or
              DynamicCast<SoftmaxPrimitive>(p) or
              DynamicCast<UnfoldPrimitive>(p) or
              DynamicCast<OutputPrimitive>(p)) {
        gen->Write() << "pass" << std::endl;
    } else {
        Unimplemented();
    }
}

static std::string TVMStyleActivationFunctionName(const ActivationType& type) {
    switch (type) {
        case GeLU:
            Unimplemented();
        case ReLU:
            return "topi.nn.relu";
        case Sigmoid:
            return "topi.sigmoid";
        case TanH:
            return "topi.tanh";
    }
    return "";
}

static std::string TVMStyleElementWiseFunctionName(const ElementWiseType& type) {
    switch (type) {
        case Abs:
            return "topi.abs";
        case Exp:
            return "topi.exp";
        case Neg:
            return "topi.negative";
        case Sin:
            return "topi.sin";
    }
    return "";
}

static std::string TVMStyleBroadcastFunctionName(const BroadcastType type) {
    switch (type) {
        case BAdd:
            return "topi.add";
        case BSub:
            return "topi.subtract";
        case BMul:
            return "topi.multiply";
    }
    return "";
}

static std::string TVMStyleFoldFunctionName(const FoldArithmeticType& type) {
    switch (type) {
        case FoldMax:
            return "tlib.fold_max";
        case FoldAvg:
            return "tlib.fold_avg";
    }
    return "";
}

// Convert pair dimensions into string
static std::string PairToArgs(const std::pair<int, int>& pair) {
    std::stringstream ss;
    assert(pair.first >= 0 and pair.second >= 0);
    assert(not (pair.first == 0 and pair.second == 0));
    if (pair.first > 0 and pair.second > 0)
        ss << "(" << pair.first << ", " << pair.second << ")";
    else
        ss << pair.first + pair.second;
    return ss.str();
};

void TVMOperatorTranslator::operator ()(CodeGen* gen, const PrimitiveSP& p) {
    for (const auto& t: br::join(p->ins, p->outs))
        assert(var_map.Count(t));

    // Handle different operators
    assert(var_map.Count(p));
    auto primitive_var = var_map[p];
    gen->Write() << "# " << p->name << ": " << primitive_var << std::endl;

    // Reshape optimization
    assert(TVMReshapeOptimizationTranslator::kOptIndex < var_map.optimizations.size());
    auto storage = DynamicCast<TVMReshapeOptimizationStorage>(
            var_map.optimizations[TVMReshapeOptimizationTranslator::kOptIndex]);
    assert(storage != nullptr);

    if (auto activation = DynamicCast<ActivationPrimitive>(p)) {
        gen->Write() << var_map[activation->outs[0]] << " = "
                     << TVMStyleActivationFunctionName(activation->type)
                     << "(" << var_map[activation->ins[0]] << ")"
                     << std::endl;
    } else if (auto broadcast = DynamicCast<BroadcastPrimitive>(p)) {
        if (broadcast->aligned) {
            gen->Write() << var_map[broadcast->outs[0]] << "_pre = "
                         << TVMStyleBroadcastFunctionName(broadcast->type) << "("
                         << var_map[broadcast->ins[0]] << ", "
                         << var_map[broadcast->ins[1]] << ")"
                         << std::endl;
            // Ensure the shape is right
            gen->Write() << var_map[broadcast->outs[0]]
                         << " = tlib.reshape("
                         << var_map[broadcast->outs[0]] << "_pre, "
                         << "(n, " << TVMStyleShape(broadcast->outs[0]->shape) << ")"
                         << ")" << std::endl;
        } else {
            gen->Write() << var_map[broadcast->outs[0]] << "_f"
                         << " = tlib.reshape("
                         << var_map[broadcast->ins[0]] << ", "
                         << "(n, " << TVMStyleBroadcasting(broadcast->prefix, {}, broadcast->lhs_pi, broadcast->suffix) << ")"
                         << ")" << std::endl;
            gen->Write() << var_map[broadcast->outs[0]] << "_t"
                         << " = tlib.reshape("
                         << var_map[broadcast->ins[1]] << ", "
                         << "(n, " << TVMStyleBroadcasting(broadcast->prefix, broadcast->multiplier, broadcast->lhs_pi, broadcast->suffix) << ")"
                         << ")" << std::endl;
            gen->Write() << var_map[broadcast->outs[0]]
                         << " = tlib.reshape("
                         << TVMStyleBroadcastFunctionName(broadcast->type)
                         << "(" << var_map[broadcast->outs[0]] << "_f, "
                         << var_map[broadcast->outs[0]] << "_t), "
                         << "(n, " << TVMStyleShape(broadcast->outs[0]->shape) << ")"
                         << ")" << std::endl;
        }
    } else if (auto dot = DynamicCast<DotPrimitive>(p)) {
        gen->Write() << var_map[dot->outs[0]] << " = "
                     << "topi.multiply("
                     << var_map[dot->ins[0]]
                     << ", "
                     << primitive_var << "_w)"
                     << std::endl;
    } else if (auto element_wise = DynamicCast<ElementWisePrimitive>(p)) {
        gen->Write() << var_map[element_wise->outs[0]] << " = "
                     << TVMStyleElementWiseFunctionName(element_wise->type)
                     << "(" << var_map[element_wise->ins[0]] << ")"
                     << std::endl;
    } else if (auto fc = DynamicCast<FCPrimitive>(p)) {
        if (fc->with_relu or fc->with_norm)
            Unimplemented();
        if (fc->ins[0]->shape.G().Empty()) { // Non-grouped FC
            if (fc->outs[0]->shape.GCKK().Empty() and fc->ins[0]->shape.CKK().Empty()) {
                gen->Write() << var_map[fc->outs[0]] << " = "
                             << "topi.multiply("
                             << var_map[fc->ins[0]]
                             << ", "
                             << var_map[fc] << "_w)"
                             << std::endl;
            } else {
                gen->Write() << var_map[fc->outs[0]]
                             << " = tlib.fc("
                             << var_map[fc->ins[0]] << ", "
                             << var_map[fc] << "_w, "
                             << TVMStyleCKKCount(fc->ins[0]->shape) << ", "
                             << "has_nd=" << (fc->outs[0]->shape.GCKK().Empty() ? "False" : "True") << ", "
                             << "name=\'" << primitive_var
                             << "\')" << std::endl;
            }
        } else {
            // The output shape is at least the group number
            assert(not fc->outs[0]->shape.GCKK().Empty());
            gen->Write() << var_map[fc->outs[0]]
                         << " = tlib.grouped_fc("
                         << var_map[fc->ins[0]] << ", "
                         << var_map[fc] << "_w, "
                         << TVMStyleCKKCount(fc->ins[0]->shape) << ", "
                         << "name=\'" << primitive_var
                         << "\')" << std::endl;
        }
        gen->Write() << var_map[fc->outs[0]]
                     << " = tlib.reshape("
                     << var_map[fc->outs[0]] << ", "
                     << "(n, " << TVMStyleShape(fc->outs[0]->shape) << ")"
                     << ")" << std::endl;
    } else if (auto fold = DynamicCast<FoldPrimitive>(p)) {
        std::pair<int, int> fold_array(0, 0);
        if (fold->type == FoldH or fold->type == FoldHW) {
            assert(not fold->ins[0]->shape.KH().Empty());
            fold_array.first = fold->ins[0]->shape.GetRelativePos(Shape::PKH) + 1;
        }
        if (fold->type == FoldW or fold->type == FoldHW) {
            assert(not fold->ins[0]->shape.KW().Empty());
            fold_array.second = fold->ins[0]->shape.GetRelativePos(Shape::PKW) + 1;
        }
        gen->Write() << var_map[fold->outs[0]] << " = "
                     << TVMStyleFoldFunctionName(fold->arith_type) << "("
                     << var_map[fold->ins[0]] << ", "
                     << PairToArgs(fold_array) << ")"
                     << std::endl;
    } else if (auto group = DynamicCast<GroupPrimitive>(p)) {
        if (storage->has_non_reshape_usage.count(group->outs[0])) {
            gen->Write() << var_map[group->outs[0]]
                         << " = tlib.reshape("
                         << var_map[group->ins[0]] << ", "
                         << "(n, " << TVMStyleShape(group->outs[0]->shape) << ")"
                         << ")" << std::endl;
        } else {
            gen->Write() << var_map[group->outs[0]]
                         << " = "
                         << var_map[group->ins[0]]
                         << std::endl;
        }
    } else if (auto input = DynamicCast<InputPrimitive>(p)) {
        gen->Write() << "pass" << std::endl;
    } else if (auto pool = DynamicCast<PoolPrimitive>(p)) {
        // Almost the same as `FoldPrimitive`
        std::pair<int, int> pool_array(0, 0);
        if (pool->type == PoolH or pool->type == PoolHW) {
            assert(not pool->ins[0]->shape.H().Empty());
            pool_array.first = pool->ins[0]->shape.GetRelativePos(Shape::PH) + 1;
        }
        if (pool->type == PoolW or pool->type == PoolHW) {
            assert(not pool->ins[0]->shape.W().Empty());
            pool_array.second = pool->ins[0]->shape.GetRelativePos(Shape::PW) + 1;
        }
        gen->Write() << var_map[pool->outs[0]]
                     << " = tlib.fold_avg("
                     << var_map[pool->ins[0]] << ", "
                     << PairToArgs(pool_array) << ")"
                     << std::endl;
    } else if (auto shift = DynamicCast<ShiftPrimitive>(p)) {
        if (shift->type == ShiftHW) {
            gen->Write() << var_map[shift->outs[0]]
                         << " = tlib.shift_2d("
                         << var_map[shift->ins[0]]
                         << ")" << std::endl;
        } else {
            int d = 0;
            if (shift->type == ShiftH)
                d = shift->ins[0]->shape.GetRelativePos(Shape::PH, true);
            else if (shift->type == ShiftW)
                d = shift->ins[0]->shape.GetRelativePos(Shape::PW, true);
            assert(d < 0);
            gen->Write() << var_map[shift->outs[0]]
                         << " = tlib.shift_1d("
                         << var_map[shift->ins[0]] << ", "
                         << d << ")"
                         << std::endl;
        }
    } else if (auto softmax = DynamicCast<SoftmaxPrimitive>(p)) {
        if (softmax->type == SoftmaxHW) {
            gen->Write() << var_map[softmax->outs[0]]
                         << " = tlib.softmax_2d("
                         << var_map[softmax->ins[0]] << ", "
                         << "name=\'" << primitive_var
                         << "\')" << std::endl;
        } else {
            int d = -1;
            if (softmax->type == SoftmaxC)
                d = softmax->ins[0]->shape.GetRelativePos(Shape::PC);
            if (softmax->type == SoftmaxH)
                d = softmax->ins[0]->shape.GetRelativePos(Shape::PH);
            if (softmax->type == SoftmaxW)
                d = softmax->ins[0]->shape.GetRelativePos(Shape::PW);
            assert(d != -1);
            gen->Write() << var_map[softmax->outs[0]]
                         << " = tlib.softmax_1d("
                         << var_map[softmax->ins[0]] << ", "
                         << d + 1 << ", "
                         << "name=\'" << primitive_var
                         << "\')" << std::endl;
        }
    } else if (auto unfold = DynamicCast<UnfoldPrimitive>(p)) {
        if (unfold->type == UnfoldHW) {
            gen->Write() << var_map[unfold->outs[0]] << " = "
                         << "tlib.unfold_2d("
                         << var_map[unfold->ins[0]] << ", "
                         << "k, "
                         << "name=\'" << primitive_var
                         << "\')"
                         << std::endl;
        } else {
            // Get the dimension to unfold
            int from_pos = -1;
            if (unfold->type == UnfoldH) {
                assert(not unfold->ins[0]->shape.H().Empty());
                from_pos = unfold->ins[0]->shape.GetRelativePos(Shape::PH);
            } else if (unfold->type == UnfoldW) {
                assert(not unfold->ins[0]->shape.W().Empty());
                from_pos = unfold->ins[0]->shape.GetRelativePos(Shape::PW);
            }
            assert(from_pos != -1);

            // Get the dimension to insert
            int to_pos = -1;
            if (unfold->type == UnfoldH) {
                assert(unfold->ins[0]->shape.KH().Empty());
                to_pos = unfold->ins[0]->shape.GetRelativePos(Shape::PKH);
            } else if (unfold->type == UnfoldW) {
                assert(unfold->ins[0]->shape.KW().Empty());
                to_pos = unfold->ins[0]->shape.GetRelativePos(Shape::PKW);
            }
            assert(to_pos != -1);

            gen->Write() << var_map[unfold->outs[0]] << " = "
                         << "tlib.unfold_1d("
                         << var_map[unfold->ins[0]] << ", k, "
                         << "f=" << from_pos + 1 << ", "
                         << "t=" << to_pos + 1 << ", "
                         << "name=\'" << primitive_var
                         << "\')"
                         << std::endl;
        }
    } else if (auto output = DynamicCast<OutputPrimitive>(p)) {
        // Not need to sum and get average
        gen->Write() << "if ic <= oc:" << std::endl;
        gen->BeginScope();
        gen->Write() << "Z = tlib.reshape(" << var_map[p->ins[0]] << ", "
                     << "(n, oc, h, w)"
                     << ")" << std::endl;
        gen->EndScope();

        // Gather to get average
        gen->Write() << "else:" << std::endl;
        gen->BeginScope();
        gen->Write() << "Z = tlib.reshape(" << var_map[p->ins[0]] << ", "
                     << "(n, a, oc, h, w)"
                     << ")" << std::endl;
        gen->Write() << "Z = tlib.fold_avg(Z, 1)" << std::endl;
        gen->EndScope();
        gen->Write() << "assert tuple(Z.shape) == (n, oc, h, w)" << std::endl;
    } else {
        Unimplemented();
    }
}

void TVMReturnTranslator::operator ()(CodeGen* gen, const PrimitiveSP& p) {
    for (const auto& t: br::join(p->ins, p->outs))
        assert(var_map.Count(t));

    // Handle different operators
    assert(var_map.Count(p));
    auto primitive_var = var_map[p];

    if (DynamicCast<InputPrimitive>(p)) {
        gen->Write(false) << "I, ";
    } else if (auto dot = DynamicCast<DotPrimitive>(p)) {
        if (not dot->ins[0]->shape.GCKK().Empty()) {
            gen->Write(false) << primitive_var << "_w, ";
        }
    } else if (auto fc = DynamicCast<FCPrimitive>(p)) {
        if (not (p->outs[0]->shape.GCKK().Empty() and p->ins[0]->shape.GCKK().Empty())) {
            gen->Write(false) << primitive_var << "_w, ";
        }
    } else if (DynamicCast<OutputPrimitive>(p)) {
        gen->Write(false) << "Z";
    } else if (DynamicCast<ActivationPrimitive>(p) or
            DynamicCast<BroadcastPrimitive>(p) or
            DynamicCast<ElementWisePrimitive>(p) or
            DynamicCast<FoldPrimitive>(p) or
            DynamicCast<GroupPrimitive>(p) or
            DynamicCast<PoolPrimitive>(p) or
            DynamicCast<ShiftPrimitive>(p) or
            DynamicCast<SoftmaxPrimitive>(p) or
            DynamicCast<UnfoldPrimitive>(p)) {
        // Write nothing
    } else {
        Unimplemented();
    }
}

Code TVMCodeGen::GenImpl(const Solution& solution, std::string name) {
    try {
        auto net_specs = solution.specs;
        auto graph = solution.graph;
        auto preferences = solution.preferences;
        auto net_fills = solution.fills;

        // Rename if with an empty name
        if (name.empty())
            name = "canvas_" + std::to_string(solution.Hash());
        bal::to_lower(name);

        // Imports
        Write() << "import math" << std::endl;
        Write() << "import tlib" << std::endl;
        Write() << "from tvm import auto_scheduler, te, topi"
                << std::endl << std::endl << std::endl;

        // Function definition
        Write() << "@auto_scheduler.register_workload(func_name="
                << "\'" << name << "\'"
                << ", override=True)" << std::endl;
        Write() << "def " << name << "(n: int, ic: int, oc: int, k: int, s: int, h: int, w: int, x: [int] = None):"
                << std::endl;
        BeginScope();

#ifdef CANVAS_DEBUG_TVM_CODEGEN_PRINT_SPECS_IN_PYTHON
        Write() << "# Print specs" << std::endl;
        Write() << "print(x)" << std::endl;
        Write() << std::endl;
#endif

        Write() << "# Configurations" << std::endl;
        Write() << "assert math.gcd(ic, oc) == min(ic, oc)" << std::endl;
        Write() << "c, a = min(ic, oc), max(ic, oc) // min(ic, oc)" << std::endl;

        Write() << "# Heuristic preferences" << std::endl;
        Write() << "g, r = " << preferences.g << ", " << preferences.r << std::endl;
        Write() << std::endl;

        // Create variable map
        VarMap var_map;

        // Add optimizations
        auto reshape_opt = std::make_shared<TVMReshapeOptimizationStorage>();
        var_map.optimizations.push_back(reshape_opt);

        // Reshape optimization
        TVMReshapeOptimizationTranslator reshape_optimization_translator(var_map);
        Travel(graph, reshape_optimization_translator, true);

        // Infer actual shapes
        TVMActualShapeInferenceTranslator actual_shape_inference_translator(var_map);
        Travel(graph, actual_shape_inference_translator);

        // Placeholder translator
        Write() << "# Placeholders" << std::endl;
        TVMPlaceholderTranslator placeholder_translator(var_map);
        Travel(graph, placeholder_translator);
        Write() << std::endl;

        // Operator translator
        Write() << "# Operators" << std::endl;
        TVMOperatorTranslator operator_translator(var_map);
        Travel(graph, operator_translator);

        // Return translator
        Write() << "return [";
        TVMReturnTranslator return_translator(var_map);
        Travel(graph, return_translator);
        Write(false) << "]" << std::endl;

        // End function body
        EndScope();
        Write() << std::endl;
    } catch (CanNotApplyTVMCodeGen& ex) {
        CriticalError("Failed to run TVM code generation on solution " + std::to_string(solution.Hash()) + " (error: " + ex.info + ")");
    }
    return Dump();
}

} // End namespace canvas

#ifdef CANVAS_DEBUG_TVM_CODEGEN_PRINT_SPECS_IN_PYTHON
#undef CANVAS_DEBUG_TVM_CODEGEN_PRINT_SPECS_IN_PYTHON
#endif
