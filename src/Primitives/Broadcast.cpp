#include "Canvas/Core/Variable.hpp"
#include "Canvas/Primitives/Broadcast.hpp"

// #define CANVAS_DEBUG_BROADCAST_PRIMITIVE_PRINT_SHAPES


namespace canvas {

void BroadcastPrimitive::SolveDynamicVar(const VarSolution& s) {
    Primitive::SolveDynamicVar(s);
    lhs_pi.SolveDynamicVar(s);
    rhs_pi.SolveDynamicVar(s);
    multiplier.SolveDynamicVar(s);
    if (not lhs_pi.MaybeInteger() or not multiplier.MaybeInteger())
        throw CanNotSolveDynamicVarOnGraph(s);
    for (auto& var: br::join(prefix, suffix)) {
        var.SolveDynamicVar(s);
        if (not var.MaybeInteger())
            throw CanNotSolveDynamicVarOnGraph(s);
    }
}

std::vector<Variable> BroadcastPrimitive::IntermediateVariables() const {
    return {lhs_pi, multiplier};
}

static void GetBroadcastMatches(const std::vector<Variable>& lhs_all, const std::vector<Variable>& rhs_all,
                                std::vector<Variable>& prefix, std::vector<Variable>& suffix,
                                Variable& lhs_pi, Variable& rhs_pi, Variable& multiplier) {
    // Some special cases: [C, KH, KH, H, W] & [C, KH, H, W], [KW, H, W] & [KW, KH, KW, H, W].
    suffix = CommonSuffix(lhs_all, rhs_all);
    assert(suffix.size() <= lhs_all.size() and suffix.size() <= rhs_all.size());
    auto lhs_cut_suffix = CutVector(lhs_all, 0, lhs_all.size() - suffix.size());
    auto rhs_cut_suffix = CutVector(rhs_all, 0, rhs_all.size() - suffix.size());
    prefix = CommonPrefix(lhs_cut_suffix, rhs_cut_suffix);
    assert(prefix.size() + suffix.size() <= lhs_all.size());
    assert(prefix.size() + suffix.size() <= rhs_all.size());
    auto lhs_cut = CutVector(lhs_all, prefix.size(), lhs_all.size() - prefix.size() - suffix.size());
    auto rhs_cut = CutVector(rhs_all, prefix.size(), rhs_all.size() - prefix.size() - suffix.size());
    lhs_pi = rhs_pi = Variable();
    for (const auto& var: lhs_cut)
        lhs_pi *= var;
    for (const auto& var: rhs_cut)
        rhs_pi *= var;
    multiplier = rhs_pi / lhs_pi;

#ifdef CANVAS_DEBUG_BROADCAST_PRIMITIVE_PRINT_SHAPES
    std::cout << "lhs_all: " << lhs_all.size() << std::endl;
    std::cout << "rhs_all: " << rhs_all.size() << std::endl;
    std::cout << "prefix: " << prefix.size() << std::endl;
    std::cout << "suffix: " << suffix.size() << std::endl;
    std::cout << "lhs_pi: " << lhs_pi << std::endl;
    std::cout << "rhs_pi: " << rhs_pi << std::endl;
    std::cout << "multiplier: " << multiplier << std::endl;
    std::cout << "lhs_cut: " << std::endl;
    for (const auto& var: lhs_cut)
        std::cout << " > " << var << std::endl;
    std::cout << "rhs_cut: " << std::endl;
    for (const auto& var: rhs_cut)
        std::cout << " > " << var << std::endl;
#endif
}

void BroadcastPrimitive::InferShapes(const TensorSP& lhs, const TensorSP& rhs) {
    auto lhs_all = lhs->shape.Continuous(), rhs_all = rhs->shape.Continuous();
    if (lhs_all != rhs_all) {
        aligned = false, input_commutative = false;
        GetBroadcastMatches(lhs_all, rhs_all, prefix, suffix, lhs_pi, rhs_pi, multiplier);
        if (not lhs_pi.MaybeInteger() or not multiplier.MaybeInteger() or not multiplier.SatisfyAssumption())
            throw CanNotApplyPrimitive(name);
    } else {
        // Element-wise multiplication is commutative.
        aligned = true, input_commutative = true;
    }
}

BroadcastPrimitive::BroadcastPrimitive(const TensorSP& lhs, const TensorSP& rhs,
                                       BroadcastType type):
        Primitive(BroadcastTypeToName(type), {lhs, rhs}, false),
        aligned(false), sign(BroadcastTypeToSign(type)), type(type) {
    InferShapes(lhs, rhs);
    outs.push_back(std::make_shared<Tensor>(rhs->shape));
}

BroadcastPrimitive::BroadcastPrimitive(const TensorSP& lhs, const TensorSP& rhs, BroadcastType type,
                                       bool aligned,
                                       const Variable& lhs_pi, const Variable& rhs_pi, const Variable& multiplier,
                                       std::vector<Variable> prefix, std::vector<Variable> suffix):
        Primitive(BroadcastTypeToName(type), {lhs, rhs}, false),
        aligned(aligned), sign(BroadcastTypeToSign(type)), type(type),
        lhs_pi(lhs_pi), rhs_pi(rhs_pi), multiplier(multiplier),
        prefix(std::move(prefix)), suffix(std::move(suffix)) {
    outs.push_back(std::make_shared<Tensor>(rhs->shape));
}

std::vector<PrimitiveApply>
BroadcastPrimitive::GetAllPossibleMatches(const TensorSP& lhs, const TensorSP& rhs, BroadcastType type, int limit) {
    auto lhs_all = lhs->shape.Continuous(), rhs_all = rhs->shape.Continuous();
    if (lhs_all == rhs_all)
        return {PrimitiveApply(std::make_shared<BroadcastPrimitive>(lhs, rhs, type))};
    std::vector<Variable> prefix, suffix;
    Variable lhs_pi, rhs_pi, multiplier;
    GetBroadcastMatches(lhs_all, rhs_all, prefix, suffix, lhs_pi, rhs_pi, multiplier);
    if (not lhs_pi.MaybeInteger() or not multiplier.MaybeInteger())
        return {};

    // Only if exactly one dynamic variable occurs in the denominator, a variable solution will be constructed.
    auto m_denominator = multiplier.Denominator();
    if (m_denominator.DynamicVarCount() == 0) {
        // No new solution.
        return {PrimitiveApply(std::make_shared<BroadcastPrimitive>(lhs, rhs, type, false,
                                                                    lhs_pi, rhs_pi, multiplier, prefix, suffix))};
    } else {
        auto m_numerator = multiplier.Numerator();
        auto all_factors = m_numerator.GetAllFactors();
        RandomShuffle(all_factors);
        std::vector<PrimitiveApply> collections;
        for (const auto& factor: all_factors) {
            // A factor of numerator equals to the denominator.
            // TODO: support dynamic variable splitting, e.g. x_1 | (x_2 = x_3 * x_4).
            auto s = VarSolution::Solve(factor, m_denominator);
            if (not s.has_value())
                continue;
            auto replaced_lhs_pi = lhs_pi, replaced_rhs_pi = rhs_pi, replaced_multiplier = multiplier;
            replaced_lhs_pi.SolveDynamicVar(s.value());
            replaced_rhs_pi.SolveDynamicVar(s.value());
            replaced_multiplier.SolveDynamicVar(s.value());
            if (replaced_lhs_pi.MaybeInteger() and replaced_rhs_pi.MaybeInteger() and replaced_multiplier.MaybeInteger()) {
                // NOTE: notice that the dynamic variables in `prefix` and `suffix` should also be processed later.
                collections.emplace_back(std::make_shared<BroadcastPrimitive>(lhs, rhs, type, false,
                                                                              replaced_lhs_pi, replaced_rhs_pi,
                                                                              replaced_multiplier, prefix, suffix),
                                         s);
                if ((-- limit) == 0)
                    break;
            }
        }
        return collections;
    }
    Unreachable();
}

} // namespace canvas

#ifdef CANVAS_DEBUG_BROADCAST_PRIMITIVE_PRINT_SHAPES
#undef CANVAS_DEBUG_BROADCAST_PRIMITIVE_PRINT_SHAPES
#endif
