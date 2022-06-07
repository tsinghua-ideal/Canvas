#include "Canvas/Core/Graph.hpp"
#include "Canvas/Primitives/Factory.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

bool PrimitiveGenOptions::Adapt(const PrimitiveApply& apply) const {
    assert(apply.primitive);

    // Check types.
    // Broadcast.
    if (auto broadcast = DynamicCast<BroadcastPrimitive>(apply.primitive)) {
        if (not b_add and broadcast->type == BAdd)
            return false;
        if (not b_mul and broadcast->type == BMul)
            return false;
        if (not b_sub and broadcast->type == BSub)
            return false;
    }

    // With parameters.
    if (not dot and DynamicCast<DotPrimitive>(apply.primitive))
        return false;
    if (not fc and DynamicCast<FCPrimitive>(apply.primitive))
        return false;

    // Arithmetic operators.
    if (auto activation = DynamicCast<ActivationPrimitive>(apply.primitive)) {
        if (not gelu and activation->type == GeLU)
            return false;
        if (not relu and activation->type == ReLU)
            return false;
        if (not sigmoid and activation->type == Sigmoid)
            return false;
        if (not tanh and activation->type == TanH)
            return false;
    }
    if (not dropout and DynamicCast<DropoutPrimitive>(apply.primitive))
        return false;
    if (not norm and DynamicCast<NormPrimitive>(apply.primitive))
        return false;
    if (auto softmax = DynamicCast<SoftmaxPrimitive>(apply.primitive)) {
        if (not softmax_c and softmax->type == SoftmaxC)
            return false;
        if (not softmax_h and softmax->type == SoftmaxH)
            return false;
        if (not softmax_w and softmax->type == SoftmaxW)
            return false;
        if (not softmax_hw and softmax->type == SoftmaxHW)
            return false;
    }
    if (auto element_wise = DynamicCast<ElementWisePrimitive>(apply.primitive)) {
        if (not abs and element_wise->type == Abs)
            return false;
        if (not exp and element_wise->type == Exp)
            return false;
        if (not neg and element_wise->type == Neg)
            return false;
        if (not sin and element_wise->type == Sin)
            return false;
    }

    // Spatial operators.
    if (not channel_shuffle and DynamicCast<ChannelShufflePrimitive>(apply.primitive))
        return false;
    if (auto fold = DynamicCast<FoldPrimitive>(apply.primitive)) {
        if (not fold_h and fold->type == FoldH)
            return false;
        if (not fold_w and fold->type == FoldW)
            return false;
        if (not fold_hw and fold->type == FoldHW)
            return false;
        if (not fold_avg and fold->arith_type == FoldAvg)
            return false;
        if (not fold_max and fold->arith_type == FoldMax)
            return false;
    }
    if (auto group = DynamicCast<GroupPrimitive>(apply.primitive)) {
        if (not group_by_factor and group->type == GroupByFactor)
            return false;
        if (not group_all and group->type == GroupAllChannels)
            return false;
    }
    if (auto pool = DynamicCast<PoolPrimitive>(apply.primitive)) {
        if (not pool_h and pool->type == PoolH)
            return false;
        if (not pool_w and pool->type == PoolW)
            return false;
        if (not pool_hw and pool->type == PoolHW)
            return false;
    }
    if (auto shift = DynamicCast<ShiftPrimitive>(apply.primitive)) {
        if (not shift_h and shift->type == ShiftH)
            return false;
        if (not shift_w and shift->type == ShiftW)
            return false;
        if (not shift_hw and shift->type == ShiftHW)
            return false;
    }
    if (not transpose and DynamicCast<TransposePrimitive>(apply.primitive))
        return false;

    // Check width.
    int delta_width = static_cast<int>(apply.primitive->outs.size());
    for (const auto& t: ToUnorderedSet(apply.primitive->ins))
        delta_width -= static_cast<int>(t->consumers.empty()); // Not applied yet.
    return delta_width <= max_delta_width;
}

std::vector<PrimitiveApply> PrimitiveFactory::GetPrimitiveApplies(const GraphSP& graph,
                                                                  bool allow_dynamic_variables,
                                                                  const std::set<TensorSP>& forbidden_successor) {
    // Get all filters.
    std::unordered_set<size_t> filter;
    for (const auto& p: graph->primitives)
        filter.insert(p->Hash(true));

    std::vector<PrimitiveApply> primitives;

    // Get primitives without inputs.
    GetPrimitiveApplies(graph, primitives, filter, allow_dynamic_variables);

    // Filter tensors.
    std::vector<TensorSP> filtered;
    for (const auto& t: graph->tensors)
        if (not forbidden_successor.count(t))
            filtered.push_back(t);

    // Get primitives with one input.
    for (const auto& t: filtered)
        GetPrimitiveApplies(graph, primitives, t, filter, allow_dynamic_variables);

    // Get primitives with two inputs.
    for (const auto& lhs: filtered)
        for (const auto& rhs: filtered)
            GetPrimitiveApplies(graph, primitives, lhs, rhs, filter, allow_dynamic_variables);

    return primitives;
}

std::vector<PrimitiveApply> PrimitiveFactory::RescalePossibilities(const std::vector<PrimitiveApply>& applies,
                                                                   bool remove_fc) {
    std::vector<PrimitiveApply> applies_copy = applies;
    RandomShuffle(applies_copy);

    // Push all the first in the shuffle into the candidates.
    std::vector<PrimitiveApply> new_applies;
    std::set<std::string> filter;
    for (const auto& apply: applies_copy) {
        if (remove_fc and DynamicCast<FCPrimitive>(apply.primitive) != nullptr)
            continue;
        if (not filter.count(apply.primitive->name)) {
            filter.insert(apply.primitive->name);
            new_applies.push_back(apply);
        }
    }
    return new_applies;
}

std::vector<PrimitiveApply> PrimitiveFactory::FilterPrimitiveApplies(const std::vector<PrimitiveApply>& applies,
                                                                     const PrimitiveGenOptions& options) {
    std::vector<PrimitiveApply> new_applies;
    for (const auto& apply: applies)
        if (options.Adapt(apply))
            new_applies.push_back(apply);
    return new_applies;
}

std::vector<PrimitiveApply> PrimitiveFactory::FilterPrimitiveApplies(const std::vector<PrimitiveApply>& applies,
                                                                     const std::vector<PrimitiveGenOptions>& or_options) {
    // No requirements, just return the original primitives.
    if (or_options.empty())
        return applies;
    std::vector<PrimitiveApply> new_applies;
    for (const auto& apply: applies) {
        if (std::any_of(or_options.begin(), or_options.end(), [apply](const PrimitiveGenOptions& options) -> bool {
            return options.Adapt(apply);
        })) new_applies.push_back(apply);
    }
    return new_applies;
}

std::vector<PrimitiveApply> PrimitiveFactory::FilterPrimitiveAppliesForOutput(const std::vector<PrimitiveApply>& applies) {
    std::vector<PrimitiveApply> new_applies;
    for (const auto& apply: applies)
        if (apply.primitive->outs.size() == 1)
            if (not (apply.primitive->outs[0]->shape.IsAllStatic() and not apply.primitive->outs[0]->shape.CouldBeReshapeToCHW()))
                new_applies.push_back(apply);
    return new_applies;
}

void PrimitiveFactory::GetPrimitiveApplies(const GraphSP &graph,
                                           std::vector<PrimitiveApply>& primitives,
                                           std::unordered_set<size_t>& filter,
                                           bool allow_dynamic_variables) {
    // Currently, none.
}

void PrimitiveFactory::GetPrimitiveApplies(const GraphSP &graph,
                                           std::vector<PrimitiveApply>& primitives,
                                           const TensorSP& t,
                                           std::unordered_set<size_t>& filter,
                                           bool allow_dynamic_variables) {
    assert(t->producer);
    // The next variable index.
    auto next_index_opt = graph->NextUnusedDynamicVarIndex();

    // Dot: no new variables.
    TryMakeAndPush<DotPrimitive>(primitives, filter, t);

    // FC: The channel could be a new variable.
    if (allow_dynamic_variables) {
        if (t->shape.G().IsStatic()) { // Could not have dynamic variables in G.
            if (next_index_opt.has_value())
                TryMakeAndPush<FCPrimitive>(primitives, filter, t, Variable::DynamicVar(next_index_opt.value()));
            else
                TryMakeAndPush<FCPrimitive>(primitives, filter, t); // By default, we retain the channel number and fold spatial information.
        }
    } else {
        TryMakeAndPush<FCPrimitive>(primitives, filter, t); // By default, we retain the channel number and fold spatial information.
    }

    // Activation: no new variables, pruning: input could not have been activated.
    if (DynamicCast<InputPrimitive>(t->producer) == nullptr and DynamicCast<ActivationPrimitive>(t->producer) == nullptr) {
        for (const auto &type: {GeLU, ReLU, Sigmoid, TanH})
            TryMakeAndPush<ActivationPrimitive>(primitives, filter, t, type);
    }

    // Dropout: no new variables, pruning: input could not have been `Dropout`-ed.
    if (DynamicCast<DropoutPrimitive>(t->producer) == nullptr)
        TryMakeAndPush<DropoutPrimitive>(primitives, filter, t);

    // Norm: no new variables, pruning: input could not have been already normalized.
    if (DynamicCast<InputPrimitive>(t->producer) == nullptr and DynamicCast<NormPrimitive>(t->producer) == nullptr)
        TryMakeAndPush<NormPrimitive>(primitives, filter, t);

    // Softmax: no new variables, pruning: the last primitive could not be the same.
    for (const auto& type: {SoftmaxC, SoftmaxH, SoftmaxW, SoftmaxHW}) {
        if (auto last_softmax = DynamicCast<SoftmaxPrimitive>(t->producer))
            if (last_softmax->type == type)
                continue;
        TryMakeAndPush<SoftmaxPrimitive>(primitives, filter, t, type);
    }

    // Element-wise: no new variables, pruning: the last primitive could not be the same.
    for (const auto& type: {Abs, Exp, Neg, Sin}) {
        if (auto element_wise = DynamicCast<ElementWisePrimitive>(t->producer)) {
            // Same type operators.
            if (element_wise->type == type)
                continue;
            // Special pruning for absolute.
            if (type == Abs and (element_wise->type == Exp or element_wise->type == Neg))
                continue;
        }
        TryMakeAndPush<ElementWisePrimitive>(primitives, filter, t, type);
    }

    // ChannelShuffle: no new variables, pruning: input could not be shuffled.
    if (DynamicCast<ChannelShufflePrimitive>(t->producer) == nullptr)
        TryMakeAndPush<ChannelShufflePrimitive>(primitives, filter, t);

    // Fold: no new variables.
    for (const auto& type: {FoldH, FoldW, FoldHW})
        for (const auto& arith_type: {FoldAvg, FoldMax})
            TryMakeAndPush<FoldPrimitive>(primitives, filter, t, type, arith_type);

    // Group: no new variables.
    for (const auto& type: {GroupByFactor, GroupAllChannels})
        TryMakeAndPush<GroupPrimitive>(primitives, filter, t, type);

    // Pool: no new variables.
    for (const auto& type: {PoolH, PoolW, PoolHW})
        TryMakeAndPush<PoolPrimitive>(primitives, filter, t, type);

    // Shift: no new variables.
    for (const auto& type: {ShiftH, ShiftW, ShiftHW})
        TryMakeAndPush<ShiftPrimitive>(primitives, filter, t, type);

    // Transpose: no new variables, pruning: input could not have been transposed.
    if (DynamicCast<TransposePrimitive>(t->producer) == nullptr)
        TryMakeAndPush<TransposePrimitive>(primitives, filter, t);
}

void PrimitiveFactory::GetPrimitiveApplies(const GraphSP &graph,
                                           std::vector<PrimitiveApply>& primitives,
                                           const TensorSP& lhs,
                                           const TensorSP& rhs,
                                           std::unordered_set<size_t>& filter,
                                           bool allow_dynamic_variables) {
    // Element-wise broadcasting operations.
    for (const auto& type: {BAdd, BSub, BMul}) {
        // Pruning for subtraction to zero.
        if (lhs == rhs and type == BSub)
            continue;
        // Pruning for multiplying 2.
        if (lhs == rhs and type == BAdd)
            continue;
        auto all_matches = BroadcastPrimitive::GetAllPossibleMatches(lhs, rhs, type);
        if (not all_matches.empty())
            TryPush(RandomChoose(all_matches), primitives, filter);
    }
}

} // namespace canvas
