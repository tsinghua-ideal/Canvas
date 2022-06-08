#include "Canvas/Core/Graph.hpp"
#include "Canvas/Primitives/Factory.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

void PrimitiveFilter::BuildFilters(const std::string& allowed_str, const std::string& forbidden_str) {
    auto Split = [](const std::string& str, std::vector<std::string>& vec) {
        boost::algorithm::split(vec, boost::algorithm::to_lower_copy(str),
                                boost::is_any_of("\t ,"),
                                boost::token_compress_on);
    };
    if (not allowed_str.empty())
        Split(allowed_str, allowed_filter);
    if (not forbidden_str.empty())
        Split(forbidden_str, forbidden_filter);
}

bool PrimitiveFilter::Filter(const PrimitiveSP& p) const {
    // Filter by width.
    int delta_width = static_cast<int>(p->outs.size());
    for (const auto& t: ToUnorderedSet(p->ins))
        delta_width -= static_cast<int>(t->consumers.empty()); // Not applied yet.
    if (delta_width > max_delta_width)
        return true;

    // Filter by hash.
    if (hash_filter.count(p->Hash(true)))
        return true;

    // Optimize FC.
    if (add_relu_bn_after_fc)
        if (auto fc = DynamicCast<FCPrimitive>(p))
            fc->with_norm = fc->with_relu = true;

    // Filter by output.
    if (p->outs[0]->shape.IsAllStatic() and not p->outs[0]->shape.CouldBeReshapeToCHW())
        return true;

    // Filter by type name.
    auto name = boost::algorithm::to_lower_copy(p->name);
    auto IsPrefix = [=](const std::string& filter) -> bool {
        return name.rfind(filter, 0) == 0;
    };
    if (not allowed_filter.empty())
        if (std::none_of(allowed_filter.begin(), allowed_filter.end(), IsPrefix))
            return true;
    if (not forbidden_filter.empty())
        if (std::any_of(forbidden_filter.begin(), forbidden_filter.end(), IsPrefix))
            return true;
    return false;
}

std::vector<PrimitiveApply> PrimitiveFactory::RescalePossibilities(const std::vector<PrimitiveApply>& applies) {
    // TODO: re-design this function.
    std::vector<PrimitiveApply> applies_copy = applies;
    RandomShuffle(applies_copy);

    // Push all the first in the shuffle into the candidates.
    std::vector<PrimitiveApply> new_applies;
    std::set<std::string> filter;
    for (const auto& apply: applies_copy) {
        if (not filter.count(apply.primitive->name)) {
            filter.insert(apply.primitive->name);
            new_applies.push_back(apply);
        }
    }
    return new_applies;
}

std::vector<PrimitiveApply> PrimitiveFactory::GetPrimitiveApplies(const GraphSP& graph,
                                                                  const PrimitiveFilter& filter) {
    std::vector<PrimitiveApply> primitives;

    // Get primitives with one input.
    for (const auto& t: graph->tensors)
        GetPrimitiveApplies(graph, primitives, t, filter);

    // Get primitives with two inputs.
    for (const auto& lhs: graph->tensors)
        for (const auto& rhs: graph->tensors)
            GetPrimitiveApplies(graph, primitives, lhs, rhs, filter);

    return primitives;
}

void PrimitiveFactory::GetPrimitiveApplies(const GraphSP &graph,
                                           std::vector<PrimitiveApply>& primitives,
                                           const TensorSP& t,
                                           const PrimitiveFilter& filter) {
    assert(t->producer);
    // The next variable index.
    auto next_index_opt = graph->NextUnusedDynamicVarIndex();

    // FC: The channel could be a new variable.
    // Could not have dynamic variables in G, consider grouping-all primitive.
    if (t->shape.G().IsStatic()) {
        if (next_index_opt.has_value())
            TryMakeAndPush<FCPrimitive>(primitives, filter, t, Variable::DynamicVar(next_index_opt.value()));
        else
            TryMakeAndPush<FCPrimitive>(primitives, filter, t); // By default, we retain the channel number and fold spatial information.
    }

    // Activation: no new variables, pruning: no double ReLU.
    for (const auto &type: {GeLU, ReLU, Sigmoid, TanH}) {
        if (auto last_activation = DynamicCast<ActivationPrimitive>(t->producer))
            if (type == ReLU and last_activation->type == ReLU)
                continue;
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

    // Element-wise: no new variables, pruning: double abs/neg.
    for (const auto& type: {Abs, Exp, Neg, Sin}) {
        if (auto last_element_wise = DynamicCast<ElementWisePrimitive>(t->producer)) {
            // Double abs.
            if (type == Abs and last_element_wise->type == Abs)
                continue;
            // Double neg.
            if (type == Neg and last_element_wise->type == Neg)
                continue;
            // Special pruning for absolute.
            if (type == Abs and (last_element_wise->type == Exp or last_element_wise->type == Neg))
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

    // Unfold: no new variables.
    for (const auto& type: {UnfoldH, UnfoldW, UnfoldHW}) {
        // TODO: support option randomness.
        TryMakeAndPush<UnfoldPrimitive>(primitives, filter, t, 3, 1, type);
    }

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
                                           const TensorSP& lhs, const TensorSP& rhs,
                                           const PrimitiveFilter& filter) {
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
