#include "Canvas/Core/Graph.hpp"
#include "Canvas/Primitives/Factory.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

void PrimitiveOptions::BuildFilters(const std::string& allowed_str, const std::string& forbidden_str) {
    if (not allowed_str.empty())
        Split(allowed_str, allowed_filter);
    if (not forbidden_str.empty())
        Split(forbidden_str, forbidden_filter);
}

bool PrimitiveOptions::Filter(const PrimitiveApply& pa) const {
    auto p = pa.primitive;

    // Filter by width.
    int delta_width = static_cast<int>(p->outs.size());
    for (const auto& t: ToUnorderedSet(p->ins))
        delta_width -= static_cast<int>(t->consumers.empty()); // Not applied yet.
    if (delta_width > max_delta_width)
        return true;

    // Too large tensor dimensions.
    for (const auto& t: p->outs)
        if (t->shape.Continuous().size() > kMaxNumberDimensions)
            return true;

    // Filter by hash.
    if (hash_filter.count(p->Hash(true)))
        return true;

    // Optimize FC.
    if (add_relu_bn_after_fc)
        if (auto fc = DynamicCast<FCPrimitive>(p))
            fc->with_norm = fc->with_relu = true;

    // Filter by output.
    if (output_filter) {
        auto pi = p->outs[0]->shape;
        if (pa.solution.has_value())
            pi.SolveDynamicVar(pa.solution.value());
        if (pi.IsStatic() and not pi.CouldBeReshapedToCHW())
            return true;
    }

    // Filter by type name.
    auto name = ToLowerCopy(p->name);
    auto IsPrefixImpl = [&name](const std::string& filter) -> bool {
        return IsPrefix(name, filter);
    };

    if (not allowed_filter.empty())
        if (std::none_of(allowed_filter.begin(), allowed_filter.end(), IsPrefixImpl))
            return true;
    if (not forbidden_filter.empty())
        if (std::any_of(forbidden_filter.begin(), forbidden_filter.end(), IsPrefixImpl))
            return true;
    return false;
}

std::vector<PrimitiveApply> PrimitiveFactory::RescalePossibilities(const std::vector<PrimitiveApply>& applies) {
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
                                                                  const PrimitiveOptions& options) {
    std::vector<PrimitiveApply> primitives;

    // Get primitives with one input.
    for (const auto& t: graph->tensors)
        GetPrimitiveApplies(graph, primitives, t, options);

    // Get primitives with two inputs.
    for (const auto& lhs: graph->tensors)
        for (const auto& rhs: graph->tensors)
            GetPrimitiveApplies(graph, primitives, lhs, rhs, options);

    return primitives;
}

void PrimitiveFactory::GetPrimitiveApplies(const GraphSP &graph,
                                           std::vector<PrimitiveApply>& primitives,
                                           const TensorSP& t,
                                           const PrimitiveOptions& options) {
    assert(t->producer);
    // The next variable index.
    auto unused_indices = graph->UnusedDynamicVarIndices();

    // TODO: add einsum primitive for matrix multiplication, a simplified version: original dot primitive.

    // FC: the channel could be a new variable.
    // Could not have dynamic variables in G, consider grouping-all primitive.
    // Norm/ReLU optimization will be added in the filter.
    // TODO: support flexible FC remapping into C, consider (C, K, K, H, W) five dimensions.
    // TODO: support multiple variable solving.
    // We may add an extra primitive for only mapping H and W, but remapping into spatial dimensions.
    // An edge case to notice: [x_0, 1, H, W] -> [x_0, x_1/x_0, H, W]
    if (t->shape.IsChannelSpatial() and not unused_indices.empty() and DynamicCast<FCPrimitive>(t->producer) == nullptr)
        MakeAndPush<FCPrimitive>(primitives, options, t, Variable::DynamicVar(unused_indices.front()));

    // Convolution: the channel could be a new variable.
    // We do not put too much convolution primitives in kernel, as much as possible to use FC.
    if (t->shape.IsChannelSpatial() and not unused_indices.empty() and
        not options.kernel_sizes.empty() and not options.dilated_sizes.empty() and
        t->shape.Channel()->KH().Empty() and t->shape.Channel()->KW().Empty() and
        not (t->shape.Spatial()->H().Empty() and t->shape.Spatial()->W().Empty())) {
        int kh = RandomChoose(options.kernel_sizes), kw = RandomChoose(options.kernel_sizes);
        int dh = RandomChoose(options.dilated_sizes), dw = RandomChoose(options.dilated_sizes);
        MakeAndPush<ConvolutionPrimitive>(primitives, options, t,
                                          Variable::DynamicVar(unused_indices[0]),
                                          Variable(), kh, kw, dh, dw);
        if (unused_indices.size() > 1) {
            kh = RandomChoose(options.kernel_sizes), kw = RandomChoose(options.kernel_sizes);
            dh = RandomChoose(options.dilated_sizes), dw = RandomChoose(options.dilated_sizes);
            MakeAndPush<ConvolutionPrimitive>(primitives, options, t,
                                              Variable::DynamicVar(unused_indices[0]),
                                              Variable::DynamicVar(unused_indices[1]),
                                              kh, kw, dh, dw);
        }
    }

    // Activation: no new variables, pruning: no double ReLU.
    for (const auto &type: {GeLU, ReLU, Sigmoid, TanH}) {
        if (auto last_activation = DynamicCast<ActivationPrimitive>(t->producer))
            if (type == ReLU and (last_activation->type == ReLU or last_activation->type == Sigmoid))
                continue;
        MakeAndPush<ActivationPrimitive>(primitives, options, t, type);
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
        if (auto last_activation = DynamicCast<ActivationPrimitive>(t->producer))
            if (type == Abs and (last_activation->type == Sigmoid or last_activation->type == ReLU))
                continue;
        MakeAndPush<ElementWisePrimitive>(primitives, options, t, type);
    }

    // Fold: no new variables.
    for (const auto& type: {FoldAvg, FoldMax}) {
        for (int d = 0; d < 2; ++ d) {
            int max_dims = DynamicCast<ChannelShape>(t->shape.dims[d]) ?
                    ChannelShape::kMaxChannelDims : SpatialShape::kMaxSpatialDims;
            for (int k = 0; k < max_dims; ++ k) {
                auto index = Shape::Index(d, k);
                if (not t->shape[index].Empty())
                    MakeAndPush<FoldPrimitive>(primitives, options, t, std::vector<Shape::Index>({index}), type);
            }

            // Build extra KH/KW double folding primitives.
            if (DynamicCast<ChannelShape>(t->shape.dims[d])) {
                auto index_kh = Shape::Index(d, ChannelShape::Index::PKH);
                auto index_kw = Shape::Index(d, ChannelShape::Index::PKW);
                if (not t->shape[index_kh].Empty() and not t->shape[index_kw].Empty()) {
                    auto indices = std::vector<Shape::Index>({index_kh, index_kw});
                    MakeAndPush<FoldPrimitive>(primitives, options, t, indices, type);
                }
            }

            // Build extra H/W double folding primitives.
            if (DynamicCast<SpatialShape>(t->shape.dims[d])) {
                auto index_h = Shape::Index(d, SpatialShape::Index::PH);
                auto index_w = Shape::Index(d, SpatialShape::Index::PW);
                if (not t->shape[index_h].Empty() and not t->shape[index_w].Empty()) {
                    auto indices = std::vector<Shape::Index>({index_h, index_w});
                    MakeAndPush<FoldPrimitive>(primitives, options, t, indices, type);
                }
            }
        }
    }

    // Unfold: no new variables.
    if (t->shape.IsChannelSpatial()) {
        auto channel = t->shape.Channel();
        auto spatial = t->shape.Spatial();
        for (const auto& type: {UnfoldH, UnfoldW, UnfoldHW}) {
            if ((type == UnfoldH or type == UnfoldHW) and (not channel->KH().Empty() or spatial->H().Empty()))
                continue;
            if ((type == UnfoldW or type == UnfoldHW) and (not channel->KW().Empty() or spatial->W().Empty()))
                continue;
            for (int k: options.kernel_sizes)
                for (int d: options.dilated_sizes)
                    MakeAndPush<UnfoldPrimitive>(primitives, options, t, k, d, type);
        }
    }

    // Group: no new variables.
    for (int d = 0; d < 2; ++ d) {
        if (auto channel = DynamicCast<ChannelShape>(t->shape.dims[d])) {
            if (not channel->G().Empty())
                continue;
            if ((channel->C() / Variable::StaticVar(StaticVarPos::VG)).MaybeInteger())
                MakeAndPush<GroupPrimitive>(primitives, options, t, d, GroupType::GroupByFactor);
            if (not channel->CKK().Empty())
                MakeAndPush<GroupPrimitive>(primitives, options, t, d, GroupType::GroupAllChannels);
        }
    }

    // Scale: no new variables.
    {
        std::vector<Shape::Index> dims[2];
        for (int d = 0; d < 2; ++ d) {
            for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
                auto index = Shape::Index(d, k);
                if (not t->shape[index].Empty())
                    dims[d].push_back(index);
            }
        }
        // Scale first, second, all and random of them.
        if (not dims[0].empty())
            MakeAndPush<ScalePrimitive>(primitives, options, t, dims[0]);
        if (not dims[1].empty())
            MakeAndPush<ScalePrimitive>(primitives, options, t, dims[1]);
        auto merged = Merge(dims[0], dims[1]);
        if (not dims[0].empty() and not dims[1].empty())
            MakeAndPush<ScalePrimitive>(primitives, options, t, merged);
        auto subset = RandomSubset(merged);
        if (not subset.empty())
            MakeAndPush<ScalePrimitive>(primitives, options, t, subset);
    }

    // Shift: no new variables.
    for (int s: options.shift_sizes) {
        for (int d = 0; d < 2; ++ d) {
            int max_dims = DynamicCast<ChannelShape>(t->shape.dims[d]) ?
                           ChannelShape::kMaxChannelDims : SpatialShape::kMaxSpatialDims;
            for (int k = 0; k < max_dims; ++ k) {
                auto index = Shape::Index(d, k);
                if (not t->shape[index].Empty())
                    MakeAndPush<ShiftPrimitive>(primitives, options, t, std::vector<Shape::Index>({index}), s);
            }

            // Build extra KH/KW double shifting primitives.
            if (DynamicCast<ChannelShape>(t->shape.dims[d])) {
                auto index_kh = Shape::Index(d, ChannelShape::Index::PKH);
                auto index_kw = Shape::Index(d, ChannelShape::Index::PKW);
                if (not t->shape[index_kh].Empty() and not t->shape[index_kw].Empty()) {
                    auto indices = std::vector<Shape::Index>({index_kh, index_kw});
                    MakeAndPush<ShiftPrimitive>(primitives, options, t, indices, s);
                }
            }

            // Build extra H/W double shifting primitives.
            if (DynamicCast<SpatialShape>(t->shape.dims[d])) {
                auto index_h = Shape::Index(d, SpatialShape::Index::PH);
                auto index_w = Shape::Index(d, SpatialShape::Index::PW);
                if (not t->shape[index_h].Empty() and not t->shape[index_w].Empty()) {
                    auto indices = std::vector<Shape::Index>({index_h, index_w});
                    MakeAndPush<ShiftPrimitive>(primitives, options, t, indices, s);
                }
            }
        }
    }
}

void PrimitiveFactory::GetPrimitiveApplies(const GraphSP &graph,
                                           std::vector<PrimitiveApply>& primitives,
                                           const TensorSP& lhs, const TensorSP& rhs,
                                           const PrimitiveOptions& options) {
    // Element-wise broadcasting operations.
    for (const auto& type: {BAdd, BSub, BMul}) {
        // Pruning for subtraction to zero.
        if (lhs == rhs and type == BSub)
            continue;
        // Pruning for multiplying 2.
        if (lhs == rhs and type == BAdd)
            continue;
        auto all_matches = BroadcastPrimitive::GetAllPossibleMatches(lhs, rhs, type, 1);
        if (not all_matches.empty())
            Push(all_matches[0], primitives, options);
    }

    // Batch matrix multiplication.
    for (const auto& pa: MatrixMultiplicationPrimitive::GetAllPossibleMatches(lhs, rhs))
        Push(pa, primitives, options);
}

} // namespace canvas
