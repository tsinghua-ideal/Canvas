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

    // FC: the channel could be a new variable.
    // Could not have dynamic variables in G, consider grouping-all primitive.
    // We may add an extra primitive for only mapping H and W, but remapping into spatial dimensions.
    if (t->shape.IsChannelSpatial()) {
        auto c = Variable::StaticVar(StaticVarPos::VC);
        if ((c / t->shape.Channel()->G()).MaybeInteger())
            MakeAndPush<FCPrimitive>(primitives, options, t, c);
        if (not unused_indices.empty())
            MakeAndPush<FCPrimitive>(primitives, options, t, Variable::DynamicVar(unused_indices.front()));
    }

    // Convolution: the channel could be a new variable.
    // We do not put too much convolution primitives in kernel, as much as possible to use FC.
    if (t->shape.IsChannelSpatial() and not unused_indices.empty() and
        not options.kernel_sizes.empty() and not options.dilated_sizes.empty() and
        t->shape.Channel()->KH().Empty() and t->shape.Channel()->KW().Empty() and
        not (t->shape.Spatial()->H().Empty() and t->shape.Spatial()->W().Empty())) {
        auto PushConv = [&](int kh, int kw, int dh, int dw) {
            MakeAndPush<ConvolutionPrimitive>(primitives, options, t,
                                              Variable::DynamicVar(unused_indices[0]),
                                              kh, kw, dh, dw, false);
            auto c = Variable::StaticVar(StaticVarPos::VC);
            if ((c / t->shape.Channel()->G()).MaybeInteger())
                MakeAndPush<ConvolutionPrimitive>(primitives, options, t, c, kh, kw, dh, dw, false);
            if (t->shape.Channel()->G().Empty()) {
                MakeAndPush<ConvolutionPrimitive>(primitives, options, t,
                                                  Variable::DynamicVar(unused_indices[0]),
                                                  kh, kw, dh, dw, true);
                if ((c / t->shape.Channel()->C()).MaybeInteger())
                    MakeAndPush<ConvolutionPrimitive>(primitives, options, t, c, kh, kw, dh, dw, true);
            }

        };
        int k = RandomChoose(options.kernel_sizes), d = RandomChoose(options.dilated_sizes);
        PushConv(k, 1, d, 1), PushConv(1, k, 1, d), PushConv(k, k, d, d);
    }

    // Mix: mapping dimensions into dimensions.
    auto PushMix = [&](const std::vector<Shape::Index>& indices) {
        if (indices.empty())
            return;
        int next_unused = 0;
        std::vector<Variable> new_dims;
        for (int i = 0; i < indices.size(); ++ i) {
            if (next_unused < unused_indices.size() and MakeChoice(kMixPossibility))
                new_dims.push_back(Variable::DynamicVar(unused_indices[next_unused])), next_unused ++;
            else
                new_dims.emplace_back();
        }
        MakeAndPush<MixPrimitive>(primitives, options, t, indices, new_dims);
    };

    // Mix case 1: mapping single meta shape.
    for (int d = 0; d < 2; ++ d) {
        std::vector<Shape::Index> indices;
        for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
            auto index = Shape::Index(d, k);
            // TODO: may add group merging primitive or add merging option.
            if (DynamicCast<ChannelShape>(t->shape.dims[d]) and k == ChannelShape::PG)
                continue;
            if (not t->shape[index].Empty())
                indices.push_back(index);
        }
        PushMix(indices);
    }

    // Mix case 2: randomly select some.
    {
        auto non_empty_indices = t->shape.GetNonEmptyIndices();
        for (int i = 0; i < kMixOpportunities; ++ i)
            PushMix(RandomSubset(non_empty_indices));
    }

    // Activation: no new variables, pruning: no double ReLU.
    for (const auto &type: {GeLU, ReLU, Sigmoid, TanH}) {
        if (auto last_activation = DynamicCast<ActivationPrimitive>(t->producer))
            if (type == ReLU and (last_activation->type == ReLU or last_activation->type == Sigmoid))
                continue;
        MakeAndPush<ActivationPrimitive>(primitives, options, t, type);
    }

    // Element-wise: no new variables, pruning: double abs/neg.
    for (const auto& type: {Abs, Exp, Neg, Sin, Sqrt, Sqr}) {
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
            // Pruning for sqrt/sqr.
            if (type == Sqrt and last_element_wise->type == Sqr)
                continue;
            if (type == Sqr and last_element_wise->type == Sqrt)
                continue;
        }
        // Pruning for activation.
        if (auto last_activation = DynamicCast<ActivationPrimitive>(t->producer))
            if (type == Abs and (last_activation->type == Sigmoid or last_activation->type == ReLU))
                continue;
        MakeAndPush<ElementWisePrimitive>(primitives, options, t, type);
    }

    // Fold: no new variables.
    for (const auto& type: {FoldAvg, FoldMax}) {
        for (int d = 0; d < 2; ++ d) {
            for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
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
            for (const auto& factor: channel->C().Numerator().GetAllFactors()) {
                if (factor.Empty() or not (channel->C() / factor).MaybeInteger())
                    continue;
                MakeAndPush<GroupPrimitive>(primitives, options, t, d, factor);
            }
            if (not unused_indices.empty())
                MakeAndPush<GroupPrimitive>(primitives, options, t, d, Variable::DynamicVar(unused_indices.front()));
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
        for (int i = 0; i < kScaleOpportunities; ++ i) {
            auto subset = RandomSubset(merged);
            if (not subset.empty())
                MakeAndPush<ScalePrimitive>(primitives, options, t, subset);
        }
    }

    // Shift: no new variables.
    for (int s: options.shift_sizes) {
        for (int d = 0; d < 2; ++ d) {
            for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
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

    // Softmax: no new variables.
    for (int d = 0; d < 2; ++ d) {
        for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
            auto index = Shape::Index(d, k);
            if (not t->shape[index].Empty())
                MakeAndPush<SoftmaxPrimitive>(primitives, options, t, index);
        }
    }
}

void PrimitiveFactory::GetPrimitiveApplies(const GraphSP &graph,
                                           std::vector<PrimitiveApply>& primitives,
                                           const TensorSP& lhs, const TensorSP& rhs,
                                           const PrimitiveOptions& options) {
    // Element-wise broadcasting operations.
    for (const auto& type: {BAdd, BSub, BMul, BMax}) {
        // Pruning.
        if (lhs == rhs and (type == BSub or type == BMax or type == BAdd))
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
