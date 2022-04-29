#include <ice-cream.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include "Canvas/Algorithms/Count.hpp"
#include "Canvas/Algorithms/Fill.hpp"
#include "Canvas/Search/RandomSample.hpp"
#include "Canvas/Primitives/Factory.hpp"

// #define CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
// #define CANVAS_DEBUG_FAILED_COUNT
// #define CANVAS_DEBUG_PRINT_STATISTICS
// #define CANVAS_FORCE_RESET_DEBUG


namespace canvas {

namespace ba = boost::adaptors;

Solution TryRandomSample(const NetSpecsSP& net_specs,
                         bool allow_dynamic, bool force_irregular, bool add_relu_bn_after_fc,
                         const Range<int>& np_range, const Range<int>& fc_range) {
    // Random graph settings
    int n_primitives = np_range.Random();
    int max_width = mw_range.Random();
    int expected_fc_count = std::min(fc_range.Random(), n_primitives);
    assert(expected_fc_count > 0 and expected_fc_count <= n_primitives);
    double fc_sample_possibility = static_cast<double>(expected_fc_count) / n_primitives;
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
    IC();
    IC(n_primitives, max_width, n_fc);
#endif

    // Take actions
    GraphSP graph = std::make_shared<Graph>();
    bool irregular_apply = false;
    for (int i = 0; i < n_primitives; ++ i) {
        int width = graph->Width();

        // Determine whether we should reduce the width every step later
        int n_steps_remaining = n_primitives - i;
        int n_steps_to_single_out = width - 1;
        // width > n_steps_remaining + 1
        if (n_steps_to_single_out > n_steps_remaining) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_reduce_to_single_out = 0;
            IC(can_not_reduce_to_single_out ++);
#endif
            return {};
        }

        // Must reduce width
        bool must_reduce_width = (n_steps_to_single_out == n_steps_remaining);

        // Already reach the maximum width limit
        // Or next iteration (if expanded), remaining: n_steps_remaining - 1, to single: width,
        //  which is equivalent to width > n_steps_remaining - 1
        bool could_not_expand_width = (width == max_width) or (width > n_steps_remaining - 1);

        // Random a primitive according to `must_reduce_width` and `could_not_expand_width`
        std::vector<PrimitiveGenOptions> or_options;
        if (must_reduce_width) // Stricter constraints
            or_options.push_back(PrimitiveGenOptions::ReduceWidth());
        else if (could_not_expand_width)
            or_options.push_back(PrimitiveGenOptions::NotExpanding());

        // Random an action and apply
        auto applies = PrimitiveFactory::GetPrimitiveApplies(graph, allow_dynamic);

        // Filter for all k == 1
        if (net_specs->no_neighbor_involved)
            applies = PrimitiveFactory::FilterPrimitiveApplies(applies, PrimitiveGenOptions::NoNeighborInvolved());

        // Filter for no reasonable grouping number
        assert(net_specs->c_gcd > 0);
        if (net_specs->c_gcd == 1)
            applies = PrimitiveFactory::FilterPrimitiveApplies(applies, PrimitiveGenOptions::NoFactorGrouping());

        // Filter for topology pruning
        if (not or_options.empty())
            applies = PrimitiveFactory::FilterPrimitiveApplies(applies, or_options);
        if (n_steps_remaining == 1)
            applies = PrimitiveFactory::FilterPrimitiveAppliesForOutput(applies);

        if (applies.empty()) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int no_available_applies = 0;
            IC(no_available_applies ++);
#endif
            return {};
        }

        auto fc_applies = PrimitiveFactory::FilterPrimitiveApplies(applies, PrimitiveGenOptions::FC());
        if (not fc_applies.empty() and MakeChoice(fc_sample_possibility))
            applies = fc_applies;
        else
            applies = PrimitiveFactory::RescalePossibilities(applies, true);
        if (applies.empty())
            return {};

        auto apply = RandomChoose(applies);
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
        IC(width, n_steps_remaining, must_reduce_width, could_not_expand_width, apply);
#endif
        try {
            graph->Apply(apply);
            if (apply.solution) {
                auto substitution = apply.solution.value().substitution;
                irregular_apply |= substitution.IsIrregular(false);
            }
            for (const auto& t: apply.primitive->outs)
                irregular_apply |= t->shape.H().Empty() or t->shape.W().Empty();
        } catch (const CanNotSolveDynamicVar& ex) {
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
            IC(ex);
#endif
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_solve_dyn_var = 0;
            IC(can_not_solve_dyn_var ++);
#endif
            return {};
        } // Failed if other exceptions
    }

    // Pruning: irregular apply
    if (force_irregular and not irregular_apply) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_satisfy_irregular = 0;
        IC(can_not_satisfy_irregular ++);
#endif
        return {};
    }

#ifdef CANVAS_FORCE_RESET_DEBUG
    // Debug (reset to depth-wise convolution)
    {
        graph = std::make_shared<Graph>();
        auto group = std::make_shared<GroupPrimitive>(graph->in, GroupAllChannels);
        graph->Apply(group);
        auto unfold = std::make_shared<UnfoldPrimitive>(group->outs[0]);
        graph->Apply(unfold);
        auto fc = std::make_shared<FCPrimitive>(unfold->outs[0], Variable(StaticVar::VC));
        graph->Apply(fc);
    }
#endif

    // FC constraints
    if (not fc_range.Contains(graph->PrimitiveCount<FCPrimitive>())) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_satisfy_fc_count = 0;
        IC(can_not_satisfy_fc_count ++);
#endif
        return {};
    }

    // Pruning: channel shuffle must cooperate with grouping
    if (graph->PrimitiveCount<ChannelShufflePrimitive>() > 0 and graph->PrimitiveCount<GroupPrimitive>() == 0) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_cooperate_channel_group = 0;
        IC(can_not_cooperate_channel_group ++);
#endif
        return {};
    }

    // Pruning: must have neighbor's information
    int n_neighbor_h = 0, n_neighbor_w = 0;
    for (const auto& p: graph->primitives) {
        if (auto unfold = DynamicCast<UnfoldPrimitive>(p)) {
            n_neighbor_h += (unfold->type == UnfoldH or unfold->type == UnfoldHW);
            n_neighbor_w += (unfold->type == UnfoldW or unfold->type == UnfoldHW);
        }
        if (auto fold = DynamicCast<FoldPrimitive>(p)) {
            n_neighbor_h += (fold->type == FoldH or fold->type == FoldHW);
            n_neighbor_w += (fold->type == FoldW or fold->type == FoldHW);
        }
        if (auto shift = DynamicCast<ShiftPrimitive>(p)) {
            n_neighbor_h += (shift->type == ShiftH or shift->type == ShiftHW);
            n_neighbor_w += (shift->type == ShiftW or shift->type == ShiftHW);
        }
        if (auto softmax = DynamicCast<SoftmaxPrimitive>(p)) {
            n_neighbor_h += (softmax->type == SoftmaxH or softmax->type == SoftmaxHW);
            n_neighbor_w += (softmax->type == SoftmaxW or softmax->type == SoftmaxHW);
        }
        if (auto pool = DynamicCast<PoolPrimitive>(p)) {
            n_neighbor_h += (pool->type == PoolH or pool->type == PoolHW);
            n_neighbor_w += (pool->type == PoolW or pool->type == PoolHW);
        }
    }
    if (not (n_neighbor_h > 0 and n_neighbor_w > 0)) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_satisfy_neighbor = 0;
        IC(can_not_satisfy_neighbor ++);
#endif
        return {};
    }

    // Pruning: reordering topologies
    {
        std::map<TensorSP, int> reorder_h, reorder_w;
        for (const auto& p: graph->primitives) {
            if (auto reorder = DynamicCast<ReorderPrimitive>(p)) {
                int delta = reorder->inverse ? -1 : 1;
                reorder_h[reorder->outs[0]] = reorder_h[reorder->ins[0]];
                reorder_w[reorder->outs[0]] = reorder_w[reorder->ins[0]];
                if (reorder->type == ReorderH or reorder->type == ReorderHW)
                    reorder_h[reorder->outs[0]] += delta;
                if (reorder->type == ReorderW or reorder->type == ReorderHW)
                    reorder_w[reorder->outs[0]] += delta;
                if (reorder_h[reorder->outs[0]] < 0 or reorder_w[reorder->outs[0]] < 0) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
                    static int reorder_too_early = 0;
                    IC(reorder_too_early ++);
#endif
                    return {};
                }
            } else {
                int in_reorder_h = 0, in_reorder_w = 0;
                if (not p->ins.empty()) {
                    // The order of reordering should be aligned
                    in_reorder_h = reorder_h[p->ins[0]];
                    in_reorder_w = reorder_w[p->ins[0]];
                    for (const auto& t: p->ins)
                        if (reorder_h[t] != in_reorder_h or reorder_w[t] != in_reorder_w) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
                            static int reorder_not_aligned = 0;
                            IC(reorder_not_aligned ++);
#endif
                            return {};
                        }
                }
                for (const auto& t: p->outs) {
                    reorder_h[t] = in_reorder_h;
                    reorder_w[t] = in_reorder_w;
                }
            }
        }
        auto out = graph->Out();
        if (reorder_h[out] != 0 or reorder_w[out] != 0) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_out_of_order_output = 0;
            IC(can_not_out_of_order_output ++);
#endif
            return {};
        }
    }

    // For debug (filter primitives)
    // if (graph->PrimitiveCount<ReorderPrimitive>() == 0)
    //     return {};

    // Add BatchNorm and ReLU after FC (connected to the output)
    if (add_relu_bn_after_fc) {
        std::set<TensorSP> has_fc_later;
        for (const auto& p: ba::reverse(graph->primitives)) {
            bool any_successor = std::any_of(p->outs.begin(), p->outs.end(), [has_fc_later](const auto& t) -> bool {
                return has_fc_later.count(t);
            });
            if (auto fc = DynamicCast<FCPrimitive>(p)) {
                if (any_successor) {
                    auto out_shape = fc->outs.front()->shape;
                    fc->with_norm = not out_shape.H().Empty() or not out_shape.W().Empty();
                    fc->with_relu = true;
                }
                has_fc_later.insert(fc->ins.front());
            } else if (any_successor) {
                for (const auto& t: p->ins)
                    has_fc_later.insert(t);
            }
        }
    }

#ifdef CANVAS_DEBUG_PRINT_STATISTICS
    static int n_total_sampled = 0, n_out_with_variable = 0;
    n_total_sampled ++;
#endif
    auto out = graph->Out();
    if (not out->shape.IsAllStatic()) {
        if (out->shape.H().Empty() or out->shape.W().Empty()) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int illegal_output_shape = 0;
            IC(illegal_output_shape ++);
#endif
            return {};
        }
        assert(out->shape.H() == Variable(StaticVar::VH));
        assert(out->shape.W() == Variable(StaticVar::VW));
        auto channel = out->shape.GCKK();
        assert(channel.DynamicVarCount() == 1); // Only C channel could have unsolved variables
        assert(channel.SatisfyAssumption());
        auto v = Variable(StaticVar::VC) / channel.StaticFactor();
        assert(v.IsStatic());
        try {
            graph->SolveDynamicVar(VarSolution(channel.GetOnlyDynamicVar(), v));
        } catch (const CanNotSolveDynamicVar& ex) {
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
            IC(ex);
#endif
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_solve_dynamic_var = 0;
            IC(can_not_solve_dynamic_var ++);
#endif
            return {};
        }
#ifdef CANVAS_DEBUG_PRINT_STATISTICS
        n_out_with_variable ++;
#endif
    }
#ifdef CANVAS_DEBUG_PRINT_STATISTICS
        double r_out_with_variable = static_cast<double>(n_out_with_variable) / static_cast<double>(n_total_sampled);
    IC(n_total_sampled, n_out_with_variable, r_out_with_variable);
#endif

    // Add output primitive and fill the budget
    assert(graph->Width() == 1);
    assert(out->shape.CouldBeReshapeToCHW());
    try {
        graph->ApplyOutput();
        size_t g = 1;
        if (not net_specs->c_gcd_factors.empty())
            g = RandomChoose(net_specs->c_gcd_factors);
        auto preferences = HeuristicPreferences(g);
        auto fills = GetFullFillsUnderBudget(net_specs, graph, preferences);
        // Success
        return {net_specs, graph, preferences, fills};
    } catch (const CanNotFillDynamicVariables& ex) {
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
        IC(ex);
#endif
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_fill_dynamic_var_or_reach_budget = 0;
        IC(can_not_fill_dynamic_var_or_reach_budget ++);
#endif
        return {};
    }
}

Solution RandomSample(const NetSpecsSP& net_specs,
                      bool allow_dynamic, bool force_irregular, bool add_relu_bn_after_fc,
                      const Range<int>& np_range, const Range<int>& fc_range, canvas_timeval_t timeout) {
    auto start_time_point = std::chrono::system_clock::now();
    int times = 0;
    while (true) {
        ++ times;
        auto solution = TryRandomSample(net_specs,
                                        allow_dynamic, force_irregular, add_relu_bn_after_fc,
                                        np_range, fc_range);
        if (not solution.Empty()) {
            auto flops = NetFLOPsCount(solution), ps = NetPsCount(solution);
            if (net_specs->SatisfyFlopsPsRange(flops, ps))
                return solution;
        }
        auto current_time_point = std::chrono::system_clock::now();
        if (timeout != std::chrono::seconds(0) and current_time_point - start_time_point > timeout)
            break;
    }
    return {};
}

} // End namespace canvas
