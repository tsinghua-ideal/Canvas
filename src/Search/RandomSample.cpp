#include <ice-cream.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include "Canvas/Search/RandomSample.hpp"
#include "Canvas/Primitives/Factory.hpp"

// #define CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
// #define CANVAS_DEBUG_FAILED_COUNT
// #define CANVAS_DEBUG_PRINT_STATISTICS
// #define CANVAS_FORCE_RESET_DEBUG


namespace canvas {

namespace ba = boost::adaptors;

Solution TryRandomSample(const NetSpecsSP& net_specs,
                         bool allow_dynamic, bool add_relu_bn_after_fc,
                         const Range<int>& np_range, const Range<int>& fc_range) {
    // Random graph settings.
    int n_primitives = np_range.Random();
    int max_width = mw_range.Random();
    int expected_fc_count = std::min(fc_range.Random(), n_primitives);
    assert(expected_fc_count > 0 and expected_fc_count <= n_primitives);
    double fc_sample_possibility = static_cast<double>(expected_fc_count) / n_primitives;
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
    IC();
    IC(n_primitives, max_width, expected_fc_count);
#endif

    // Take actions.
    GraphSP graph = std::make_shared<Graph>();
    for (int i = 0; i < n_primitives; ++ i) {
        int width = graph->Width();

        // Determine whether we should reduce the width every step later.
        int n_steps_remaining = n_primitives - i;
        int n_steps_to_single_out = width - 1;
        // width > n_steps_remaining + 1.
        if (n_steps_to_single_out > n_steps_remaining) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_reduce_to_single_out = 0;
            IC(can_not_reduce_to_single_out ++);
#endif
            return {};
        }

        // Must reduce width.
        bool must_reduce_width = (n_steps_to_single_out == n_steps_remaining);

        // Already reach the maximum width limit.
        // Or next iteration (if expanded), remaining: n_steps_remaining - 1, to single: width,
        //  which is equivalent to width > n_steps_remaining - 1.
        bool could_not_expand_width = (width == max_width) or (width > n_steps_remaining - 1);

        // Random a primitive according to `must_reduce_width` and `could_not_expand_width`.
        std::vector<PrimitiveGenOptions> or_options;
        if (must_reduce_width) // Stricter constraints
            or_options.push_back(PrimitiveGenOptions::ReduceWidth());
        else if (could_not_expand_width)
            or_options.push_back(PrimitiveGenOptions::NotExpanding());

        // Random an action and apply.
        auto applies = PrimitiveFactory::GetPrimitiveApplies(graph, allow_dynamic);

        // Filter for no reasonable grouping number.
        if (net_specs->c_gcd == 1)
            applies = PrimitiveFactory::FilterPrimitiveApplies(applies, PrimitiveGenOptions::NoFactorGrouping());

        // Filter for topology pruning.
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
            if (apply.solution)
                auto substitution = apply.solution.value().substitution;
        } catch (const CanNotSolveDynamicVar& ex) {
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
            IC(ex);
#endif
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_solve_dyn_var = 0;
            IC(can_not_solve_dyn_var ++);
#endif
            return {};
        } // Failed if other exceptions.
    }

#ifdef CANVAS_FORCE_RESET_DEBUG
    // Debug (reset to depth-wise convolution)
    {
        graph = std::make_shared<Graph>();
        auto group = std::make_shared<GroupPrimitive>(graph->in, GroupAllChannels);
        graph->Apply(group);
        auto unfold = std::make_shared<UnfoldPrimitive>(group->outs[0]);
        graph->Apply(unfold);
        auto fc = std::make_shared<FCPrimitive>(unfold->outs[0], Variable(StaticVarPos::VC));
        graph->Apply(fc);
    }
#endif

    // FC constraints.
    if (not fc_range.Contains(graph->PrimitiveCount<FCPrimitive>())) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_satisfy_fc_count = 0;
        IC(can_not_satisfy_fc_count ++);
#endif
        return {};
    }

    // Pruning: channel shuffle must cooperate with grouping.
    if (graph->PrimitiveCount<ChannelShufflePrimitive>() > 0 and graph->PrimitiveCount<GroupPrimitive>() == 0) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_cooperate_channel_group = 0;
        IC(can_not_cooperate_channel_group ++);
#endif
        return {};
    }

    // For debug (filter primitives).
    // if (graph->PrimitiveCount<ReorderPrimitive>() == 0)
    //     return {};

    // Add BatchNorm and ReLU after FC (connected to the output).
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
        assert(out->shape.H() == Variable::StaticVar(StaticVarPos::VH));
        assert(out->shape.W() == Variable::StaticVar(StaticVarPos::VW));
        auto channel = out->shape.GCKK();
        assert(channel.DynamicVarCount() == 1); // Only C channel could have unsolved variables.
        assert(channel.SatisfyAssumption());
        auto v = Variable::StaticVar(StaticVarPos::VC) / channel.StaticFactor();
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

    // Add output primitive and fill the budget.
    assert(graph->Width() == 1);
    assert(out->shape.CouldBeReshapeToCHW());
    graph->ApplyOutput();

    // TODO: rewrite with new strategies.
    if (graph->DynamicVarCount() > 0)
        return {};

    // The final check, fill the solution with concise values.
    // TODO: g factor sampling.
    for (const auto& kernel: net_specs->kernel_specs)
        if (not graph->AlgebraCheck({1, kernel.c, kernel.h, kernel.w}))
            return {};
    return {net_specs, graph};
}

Solution RandomSample(const NetSpecsSP& net_specs,
                      bool allow_dynamic, bool add_relu_bn_after_fc,
                      const Range<int>& np_range, const Range<int>& fc_range, canvas_timeval_t timeout) {
    auto start_time_point = std::chrono::system_clock::now();
    int times = 0;
    while (true) {
        ++ times;
        auto solution = TryRandomSample(net_specs,
                                        allow_dynamic, add_relu_bn_after_fc,
                                        np_range, fc_range);
        if (not solution.Empty())
            return solution;

        auto current_time_point = std::chrono::system_clock::now();
        if (timeout != std::chrono::seconds(0) and current_time_point - start_time_point > timeout)
            break;
    }
    return {};
}

} // namespace canvas
