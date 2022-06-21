#include <ice-cream.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include "Canvas/Search/RandomSample.hpp"
#include "Canvas/Primitives/Factory.hpp"
#include "Canvas/Utils/Exceptions.hpp"

// #define CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
// #define CANVAS_DEBUG_FAILED_COUNT
// #define CANVAS_DEBUG_PRINT_STATISTICS


namespace canvas {

namespace ba = boost::adaptors;

Solution TryRandomSample(const NetSpecsSP& net_specs, const SampleOptions& options) {
    // Random graph settings.
    int n_primitives = options.np_range.Random();
    int max_width = options.mw_range.Random();
    int expected_fc_count = std::min(options.fc_range.Random(), n_primitives);
    assert(expected_fc_count > 0 and expected_fc_count <= n_primitives);
    double fc_sample_possibility = static_cast<double>(expected_fc_count) / n_primitives;
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
    IC();
    IC(n_primitives, max_width, expected_fc_count);
#endif
    if (expected_fc_count > options.max_fc_ratio * n_primitives) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int too_many_fc_primitives = 0;
        IC(too_many_fc_primitives ++);
#endif
        return {};
    }

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

        // Random a primitive according to the filters.
        PrimitiveOptions primitive_options(options.allowed_filter, options.forbidden_filter,
                                           options.kernel_sizes, options.dilated_sizes, options.shift_sizes,
                                           options.add_relu_bn_after_fc);

        // Hash filters.
        for (const auto& p: graph->primitives)
            primitive_options.hash_filter.insert(p->Hash(true));

        // Width filters.
        if (must_reduce_width)
            primitive_options.max_delta_width = -1;
        else if (could_not_expand_width)
            primitive_options.max_delta_width = 0;

        // Grouping number filters.
        if (net_specs->c_gcd == 1)
            primitive_options.forbidden_filter.emplace_back(GroupTypeToName(GroupByFactor));

        // Must be output next step.
        if (n_steps_remaining == 1)
            primitive_options.output_filter = true;

        // FC selectivity filter.
        if (MakeChoice(fc_sample_possibility))
            primitive_options.allowed_filter = {"fc"};
        else
            primitive_options.forbidden_filter.emplace_back("fc");

        // Get all available applies.
        auto applies = PrimitiveFactory::GetPrimitiveApplies(graph, primitive_options);

        // Rescale possibilities.
        applies = PrimitiveFactory::RescalePossibilities(applies);

        if (applies.empty()) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int no_available_applies = 0;
            IC(no_available_applies ++);
#endif
            return {};
        }

        // Random choose and apply.
        auto apply = RandomChoose(applies);
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
        IC(width, n_steps_remaining, must_reduce_width, could_not_expand_width, apply);
#endif
        try {
            graph->Apply(apply);
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

    // FC constraints.
    if (not options.fc_range.Contains(graph->PrimitiveCount<FCPrimitive>())) {
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

    // Filter by user options.
    if (options.Filter(graph))
        return {};

    // For debug (filter primitives).
    // if (graph->PrimitiveCount<ReorderPrimitive>() == 0)
    //     return {};

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

    try {
        auto current_vars = graph->DynamicVars();
        // TODO: better strategies may exist, e.g. with a reduction factor.
        for (int index: current_vars)
            graph->SolveDynamicVar(VarSolution(index, Variable::StaticVar(StaticVarPos::VC)));
    } catch (const CanNotSolveDynamicVar& ex) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_solve_remaining_dynamic_var = 0;
        IC(can_not_solve_remaining_dynamic_var ++);
#endif
        return {};
    }

    // Sample a grouping factor.
    std::vector<int> available_g_factors;
    for (int i = 0; i <= SampleOptions::kMaxGroupFactor; ++ i)
        if (net_specs->c_gcd % (1 << i) == 0)
            available_g_factors.push_back(i);
    auto global_specs = GlobalSpecs(1 << RandomChoose(available_g_factors));

    // The final check, fill the solution with concise values.
    for (const auto& kernel: net_specs->kernel_specs)
        if (not graph->AlgebraCheck(Merge(global_specs, kernel)))
            return {};
    return {net_specs, graph, global_specs};
}

Solution RandomSample(const NetSpecsSP& net_specs, const SampleOptions& options) {
    auto start_time_point = std::chrono::system_clock::now();
    int times = 0;
    while (true) {
        ++ times;
        auto solution = TryRandomSample(net_specs, options);
        if (not solution.Empty())
            return solution;

        auto current_time_point = std::chrono::system_clock::now();
        if (options.timeout != std::chrono::seconds(0) and current_time_point - start_time_point > options.timeout)
            throw TimeoutException(options.timeout);
    }
    Unreachable();
}

} // namespace canvas
