#include <ice-cream.hpp>
#include <thread>

#include "Canvas/Search/RandomSample.hpp"
#include "Canvas/Search/ReceptiveAnalyzer.hpp"
#include "Canvas/Primitives/Factory.hpp"
#include "Canvas/Utils/Exceptions.hpp"

// #define CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
// #define CANVAS_DEBUG_FAILED_COUNT
// #define CANVAS_DEBUG_PRINT_STATISTICS
// #define CANVAS_DEBUG_STATIC_KERNEL


namespace canvas {

Solution TryRandomSample(const NetSpecsSP& net_specs, const SampleOptions& options) {
    // Random graph settings.
    int n_primitives = options.np_range.Random();
    int max_width = options.mw_range.Random();
    int expected_weighted_count = std::min(options.weighted_range.Random(), n_primitives);
    assert(expected_weighted_count > 0 and expected_weighted_count <= n_primitives);
    double weighted_sample_possibility = static_cast<double>(expected_weighted_count) / n_primitives;
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
    IC();
    IC(n_primitives, max_width, expected_weighted_count);
#endif
    if (expected_weighted_count > options.max_weighted_ratio * n_primitives) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int too_many_weighted_primitives = 0;
        IC(too_many_weighted_primitives ++);
#endif
        return {};
    }

    // Sample a grouping factor.
    std::vector<int> available_g_factors;
    for (int i = 0; i <= SampleOptions::kMaxGroupFactor; ++ i)
        if (net_specs->c_gcd % (1 << i) == 0)
            available_g_factors.push_back(i);
    auto global_specs = GlobalSpecs(1 << RandomChoose(available_g_factors));

    // Determine spatial dims and input/output shapes
    size_t spatial_dims = 2;
    if (not net_specs->Empty()) {
        spatial_dims = net_specs->kernel_specs.front().spatial_dims;
        for (const auto& specs: net_specs->kernel_specs)
            assert(spatial_dims == specs.spatial_dims);
    }
    auto io_shape = Shape::MakeShapeCHW(spatial_dims);

    // Take actions.
    GraphSP graph = std::make_shared<Graph>(io_shape);
    std::unordered_set<TensorSP> algebra_checked;
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
                                           io_shape, options.ensure_spatial_invariance);

        // Hash filters.
        for (const auto& p: graph->primitives)
            primitive_options.hash_filter.insert(p->Hash(true));

        // Width filters.
        if (must_reduce_width)
            primitive_options.max_delta_width = -1;
        else if (could_not_expand_width)
            primitive_options.max_delta_width = 0;

        // Must be output next step.
        if (n_steps_remaining == 1)
            primitive_options.output_filter = true;

        // FC selectivity filter.
        if (MakeChoice(weighted_sample_possibility)) {
            primitive_options.allowed_filter = {"fc", "conv", "scale", "mix"};
        } else {
            primitive_options.forbidden_filter.emplace_back("fc");
            primitive_options.forbidden_filter.emplace_back("conv");
            primitive_options.forbidden_filter.emplace_back("scale");
            primitive_options.forbidden_filter.emplace_back("mix");
        }

        // Get all available applies.
        auto applies = PrimitiveFactory::GetPrimitiveApplies(graph, primitive_options);

        if (applies.empty()) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int no_available_applies = 0;
            IC(i, n_primitives, primitive_options.max_delta_width, no_available_applies ++);
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
        } catch (const CanNotSolveDynamicVarOnGraph& ex) {
#ifdef CANVAS_DEBUG_PRINT_RANDOM_SAMPLE_STEPS
            IC(ex);
#endif
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_solve_dyn_var = 0;
            IC(can_not_solve_dyn_var ++);
#endif
            return {};
        } // Failed if other exceptions.

        // Early algebra checks.
        for (const auto& kernel: net_specs->kernel_specs) {
            auto specs = Merge(global_specs, kernel);
            for (const auto& t: graph->tensors) {
                if (not t->shape.IsStatic())
                    continue;
                if (algebra_checked.count(t))
                    continue;
                algebra_checked.insert(t);
                if (not (t->shape.FillToStaticShape(specs)).IsValid()) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
                    static int can_not_pass_algebra_check = 0;
                    IC(can_not_pass_algebra_check ++);
#endif
                    return {};
                }
            }
        }
    }

    // Weighted constraints.
    auto weighted_count = graph->PrimitiveCount<FCPrimitive>();
    weighted_count += graph->PrimitiveCount<ScalePrimitive>();
    weighted_count += graph->PrimitiveCount<ConvolutionPrimitive>();
    if (not options.weighted_range.Contains(weighted_count)) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_satisfy_weighted_count = 0;
        IC(can_not_satisfy_weighted_count ++);
#endif
        return {};
    }

    // Filter by user options.
    if (options.Filter(graph)) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_pass_option_filter = 0;
        IC(can_pass_option_filter ++);
#endif
        return {};
    }

#ifdef CANVAS_DEBUG_PRINT_STATISTICS
    static int n_total_sampled = 0, n_out_with_variable = 0;
    n_total_sampled ++;
#endif
    auto out = graph->Out();
    if (not out->shape.IsStatic()) {
        auto pi = out->shape.Pi();
        auto var_sol = VarSolution::Solve(pi, io_shape.Pi());
        if (pi.DynamicVarCount() > 1 or not var_sol.has_value()) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_solve_output_shape = 0;
            IC(can_not_solve_output_shape ++);
#endif
            return {};
        }
        try {
            graph->SolveDynamicVar(var_sol.value());
        } catch (const CanNotSolveDynamicVarOnGraph& ex) {
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
    graph->ApplyOutput(io_shape);

    try {
        std::vector<int> available_reduction_factors;
        for (int i = 0; i <= SampleOptions::kMaxReductionFactor; ++ i)
            if (net_specs->c_gcd % (1 << i) == 0)
                available_reduction_factors.push_back(1 << i);
        assert(not available_reduction_factors.empty());
        int reduction_factor = RandomChoose(available_reduction_factors);
        auto current_vars = graph->DynamicVars();
        for (int index: current_vars) {
            auto related_vars = graph->GetRelatedVariables(index);
            bool has_numerator_gcd = false;
            Variable numerator_gcd, denominator_lcm;
            for (const auto& related: related_vars) {
                assert(related.dynamic_power[index] != 0);
                if (std::abs(related.dynamic_power[index]) != 1) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
                    static int can_not_solve_remaining_dynamic_var_not_linear = 0;
                    IC(can_not_solve_remaining_dynamic_var_not_linear ++);
#endif
                    return {};
                }
                if (related.dynamic_power[index] == 1) {
                    denominator_lcm = Variable::Lcm(denominator_lcm, related.Denominator());
                } else {
                    assert(related.dynamic_power[index] == -1);
                    if (not has_numerator_gcd)
                        has_numerator_gcd = true, numerator_gcd = related.Numerator();
                    else
                        numerator_gcd = Variable::Gcd(numerator_gcd, related.Numerator());
                }
            }
            Variable repl = Variable::StaticVar(StaticVarPos::VC) / Variable::Number(reduction_factor);
            if (has_numerator_gcd)
                repl = numerator_gcd;
            else if (not denominator_lcm.Empty())
                repl = denominator_lcm;
            graph->SolveDynamicVar(VarSolution(index, repl));
        }
    } catch (const CanNotSolveDynamicVarOnGraph& ex) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_solve_remaining_dynamic_var = 0;
        IC(can_not_solve_remaining_dynamic_var ++);
#endif
        return {};
    }

    // The final check, fill the solution with concise values.
    if (not graph->AlgebraCheck(options.ensure_spatial_invariance))
        return {};
    for (const auto& kernel: net_specs->kernel_specs) {
        if (not graph->AlgebraCheck(Merge(global_specs, kernel))) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
            static int can_not_pass_algebra_check = 0;
            IC(can_not_pass_algebra_check ++);
#endif
            return {};
        }
    }

    // Filter for receptive field size.
    int receptive_size = ReceptiveAnalyzer::GetReceptiveSize(graph);
    if (receptive_size < options.min_receptive_size) {
#ifdef CANVAS_DEBUG_FAILED_COUNT
        static int can_not_satisfy_receptive_size = 0;
        IC(can_not_satisfy_receptive_size ++);
#endif
        return {};
    }

    // Successfully sampled!
    return {net_specs, graph, global_specs, options.ensure_spatial_invariance};
}

Solution RandomSample(const NetSpecsSP& net_specs, const SampleOptions& options) {
    bool force_bmm = MakeChoice(options.force_bmm_possibility);
    auto start_time_point = std::chrono::system_clock::now();

    Solution final_solution;
    std::mutex lock;
    auto task = [&]() {
        while (true) {
            auto solution = TryRandomSample(net_specs, options);
            if (not solution.Empty()) {
                if ((not force_bmm) or solution.graph->PrimitiveCount<MatrixMultiplicationPrimitive>() > 0) {
                    // An available solution.
                    lock.lock();
                    if (final_solution.Empty())
                        final_solution = solution;
                    lock.unlock();
                    return;
                }
            }

            // Check current solution.
            if (not final_solution.Empty())
                return;

            // Check timeout.
            auto current_time_point = std::chrono::system_clock::now();
            if (options.timeout != std::chrono::seconds(0) and current_time_point - start_time_point > options.timeout)
                return;
        }
    };

    // Span workers.
    assert(options.workers > 0);
    std::vector<std::thread> workers;
    for (int i = 0; i < options.workers; ++ i)
        workers.emplace_back(task);
    for (auto& worker: workers)
        worker.join();

    // Check timeout.
    if (final_solution.Empty())
        throw TimeoutException(options.timeout);

    return final_solution;
}

} // namespace canvas
