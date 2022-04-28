#include "Canvas/Algorithms/Count.hpp"
#include "Canvas/Algorithms/Fill.hpp"

// #define CANVAS_DEBUG_EVALUATOR_DOUBLE_ALGORITHM_PRINT_ROUND


namespace canvas {

NetFillsSP GetMinimumFills(const NetSpecsSP& net_specs,
                           const GraphSP& graph,
                           const HeuristicPreferences& preferences) {
    // Get minimum fills individually for every kernel
    assert(net_specs);
    auto net_fills = std::make_shared<NetFills>();
    for (const auto& kernel_specs: net_specs->kernel_specs)
        net_fills->Push(graph->GetMinimumFills(MergeIntoStaticSpecs(preferences, kernel_specs)));

    // Scale the parameters by kernel channels
    int min_i = 0;
    for (int i = 1; i < net_specs->kernel_specs.size(); ++ i)
        if (net_specs->kernel_specs[i].ChannelGeometricMean() < net_specs->kernel_specs[min_i].ChannelGeometricMean())
            min_i = i;
    assert(net_fills->Size() == net_specs->kernel_specs.size());
    for (int i = 0; i < net_specs->kernel_specs.size(); ++ i) {
        size_t c1 = net_specs->kernel_specs[min_i].ChannelGeometricMean();
        size_t c2 = net_specs->kernel_specs[i].ChannelGeometricMean();
        for (int j = 0; j < Variable::kDynamicVarCount; ++ j) {
            size_t q1 = net_fills->At(min_i).x[j], q2 = net_fills->At(i).x[j];
            // Scale q2 by ceil(c2 * q1 / (c1 * q2))
            size_t a = c2 * q1, b = c1 * q2, s;
            assert(b > 0);
            if (a % b == 0) // Use integer division to avoid precision problems
                s = a / b;
            else
                s = (size_t) std::ceil((double) (a) / (double) (b));
            assert(s > 0);
            net_fills->At(i).x[j] *= s;
        }
    }

    // Check minimals
    size_t flops = NetFLOPsCount(net_specs, graph, preferences, net_fills);
    size_t ps = NetPsCount(net_specs, graph, preferences, net_fills);
    if (not net_specs->BelowFlopsPsBudget(flops, ps))
        throw CanNotFillDynamicVariables("the minimum fills have already reach the budget");
    if (not NetCheckTensorSizeOverflow(net_specs, graph, preferences, net_fills))
        throw CanNotFillDynamicVariables("the minimum fills lead to a tensor size overflow");

    return net_fills;
}

void UpdateFullFillsUnderBudget(const NetSpecsSP& net_specs,
                                const GraphSP& graph,
                                const HeuristicPreferences& preferences,
                                const NetFillsSP& fills) {
    if (graph->DynamicVarCount() == 0)
        return;

    int n_kernels = static_cast<int>(fills->Size());
    assert(n_kernels > 0);
    std::vector<Variable::StaticSpecs> static_specs;
    static_specs.reserve(n_kernels);
    for (int i = 0; i < n_kernels; ++ i)
        static_specs.push_back(MergeIntoStaticSpecs(preferences, net_specs->kernel_specs[i]));

    size_t total_flops = 0, total_ps = 0;
    std::vector<size_t> flops(n_kernels), ps(n_kernels);
    for (int i = 0; i < n_kernels; ++ i) {
        flops[i] = KernelFLOPsCount(graph, static_specs[i], fills->At(i));
        ps[i] = KernelPsCount(graph, static_specs[i], fills->At(i));
        total_flops += flops[i], total_ps += ps[i];
    }
    assert(net_specs->BelowFlopsPsBudget(total_flops, total_ps));

#ifdef CANVAS_DEBUG_EVALUATOR_DOUBLE_ALGORITHM_PRINT_ROUND
    int round = 0;
    std::cout << "Round 0 (with minimum fills):" << std::endl;
    std::cout << " > FLOPs: " << total_flops << std::endl;
    std::cout << " > Ps: " << total_ps << std::endl;
#endif

    while (true) {
#ifdef CANVAS_DEBUG_EVALUATOR_DOUBLE_ALGORITHM_PRINT_ROUND
        std::cout << "Round " << ++ round << ":" << std::endl;
#endif

        // Get all deltas of FLOPs and Ps
        /// Index, delta of FLOPs, delta of Ps
        typedef std::tuple<int, size_t, size_t> KernelDelta;
        std::vector<KernelDelta> deltas;
        for (int i = 0; i < n_kernels; ++ i) {
            auto copied_fills = fills->At(i);
            copied_fills.Double();
            auto new_flops = KernelFLOPsCount(graph, static_specs[i], copied_fills);
            auto new_ps = KernelPsCount(graph, static_specs[i], copied_fills);
            assert(new_flops >= flops[i] and new_ps >= ps[i]);
            assert(new_flops > flops[i] or new_ps > ps[i]);
            if (KernelCheckTensorSizeOverflow(graph, static_specs[i], copied_fills))
                deltas.emplace_back(i, new_flops - flops[i], new_ps - ps[i]);
        }

        // Sort by delta of Ps (heuristically, maybe FLOPs also work)
        if (deltas.empty())
            break;
        std::sort(deltas.begin(), deltas.end(), [](const KernelDelta &lhs, const KernelDelta& rhs) -> bool {
            return std::get<2>(lhs) < std::get<2>(rhs);
        });

        // Double by the sorted order
        int doubled_count = 0;
        for (const auto& [i, flops_delta, ps_delta]: deltas) {
            if (net_specs->BelowFlopsPsBudget(total_flops + flops_delta, total_ps + ps_delta)) {
                ++ doubled_count;
#ifdef CANVAS_DEBUG_EVALUATOR_DOUBLE_ALGORITHM_PRINT_ROUND
                std::cout << " > Kernel#" << i << " doubled from " << fills->kernel_fills[i] << " to ";
#endif
                fills->At(i).Double();
#ifdef CANVAS_DEBUG_EVALUATOR_DOUBLE_ALGORITHM_PRINT_ROUND
                std::cout << fills->kernel_fills[i] << std::endl;
#endif
                flops[i] += flops_delta, ps[i] += ps_delta;
                total_flops += flops_delta, total_ps += ps_delta;
            }
        }
#ifdef CANVAS_DEBUG_EVALUATOR_DOUBLE_ALGORITHM_PRINT_ROUND
        std::cout << " > " << doubled_count << " kernel(s) doubled" << std::endl;
        std::cout << " > FLOPs: "<< total_flops << std::endl;
        std::cout << " > Ps: " << total_ps << std::endl;
#endif
        if (doubled_count != deltas.size())
            break;
    }
}

NetFillsSP GetFullFillsUnderBudget(const NetSpecsSP& net_specs,
                                   const GraphSP& graph,
                                   const HeuristicPreferences& preferences) {
    auto net_fills = GetMinimumFills(net_specs, graph, preferences);
    UpdateFullFillsUnderBudget(net_specs, graph, preferences, net_fills);
    return net_fills;
}

} // End namespace canvas

#ifdef CANVAS_DEBUG_EVALUATOR_DOUBLE_ALGORITHM_PRINT_ROUND
#undef CANVAS_DEBUG_EVALUATOR_DOUBLE_ALGORITHM_PRINT_ROUND
#endif
