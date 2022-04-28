#include <boost/range/adaptor/reversed.hpp>

#include "Canvas/CodeGen/CodeGen.hpp"


namespace canvas {

namespace ba = boost::adaptors;

void CodeGen::Travel(const GraphSP& graph, const std::function<void(CodeGen*, PrimitiveSP)>& func, bool reverse) {
    // Assume that `graph->primitives` is in topological orders
    if (not reverse) {
        for (const auto& p: graph->primitives)
            func(this, p);
    } else {
        for (const auto& p: ba::reverse(graph->primitives))
            func(this, p);
    }
}

void CodeGen::CommonChecks(const Solution& solution) {
    auto net_specs = solution.specs;
    auto graph = solution.graph;
    auto preferences = solution.preferences;
    auto net_fills = solution.fills;

    // Fills and specs need each other to check algebra legality
    assert((net_fills == nullptr) == (net_specs == nullptr));

    // Graph should be topologically finished
    if (not graph->IsTopologicalFinished())
        CriticalError("Unable to generate code for an unfinished graph (topologically)");

    // All the shapes satisfy the assumptions
    for (const auto& t: graph->tensors)
        if (not t->shape.SatisfyAssumption())
            CriticalError("Unable to generate code for a solution which has illegal variables");

    if (net_fills == nullptr) {
        for (const auto& t: graph->tensors)
            if (not t->shape.IsAllStatic())
                CriticalError("Unable to generate code for a solution with dynamic variables, but without fills");
    } else {
        assert(net_specs->layer_specs.size() == net_fills->Size());
        for (int i = 0; i < net_specs->layer_specs.size(); ++ i)
            if (not graph->AlgebraCheck(MergeIntoStaticSpecs(preferences, net_specs->layer_specs[i]),
                                        net_fills->At(i)))
                CriticalError("Unable to generate code with an illegal dynamic variable fills");
    }
}

} // End namespace canvas
