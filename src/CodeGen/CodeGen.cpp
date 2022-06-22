#include <boost/range/adaptor/reversed.hpp>

#include "Canvas/CodeGen/CodeGen.hpp"


namespace canvas {

namespace ba = boost::adaptors;

void CodeGen::Travel(const GraphSP& graph, const std::function<void(CodeGen*, PrimitiveSP)>& func, bool reverse) {
    // Assume that `graph->primitives` is in topological orders.
    if (not reverse) {
        for (const auto& p: graph->primitives)
            func(this, p);
    } else {
        for (const auto& p: ba::reverse(graph->primitives))
            func(this, p);
    }
}

void CodeGen::CommonChecks(const Solution& solution) {
    auto graph = solution.graph;
    auto net_specs = solution.net_specs;

    // Algebra checks.
    for (const auto& kernel: net_specs->kernel_specs)
        if (not graph->AlgebraCheck(Merge(solution.global_specs, kernel)))
            CriticalError("Unable to generate code for a graph with illegal shapes");

    // Graph should be topologically finished.
    if (not graph->IsTopologicalFinished())
        CriticalError("Unable to generate code for an unfinished graph (topologically)");

    // Check remaining dynamic variables.
    for (const auto& t: graph->tensors)
        if (not t->shape.IsStatic())
            CriticalError("Unable to generate code for a solution with dynamic variables");
}

} // namespace canvas
