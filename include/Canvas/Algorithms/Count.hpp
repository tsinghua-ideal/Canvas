#pragma once

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Solution.hpp"


namespace canvas {

/// The predefined max batch size
static constexpr size_t kPredefinedMaxBatchSize = 512;

size_t KernelPsCount(const GraphSP& graph,
                     const Variable::StaticSpecs& specs,
                     const Variable::DynamicFills& fills=Variable::DynamicFills());

size_t KernelFLOPsCount(const GraphSP& graph,
                        const Variable::StaticSpecs& specs,
                        const Variable::DynamicFills& fills=Variable::DynamicFills());

bool KernelCheckTensorSizeOverflow(const GraphSP& graph,
                                   const Variable::StaticSpecs& specs,
                                   const Variable::DynamicFills& fills=Variable::DynamicFills());

size_t NetPsCount(const NetSpecsSP& specs,
                  const GraphSP& graph,
                  const HeuristicPreferences& preferences=HeuristicPreferences(),
                  const NetFillsSP& fills=nullptr);

size_t NetFLOPsCount(const NetSpecsSP& specs,
                     const GraphSP& graph,
                     const HeuristicPreferences& preferences=HeuristicPreferences(),
                     const NetFillsSP& fills=nullptr);

bool NetCheckTensorSizeOverflow(const NetSpecsSP& specs,
                                const GraphSP& graph,
                                const HeuristicPreferences& preferences=HeuristicPreferences(),
                                const NetFillsSP& fills=nullptr);

static size_t NetPsCount(const Solution& solution) {
    return NetPsCount(solution.specs, solution.graph, solution.preferences, solution.fills);
}

static size_t NetFLOPsCount(const Solution& solution) {
    return NetFLOPsCount(solution.specs, solution.graph, solution.preferences, solution.fills);
}

} // End namespace canvas
