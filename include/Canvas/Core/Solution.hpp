#pragma once

#include <utility>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/NetSpecs.hpp"


namespace canvas {

struct Solution {
    NetSpecsSP specs = nullptr;
    GraphSP graph = nullptr;

    Solution() = default;

    Solution(NetSpecsSP specs, GraphSP graph):
            specs(std::move(specs)), graph(std::move(graph)) {
        assert(this->specs and this->graph);
    }

    [[nodiscard]] bool Empty() const { return graph == nullptr; }

    [[nodiscard]] size_t Hash() const {
        size_t hash = specs ? specs->Hash() : 0;
        hash = IterateHash(hash, graph->Hash());
        return hash;
    }
};

} // namespace canvas
