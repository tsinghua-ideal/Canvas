#pragma once

#include <utility>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Specs.hpp"


namespace canvas {

struct Solution {
    NetSpecsSP net_specs = nullptr;
    GraphSP graph = nullptr;
    GlobalSpecs global_specs;

    Solution() = default;

    Solution(NetSpecsSP specs, GraphSP graph, GlobalSpecs global_specs=GlobalSpecs()):
            net_specs(std::move(specs)), graph(std::move(graph)), global_specs(global_specs) {
        assert(this->net_specs and this->graph);
    }

    [[nodiscard]] bool Empty() const { return graph == nullptr; }

    [[nodiscard]] size_t Hash() const {
        size_t hash = net_specs ? net_specs->Hash() : 0;
        hash = IterateHash(hash, global_specs.Hash());
        hash = IterateHash(hash, graph->Hash());
        return hash;
    }
};

} // namespace canvas
