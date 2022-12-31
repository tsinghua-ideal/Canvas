#pragma once

#include <utility>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Specs.hpp"


namespace canvas {

struct Solution {
    NetSpecsSP net_specs = nullptr;
    GraphSP graph = nullptr;
    GlobalSpecs global_specs;
    bool ensure_spatial_invariance = true;

    Solution() = default;

    explicit Solution(GraphSP graph,
                      GlobalSpecs global_specs=GlobalSpecs(),
                      bool check_spatial_dims=true):
            net_specs(std::make_shared<NetSpecs>()), graph(std::move(graph)),
            global_specs(global_specs), ensure_spatial_invariance(check_spatial_dims) {
        assert(this->graph);
    }

    Solution(NetSpecsSP specs, GraphSP graph,
             GlobalSpecs global_specs=GlobalSpecs(),
             bool check_spatial_dims=true):
            net_specs(std::move(specs)), graph(std::move(graph)),
            global_specs(global_specs), ensure_spatial_invariance(check_spatial_dims) {
        assert(this->net_specs and this->graph);
    }

    [[nodiscard]] bool Empty() const { return graph == nullptr; }

    [[nodiscard]] size_t Hash() const {
        size_t hash = net_specs ? net_specs->Hash() : 0;
        hash = IterateHash(hash, global_specs.Hash());
        hash = IterateHash(hash, graph->Hash());
        hash = IterateHash(hash, ensure_spatial_invariance);
        return hash;
    }
};

} // namespace canvas
