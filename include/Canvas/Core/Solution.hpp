#pragma once

#include <utility>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/NetSpecs.hpp"


namespace canvas {

struct Solution {
    NetSpecsSP specs = nullptr;
    GraphSP graph = nullptr;
    HeuristicPreferences preferences = HeuristicPreferences();
    NetFillsSP fills = nullptr;

    Solution() = default;

    Solution(GraphSP graph): graph(std::move(graph)) { assert(this->graph); } // NOLINT(google-explicit-constructor)

    Solution(NetSpecsSP specs, // Problem specifications
             GraphSP graph, // Graph-level
             const HeuristicPreferences& preferences, NetFillsSP fills): // Heuristic fills
            specs(std::move(specs)), graph(std::move(graph)), preferences(preferences), fills(std::move(fills)) {
        assert(this->specs and this->graph and this->fills);
    }

    [[nodiscard]] bool Empty() const { return graph == nullptr; }

    [[nodiscard]] size_t Hash() const {
        size_t hash = specs ? specs->Hash() : 0;
        hash = IterateHash(hash, graph->Hash());
        hash = IterateHash(hash, preferences.Hash());
        hash = IterateHash(hash, fills ? fills->Hash() : 0);
        return hash;
    }
};

} // End namespace canvas
