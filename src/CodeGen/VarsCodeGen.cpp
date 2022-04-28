#include <sstream>

#include <json.hpp>

#include "Canvas/CodeGen/VarsCodeGen.hpp"


namespace canvas {

Code VarsCodeGen::GenImpl(const Solution& solution, std::string name) {
    auto net_specs = solution.specs;
    auto graph = solution.graph;
    auto preferences = solution.preferences;
    auto net_fills = solution.fills;

    nlohmann::json j;
    j["g"] = preferences.g;
    j["r"] = preferences.r;
    if (net_fills and graph->DynamicVarCount() != 0) {
        j["x"] = nlohmann::json::array();
        int n_kernels = (int) net_fills->Size();
        for (int i = 0; i < n_kernels; ++ i)
            j["x"].push_back(net_fills->At(i).x);
    }

    Write() << j << std::endl;
    return Dump();
}

} // End namespace canvas
