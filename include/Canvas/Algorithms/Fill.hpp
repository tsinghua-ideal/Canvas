#pragma once

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/NetSpecs.hpp"


namespace canvas {

class CanNotFillDynamicVariables: public ExceptionWithInfo {
public:
    explicit CanNotFillDynamicVariables(const std::string& reason) {
        std::stringstream ss;
        ss << "Can not automatically fill dynamic variables: " << reason;
        info = ss.str();
    }
};

NetFillsSP GetMinimumFills(const NetSpecsSP& net_specs,
                           const GraphSP& graph,
                           const HeuristicPreferences& preferences=HeuristicPreferences());

NetFillsSP GetFullFillsUnderBudget(const NetSpecsSP& net_specs,
                                   const GraphSP& graph,
                                   const HeuristicPreferences& preferences=HeuristicPreferences());

}
