#pragma once

#include "Canvas/Core/Graph.hpp"


namespace canvas {

class Pass {
public:
    Pass() = default;

    [[nodiscard]] virtual GraphSP Optimize(const GraphSP& graph) = 0;
};

} // namespace canvas
