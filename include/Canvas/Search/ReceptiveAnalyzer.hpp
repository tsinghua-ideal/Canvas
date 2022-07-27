#pragma once

#include <tuple>

#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Graph.hpp"


namespace canvas {

class ReceptiveAnalyzer {
private:
    static constexpr int kMaxReceptiveLength = 21;
    static constexpr int kCenterIndex = kMaxReceptiveLength / 2;

public:
    static int GetReceptiveSize(const GraphSP& graph);
};

} // namespace canvas
