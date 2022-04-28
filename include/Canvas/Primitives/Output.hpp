#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

/// Output primitive for ending the graph
struct OutputPrimitive: Primitive {
    explicit OutputPrimitive(const TensorSP& t);

    CanvasPrimitiveCopyTemplate(OutputPrimitive);
};

} // End namespace canvas
