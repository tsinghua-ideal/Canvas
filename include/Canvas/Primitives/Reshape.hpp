#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct ReshapePrimitive: Primitive {
    ReshapePrimitive(const TensorSP& t, const Shape& new_shape);

    CanvasPrimitiveCopyTemplate(ReshapePrimitive);
};

} // End namespace canvas
