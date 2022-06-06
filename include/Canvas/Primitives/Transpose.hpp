#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct TransposePrimitive: Primitive {
    explicit TransposePrimitive(const TensorSP& t);

    CanvasPrimitiveCopyTemplate(TransposePrimitive);
};

} // namespace canvas
