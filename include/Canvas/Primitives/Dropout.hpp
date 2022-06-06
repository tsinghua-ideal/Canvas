#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct DropoutPrimitive: Primitive {
    explicit DropoutPrimitive(const TensorSP& t);

    CanvasPrimitiveCopyTemplate(DropoutPrimitive);
};

} // namespace canvas
