#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct DotPrimitive: Primitive {
    explicit DotPrimitive(const TensorSP& t);

    CanvasPrimitiveCopyTemplate(DotPrimitive);
};

} // namespace canvas
