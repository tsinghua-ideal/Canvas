#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct InputPrimitive: Primitive {
    InputPrimitive();

    CanvasPrimitiveCopyTemplate(InputPrimitive);
};

} // namespace canvas
