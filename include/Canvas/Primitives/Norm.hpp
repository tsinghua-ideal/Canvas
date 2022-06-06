#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct NormPrimitive: Primitive {
    explicit NormPrimitive(const TensorSP& t);

    CanvasPrimitiveCopyTemplate(NormPrimitive);
};

} // namespace canvas
