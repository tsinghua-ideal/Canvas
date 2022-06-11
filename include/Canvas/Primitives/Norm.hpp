#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

// TODO: may add different types of normalizations.
struct NormPrimitive: Primitive {
    explicit NormPrimitive(const TensorSP& t);

    CanvasPrimitiveCopyTemplate(NormPrimitive);
};

} // namespace canvas
