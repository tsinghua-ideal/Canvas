#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct InputPrimitive: Primitive {
    InputPrimitive(): Primitive("Input", {}) {
        outs.push_back(std::make_shared<Tensor>(Shape::StandardCHW()));
    }

    CanvasPrimitiveCopyTemplate(InputPrimitive);
};

} // namespace canvas
