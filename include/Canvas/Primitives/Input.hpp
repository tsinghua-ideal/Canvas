#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct InputPrimitive: Primitive {
    explicit InputPrimitive(const Shape& input_shape=Shape::MakeShapeCHW()): Primitive("Input", {}) {
        outs.push_back(std::make_shared<Tensor>(input_shape));
    }

    CanvasPrimitiveCopyTemplate(InputPrimitive);
};

} // namespace canvas
