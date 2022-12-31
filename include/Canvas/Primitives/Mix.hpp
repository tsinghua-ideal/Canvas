#pragma once

#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct MixPrimitive: Primitive {
    std::vector<Shape::Index> indices;

    MixPrimitive(const TensorSP& t,
                 const std::vector<Shape::Index>& indices,
                 const std::vector<Variable>& new_dims);

    [[nodiscard]] std::vector<Variable> ParamShape() const final;

    CanvasPrimitiveCopyTemplate(MixPrimitive);
};

} // namespace canvas
