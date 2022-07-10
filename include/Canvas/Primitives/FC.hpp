#pragma once

#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct FCPrimitive: Primitive {
    FCPrimitive(const TensorSP& t, const Variable& nc);

    [[nodiscard]] std::vector<Variable> IntermediateVariables() const final;

    CanvasPrimitiveCopyTemplate(FCPrimitive);
};

} // namespace canvas
