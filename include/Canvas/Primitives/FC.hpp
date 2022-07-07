#pragma once

#include <tuple>

#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct FCPrimitive: Primitive {
    bool with_norm = false, with_relu = false;

    FCPrimitive(const TensorSP& t, const Variable& nc);

    [[nodiscard]] std::vector<Variable> IntermediateVariables() const final;

    CanvasPrimitiveCopyTemplate(FCPrimitive);
};

} // namespace canvas
