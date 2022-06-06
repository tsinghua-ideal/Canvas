#pragma once

#include <tuple>

#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct FCPrimitive: Primitive {
    bool with_norm = false, with_relu = false;

    explicit FCPrimitive(const TensorSP& t);

    FCPrimitive(const TensorSP& t, const Variable& nc);

    [[nodiscard]] std::vector<Variable> IntermediateVariables() const override;

    CanvasPrimitiveCopyTemplate(FCPrimitive);
};

} // namespace canvas
