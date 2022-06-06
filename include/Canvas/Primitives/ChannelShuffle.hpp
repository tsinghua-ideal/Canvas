#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct ChannelShufflePrimitive: Primitive {
    explicit ChannelShufflePrimitive(const TensorSP& t);

    [[nodiscard]] std::vector<Variable> IntermediateVariables() const override;

    CanvasPrimitiveCopyTemplate(ChannelShufflePrimitive);
};

} // namespace canvas
