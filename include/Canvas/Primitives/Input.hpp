#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct InputPrimitive: Primitive {
    InputPrimitive();

    CanvasPrimitiveCopyTemplate(InputPrimitive);

    [[nodiscard]] size_t FLOPsCount(const Variable::StaticSpecs& specs,
                                    const Variable::DynamicFills& fills=Variable::DynamicFills()) const override;
};

} // End namespace canvas
