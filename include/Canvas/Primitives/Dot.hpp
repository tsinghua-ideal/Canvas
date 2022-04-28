#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

struct DotPrimitive: Primitive {
    explicit DotPrimitive(const TensorSP& t);

    [[nodiscard]] size_t PsCount(const Variable::StaticSpecs& specs,
                                 const Variable::DynamicFills& fills=Variable::DynamicFills()) const override;

    [[nodiscard]] size_t FLOPsCount(const Variable::StaticSpecs& specs,
                                    const Variable::DynamicFills& fills=Variable::DynamicFills()) const override;

    CanvasPrimitiveCopyTemplate(DotPrimitive);
};

} // End namespace canvas
