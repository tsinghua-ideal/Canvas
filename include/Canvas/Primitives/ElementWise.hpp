#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum ElementWiseType {
    Abs,
    Exp,
    Neg,
    Sin
};

static constexpr const char* ElementWiseTypeToName(ElementWiseType type) {
    switch (type) {
        case Abs: return "Abs";
        case Exp: return "Exp";
        case Neg: return "Neg";
        case Sin: return "Sin";
    }
    return "";
}

struct ElementWisePrimitive: Primitive {
    ElementWiseType type;

    explicit ElementWisePrimitive(const TensorSP& t, ElementWiseType type);

    [[nodiscard]] size_t FLOPsCount(const Variable::StaticSpecs& specs,
                                    const Variable::DynamicFills& fills) const override;

    CanvasPrimitiveCopyTemplate(ElementWisePrimitive);
};

} // End namespace canvas
