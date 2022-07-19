#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum ElementWiseType {
    Abs,
    Exp,
    Neg,
    Sin,
    Sqrt,
    Sqr
};

static constexpr const char* ElementWiseTypeToName(ElementWiseType type) {
    switch (type) {
        case Abs:  return "Abs";
        case Exp:  return "Exp";
        case Neg:  return "Neg";
        case Sin:  return "Sin";
        case Sqrt: return "Sqrt";
        case Sqr:  return "Sqr";
    }
    return "";
}

struct ElementWisePrimitive: Primitive {
    ElementWiseType type;

    explicit ElementWisePrimitive(const TensorSP& t, ElementWiseType type);

    CanvasPrimitiveCopyTemplate(ElementWisePrimitive);
};

} // namespace canvas
