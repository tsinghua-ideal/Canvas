#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum ShiftType {
    ShiftH,
    ShiftW,
    ShiftHW
};

static std::string ShiftTypeToName(ShiftType type, int k) {
    std::string name;
    switch (type) {
        case ShiftH: name = "ShiftH"; break;
        case ShiftW: name = "ShiftW"; break;
        case ShiftHW: name = "ShiftHW"; break;
        default: Unreachable();
    }
    return name + "_K" + std::to_string(k);
}

// TODO: rewrite shift primitive to support any dimension (like fold).
struct ShiftPrimitive: Primitive {
    ShiftType type;
    int k;

    explicit ShiftPrimitive(const TensorSP& t, ShiftType type, int k=1);

    CanvasPrimitiveCopyTemplate(ShiftPrimitive);
};

} // namespace canvas
