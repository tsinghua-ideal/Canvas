#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum ShiftType {
    ShiftH,
    ShiftW,
    ShiftHW
};

static constexpr const char* ShiftTypeToName(ShiftType type) {
    switch (type) {
        case ShiftH: return "ShiftH";
        case ShiftW: return "ShiftW";
        case ShiftHW: return "ShiftHW";
    }
    return "";
}

struct ShiftPrimitive: Primitive {
    ShiftType type;

    explicit ShiftPrimitive(const TensorSP& t, ShiftType type);

    CanvasPrimitiveCopyTemplate(ShiftPrimitive);
};

} // namespace canvas
