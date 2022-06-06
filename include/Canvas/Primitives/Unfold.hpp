#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum UnfoldType {
    UnfoldH,
    UnfoldW,
    UnfoldHW
};

static constexpr const char* UnfoldTypeToName(UnfoldType type) {
    switch (type) {
        case UnfoldH: return "UnfoldH";
        case UnfoldW: return "UnfoldW";
        case UnfoldHW: return "UnfoldHW";
    }
    return "";
}

struct UnfoldPrimitive: Primitive {
    UnfoldType type;

    explicit UnfoldPrimitive(const TensorSP& t, UnfoldType type=UnfoldHW);

    CanvasPrimitiveCopyTemplate(UnfoldPrimitive);
};

} // namespace canvas
