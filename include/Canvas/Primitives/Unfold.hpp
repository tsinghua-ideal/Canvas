#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum UnfoldType {
    UnfoldH,
    UnfoldW,
    UnfoldHW
};

// TODO: refactor this name.
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
    int k = 3, d = 1;

    explicit UnfoldPrimitive(const TensorSP& t, int k=3, int d=1, UnfoldType type=UnfoldHW);

    CanvasPrimitiveCopyTemplate(UnfoldPrimitive);
};

} // namespace canvas
