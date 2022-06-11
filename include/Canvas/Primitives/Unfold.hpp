#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum UnfoldType {
    UnfoldH,
    UnfoldW,
    UnfoldHW
};

static std::string UnfoldTypeToName(UnfoldType type, int k, int d) {
    std::string name;
    switch (type) {
        case UnfoldH: name = "UnfoldH"; break;
        case UnfoldW: name = "UnfoldW"; break;
        case UnfoldHW: name = "UnfoldHW"; break;
        default: Unreachable();
    }
    return name + "_K" + std::to_string(k) + "_D" + std::to_string(d);
}

struct UnfoldPrimitive: Primitive {
    UnfoldType type;
    int k = 3, d = 1;

    explicit UnfoldPrimitive(const TensorSP& t, int k=3, int d=1, UnfoldType type=UnfoldHW);

    CanvasPrimitiveCopyTemplate(UnfoldPrimitive);
};

} // namespace canvas
