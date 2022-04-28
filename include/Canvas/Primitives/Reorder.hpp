#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum ReorderType {
    ReorderH,
    ReorderW,
    ReorderHW
};

static std::string ReorderTypeToName(ReorderType type, bool inverse) {
    std::string type_string;
    switch (type) {
        case ReorderH: type_string = "ReorderH"; break;
        case ReorderW: type_string = "ReorderW"; break;
        case ReorderHW: type_string = "ReorderHW"; break;
    }
    return (inverse ? "Inverse" : "") + type_string;
}

struct ReorderPrimitive: Primitive {
    ReorderType type;
    bool inverse;

    explicit ReorderPrimitive(const TensorSP& t, ReorderType type, bool inverse);

    CanvasPrimitiveCopyTemplate(ReorderPrimitive);
};

} // End namespace canvas
