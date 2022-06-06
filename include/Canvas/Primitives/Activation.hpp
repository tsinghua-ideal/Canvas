#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum ActivationType {
    GeLU,
    ReLU,
    Sigmoid,
    TanH
};

static constexpr const char* ActivationTypeToName(ActivationType type) {
    switch (type) {
        case GeLU: return "GeLU";
        case ReLU: return "ReLU";
        case Sigmoid: return "Sigmoid";
        case TanH: return "TanH";
    }
    return "";
}

struct ActivationPrimitive: Primitive {
    ActivationType type;

    explicit ActivationPrimitive(const TensorSP& t, ActivationType type=ReLU);

    CanvasPrimitiveCopyTemplate(ActivationPrimitive);
};

} // namespace canvas
