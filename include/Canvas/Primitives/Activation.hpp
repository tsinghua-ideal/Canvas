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

static constexpr int ActivationFLOPsFactor(ActivationType type) {
    switch (type) {
        case GeLU: return 4;
        case ReLU: return 1;
        case Sigmoid: return 3;
        case TanH: return 5;
    }
    return 0;
}

struct ActivationPrimitive: Primitive {
    ActivationType type;

    explicit ActivationPrimitive(const TensorSP& t, ActivationType type=ReLU);

    CanvasPrimitiveCopyTemplate(ActivationPrimitive);
};

} // namespace canvas
