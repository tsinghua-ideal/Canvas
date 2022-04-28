#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum SoftmaxType {
    SoftmaxC,
    SoftmaxH,
    SoftmaxW,
    SoftmaxHW
};

static constexpr const char* SoftmaxTypeToName(SoftmaxType type) {
    switch (type) {
        case SoftmaxC: return "SoftmaxC";
        case SoftmaxH: return "SoftmaxH";
        case SoftmaxW: return "SoftmaxW";
        case SoftmaxHW: return "SoftmaxHW";
    }
    return "";
}

struct SoftmaxPrimitive: Primitive {
    SoftmaxType type;

    explicit SoftmaxPrimitive(const TensorSP& t, SoftmaxType type=SoftmaxHW);

    [[nodiscard]] size_t FLOPsCount(const Variable::StaticSpecs& specs,
                                    const Variable::DynamicFills& fills) const override;

    CanvasPrimitiveCopyTemplate(SoftmaxPrimitive);
};

} // End namespace canvas
