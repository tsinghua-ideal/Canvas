#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum FoldType {
    FoldH,
    FoldW,
    FoldHW
};

enum FoldArithmeticType {
    FoldAvg,
    FoldMax
};

static constexpr const char* FoldTypeToName(FoldType type) {
    switch (type) {
        case FoldH: return "FoldH";
        case FoldW: return "FoldW";
        case FoldHW: return "FoldHW";
    }
    return "";
}

static constexpr const char* FoldArithmeticTypeToName(FoldArithmeticType type) {
    switch (type) {
        case FoldAvg: return "Sum";
        case FoldMax: return "Max";
    }
    return "";
}

struct FoldPrimitive: Primitive {
    FoldType type;
    FoldArithmeticType arith_type;

    explicit FoldPrimitive(const TensorSP& t, FoldType type=FoldHW, FoldArithmeticType arith_type=FoldAvg);

    [[nodiscard]] size_t FLOPsCount(const Variable::StaticSpecs& specs,
                                    const Variable::DynamicFills& fills) const override;

    CanvasPrimitiveCopyTemplate(FoldPrimitive);
};

} // End namespace canvas
