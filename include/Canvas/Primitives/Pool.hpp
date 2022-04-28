#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum PoolType {
    PoolH,
    PoolW,
    PoolHW
};

static constexpr const char* PoolTypeToName(PoolType type) {
    switch (type) {
        case PoolH: return "PoolH";
        case PoolW: return "PoolW";
        case PoolHW: return "PoolHW";
    }
    return "";
}

struct PoolPrimitive: Primitive {
    PoolType type;

    explicit PoolPrimitive(const TensorSP& t, PoolType type=PoolHW);

    [[nodiscard]] size_t FLOPsCount(const Variable::StaticSpecs& specs,
                                    const Variable::DynamicFills& fills) const override;

    CanvasPrimitiveCopyTemplate(PoolPrimitive);
};

} // End namespace canvas
