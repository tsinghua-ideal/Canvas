#pragma once

#include "Canvas/Core/Shape.hpp"
#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum FoldType {
    FoldAvg,
    FoldMax
};

static std::string FoldTypeToName(const std::vector<Shape::DimPos>& pos_vec, FoldType type) {
    std::string prefix = "Fold";
    for (const auto& pos: pos_vec)
        prefix += "_" + Shape::DimPosToName(pos);
    switch (type) {
        case FoldAvg: return prefix + "_Avg";
        case FoldMax: return prefix + "_Max";
    }
    Unreachable();
}

struct FoldPrimitive: Primitive {
    std::vector<Shape::DimPos> pos_vec;
    FoldType type;

    explicit FoldPrimitive(const TensorSP& t, const std::vector<Shape::DimPos>& pos, FoldType type=FoldAvg);

    CanvasPrimitiveCopyTemplate(FoldPrimitive);
};

} // namespace canvas
