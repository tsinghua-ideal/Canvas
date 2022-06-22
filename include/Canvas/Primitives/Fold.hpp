#pragma once

#include "Canvas/Core/Shape.hpp"
#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum FoldType {
    FoldAvg,
    FoldMax
};

static std::string FoldTypeToName(const Shape& s, const std::vector<Shape::Index>& indices, FoldType type) {
    std::stringstream ss;
    ss << "Fold";
    for (const auto& index: indices)
        ss << "_" << index.d << "/" << index.k << "/" << s.IndexToName(index);
    switch (type) {
        case FoldAvg: return ss.str() + "_Avg";
        case FoldMax: return ss.str() + "_Max";
    }
    Unreachable();
}

struct FoldPrimitive: Primitive {
    std::vector<Shape::Index> indices;
    FoldType type;

    explicit FoldPrimitive(const TensorSP& t, const std::vector<Shape::Index>& indices, FoldType type=FoldAvg);

    CanvasPrimitiveCopyTemplate(FoldPrimitive);
};

} // namespace canvas
