#pragma once

#include <sstream>
#include <vector>

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

static std::string ShiftToName(const Shape& s, const std::vector<Shape::Index>& indices, int k) {
    std::stringstream ss;
    ss << "Shift";
    for (const auto& index: indices)
        ss << "_" << index.d << "/" << index.k << "/" << s.IndexToName(index);
    ss << "_K" << k;
    return ss.str();
}

struct ShiftPrimitive: Primitive {
    int k;
    std::vector<Shape::Index> indices;

    explicit ShiftPrimitive(const TensorSP& t, const std::vector<Shape::Index>& indices, int k=1);

    CanvasPrimitiveCopyTemplate(ShiftPrimitive);
};

} // namespace canvas
