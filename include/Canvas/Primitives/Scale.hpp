#pragma once

#include <sstream>
#include <vector>

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

static std::string ScaleToName(const Shape& s, const std::vector<Shape::Index>& indices) {
    std::stringstream ss;
    ss << "Scale";
    for (const auto& index: indices)
        ss << "_" << index.d << "/" << index.k << "/" << s.IndexToName(index);
    return ss.str();
}

struct ScalePrimitive: Primitive {
    std::vector<Shape::Index> indices;

    explicit ScalePrimitive(const TensorSP& t, const std::vector<Shape::Index>& indices);

    CanvasPrimitiveCopyTemplate(ScalePrimitive);
};

} // namespace canvas
