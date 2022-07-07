#pragma once

#include <sstream>
#include <vector>

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

static std::string SoftmaxToName(const Shape& s, const Shape::Index& index) {
    std::stringstream ss;
    ss << "Softmax" << "_" << index.d << "/" << index.k << "/" << s.IndexToName(index);
    return ss.str();
}

struct SoftmaxPrimitive: Primitive {
    Shape::Index index;

    explicit SoftmaxPrimitive(const TensorSP& t, const Shape::Index& index);

    CanvasPrimitiveCopyTemplate(SoftmaxPrimitive);
};

} // namespace canvas
