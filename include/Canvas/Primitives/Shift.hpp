#pragma once

#include <sstream>
#include <vector>

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

static std::string ShiftToName(const std::vector<Shape::DimPos>& pos_vec, int k) {
    std::stringstream ss;
    ss << "Shift";
    for (const auto& pos: pos_vec)
        ss << "_" << Shape::DimPosToName(pos);
    ss << "_K" << std::to_string(k);
    return ss.str();
}

struct ShiftPrimitive: Primitive {
    std::vector<Shape::DimPos> pos_vec;
    int k;

    explicit ShiftPrimitive(const TensorSP& t, const std::vector<Shape::DimPos>& pos_vec, int k=1);

    CanvasPrimitiveCopyTemplate(ShiftPrimitive);
};

} // namespace canvas
