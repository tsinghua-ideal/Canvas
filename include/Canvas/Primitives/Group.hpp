#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

static std::string GroupTypeToName(int d, const Variable& factor) {
    std::stringstream ss;
    ss << "Group_" << d << "_" << factor;
    return ss.str();
}

struct GroupPrimitive: Primitive {
    explicit GroupPrimitive(const TensorSP& t, int d, const Variable& factor);

    CanvasPrimitiveCopyTemplate(GroupPrimitive);
};

} // namespace canvas
