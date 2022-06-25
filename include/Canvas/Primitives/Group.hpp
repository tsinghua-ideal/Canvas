#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum GroupType {
    GroupByFactor,
    GroupAllChannels,
};

static std::string GroupTypeToName(int d, GroupType type) {
    switch (type) {
        case GroupByFactor: return "GroupByFactor_" + std::to_string(d);
        case GroupAllChannels: return "GroupAll_" + std::to_string(d);
    }
    Unreachable();
}

struct GroupPrimitive: Primitive {
    int d;
    GroupType type;

    explicit GroupPrimitive(const TensorSP& t, int d, GroupType type=GroupByFactor);

    CanvasPrimitiveCopyTemplate(GroupPrimitive);
};

} // namespace canvas
