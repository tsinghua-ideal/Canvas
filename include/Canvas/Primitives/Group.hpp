#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

enum GroupType {
    GroupByFactor,
    GroupAllChannels,
};

static constexpr const char* GroupTypeToName(GroupType type) {
    switch (type) {
        case GroupByFactor: return "GroupByFactor";
        case GroupAllChannels: return "GroupAll";
    }
    return "";
}

struct GroupPrimitive: Primitive {
    GroupType type;

    explicit GroupPrimitive(const TensorSP& t, GroupType type=GroupByFactor);

    CanvasPrimitiveCopyTemplate(GroupPrimitive);
};

} // End namespace canvas
