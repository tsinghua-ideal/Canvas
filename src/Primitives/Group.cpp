#include "Canvas/Primitives/Group.hpp"


namespace canvas {

GroupPrimitive::GroupPrimitive(const TensorSP& t, int d, GroupType type):
        d(d), type(type), Primitive(GroupTypeToName(d, type), {t}, false) {
    assert(0 <= d and d < 2 and DynamicCast<ChannelShape>(t->shape.dims[d]));
    Shape new_shape = t->shape;
    auto channel = DynamicCast<ChannelShape>(new_shape.dims[d]);
    if (type == GroupByFactor) {
        channel->C() /= StaticVarPos::VG;
        assert(channel->C().MaybeInteger());
        channel->G() = StaticVarPos::VG;
    } else if (type == GroupAllChannels) {
        assert(not channel->CKK().Empty());
        channel->G() = channel->CKK();
        channel->C().Reset(), channel->KH().Reset(), channel->KW().Reset();
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
