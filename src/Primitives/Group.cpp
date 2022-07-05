#include "Canvas/Primitives/Group.hpp"


namespace canvas {

GroupPrimitive::GroupPrimitive(const TensorSP& t, int d, const Variable& factor):
        Primitive(GroupTypeToName(d, factor), {t}, false) {
    assert(not factor.Empty());
    assert(0 <= d and d < 2 and DynamicCast<ChannelShape>(t->shape.dims[d]));
    Shape new_shape = t->shape;
    auto channel = DynamicCast<ChannelShape>(new_shape.dims[d]);
    assert(channel->G().Empty());
    channel->C() /= factor;
    assert(channel->C().MaybeInteger());
    channel->G() = factor;
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
