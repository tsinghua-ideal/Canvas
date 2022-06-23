#include "Canvas/Primitives/Unfold.hpp"


namespace canvas {

UnfoldPrimitive::UnfoldPrimitive(const TensorSP& t, int k, int d, UnfoldType type):
        k(k), d(d), type(type), Primitive(UnfoldTypeToName(type, k, d), {t}) {
    assert(t->shape.IsChannelSpatial());
    assert(k > 1 and k % 2 == 1 and d >= 1);
    auto new_shape = t->shape;
    auto channel = new_shape.Channel();
    auto spatial = new_shape.Spatial();
    if (type == UnfoldH or type == UnfoldHW) {
        assert(channel->KH().Empty() and not spatial->H().Empty());
        channel->KH() = Variable::Number(k);
    }
    if (type == UnfoldW or type == UnfoldHW) {
        assert(channel->KW().Empty() and not spatial->W().Empty());
        channel->KW() = Variable::Number(k);
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
