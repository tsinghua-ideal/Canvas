#include "Canvas/Primitives/Convolution.hpp"


namespace canvas {

ConvolutionPrimitive::ConvolutionPrimitive(const TensorSP& t,
                                           const Variable& oc,
                                           int kh, int kw, int dh, int dw,
                                           bool depth_wise):
        Primitive(ConvolutionToName(kh, kw, dh, dw, depth_wise), {t}, false),
        kh(kh), kw(kw), dh(dh), dw(dw), depth_wise(depth_wise) {
    assert(t->shape.IsChannelSpatial());
    assert(kh >= 1 and kh % 2 == 1 and kw >= 1 and kw % 2 == 1);
    assert(dh >= 1 and dw >= 1);
    auto new_shape = t->shape;
    auto channel = new_shape.Channel();
    auto spatial = new_shape.Spatial();
    assert(channel->KH().Empty() and channel->KW().Empty());
    assert(not (spatial->H().Empty() and spatial->W().Empty()));
    if (not depth_wise) {
        assert((oc / channel->G()).MaybeInteger());
    } else {
        assert(channel->G().Empty());
        assert((oc / channel->C()).MaybeInteger());
    }
    channel->Reset();
    channel->C() = oc;
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

std::vector<Variable> ConvolutionPrimitive::IntermediateVariables() const {
    auto channel = ins[0]->shape.Channel();
    auto oc = outs[0]->shape.Channel()->Pi();
    return {depth_wise ? (oc / channel->C()) : (oc / channel->G())};
}

std::vector<Variable> ConvolutionPrimitive::ParamShape() const {
    auto g = ins[0]->shape.Channel()->G();
    auto ic = ins[0]->shape.Channel()->Pi();
    auto oc = outs[0]->shape.Channel()->Pi();
    return {ic, oc, depth_wise ? ic : g};
}

} // namespace canvas
