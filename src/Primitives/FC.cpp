#include "Canvas/Primitives/FC.hpp"


namespace canvas {

FCPrimitive::FCPrimitive(const TensorSP& t, const Variable& nc):
        Primitive("FC", {t}) {
    Shape new_shape = Shape::MakeChannelSpatial();
    new_shape.Channel()->C() = nc;
    new_shape.Spatial()->H() = t->shape.Spatial()->H();
    new_shape.Spatial()->W() = t->shape.Spatial()->W();
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

std::vector<Variable> FCPrimitive::IntermediateVariables() const {
    return {outs[0]->shape.Channel()->Pi() / ins[0]->shape.Channel()->G()};
}

std::vector<Variable> FCPrimitive::ParamShape() const {
    auto g = ins[0]->shape.Channel()->G();
    auto ic = ins[0]->shape.Channel()->Pi();
    auto oc = outs[0]->shape.Channel()->Pi();
    return {ic, oc, g};
}

} // namespace canvas
