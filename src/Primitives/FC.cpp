#include "Canvas/Primitives/FC.hpp"


namespace canvas {

FCPrimitive::FCPrimitive(const TensorSP& t):
        Primitive("FC", {t}) {
    assert(t->shape.IsChannelSpatial());
    Shape new_shape = Shape::MakeChannelSpatial();
    new_shape.Channel()->C() = Variable::StaticVar(StaticVarPos::VC);
    new_shape.Spatial()->H() = t->shape.Spatial()->H();
    new_shape.Spatial()->W() = t->shape.Spatial()->W();
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

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

} // namespace canvas
