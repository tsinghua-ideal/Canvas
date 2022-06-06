#include "Canvas/Primitives/FC.hpp"


namespace canvas {

FCPrimitive::FCPrimitive(const TensorSP& t):
        Primitive("FC", {t}) {
    auto& s = t->shape;
    Shape new_shape;
    new_shape.C() = StaticVar::VC;
    new_shape.H() = s.H(), new_shape.W() = s.W();
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

FCPrimitive::FCPrimitive(const TensorSP& t, const Variable& nc):
        Primitive("FC", {t}) {
    auto& s = t->shape;
    Shape new_shape;
    new_shape.C() = nc;
    new_shape.H() = s.H(), new_shape.W() = s.W();
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

std::vector<Variable> FCPrimitive::IntermediateVariables() const {
    return {outs[0]->shape.GCKK() / ins[0]->shape.G()};
}

} // namespace canvas
