#include "Canvas/Primitives/Norm.hpp"


namespace canvas {

NormPrimitive::NormPrimitive(const TensorSP& t):
        Primitive("Norm", {t}, false) {
    auto& s = t->shape;
    if (s.H().Empty() and s.W().Empty())
        throw CanNotApplyPrimitive("Norm");
    outs.push_back(std::make_shared<Tensor>(s));
}

} // namespace canvas
