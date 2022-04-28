#include "Canvas/Primitives/Transpose.hpp"


namespace canvas {

TransposePrimitive::TransposePrimitive(const TensorSP& t):
        Primitive("Transpose", {t}, false) {
    auto& s = t->shape;
    if (s.H().Empty() or s.W().Empty())
        throw CanNotApplyPrimitive("Transpose");
    outs.push_back(std::make_shared<Tensor>(s));
}

} // End namespace canvas
