#include "Canvas/Primitives/Reshape.hpp"


namespace canvas {

ReshapePrimitive::ReshapePrimitive(const TensorSP& t, const Shape& new_shape):
        Primitive("Reshape", {t}, false) {
    if (t->shape.Pi() != new_shape.Pi())
        throw CanNotApplyPrimitive("Reshape");
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // End namespace canvas
