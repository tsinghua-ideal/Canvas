#include "Canvas/Primitives/Dropout.hpp"


namespace canvas {

DropoutPrimitive::DropoutPrimitive(const TensorSP& t):
        Primitive("Dropout", {t}, false) {
    outs.push_back(std::make_shared<Tensor>(t->shape));
}

} // namespace canvas
