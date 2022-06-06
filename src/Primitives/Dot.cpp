#include "Canvas/Primitives/Dot.hpp"


namespace canvas {

DotPrimitive::DotPrimitive(const TensorSP& t):
        Primitive("Dot", {t}, false) {
    outs.push_back(std::make_shared<Tensor>(t->shape));
}

} // namespace canvas
