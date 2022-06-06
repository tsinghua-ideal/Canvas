#include "Canvas/Primitives/Activation.hpp"


namespace canvas {

ActivationPrimitive::ActivationPrimitive(const TensorSP& t, ActivationType type):
        type(type), Primitive(ActivationTypeToName(type), {t}, false) {
    outs.push_back(std::make_shared<Tensor>(t->shape));
}

} // namespace canvas
