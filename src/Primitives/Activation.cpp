#include "Canvas/Primitives/Activation.hpp"


namespace canvas {

ActivationPrimitive::ActivationPrimitive(const TensorSP& t, ActivationType type):
        type(type), Primitive(ActivationTypeToName(type), {t}, false) {
    outs.push_back(std::make_shared<Tensor>(t->shape));
}

size_t ActivationPrimitive::FLOPsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    return ins[0]->shape.Pi().FillToInteger(specs, fills) * ActivationFLOPsFactor(type);
}

} // End namespace canvas
