#include "Canvas/Primitives/Softmax.hpp"


namespace canvas {

SoftmaxPrimitive::SoftmaxPrimitive(const TensorSP& t, SoftmaxType type):
        type(type), Primitive(SoftmaxTypeToName(type), {t}, false) {
    auto& s = t->shape;
    if (type == SoftmaxC and s.GCKK().Empty())
        throw CanNotApplyPrimitive(SoftmaxTypeToName(type));
    if ((type == SoftmaxH or type == SoftmaxHW) and s.H().Empty())
        throw CanNotApplyPrimitive(SoftmaxTypeToName(type));
    if ((type == SoftmaxW or type == SoftmaxHW) and s.W().Empty())
        throw CanNotApplyPrimitive(SoftmaxTypeToName(type));
    outs.push_back(std::make_shared<Tensor>(s));
}

size_t SoftmaxPrimitive::FLOPsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    return ins[0]->shape.Pi().FillToInteger(specs, fills) * 3; // Exp, add and division
}

} // End namespace canvas
