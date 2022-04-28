#include "Canvas/Primitives/Norm.hpp"


namespace canvas {

NormPrimitive::NormPrimitive(const TensorSP& t):
        Primitive("Norm", {t}, false) {
    auto& s = t->shape;
    if (s.H().Empty() and s.W().Empty())
        throw CanNotApplyPrimitive("Norm");
    outs.push_back(std::make_shared<Tensor>(s));
}

size_t NormPrimitive::PsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    // \gamma, \beta, moving_mean and moving_variance
    return ins[0]->shape.GCKK().FillToInteger(specs, fills) * 4;
}

size_t NormPrimitive::FLOPsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    // FMA counts 2 FLOPs
    return ins[0]->shape.Pi().FillToInteger(specs, fills) * 2;
}

} // End namespace canvas
