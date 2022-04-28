#include "Canvas/Primitives/Dot.hpp"


namespace canvas {

DotPrimitive::DotPrimitive(const TensorSP& t):
        Primitive("Dot", {t}, false) {
    outs.push_back(std::make_shared<Tensor>(t->shape));
}

size_t DotPrimitive::PsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    return ins[0]->shape.GCKK().FillToInteger(specs, fills);
}

size_t DotPrimitive::FLOPsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    return ins[0]->shape.Pi().FillToInteger(specs, fills);
}

} // End namespace canvas
