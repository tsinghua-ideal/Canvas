#include "Canvas/Primitives/Input.hpp"


namespace canvas {

InputPrimitive::InputPrimitive():
        Primitive("Input", {}) {
    Shape shape;
    shape.C() = StaticVar::VC, shape.H() = StaticVar::VH, shape.W() = StaticVar::VW;
    outs.push_back(std::make_shared<Tensor>(shape));
}

size_t InputPrimitive::FLOPsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    if (specs.s == 1)
        return 0;
    return outs[0]->shape.FillToStaticShape(specs, fills).Pi() * specs.s * specs.s;
}

} // End namespace canvas
