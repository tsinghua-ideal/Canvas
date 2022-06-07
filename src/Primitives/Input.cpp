#include "Canvas/Primitives/Input.hpp"


namespace canvas {

InputPrimitive::InputPrimitive():
        Primitive("Input", {}) {
    Shape shape;
    shape.C() = StaticVarPos::VC, shape.H() = StaticVarPos::VH, shape.W() = StaticVarPos::VW;
    outs.push_back(std::make_shared<Tensor>(shape));
}

} // namespace canvas
