#include "Canvas/Primitives/Input.hpp"


namespace canvas {

InputPrimitive::InputPrimitive():
        Primitive("Input", {}) {
    Shape shape;
    shape.C() = StaticVar::VC, shape.H() = StaticVar::VH, shape.W() = StaticVar::VW;
    outs.push_back(std::make_shared<Tensor>(shape));
}

} // namespace canvas
