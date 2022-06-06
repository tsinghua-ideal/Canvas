#include "Canvas/Primitives/Output.hpp"


namespace canvas {

OutputPrimitive::OutputPrimitive(const TensorSP& t):
        Primitive("Output", {t}, false) {
    if (not t->consumers.empty())
        throw CanNotApplyPrimitive("Output");
    if (not t->shape.CouldBeReshapeToCHW())
        throw CanNotApplyPrimitive("OutputPrimitive");
    outs.push_back(std::make_shared<Tensor>(Shape::StandardCHW()));
}

} // namespace canvas
