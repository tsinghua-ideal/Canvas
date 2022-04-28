#include "Canvas/Primitives/Shift.hpp"


namespace canvas {

ShiftPrimitive::ShiftPrimitive(const TensorSP& t, ShiftType type):
        type(type), Primitive(ShiftTypeToName(type), {t}, false) {
    auto& s = t->shape;
    if ((type == ShiftH or type == ShiftHW) and s.H().Empty())
        throw CanNotApplyPrimitive(ShiftTypeToName(type));
    if ((type == ShiftW or type == ShiftHW) and s.W().Empty())
        throw CanNotApplyPrimitive(ShiftTypeToName(type));
    outs.push_back(std::make_shared<Tensor>(s));
}

} // End namespace canvas
