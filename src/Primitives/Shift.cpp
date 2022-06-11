#include "Canvas/Primitives/Shift.hpp"


namespace canvas {

ShiftPrimitive::ShiftPrimitive(const TensorSP& t, ShiftType type, int k):
        type(type), k(k), Primitive(ShiftTypeToName(type, k), {t}, false) {
    auto& s = t->shape;
    if ((type == ShiftH or type == ShiftHW) and s.H().Empty())
        throw CanNotApplyPrimitive(ShiftTypeToName(type, k));
    if ((type == ShiftW or type == ShiftHW) and s.W().Empty())
        throw CanNotApplyPrimitive(ShiftTypeToName(type, k));
    outs.push_back(std::make_shared<Tensor>(s));
}

} // namespace canvas
