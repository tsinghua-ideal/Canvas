#include "Canvas/Primitives/Shift.hpp"


namespace canvas {

ShiftPrimitive::ShiftPrimitive(const TensorSP& t, const std::vector<Shape::DimPos>& pos_vec, int k):
        pos_vec(pos_vec), k(k), Primitive(ShiftToName(pos_vec, k), {t}, false) {
    assert(not pos_vec.empty());
    auto& s = t->shape;
    for (const auto& pos: pos_vec)
        if (s.dims[pos].Empty())
            throw CanNotApplyPrimitive(ShiftToName(pos_vec, k));
    outs.push_back(std::make_shared<Tensor>(s));
}

} // namespace canvas
