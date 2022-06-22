#include "Canvas/Primitives/Shift.hpp"


namespace canvas {

ShiftPrimitive::ShiftPrimitive(const TensorSP& t, const std::vector<Shape::Index>& indices, int k):
        indices(indices), k(k), Primitive(ShiftToName(t->shape, indices, k), {t}, false) {
    assert(not indices.empty());
    auto& s = t->shape;
    for (const auto& index: indices)
        if (s[index].Empty())
            throw CanNotApplyPrimitive(ShiftToName(t->shape, indices, k));
    outs.push_back(std::make_shared<Tensor>(s));
}

} // namespace canvas
