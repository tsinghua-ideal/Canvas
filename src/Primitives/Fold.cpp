#include "Canvas/Primitives/Fold.hpp"


namespace canvas {

FoldPrimitive::FoldPrimitive(const TensorSP& t, const std::vector<Shape::Index>& indices, FoldType type):
        indices(indices), type(type), Primitive(FoldTypeToName(t->shape, indices, type), {t}, false) {
    assert(not indices.empty());
    Shape new_shape = t->shape;
    for (const auto& index: indices) {
        assert(not new_shape[index].Empty());
        new_shape[index].Reset();
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
