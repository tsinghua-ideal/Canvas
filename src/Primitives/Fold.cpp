#include "Canvas/Primitives/Fold.hpp"


namespace canvas {

FoldPrimitive::FoldPrimitive(const TensorSP& t, const std::vector<Shape::DimPos>& pos_vec, FoldType type):
        pos_vec(pos_vec), type(type), Primitive(FoldTypeToName(pos_vec, type), {t}, false) {
    assert(not pos_vec.empty());
    Shape new_shape = t->shape;
    for (const auto& pos: pos_vec) {
        int i = static_cast<int>(pos);
        assert(0 <= i and i < Shape::kShapeMaxDim);
        if (new_shape.dims[i].Empty())
            throw CanNotApplyPrimitive(FoldTypeToName(pos_vec, type));
        else
            new_shape.dims[i].Reset();
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
