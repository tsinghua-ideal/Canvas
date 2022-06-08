#include "Canvas/Primitives/Unfold.hpp"


namespace canvas {

UnfoldPrimitive::UnfoldPrimitive(const TensorSP& t, int k, int d, UnfoldType type):
        k(k), d(d), type(type), Primitive(UnfoldTypeToName(type), {t}) {
    assert(k > 1 and k % 2 == 1 and d >= 1);
    auto new_shape = t->shape;
    if (type == UnfoldH or type == UnfoldHW) {
        if (not new_shape.KH().Empty() or new_shape.H().Empty())
            throw CanNotApplyPrimitive(UnfoldTypeToName(type));
        new_shape.KH() = Variable::Number(k);
    }
    if (type == UnfoldW or type == UnfoldHW) {
        if (not new_shape.KW().Empty() or new_shape.W().Empty())
            throw CanNotApplyPrimitive(UnfoldTypeToName(type));
        new_shape.KW() = Variable::Number(k);
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
