#include "Canvas/Primitives/Unfold.hpp"


namespace canvas {

UnfoldPrimitive::UnfoldPrimitive(const TensorSP& t, UnfoldType type):
        type(type), Primitive(UnfoldTypeToName(type), {t}) {
    auto new_shape = t->shape;
    if (type == UnfoldH or type == UnfoldHW) {
        if (not new_shape.KH().Empty() or new_shape.H().Empty())
            throw CanNotApplyPrimitive(UnfoldTypeToName(type));
        new_shape.KH() = StaticVar::VKH;
    }
    if (type == UnfoldW or type == UnfoldHW) {
        if (not new_shape.KW().Empty() or new_shape.W().Empty())
            throw CanNotApplyPrimitive(UnfoldTypeToName(type));
        new_shape.KW() = StaticVar::VKW;
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
