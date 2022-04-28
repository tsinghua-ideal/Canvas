#include "Canvas/Primitives/Reorder.hpp"


namespace canvas {

ReorderPrimitive::ReorderPrimitive(const TensorSP& t, ReorderType type, bool inverse):
        Primitive(ReorderTypeToName(type, inverse), {t}, false), type(type), inverse(inverse) {
    auto& s = t->shape;
    if (type == ReorderH or type == ReorderHW)
        if (s.H().Empty())
            throw CanNotApplyPrimitive(ReorderTypeToName(type, inverse));
    if (type == ReorderW or type == ReorderHW)
        if (s.W().Empty())
            throw CanNotApplyPrimitive(ReorderTypeToName(type, inverse));
    outs.push_back(std::make_shared<Tensor>(s));
}

} // End namespace canvas
