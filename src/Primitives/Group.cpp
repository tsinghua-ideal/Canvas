#include "Canvas/Primitives/Group.hpp"


namespace canvas {

GroupPrimitive::GroupPrimitive(const TensorSP& t, GroupType type):
        type(type), Primitive(GroupTypeToName(type), {t}, false) {
    Shape new_shape = t->shape;
    if (type == GroupByFactor) {
        new_shape.C() /= StaticVarPos::VG;
        if (not new_shape.G().Empty() or not new_shape.C().MaybeInteger())
            throw CanNotApplyPrimitive(GroupTypeToName(type));
        new_shape.G() = StaticVarPos::VG;
    } else if (type == GroupAllChannels) {
        if (not new_shape.G().Empty() or new_shape.CKK().Empty())
            throw CanNotApplyPrimitive(GroupTypeToName(type));
        new_shape.G() = new_shape.CKK();
        new_shape.C().Reset(), new_shape.KH().Reset(), new_shape.KW().Reset();
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
