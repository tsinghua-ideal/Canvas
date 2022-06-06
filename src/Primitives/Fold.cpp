#include "Canvas/Primitives/Fold.hpp"


namespace canvas {

FoldPrimitive::FoldPrimitive(const TensorSP& t, FoldType type, FoldArithmeticType arith_type):
        type(type), arith_type(arith_type),
        Primitive(FoldTypeToName(type) + std::string("_") + FoldArithmeticTypeToName(arith_type), {t}, false) {
    Shape new_shape = t->shape;
    if (type == FoldH or type == FoldHW) {
        if (t->shape.KH().Empty())
            throw CanNotApplyPrimitive(FoldTypeToName(type));
        else
            new_shape.KH().Reset();
    }
    if (type == FoldW or type == FoldHW) {
        if (t->shape.KW().Empty())
            throw CanNotApplyPrimitive(FoldTypeToName(type));
        else
            new_shape.KW().Reset();
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

} // namespace canvas
