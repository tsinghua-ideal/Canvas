#include "Canvas/Primitives/Pool.hpp"


namespace canvas {

PoolPrimitive::PoolPrimitive(const TensorSP& t, PoolType type):
        type(type), Primitive(PoolTypeToName(type), {t}, false) {
    auto new_shape = t->shape;
    if (type == PoolH or type == PoolHW) {
        if (new_shape.H().Empty())
            throw CanNotApplyPrimitive(PoolTypeToName(type));
        new_shape.H().Reset();
    }
    if (type == PoolW or type == PoolHW) {
        if (new_shape.W().Empty())
            throw CanNotApplyPrimitive(PoolTypeToName(type));
        new_shape.W().Reset();
    }
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

size_t PoolPrimitive::FLOPsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    return ins[0]->shape.Pi().FillToInteger(specs, fills);
}

} // End namespace canvas
