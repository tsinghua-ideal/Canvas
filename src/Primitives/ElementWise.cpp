#include "Canvas/Primitives/ElementWise.hpp"


namespace canvas {

ElementWisePrimitive::ElementWisePrimitive(const TensorSP& t, ElementWiseType type):
        type(type), Primitive(ElementWiseTypeToName(type), {t}, false) {
    outs.push_back(std::make_shared<Tensor>(t->shape));
}

size_t ElementWisePrimitive::FLOPsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    return ins[0]->shape.Pi().FillToInteger(specs, fills); // TODO: add FLOPs factor
}

} // End namespace canvas
