#include "Canvas/Primitives/ElementWise.hpp"


namespace canvas {

ElementWisePrimitive::ElementWisePrimitive(const TensorSP& t, ElementWiseType type):
        type(type), Primitive(ElementWiseTypeToName(type), {t}, false) {
    outs.push_back(std::make_shared<Tensor>(t->shape));
}

} // namespace canvas
