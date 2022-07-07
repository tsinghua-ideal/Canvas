#include "Canvas/Primitives/Softmax.hpp"


namespace canvas {

SoftmaxPrimitive::SoftmaxPrimitive(const TensorSP& t, const Shape::Index& index):
        index(index), Primitive(SoftmaxToName(t->shape, index), {t}, false) {
    assert(not t->shape[index].Empty());
    outs.push_back(std::make_shared<Tensor>(t->shape));
}

} // namespace canvas
