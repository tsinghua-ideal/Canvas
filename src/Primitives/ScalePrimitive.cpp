#include "Canvas/Primitives/Scale.hpp"

namespace canvas {

ScalePrimitive::ScalePrimitive(const TensorSP& t, const std::vector<Shape::Index>& indices):
        indices(indices), Primitive(ScaleToName(t->shape, indices), {t}, false) {
    assert(not indices.empty());
    auto& s = t->shape;
    for (const auto& index: indices)
        assert(not s[index].Empty());
    outs.push_back(std::make_shared<Tensor>(s));
}

} // namespace canvas
