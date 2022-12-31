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

std::vector<Variable> ScalePrimitive::ParamShape() const {
    auto sorted = indices;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& lhs, const auto& rhs) -> bool {
                  return lhs.d == rhs.d ? lhs.k < rhs.k : lhs.d < rhs.d;
              });

    auto t_shape = ins[0]->shape;
    int next = 0;
    std::vector<Variable> weight_shape;
    for (int d = 0; d < 2; ++ d) {
        for (int k = 0; k < MetaShape::kMaxMetaDims; ++ k) {
            auto index = Shape::Index(d, k);
            if (t_shape[index].Empty())
                continue;
            if (next < sorted.size() and index == sorted[next])
                weight_shape.emplace_back(t_shape[index]);
            else
                weight_shape.emplace_back();
        }
    }
    return weight_shape;
}

} // namespace canvas
