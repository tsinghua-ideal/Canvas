#include "Canvas/Primitives/Mix.hpp"


namespace canvas {

MixPrimitive::MixPrimitive(const TensorSP& t,
                           const std::vector<Shape::Index>& indices,
                           const std::vector<Variable>& new_dims):
        indices(indices), Primitive("Mix", {t}) {
    assert(indices.size() == new_dims.size());
    auto new_shape = t->shape;
    for (int i = 0; i < indices.size(); ++ i)
        new_shape[indices[i]] = new_dims[i];
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

std::vector<Variable> MixPrimitive::ParamShape() const {
    std::vector<Variable> weight_shape;
    for (const auto& index: indices)
        weight_shape.emplace_back(ins[0]->shape[index]);
    for (const auto& index: indices)
        weight_shape.emplace_back(outs[0]->shape[index]);
    return weight_shape;
}

} // namespace canvas
