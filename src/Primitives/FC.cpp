#include "Canvas/Primitives/FC.hpp"


namespace canvas {

FCPrimitive::FCPrimitive(const TensorSP& t):
        Primitive("FC", {t}) {
    auto& s = t->shape;
    Shape new_shape;
    new_shape.C() = StaticVar::VC;
    new_shape.H() = s.H(), new_shape.W() = s.W();
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

FCPrimitive::FCPrimitive(const TensorSP& t, const Variable& nc):
        Primitive("FC", {t}) {
    auto& s = t->shape;
    Shape new_shape;
    new_shape.C() = nc;
    new_shape.H() = s.H(), new_shape.W() = s.W();
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

std::vector<Variable> FCPrimitive::IntermediateVariables() const {
    return {outs[0]->shape.GCKK() / ins[0]->shape.G()};
}

std::tuple<size_t, size_t, size_t, size_t>
FCPrimitive::GetConcreteSpecs(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    auto filter_shape = ins[0]->shape;
    filter_shape.G() = Variable(), filter_shape.H() = Variable(), filter_shape.W() = Variable();
    size_t n_groups = ins[0]->shape.G().FillToInteger(specs, fills);
    size_t ic_per_group = filter_shape.Pi().FillToInteger(specs, fills);
    size_t oc = outs[0]->shape.GCKK().FillToInteger(specs, fills);
    assert(oc > 0 and oc % n_groups == 0);
    size_t oc_per_group = oc / n_groups;
    size_t map_size = (ins[0]->shape.H() * ins[0]->shape.W()).FillToInteger(specs, fills);
    return {n_groups, ic_per_group, oc_per_group, map_size};
}

size_t FCPrimitive::PsCount(const Variable::StaticSpecs& specs,
                            const Variable::DynamicFills& fills) const {
    auto [n_groups, ic_per_group, oc_per_group, map_size] = GetConcreteSpecs(specs, fills);
    size_t total = n_groups * ic_per_group * oc_per_group; // Without bias
    if (with_norm)
        total += n_groups * oc_per_group * 4;
    return total;
}

size_t FCPrimitive::FLOPsCount(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    auto [n_groups, ic_per_group, oc_per_group, map_size] = GetConcreteSpecs(specs, fills);
    size_t total = n_groups * ic_per_group * oc_per_group * map_size * 2; // Without bias
    if (with_norm)
        total += n_groups * oc_per_group * map_size * 2;
    if (with_relu)
        total += n_groups * oc_per_group * map_size;
    return total;
}

} // End namespace canvas
