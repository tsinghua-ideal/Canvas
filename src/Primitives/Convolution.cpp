#include "Canvas/Primitives/Convolution.hpp"


namespace canvas {

ConvolutionPrimitive::ConvolutionPrimitive(const TensorSP& t,
                                           const Variable& oc, const Variable& g,
                                           int kh, int kw, int dh, int dw):
        Primitive(ConvolutionToName(g, kh, kw, dh, dw), {t}, false),
        oc(oc), g(g), kh(kh), kw(kw), dh(dh), dw(dw) {
    assert(t->shape.IsChannelSpatial());
    assert(kh >= 1 and kh % 2 == 1 and kw >= 1 and kw % 2 == 1);
    assert(dh >= 1 and dw >= 1);
    auto new_shape = t->shape;
    auto channel = new_shape.Channel();
    channel->Reset();
    channel->C() = oc;
    outs.push_back(std::make_shared<Tensor>(new_shape));
}

void ConvolutionPrimitive::SolveDynamicVar(const VarSolution& s) {
    Primitive::SolveDynamicVar(s);
    g.SolveDynamicVar(s);
    oc.SolveDynamicVar(s);
    if (not g.MaybeInteger() or not oc.MaybeInteger())
        throw CanNotSolveDynamicVarOnGraph(s);
}

std::vector<Variable> ConvolutionPrimitive::IntermediateVariables() const {
    return {ins[0]->shape.Channel()->Pi() / g, oc / g};
}

} // namespace canvas
