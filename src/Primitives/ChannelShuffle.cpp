#include "Canvas/Primitives/ChannelShuffle.hpp"


namespace canvas {

ChannelShufflePrimitive::ChannelShufflePrimitive(const TensorSP& t):
        Primitive("ChannelShuffle", {t}, false) {
    auto& s = t->shape;
    if (not (s.GCKK() / StaticVar::VG).MaybeInteger())
        throw CanNotApplyPrimitive("ChannelShuffle");
    outs.push_back(std::make_shared<Tensor>(s));
}

std::vector<Variable> ChannelShufflePrimitive::IntermediateVariables() const {
    return {ins[0]->shape.GCKK() / StaticVar::VG};
}

} // namespace canvas
