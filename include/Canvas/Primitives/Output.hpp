#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

/// Output primitive for ending the graph.
struct OutputPrimitive: Primitive {
    explicit OutputPrimitive(const TensorSP& t):
            Primitive("Output", {t}, false) {
        if (not t->consumers.empty())
            throw CanNotApplyPrimitive("Output");
        if (not t->shape.CouldBeReshapedToCHW())
            throw CanNotApplyPrimitive("Output");
        outs.push_back(std::make_shared<Tensor>(Shape::MakeShapeCHW()));
    }

    CanvasPrimitiveCopyTemplate(OutputPrimitive);
};

} // namespace canvas
