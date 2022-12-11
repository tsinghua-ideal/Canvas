#pragma once

#include "Canvas/Core/Primitive.hpp"


namespace canvas {

/// Output primitive for ending the graph.
struct OutputPrimitive: Primitive {
    explicit OutputPrimitive(const TensorSP& t, const Shape& output_shape=Shape::MakeShapeCHW()):
            Primitive("Output", {t}, false) {
        if (not t->consumers.empty())
            throw CanNotApplyPrimitive("Output");
        if (t->shape.Pi() != output_shape.Pi())
            throw CanNotApplyPrimitive("Output");
        outs.push_back(std::make_shared<Tensor>(output_shape));
    }

    CanvasPrimitiveCopyTemplate(OutputPrimitive);
};

} // namespace canvas
