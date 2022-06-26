#pragma once

#include "Canvas/Core/Primitive.hpp"

namespace canvas {

static std::string ConvolutionToName(const Variable& g, int kh, int kw, int dh, int dw) {
    std::stringstream ss;
    ss << "Convolution_" << g << "_" << kh << "_" << kw << "_" << dh << "_" << dw;
    return ss.str();
}

struct ConvolutionPrimitive: Primitive {
    Variable oc, g;
    int kh, kw, dh, dw;

    ConvolutionPrimitive(const TensorSP& t,
                         const Variable& oc, const Variable& g,
                         int kh, int kw, int dh, int dw);

    void SolveDynamicVar(const VarSolution& s) final;

    [[nodiscard]] std::vector<Variable> IntermediateVariables() const final;

    CanvasPrimitiveCopyTemplate(ConvolutionPrimitive);
};

}
