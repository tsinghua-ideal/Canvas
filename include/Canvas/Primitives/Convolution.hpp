#pragma once

#include "Canvas/Core/Primitive.hpp"

namespace canvas {

static std::string ConvolutionToName(int kh, int kw, int dh, int dw, bool depth_wise) {
    std::stringstream ss;
    ss << "Convolution_" << kh << "x" << kw << "_" << dh << "x" << dw << "_DW" << depth_wise;
    return ss.str();
}

struct ConvolutionPrimitive: Primitive {
    int kh, kw, dh, dw;
    bool depth_wise = false;

    ConvolutionPrimitive(const TensorSP& t,
                         const Variable& oc,
                         int kh, int kw, int dh, int dw,
                         bool depth_wise);

    [[nodiscard]] std::vector<Variable> IntermediateVariables() const final;

    CanvasPrimitiveCopyTemplate(ConvolutionPrimitive);
};

}
