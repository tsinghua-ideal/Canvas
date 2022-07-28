#pragma once

#include <vector>

#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Variable.hpp"


namespace canvas {

static std::string MatrixMultiplicationToName(bool transpose_lhs, bool transpose_rhs) {
    std::stringstream ss;
    ss << "BMM_" << transpose_lhs << "_" << transpose_rhs;
    return ss.str();
}

struct MatrixMultiplicationPrimitive: Primitive {
    bool transpose_lhs, transpose_rhs;
    bool with_softmax;

    MatrixMultiplicationPrimitive(const TensorSP& lhs, const TensorSP& rhs,
                                  bool transpose_lhs, bool transpose_rhs,
                                  bool with_softmax=false);

    static std::vector<PrimitiveApply> GetAllPossibleMatches(const TensorSP& lhs, const TensorSP& rhs, bool with_softmax);

    CanvasPrimitiveCopyTemplate(MatrixMultiplicationPrimitive);
};

} // namespace canvas
