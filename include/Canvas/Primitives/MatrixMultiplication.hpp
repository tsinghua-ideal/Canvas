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

    MatrixMultiplicationPrimitive(const TensorSP& lhs, const TensorSP& rhs,
                                  bool transpose_lhs, bool transpose_rhs);

    static std::vector<PrimitiveApply> GetAllPossibleMatches(const TensorSP& lhs, const TensorSP& rhs);

    CanvasPrimitiveCopyTemplate(MatrixMultiplicationPrimitive);
};

} // namespace canvas
