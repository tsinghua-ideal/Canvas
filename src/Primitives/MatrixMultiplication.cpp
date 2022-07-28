#include "Canvas/Primitives/MatrixMultiplication.hpp"


namespace canvas {

MatrixMultiplicationPrimitive::MatrixMultiplicationPrimitive(const TensorSP& lhs, const TensorSP& rhs,
                                                             bool transpose_lhs, bool transpose_rhs,
                                                             bool with_softmax):
        Primitive(MatrixMultiplicationToName(transpose_lhs, transpose_rhs), {lhs, rhs}, false),
        transpose_lhs(transpose_lhs), transpose_rhs(transpose_rhs), with_softmax(with_softmax) {
    // No checks during construction, dynamic variable solving may occur before code generation.
    auto shape = Shape(lhs->shape.dims[transpose_lhs]->Copy(), rhs->shape.dims[not transpose_rhs]->Copy());
    outs.push_back(std::make_shared<Tensor>(shape));
}

std::vector<PrimitiveApply> MatrixMultiplicationPrimitive::GetAllPossibleMatches(const TensorSP& lhs,
                                                                                 const TensorSP& rhs,
                                                                                 bool with_softmax) {
    std::vector<PrimitiveApply> collections;
    for (int transpose_lhs = 0; transpose_lhs < 2; ++ transpose_lhs) {
        for (int transpose_rhs = 0; transpose_rhs < 2; ++ transpose_rhs) {
            // BMM: [N, A, B] * [N, B, C]
            auto lhs_b = lhs->shape.dims[transpose_lhs ^ 1]->Pi();
            auto rhs_b = rhs->shape.dims[transpose_rhs]->Pi();
            auto p = std::make_shared<MatrixMultiplicationPrimitive>(lhs, rhs, transpose_lhs, transpose_rhs, with_softmax);
            if (lhs_b == rhs_b) {
                collections.emplace_back(p);
            } else {
                auto var_sol = VarSolution::Solve(lhs_b, rhs_b);
                if (var_sol.has_value())
                    collections.emplace_back(p, var_sol);
            }
        }
    }
    return collections;
}

} // namespace canvas
