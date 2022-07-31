#pragma once

#include <vector>

#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Variable.hpp"


namespace canvas {

enum BroadcastType {
    BAdd,
    BMul,
    BSub,
    BMax,
    BMin
};

static constexpr const char* BroadcastTypeToName(BroadcastType type) {
    switch (type) {
        case BAdd: return "BAdd";
        case BMul: return "BMul";
        case BSub: return "BSub";
        case BMax: return "BMax";
        case BMin: return "BMin";
    }
    return "";
}

/// Broadcasting primitives (add, sub, mul)
///   Broadcast `lhs` (with a shape of [CP, L_1, L_2, ..., CS]) into `rhs` (with a shape of [CP, R_1, R_2, ..., CS])
///   Then let `lhs_pi` = \Pi_{L_i} (ensure to be an integer),
///     `rhs_pi` = \Pi_{R_i}, multiplier = `rhs_pi` / `lhs_pi` (ensure to be an integer);
///   Reshape `lhs` into a shape of [CP, 1, `lhs_pi`, CS];
///   Reshape `rhs` into a shape of [CP, multiplier, `lhs_pi`, CS].
struct BroadcastPrimitive: Primitive {
    static constexpr int kPossibleSolutionsLimit = 12;

    bool aligned;
    BroadcastType type;
    Variable lhs_pi, rhs_pi, multiplier;
    std::vector<Variable> prefix, suffix;

    /// Broadcast `lhs` onto `rhs`
    BroadcastPrimitive(const TensorSP& lhs, const TensorSP& rhs, BroadcastType type);

    BroadcastPrimitive(const TensorSP& lhs, const TensorSP& rhs, BroadcastType type, bool aligned,
                       const Variable& lhs_pi, const Variable& rhs_pi, const Variable& multiplier,
                       std::vector<Variable> prefix, std::vector<Variable> suffix);

    static std::vector<PrimitiveApply> GetAllPossibleMatches(const TensorSP& lhs, const TensorSP& rhs,
                                                             BroadcastType type, int limit=kPossibleSolutionsLimit);

    void InferShapes(const TensorSP& lhs, const TensorSP& rhs);

    void SolveDynamicVar(const VarSolution& s) final;

    [[nodiscard]] std::vector<Variable> IntermediateVariables() const final;

    [[nodiscard]] char TypeToSign() const {
        switch (type) {
            case BAdd: return '+';
            case BMul: return '*';
            case BSub: return '-';
            case BMax: Unreachable();
            case BMin: Unreachable();
        }
        Unreachable();
    }

    CanvasPrimitiveCopyTemplate(BroadcastPrimitive);
};

} // namespace canvas
