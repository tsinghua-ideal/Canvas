#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "Canvas/Utils/Common.hpp"
#include "Canvas/Utils/Exceptions.hpp"


namespace canvas {

struct Variable;
struct VarSolution;

struct Variable {
    static constexpr int kStaticVarCount = 4;
    static constexpr int kDynamicVarCount = 8;
    static constexpr int kFactorThreshold = 1000;

    struct VarSpecs {
        size_t g, c, h, w;

        VarSpecs(size_t g, size_t c, size_t h, size_t w): g(g), c(c), h(h), w(w) {}
    };

    /// Variable position indices.
    enum StaticVarPos {
        VG = 0,     // Groups.
        VC = 1,     // Channels.
        VH = 2,     // Height.
        VW = 3,     // Width.
        VDG = 4,    // Divided by groups (not really exists).
    };

    static const char* var_info[kStaticVarCount];

    size_t numeric_numerator = 1, numeric_denominator = 1;
    int static_power[kStaticVarCount] = {0}, dynamic_power[kDynamicVarCount] = {0};

    Variable() = default;

    Variable(size_t numeric_numerator, size_t numeric_denominator):
            numeric_numerator(numeric_numerator), numeric_denominator(numeric_denominator) {}

    Variable(const Variable& rhs) = default;

    [[nodiscard]] static Variable Number(size_t numeric_numerator=1, size_t numeric_denominator=1) {
        assert(numeric_numerator > 0 and numeric_denominator > 0);
        Variable var(numeric_numerator, numeric_denominator);
        var.Simplify();
        return var;
    }

    [[nodiscard]] static Variable StaticVar(const StaticVarPos& dim) {
        Variable var;
        var.static_power[dim] = 1;
        return var;
    }

    [[nodiscard]] static Variable DynamicVar(int i) {
        assert(0 <= i and i < kDynamicVarCount);
        Variable var;
        var.dynamic_power[i] = 1;
        return var;
    }

    [[nodiscard]] static Variable Compose(const std::initializer_list<StaticVarPos>& dims,
                                          size_t numeric_numerator=1, size_t numeric_denominator=1,
                                          const std::initializer_list<int>& dyn_vars={});

    [[nodiscard]] static Variable CHW() {
        return Compose({StaticVarPos::VC, StaticVarPos::VH, StaticVarPos::VW});
    }

    void Reset() {
        std::memset(this, 0, sizeof(Variable));
        numeric_numerator = numeric_denominator = 1;
    }

    [[nodiscard]] size_t FillToInteger(const VarSpecs& specs) const;

    void Simplify() {
        auto gcd = std::gcd(numeric_numerator, numeric_denominator);
        numeric_numerator /= gcd, numeric_denominator /= gcd;
    }

    [[nodiscard]] bool SatisfyAssumption() const {
        assert(numeric_numerator > 0 and numeric_denominator > 0);
        return Denominator().IsStatic() and Numerator().DynamicVarCount() <= 1;
    }

    [[nodiscard]] bool IsNumber() const {
        return std::all_of(static_power, static_power + kStaticVarCount,
                           [](uint8_t x) -> bool { return x == 0; });
    }

    [[nodiscard]] bool IsStatic() const {
        return std::all_of(dynamic_power, dynamic_power + kDynamicVarCount,
                           [](uint8_t x) -> bool { return x == 0; });
    }

    [[nodiscard]] int DynamicVarCount() const;

    [[nodiscard]] int GetFirstLinearDynamicVar() const;

    [[nodiscard]] int GetOnlyDynamicVar() const;

    /// Solve dynamic variable `i` with `x`.
    void SolveDynamicVar(const VarSolution &s);

    [[nodiscard]] Variable Reciprocal() const;

    [[nodiscard]] Variable NumberFactor() const {
        Variable factor;
        factor.numeric_numerator = numeric_numerator;
        factor.numeric_denominator = numeric_denominator;
        return factor;
    }

    [[nodiscard]] Variable StaticVarFactor() const {
        Variable factor;
        for (int i = 0; i < kStaticVarCount; ++ i)
            factor.static_power[i] = static_power[i];
        return factor;
    }

    [[nodiscard]] Variable StaticFactor() const {
        return NumberFactor() * StaticVarFactor();
    }

    [[nodiscard]] Variable Numerator() const;

    [[nodiscard]] Variable Denominator() const;

    /// Return whether maybe an integer for early pruning.
    [[nodiscard]] bool MaybeInteger() const;

    [[nodiscard]] bool Empty() const {
        auto func = [](int x) -> bool { return x == 0; };
        return numeric_numerator == 1 and numeric_denominator == 1 and
               std::all_of(static_power, static_power + kStaticVarCount, func) and
               std::all_of(dynamic_power, dynamic_power + kDynamicVarCount, func);
    }

    Variable& operator = (const StaticVarPos& dim) {
        Reset();
        static_power[dim] = 1;
        return *this;
    }

    [[nodiscard]] bool operator == (const Variable& rhs) const;

    [[nodiscard]] bool operator != (const Variable& rhs) const {
        return not (*this == rhs);
    }

    [[nodiscard]] Variable operator * (const Variable& rhs) const;

    Variable& operator *= (const Variable& rhs) {
        *this = *this * rhs;
        return *this;
    }

    [[nodiscard]] Variable operator * (const StaticVarPos& rhs) const {
        return *this * Variable::StaticVar(rhs);
    }

    Variable& operator *= (const StaticVarPos& rhs) {
        *this = *this * rhs;
        return *this;
    }

    Variable operator / (const Variable& rhs) const {
        return *this * rhs.Reciprocal();
    }

    Variable& operator /= (const Variable& rhs) {
        *this = *this / rhs;
        return *this;
    }

    Variable operator / (const StaticVarPos& rhs) const {
        return *this / Variable::StaticVar(rhs);
    }

    Variable& operator /= (const StaticVarPos& rhs) {
        *this = *this / rhs;
        return *this;
    }

    friend Variable operator * (const StaticVarPos& lhs, const StaticVarPos &rhs) {
        return Variable::StaticVar(lhs) * Variable::StaticVar(rhs);
    }

    friend Variable operator / (const StaticVarPos& lhs, const StaticVarPos &rhs) {
        return Variable::StaticVar(lhs) / Variable::StaticVar(rhs);
    }

    [[nodiscard]] size_t Hash() const;

    [[nodiscard]] std::vector<Variable> GetAllFactors() const;

    std::string Format(const char* *info, const std::string& mul, const std::string& div,
                       const std::string& x_prefix="x_", const std::string& x_suffix="") const;

    friend std::ostream& operator << (std::ostream& os, const Variable& rhs) {
        return os << rhs.Format(var_info, "*", "/");
    }

    static Variable Lcm(const Variable& lhs, const Variable& rhs);

    static Variable Gcd(const Variable& lhs, const Variable& rhs);
};

struct VarSolution {
    int index;
    Variable substitution;

    VarSolution(int index, const Variable& substitution): index(index), substitution(substitution) {}

    static std::optional<VarSolution> Solve(const Variable& lhs, const Variable& rhs) {
        // Solve variable relationships between `lhs` and `rhs`, returning dynamic variable substitution.
        auto simplified = lhs / rhs;
        auto equ_lhs = simplified.Numerator(), equ_rhs = simplified.Denominator();
        assert(equ_lhs.Denominator().Empty() and equ_rhs.Denominator().Empty());
        int linear_dyn_var = equ_lhs.GetFirstLinearDynamicVar();
        if (linear_dyn_var == kInvalidIndex) {
            std::swap(equ_lhs, equ_rhs);
            linear_dyn_var = equ_lhs.GetFirstLinearDynamicVar();
        }
        if (linear_dyn_var == kInvalidIndex)
            return std::nullopt;
        assert(equ_lhs.dynamic_power[linear_dyn_var] == 1);
        equ_lhs.dynamic_power[linear_dyn_var] = 0;
        auto repl = equ_rhs / equ_lhs;
        return repl.MaybeInteger() ? std::make_optional<VarSolution>(linear_dyn_var, repl) : std::nullopt;
    }
};

using StaticVarPos = Variable::StaticVarPos;

class CanNotSolveDynamicVarOnGraph: public ExceptionWithInfo {
public:
    explicit CanNotSolveDynamicVarOnGraph(const VarSolution& s) {
        std::stringstream ss;
        ss << "Can not apply \"x_" << s.index << " = " << s.substitution << "\" on the graph";
        info = ss.str();
    }
};

} // namespace canvas

CanvasHashTemplate(canvas::Variable, .Hash());
