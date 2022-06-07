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


namespace canvas {

struct Variable;
struct VarSolution;

struct Variable {
    static constexpr int kStaticVarCount = 4;
    static constexpr int kDynamicVarCount = 8;

    /// Variable position indices.
    enum StaticVarPos {
        VG = 0,     // Groups.
        VC = 1,     // Channels.
        VH = 2,     // Height.
        VW = 3,     // Width.
        VDG = 4,    // Divided by groups (not really exists).
    };

    static const char* var_info[kStaticVarCount];

    int numeric_numerator = 1, numeric_denominator = 1;
    int static_power[kStaticVarCount] = {0}, dynamic_power[kDynamicVarCount] = {0};

    Variable() = default;

    Variable(const Variable& rhs) = default;

    [[nodiscard]] static Variable Number(int numeric_numerator=1, int numeric_denominator=1) {
        assert(numeric_numerator > 0 and numeric_denominator > 0);
        int gcd = std::gcd(numeric_numerator, numeric_denominator);
        Variable var;
        var.numeric_numerator = numeric_numerator / gcd;
        var.numeric_denominator = numeric_denominator / gcd;
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
                                          int numeric_numerator=1, int numeric_denominator=1,
                                          const std::initializer_list<int>& dyn_vars={}) {
        Variable var;
        for (const auto& dim: dims) {
            if (dim == VDG)
                var.static_power[StaticVarPos::VG] -= 1;
            else
                var.static_power[dim] += 1;
        }
        for (const auto& dyn_var: dyn_vars) {
            assert(dyn_var < kDynamicVarCount);
            var.dynamic_power[dyn_var] += 1;
        }
        int gcd = std::gcd(numeric_numerator, numeric_denominator);
        var.numeric_numerator = numeric_numerator / gcd;
        var.numeric_denominator = numeric_denominator / gcd;
        return var;
    }

    void Reset() {
        std::memset(this, 0, sizeof(Variable));
        numeric_numerator = numeric_denominator = 1;
    }

    [[nodiscard]] bool SatisfyAssumption() const {
        return Denominator().IsStatic() and Numerator().DynamicVarCount() <= 1;
    }

    [[nodiscard]] bool IsStatic() const {
        return std::all_of(dynamic_power, dynamic_power + kDynamicVarCount,
                           [](uint8_t x) -> bool { return x == 0; });
    }

    [[nodiscard]] bool IsDynamic() const { return not IsStatic(); }

    [[nodiscard]] int DynamicVarCount() const {
        int count = 0;
        for (const auto& var: dynamic_power)
            count += (var != 0);
        return count;
    }

    [[nodiscard]] int GetOnlyDynamicVar() const {
        int index = kInvalidIndex;
        for (int i = 0; i < kDynamicVarCount; ++ i) {
            if (dynamic_power[i]) {
                assert(index == kInvalidIndex);
                index = i;
            }
        }
        assert(index != kInvalidIndex);
        return index;
    }

    [[nodiscard]] bool IsIrregular(bool empty_irregular=true) const {
        if (*this == StaticVar(VC))
            return false;
        if (not empty_irregular and Empty())
            return false;
        if (DynamicVarCount() >= 1 and StaticFactor().Empty())
            return false;
        return true;
    }

    /// Solve dynamic variable `i` with `x`.
    void SolveDynamicVar(const VarSolution &s);

    [[nodiscard]] Variable Reciprocal() const {
        Variable reciprocal;
        for (int i = 0; i < kStaticVarCount; ++ i)
            reciprocal.static_power[i] = -static_power[i];
        for (int i = 0; i < kDynamicVarCount; ++ i)
            reciprocal.dynamic_power[i] = -dynamic_power[i];
        reciprocal.numeric_numerator = numeric_denominator;
        reciprocal.numeric_denominator = numeric_numerator;
        return reciprocal;
    }

    [[nodiscard]] Variable StaticFactor() const {
        Variable factor;
        for (int i = 0; i < kStaticVarCount; ++ i)
            factor.static_power[i] = static_power[i];
        return factor;
    }

    [[nodiscard]] Variable DynamicFactor() const {
        Variable factor;
        for (int i = 0; i < kDynamicVarCount; ++ i)
            factor.dynamic_power[i] = dynamic_power[i];
        return factor;
    }

    [[nodiscard]] Variable Numerator() const {
        Variable numerator;
        for (int i = 0; i < kStaticVarCount; ++ i)
            numerator.static_power[i] = std::max(static_power[i], 0);
        for (int i = 0; i < kDynamicVarCount; ++ i)
            numerator.dynamic_power[i] = std::max(dynamic_power[i], 0);
        return numerator;
    }

    [[nodiscard]] Variable Denominator() const {
        Variable denominator;
        for (int i = 0; i < kStaticVarCount; ++ i)
            denominator.static_power[i] = std::abs(std::min(static_power[i], 0));
        for (int i = 0; i < kDynamicVarCount; ++ i)
            denominator.dynamic_power[i] = std::abs(std::min(dynamic_power[i], 0));
        return denominator;
    }

    /// Judge whether a variable is static and an integer.
    [[nodiscard]] bool IsStaticInteger() const;

    [[nodiscard]] bool MaybeInteger() const {
        // With dynamic variables in the numerator, we can always eliminate the denominator.
        return IsDynamic() or IsStaticInteger();
    }

    [[nodiscard]] bool Empty() const {
        auto func = [](uint8_t x) -> bool { return x == 0; };
        return numeric_numerator == 1 and numeric_denominator == 1 and
               std::all_of(static_power, static_power + kStaticVarCount, func) and
               std::all_of(dynamic_power, dynamic_power + kDynamicVarCount, func);
    }

    Variable& operator = (const StaticVarPos& dim) {
        Reset();
        static_power[dim] = 1;
        return *this;
    }

    [[nodiscard]] bool operator == (const Variable& rhs) const {
        for (int i = 0; i < kStaticVarCount; ++ i)
            if (static_power[i] != rhs.static_power[i])
                return false;
        for (int i = 0; i < kDynamicVarCount; ++ i)
            if (dynamic_power[i] != rhs.dynamic_power[i])
                return false;
        return numeric_numerator == rhs.numeric_numerator and numeric_denominator == rhs.numeric_denominator;
    }

    [[nodiscard]] bool operator != (const Variable& rhs) const {
        return not (*this == rhs);
    }

    [[nodiscard]] Variable operator * (const Variable& rhs) const {
        Variable pi;
        for (int i = 0; i < kStaticVarCount; ++ i)
            pi.static_power[i] = static_power[i] + rhs.static_power[i];
        for (int i = 0; i < kDynamicVarCount; ++ i)
            pi.dynamic_power[i] = dynamic_power[i] + rhs.dynamic_power[i];
        pi.numeric_numerator = numeric_numerator * rhs.numeric_numerator;
        pi.numeric_denominator = numeric_denominator * rhs.numeric_denominator;
        int gcd = std::gcd(pi.numeric_numerator, pi.numeric_denominator);
        pi.numeric_numerator /= gcd;
        pi.numeric_denominator /= gcd;
        return pi;
    }

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

    [[nodiscard]] size_t Hash() const {
        size_t value = 0;
        for (const auto& power: static_power)
            value = IterateHash(value, power);
        for (const auto& power: dynamic_power)
            value = IterateHash(value, power);
        return value;
    }

    void RecursiveGetFactors(int i, Variable &current, std::vector<Variable>& collections,
                             bool except_hw, int extra_g_factor=0, int extra_cg_factor=0) const;

    [[nodiscard]] std::vector<Variable> GetAllFactors(bool except_hw=false) const;

    std::string Format(const char* *info, const std::string& mul, const std::string& div,
                       const std::string& x_prefix="x_", const std::string& x_suffix="") const;

    friend std::ostream& operator << (std::ostream& os, const Variable& rhs) {
        return os << rhs.Format(var_info, "*", "/");
    }
};

struct VarSolution {
    int index;
    Variable substitution;

    VarSolution(int index, const Variable& substitution): index(index), substitution(substitution) {}
};

using StaticVarPos = Variable::StaticVarPos;

class CanNotSolveDynamicVar: public ExceptionWithInfo {
public:
    explicit CanNotSolveDynamicVar(const VarSolution& s) {
        std::stringstream ss;
        ss << "Can not apply \"x_" << s.index << " = " << s.substitution << "\" on the graph";
        info = ss.str();
    }
};

} // namespace canvas

CanvasHashTemplate(canvas::Variable, .Hash());
