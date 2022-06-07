#pragma once

#include <algorithm>
#include <cassert>
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
    static constexpr int kStaticVarCount = 6;
    static constexpr int kDynamicVarCount = 8;

    struct DynamicFills {
        size_t x[kDynamicVarCount] = {0};

        explicit DynamicFills(size_t x0=1, size_t x1=1, size_t x2=1, size_t x3=1,
                              size_t x4=1, size_t x5=1, size_t x6=1, size_t x7=1) {
            x[0] = x0, x[1] = x1, x[2] = x2, x[3] = x3;
            x[4] = x4, x[5] = x5, x[6] = x6, x[7] = x7;
        }

        DynamicFills(const DynamicFills& rhs) { std::memcpy(x, rhs.x, sizeof(x)); }

        [[nodiscard]] std::vector<size_t> ToVector() const {
            return {x, x + kDynamicVarCount};
        }

        void Double() {
            for (auto& v: x)
                v *= 2;
        }

        [[nodiscard]] size_t Hash() const {
            size_t hash = 0;
            for (const auto& v: x)
                hash = IterateHash(hash, v);
            return hash;
        }

        [[nodiscard]] bool operator == (const DynamicFills& rhs) const {
            for (int i = 0; i < kDynamicVarCount; ++ i)
                if (x[i] != rhs.x[i])
                    return false;
            return true;
        }

        [[nodiscard]] bool operator != (const DynamicFills& rhs) const {
            return not (*this == rhs);
        }

        friend std::ostream& operator << (std::ostream& os, const DynamicFills& fills);
    };

    /// Variable position indices.
    enum StaticVarPos {
        VG = 0,     // Groups.
        VC = 1,     // Channels.
        VKH = 2,    // Kernel height.
        VKW = 3,    // Kernel width.
        VH = 4,     // Height.
        VW = 5,     // Width.
        VDG = 6,    // Divided by groups (not really exists).
    };

    static const char* var_info[kStaticVarCount];

    int static_power[kStaticVarCount] = {0}, dynamic_power[kDynamicVarCount] = {0};

    Variable() = default;

    Variable(const Variable& rhs) = default;

    explicit Variable(const StaticVarPos& dim) { static_power[dim] = 1; }

    Variable(const std::initializer_list<StaticVarPos>& dims,
             const std::initializer_list<int>& vars={});

    [[nodiscard]] static Variable Static(const StaticVarPos& dim) { return Variable(dim); }

    [[nodiscard]] static Variable Dynamic(int i) {
        assert(0 <= i and i < kDynamicVarCount);
        Variable var;
        var.dynamic_power[i] = 1;
        return var;
    }

    void Reset() { std::memset(this, 0, sizeof(Variable)); }

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
        if (*this == Static(VC))
            return false;
        if (not empty_irregular and Empty())
            return false;
        if (DynamicVarCount() >= 1 and StaticFactor().Empty())
            return false;
        return true;
    }

    /// Solve dynamic variable `i` with `x`.
    void SolveDynamicVar(const VarSolution &s);

    [[nodiscard]] std::vector<int> UnsolvedIndices() {
        std::vector<int> indices;
        for (int i = 0; i < kDynamicVarCount; ++ i)
            if (dynamic_power[i] != 0)
                indices.push_back(i);
        return indices;
    }

    [[nodiscard]] Variable Reciprocal() const {
        Variable reciprocal;
        for (int i = 0; i < kStaticVarCount; ++ i)
            reciprocal.static_power[i] = -static_power[i];
        for (int i = 0; i < kDynamicVarCount; ++ i)
            reciprocal.dynamic_power[i] = -dynamic_power[i];
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

    [[nodiscard]] bool HasNumerator() const {
        auto positive_check = [](int x) -> bool { return x > 0; };
        return std::any_of(static_power, static_power + kStaticVarCount, positive_check) or
               std::any_of(dynamic_power, dynamic_power + kDynamicVarCount, positive_check);
    }

    [[nodiscard]] bool HasDenominator() const {
        auto negative_check = [](int x) -> bool { return x < 0; };
        return std::any_of(static_power, static_power + kStaticVarCount, negative_check) or
               std::any_of(dynamic_power, dynamic_power + kDynamicVarCount, negative_check);
    }

    /// Judge whether a variable is static and an integer: * or, *C*/G, or *C*/R.
    [[nodiscard]] bool IsStaticInteger() const;

    [[nodiscard]] bool MaybeInteger() const {
        // With dynamic variables in the numerator, we can always eliminate the denominator.
        return IsDynamic() or IsStaticInteger();
    }

    [[nodiscard]] bool Empty() const {
        auto func = [](uint8_t x) -> bool { return x == 0; };
        return std::all_of(static_power, static_power + kStaticVarCount, func) and
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
        return true;
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
        return pi;
    }

    Variable& operator *= (const Variable& rhs) {
        *this = *this * rhs;
        return *this;
    }

    [[nodiscard]] Variable operator * (const StaticVarPos& rhs) const {
        return *this * Variable(rhs);
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
        return *this / Variable(rhs);
    }

    Variable& operator /= (const StaticVarPos& rhs) {
        *this = *this / rhs;
        return *this;
    }

    friend Variable operator * (const StaticVarPos& lhs, const StaticVarPos &rhs) {
        return Variable(lhs) * Variable(rhs);
    }

    friend Variable operator / (const StaticVarPos& lhs, const StaticVarPos &rhs) {
        return Variable(lhs) / Variable(rhs);
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

using StaticVar = Variable::StaticVarPos;

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
