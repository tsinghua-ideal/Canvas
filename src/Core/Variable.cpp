#include <functional>
#include <sstream>

#include "Canvas/Core/Variable.hpp"


namespace canvas {

static_assert(Variable::kStaticVarCount == 4);
const char* Variable::var_info[kStaticVarCount] = {"G", "C", "H", "W"};

Variable Variable::Compose(const std::initializer_list<StaticVarPos>& dims,
                           size_t numeric_numerator, size_t numeric_denominator,
                           const std::initializer_list<int>& dyn_vars) {
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
    var.numeric_numerator = numeric_numerator;
    var.numeric_denominator = numeric_denominator;
    var.Simplify();
    return var;
}

size_t Variable::FillToInteger(const Variable::VarSpecs& specs) const {
    assert(IsStatic());
    auto Product = [=](bool positive) -> size_t {
        size_t result = positive ? numeric_numerator : numeric_denominator;
        result *= (static_power[VG] >= 0) == positive ? Power(specs.g, std::abs(static_power[VG])) : 1;
        result *= (static_power[VC] >= 0) == positive ? Power(specs.c, std::abs(static_power[VC])) : 1;
        result *= (static_power[VH] >= 0) == positive ? Power(specs.h, std::abs(static_power[VH])) : 1;
        result *= (static_power[VW] >= 0) == positive ? Power(specs.w, std::abs(static_power[VW])) : 1;
        return result;
    };
    size_t numerator = Product(true), denominator = Product(false);
    if (denominator == 0 or numerator % denominator != 0)
        return 0;
    return numerator / denominator;
}

bool Variable::MaybeInteger() const {
    // With dynamic variables in the numerator, we can always eliminate the denominator.
    for (const auto& var: dynamic_power)
        if (var != 0)
            return true;

    // Pure number.
    assert(numeric_numerator != 0);
    if (IsNumber())
        return numeric_numerator % numeric_denominator == 0;

    // Divide by C, H or W.
    for (int i = 0; i < kStaticVarCount; ++ i) {
        if (i != StaticVarPos::VG and static_power[i] < 0)
            return false;
    }

    // Each G eliminates one C.
    return static_power[StaticVarPos::VC] + static_power[StaticVarPos::VG] >= 0;
}

void Variable::SolveDynamicVar(const VarSolution& solution) {
    auto [index, substitution] = solution;
    if (dynamic_power[index] != 0) {
        assert(substitution.dynamic_power[index] == 0);
        for (int k = 0; k < kStaticVarCount; ++ k)
            static_power[k] += dynamic_power[index] * substitution.static_power[k];
        for (int k = 0; k < kDynamicVarCount; ++ k)
            dynamic_power[k] += dynamic_power[index] * substitution.dynamic_power[k];
        if (dynamic_power[index] > 0) {
            numeric_numerator *= Power(substitution.numeric_numerator, dynamic_power[index]);
            numeric_denominator *= Power(substitution.numeric_denominator, dynamic_power[index]);
        } else {
            numeric_numerator *= Power(substitution.numeric_denominator, -dynamic_power[index]);
            numeric_denominator *= Power(substitution.numeric_numerator, -dynamic_power[index]);
        }
        Simplify();
        dynamic_power[index] = 0;
    }
}

Variable Variable::operator * (const Variable& rhs) const {
    Variable pi;
    for (int i = 0; i < kStaticVarCount; ++ i)
        pi.static_power[i] = static_power[i] + rhs.static_power[i];
    for (int i = 0; i < kDynamicVarCount; ++ i)
        pi.dynamic_power[i] = dynamic_power[i] + rhs.dynamic_power[i];
    pi.numeric_numerator = numeric_numerator * rhs.numeric_numerator;
    pi.numeric_denominator = numeric_denominator * rhs.numeric_denominator;
    pi.Simplify();
    return pi;
}

Variable Variable::Reciprocal() const {
    Variable reciprocal;
    for (int i = 0; i < kStaticVarCount; ++ i)
        reciprocal.static_power[i] = -static_power[i];
    for (int i = 0; i < kDynamicVarCount; ++ i)
        reciprocal.dynamic_power[i] = -dynamic_power[i];
    reciprocal.numeric_numerator = numeric_denominator;
    reciprocal.numeric_denominator = numeric_numerator;
    return reciprocal;
}

int Variable::GetFirstLinearDynamicVar() const {
    for (int i = 0; i < kDynamicVarCount; ++ i)
        if (dynamic_power[i] == 1)
            return i;
    return kInvalidIndex;
}

int Variable::GetOnlyDynamicVar() const {
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

int Variable::DynamicVarCount() const {
    int count = 0;
    for (const auto& var: dynamic_power)
        count += (var != 0);
    return count;
}

Variable Variable::Numerator() const {
    Variable numerator;
    for (int i = 0; i < kStaticVarCount; ++ i)
        numerator.static_power[i] = std::max(static_power[i], 0);
    for (int i = 0; i < kDynamicVarCount; ++ i)
        numerator.dynamic_power[i] = std::max(dynamic_power[i], 0);
    numerator.numeric_numerator = numeric_numerator;
    return numerator;
}

Variable Variable::Denominator() const {
    Variable denominator;
    for (int i = 0; i < kStaticVarCount; ++ i)
        denominator.static_power[i] = std::abs(std::min(static_power[i], 0));
    for (int i = 0; i < kDynamicVarCount; ++ i)
        denominator.dynamic_power[i] = std::abs(std::min(dynamic_power[i], 0));
    denominator.numeric_numerator = numeric_denominator;
    return denominator;
}

bool Variable::operator == (const Variable& rhs) const {
    for (int i = 0; i < kStaticVarCount; ++ i)
        if (static_power[i] != rhs.static_power[i])
            return false;
    for (int i = 0; i < kDynamicVarCount; ++ i)
        if (dynamic_power[i] != rhs.dynamic_power[i])
            return false;
    return numeric_numerator == rhs.numeric_numerator and numeric_denominator == rhs.numeric_denominator;
}

size_t Variable::Hash() const {
    size_t value = numeric_numerator;
    value = IterateHash(value, numeric_denominator);
    for (const auto& power: static_power)
        value = IterateHash(value, power);
    for (const auto& power: dynamic_power)
        value = IterateHash(value, power);
    return value;
}

std::vector<Variable> Variable::GetAllFactors() const {
    std::vector<std::pair<Variable, size_t>> primes;
    auto PushPrime = [&primes](Variable var, int power) {
        if (power == 0)
            return;
        if (power < 0)
            var = var.Reciprocal(), power = -power;
        primes.emplace_back(std::make_pair(var, power));
    };

    // Push prime of numbers into vector.
    auto DecomposeNumber = [&](size_t x, bool is_numerator) {
        for (size_t i = 2; i <= x; ++ i) {
            int count = 0;
            while (x % i == 0)
                x /= i, ++ count;
            if (not is_numerator)
                count = -count;
            PushPrime(Variable::Number(i), count);
        }
    };

    // Push prime of variables into vector.
    auto DecomposeStaticVariable = [&](int pos) {
        if (static_power[pos] == 0)
            return;
        if (pos == StaticVarPos::VC) {
            PushPrime(Variable::StaticVar(StaticVarPos::VG), static_power[pos]);
            PushPrime(Variable::Compose({StaticVarPos::VC, StaticVarPos::VDG}), static_power[pos]);
        } else {
            PushPrime(Variable::StaticVar(StaticVarPos(pos)), static_power[pos]);
        }
    };

    // Collect them into the prime vector.
    DecomposeNumber(numeric_numerator, true);
    DecomposeNumber(numeric_denominator, false);
    for (int i = 0; i < kStaticVarCount; ++ i)
        DecomposeStaticVariable(i);
    for (int i = 0; i < kDynamicVarCount; ++ i) {
        assert(dynamic_power[i] >= 0);
        PushPrime(Variable::DynamicVar(i), dynamic_power[i]);
    }

    // Check possible size.
    size_t total_candidates = 1;
    for (const auto& p: primes)
        total_candidates *= p.second + 1;
    if (total_candidates > kFactorThreshold) {
        std::stringstream ss;
        ss << "~" << total_candidates << " factors found for variable " << *this;
        Warning(ss.str());
    }

    // Enumerate all possible combination of primes.
    std::vector<Variable> collections;
    std::function<void(int, Variable)> EnumerateAll;
    EnumerateAll = [&](int pos, Variable pi) {
        if (pos == primes.size()) {
            collections.push_back(pi);
            return;
        }
        for (int i = 0; i <= primes[pos].second; ++ i) {
            EnumerateAll(pos + 1, pi);
            pi = pi * primes[pos].first;
        }
    };
    EnumerateAll(0, Variable());

    // Return.
    return UniqueByHash(collections);
}

std::string Variable::Format(const char** info, const std::string& mul, const std::string& div,
                             const std::string& x_prefix, const std::string& x_suffix) const {
    assert(numeric_numerator > 0 and numeric_denominator > 0);
    std::stringstream ss;

    // Print variable numerator.
    bool displayed = false;
    for (int i = 0; i < Variable::kStaticVarCount; ++ i) {
        if (static_power[i] > 0) {
            for (int j = 0; j < static_power[i]; ++ j)
                ss << (displayed ? mul : "") << info[i], displayed = true;
        }
    }
    for (int i = 0; i < Variable::kDynamicVarCount; ++ i) {
        if (dynamic_power[i] > 0) {
            for (int j = 0; j < dynamic_power[i]; ++ j)
                ss << (displayed ? mul : "") << x_prefix << i << x_suffix, displayed = true;
        }
    }
    if (not displayed)
        ss << numeric_numerator;
    else if (numeric_numerator != 1)
        ss << mul << numeric_numerator;

    // Print denominator.
    if (numeric_denominator != 1)
        ss << div << numeric_denominator;
    for (int i = 0; i < Variable::kStaticVarCount; ++ i)
        for (int j = 0; j < -static_power[i]; ++ j)
            ss << div << info[i];
    for (int i = 0; i < Variable::kDynamicVarCount; ++ i)
        for (int j = 0; j < -dynamic_power[i]; ++ j)
            ss << div << x_prefix << i << x_suffix;
    return ss.str();
}

Variable Variable::Lcm(const Variable& lhs, const Variable& rhs) {
    assert(lhs.Denominator().Empty() and rhs.Denominator().Empty());
    Variable lcm;
    lcm.numeric_numerator = std::lcm(lhs.numeric_numerator, rhs.numeric_numerator);
    assert(lhs.numeric_denominator == 1 and rhs.numeric_denominator == 1);
    for (int i = 0; i < kStaticVarCount; ++ i) {
        assert(lhs.static_power[i] >= 0 and rhs.static_power[i] >= 0);
        lcm.static_power[i] = std::max(lhs.static_power[i], rhs.static_power[i]);
    }
    for (int i = 0; i < kDynamicVarCount; ++ i) {
        assert(lhs.dynamic_power[i] >= 0 and rhs.dynamic_power[i] >= 0);
        lcm.dynamic_power[i] = std::max(lhs.dynamic_power[i], rhs.dynamic_power[i]);
    }
    return lcm;
}

Variable Variable::Gcd(const Variable& lhs, const Variable& rhs) {
    assert(lhs.Denominator().Empty() and rhs.Denominator().Empty());
    Variable lcm;
    lcm.numeric_numerator = std::gcd(lhs.numeric_numerator, rhs.numeric_numerator);
    assert(lhs.numeric_denominator == 1 and rhs.numeric_denominator == 1);
    for (int i = 0; i < kStaticVarCount; ++ i) {
        assert(lhs.static_power[i] >= 0 and rhs.static_power[i] >= 0);
        lcm.static_power[i] = std::min(lhs.static_power[i], rhs.static_power[i]);
    }
    for (int i = 0; i < kDynamicVarCount; ++ i) {
        assert(lhs.dynamic_power[i] >= 0 and rhs.dynamic_power[i] >= 0);
        lcm.dynamic_power[i] = std::min(lhs.dynamic_power[i], rhs.dynamic_power[i]);
    }
    return lcm;
}

} // namespace canvas
