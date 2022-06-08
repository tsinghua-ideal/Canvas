#include <functional>
#include <sstream>

#include "Canvas/Core/Variable.hpp"


namespace canvas {

static_assert(Variable::kStaticVarCount == 4);
const char* Variable::var_info[kStaticVarCount] = {"G", "C", "H", "W"};

Variable Variable::Compose(const std::initializer_list<StaticVarPos>& dims,
                           int numeric_numerator, int numeric_denominator,
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
    int gcd = std::gcd(numeric_numerator, numeric_denominator);
    var.numeric_numerator = numeric_numerator / gcd;
    var.numeric_denominator = numeric_denominator / gcd;
    return var;
}

int Variable::FillToInteger(const Variable::VarSpecs& specs) const {
    assert(IsStatic());
    auto Product = [=](bool positive) -> int {
        int result = positive ? numeric_numerator : numeric_denominator;
        result *= (static_power[VG] >= 0) == positive ? Power(specs.g, std::abs(static_power[VG])) : 1;
        result *= (static_power[VC] >= 0) == positive ? Power(specs.c, std::abs(static_power[VC])) : 1;
        result *= (static_power[VH] >= 0) == positive ? Power(specs.h, std::abs(static_power[VH])) : 1;
        result *= (static_power[VW] >= 0) == positive ? Power(specs.w, std::abs(static_power[VW])) : 1;
        return result;
    };
    int numerator = Product(true), denominator = Product(false);
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
        // TODO: extract as a function.
        int gcd = std::gcd(numeric_numerator, numeric_denominator);
        numeric_numerator /= gcd, numeric_denominator /= gcd;
        dynamic_power[index] = 0;
    }
}

std::vector<Variable> Variable::GetAllFactors() const {
    std::vector<std::pair<Variable, int>> primes;
    auto PushPrime = [&primes](Variable var, int power) {
        if (power == 0)
            return;
        if (power < 0)
            var = var.Reciprocal(), power = -power;
        primes.emplace_back(std::make_pair(var, power));
    };

    // Push prime of numbers into vector.
    auto DecomposeNumber = [&](int x, bool is_numerator) {
        for (int i = 2; i <= x; ++ i) {
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

} // namespace canvas
