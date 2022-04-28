#include <sstream>

#include "Canvas/Core/Variable.hpp"


namespace canvas {

static_assert(Variable::kStaticVarCount == 8);
const char* Variable::var_info[kStaticVarCount] = {"G", "A", "C", "KH", "KW", "H", "W", "R"};

Variable::Variable(const std::initializer_list<StaticVarPos>& dims, const std::initializer_list<int>& vars) {
    Reset();
    for (const auto& dim: dims) {
        if (dim == VDG)
            static_power[StaticVarPos::VG] -= 1;
        else if (dim == VDR)
            static_power[StaticVarPos::VR] -= 1;
        else
            static_power[dim] += 1;
    }
    for (const auto& var: vars) {
        assert(var < kDynamicVarCount);
        dynamic_power[var] += 1;
    }
}

std::ostream& operator <<(std::ostream& os, const Variable::DynamicFills& fills) {
    assert(Variable::kDynamicVarCount > 0);
    os << "DynamicFills[" << fills.x[0];
    for (int i = 1; i < Variable::kDynamicVarCount; ++ i)
        os << ", " << fills.x[i];
    return os << "]";
}

bool Variable::IsStaticInteger() const {
    // Without dynamic variables
    for (const auto& var: dynamic_power)
        if (var != 0)
            return false;

    // Divide by C, KH, KW, H or W
    for (int i = 0; i < kStaticVarCount; ++ i) {
        if (i != StaticVarPos::VR and i != StaticVarPos::VG and static_power[i] < 0)
            return false;
    }

    // Each R or G eliminates one C
    return static_power[StaticVarPos::VC] +
           std::min(static_power[StaticVarPos::VG], 0) +
           std::min(static_power[StaticVarPos::VR], 0) >= 0;
}

void Variable::SolveDynamicVar(const VarSolution& solution) {
    auto [index, substitution] = solution;
    if (dynamic_power[index] != 0) {
        assert(substitution.dynamic_power[index] == 0);
        for (int k = 0; k < kStaticVarCount; ++ k)
            static_power[k] += dynamic_power[index] * substitution.static_power[k];
        for (int k = 0; k < kDynamicVarCount; ++ k)
            dynamic_power[k] += dynamic_power[index] * substitution.dynamic_power[k];
        dynamic_power[index] = 0;
    }
}

void Variable::RecursiveGetFactors(int i, Variable &current, std::vector<Variable>& collections,
                                   bool except_hw, int extra_g_factor, int extra_cg_factor) const {
    if (i == kStaticVarCount + kDynamicVarCount) {
        collections.push_back(current);
        collections.back().static_power[StaticVarPos::VG] += extra_g_factor;
        collections.back().static_power[StaticVarPos::VC] += extra_cg_factor;
        collections.back().static_power[StaticVarPos::VG] -= extra_cg_factor;
        return;
    }
    if (i < kStaticVarCount) {
        int power_limit = static_power[i];
        if (except_hw and (i == StaticVarPos::VH or i == StaticVarPos::VW))
            power_limit = 0;
        for (int k = 0; k <= power_limit; ++ k) {
            current.static_power[i] = k;
            int max_extra_g_or_cg_factor = 0;
            if (i == StaticVarPos::VC)
                max_extra_g_or_cg_factor = static_power[i] - k;
            for (int j = 0; j <= max_extra_g_or_cg_factor; ++ j)
                RecursiveGetFactors(i + 1, current, collections, except_hw, extra_g_factor + j, extra_cg_factor);
            for (int j = 0; j <= max_extra_g_or_cg_factor; ++ j)
                RecursiveGetFactors(i + 1, current, collections, except_hw, extra_g_factor, extra_cg_factor + j);
        }
    } else {
        for (int k = 0; k <= dynamic_power[i - kStaticVarCount]; ++ k) {
            current.dynamic_power[i - kStaticVarCount] = k;
            RecursiveGetFactors(i + 1, current, collections, except_hw, extra_g_factor, extra_cg_factor);
        }
    }
}

std::vector<Variable> Variable::GetAllFactors(bool except_hw) const {
    Variable current;
    std::vector<Variable> collections;
    RecursiveGetFactors(0, current, collections, except_hw);
    return UniqueByHash(collections);
}

std::string Variable::Format(const char** info, const std::string& mul, const std::string& div,
                             const std::string& x_prefix, const std::string& x_suffix) const {
    std::stringstream ss;
    bool has_numerator = HasNumerator();
    bool has_denominator = HasDenominator();

    // Empty variable
    if (not has_numerator and not has_denominator)
        return "1";

    // Print numerator
    if (has_numerator) {
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
    } else {
        ss << "1";
    }

    // Print denominator
    for (int i = 0; i < Variable::kStaticVarCount; ++ i)
        for (int j = 0; j < -static_power[i]; ++ j)
            ss << div << info[i];
    for (int i = 0; i < Variable::kDynamicVarCount; ++ i)
        for (int j = 0; j < -dynamic_power[i]; ++ j)
            ss << div << x_prefix << i << x_suffix;
    return ss.str();
}

size_t Variable::FillToInteger(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    auto Product = [=](bool positive) -> size_t {
        size_t result = 1;
        result *= (static_power[VG] >= 0) == positive ? Power(specs.g, std::abs(static_power[VG])) : 1;
        result *= (static_power[VR] >= 0) == positive ? Power(specs.r, std::abs(static_power[VR])) : 1;
        result *= (static_power[VA] >= 0) == positive ? Power(specs.a, std::abs(static_power[VA])) : 1;
        result *= (static_power[VC] >= 0) == positive ? Power(specs.c, std::abs(static_power[VC])) : 1;
        result *= (static_power[VKH] >= 0) == positive ? Power(specs.k, std::abs(static_power[VKH])) : 1;
        result *= (static_power[VKW] >= 0) == positive ? Power(specs.k, std::abs(static_power[VKW])) : 1;
        result *= (static_power[VH] >= 0) == positive ? Power(specs.h / specs.s, std::abs(static_power[VH])) : 1;
        result *= (static_power[VW] >= 0) == positive ? Power(specs.w / specs.s, std::abs(static_power[VW])) : 1;
        for (int i = 0; i < Variable::kDynamicVarCount; ++ i)
            result *= (dynamic_power[i] >= 0) == positive ? Power(fills.x[i], std::abs(dynamic_power[i])) : 1;
        return result;
    };
    size_t numerator = Product(true);
    size_t denominator = Product(false);
    if (denominator == 0 or numerator % denominator != 0)
        return 0;
    return numerator / denominator;
}

void Variable::UpdateMinimumFills(Variable::DynamicFills& fills, const Variable::StaticSpecs& specs) const {
    assert(SatisfyAssumption());
    auto numerator = Numerator(), denominator = Denominator();
    if (numerator.IsStatic())
        return;
    assert(numerator.DynamicVarCount() == 1);
    auto p = numerator.StaticFactor().FillToInteger(specs);
    auto q = denominator.FillToInteger(specs);
    assert(p > 0 and q > 0);
    // v = p * x / q, then min_value = q / gcd(p, q)
    auto min_value = q / std::gcd(p, q);
    int i = numerator.GetOnlyDynamicVar();
    fills.x[i] = std::lcm(fills.x[i], min_value);
}

} // End namespace canvas
