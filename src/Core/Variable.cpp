#include <sstream>

#include "Canvas/Core/Variable.hpp"


namespace canvas {

static_assert(Variable::kStaticVarCount == 6);
const char* Variable::var_info[kStaticVarCount] = {"G", "C", "KH", "KW", "H", "W"};

Variable::Variable(const std::initializer_list<StaticVarPos>& dims, const std::initializer_list<int>& vars) {
    Reset();
    for (const auto& dim: dims) {
        if (dim == VDG)
            static_power[StaticVarPos::VG] -= 1;
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
    // Without dynamic variables.
    for (const auto& var: dynamic_power)
        if (var != 0)
            return false;

    // Divide by C, KH, KW, H or W.
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
        dynamic_power[index] = 0;
    }
}

void Variable::RecursiveGetFactors(int i, Variable &current, std::vector<Variable>& collections, // NOLINT(misc-no-recursion)
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

    // Empty variable.
    if (not has_numerator and not has_denominator)
        return "1";

    // Print numerator.
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

    // Print denominator.
    for (int i = 0; i < Variable::kStaticVarCount; ++ i)
        for (int j = 0; j < -static_power[i]; ++ j)
            ss << div << info[i];
    for (int i = 0; i < Variable::kDynamicVarCount; ++ i)
        for (int j = 0; j < -dynamic_power[i]; ++ j)
            ss << div << x_prefix << i << x_suffix;
    return ss.str();
}

} // namespace canvas
