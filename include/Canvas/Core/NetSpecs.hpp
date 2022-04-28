#pragma once

#include <iostream>
#include <numeric>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "Canvas/Core/Variable.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

/// Heuristic filled values or strategies
struct HeuristicPreferences {
    static constexpr size_t kDefaultHeuristicGroup = 4;
    static constexpr size_t kDefaultHeuristicReduction = 4;

    size_t g, r;

    explicit HeuristicPreferences(size_t g=kDefaultHeuristicGroup, size_t r=kDefaultHeuristicReduction):
            g(g), r(r) {}

    [[nodiscard]] size_t Hash() const {
        size_t hash = g;
        return IterateHash(hash, r);
    }
};

/// Specifications for a single kernel
struct KernelSpecs {
    size_t ic, oc, k, h, w, s;

    KernelSpecs(size_t ic, size_t oc, size_t k, size_t h, size_t w, size_t s):
            ic(ic), oc(oc), k(k), h(h), w(w), s(s) {}

    [[nodiscard]] size_t ChannelGeometricMean() const {
        return std::ceil(std::sqrt(ic * oc));
    }

    [[nodiscard]] size_t Hash() const {
        size_t hash = ic;
        hash = IterateHash(hash, oc);
        hash = IterateHash(hash, k);
        hash = IterateHash(hash, h);
        hash = IterateHash(hash, w);
        hash = IterateHash(hash, s);
        return hash;
    }
};

/// Specifications for a whole net (i.e. problem settings)
struct NetSpecs {
private:
    bool hash_cached = false;
    size_t hash_value = 0;

    static constexpr double kMinCheckRatio = 0;
    static constexpr double kMaxCheckRatio = 10.0;

public:
    Range<double> flops_ratio_range = {0, 0}, ps_ratio_range = {0, 0};
    Range<size_t> flops_range = {0, 0}, ps_range = {0, 0};

    size_t c_gcd = 0;
    std::vector<size_t> c_gcd_factors;
    bool no_neighbor_involved = false;
    std::vector<KernelSpecs> kernel_specs;
    size_t standard_conv_flops = 0, standard_conv_ps = 0;

    // Convenient for debugging
    explicit NetSpecs(size_t flops_budget=kSizeTUnlimited,
                      size_t ps_budget=kSizeTUnlimited):
            flops_range(0, flops_budget), ps_range(0, ps_budget) {}

    explicit NetSpecs(const std::string& str);

    // NetSpecs(size_t flops_budget, size_t ps_budget, std::vector<KernelSpecs> kernel_specs);

    [[nodiscard]] size_t Hash() {
        if (hash_cached)
            return hash_value;
        hash_cached = true;
        hash_value = 0;
        for (const auto& specs: kernel_specs)
            hash_value = IterateHash(hash_value, specs.Hash());
        return hash_value;
    }

    [[nodiscard]] bool SatisfyFlopsPsRange(size_t flops, size_t ps) const {
        return flops_range.Contains(flops) and ps_range.Contains(ps);
    }

    [[nodiscard]] bool BelowFlopsPsBudget(size_t flops, size_t ps) const {
        return flops <= flops_range.max and ps <= ps_range.max;
    }

    [[nodiscard]] bool Empty() const { return kernel_specs.empty(); }

    void PushKernel(const KernelSpecs& specs) { kernel_specs.push_back(specs); }

    friend std::ostream& operator << (std::ostream& os, const NetSpecs& rhs);
};

/// Dynamic fills for a whole net
struct NetFills {
private:
    bool hash_cached = false;
    size_t hash_value = 0;

    std::vector<Variable::DynamicFills> kernel_fills;

public:
    NetFills() = default;

    [[nodiscard]] size_t Size() const { return kernel_fills.size(); }

    [[nodiscard]] Variable::DynamicFills& At(int i) {
        hash_cached = false;
        return kernel_fills[i];
    }

    size_t Hash() {
        if (hash_cached)
            return hash_value;
        hash_cached = true;
        hash_value = 0;
        for (const auto& fills: kernel_fills)
            hash_value = IterateHash(hash_value, fills.Hash());
        return hash_value;
    }

    void Double() {
        hash_cached = false;
        for (auto& fills: kernel_fills)
            fills.Double();
    }

    void Push(const Variable::DynamicFills& fills) {
        hash_cached = false;
        kernel_fills.push_back(fills);
    }

    friend std::ostream& operator << (std::ostream& os, const NetFills& fills);

    [[nodiscard]] bool operator == (const NetFills& rhs) const {
        if (kernel_fills.size() != rhs.kernel_fills.size())
            return false;
        for (int i = 0; i < kernel_fills.size(); ++ i)
            if (kernel_fills[i] != rhs.kernel_fills[i])
                return false;
        return true;
    }

    [[nodiscard]] bool operator != (const NetFills& rhs) const {
        return not (*this == rhs);
    }
};

static Variable::StaticSpecs MergeIntoStaticSpecs(const HeuristicPreferences& preferences,
                                                  const KernelSpecs& kernel_specs) {
    return {preferences.g, preferences.r, kernel_specs.ic, kernel_specs.oc,
            kernel_specs.k, kernel_specs.h, kernel_specs.w, kernel_specs.s};
}

typedef std::shared_ptr<NetSpecs> NetSpecsSP;
typedef std::shared_ptr<NetFills> NetFillsSP;

} // End namespace canvas
