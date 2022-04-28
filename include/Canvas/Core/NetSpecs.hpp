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
    std::vector<KernelSpecs> layer_specs;
    size_t standard_conv_flops = 0, standard_conv_ps = 0;

    // Convenient for debugging
    explicit NetSpecs(size_t flops_budget=kSizeTUnlimited,
                      size_t ps_budget=kSizeTUnlimited):
            flops_range(0, flops_budget), ps_range(0, ps_budget) {}

    explicit NetSpecs(const std::string& str);

    [[nodiscard]] size_t Hash() {
        if (hash_cached)
            return hash_value;
        hash_cached = true;
        hash_value = 0;
        for (const auto& specs: layer_specs)
            hash_value = IterateHash(hash_value, specs.Hash());
        return hash_value;
    }

    [[nodiscard]] bool SatisfyFlopsPsRange(size_t flops, size_t ps) const {
        return flops_range.Contains(flops) and ps_range.Contains(ps);
    }

    [[nodiscard]] bool BelowFlopsPsBudget(size_t flops, size_t ps) const {
        return flops <= flops_range.max and ps <= ps_range.max;
    }

    [[nodiscard]] bool Empty() const { return layer_specs.empty(); }

    void PushLayer(const KernelSpecs& specs) { layer_specs.push_back(specs); }

    friend std::ostream& operator << (std::ostream& os, const NetSpecs& rhs);
};

/// Dynamic fills for a whole net
struct NetFills {
private:
    bool hash_cached = false;
    size_t hash_value = 0;

    std::vector<Variable::DynamicFills> layer_fills;

public:
    NetFills() = default;

    [[nodiscard]] size_t Size() const { return layer_fills.size(); }

    [[nodiscard]] Variable::DynamicFills& At(int i) {
        hash_cached = false;
        return layer_fills[i];
    }

    size_t Hash() {
        if (hash_cached)
            return hash_value;
        hash_cached = true;
        hash_value = 0;
        for (const auto& fills: layer_fills)
            hash_value = IterateHash(hash_value, fills.Hash());
        return hash_value;
    }

    void Double() {
        hash_cached = false;
        for (auto& fills: layer_fills)
            fills.Double();
    }

    void Push(const Variable::DynamicFills& fills) {
        hash_cached = false;
        layer_fills.push_back(fills);
    }

    friend std::ostream& operator << (std::ostream& os, const NetFills& fills);

    [[nodiscard]] bool operator == (const NetFills& rhs) const {
        if (layer_fills.size() != rhs.layer_fills.size())
            return false;
        for (int i = 0; i < layer_fills.size(); ++ i)
            if (layer_fills[i] != rhs.layer_fills[i])
                return false;
        return true;
    }

    [[nodiscard]] bool operator != (const NetFills& rhs) const {
        return not (*this == rhs);
    }
};

static Variable::StaticSpecs MergeIntoStaticSpecs(const HeuristicPreferences& preferences,
                                                  const KernelSpecs& layer_specs) {
    return {preferences.g, preferences.r, layer_specs.ic, layer_specs.oc,
            layer_specs.k, layer_specs.h, layer_specs.w, layer_specs.s};
}

typedef std::shared_ptr<NetSpecs> NetSpecsSP;
typedef std::shared_ptr<NetFills> NetFillsSP;

} // End namespace canvas
