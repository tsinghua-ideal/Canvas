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

/// Global settings for a kernel design.
struct GlobalSpecs {
    size_t g;

    explicit GlobalSpecs(size_t g=1): g(g) {}

    GlobalSpecs(const GlobalSpecs& specs) = default;

    [[nodiscard]] size_t Hash() const {
        return g;
    }
};

/// Specifications for a single kernel.
struct KernelSpecs {
    size_t c, h, w, spatial_dims;

    KernelSpecs(size_t c, size_t h, size_t w, size_t spatial_dims=3): c(c), h(h), w(w), spatial_dims(spatial_dims) {
        if (spatial_dims < 1)
            assert(h == 1);
        if (spatial_dims < 2)
            assert(w == 1);
    }

    [[nodiscard]] size_t Hash() const {
        size_t hash = c;
        if (spatial_dims >= 1)
            hash = IterateHash(hash, h);
        if (spatial_dims >= 2)
            hash = IterateHash(hash, w);
        return hash;
    }
};

/// Specifications for a whole net (i.e. problem settings).
struct NetSpecs {
private:
    bool hash_cached = false;
    size_t hash_value = 0;

public:
    size_t c_gcd = 0;
    std::vector<size_t> c_gcd_factors;
    std::vector<KernelSpecs> kernel_specs;

    NetSpecs() = default;

    explicit NetSpecs(std::vector<KernelSpecs> kernel_specs);

    [[nodiscard]] size_t Hash() {
        if (hash_cached)
            return hash_value;
        hash_cached = true;
        hash_value = 0;
        for (const auto& specs: kernel_specs)
            hash_value = IterateHash(hash_value, specs.Hash());
        return hash_value;
    }

    [[nodiscard]] bool Empty() const { return kernel_specs.empty(); }

    friend std::ostream& operator << (std::ostream& os, const NetSpecs& rhs);
};

typedef std::shared_ptr<NetSpecs> NetSpecsSP;

static Variable::VarSpecs Merge(const GlobalSpecs& global_specs, const KernelSpecs& kernel_specs) {
    return {global_specs.g, kernel_specs.c, kernel_specs.h, kernel_specs.w};
}

} // namespace canvas
