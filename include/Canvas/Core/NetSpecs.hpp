#pragma once

#include <cmath>

#include "Canvas/Utils/Common.hpp"


namespace canvas {

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

} // End namespace canvas
