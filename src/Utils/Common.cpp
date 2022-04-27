#include "Canvas/Utils/Common.hpp"

namespace canvas {

[[maybe_unused]] Random<int> global_int_random(0, kIntUnlimited, false, kDefaultGlobalRandomSeed);
[[maybe_unused]] Random<uint64_t> global_uint64_random(0, kUInt64Unlimited, false, kDefaultGlobalRandomSeed + 1);
[[maybe_unused]] Random<double> global_norm_uniform_random(0, 1, false, kDefaultGlobalRandomSeed + 2);

void InitRandomEngine(bool pure, uint32_t seed) {
    global_int_random = Random<int>(0, kIntUnlimited, pure, seed);
    global_uint64_random = Random<uint64_t>(0, kUInt64Unlimited, pure, seed + 1);
    global_norm_uniform_random = Random<double>(0, 1, pure, seed + 2);
}

} // End namespace canvas
