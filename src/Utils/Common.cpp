#include "Canvas/Utils/Common.hpp"


namespace canvas {

RandomGen<int> global_int_random(0, std::numeric_limits<int>::max(), false, kDefaultGlobalRandomSeed); // NOLINT(cert-err58-cpp)
RandomGen<double> global_uniform_random(0, 1, false, kDefaultGlobalRandomSeed + 2); // NOLINT(cert-err58-cpp)

void ResetRandomSeed(bool pure, uint32_t seed) {
    global_int_random = RandomGen<int>(0, std::numeric_limits<int>::max(),
                                       pure, seed);
    global_uniform_random = RandomGen<double>(0, 1, pure, seed + 2);
}

} // namespace canvas
