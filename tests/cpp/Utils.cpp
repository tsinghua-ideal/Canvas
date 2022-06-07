#include <gtest/gtest.h>
#include <vector>

#include "Canvas/Utils/Common.hpp"


using namespace canvas;

TEST(Utils, GlobalRandomEngine) {
    std::vector<int> vec = {0, 1, 2, 3, 4};
    int v0 = RandomInt();
    int v1 = RandomChoose(vec);
    RandomShuffle(vec);

    ResetRandomSeed(false);
    std::vector<int> r_vec = {0, 1, 2, 3, 4};
    int r_v0 = RandomInt();
    int r_v1 = RandomChoose(r_vec);
    RandomShuffle(r_vec);
    ASSERT_EQ(r_v0, v0);
    ASSERT_EQ(r_v1, v1);
    ASSERT_EQ(r_vec, vec);
}
