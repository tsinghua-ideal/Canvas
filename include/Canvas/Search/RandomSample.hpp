#include <utility>

#include "Canvas/Core/Specs.hpp"
#include "Canvas/Core/Solution.hpp"


namespace canvas {

struct SampleOptions {
    // Temporarily not visible to users.
    static constexpr int kMaxGroupFactor = 10;

    const std::string allowed_filter, forbidden_filter;

    const std::vector<int> kernel_sizes, dilated_sizes;

    const bool add_relu_bn_after_fc;
    const Range<int> np_range, mw_range, fc_range;

    const canvas_timeval_t timeout;

    SampleOptions():
            kernel_sizes({3, 5, 7}), dilated_sizes({1, 2, 3}),
            add_relu_bn_after_fc(false),
            np_range(3, 25), mw_range(2, 8), fc_range(1, 8),
            timeout(std::chrono::seconds::zero()) {}

    SampleOptions(std::string allowed_filter,
                  std::string forbidden_filter,
                  std::vector<int> kernel_sizes,
                  std::vector<int> dilated_sizes,
                  bool add_relu_bn_after_fc,
                  int np_range_min, int np_range_max,
                  int mw_range_min, int mw_range_max,
                  int fc_range_min, int fc_range_max,
                  int timeout):
        allowed_filter(std::move(allowed_filter)),
        forbidden_filter(std::move(forbidden_filter)),
        kernel_sizes(std::move(kernel_sizes)),
        dilated_sizes(std::move(dilated_sizes)),
        add_relu_bn_after_fc(add_relu_bn_after_fc),
        np_range(np_range_min, np_range_max),
        mw_range(mw_range_min, mw_range_max),
        fc_range(fc_range_min, fc_range_max),
        timeout(std::chrono::seconds(timeout)) {
        for (int k: this->kernel_sizes)
            assert(k > 0 and k % 2 == 1);
        for (int d: this->dilated_sizes)
            assert(d > 0);
        assert(timeout > 0);
    }
};

Solution TryRandomSample(const NetSpecsSP& net_specs, const SampleOptions& options=SampleOptions());

Solution RandomSample(const NetSpecsSP& net_specs, const SampleOptions& options=SampleOptions());

}
