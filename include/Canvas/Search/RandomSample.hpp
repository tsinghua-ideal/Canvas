#include <utility>

#include "Canvas/Core/Specs.hpp"
#include "Canvas/Core/Solution.hpp"


namespace canvas {

struct SampleOptions {
    // Temporarily not visible to users.
    static constexpr int kMaxGroupFactor = 10;

    const std::string allowed_filter, forbidden_filter;

    const bool add_relu_bn_after_fc;
    const Range<int> np_range, mw_range, fc_range;

    const canvas_timeval_t timeout;

    SampleOptions():
            add_relu_bn_after_fc(false),
            np_range(3, 25), mw_range(2, 8), fc_range(1, 8),
            timeout(std::chrono::seconds::zero()) {}

    SampleOptions(std::string allowed_filter,
                  std::string forbidden_filter,
                  bool add_relu_bn_after_fc,
                  int np_range_min, int np_range_max,
                  int mw_range_min, int mw_range_max,
                  int fc_range_min, int fc_range_max,
                  int timeout):
        allowed_filter(std::move(allowed_filter)),
        forbidden_filter(std::move(forbidden_filter)),
        add_relu_bn_after_fc(add_relu_bn_after_fc),
        np_range(np_range_min, np_range_max),
        mw_range(mw_range_min, mw_range_max),
        fc_range(fc_range_min, fc_range_max),
        timeout(std::chrono::seconds(timeout)) {}
};

Solution TryRandomSample(const NetSpecsSP& net_specs, const SampleOptions& options=SampleOptions());

Solution RandomSample(const NetSpecsSP& net_specs, const SampleOptions& options=SampleOptions());

}
