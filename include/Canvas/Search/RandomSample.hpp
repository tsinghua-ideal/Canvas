#include "Canvas/Core/Specs.hpp"
#include "Canvas/Core/Solution.hpp"


namespace canvas {

static const Range<int> mw_range = {1, 8}; // NOLINT(cert-err58-cpp)

Solution TryRandomSample(const NetSpecsSP& net_specs,
                         bool allow_dynamic,
                         bool add_relu_bn_after_fc,
                         const Range<int>& np_range,
                         const Range<int>& fc_range);

Solution RandomSample(const NetSpecsSP& net_specs,
                      bool allow_dynamic,
                      bool add_relu_bn_after_fc,
                      const Range<int>& np_range,
                      const Range<int>& fc_range,
                      canvas_timeval_t timeout);

}
