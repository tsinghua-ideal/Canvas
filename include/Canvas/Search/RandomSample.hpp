#include "Canvas/Core/NetSpecs.hpp"
#include "Canvas/Core/Solution.hpp"


namespace canvas {

Solution RandomSample(const NetSpecsSP& net_specs,
                      bool allow_dynamic,
                      bool force_irregular,
                      bool add_relu_bn_after_fc,
                      const Range<int>& np_range,
                      const Range<int>& fc_range,
                      canvas_timeval_t timeout);

}
