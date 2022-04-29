#include <ice-cream.hpp>

#include "Canvas/Search/RandomSample.hpp"


namespace canvas {

Solution RandomSample(const NetSpecsSP& net_specs, bool allow_dynamic, bool force_irregular, bool add_relu_bn_after_fc,
                      const Range<int>& np_range, const Range<int>& fc_range, canvas_timeval_t timeout) {
    IC(net_specs->flops_ratio_range);
    IC(net_specs->ps_ratio_range);
    IC(allow_dynamic, force_irregular, add_relu_bn_after_fc);
    IC(np_range, fc_range);
    IC(timeout);
    Unimplemented();
}

}
