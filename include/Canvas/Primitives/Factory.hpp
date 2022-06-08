#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <vector>
#include <unordered_set>

#include "Canvas/Primitives/Activation.hpp"
#include "Canvas/Primitives/Broadcast.hpp"
#include "Canvas/Primitives/ChannelShuffle.hpp"
#include "Canvas/Primitives/Dropout.hpp"
#include "Canvas/Primitives/ElementWise.hpp"
#include "Canvas/Primitives/FC.hpp"
#include "Canvas/Primitives/Fold.hpp"
#include "Canvas/Primitives/Group.hpp"
#include "Canvas/Primitives/Input.hpp"
#include "Canvas/Primitives/Norm.hpp"
#include "Canvas/Primitives/Output.hpp"
#include "Canvas/Primitives/Pool.hpp"
#include "Canvas/Primitives/Shift.hpp"
#include "Canvas/Primitives/Softmax.hpp"
#include "Canvas/Primitives/Transpose.hpp"
#include "Canvas/Primitives/Unfold.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

struct Graph;
typedef std::shared_ptr<Graph> GraphSP;

struct PrimitiveFilter {
    /// Must be output.
    bool output_filter = false;

    /// Hash filters.
    std::unordered_set<size_t> hash_filter;

    /// Filters.
    std::vector<std::string> allowed_filter, forbidden_filter;

    /// `max_delta_width` could be -1 (reducing width), 0 (retaining width), 1 (unlimited).
    int max_delta_width = 1;

    // Optimize FC.
    bool add_relu_bn_after_fc = false;

    explicit PrimitiveFilter(const std::string& allowed_str="",
                             const std::string& forbidden_str="",
                             bool add_relu_bn_after_fc=false):
            add_relu_bn_after_fc(add_relu_bn_after_fc) {
        BuildFilters(allowed_str, forbidden_str);
    }

    explicit PrimitiveFilter(int max_delta_width): max_delta_width(max_delta_width) {}

    void BuildFilters(const std::string& allowed_str="", const std::string& forbidden_str="");

    [[nodiscard]] bool Filter(const PrimitiveSP& p) const;
};

/// Register all primitive constructions here.
struct PrimitiveFactory {
    static std::vector<PrimitiveApply> RescalePossibilities(const std::vector<PrimitiveApply>& applies);

    /// Get all primitives for a graph.
    static std::vector<PrimitiveApply> GetPrimitiveApplies(const GraphSP& graph,
                                                           const PrimitiveFilter& filter=PrimitiveFilter());

    /// Get primitives with one input.
    static void GetPrimitiveApplies(const GraphSP &graph,
                                    std::vector<PrimitiveApply>& primitives,
                                    const TensorSP& t,
                                    const PrimitiveFilter& filter);

    /// Get primitives with two inputs.
    static void GetPrimitiveApplies(const GraphSP &graph,
                                    std::vector<PrimitiveApply>& primitives,
                                    const TensorSP& lhs,
                                    const TensorSP& rhs,
                                    const PrimitiveFilter& filter);
};

static void TryPush(const PrimitiveApply& pa,
                    std::vector<PrimitiveApply>& vec,
                    const PrimitiveFilter& filter) {
    if (not filter.Filter(pa.primitive))
        vec.push_back(pa);
}

template <typename PrimitiveType, class ... Args>
static void TryMakeAndPush(std::vector<PrimitiveApply>& vec,
                           const PrimitiveFilter& filter,
                           Args&&... args) {
    try {
        auto p = std::make_shared<PrimitiveType>(args...);
        if (not filter.Filter(p))
            vec.push_back(PrimitiveApply(p));
    } catch (CanNotApplyPrimitive& e) {}
}

} // namespace canvas
