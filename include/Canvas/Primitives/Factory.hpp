#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <utility>
#include <vector>
#include <unordered_set>

#include "Canvas/Primitives/Activation.hpp"
#include "Canvas/Primitives/Broadcast.hpp"
#include "Canvas/Primitives/Convolution.hpp"
#include "Canvas/Primitives/ElementWise.hpp"
#include "Canvas/Primitives/FC.hpp"
#include "Canvas/Primitives/Fold.hpp"
#include "Canvas/Primitives/Group.hpp"
#include "Canvas/Primitives/Input.hpp"
#include "Canvas/Primitives/MatrixMultiplication.hpp"
#include "Canvas/Primitives/Output.hpp"
#include "Canvas/Primitives/Scale.hpp"
#include "Canvas/Primitives/Shift.hpp"
#include "Canvas/Primitives/Softmax.hpp"
#include "Canvas/Primitives/Unfold.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

struct Graph;
typedef std::shared_ptr<Graph> GraphSP;

struct PrimitiveOptions {
    static constexpr int kMaxNumberDimensions = 7;

    // Kernel/dilated/shift sizes.
    std::vector<int> kernel_sizes = {3, 5, 7}, dilated_sizes = {1, 2, 3}, shift_sizes = {1, 2, 3};

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

    explicit PrimitiveOptions(const std::string& allowed_str="",
                              const std::string& forbidden_str="",
                              std::vector<int> kernel_sizes={3, 5, 7},
                              std::vector<int> dilated_sizes={1, 2, 3},
                              std::vector<int> shift_sizes={1, 2, 3},
                              bool add_relu_bn_after_fc=false):
            kernel_sizes(std::move(kernel_sizes)),
            dilated_sizes(std::move(dilated_sizes)),
            shift_sizes(std::move(shift_sizes)),
            add_relu_bn_after_fc(add_relu_bn_after_fc) {
        BuildFilters(allowed_str, forbidden_str);
    }

    explicit PrimitiveOptions(int max_delta_width): max_delta_width(max_delta_width) {}

    void BuildFilters(const std::string& allowed_str="", const std::string& forbidden_str="");

    [[nodiscard]] bool Filter(const PrimitiveApply& pa) const;
};

/// Register all primitive constructions here.
struct PrimitiveFactory {
    static std::vector<PrimitiveApply> RescalePossibilities(const std::vector<PrimitiveApply>& applies);

    /// Get all primitives for a graph.
    static std::vector<PrimitiveApply> GetPrimitiveApplies(const GraphSP& graph,
                                                           const PrimitiveOptions& options=PrimitiveOptions());

    /// Get primitives with one input.
    static void GetPrimitiveApplies(const GraphSP &graph,
                                    std::vector<PrimitiveApply>& primitives,
                                    const TensorSP& t,
                                    const PrimitiveOptions& options);

    /// Get primitives with two inputs.
    static void GetPrimitiveApplies(const GraphSP &graph,
                                    std::vector<PrimitiveApply>& primitives,
                                    const TensorSP& lhs,
                                    const TensorSP& rhs,
                                    const PrimitiveOptions& options);
};

static void Push(const PrimitiveApply& pa,
                 std::vector<PrimitiveApply>& vec,
                 const PrimitiveOptions& options) {
    if (not options.Filter(pa))
        vec.push_back(pa);
}

template <typename PrimitiveType, class ... Args>
static void MakeAndPush(std::vector<PrimitiveApply>& vec,
                        const PrimitiveOptions& options,
                        Args&&... args) {
    auto pa = PrimitiveApply(std::make_shared<PrimitiveType>(args...));
    if (not options.Filter(pa))
        vec.push_back(pa);
}

} // namespace canvas
