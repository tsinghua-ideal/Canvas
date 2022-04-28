#pragma once

#include <vector>
#include <unordered_set>

#include "Canvas/Primitives/Activation.hpp"
#include "Canvas/Primitives/Broadcast.hpp"
#include "Canvas/Primitives/ChannelShuffle.hpp"
#include "Canvas/Primitives/Dot.hpp"
#include "Canvas/Primitives/Dropout.hpp"
#include "Canvas/Primitives/ElementWise.hpp"
#include "Canvas/Primitives/FC.hpp"
#include "Canvas/Primitives/Fold.hpp"
#include "Canvas/Primitives/Group.hpp"
#include "Canvas/Primitives/Input.hpp"
#include "Canvas/Primitives/Norm.hpp"
#include "Canvas/Primitives/Output.hpp"
#include "Canvas/Primitives/Pool.hpp"
#include "Canvas/Primitives/Reorder.hpp"
#include "Canvas/Primitives/Reshape.hpp"
#include "Canvas/Primitives/Shift.hpp"
#include "Canvas/Primitives/Softmax.hpp"
#include "Canvas/Primitives/Transpose.hpp"
#include "Canvas/Primitives/Unfold.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

struct Graph;
typedef std::shared_ptr<Graph> GraphSP;

struct PrimitiveGenOptions {
    // Dynamic variables
    bool allow_dynamic_variables = true;

    // Optimize FC
    bool optimize_fc = false;

    // Force irregular applies
    bool force_irregular = false;

    // Broadcast
    bool b_add = false, b_mul = false, b_sub = false;

    // With parameters
    bool dot = false, fc = false;

    // Arithmetic operators
    bool dropout = false, norm = false;
    bool gelu = false, relu = false, sigmoid = false, tanh = false;
    bool softmax_c = false, softmax_h = false, softmax_w = false, softmax_hw = false;
    bool abs = false, exp = false, neg = false, sin = false;

    // Spatial operators
    bool channel_shuffle = false;
    bool fold_h = false, fold_w = false, fold_hw = false;
    bool fold_avg = false, fold_max = false;
    bool unfold_h = false, unfold_w = false, unfold_hw = false;
    bool group_all = false, group_by_factor = false;
    bool pool_h = false, pool_w = false, pool_hw = false;
    bool reorder_h = false, reorder_w = false, reorder_hw = false;
    bool shift_h = false, shift_w = false, shift_hw = false;
    bool transpose = false;

    /// `max_delta_width` could be -1 (reducing width), 0 (retaining width), 1 (unlimited)
    int max_delta_width = 1;

    PrimitiveGenOptions(bool enable_all, int max_delta_width, bool only_fc=false):
            max_delta_width(max_delta_width) {
        if (enable_all) {
            assert(not only_fc);
            allow_dynamic_variables = true;
            force_irregular = false;
            optimize_fc = false;
            b_add = b_mul = b_sub = true;
            dot = fc = true;
            dropout = norm = gelu = relu = sigmoid = tanh = true;
            abs = exp = neg = sin = true;
            softmax_c = softmax_h = softmax_w = softmax_hw = true;
            channel_shuffle = true;
            fold_h = fold_w = fold_hw = true;
            fold_avg = fold_max = true;
            unfold_h = unfold_w = unfold_hw = true;
            group_all = group_by_factor = true;
            pool_h = pool_w = pool_hw = true;
            reorder_h = reorder_w = reorder_hw = true;
            shift_h = shift_w = shift_hw = true;
            transpose = true;
        }
        if (only_fc) {
            allow_dynamic_variables = true;
            force_irregular = false;
            optimize_fc = false;
            fc = true;
        }
    }

    static PrimitiveGenOptions Recommended() {
        auto options = PrimitiveGenOptions(true, 1, false);
        options.allow_dynamic_variables = true, options.force_irregular = false, options.optimize_fc = false;
        // options.dropout = options.norm = options.gelu = options.tanh = false;
        // options.abs = options.exp = options.neg = options.sin = false;
        // options.channel_shuffle = false;
        // options.transpose = false;
        // options.softmax_c = options.softmax_h = options.softmax_w = options.softmax_hw = false;
        return options;
    }

    static PrimitiveGenOptions Unlimited() { return {true, 1}; }

    static PrimitiveGenOptions NoFactorGrouping() {
        auto options = Unlimited();
        options.group_by_factor = false;
        return options;
    }

    static PrimitiveGenOptions NoNeighborInvolved() {
        auto options = Unlimited();
        options.unfold_hw = options.unfold_h = options.unfold_w = false;
        options.fold_h = options.fold_w = options.fold_hw = options.fold_max = options.fold_avg = false;
        return options;
    }

    static PrimitiveGenOptions ReduceWidth() { return {true, -1}; }

    static PrimitiveGenOptions NotExpanding() { return {true, 0}; }

    static PrimitiveGenOptions FC(int max_delta_width=1) { return {false, max_delta_width, true}; }

    static PrimitiveGenOptions None() { return {false, 1}; }

    [[nodiscard]] bool Adapt(const PrimitiveApply& apply) const;

    friend PrimitiveGenOptions operator and (const PrimitiveGenOptions& lhs, const PrimitiveGenOptions& rhs) {
        PrimitiveGenOptions options = PrimitiveGenOptions::Unlimited();
#define CANVAS_AND_OPT(name) options.name = lhs.name and rhs.name
#define CANVAS_OR_OPT(name) options.name = lhs.name or rhs.name
        CANVAS_AND_OPT(allow_dynamic_variables);
        CANVAS_OR_OPT(force_irregular), CANVAS_OR_OPT(optimize_fc);
        CANVAS_AND_OPT(b_add), CANVAS_AND_OPT(b_mul), CANVAS_AND_OPT(b_sub);
        CANVAS_AND_OPT(dot), CANVAS_AND_OPT(fc);
        CANVAS_AND_OPT(dropout), CANVAS_AND_OPT(norm);
        CANVAS_AND_OPT(gelu), CANVAS_AND_OPT(relu), CANVAS_AND_OPT(sigmoid), CANVAS_AND_OPT(tanh);
        CANVAS_AND_OPT(softmax_c), CANVAS_AND_OPT(softmax_h), CANVAS_AND_OPT(softmax_w), CANVAS_AND_OPT(softmax_hw);
        CANVAS_AND_OPT(abs), CANVAS_AND_OPT(exp), CANVAS_AND_OPT(neg), CANVAS_AND_OPT(sin);
        CANVAS_AND_OPT(channel_shuffle);
        CANVAS_AND_OPT(fold_h), CANVAS_AND_OPT(fold_w), CANVAS_AND_OPT(fold_hw);
        CANVAS_AND_OPT(fold_avg), CANVAS_AND_OPT(fold_max);
        CANVAS_AND_OPT(unfold_h), CANVAS_AND_OPT(unfold_w), CANVAS_AND_OPT(unfold_hw);
        CANVAS_AND_OPT(group_by_factor), CANVAS_AND_OPT(group_all);
        CANVAS_AND_OPT(pool_h), CANVAS_AND_OPT(pool_w), CANVAS_AND_OPT(pool_hw);
        CANVAS_AND_OPT(reorder_h), CANVAS_AND_OPT(reorder_w), CANVAS_AND_OPT(reorder_hw);
        CANVAS_AND_OPT(shift_h), CANVAS_AND_OPT(shift_w), CANVAS_AND_OPT(shift_hw);
        CANVAS_AND_OPT(transpose);
#undef CANVAS_AND_OPT
        options.max_delta_width = std::min(lhs.max_delta_width, rhs.max_delta_width);
        return options;
    }
};

/// Register all primitive constructions here
struct PrimitiveFactory {
    /// Get all primitives for a graph
    static std::vector<PrimitiveApply> GetPrimitiveApplies(const GraphSP& graph,
                                                           bool allow_dynamic_variables=true,
                                                           const std::set<TensorSP>& forbidden_successor={});

    /// Filter primitive applies with conditions
    static std::vector<PrimitiveApply> FilterPrimitiveApplies(const std::vector<PrimitiveApply>& applies,
                                                              const PrimitiveGenOptions& options);

    /// Filter primitive applies with conditions
    static std::vector<PrimitiveApply> FilterPrimitiveApplies(const std::vector<PrimitiveApply>& applies,
                                                              const std::vector<PrimitiveGenOptions>& or_options);

    /// Filter for output
    static std::vector<PrimitiveApply> FilterPrimitiveAppliesForOutput(const std::vector<PrimitiveApply>& applies);

    /// Rescale with possibilities
    static std::vector<PrimitiveApply> RescalePossibilities(const std::vector<PrimitiveApply>& applies,
                                                            bool remove_fc=false);

    /// Get primitives without inputs
    static void GetPrimitiveApplies(const GraphSP &graph,
                                    std::vector<PrimitiveApply>& primitives,
                                    std::unordered_set<size_t>& filter,
                                    bool allow_dynamic_variables=true);

    /// Get primitives with one input
    static void GetPrimitiveApplies(const GraphSP &graph,
                                    std::vector<PrimitiveApply>& primitives,
                                    const TensorSP& t,
                                    std::unordered_set<size_t>& filter,
                                    bool allow_dynamic_variables=true);

    /// Get primitives with two inputs
    static void GetPrimitiveApplies(const GraphSP &graph,
                                    std::vector<PrimitiveApply>& primitives,
                                    const TensorSP& lhs,
                                    const TensorSP& rhs,
                                    std::unordered_set<size_t>& filter,
                                    bool allow_dynamic_variables=true);
};

static void TryPush(const PrimitiveApply& pa, std::vector<PrimitiveApply>& vec,
                    std::unordered_set<size_t>& filter) {
    // Pruning: not duplicate tensors
    auto hash_value = pa.primitive->Hash(true);
    if (not filter.count(hash_value)) {
        vec.push_back(pa);
        filter.insert(hash_value);
    }
}

template <typename PrimitiveType, class ... Args>
static void TryMakeAndPush(std::vector<PrimitiveApply>& vec,
                           std::unordered_set<size_t>& filter,
                           Args&&... args) {
    try {
        auto p = std::make_shared<PrimitiveType>(args...);
        TryPush(PrimitiveApply(p), vec, filter);
    } catch (CanNotApplyPrimitive& e) {}
}

} // End namespace canvas
