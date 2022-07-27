#include <utility>

#include "Canvas/Core/Specs.hpp"
#include "Canvas/Core/Solution.hpp"


namespace canvas {

struct SampleOptions {
    // Temporarily not visible to users.
    static constexpr int kMaxGroupFactor = 10;
    static constexpr int kMaxReductionFactor = 3;

    const std::string allowed_filter, forbidden_filter, necessary_filter;
    std::vector<std::string> necessaries;

    const std::vector<int> kernel_sizes, dilated_sizes, shift_sizes;

    const Range<int> np_range, mw_range, weighted_range;
    double max_weighted_ratio, force_bmm_possibility;

    int min_receptive_size;

    const canvas_timeval_t timeout;

    SampleOptions():
            necessary_filter("unfold"),
            kernel_sizes({3, 5, 7}), dilated_sizes({1, 2, 3}), shift_sizes({1, 2, 3}),
            np_range(3, 25), mw_range(2, 8), weighted_range(1, 8),
            max_weighted_ratio(0.6), force_bmm_possibility(0),
            min_receptive_size(1),
            timeout(std::chrono::seconds::zero()) {
        BuildNecessaryFilters();
    }

    SampleOptions(std::string allowed_filter,
                  std::string forbidden_filter,
                  std::string necessary_filter,
                  std::vector<int> kernel_sizes,
                  std::vector<int> dilated_sizes,
                  std::vector<int> shift_sizes,
                  int np_range_min, int np_range_max,
                  int mw_range_min, int mw_range_max,
                  int weighted_range_min, int weighted_range_max,
                  double max_weighted_ratio, double force_bmm_possibility,
                  int min_receptive_size,
                  int timeout):
            allowed_filter(std::move(allowed_filter)),
            forbidden_filter(std::move(forbidden_filter)),
            necessary_filter(std::move(necessary_filter)),
            kernel_sizes(std::move(kernel_sizes)),
            dilated_sizes(std::move(dilated_sizes)),
            shift_sizes(std::move(shift_sizes)),
            np_range(np_range_min, np_range_max),
            mw_range(mw_range_min, mw_range_max),
            weighted_range(weighted_range_min, weighted_range_max),
            max_weighted_ratio(max_weighted_ratio),
            force_bmm_possibility(force_bmm_possibility),
            min_receptive_size(min_receptive_size),
            timeout(std::chrono::seconds(timeout)) {
        for (int k: this->kernel_sizes)
            assert(k > 0 and k % 2 == 1);
        for (int d: this->dilated_sizes)
            assert(d > 0);
        for (int k: this->shift_sizes)
            assert(k > 0);
        assert(timeout > 0);
        assert(0 <= max_weighted_ratio and max_weighted_ratio <= 1.0);
        assert(0 <= force_bmm_possibility and force_bmm_possibility <= 1.0);
        assert(min_receptive_size > 0);
        BuildNecessaryFilters();
    }

    void BuildNecessaryFilters() {
        if (not necessary_filter.empty())
            Split(necessary_filter, necessaries);
    }

    [[nodiscard]] bool Filter(const GraphSP& graph) const {
        // Remaining options have been controlled during training.
        if (necessaries.empty())
            return false;
        std::vector<int> counter(necessaries.size(), 0);
        for (const auto& p: graph->primitives) {
            auto name = ToLowerCopy(p->name);
            for (int i = 0; i < necessaries.size(); ++ i)
                if (IsPrefix(name, necessaries[i]))
                    counter[i] ++;
        }
        return std::any_of(counter.begin(), counter.end(), [](int c) { return c == 0; });
    }
};

Solution TryRandomSample(const NetSpecsSP& net_specs, const SampleOptions& options=SampleOptions());

Solution RandomSample(const NetSpecsSP& net_specs, const SampleOptions& options=SampleOptions());

}
