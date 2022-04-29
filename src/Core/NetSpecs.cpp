#include "Canvas/Core/NetSpecs.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

NetSpecs::NetSpecs(const std::string& str) {
    /*
     * `NetSpecs` format (in a single line):
     *   FLOPs_Min_Ratio, FLOPs_Max_Ratio, Ps_Min_Ratio, Ps_Max_Ratio,
     *   Number_of_Kernels, [C_i, OC_i, K_i, S_i, H_i, W_i]*
     * Example:
     *   0.1, 1.0, 0.1, 1.0, [19, [64, 64, 3, 1, 32, 32], [64, 64, 3, 1, 32, 32], [64, 64, 3, 1, 32, 32], [64, 64, 3, 1, 32, 32], [64, 128, 3, 2, 32, 32], [128, 128, 3, 1, 16, 16], [64, 128, 1, 2, 32, 32], [128, 128, 3, 1, 16, 16], [128, 128, 3, 1, 16, 16], [128, 256, 3, 2, 16, 16], [256, 256, 3, 1, 8, 8], [128, 256, 1, 2, 16, 16], [256, 256, 3, 1, 8, 8], [256, 256, 3, 1, 8, 8], [256, 512, 3, 2, 8, 8], [512, 512, 3, 1, 4, 4], [256, 512, 1, 2, 8, 8], [512, 512, 3, 1, 4, 4], [512, 512, 3, 1, 4, 4]]
     * */
    int next_pos = 0;
    auto ReadNextToken = [&next_pos, str]() -> std::string {
        while (next_pos < str.length() and not std::isdigit(str.at(next_pos)) and str.at(next_pos) != '.')
            ++ next_pos;
        if (next_pos == str.length())
            return "";
        int start_pos = next_pos;
        while (next_pos < str.length() and (std::isdigit(str.at(next_pos)) or str.at(next_pos) == '.'))
            ++ next_pos;
        return str.substr(start_pos, next_pos - start_pos);
    };

    // 0 for no limits
    auto lhs = StringTo<double>(ReadNextToken()), rhs = StringTo<double>(ReadNextToken());
    flops_ratio_range = Range<double>(lhs, rhs);
    lhs = StringTo<double>(ReadNextToken()), rhs = StringTo<double>(ReadNextToken());
    ps_ratio_range = Range<double>(lhs, rhs);
    static Range<double> checker(kMinCheckRatio, kMaxCheckRatio);
    if (not checker.Contains(flops_ratio_range.min) or not checker.Contains(flops_ratio_range.max)
        or not checker.Contains(ps_ratio_range.min) or not checker.Contains(ps_ratio_range.max))
        CriticalError("The ratio range is so low or so high");

    int n = StringTo<int>(ReadNextToken());
    if (n <= 0)
        CriticalError("Illegal number of kernels");

    kernel_specs.reserve(n);
    for (int i = 0; i < n; ++ i) {
        size_t ic, oc, k, h, w, s;
        ic = StringTo<size_t>(ReadNextToken());
        oc = StringTo<size_t>(ReadNextToken());
        k = StringTo<size_t>(ReadNextToken());
        s = StringTo<size_t>(ReadNextToken());
        h = StringTo<size_t>(ReadNextToken());
        w = StringTo<size_t>(ReadNextToken());
        kernel_specs.emplace_back(ic, oc, k, h, w, s);
    }

    BuildPreferences();
}

NetSpecs::NetSpecs(const Range<double>& flops_ratio_range,
                   const Range<double>& ps_ratio_range,
                   std::vector<KernelSpecs> kernel_specs):
        flops_ratio_range(flops_ratio_range),
        ps_ratio_range(ps_ratio_range),
        kernel_specs(std::move(kernel_specs)) {
    BuildPreferences();
}

void NetSpecs::BuildPreferences() {
    c_gcd = 0;
    c_gcd_factors.clear();

    standard_conv_flops = 0, standard_conv_ps = 0;
    no_neighbor_involved = true;
    for (const auto& kernel: this->kernel_specs) {
        assert(std::gcd(kernel.ic, kernel.oc) == std::min(kernel.ic, kernel.oc));
        if (kernel.k > 1)
            no_neighbor_involved = false;
        if (kernel.h % kernel.s != 0 or kernel.w % kernel.s != 0)
            CriticalError("Height and width should be dividable by striding number");

        c_gcd = c_gcd == 0 ? std::min(kernel.ic, kernel.oc) : std::gcd(std::min(kernel.ic, kernel.oc), c_gcd);
        standard_conv_flops += kernel.ic * kernel.k * kernel.k * kernel.oc * kernel.h / kernel.s * kernel.w / kernel.s * 2;
        standard_conv_ps += kernel.ic * kernel.k * kernel.k * kernel.oc;
    }

    size_t min, max;
    min = static_cast<size_t>(static_cast<double>(standard_conv_flops) * flops_ratio_range.min);
    max = static_cast<size_t>(static_cast<double>(standard_conv_flops) * flops_ratio_range.max);
    assert(0 <= min and min < max);
    flops_range = Range<size_t>(min, max);

    min = static_cast<size_t>(static_cast<double>(standard_conv_ps) * ps_ratio_range.min);
    max = static_cast<size_t>(static_cast<double>(standard_conv_ps) * ps_ratio_range.max);
    assert(0 <= min and min < max);
    ps_range = Range<size_t>(min, max);

    for (size_t i = 2; i <= c_gcd; ++ i)
        if (c_gcd % i == 0)
            c_gcd_factors.push_back(i);
}

std::ostream& operator << (std::ostream& os, const NetSpecs& rhs) {
    os << "NetSpecs:" << std::endl;
    if (rhs.standard_conv_flops > 0 and rhs.standard_conv_ps > 0) {
        os << " > Standard convolution FLOPs: " << rhs.standard_conv_flops << std::endl;
        os << " > Standard convolution Ps: " << rhs.standard_conv_ps << std::endl;
    }
    if (not rhs.flops_ratio_range.IsPoint() and not rhs.ps_ratio_range.IsPoint()) {
        os << " > FLOPs ratio range: " << rhs.flops_ratio_range << std::endl;
        os << " > Ps ratio range: " << rhs.ps_ratio_range << std::endl;
    }
    os << " > FLOPs range: " << rhs.flops_range << std::endl;
    os << " > Ps range: " << rhs.ps_range << std::endl;
    os << " > Number of kernels: " << rhs.kernel_specs.size() << std::endl;
    for (int i = 0; i < rhs.kernel_specs.size(); ++ i) {
        const auto& kernel = rhs.kernel_specs.at(i);
        os << "   > Kernel#" << i << ": "
           << kernel.ic << ", " << kernel.oc << ", "
           << kernel.k << ", " << kernel.h << ", "
           << kernel.w << ", " << kernel.s
           << std::endl;
    }
    return os;
}

std::ostream& operator <<(std::ostream& os, const NetFills& fills) {
    os << "NetFills (" << fills.kernel_fills.size() << " kernels): ";
    bool displayed = false;
    for (const auto& dynamic_fills: fills.kernel_fills)
        os << (displayed ? ", " : "") << dynamic_fills, displayed = true;
    return displayed ? os : (os << "null");
}

} // End namespace canvas
