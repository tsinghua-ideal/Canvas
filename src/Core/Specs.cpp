#include "Canvas/Core/Specs.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

NetSpecs::NetSpecs(std::vector<KernelSpecs> kernel_specs): kernel_specs(std::move(kernel_specs)) {
    c_gcd = 0;
    c_gcd_factors.clear();

    for (const auto& kernel: this->kernel_specs)
        c_gcd = c_gcd == 0 ? kernel.c : std::gcd(kernel.c, c_gcd);
    for (int i = 2; i <= c_gcd; ++ i)
        if (c_gcd % i == 0)
            c_gcd_factors.push_back(i);
}

std::ostream& operator << (std::ostream& os, const NetSpecs& rhs) {
    os << "NetSpecs:" << std::endl;
    os << " > Number of kernels: " << rhs.kernel_specs.size() << std::endl;
    for (int i = 0; i < rhs.kernel_specs.size(); ++ i) {
        const auto& kernel = rhs.kernel_specs.at(i);
        os << "   > Kernel#" << i << ": "
           << kernel.c << ", " << kernel.h << ", " << kernel.w << std::endl;
    }
    return os;
}

} // namespace canvas
