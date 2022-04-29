#include <chrono>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Canvas/Core/NetSpecs.hpp"
#include "Canvas/Search/RandomSample.hpp"
#include "Canvas/Utils/Common.hpp"


PYBIND11_MODULE(cpp_canvas, m) {
    // Document
    m.doc() = "Python/C++ API for Canvas";

    // The `canvas.sample` function, sampling a kernel from the search space
    m.def("sample",
          [](const std::vector<canvas::KernelSpecs>& kernels,
             double flops_min, double flops_max,
             double params_min, double params_max,
             bool allow_dynamic,
             bool force_irregular,
             bool add_relu_bn_after_fc,
             int np_min, int np_max,
             int fc_min, int fc_max,
             int timeout) -> void {
              auto net_specs = std::make_shared<canvas::NetSpecs>(
                      canvas::Range(flops_min, flops_max),
                      canvas::Range(params_min, params_max),
                      kernels);
              auto solution = canvas::RandomSample(net_specs,
                                                   allow_dynamic,
                                                   force_irregular,
                                                   add_relu_bn_after_fc,
                                                   canvas::Range(np_min, np_max),
                                                   canvas::Range(fc_min, fc_max),
                                                   std::chrono::seconds(timeout));
          },
          "Sample a kernel from the space specified by the configuration.");

    // The `canvas.seed` function, setting seed for the random engine
    m.def("seed",
          [](uint32_t seed) -> void {
              canvas::InitRandomEngine(false, seed);
          },
          "Set the global seed for the C++ random engine.");

    // The kernel specification class
    pybind11::class_<canvas::KernelSpecs>(m, "KernelSpecs")
            .def(pybind11::init<size_t, size_t, size_t, size_t, size_t, size_t>())
            .def_readwrite("ic", &canvas::KernelSpecs::ic)
            .def_readwrite("oc", &canvas::KernelSpecs::oc)
            .def_readwrite("k", &canvas::KernelSpecs::k)
            .def_readwrite("h", &canvas::KernelSpecs::h)
            .def_readwrite("w", &canvas::KernelSpecs::w)
            .def_readwrite("s", &canvas::KernelSpecs::s);
}
