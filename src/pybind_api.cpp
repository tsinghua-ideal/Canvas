#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Canvas/Core/NetSpecs.hpp"
#include "Canvas/Utils/Common.hpp"


PYBIND11_MODULE(cpp_canvas, m) {
    // Document
    m.doc() = "Python/C++ API for Canvas";

    // The `canvas.sample` function, sampling a kernel from the search space
    m.def("sample",
          [](const std::vector<canvas::KernelSpecs>& layers,
             double flops_budget,
             double params_budget,
             bool allow_dynamic,
             bool force_irregular,
             bool add_relu_bn_after_fc,
             int np_min, int np_max,
             int fc_min, int fc_max,
             int timeout) -> void {
              canvas::Unimplemented();
          },
          "Sample a kernel from the space specified by the configuration.");

    // The `canvas.seed` function, setting seed for the random engine
    m.def("seed",
          [](uint32_t seed) -> void {
              canvas::InitRandomEngine(false, seed);
          },
          "Set the global seed for the C++ random engine.");

    // The layer specification class
    pybind11::class_<canvas::KernelSpecs>(m, "KernelSpecs")
            .def(pybind11::init<size_t, size_t, size_t, size_t, size_t, size_t>())
            .def_readwrite("ic", &canvas::KernelSpecs::ic)
            .def_readwrite("oc", &canvas::KernelSpecs::oc)
            .def_readwrite("k", &canvas::KernelSpecs::k)
            .def_readwrite("h", &canvas::KernelSpecs::h)
            .def_readwrite("w", &canvas::KernelSpecs::w)
            .def_readwrite("s", &canvas::KernelSpecs::s);
}
