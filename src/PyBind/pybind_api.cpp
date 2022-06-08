#include <chrono>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Canvas/Core/Specs.hpp"
#include "Canvas/CodeGen/DotCodeGen.hpp"
#include "Canvas/CodeGen/PyTorchCodeGen.hpp"
#include "Canvas/PyBind/KernelPack.hpp"
#include "Canvas/Search/RandomSample.hpp"
#include "Canvas/Utils/Common.hpp"


PYBIND11_MODULE(cpp_canvas, m) {
    // Document.
    m.doc() = "Python/C++ API for Canvas";

    // The kernel specification class.
    pybind11::class_<canvas::KernelSpecs>(m, "KernelSpecs")
            .def(pybind11::init<int, int, int>())
            .def_readwrite("c", &canvas::KernelSpecs::c)
            .def_readwrite("h", &canvas::KernelSpecs::h)
            .def_readwrite("w", &canvas::KernelSpecs::w);

    // The kernel pack (solution class in Python).
    pybind11::class_<canvas::KernelPack>(m, "KernelPackImpl")
            .def_readwrite("torch_code", &canvas::KernelPack::torch_code)
            .def_readwrite("graphviz_code", &canvas::KernelPack::graphviz_code);

    // The `canvas.sample` function, sampling a kernel from the search space.
    // TODO: wrap the sample options into a class.
    m.def("sample",
          [](const std::vector<canvas::KernelSpecs>& kernels,
             bool allow_dynamic,
             bool add_relu_bn_after_fc,
             int np_min, int np_max,
             int fc_min, int fc_max,
             int timeout) -> canvas::KernelPack {
              auto net_specs = std::make_shared<canvas::NetSpecs>(kernels);
              auto solution = canvas::RandomSample(net_specs,
                                                   allow_dynamic, add_relu_bn_after_fc,
                                                   canvas::Range(np_min, np_max),
                                                   canvas::Range(fc_min, fc_max),
                                                   std::chrono::seconds(timeout));
              auto torch_code = canvas::PyTorchCodeGen().Gen(solution);
              auto graphviz_code = canvas::DotCodeGen().Gen(solution);
              return {torch_code.ToString(), graphviz_code.ToString()};
          },
          "Sample a kernel from the space specified by the configuration.");

    // The `canvas.seed` function, setting seed for the random engine.
    m.def("seed",
          [](uint32_t seed) -> void {
              canvas::ResetRandomSeed(false, seed);
          },
          "Set the global seed for the C++ random engine.");
}
