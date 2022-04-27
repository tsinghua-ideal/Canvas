#include <pybind11/pybind11.h>

#include "Canvas/Utils/Common.hpp"


PYBIND11_MODULE(cpp_canvas, m) {
    // Document
    m.doc() = "Python/C++ API for Canvas";

    // The `canvas.sample` function, sampling a kernel from the search space
    m.def("sample",
          []() -> void {
              canvas::Unimplemented();
          },
          "Sample a kernel from the space specified by the configuration.");

    // The `canvas.seed` function, setting seed for the random engine
    m.def("seed",
          [](uint32_t seed) -> void {
              canvas::InitRandomEngine(false, seed);
          },
          "Set the global seed for the C++ random engine.");
}
