#include <pybind11/pybind11.h>

namespace py = pybind11;

std::string Sample() {
    return "Hello, world!";
}

PYBIND11_MODULE(cc_canvas, m) {
    // Document
    m.doc() = "Python/C++ API for Canvas";

    // Sampling function
    m.def("sample", &Sample,
          "Sample a kernel from the space specified by the configuration");
}
