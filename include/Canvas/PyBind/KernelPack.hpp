#pragma once

#include <string>

#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct KernelPack {
    std::string torch_code, graphviz_code;
    std::vector<std::vector<size_t>> fills;

    KernelPack(std::string torch_code,
               std::string graphviz_code,
               std::vector<std::vector<size_t>> fills):
               torch_code(std::move(torch_code)),
               graphviz_code(std::move(graphviz_code)),
               fills(std::move(fills)) {}
};

} // End namespace canvas
