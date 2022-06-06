#pragma once

#include <string>

#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct KernelPack {
    std::string torch_code, graphviz_code;

    KernelPack(std::string torch_code, std::string graphviz_code):
               torch_code(std::move(torch_code)),
               graphviz_code(std::move(graphviz_code)) {}
};

} // namespace canvas
