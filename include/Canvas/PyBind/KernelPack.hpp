#pragma once

#include <string>

#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct KernelPack {
    std::string torch_code, graphviz_code, hash;

    KernelPack(std::string torch_code, std::string graphviz_code, std::string hash):
               torch_code(std::move(torch_code)),
               graphviz_code(std::move(graphviz_code)),
               hash(std::move(hash)) {}
};

} // namespace canvas
