#pragma once

#include <string>
#include <utility>

#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct KernelPack {
    std::string torch_code, graphviz_code, hash, exception_info;

    KernelPack(std::string exception_info): exception_info(std::move(exception_info)) {}

    KernelPack(std::string torch_code, std::string graphviz_code, std::string hash):
               torch_code(std::move(torch_code)),
               graphviz_code(std::move(graphviz_code)),
               hash(std::move(hash)) {}
};

} // namespace canvas
