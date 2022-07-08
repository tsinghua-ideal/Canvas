#include <gtest/gtest.h>

#include "Canvas/CodeGen/DotCodeGen.hpp"
#include "Canvas/CodeGen/PyTorchCodeGen.hpp"
#include "Canvas/Impls/Impls.hpp"


using namespace canvas;

TEST(CodeGen, LKA) {
    auto solution = ImplLKA();
    std::cout << DotCodeGen().Gen(solution) << std::endl;
    std::cout << PyTorchCodeGen().Gen(solution) << std::endl;
}
