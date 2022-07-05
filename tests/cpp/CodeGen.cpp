#include <gtest/gtest.h>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/CodeGen/DotCodeGen.hpp"
#include "Canvas/CodeGen/PyTorchCodeGen.hpp"


using namespace canvas;

TEST(CodeGen, LKA) {
    auto graph = std::make_shared<Graph>();
    auto t_0 = graph->in;
    auto c = Variable::StaticVar(StaticVarPos::VC);
    auto proj_1 = std::make_shared<FCPrimitive>(t_0, c);
    graph->Apply(proj_1);
    auto activation = std::make_shared<ActivationPrimitive>(proj_1->outs[0], ActivationType::GeLU);
    graph->Apply(activation);
    auto lka_conv0 = std::make_shared<ConvolutionPrimitive>(activation->outs[0], c, c, 5, 5, 1, 1);
    graph->Apply(lka_conv0);
    auto lka_conv_spatial = std::make_shared<ConvolutionPrimitive>(lka_conv0->outs[0], c, c, 7, 7, 3, 3);
    graph->Apply(lka_conv_spatial);
    auto lka_conv1 = std::make_shared<FCPrimitive>(lka_conv_spatial->outs[0], c);
    graph->Apply(lka_conv1);
    auto b_mul = std::make_shared<BroadcastPrimitive>(lka_conv1->outs[0], activation->outs[0], BroadcastType::BMul);
    graph->Apply(b_mul);
    auto proj_2 = std::make_shared<FCPrimitive>(b_mul->outs[0], c);
    graph->Apply(proj_2);
    auto b_add = std::make_shared<BroadcastPrimitive>(t_0, proj_2->outs[0], BroadcastType::BAdd);
    graph->Apply(b_add);
    graph->ApplyOutput();

    auto solution = Solution(graph);
    std::cout << DotCodeGen().Gen(solution) << std::endl;
    std::cout << PyTorchCodeGen().Gen(solution) << std::endl;
}
