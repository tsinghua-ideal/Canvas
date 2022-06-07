#include <gtest/gtest.h>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Shape.hpp"
#include "Canvas/Core/Variable.hpp"
#include "Canvas/Primitives/Factory.hpp"


using namespace canvas;

TEST(Core, Variable) {
    Variable x({StaticVar::VC, StaticVar::VDG}, {0, 4});
    Variable y({StaticVar::VKH, StaticVar::VKW}, {1});
    Variable a({StaticVar::VG, StaticVar::VH, StaticVar::VW}, {2});

    std::stringstream ss_x;
    ss_x << x;
    ASSERT_EQ(ss_x.str(), "C*x_0*x_4/G");

    std::stringstream ss_y;
    ss_y << y;
    ASSERT_EQ(ss_y.str(), "KH*KW*x_1");

    std::stringstream ss_xy;
    ss_xy << x * y;
    ASSERT_EQ(ss_xy.str(), "C*KH*KW*x_0*x_1*x_4/G");

    std::stringstream ss_ax;
    ss_ax << a * x;
    ASSERT_EQ(ss_ax.str(), "C*H*W*x_0*x_2*x_4");
}

TEST(Core, DynamicVariable) {
    Variable a({StaticVar::VC, StaticVar::VDG}, {0, 1, 1, 4});
    Variable x1({StaticVar::VC, StaticVar::VKH, StaticVar::VKH});

    a.SolveDynamicVar({1, x1});
    std::stringstream ss;
    ss << a;
    ASSERT_EQ(ss.str(), "C*C*C*KH*KH*KH*KH*x_0*x_4/G");
}

TEST(Core, VariableFactors) {
    Variable x({StaticVar::VC, StaticVar::VC, StaticVar::VG}, {0, 2, 2});
    std::cout << "All factors of " << x << ":" << std::endl;
    for (const auto& factor: x.GetAllFactors())
        std::cout << " > Factor: " << factor << std::endl;

    Variable y({StaticVar::VC, StaticVar::VKH, StaticVar::VKW});
    std::cout << "All factors of " << y << ":" << std::endl;
    for (const auto& factor: y.GetAllFactors())
        std::cout << " > Factor: " << factor << std::endl;
}

TEST(Core, Shape) {
    Shape s;
    s.C() = StaticVar::VC, s.H() = StaticVar::VH, s.W() = StaticVar::VW;
    std::stringstream ss;
    ss << s;
    ASSERT_EQ(ss.str(), "[C, H, W]");
}

TEST(Core, GraphDeconstruction) {
    int prev_num_tensor_deconstruction = Tensor::num_deconstruction;
    int prev_num_primitive_deconstruction = Primitive::num_deconstruction;
    {
        auto graph = std::make_shared<Graph>();
        auto in = graph->in;
        graph->Apply(std::make_shared<UnfoldPrimitive>(in));
    }
    ASSERT_EQ(prev_num_tensor_deconstruction + 2, Tensor::num_deconstruction);
    ASSERT_EQ(prev_num_primitive_deconstruction + 2, Primitive::num_deconstruction);
}

TEST(Core, GraphCopy) {
    int prev_num_tensor_deconstruction = Tensor::num_deconstruction;
    int prev_num_primitive_deconstruction = Primitive::num_deconstruction;
    {
        auto graph = std::make_shared<Graph>();
        auto in = graph->in;
        graph->Apply(std::make_shared<UnfoldPrimitive>(in));

        // Copy
        {
            auto copy = std::make_shared<Graph>(*graph);
        }
        ASSERT_EQ(prev_num_tensor_deconstruction + 2, Tensor::num_deconstruction);
        ASSERT_EQ(prev_num_primitive_deconstruction + 2, Primitive::num_deconstruction);

        // Copy and apply
        {
            auto unfold_2 = std::make_shared<UnfoldPrimitive>(in);
#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedLocalVariable"
            auto [copy, remapped] = graph->CopyAndApply(unfold_2);
#pragma clang diagnostic pop
        }
        ASSERT_EQ(prev_num_tensor_deconstruction + 6, Tensor::num_deconstruction);
        ASSERT_EQ(prev_num_primitive_deconstruction + 6, Primitive::num_deconstruction);
    }
    ASSERT_EQ(prev_num_tensor_deconstruction + 8, Tensor::num_deconstruction);
    ASSERT_EQ(prev_num_primitive_deconstruction + 8, Primitive::num_deconstruction);
}
