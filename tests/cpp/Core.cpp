#include <gtest/gtest.h>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Shape.hpp"
#include "Canvas/Core/Variable.hpp"
#include "Canvas/Primitives/Factory.hpp"


using namespace canvas;

TEST(Core, Variable) {
    auto x = Variable::Compose({StaticVarPos::VC, StaticVarPos::VDG}, 1, 1, {0, 4});
    auto y = Variable::Number(10, 8);
    auto z = x * y;
    auto a = Variable::Compose({StaticVarPos::VG, StaticVarPos::VH, StaticVarPos::VW}, 1, 1, {2});

    std::stringstream ss_x;
    ss_x << x;
    ASSERT_EQ(ss_x.str(), "C*x_0*x_4/G");

    std::stringstream ss_y;
    ss_y << y;
    ASSERT_EQ(ss_y.str(), "5/4");
    ASSERT_FALSE(y.MaybeInteger());

    std::stringstream ss_z;
    ss_z << z;
    ASSERT_EQ(ss_z.str(), "C*x_0*x_4*5/4/G");
    ASSERT_TRUE(z.MaybeInteger());

    std::stringstream ss_ax;
    ss_ax << a * x;
    ASSERT_EQ(ss_ax.str(), "C*H*W*x_0*x_2*x_4");
    ASSERT_TRUE((a * x).MaybeInteger());
}

TEST(Core, DynamicVariable) {
    auto a = Variable::Compose({StaticVarPos::VC, StaticVarPos::VDG}, 1, 1, {0, 1, 1, 4});
    auto x1 = Variable::StaticVar(StaticVarPos::VC);

    a.SolveDynamicVar({1, x1});
    std::stringstream ss;
    ss << a;
    ASSERT_EQ(ss.str(), "C*C*C*x_0*x_4/G");
}

TEST(Core, VariableFactors) {
    auto x = Variable::Compose({StaticVarPos::VC, StaticVarPos::VC, StaticVarPos::VG}, 4, 5, {0, 2, 2});
    std::cout << "All factors of " << x << ":" << std::endl;
    for (const auto& factor: x.GetAllFactors())
        std::cout << " > Factor: " << factor << std::endl;

    auto y = Variable::Compose({StaticVarPos::VC, StaticVarPos::VG});
    std::cout << "All factors of " << y << ":" << std::endl;
    for (const auto& factor: y.GetAllFactors())
        std::cout << " > Factor: " << factor << std::endl;
}

TEST(Core, Shape) {
    Shape s;
    s.C() = StaticVarPos::VC, s.H() = StaticVarPos::VH, s.W() = StaticVarPos::VW;
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
        graph->Apply(std::make_shared<ActivationPrimitive>(in));
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
        graph->Apply(std::make_shared<ActivationPrimitive>(in));

        // Copy.
        {
            auto copy = std::make_shared<Graph>(*graph);
        }
        ASSERT_EQ(prev_num_tensor_deconstruction + 2, Tensor::num_deconstruction);
        ASSERT_EQ(prev_num_primitive_deconstruction + 2, Primitive::num_deconstruction);

        // Copy and apply.
        {
            auto unfold_2 = std::make_shared<ActivationPrimitive>(in);
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
