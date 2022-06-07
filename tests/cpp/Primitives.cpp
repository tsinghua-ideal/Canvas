#include <gtest/gtest.h>
#include <unordered_set>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Shape.hpp"
#include "Canvas/Core/Tensor.hpp"
#include "Canvas/Primitives/Factory.hpp"


using namespace canvas;

TEST(Primitives, DuplicatePrimitiveChecking) {
    auto graph = std::make_shared<Graph>();
    auto group = std::make_shared<GroupPrimitive>(graph->in);
    graph->Apply(group);

    int count = 0;
    auto applies = PrimitiveFactory::GetPrimitiveApplies(graph);
    for (const auto& apply: applies) {
        if (apply.primitive->ins[0] == graph->in) {
            if (auto p = DynamicCast<GroupPrimitive>(apply.primitive)) {
                ++ count;
                ASSERT_TRUE(p->type == GroupAllChannels);
            }
        }
    }
    ASSERT_EQ(count, 1);
}

TEST(Primitives, BroadcastDynamicMatching) {
    Shape lhs_shape, rhs_shape;
    lhs_shape.G() = StaticVar::VG, lhs_shape.C() = Variable::Dynamic(2) / StaticVar::VG;
    lhs_shape.H() = StaticVar::VH, lhs_shape.W() = StaticVar::VW;
    rhs_shape.G() = StaticVar::VG, rhs_shape.C() = StaticVar::VC / StaticVar::VG;
    rhs_shape.KH() = StaticVar::VKH, rhs_shape.KW() = StaticVar::VKW;
    rhs_shape.H() = StaticVar::VH, rhs_shape.W() = StaticVar::VW;
    auto lhs = std::make_shared<Tensor>(lhs_shape), rhs = std::make_shared<Tensor>(rhs_shape);
    auto all_matches = BroadcastPrimitive::GetAllPossibleMatches(lhs, rhs, BMul);
    std::cout << "Solutions:" << std::endl;
    for (const auto& [p, s]: all_matches) {
        if (s.has_value())
            std::cout << " > x_" << s.value().index << " = " << s.value().substitution << std::endl;
        else
            std::cout << " > empty match" << std::endl;
    }
}

TEST(Primitives, BroadcastDynamicMatchingHW) {
    Shape lhs_shape, rhs_shape;
    lhs_shape.C() = Variable::Dynamic(3);
    rhs_shape.C() = StaticVar::VC, rhs_shape.H() = StaticVar::VH, rhs_shape.W() = StaticVar::VW;
    auto lhs = std::make_shared<Tensor>(lhs_shape), rhs = std::make_shared<Tensor>(rhs_shape);
    auto all_matches = BroadcastPrimitive::GetAllPossibleMatches(lhs, rhs, BMul);
    std::cout << "Solutions:" << std::endl;
    for (const auto& [p, s]: all_matches) {
        if (s.has_value())
            std::cout << " > x_" << s.value().index << " = " << s.value().substitution << std::endl;
        else
            std::cout << " > empty match" << std::endl;
    }
}

TEST(Primitives, BroadcastSpecialCases) {
    Shape s1, s2;
    s1.C() = StaticVar::VC, s1.KH() = StaticVar::VKH, s1.KW() = StaticVar::VKH, s1.H() = StaticVar::VH, s1.W() = StaticVar::VW;
    s2.C() = StaticVar::VC, s2.KH() = StaticVar::VKH, s2.H() = StaticVar::VH, s2.W() = StaticVar::VW;
    auto t1 = std::make_shared<Tensor>(s1), t2 = std::make_shared<Tensor>(s2);
    auto b_add = std::make_shared<BroadcastPrimitive>(t2, t1, BroadcastType::BAdd);
    std::cout << *b_add << std::endl;
    std::cout << "Prefix:" << std::endl;
    for (const auto& p: b_add->prefix)
        std::cout << " > " << p << std::endl;
    std::cout << "Suffix:" << std::endl;
    for (const auto& p: b_add->suffix)
        std::cout << " > " << p << std::endl;
    std::cout << "Multiplier:" << b_add->multiplier << std::endl;
    std::cout << std::endl;
}
