#include <gtest/gtest.h>
#include <unordered_set>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Shape.hpp"
#include "Canvas/Core/Tensor.hpp"
#include "Canvas/Primitives/Factory.hpp"


using namespace canvas;

TEST(Primitives, DuplicatePrimitiveChecking) {
    auto graph = std::make_shared<Graph>();
    auto group = std::make_shared<GroupPrimitive>(graph->in, 0);
    graph->Apply(group);

    PrimitiveOptions filter;
    filter.hash_filter.insert(group->Hash(true));

    int count = 0;
    auto applies = PrimitiveFactory::GetPrimitiveApplies(graph, filter);
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
    Shape lhs_shape = Shape::MakeChannelSpatial(), rhs_shape = Shape::MakeChannelSpatial();
    lhs_shape.Channel()->G() = StaticVarPos::VG, lhs_shape.Channel()->C() = Variable::DynamicVar(2) / StaticVarPos::VG;
    lhs_shape.Spatial()->H() = StaticVarPos::VH, lhs_shape.Spatial()->W() = StaticVarPos::VW;

    rhs_shape.Channel()->G() = StaticVarPos::VG, rhs_shape.Channel()->C() = StaticVarPos::VC / StaticVarPos::VG;
    rhs_shape.Spatial()->H() = StaticVarPos::VH, rhs_shape.Spatial()->W() = StaticVarPos::VW;
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
    Shape lhs_shape = Shape::MakeChannelSpatial(), rhs_shape = Shape::MakeShapeCHW();
    lhs_shape.Channel()->C() = Variable::DynamicVar(3);
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
    auto t1 = std::make_shared<Tensor>(Shape::MakeShapeCHW()), t2 = std::make_shared<Tensor>(Shape::MakeShapeCHW());
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

// TODO: check whether all primitives exist.
