#include <gtest/gtest.h>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Variable.hpp"
#include "Canvas/Primitives/Factory.hpp"

// #define CANVAS_DEBUG_SEARCH_TEST_PRINT_ALL_ACTIONS


using namespace canvas;

GraphSP RandomGraph(int n_primitives, const std::vector<PrimitiveGenOptions>& or_options={}) {
    auto TryToGenerate = [&]() -> GraphSP {
        auto graph = std::make_shared<Graph>();
        for (int i = 1; i <= n_primitives; ++ i) {
            auto applies = PrimitiveFactory::GetPrimitiveApplies(graph);
            std::cout << "Round " << i << ":" << std::endl;
#ifdef CANVAS_DEBUG_SEARCH_TEST_PRINT_ALL_ACTIONS
            std::cout << " * Before filtering:" << std::endl;
            for (const auto& [p, s]: applies)
                std::cout << "  > " << p->name << ": " << *p << std::endl;
#endif
            if (not or_options.empty()) {
                applies = PrimitiveFactory::FilterPrimitiveApplies(applies, or_options);
#ifdef CANVAS_DEBUG_SEARCH_TEST_PRINT_ALL_ACTIONS
                std::cout << " * After filtering:" << std::endl;
                for (const auto& [p, s]: applies)
                    std::cout << "  > " << p->name << ": " << *p << std::endl;
#endif
            }
            auto apply = RandomChoose(applies);
            std::cout << " * Choose " << apply << std::endl;
            graph->Apply(apply);
        }
        return graph;
    };

    int try_times = 0;
    GraphSP graph = nullptr;
    while (graph == nullptr) {
        try {
            ++ try_times;
            graph = TryToGenerate();
        } catch (const CanNotSolveDynamicVar& _) {}
    }
    std::cout << "Generate a graph with " << try_times << " tries" << std::endl;
    return graph;
}

TEST(Search, PrimitiveFactory) {
    int n_primitives = 10;
    auto graph = RandomGraph(n_primitives);
    std::cout << graph->Hash() << std::endl;
}

TEST(Search, PrimitiveFactoryFixedSingleWidth) {
    int n_primitives = 10;
    auto graph = RandomGraph(n_primitives, {
        PrimitiveGenOptions::NotExpanding()
    });
    std::cout << graph->Hash() << std::endl;
}

TEST(Search, PrimitiveFactoryFixedSingleWidthOrFC) {
    int n_primitives = 20;
    auto graph = RandomGraph(n_primitives, {
        PrimitiveGenOptions::NotExpanding(), PrimitiveGenOptions::FC()
    });
    std::cout << graph->Hash() << std::endl;
}

TEST(Search, PrimitiveFactoryOnlyFC) {
    int n_primitives = 10;
    auto graph = RandomGraph(n_primitives, {
        PrimitiveGenOptions::FC()
    });
    std::cout << graph->Hash() << std::endl;
}

TEST(Search, PrimitiveFactoryReduceWidth) {
    static int n_primitives = 10;

    while (true) {
        try {
            GraphSP graph;
            while (true) {
                graph = RandomGraph(n_primitives);
                if (graph->Width() < 5)
                    break;
            }
            std::cout << graph->Hash() << std::endl;

            // Try to reduce to 1.
            std::cout << "Reducing width to 1" << std::endl;
            int width = static_cast<int>(graph->Width());
            for (int i = 1; i < width; ++ i) {
                auto applies = PrimitiveFactory::GetPrimitiveApplies(graph);
                applies = PrimitiveFactory::FilterPrimitiveApplies(applies, {
                        PrimitiveGenOptions::ReduceWidth()
                });
                if (applies.empty())
                    break;
                std::cout << " * Reduce kernel graph width by 1" << std::endl;
                auto apply = RandomChoose(applies);
                std::cout << " * Choose " << apply << std::endl;
                graph->Apply(apply);
                ASSERT_EQ(graph->Width(), width - i);
            }
            std::cout << "Final kernel width: " << graph->Width() << std::endl;
            std::cout << graph->Hash() << std::endl;
            break;
        } catch (const CanNotSolveDynamicVar& _) {}
    }
}

#ifdef CANVAS_DEBUG_SEARCH_TEST_PRINT_ALL_ACTIONS
#undef CANVAS_DEBUG_SEARCH_TEST_PRINT_ALL_ACTIONS
#endif
