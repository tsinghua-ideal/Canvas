#include <gtest/gtest.h>

#include "Canvas/CodeGen/DotCodeGen.hpp"
#include "Canvas/CodeGen/PyTorchCodeGen.hpp"
#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Variable.hpp"
#include "Canvas/Impls/Impls.hpp"
#include "Canvas/Primitives/Factory.hpp"
#include "Canvas/Search/RandomSample.hpp"
#include "Canvas/Search/ReceptiveAnalyzer.hpp"


using namespace canvas;

GraphSP RandomGraph(int n_primitives, const PrimitiveOptions& filter=PrimitiveOptions()) {
    auto TryToGenerate = [&]() -> GraphSP {
        auto graph = std::make_shared<Graph>();
        for (int i = 1; i <= n_primitives; ++ i) {
            auto applies = PrimitiveFactory::GetPrimitiveApplies(graph, filter);
            std::cout << "Round " << i << ":" << std::endl;
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
        } catch (const CanNotSolveDynamicVarOnGraph& _) {}
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
    auto graph = RandomGraph(n_primitives, PrimitiveOptions(0));
    std::cout << graph->Hash() << std::endl;
}

TEST(Search, PrimitiveFactoryOnlyFC) {
    int n_primitives = 10;
    auto graph = RandomGraph(n_primitives, PrimitiveOptions("fc"));
    std::cout << graph->Hash() << std::endl;
}

TEST(Search, ReceptiveAnalyzer) {
    auto lka = ImplLKA().graph;
    ASSERT_EQ(ReceptiveAnalyzer::GetReceptiveSize(lka), 17 * 17);
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
                auto applies = PrimitiveFactory::GetPrimitiveApplies(graph, PrimitiveOptions(-1));
                if (applies.empty())
                    break;
                std::cout << " * Reduce kernel graph width by 1" << std::endl;
                auto apply = RandomChoose(applies);
                std::cout << " * Choose " << apply << std::endl;
                graph->Apply(apply);
                ASSERT_EQ(graph->Width(), width - i);
            }
            std::cout << "Final kernel width: " << graph->Width() << std::endl;
            std::cout << "Graph hash:" << graph->Hash() << std::endl;
            break;
        } catch (const CanNotSolveDynamicVarOnGraph& _) {}
    }
}

void TestAPI(const NetSpecsSP& net_specs) {
    // Random and generate code.
    for (int i = 0; i < 30; ++ i) {
        auto options = SampleOptions();
        options.ensure_spatial_invariance = MakeChoice();
        auto solution = RandomSample(net_specs, options);
        std::cout << ConsoleUtils::blue
                  << "# Sample kernel " << i + 1 << ": "
                  << ConsoleUtils::reset << std::endl;
        auto torch_code = PyTorchCodeGen().Gen(solution);
        auto graphviz_code = DotCodeGen().Gen(solution);
        std::cout << torch_code << std::endl;
        std::cout << graphviz_code << std::endl;
        std::cout << std::endl;
    }
}

TEST(Search, RandomSampleAPI) {
    // Create network specifications.
    std::vector<KernelSpecs> kernels;
    kernels.emplace_back(32, 56, 56);
    kernels.emplace_back(64, 28, 28);
    kernels.emplace_back(160, 14, 14);
    kernels.emplace_back(256, 7, 7);
    TestAPI(std::make_shared<NetSpecs>(kernels));
}

TEST(Search, SingleSpatialRandomSampleAPI) {
    // Create network specifications.
    std::vector<KernelSpecs> kernels;
    kernels.emplace_back(32, 56, 1, 1);
    kernels.emplace_back(64, 28, 1, 1);
    kernels.emplace_back(160, 14, 1, 1);
    kernels.emplace_back(256, 7, 1, 1);
    TestAPI(std::make_shared<NetSpecs>(kernels));
}

TEST(Search, NoneSpatialRandomSampleAPI) {
    // Create network specifications.
    std::vector<KernelSpecs> kernels;
    kernels.emplace_back(32, 1, 1, 0);
    kernels.emplace_back(64, 1, 1, 0);
    kernels.emplace_back(160, 1, 1, 0);
    kernels.emplace_back(256, 1, 1, 0);
    TestAPI(std::make_shared<NetSpecs>(kernels));
}

TEST(Search, EmptyRandomSampleAPI) {
    // Test empty kernels.
    TestAPI(std::make_shared<NetSpecs>(std::vector<KernelSpecs>()));
}

#ifdef CANVAS_DEBUG_SEARCH_TEST_PRINT_ALL_ACTIONS
#undef CANVAS_DEBUG_SEARCH_TEST_PRINT_ALL_ACTIONS
#endif
