#include "Canvas/Kernels/Convolution.hpp"
#include "Canvas/Primitives/Factory.hpp"


namespace canvas {

GraphSP CreateConvolutionGraph() {
    auto graph = std::make_shared<Graph>();

    auto unfold = std::make_shared<UnfoldPrimitive>(graph->in);
    graph->Apply(unfold);

    auto fc = std::make_shared<FCPrimitive>(unfold->outs[0], Variable(StaticVar::VC));
    graph->Apply(fc);

    graph->ApplyOutput();
    return graph;
}

GraphSP CreateDepthWiseConvolutionGraph() {
    auto graph = std::make_shared<Graph>();

    auto group = std::make_shared<GroupPrimitive>(graph->in, GroupAllChannels);
    graph->Apply(group);

    auto unfold = std::make_shared<UnfoldPrimitive>(group->outs[0]);
    graph->Apply(unfold);

    auto fc = std::make_shared<FCPrimitive>(unfold->outs[0], Variable(StaticVar::VC));
    graph->Apply(fc);

    graph->ApplyOutput();
    return graph;
}

GraphSP CreateDepthWiseSeparableConvolutionGraph() {
    auto graph = std::make_shared<Graph>();

    auto group = std::make_shared<GroupPrimitive>(graph->in, GroupAllChannels);
    graph->Apply(group);

    auto unfold = std::make_shared<UnfoldPrimitive>(group->outs[0]);
    graph->Apply(unfold);

    auto depth_wise = std::make_shared<FCPrimitive>(unfold->outs[0], Variable(StaticVar::VC));
    graph->Apply(depth_wise);

    auto relu = std::make_shared<ActivationPrimitive>(depth_wise->outs[0]);
    auto norm = std::make_shared<NormPrimitive>(relu->outs[0]);
    graph->Apply(relu); graph->Apply(norm);

    auto point_wise = std::make_shared<FCPrimitive>(norm->outs[0], Variable(StaticVar::VC));
    graph->Apply(point_wise);

    graph->ApplyOutput();
    return graph;
}

GraphSP CreateGroupedConvolutionGraph() {
    auto graph = std::make_shared<Graph>();

    auto unfold = std::make_shared<UnfoldPrimitive>(graph->in);
    graph->Apply(unfold);

    auto group = std::make_shared<GroupPrimitive>(unfold->outs[0]);
    graph->Apply(group);

    auto fc = std::make_shared<FCPrimitive>(group->outs[0], Variable(StaticVar::VC));
    graph->Apply(fc);

    graph->ApplyOutput();
    return graph;
}

GraphSP CreateInvolutionGraph() {
    auto graph = std::make_shared<Graph>();

    auto fc_1 = std::make_shared<FCPrimitive>(graph->in, StaticVar::VC / StaticVar::VR);
    graph->Apply(fc_1);

    // Batch normalization seems bad for Involution
    // auto norm = std::make_shared<NormPrimitive>(fc_1->outs[0]);
    // graph->Apply(norm);

    auto relu = std::make_shared<ActivationPrimitive>(fc_1->outs[0]);
    graph->Apply(relu);

    auto fc_2 = std::make_shared<FCPrimitive>(relu->outs[0],
                                              Variable({StaticVar::VG, StaticVar::VKH, StaticVar::VKW}));
    graph->Apply(fc_2);

    auto unfold = std::make_shared<UnfoldPrimitive>(graph->in);
    graph->Apply(unfold);

    auto group_1 = std::make_shared<GroupPrimitive>(unfold->outs[0]);
    graph->Apply(group_1);

    auto group_2 = std::make_shared<GroupPrimitive>(fc_2->outs[0]);
    graph->Apply(group_2);

    auto bmul = std::make_shared<BroadcastPrimitive>(group_2->outs[0], group_1->outs[0], BMul);
    graph->Apply(bmul);

    auto fold = std::make_shared<FoldPrimitive>(bmul->outs[0]);
    graph->Apply(fold);

    graph->ApplyOutput();
    return graph;
}

GraphSP CreateSEConvolutionGraph() {
    auto graph = std::make_shared<Graph>();

    auto pool = std::make_shared<PoolPrimitive>(graph->in);
    graph->Apply(pool);

    auto fc_1 = std::make_shared<FCPrimitive>(pool->outs[0], StaticVar::VC / StaticVar::VR);
    graph->Apply(fc_1);

    auto relu = std::make_shared<ActivationPrimitive>(fc_1->outs[0]);
    graph->Apply(relu);

    auto fc_2 = std::make_shared<FCPrimitive>(relu->outs[0], Variable(StaticVar::VC));
    graph->Apply(fc_2);

    auto sigmoid = std::make_shared<ActivationPrimitive>(fc_2->outs[0], Sigmoid);
    graph->Apply(sigmoid);

    auto bmul = std::make_shared<BroadcastPrimitive>(sigmoid->outs[0], graph->in, BMul);
    graph->Apply(bmul);

    graph->ApplyOutput();
    return graph;
}

} // End namespace canvas
