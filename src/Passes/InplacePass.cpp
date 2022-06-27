#include "Canvas/Passes/InplacePass.hpp"


namespace canvas {

GraphSP InplacePass::Optimize(const GraphSP& graph) {
    return graph;
}

}
