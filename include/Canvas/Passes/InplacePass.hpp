#pragma once

#include "Canvas/Core/Pass.hpp"


namespace canvas {

class InplacePass: public Pass {
public:
    InplacePass() = default;

    [[nodiscard]] GraphSP Optimize(const GraphSP &graph) final;
};

} // namespace canvas
