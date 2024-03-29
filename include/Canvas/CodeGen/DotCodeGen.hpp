#pragma once

#include <string>
#include <utility>

#include "Canvas/CodeGen/CodeGen.hpp"
#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Primitives/Factory.hpp"
#include "Canvas/Utils/Common.hpp"
#include "Canvas/Utils/Indent.hpp"


namespace canvas {

class DotCodeGen: public CodeGen {
public:
    explicit DotCodeGen(): CodeGen("Dot") {}

    Code GenImpl(const Solution& solution, std::string name) override;
};

} // namespace canvas
