#pragma once

#include <string>
#include <utility>

#include "Canvas/CodeGen/CodeGen.hpp"
#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Primitives/Factory.hpp"
#include "Canvas/Utils/Common.hpp"
#include "Canvas/Utils/Format.hpp"


namespace canvas {

class VarsCodeGen: public CodeGen {
public:
    explicit VarsCodeGen(): CodeGen("Vars") {}

    Code GenImpl(const Solution& solution, std::string name) override;

    [[nodiscard]] static PrimitiveGenOptions SupportedOptions() {
        return PrimitiveGenOptions::Unlimited();
    }
};

} // End namespace canvas
