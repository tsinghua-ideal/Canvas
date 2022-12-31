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

struct PyTorchInitTranslator {
    VarMap& var_map;
    bool ensure_spatial_invariance;

    PyTorchInitTranslator(VarMap& var_map, bool ensure_spatial_invariance):
        var_map(var_map), ensure_spatial_invariance(ensure_spatial_invariance) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

struct PyTorchForwardTranslator {
    VarMap& var_map;
    bool ensure_spatial_invariance;

    PyTorchForwardTranslator(VarMap& var_map, bool ensure_spatial_invariance):
        var_map(var_map), ensure_spatial_invariance(ensure_spatial_invariance) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

class PyTorchCodeGen: public CodeGen {
public:
    explicit PyTorchCodeGen(): CodeGen("PyTorch") {}

    Code GenImpl(const Solution& solution, std::string name) override;
};

class CanNotApplyPyTorchCodeGen: public ExceptionWithInfo {
public:
    explicit CanNotApplyPyTorchCodeGen(const std::string& reason) {
        std::stringstream ss;
        ss << "Can not apply PyTorch code generation on the graph: " << reason;
        info = ss.str();
    }
};

} // namespace canvas
