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

struct PyTorchNCHWRecorder {
    CodeGen *gen;
    VarMap& var_map;

    bool need_view = false, force_record_os = false;
    std::string reference;
    TensorSP out;

    PyTorchNCHWRecorder(CodeGen *gen, VarMap& var_map, const TensorSP& in, const TensorSP& out,
                        bool record_os, bool force_record_os=false);

    void GenCopyShapeCode();
};

struct PyTorchInitTranslator {
    VarMap& var_map;

    explicit PyTorchInitTranslator(VarMap& var_map): var_map(var_map) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

struct PyTorchForwardTranslator {
    VarMap& var_map;

    explicit PyTorchForwardTranslator(VarMap& var_map): var_map(var_map) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

class PyTorchCodeGen: public CodeGen {
public:
    explicit PyTorchCodeGen(): CodeGen("PyTorch") {}

    Code GenImpl(const Solution& solution, std::string name) override;

    [[nodiscard]] static PrimitiveGenOptions SupportedOptions() {
        return PrimitiveGenOptions::Unlimited();
    }
};

class CanNotApplyPyTorchCodeGen: public ExceptionWithInfo {
public:
    explicit CanNotApplyPyTorchCodeGen(const std::string& reason) {
        std::stringstream ss;
        ss << "Can not apply PyTorch code generation on the graph: " << reason;
        info = ss.str();
    }
};

} // End namespace canvas
