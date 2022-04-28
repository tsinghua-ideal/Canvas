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

struct TVMReshapeOptimizationStorage: OptimizationStorage {
    std::set<TensorSP> has_non_reshape_usage;
    std::map<TensorSP, Shape> actual_shape;
};

typedef std::shared_ptr<TVMReshapeOptimizationStorage> TVMReshapeOptimizationStorageSP;

struct TVMReshapeOptimizationTranslator {
    static constexpr int kOptIndex = 0;

    VarMap& var_map;

    explicit TVMReshapeOptimizationTranslator(VarMap& var_map): var_map(var_map) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

struct TVMActualShapeInferenceTranslator {
    static constexpr int kOptIndex = 0;

    VarMap& var_map;

    explicit TVMActualShapeInferenceTranslator(VarMap& var_map): var_map(var_map) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

struct TVMPlaceholderTranslator {
    VarMap& var_map;

    explicit TVMPlaceholderTranslator(VarMap& var_map): var_map(var_map) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

struct TVMOperatorTranslator {
    VarMap& var_map;

    explicit TVMOperatorTranslator(VarMap& var_map): var_map(var_map) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

struct TVMReturnTranslator {
    VarMap& var_map;

    explicit TVMReturnTranslator(VarMap& var_map): var_map(var_map) {}

    void operator () (CodeGen* gen, const PrimitiveSP& p);
};

class TVMCodeGen: public CodeGen {
public:
    explicit TVMCodeGen(): CodeGen("TVM") {}

    Code GenImpl(const Solution& solution, std::string name) override;

    [[nodiscard]] static PrimitiveGenOptions SupportedOptions() {
        auto options = PrimitiveGenOptions::None();
        options.allow_dynamic_variables = true;
        options.force_irregular = false;
        options.dot = true;
        options.group_by_factor = true, options.group_all = true;
        options.fc = true, options.unfold_hw = true, options.unfold_h = true, options.unfold_w = true;
        options.fold_avg = true, options.fold_max = true;
        options.fold_h = true, options.fold_w = true, options.fold_hw = true;
        options.pool_h = true, options.pool_w = true, options.pool_hw = true;
        options.shift_h = options.shift_w = options.shift_hw = true;
        options.relu = true, options.tanh = true, options.sigmoid = true;
        options.exp = true; options.abs = true; options.neg = true; options.sin = true;
        options.softmax_c = true, options.softmax_h = true, options.softmax_w = true, options.softmax_hw = true;
        options.b_add = true, options.b_sub = true, options.b_mul = true;
        return options;
    }
};

class CanNotApplyTVMCodeGen: public ExceptionWithInfo {
public:
    explicit CanNotApplyTVMCodeGen(const std::string& reason) {
        std::stringstream ss;
        ss << "Can not apply TVM code generation on the graph: " << reason;
        info = ss.str();
    }
};

} // End namespace canvas
