#pragma once

#include <tuple>

#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Variable.hpp"


namespace canvas {

struct FCPrimitive: Primitive {
    bool with_norm = false, with_relu = false;

    explicit FCPrimitive(const TensorSP& t);

    FCPrimitive(const TensorSP& t, const Variable& nc);

    /// This function returns `n_groups`, `ic_per_group`, `oc_per_group` and `map_size`
    [[nodiscard]] std::tuple<size_t, size_t, size_t, size_t>
    GetConcreteSpecs(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const;

    [[nodiscard]] std::vector<Variable> IntermediateVariables() const override;

    [[nodiscard]] size_t PsCount(const Variable::StaticSpecs& specs,
                                 const Variable::DynamicFills& fills=Variable::DynamicFills()) const override;

    [[nodiscard]] size_t FLOPsCount(const Variable::StaticSpecs& specs,
                                    const Variable::DynamicFills& fills=Variable::DynamicFills()) const override;

    CanvasPrimitiveCopyTemplate(FCPrimitive);
};

} // End namespace canvas
