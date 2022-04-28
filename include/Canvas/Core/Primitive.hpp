#pragma once

#include <boost/container/small_vector.hpp>
#include <boost/range/join.hpp>
#include <utility>
#include <vector>

#include "Canvas/Core/Preferences.hpp"
#include "Canvas/Core/Tensor.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

namespace bc = boost::container;
namespace br = boost::range;

struct Primitive {
    static constexpr int kPreservedNumLinks = 2;

#ifdef CANVAS_ENABLE_BOOST_STYLE_SMALL_VECTOR
    typedef bc::small_vector<TensorSP, kPreservedNumLinks> TensorArray;
#else
    typedef std::vector<TensorSP> TensorArray;
#endif

    static int num_deconstruction;

    int id = kInvalidIndex;
    bool input_commutative; // Outputs must be not commutative
    std::string name;
    TensorArray ins, outs;

    explicit Primitive(std::string name,
              const std::initializer_list<TensorSP>& ins={},
              bool input_commutative=false):
        name(std::move(name)), ins(ins),
        input_commutative(input_commutative) {
        for (const auto& t: br::join(ins, outs))
            assert(t);
    }

    Primitive(const Primitive& rhs) = default;

    ~Primitive() { ++ num_deconstruction; }

    virtual void AmplifyIntermediateVariables() {}

    /// Note the returned values are not references but copies
    [[nodiscard]] virtual std::vector<Variable> IntermediateVariables() const { return {}; }

    [[nodiscard]] virtual size_t PsCount(const Variable::StaticSpecs& specs,
                                         const Variable::DynamicFills& fills) const {
        return 0;
    }

    [[nodiscard]] virtual size_t FLOPsCount(const Variable::StaticSpecs& specs,
                                            const Variable::DynamicFills& fills) const {
        return 0;
    }

    /// Only solve the special variables in the primitive (inputs and outputs' shape not included)
    virtual void SolveDynamicVar(const VarSolution& s) {}

    [[nodiscard]] virtual size_t Hash(bool ignore_outs) const;

    [[nodiscard]] virtual PrimitiveSP Copy() const = 0;

    friend std::ostream& operator << (std::ostream& os, const Primitive& rhs);
};

struct PrimitiveApply {
    PrimitiveSP primitive;
    std::optional<VarSolution> solution;

    explicit PrimitiveApply(PrimitiveSP p): primitive(std::move(p)), solution(std::nullopt) { assert(this->primitive); }

    PrimitiveApply(PrimitiveSP p, const VarSolution& solution):
            primitive(std::move(p)), solution(solution) { assert(this->primitive); }

    PrimitiveApply(PrimitiveSP p, std::optional<VarSolution> solution):
            primitive(std::move(p)), solution(solution) { assert(this->primitive); }

    friend std::ostream& operator << (std::ostream& os, const PrimitiveApply& apply) {
        os << "[Apply " << apply.primitive->name << ": " << *apply.primitive;
        if (apply.solution.has_value())
            os << " (x_" << apply.solution.value().index << " = " << apply.solution.value().substitution << ")";
        return os << "]";
    }
};

class CanNotApplyPrimitive: public ExceptionWithInfo {
public:
    explicit CanNotApplyPrimitive(const std::string& name="Unknown") {
        info = "Can not apply \"" + name + "\" primitive on tensors";
    }
};

} // End namespace canvas

#define CanvasPrimitiveCopyTemplate(Type)  Type(const Type& rhs) = default; \
[[nodiscard]] PrimitiveSP Copy() const override { \
    return std::make_shared<Type>(*this); \
}