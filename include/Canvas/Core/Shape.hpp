#pragma once

#include <algorithm>
#include <initializer_list>

#include "Canvas/Core/Variable.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

struct Shape {
    static constexpr int kShapeMaxDim = 6;

    /// Shape dim position indices.
    enum DimPos {
        PG = 0,     // Groups.
        PC = 1,     // Channels.
        PKH = 2,    // Kernel height.
        PKW = 3,    // Kernel width.
        PH = 4,     // Height.
        PW = 5,     // Width.
    };

    struct ShapeSpecs {
        int dims[kShapeMaxDim] = {};

        [[nodiscard]] int& G() { return dims[DimPos::PG]; }
        [[nodiscard]] int& C() { return dims[DimPos::PC]; }
        [[nodiscard]] int& KH() { return dims[DimPos::PKH]; }
        [[nodiscard]] int& KW() { return dims[DimPos::PKW]; }
        [[nodiscard]] int& H() { return dims[DimPos::PH]; }
        [[nodiscard]] int& W() { return dims[DimPos::PW]; }

        [[nodiscard]] int Pi() const {
            int pi = 1;
            for (const auto& dim: dims)
                pi *= dim;
            assert(pi > 0);
            return pi;
        }
    };

    Variable dims[kShapeMaxDim] = {};

    Shape() = default;

    Shape(const Shape& rhs) = default;

    [[nodiscard]] Variable& G() { return dims[DimPos::PG]; }
    [[nodiscard]] Variable& C() { return dims[DimPos::PC]; }
    [[nodiscard]] Variable& KH() { return dims[DimPos::PKH]; }
    [[nodiscard]] Variable& KW() { return dims[DimPos::PKW]; }
    [[nodiscard]] Variable& H() { return dims[DimPos::PH]; }
    [[nodiscard]] Variable& W() { return dims[DimPos::PW]; }

    [[nodiscard]] Variable& Get(const DimPos& i) { return dims[i]; }
    [[nodiscard]] Variable& Get(const int& i) { return dims[i]; }

    [[nodiscard]] ShapeSpecs FillToStaticShape(const Variable::VarSpecs& specs) {
        ShapeSpecs static_shape;
        for (int i = 0; i < kShapeMaxDim; ++ i) {
            int static_value = dims[i].FillToInteger(specs);
            assert(static_value > 0);
            static_shape.dims[i] = static_value;
        }
        return static_shape;
    }

    [[nodiscard]] int GetRelativePos(const DimPos& i, bool backward=false) {
        int count = 0;
        if (not backward) {
            for (int k = 0; k < i; ++ k)
                if (not dims[k].Empty())
                    ++ count;
        } else {
            for (int k = i + 1; k < kShapeMaxDim; ++ k)
                if (not dims[k].Empty())
                    ++ count;
            count = -count - 1;
        }
        return count;
    }

    [[nodiscard]] static Shape StandardCHW() {
        Shape s;
        s.C() = StaticVarPos::VC, s.H() = StaticVarPos::VH, s.W() = StaticVarPos::VW;
        return s;
    }

    [[nodiscard]] Variable CKK() const {
        return dims[DimPos::PC] * dims[DimPos::PKH] * dims[DimPos::PKW];
    }

    [[nodiscard]] Variable GCKK() const {
        return dims[DimPos::PG] * dims[DimPos::PC] * dims[DimPos::PKH] * dims[DimPos::PKW];
    }

    [[nodiscard]] Variable Pi() const {
        Variable pi;
        for (const auto& dim: dims)
            pi *= dim;
        return pi;
    }

    void SolveDynamicVar(const VarSolution& s) {
        for (auto& dim: dims)
            dim.SolveDynamicVar(s);
    }

    [[nodiscard]] bool CouldBeReshapeToCHW() const {
        return Pi() == Variable::Compose({StaticVarPos::VC, StaticVarPos::VH, StaticVarPos::VW});
    }

    [[nodiscard]] bool SatisfyAssumption() const {
        return std::all_of(dims, dims + kShapeMaxDim, [](const Variable& dim) -> bool {
            return dim.SatisfyAssumption();
        });
    }

    [[nodiscard]] bool IsAllStatic() const {
        return std::all_of(dims, dims + kShapeMaxDim, [](const Variable& dim) -> bool {
            return dim.IsStatic();
        });
    }

    [[nodiscard]] std::vector<Variable> Continuous() {
        std::vector<Variable> array;
        for (const auto& dim: dims)
            if (not dim.Empty())
                array.push_back(dim);
        return array;
    }

    [[nodiscard]] size_t Hash() const {
        size_t value = 0;
        for (const auto& dim: dims)
            value = IterateHash(value, dim.Hash());
        return value;
    }

    [[nodiscard]] bool Empty() const {
        return std::none_of(dims, dims + kShapeMaxDim, [](const Variable& d) -> bool {
            return not d.Empty();
        });
    }

    [[nodiscard]] bool operator == (const Shape& rhs) const {
        for (int i = 0; i < kShapeMaxDim; ++ i)
            if (dims[i] != rhs.dims[i])
                return false;
        return true;
    }

    [[nodiscard]] bool operator != (const Shape& rhs) const {
        return not (*this == rhs);
    }

    friend std::ostream& operator << (std::ostream& os, const Shape& rhs);
};

} // namespace canvas

CanvasHashTemplate(canvas::Shape, .Hash());
