#pragma once

#include <algorithm>
#include <initializer_list>
#include <utility>

#include "Canvas/Core/Variable.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

struct MetaDims;
struct ChannelDims;
struct SpatialDims;

typedef std::shared_ptr<MetaDims> MetaDimsSP;

struct MetaDims {
    static constexpr int kMaxMetaDims = 4;

    Variable dims[kMaxMetaDims];

    MetaDims() = default;

    [[nodiscard]] virtual std::string DimPosToName(int pos) const = 0;

    [[nodiscard]] bool IsStatic() const {
        return std::all_of(dims, dims + kMaxMetaDims, [](const Variable& dim) -> bool {
            return dim.IsStatic();
        });
    }

    [[nodiscard]] bool Empty() const {
        return std::none_of(dims, dims + kMaxMetaDims, [](const Variable& d) -> bool {
            return not d.Empty();
        });
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

    [[nodiscard]] bool operator == (const MetaDims& rhs) const {
        for (int i = 0; i < kMaxMetaDims; ++ i)
            if (dims[i] != rhs.dims[i])
                return false;
        return true;
    }
};

struct ChannelDims: MetaDims {
    static constexpr int kMaxChannelDims = 4;

    enum ChannelDimPos {
        PG = 0,     // Groups.
        PC = 1,     // Channels.
        PKH = 2,    // Kernel height.
        PKW = 3,    // Kernel width.
    };

    ChannelDims() = default;

    [[nodiscard]] static MetaDimsSP StandardC() {
        auto meta_dims = std::make_shared<ChannelDims>();
        meta_dims->dims[PC] = Variable::StaticVar(StaticVarPos::VC);
        return meta_dims;
    }

    [[nodiscard]] std::string DimPosToName(int pos) const final {
        static_assert(kMaxChannelDims == 4);
        switch (pos) {
            case PG: return "G";
            case PC: return "C";
            case PKH: return "KH";
            case PKW: return "KW";
            default: Unreachable();
        }
        Unreachable();
    }
};

struct SpatialDims: MetaDims {
    static constexpr int kMaxSpatialDims = 2;

    enum SpatialDimPos {
        PH = 0,    // Image height.
        PW = 1,    // Image width.
    };

    SpatialDims() = default;

    [[nodiscard]] static MetaDimsSP StandardHW() {
        auto meta_dims = std::make_shared<SpatialDims>();
        meta_dims->dims[PH] = Variable::StaticVar(StaticVarPos::VH);
        meta_dims->dims[PW] = Variable::StaticVar(StaticVarPos::VW);
        return meta_dims;
    }

    [[nodiscard]] std::string DimPosToName(int pos) const final {
        static_assert(kMaxSpatialDims == 2);
        switch (pos) {
            case PH: return "H";
            case PW: return "W";
            default: Unreachable();
        }
        Unreachable();
    }
};

struct Shape {
    struct ShapeSpecs {
        std::vector<int> dims;

        explicit ShapeSpecs(std::vector<int> dims): dims(std::move(dims)) {}

        [[nodiscard]] int Pi() const {
            int pi = 1;
            for (const auto& dim: dims)
                pi *= dim;
            return pi;
        }

        [[nodiscard]] bool IsValid() const {
            return std::all_of(dims.begin(), dims.end(), [](int x) -> bool { return x > 0; });
        }
    };

    MetaDimsSP dims[2];

    Shape() = default;

    Shape(const Shape& rhs) = default;

    Shape(const MetaDimsSP& first, const MetaDimsSP& second) {
        assert(first and second);
        dims[0] = first, dims[1] = second;
    }

    [[nodiscard]] ShapeSpecs FillToStaticShape(const Variable::VarSpecs& specs) const {
        std::vector<int> dim_specs;
        for (const auto& dim: Continuous())
            dim_specs.push_back(dim.FillToInteger(specs));
        return ShapeSpecs(dim_specs);
    }

    [[nodiscard]] static Shape StandardCHW() {
        return {ChannelDims::StandardC(), SpatialDims::StandardHW()};
    }

    [[nodiscard]] bool IsStatic() const {
        return dims[0]->IsStatic() and dims[1]->IsStatic();
    }

    [[nodiscard]] bool Empty() const {
        return dims[0]->Empty() and dims[1]->Empty();
    }

    [[nodiscard]] Variable Pi() const {
        return dims[0]->Pi() * dims[1]->Pi();
    }

    [[nodiscard]] bool CouldBeReshapedToCHW() const {
        return Pi() == Variable::Compose({StaticVarPos::VC, StaticVarPos::VH, StaticVarPos::VW});
    }

    void SolveDynamicVar(const VarSolution& s) {
        dims[0]->SolveDynamicVar(s);
        dims[1]->SolveDynamicVar(s);
    }

    [[nodiscard]] std::vector<Variable> Continuous() const {
        return Merge(dims[0]->Continuous(), dims[1]->Continuous());
    }

    [[nodiscard]] size_t Hash() const {
        return IterateHash(dims[0]->Hash(), dims[1]->Hash());
    }

    [[nodiscard]] bool operator == (const Shape& rhs) const {
        return *dims[0] == *rhs.dims[0] and *dims[1] == *rhs.dims[1];
    }

    [[nodiscard]] bool operator != (const Shape& rhs) const {
        return not (*this == rhs);
    }

    friend std::ostream& operator << (std::ostream& os, const Shape& rhs);
};

} // namespace canvas

CanvasHashTemplate(canvas::Shape, .Hash());
