#pragma once

#include <algorithm>
#include <initializer_list>
#include <utility>

#include "Canvas/Core/Variable.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

struct MetaShape;
struct ChannelShape;
struct SpatialShape;

typedef std::shared_ptr<MetaShape> MetaShapeSP;
typedef std::shared_ptr<ChannelShape> ChannelShapeSP;
typedef std::shared_ptr<SpatialShape> SpatialShapeSP;

struct MetaShape {
    static constexpr int kMaxMetaDims = 4;

    Variable dims[kMaxMetaDims];

    MetaShape() = default;

    [[nodiscard]] virtual MetaShapeSP Copy() const = 0;

    [[nodiscard]] virtual std::string IndexToName(int i) const = 0;

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

    [[nodiscard]] bool operator == (const MetaShape& rhs) const {
        for (int i = 0; i < kMaxMetaDims; ++ i)
            if (dims[i] != rhs.dims[i])
                return false;
        return true;
    }
};

struct ChannelShape: MetaShape {
    static constexpr int kMaxChannelDims = 4;

    enum Index {
        PG = 0,     // Groups.
        PC = 1,     // Channels.
        PKH = 2,    // Kernel height.
        PKW = 3,    // Kernel width.
    };

    ChannelShape() = default;

    ChannelShape(const ChannelShape& rhs) = default;

    [[nodiscard]] MetaShapeSP Copy() const final {
        return std::make_shared<ChannelShape>(*this);
    }

    [[nodiscard]] static MetaShapeSP MakeShapeC() {
        auto meta_dims = std::make_shared<ChannelShape>();
        meta_dims->dims[PC] = Variable::StaticVar(StaticVarPos::VC);
        return meta_dims;
    }

    [[nodiscard]] Variable& G() { return dims[Index::PG]; }
    [[nodiscard]] Variable& C() { return dims[Index::PC]; }
    [[nodiscard]] Variable& KH() { return dims[Index::PKH]; }
    [[nodiscard]] Variable& KW() { return dims[Index::PKW]; }

    [[nodiscard]] Variable CKK() const {
        return dims[Index::PC] * dims[Index::PKH] * dims[Index::PKW];
    }

    [[nodiscard]] std::string IndexToName(int i) const final {
        static_assert(kMaxChannelDims == 4);
        switch (i) {
            case PG: return "G";
            case PC: return "C";
            case PKH: return "KH";
            case PKW: return "KW";
            default: assert(false);
        }
        Unreachable();
    }
};

struct SpatialShape: MetaShape {
    static constexpr int kMaxSpatialDims = 2;

    enum Index {
        PH = 0,    // Image height.
        PW = 1,    // Image width.
    };

    SpatialShape() = default;

    SpatialShape(const SpatialShape& rhs) = default;

    [[nodiscard]] MetaShapeSP Copy() const final {
        return std::make_shared<SpatialShape>(*this);
    }

    [[nodiscard]] static MetaShapeSP MakeShapeHW() {
        auto meta_dims = std::make_shared<SpatialShape>();
        meta_dims->dims[PH] = Variable::StaticVar(StaticVarPos::VH);
        meta_dims->dims[PW] = Variable::StaticVar(StaticVarPos::VW);
        return meta_dims;
    }

    [[nodiscard]] Variable& H() { return dims[Index::PH]; }
    [[nodiscard]] Variable& W() { return dims[Index::PW]; }

    [[nodiscard]] std::string IndexToName(int i) const final {
        static_assert(kMaxSpatialDims == 2);
        switch (i) {
            case PH: return "H";
            case PW: return "W";
            default: assert(false);
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

    struct Index {
        const int d, k;

        Index(int d, int k): d(d), k(k) {
            assert(0 <= d and d < 2);
            assert(0 <= k and k < MetaShape::kMaxMetaDims);
        }
    };

    MetaShapeSP dims[2];

    Shape(const Shape& rhs) {
        dims[0] = rhs.dims[0]->Copy();
        dims[1] = rhs.dims[1]->Copy();
    }

    Shape(const MetaShapeSP& first, const MetaShapeSP& second) {
        assert(first and second);
        assert(not (DynamicCast<SpatialShape>(first) and DynamicCast<ChannelShape>(second)));
        dims[0] = first, dims[1] = second;
    }

    [[nodiscard]] static Shape MakeChannelSpatial() {
        return {std::make_shared<ChannelShape>(), std::make_shared<SpatialShape>()};
    }

    [[nodiscard]] static Shape MakeShapeCHW() {
        return {ChannelShape::MakeShapeC(), SpatialShape::MakeShapeHW()};
    }

    [[nodiscard]] bool IsChannelSpatial() const {
        return DynamicCast<ChannelShape>(dims[0]) and DynamicCast<SpatialShape>(dims[1]);
    }

    [[nodiscard]] ChannelShapeSP Channel() {
        assert(IsChannelSpatial());
        return DynamicCast<ChannelShape>(dims[0]);
    }

    [[nodiscard]] SpatialShapeSP Spatial() {
        assert(IsChannelSpatial());
        return DynamicCast<SpatialShape>(dims[1]);
    }

    [[nodiscard]] std::pair<ChannelShapeSP, SpatialShapeSP> ChannelSpatial() {
        assert(IsChannelSpatial());
        return {DynamicCast<ChannelShape>(dims[0]), DynamicCast<SpatialShape>(dims[1])};
    }

    [[nodiscard]] ShapeSpecs FillToStaticShape(const Variable::VarSpecs& specs) const {
        std::vector<int> dim_specs;
        for (const auto& dim: Continuous())
            dim_specs.push_back(dim.FillToInteger(specs));
        return ShapeSpecs(dim_specs);
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

    [[nodiscard]] std::string IndexToName(const Index& index) const {
        return dims[index.d]->IndexToName(index.k);
    }

    Variable& operator [] (const Index& index) {
        assert(dims[0] and dims[1]);
        return dims[index.d]->dims[index.k];
    }
};

} // namespace canvas

CanvasHashTemplate(canvas::Shape, .Hash());
