#include <cassert>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Primitives/Broadcast.hpp"
#include "Canvas/Primitives/Dot.hpp"
#include "Canvas/Primitives/FC.hpp"
#include "Canvas/Primitives/Input.hpp"
#include "Canvas/Primitives/Output.hpp"
#include "Canvas/Utils/Common.hpp"
#include "Canvas/Utils/Format.hpp"


namespace canvas {

Graph::Graph(const Graph& rhs) {
    // Copy hash
    hash_cached = rhs.hash_cached;
    hash_value = rhs.hash_value;

    // Copy every tensor
    tensors = rhs.tensors;
    for (auto &t: tensors) {
        t = std::make_shared<Tensor>(*t);
        assert(t->id != kInvalidIndex);
    }
    in = tensors[rhs.in->id];

    // Copy every primitive, change inputs and outputs into new tensors
    primitives = rhs.primitives;
    for (auto &p: primitives) {
        p = p->Copy();
        for (auto& p_t: br::join(p->ins, p->outs))
            p_t = tensors[p_t->id];
    }

    // Change tensors' producers and consumers
    for (const auto& t: tensors) {
        t->producer = primitives[t->producer->id];
        for (auto& p: t->consumers)
            p = primitives[p->id];
    }
}

Graph::~Graph() {
    // Clear one of the links between tensors and primitives
    in = nullptr;
    for (const auto& t: tensors) {
        t->producer = nullptr;
        t->consumers.clear();
    }
}

void Graph::LegalityCheck() const {
    // Topology checks
    std::set<TensorSP> t_set(tensors.begin(), tensors.end());
    std::set<PrimitiveSP> p_set(primitives.begin(), primitives.end());
    assert(t_set.size() == tensors.size());
    assert(p_set.size() == primitives.size());
    for (const auto& t: tensors) {
        assert(p_set.count(t->producer));
        for (const auto& p: t->consumers)
            assert(p_set.count(p));
    }
    for (const auto& p: primitives)
        for (const auto& t: br::join(p->ins, p->outs))
            assert(t_set.count(t));

    // Shape checks
    for (const auto& t: tensors) {
        auto& shape = t->shape;
        // G
        assert(shape.G().SatisfyAssumption());
        // C
        assert(shape.C().SatisfyAssumption());
        assert(not shape.C().Amplified());
        // KH, KW, H, W
        assert(shape.KH().IsStaticInteger());
        assert(not shape.KH().Amplified());
        assert(shape.KW().IsStaticInteger());
        assert(not shape.KW().Amplified());
        assert(shape.H() == Variable(StaticVar::VH) or shape.H().Empty());
        assert(not shape.H().Amplified());
        assert(shape.W() == Variable(StaticVar::VW) or shape.W().Empty());
        assert(not shape.W().Amplified());
    }
}

bool Graph::AlgebraCheck(const Variable::StaticSpecs& specs, const Variable::DynamicFills& fills) const {
    // Get all parameters
    size_t ic = specs.ic, oc = specs.oc;
    auto k = specs.k, h = specs.h, w = specs.w;
    size_t s = specs.s, g = specs.g, r = specs.r;
    assert(h % s == 0 and w % s == 0);
    assert(ic > 0 and oc > 0 and k > 0 and h > 0 and w > 0 and g > 0 and r > 0 and s > 0);
    assert(std::gcd(ic, oc) == std::min(ic, oc));

    // Run dynamic variable substitutions
    for (const auto& t: tensors)
        assert(t->shape.CouldBeFilledToInteger(specs, fills));
    for (const auto& p: primitives)
        for (const auto& var: p->IntermediateVariables())
            assert(var.FillToInteger(specs, fills) != 0);
    return true;
}

bool Graph::IsTopologicalFinished() const {
    auto out = Out();
    if (not out)
        return false;
    int output_count = 0;
    for (const auto& p: primitives)
        output_count += (DynamicCast<OutputPrimitive>(p) != nullptr);
    return output_count == 1
        and DynamicCast<OutputPrimitive>(out->producer) != nullptr
        and out->shape == Shape::StandardACHW(); // `ACHW` means that the amplifier rewriting pass is done
}

int Graph::DynamicVarCount() const {
    bool used[Variable::kDynamicVarCount] = {false};
    for (const auto& t: tensors)
        for (const auto& dim: t->shape.dims)
            for (int i = 0; i < Variable::kDynamicVarCount; ++ i)
                if (dim.dynamic_power[i] != 0)
                    used[i] = true;
    int count = 0;
    for (const auto& u: used)
        count += static_cast<int>(u);
    return count;
}

std::optional<int> Graph::NextUnusedDynamicVarIndex() const {
    bool used[Variable::kDynamicVarCount] = {false};
    for (const auto& t: tensors)
        for (const auto& dim: t->shape.dims)
            for (int i = 0; i < Variable::kDynamicVarCount; ++ i)
                if (dim.dynamic_power[i] != 0)
                    used[i] = true;
    for (int i = 0; i < Variable::kDynamicVarCount; ++ i)
        if (not used[i])
            return i;
    return std::nullopt;
}

TensorSP Graph::Out() const {
    TensorSP out = nullptr;
    for (const auto& t: tensors) {
        if (t->consumers.empty()) {
            if (out) // Only allow one output
                return nullptr;
            out = t;
        }
    }
    return out;
}

Graph::TensorArray Graph::Outs() const {
    TensorArray outs;
    for (const auto& t: tensors)
        if (t->consumers.empty())
            outs.push_back(t);
    return outs;
}

int Graph::Width() const {
    int n = 0;
    for (const auto& t: tensors)
        n += t->consumers.empty();
    return n;
}

size_t Graph::CalculateSubgraphHash(const TensorSP& t, SizeTArray& cache) {
    assert(t);
    if (cache[t->id])
        return cache[t->id];

    // Get all Hash of the consumer primitives
    SizeTArray consumers_hash;
    consumers_hash.reserve(t->consumers.size());
    for (const auto& p: t->consumers) {
        // Type Hash of the primitive
        size_t p_hash = HashStr(p->name);
        // Tensors must be in some order (guaranteed by the primitive implement)
        for (const auto& out_t: p->outs)
            p_hash = IterateHash(p_hash, CalculateSubgraphHash(out_t, cache));
        consumers_hash.push_back(p_hash);
    }

    // Sort them by Hash to discover isomorphism
    std::sort(consumers_hash.begin(), consumers_hash.end());

    // Calculate this subgraph, use shape Hash to discover isomorphism
    size_t subgraph_hash = t->shape.Hash();
    for (const auto& h: consumers_hash)
        subgraph_hash = IterateHash(subgraph_hash, h);
    return cache[t->id] = subgraph_hash;
}

size_t Graph::CalculateHash() {
    // For a very small possibility, the hash value does not equal to zero
    SizeTArray cache(tensors.size(), 0);
    hash_cached = true;
    return hash_value = CalculateSubgraphHash(in, cache);
}

Variable::DynamicFills Graph::GetMinimumFills(const Variable::StaticSpecs& specs) {
    Variable::DynamicFills minimum_fills;
    for (const auto& t: tensors)
        t->shape.UpdateMinimumFills(minimum_fills, specs);
    for (const auto& p: primitives)
        for (const auto& v: p->IntermediateVariables())
            v.UpdateMinimumFills(minimum_fills, specs);
    return minimum_fills;
}

void Graph::Apply(const PrimitiveSP& p) {
    hash_cached = false;
    for (const auto& t: p->ins)
        t->consumers.push_back(p);
    for (const auto& t: p->outs) {
        t->id = static_cast<int>(tensors.size());
        t->producer = p;
        tensors.push_back(t);
    }
    p->id = static_cast<int>(primitives.size());
    primitives.push_back(p);
    LegalityCheck();
}

void Graph::Apply(const PrimitiveApply& pa) {
    Apply(pa.primitive);
    if (pa.solution.has_value())
        SolveDynamicVar(pa.solution.value());
}

void Graph::ApplyOutput() {
    // Apply output primitive
    auto out = Out();
    assert(out and DynamicCast<OutputPrimitive>(out->producer) == nullptr);
    Apply(std::make_shared<OutputPrimitive>(out));

    // Rewrite with amplifier
    for (const auto& t: tensors)
        t->shape.G() *= StaticVar::VA;
    for (const auto& p: primitives)
        p->AmplifyIntermediateVariables();
}

PrimitiveSP Graph::RemapPrimitive(const PrimitiveSP& p) const {
    auto copy = p->Copy();
    for (auto& t: copy->ins)
        t = tensors[t->id];
    for (auto& t: copy->outs)
        t = std::make_shared<Tensor>(*t);
    return copy;
}

std::pair<GraphSP, PrimitiveSP> Graph::CopyAndApply(const PrimitiveSP& p) const {
    auto copy = Copy();
    auto remapped = copy->RemapPrimitive(p);
    copy->Apply(remapped);
    return {copy, remapped};
}

std::pair<GraphSP, PrimitiveApply> Graph::CopyAndApply(const PrimitiveApply& pa) const {
    auto copy = Copy();
    auto remapped = copy->RemapPrimitive(pa.primitive);
    copy->Apply(pa);
    return {copy, PrimitiveApply(remapped, pa.solution)};
}

void Graph::SolveDynamicVar(const VarSolution& s) {
    for (const auto &t: tensors) {
        t->shape.SolveDynamicVar(s);
        for (const auto& dim: t->shape.dims)
            if (not dim.MaybeInteger())
                throw CanNotSolveDynamicVar(s);
    }
    for (const auto& p: primitives) {
        p->SolveDynamicVar(s);
        for (const auto& dim: p->IntermediateVariables())
            if (not dim.MaybeInteger())
                throw CanNotSolveDynamicVar(s);
    }
}

} // End namespace canvas
