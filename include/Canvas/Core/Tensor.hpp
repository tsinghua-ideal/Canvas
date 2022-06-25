#pragma once

#include <boost/container/small_vector.hpp>
#include <memory>
#include <utility>

#include "Canvas/Core/Shape.hpp"
#include "Canvas/Core/Tensor.hpp"
#include "Canvas/Utils/Common.hpp"


namespace canvas {

namespace bc = boost::container;

struct Primitive;
typedef std::shared_ptr<Primitive> PrimitiveSP;

struct Tensor {
    typedef std::vector<PrimitiveSP> PrimitiveArray;

    int id = kInvalidIndex;
    Shape shape;

    static int num_deconstruction;

    // Structures in graph (only after applying in graph, the links will be built).
    PrimitiveSP producer;
    PrimitiveArray consumers;

    explicit Tensor(const Shape& shape, PrimitiveSP producer=nullptr):
        shape(shape), producer(std::move(producer)) {}

    Tensor(const Tensor& rhs) = default;

    ~Tensor() { ++ num_deconstruction; }
};

typedef std::shared_ptr<Tensor> TensorSP;

} // namespace canvas
