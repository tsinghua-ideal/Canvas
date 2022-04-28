#include "Canvas/Core/Primitive.hpp"


namespace canvas {

int Primitive::num_deconstruction = 0;

size_t Primitive::Hash(bool ignore_outs) const {
    auto hash_value = HashStr(name);
    auto IterateVecHash = [&hash_value](const TensorArray& vec, bool commutative) {
        bc::small_vector<size_t, kPreservedNumLinks> values;
        values.reserve(vec.size());
        for (const auto& t: vec)
            values.push_back(t->id); // Use `tensor->id` for topology filtering
        if (commutative)
            std::sort(values.begin(), values.end());
        for (const auto& v: values)
            hash_value = IterateHash(hash_value, v);
    };
    IterateVecHash(ins, input_commutative);
    if (not ignore_outs)
        IterateVecHash(outs, false);
    return hash_value;
}

std::ostream& operator << (std::ostream& os, const Primitive& rhs) {
    auto Print = [&os](const auto& vec) -> std::ostream& {
        os << "[";
        bool displayed = false;
        for (const auto& t: vec) {
            if (displayed)
                os << ", ";
            displayed = true, os << "T" << t->id;
        }
        return os << "]";
    };
    Print(rhs.ins);
    os << " => ";
    return Print(rhs.outs);
}

} // End namespace canvas
