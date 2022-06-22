#include "Canvas/Core/Shape.hpp"


namespace canvas {

std::ostream& operator << (std::ostream& os, const Shape& rhs) {
    os << "[";
    bool displayed = false;
    for (const auto& dim: rhs.Continuous()) {
        if (displayed)
            os << ", ";
        displayed = true, os << dim;
    }
    if (not displayed)
        os << "Scalar";
    return os << "]";
}

} // namespace canvas
