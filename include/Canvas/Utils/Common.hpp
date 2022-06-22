#pragma once

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cassert>
#include <chrono>
#include <exception>
#include <ice-cream.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>


namespace canvas {

static constexpr int kInvalidIndex = -1;

/// Console colors.
class [[maybe_unused]] ConsoleUtils {
public:
    [[maybe_unused]] static constexpr const char *reset  = "\033[0m";
    [[maybe_unused]] static constexpr const char *black  = "\033[30m";
    [[maybe_unused]] static constexpr const char *red    = "\033[31m";
    [[maybe_unused]] static constexpr const char *green  = "\033[32m";
    [[maybe_unused]] static constexpr const char *yellow = "\033[33m";
    [[maybe_unused]] static constexpr const char *blue   = "\033[34m";
    [[maybe_unused]] static constexpr const char *white  = "\033[37m";
    [[maybe_unused]] static constexpr const char *clear  = "\033[2K\r";
};

/// Warning
[[maybe_unused]] static void Warning(const std::string& info) {
    std::cout << ConsoleUtils::green;
    std::cout << "Warning: " << info;
    std::cout << ConsoleUtils::reset << std::endl;
}

/// An unimplemented error raiser
[[noreturn]] [[maybe_unused]] static void UnimplementedImpl(int line, const char *file) {
    std::cerr << ConsoleUtils::red;
    std::cerr << "Unimplemented part at line " << line << " in file " << file;
    std::cerr << ConsoleUtils::reset << std::endl;
    std::exit(EXIT_FAILURE);
}

/// An unimplemented error raiser.
#define Unimplemented() UnimplementedImpl(__LINE__, __FILE__)

/// An unreachable error raiser.
[[noreturn]] [[maybe_unused]] static void UnreachableImpl(int line, const char *file) {
    std::cerr << ConsoleUtils::red;
    std::cerr << "Unreachable part at line " << line << " in file " << file;
    std::cerr << ConsoleUtils::reset << std::endl;
    std::exit(EXIT_FAILURE);
}

/// An unreachable error raiser.
#define Unreachable() UnreachableImpl(__LINE__, __FILE__)

/// A critical error raiser.
[[noreturn]] [[maybe_unused]] static void CriticalErrorImpl(int line, const char* file, const std::string& info) {
    std::cerr << ConsoleUtils::red;
    std::cerr << "Critical error at line " << line << " in file " << file << ":" << std::endl;
    std::cerr << "  " << info << ConsoleUtils::reset << std::endl;
    std::exit(EXIT_FAILURE);
}

/// A critical error raiser.
#define CriticalError(info) CriticalErrorImpl(__LINE__, __FILE__, info)

/// Pretty numeric ratio.
template <typename T1, typename T2>
double NumericRatio(const T1& lhs, const T2& rhs) {
    assert(rhs != 0);
    return static_cast<double>(lhs) / static_cast<double>(rhs);
}

/// Pretty numeric ratio.
template <typename T1, typename T2>
std::string PrettyRatio(const T1& lhs, const T2& rhs) {
    auto ratio = NumericRatio(lhs, rhs);
    return std::to_string(ratio * 100) + "%";
}

template <typename ValueType>
class [[maybe_unused]] RandomGen {
private:
    typedef typename std::conditional<
            std::is_integral<ValueType>::value,
            std::uniform_int_distribution<ValueType>,
            std::uniform_real_distribution<ValueType>>::type DistType;

    std::default_random_engine engine;
    DistType dist;

public:
    /// Random a value between [`min`, `max`].
    [[maybe_unused]] RandomGen(ValueType min, ValueType max, bool pure=true, uint32_t seed=0) { // NOLINT(cert-msc51-cpp)
        assert(min <= max);
        if (pure)
            seed = std::random_device()();
        engine = std::default_random_engine(seed);
        dist = DistType(min, max);
    }

    [[maybe_unused]] ValueType operator () () {
        return dist(engine);
    }
};

constexpr static uint32_t kDefaultGlobalRandomSeed = 19981011;

extern RandomGen<int> global_int_random;
extern RandomGen<double> global_uniform_random;

void ResetRandomSeed(bool pure=true, uint32_t seed=kDefaultGlobalRandomSeed);

static int RandomInt(int min=0, int max=std::numeric_limits<int>::max()) {
    return (RandomGen<int>(min, max, false, global_int_random()))();
}

/// Random choice with possibility.
static bool MakeChoice(double p) {
    assert(0 <= p and p <= 1);
    return global_uniform_random() <= p;
}

/// Range: [`min`, `max`].
template <typename RangeType=int>
struct Range {
    RangeType min, max;

    Range(const Range<RangeType>& rhs) = default;

    Range(RangeType min, RangeType max): min(min), max(max) {
        assert(min <= max);
    }

    [[nodiscard]] bool Contains(const RangeType& v) const { return min <= v and v <= max; }

    [[nodiscard]] RangeType Random() const {
        return (RandomGen<RangeType>(min, max, false, global_int_random()))();
    }

    friend std::ostream& operator << (std::ostream& os, const Range<RangeType>& range) {
        return os << "[" << range.min << ", " << range.max << "]";
    }
};

/// Randomly choose from a vector-like thing.
template <typename VecType>
auto RandomChoose(const VecType& vec) {
    assert(not vec.empty());
    return vec[RandomInt(0, vec.size() - 1)];
}

/// Randomly shuffle.
template <typename VecType>
void RandomShuffle(VecType& vec) {
    std::shuffle(vec.begin(), vec.end(), std::mt19937(RandomInt()));
}

/// Merge two vectors.
template <typename VecType>
[[maybe_unused]] VecType Merge(const VecType &lhs, const VecType& rhs) {
    VecType merged = lhs;
    merged.insert(merged.end(), rhs.begin(), rhs.end());
    return merged;
}

/// Filter in a vector-like thing.
template <typename VecType, typename Function>
[[maybe_unused]] VecType Filter(const VecType &vec, const Function &f) {
    VecType filtered;
    for (const auto &item: vec)
        if (f(item))
            filtered.push_back(item);
    return filtered;
}

/// Unique by hash function.
template <typename VecType>
[[maybe_unused]] VecType UniqueByHash(const VecType& vec) {
    VecType unique;
    std::unordered_set<size_t> hash_set;
    std::hash<typename VecType::value_type> hash_func;
    for (const auto& item: vec) {
        auto hash_value = hash_func(item);
        if (not hash_set.count(hash_value)) {
            hash_set.insert(hash_value);
            unique.push_back(item);
        }
    }
    return unique;
}

/// Iterate hash for the next.
static constexpr size_t IterateHash(size_t hash, size_t next, size_t seed=131) {
    return hash * seed + next;
}

/// Time point type.
typedef std::chrono::system_clock::time_point canvas_time_point_t;

/// Time interval type.
typedef decltype(std::chrono::seconds()) canvas_timeval_t;

/// Dynamic pointer cast.
template <typename Type, typename PtrType>
static auto DynamicCast(const PtrType& ptr) {
    return std::dynamic_pointer_cast<Type>(ptr);
}

static std::hash<std::string> HashStr;

/// Get the common prefix of two vectors.
template <typename Type>
static std::vector<Type> CommonPrefix(const std::vector<Type>& lhs,
                                      const std::vector<Type>& rhs) {
    std::vector<Type> vec;
    size_t size = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < size; ++ i) {
        if (lhs.at(i) == rhs.at(i))
            vec.push_back(lhs.at(i));
        else
            break;
    }
    return vec;
}

/// Get the common suffix of two vectors.
template <typename Type>
static std::vector<Type> CommonSuffix(const std::vector<Type>& lhs,
                                      const std::vector<Type>& rhs) {
    std::vector<Type> vec;
    size_t size = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < size; ++ i) {
        if (lhs.at(lhs.size() - i - 1) == rhs.at(rhs.size() - i - 1))
            vec.push_back(lhs.at(lhs.size() - i - 1));
        else
            break;
    }
    std::reverse(vec.begin(), vec.end());
    return vec;
}

/// Cut a vector.
template <typename Type>
static std::vector<Type> CutVector(const std::vector<Type>& vec, size_t shift, size_t size) {
    return {vec.begin() + shift, vec.begin() + shift + size};
}

/// Convert a vector into a set.
template <typename Type>
static std::set<Type> ToSet(const std::vector<Type>& vec) {
    return std::set<Type>(vec.begin(), vec.end());
}

/// Convert a vector into an unordered set.
template <typename Type>
static std::unordered_set<Type> ToUnorderedSet(const std::vector<Type>& vec) {
    return std::unordered_set<Type>(vec.begin(), vec.end());
}

/// Convert a string into a templated type.
template <typename Type>
static Type StringTo(const std::string& str) {
    assert(str.length() > 0);
    std::stringstream ss(str);
    Type value;
    ss >> value;
    return value;
}

/// Fast algorithm of powering.
template <typename Type>
static Type Power(Type base, int power) {
    assert(power >= 0);
    if (power > 0)
        assert(base > 0);
    Type result = 1;
    while (power > 0) {
        if (power & 1)
            result *= base;
        base *= base, power /= 2;
    }
    return result;
}

/// Split a string into vector.
static void Split(const std::string& str, std::vector<std::string>& vec) {
    boost::algorithm::split(vec, boost::algorithm::to_lower_copy(str),
                            boost::is_any_of("\t ,"),
                            boost::token_compress_on);
}

static std::string ToLowerCopy(const std::string& str) {
    return boost::algorithm::to_lower_copy(str);
}

/// Judge whether one is a prefix of the other.
static bool IsPrefix(const std::string& name, const std::string& filter) {
    return name.rfind(filter, 0) == 0;
}

} // namespace canvas

#define CanvasHashTemplate(Type, expr)    template <> \
struct std::hash<Type> { \
    size_t operator () (Type const& instance) const noexcept { \
        return instance expr; \
    } \
}
