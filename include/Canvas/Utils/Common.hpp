#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <ice-cream.hpp>


namespace canvas {

static constexpr int kInvalidIndex = -1;

// TODO: refactor random utils.
static constexpr int kIntUnlimited = std::numeric_limits<int>::max();
static constexpr uint64_t kUInt64Unlimited = std::numeric_limits<uint64_t>::max();

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

/// A random number generator.
template <typename ValueType>
class [[maybe_unused]] Random {
private:
    typedef typename std::conditional<
            std::is_integral<ValueType>::value,
            std::uniform_int_distribution<ValueType>,
            std::uniform_real_distribution<ValueType>>::type DistType;

    std::default_random_engine engine;
    DistType dist;

public:
    /// The interval is closed ([`min`, `max`]).
    [[maybe_unused]] Random(ValueType min, ValueType max,
                            bool pure=true, uint32_t seed=0) { // NOLINT(cert-msc51-cpp)
        assert(min <= max);
        if (pure)
            seed = std::random_device()();
        engine = std::default_random_engine(seed);
        dist = DistType(min, max);
    }

    /// Generate a random number.
    [[maybe_unused]] ValueType operator () () {
        return dist(engine);
    }
};

constexpr static uint32_t kDefaultGlobalRandomSeed = 19981011;

extern Random<int> global_int_random;
extern Random<uint64_t> global_uint64_random;
extern Random<double> global_norm_uniform_random;

/// Init global random engine.
void InitRandomEngine(bool pure=true, uint32_t seed=kDefaultGlobalRandomSeed);

/// Random a positive integer of 32 bits.
static int RandomInt(int min=0, int max=kIntUnlimited) {
    assert(0 <= min and min <= max);
    // Overflow with `max - min + 1`
    if (min == 0 and max == kIntUnlimited)
        return global_int_random();
    return global_int_random() % (max - min + 1) + min;
}

/// Random a positive integer of 64 bits.
static uint64_t RandomUInt64(uint64_t min=0, uint64_t max=kUInt64Unlimited) {
    assert(min <= max);
    // Overflow with `max - min + 1`
    if (min == 0 and max == kUInt64Unlimited)
        return global_uint64_random();
    return global_uint64_random() % (max - min + 1) + min;
}

/// Random choice with possibility.
static bool MakeChoice(double p) {
    assert(0 <= p and p <= 1);
    return global_norm_uniform_random() <= p;
}

/// Range: [`min`, `max`]
template <typename RangeType=int>
struct Range {
    RangeType min, max;

    Range(const Range<RangeType>& rhs) = default;

    Range(RangeType min, RangeType max): min(min), max(max) { assert(min <= max); }

    [[nodiscard]] bool Contains(const RangeType& v) const { return min <= v and v <= max; }

    [[nodiscard]] RangeType Random() const {
        return RandomUInt64(static_cast<RangeType>(min), static_cast<RangeType>(max));
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

/// Exception class.
class ExceptionWithInfo: public std::exception {
public:
    std::string info;

    explicit ExceptionWithInfo(std::string info="An unknown exception occurs"):
        info(std::move(info)) {}

    [[nodiscard]] const char* what() const noexcept override {
        return info.c_str();
    }
};

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

/// Fast algorithm of powering
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

} // namespace canvas

#define CanvasHashTemplate(Type, expr)    template <> \
struct std::hash<Type> { \
    size_t operator () (Type const& instance) const noexcept { \
        return instance expr; \
    } \
}
