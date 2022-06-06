#pragma once

#include <boost/compute/detail/lru_cache.hpp>
#include <functional>
#include <sstream>

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Core/NetSpecs.hpp"
#include "Canvas/Core/Primitive.hpp"
#include "Canvas/Core/Solution.hpp"
#include "Canvas/Core/Tensor.hpp"
#include "Canvas/Utils/Common.hpp"
#include "Canvas/Utils/Format.hpp"


namespace canvas {

namespace bcd = boost::compute::detail;

/// String-format code.
struct Code {
    std::string code;

    Code() = default;

    explicit Code(std::string code): code(std::move(code)) {}

    [[nodiscard]] const std::string& ToString() const { return code; }

    [[nodiscard]] bool Empty() const { return code.empty(); }

    void SaveIntoFile(const std::string& path) const {
        std::fstream file(path, std::ios::out);
        file << code << std::endl;
        file.close();
    }

    friend std::ostream& operator << (std::ostream& os, const Code& rhs) { return os << rhs.ToString(); }
};

struct OptimizationStorage {
    virtual ~OptimizationStorage() = default; // To make polymorphic.
};

typedef std::shared_ptr<OptimizationStorage> OptimizationStorageSP;

/// Variable map.
struct VarMap {
    std::map<TensorSP, std::string> tensor_map;
    std::map<PrimitiveSP, std::string> primitive_map;
    std::vector<OptimizationStorageSP> optimizations;

    [[nodiscard]] size_t TensorSize() const { return tensor_map.size(); }

    [[nodiscard]] size_t PrimitiveSize() const { return primitive_map.size(); }

    [[nodiscard]] bool Count(const TensorSP& t) const { return tensor_map.count(t); }

    [[nodiscard]] bool Count(const PrimitiveSP& p) const { return primitive_map.count(p); }

    std::string& operator [] (const TensorSP& t) { return tensor_map[t]; }

    std::string& operator [] (const PrimitiveSP& p) { return primitive_map[p]; }
};

/// Code generator prototype.
class CodeGen {
private:
    static constexpr int kDefaultNetFillsCacheSize = 65536;

    std::string language = "null";
    std::stringstream code_stream;
    IndentOS indent_os;

public:
    explicit CodeGen(std::string language): language(std::move(language)), indent_os(code_stream) {}

    static void CommonChecks(const Solution& solution);

    /// Generate code for specific graphs and configurations.
    virtual Code GenImpl(const Solution& solution, std::string name) = 0;

    Code Gen(const Solution& solution, const std::string& name="") {
        auto solution_hash = solution.Hash();

        static bcd::lru_cache<size_t, Code> cache = {kDefaultNetFillsCacheSize};
        auto code_cache_hash = HashStr(language);
        code_cache_hash = IterateHash(code_cache_hash, solution_hash);
        if (cache.contains(code_cache_hash))
            return cache.get(code_cache_hash).value();

        static bcd::lru_cache<size_t, bool> check_cache = {kDefaultNetFillsCacheSize};
        auto check_cache_hash = solution_hash;
        if (not check_cache.contains(check_cache_hash)) {
            CommonChecks(solution);
            check_cache.insert(check_cache_hash, true);
        }

        auto code = GenImpl(solution, name);
        cache.insert(code_cache_hash, code);
        return code;
    }

    void BeginScope() { indent_os.BeginScope(); }

    void EndScope() { indent_os.EndScope(); }

    std::ostream& Write(bool do_indent=true) { return indent_os(do_indent); }

    /// Travel the entire graph in a topological order.
    void Travel(const GraphSP& graph, const std::function<void(CodeGen*, PrimitiveSP)>& func, bool reverse=false);

    Code Dump() {
        Code code(code_stream.str());
        code_stream.clear();
        return code;
    }
};

} // namespace canvas
