/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
// Shared by standalone runner and GTest. Golden always calls cblas_sgemm (not NumPy).
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef GB_CPU_MOCK
#include "mock_runtime.h"
#endif
namespace gb_test {
using Row = std::map<std::string, std::string>;
struct Skip : std::runtime_error {
    using std::runtime_error::runtime_error;
};
inline void CheckACL(aclError r, const char* where)
{
    if (r != ACL_SUCCESS)
        throw std::runtime_error(std::string(where) + " failed: " + std::to_string(r));
}
inline void CheckBLAS(aclblasStatus_t r, const char* where)
{
    if (r != ACLBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(where) + " failed: " + std::to_string(int(r)));
}
inline std::string Cell(const Row& r, const std::string& key, const std::string& fallback = "")
{
    auto i = r.find(key);
    return i == r.end() ? fallback : i->second;
}
inline std::vector<std::string> Split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::string t;
    for (char c : s) {
        if (c == sep) {
            out.push_back(t);
            t.clear();
        } else
            t += c;
    }
    out.push_back(t);
    return out;
}
inline std::vector<std::vector<std::string>> ParseCSV(std::istream& in)
{
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    char c;
    while (in.get(c)) {
        if (c == '"') {
            if (quoted && in.peek() == '"') {
                field += '"';
                in.get(c);
            } else
                quoted = !quoted;
        } else if (!quoted && c == ',') {
            row.push_back(field);
            field.clear();
        } else if (!quoted && c == '\n') {
            if (!field.empty() && field.back() == '\r')
                field.pop_back();
            row.push_back(field);
            field.clear();
            rows.push_back(row);
            row.clear();
        } else
            field += c;
    }
    if (quoted)
        throw std::runtime_error("unterminated CSV quote");
    if (!row.empty() || !field.empty()) {
        row.push_back(field);
        rows.push_back(row);
    }
    return rows;
}
inline std::vector<Row> ReadCSV(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read CSV: " + path);
    auto cells = ParseCSV(f);
    if (cells.empty())
        throw std::runtime_error("empty CSV: " + path);
    std::vector<Row> rows;
    std::map<std::string, bool> names;
    for (size_t i = 1; i < cells.size(); ++i) {
        if (cells[i].size() == 1 && cells[i][0].empty())
            continue;
        if (cells[i].size() != cells[0].size())
            throw std::runtime_error("CSV column count at row " + std::to_string(i + 1));
        Row r;
        for (size_t j = 0; j < cells[0].size(); ++j)
            r[cells[0][j]] = cells[i][j];
        if (!names.emplace(Cell(r, "case_name"), true).second)
            throw std::runtime_error("duplicate case_name");
        rows.push_back(std::move(r));
    }
    if (rows.empty())
        throw std::runtime_error("CSV has no cases");
    return rows;
}
inline int Int(const std::string& s)
{
    size_t end = 0;
    auto v = std::stoll(s, &end);
    if (end != s.size() || v < INT32_MIN || v > INT32_MAX)
        throw std::runtime_error("bad int32: " + s);
    return static_cast<int>(v);
}
inline float Float(const std::string& s)
{
    // std::stof may throw on valid subnormals (ERANGE). strtof preserves their rounded FP32 value.
    char* end = nullptr;
    errno = 0;
    const float v = std::strtof(s.c_str(), &end);
    if (end == s.c_str() || end != s.c_str() + s.size())
        throw std::runtime_error("bad float: " + s);
    if (errno == ERANGE && std::isinf(v))
        throw std::runtime_error("float overflow: " + s);
    return v;
}

inline aclblasOperation_t Op(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    if (s == "N" || s == "ACLBLAS_OP_N" || s == "0")
        return ACLBLAS_OP_N;
    if (s == "T" || s == "ACLBLAS_OP_T" || s == "1")
        return ACLBLAS_OP_T;
    if (s == "C" || s == "ACLBLAS_OP_C" || s == "2")
        return ACLBLAS_OP_C;
    if (s == "INVALID" || s == "ACLBLAS_OP_INVALID")
        return static_cast<aclblasOperation_t>(99);
    return static_cast<aclblasOperation_t>(Int(s));
}
inline aclblasStatus_t Expected(const Row& r)
{
    const auto s = Cell(r, "expect_result");
    if (s == "ACLBLAS_STATUS_SUCCESS")
        return ACLBLAS_STATUS_SUCCESS;
    if (s == "ACLBLAS_STATUS_INVALID_VALUE")
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (s == "ACLBLAS_STATUS_HANDLE_IS_NULLPTR")
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    throw std::runtime_error("unknown expected status: " + s);
}
template <class T, class F>
inline std::vector<T> ParseArray(const Row& r, const char* key, int count, F convert)
{
    std::string s = Cell(r, key);
    const size_t n = static_cast<size_t>(std::max(0, count));
    if (s == "NULLPTR" || s.empty())
        return std::vector<T>(std::max<size_t>(n, 1));
    auto fields = Split(s, ';');
    if (n && fields.size() != n)
        throw std::runtime_error(std::string("wrong array length: ") + key);
    std::vector<T> out;
    for (const auto& x : fields)
        out.push_back(convert(x));
    return out;
}
struct Context {
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    int device = 0;
    bool initialized = false, set = false;
    explicit Context(int id = 0) : device(id)
    {
        try {
            CheckACL(aclInit(nullptr), "aclInit");
            initialized = true;
            CheckACL(aclrtSetDevice(device), "aclrtSetDevice");
            set = true;
            CheckACL(aclrtCreateStream(&stream), "aclrtCreateStream");
            CheckBLAS(aclblasCreate(&handle), "aclblasCreate");
            CheckBLAS(aclblasSetStream(handle, stream), "aclblasSetStream");
        } catch (...) {
            Release();
            throw;
        }
    }
    Context(const Context&) = delete;
    void Release() noexcept
    {
        if (stream)
            aclrtSynchronizeStream(stream);
        if (handle)
            aclblasDestroy(handle);
        if (stream)
            aclrtDestroyStream(stream);
        if (set)
            aclrtResetDevice(device);
        if (initialized)
            aclFinalize();
        stream = nullptr;
        handle = nullptr;
        set = initialized = false;
    }
    ~Context() { Release(); }
};
constexpr float kSentinel = 12345.25f;
// A 512-byte prefix preserves allocation alignment; explicit offsets remain effective.
// ReadChecked still validates every prefix and suffix guard element.
struct Buffer {
    void* base = nullptr;
    size_t elements = 0, prefix = 128, suffix = 8;
    Buffer(size_t n, size_t offset) : elements(std::max<size_t>(n, 1)), prefix(128 + offset)
    {
        CheckACL(aclrtMalloc(&base, Bytes(), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    Buffer(const Buffer&) = delete;
    ~Buffer()
    {
        if (base)
            aclrtFree(base);
    }
    size_t Bytes() const { return (elements + prefix + suffix) * sizeof(float); }
    float* Data() const { return static_cast<float*>(base) + prefix; }
    void Upload(const std::vector<float>& payload)
    {
        if (payload.size() != elements)
            throw std::runtime_error("upload length mismatch");
        std::vector<float> all(elements + prefix + suffix, kSentinel);
        std::copy(payload.begin(), payload.end(), all.begin() + prefix);
        CheckACL(aclrtMemcpy(base, Bytes(), all.data(), Bytes(), ACL_MEMCPY_HOST_TO_DEVICE), "upload");
    }
    std::vector<float> ReadChecked() const
    {
        std::vector<float> all(elements + prefix + suffix);
        CheckACL(aclrtMemcpy(all.data(), Bytes(), base, Bytes(), ACL_MEMCPY_DEVICE_TO_HOST), "readback");
        for (size_t i = 0; i < prefix; ++i)
            if (all[i] != kSentinel)
                throw std::runtime_error("prefix guard modified");
        for (size_t i = prefix + elements; i < all.size(); ++i)
            if (all[i] != kSentinel)
                throw std::runtime_error("suffix guard modified");
        return std::vector<float>(all.begin() + prefix, all.begin() + prefix + elements);
    }
};
struct MatrixSpec {
    int rows, cols, ld;
    std::string fill;
    uint32_t seed;
};
inline size_t Elements(const MatrixSpec& s)
{
    uint64_t n = static_cast<uint64_t>(std::max(0, s.cols)) * static_cast<uint64_t>(std::max(1, s.ld));
    if (n > std::numeric_limits<size_t>::max() / sizeof(float))
        throw Skip("matrix byte extent overflow");
    return static_cast<size_t>(std::max<uint64_t>(1, n));
}
inline std::vector<float> Generate(const MatrixSpec& s)
{
    std::vector<float> out(Elements(s), kSentinel);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
    float mu = 0, sigma = 1;
    const bool normal = s.fill.find("NORMAL:") == 0;
    if (normal) {
        auto f = Split(s.fill, ':');
        if (f.size() != 3)
            throw std::runtime_error("bad normal fill");
        mu = Float(f[1]);
        sigma = Float(f[2]);
        if (!(sigma > 0))
            throw std::runtime_error("normal sigma must be positive");
    }
    std::normal_distribution<float> norm(mu, sigma);
    for (int j = 0; j < std::max(0, s.cols); ++j)
        for (int i = 0; i < std::max(0, s.rows); ++i) {
            float v;
            if (normal)
                v = norm(gen);
            else if (s.fill == "RANDOM_NORM_5_5" || s.fill == "UNIFORM")
                v = uniform(gen);
            else if (s.fill == "VALUE_NORM_0")
                v = 0;
            else if (s.fill == "VALUE_NORM_INF")
                v = std::numeric_limits<float>::infinity();
            else if (s.fill == "VALUE_NORM_NAN")
                v = std::numeric_limits<float>::quiet_NaN();
            else if (s.fill == "RANDOM_ALTER")
                v = ((i + j) & 1 ? -1.0f : 1.0f) * std::fabs(uniform(gen));
            else if (s.fill == "RANDOM_EXTREME") {
                // Explicit, reproducible wide-dynamic-range profile, not a claim about undocumented upstream fillers.
                const float values[] = {1.0e-4f, -1.0e-4f, 1.0e4f, -1.0e4f, 0.0f};
                v = values[gen() % 5];
            } else
                throw std::runtime_error("unsupported fill: " + s.fill);
            out[static_cast<size_t>(j) * s.ld + i] = v;
        }
    return out;
}
struct Parameters {
    Row row;
    int gc;
    std::vector<aclblasOperation_t> ta, tb;
    std::vector<int> m, n, k, lda, ldb, ldc, batch;
    std::vector<float> alpha, beta;
    std::vector<const float*> a, b;
    std::vector<float*> c;
    explicit Parameters(const Row& r) : row(r), gc(Int(Cell(r, "groupCount")))
    {
        ta = ParseArray<aclblasOperation_t>(r, "transaArray", gc, Op);
        tb = ParseArray<aclblasOperation_t>(r, "transbArray", gc, Op);
        m = ParseArray<int>(r, "mArray", gc, Int);
        n = ParseArray<int>(r, "nArray", gc, Int);
        k = ParseArray<int>(r, "kArray", gc, Int);
        lda = ParseArray<int>(r, "ldaArray", gc, Int);
        ldb = ParseArray<int>(r, "ldbArray", gc, Int);
        ldc = ParseArray<int>(r, "ldcArray", gc, Int);
        batch = ParseArray<int>(r, "groupSizeArray", gc, Int);
        alpha = ParseArray<float>(r, "alphaArray", gc, Float);
        beta = ParseArray<float>(r, "betaArray", gc, Float);
        size_t count = 0;
        for (int g = 0; g < gc; ++g)
            count += static_cast<size_t>(std::max(0, batch[g]));
        if (count > 10000000)
            throw Skip("test fixture pointer-count safety limit");
        a.resize(std::max<size_t>(1, count));
        b.resize(a.size());
        c.resize(a.size());
    }
    template <class T>
    const T* Ptr(const std::vector<T>& v, const char* field) const
    {
        return Cell(row, field) == "NULLPTR" ? nullptr : v.data();
    }
    void SetNullElements()
    {
        auto apply = [&](const char* key, auto& list) {
            const auto cell = Cell(row, key);
            if (cell.empty() || cell == "NULLPTR")
                return;
            auto flags = Split(cell, ';');
            size_t offset = 0;
            for (int g = 0; g < gc; ++g) {
                if (static_cast<size_t>(g) < flags.size() && Int(flags[g]) != 0 && batch[g] > 0)
                    list[offset] = nullptr;
                offset += static_cast<size_t>(std::max(0, batch[g]));
            }
        };
        apply("Aarray_null", a);
        apply("Barray_null", b);
        apply("Carray_null", c);
        // Supplemental exact flattened-element null injection, independent of the original group flags.
        for (const auto& key : {"A_null_index", "B_null_index", "C_null_index"}) {
            const auto v = Cell(row, key);
            if (v.empty())
                continue;
            size_t i = static_cast<size_t>(Int(v));
            if (i >= a.size())
                throw std::runtime_error("null index outside pointer array");
            if (key[0] == 'A')
                a[i] = nullptr;
            else if (key[0] == 'B')
                b[i] = nullptr;
            else
                c[i] = nullptr;
        }
    }
    aclblasStatus_t Call(aclblasHandle_t handle) const
    {
        return aclblasSgemmGroupedBatched(
            Cell(row, "description") == "null_handle" ? nullptr : handle, gc, Ptr(ta, "transaArray"),
            Ptr(tb, "transbArray"), Ptr(m, "mArray"), Ptr(n, "nArray"), Ptr(k, "kArray"), Ptr(alpha, "alphaArray"),
            Cell(row, "Aarray_null") == "NULLPTR" ? nullptr : a.data(), Ptr(lda, "ldaArray"),
            Cell(row, "Barray_null") == "NULLPTR" ? nullptr : b.data(), Ptr(ldb, "ldbArray"), Ptr(beta, "betaArray"),
            Cell(row, "Carray_null") == "NULLPTR" ? nullptr : c.data(), Ptr(ldc, "ldcArray"),
            Ptr(batch, "groupSizeArray"));
    }
    uint64_t Flops() const
    {
        uint64_t total = 0;
        for (int g = 0; g < gc; ++g) {
            long double x = 2.0L * std::max(0, m[g]) * std::max(0, n[g]) * std::max(0, k[g]) * std::max(0, batch[g]);
            if (x > std::numeric_limits<uint64_t>::max() - total)
                return UINT64_MAX;
            total += static_cast<uint64_t>(x);
        }
        return total;
    }
};
struct Metrics {
    uint64_t total = 0, matched = 0, nonfinite = 0, mismatchedNonfinite = 0, hardFailures = 0;
    double maxAbs = 0, maxRel = 0;
    bool Passed() const
    {
        return (!total || double(matched) / double(total) >= 0.99) && !hardFailures && !mismatchedNonfinite;
    }
    void Add(float actual, float golden, bool ulpLimit = false)
    {
        ++total;
        if (!std::isfinite(golden) || !std::isfinite(actual)) {
            ++nonfinite;
            if ((std::isnan(golden) && std::isnan(actual)) || (std::isinf(golden) && actual == golden))
                ++matched;
            else {
                ++mismatchedNonfinite;
                ++hardFailures;
            }
            return;
        }
        const double error = std::fabs(double(actual) - double(golden));
        maxAbs = std::max(maxAbs, error);
        if (golden != 0)
            maxRel = std::max(maxRel, error / std::fabs(double(golden)));
        if (error <= std::ldexp(1.0, -13) * (1.0 + std::fabs(double(golden))))
            ++matched;
        double hard = 0.01;
        if (ulpLimit) {
            const float magnitude = std::fabs(golden);
            const float next = std::nextafter(magnitude, std::numeric_limits<float>::infinity());
            const double ulp =
                std::isfinite(next) ? double(next) - magnitude : double(magnitude) - std::nextafter(magnitude, 0.0f);
            hard = std::max(hard, 32.0 * ulp);
        }
        if (error > hard)
            ++hardFailures;
    }
};
struct Options {
    std::string mode = "accuracy";
    uint64_t maxFlops = 0, maxBytes = uint64_t{32} << 30;
    int warmup = 5, samples = 20;
    bool ulpLimit = true, checkInputs = true;
};
struct Result {
    std::string name, status, message, backend, mode;
    Metrics metrics;
    uint64_t bytes = 0, flops = 0;
    size_t launches = 0;
    std::vector<double> eventUs;
};
struct DeviceMatrix {
    int group, batch;
    MatrixSpec as, bs, cs;
    std::unique_ptr<Buffer> a, b, c;
};
} // namespace gb_test
#include "gb_test_runtime.h"
