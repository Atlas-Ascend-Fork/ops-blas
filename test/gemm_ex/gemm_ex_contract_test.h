/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cblas.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <type_traits>
#include <thread>
#include <chrono>
#include "blas_test.h"
#include "csv_loader.h"

namespace gemm_ex_test {
inline aclDataType Dtype(const std::string& text)
{
    if (text == "ACL_FLOAT") {
        return ACL_FLOAT;
    }
    if (text == "ACL_COMPLEX64") {
        return ACL_COMPLEX64;
    }
    if (text == "ACL_FLOAT16") {
        return ACL_FLOAT16;
    }
    if (text == "ACL_BF16") {
        return ACL_BF16;
    }
    if (text == "ACL_DOUBLE") {
        return ACL_DOUBLE;
    }
    if (text == "ACL_INT32") {
        return ACL_INT32;
    }
    return static_cast<aclDataType>(std::stoi(text));
}
struct Param : BlasTestParamBase {
    csv_map fields;
    explicit Param(const csv_map& f) : BlasTestParamBase(f), fields(f) {}
    std::string Text(const std::string& key, const std::string& fallback = "") const
    {
        return ReadMap(fields, key, fallback);
    }
    int Int(const std::string& key) const { return std::stoi(Text(key)); }
};
inline Param ReviewCase(
    const std::string& name, int m, int n, int k, const std::string& alpha, const std::string& beta,
    bool transposeA = false, bool transposeB = false)
{
    return Param(
        {{"case_name", name},
         {"description", "Review regression: general scalars and short-K neighbors"},
         {"m", std::to_string(m)},
         {"n", std::to_string(n)},
         {"k", std::to_string(k)},
         {"lda", std::to_string(transposeA ? k : m)},
         {"ldb", std::to_string(transposeB ? n : k)},
         {"ldc", std::to_string(m)},
         {"transA", transposeA ? "ACLBLAS_OP_T" : "ACLBLAS_OP_N"},
         {"transB", transposeB ? "ACLBLAS_OP_T" : "ACLBLAS_OP_N"},
         {"alpha", alpha},
         {"beta", beta},
         {"Atype", "ACL_FLOAT"},
         {"Btype", "ACL_FLOAT"},
         {"Ctype", "ACL_FLOAT"},
         {"a_fill", "RANDOM_NORM_5_5"},
         {"b_fill", "RANDOM_NORM_5_5"},
         {"c_fill", "RANDOM_NORM_5_5"},
         {"computeType", "ACLBLAS_COMPUTE_32F"},
         {"algo", "ACLBLAS_GEMM_DEFAULT"},
         {"random_seed", "20261105"},
         {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
         {"mere_threshold", "0.00012207031"},
         {"mare_multiplier", "10.0"}});
}
inline std::vector<Param> ReviewRegressionCases()
{
    std::vector<Param> cases{
        ReviewCase("AlphaHalf1024NN", 1024, 1024, 1024, "0.5", "0"),
        ReviewCase("AlphaNegative1024TT", 1024, 1024, 1024, "-1", "0.25", true, true),
        ReviewCase("AlphaDecimal1024NN", 1024, 1024, 1024, "0.3", "-0.7"),
        ReviewCase("AlphaFraction1024NT", 1024, 1024, 1024, "1.25", "0.5", false, true),
        ReviewCase("SmallOutputLongK", 16, 16, 8192, "0.5", "0.25")};
    for (int m : {368, 369, 384, 512, 704, 720, 721, 729, 736, 737}) {
        for (int n : {16, 551}) {
            for (int k : {17, 29, 32}) {
                const auto name = "ShortNN_" + std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k);
                cases.push_back(ReviewCase(name, m, n, k, "1", "0"));
            }
        }
    }
    return cases;
}
struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    explicit DeviceBuffer(size_t n) : bytes(n)
    {
        if (n && aclrtMalloc(&ptr, n, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            throw std::runtime_error("device allocation failed");
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if (ptr) {
            aclrtFree(ptr);
        }
    }
    void Upload(const void* source)
    {
        if (bytes && aclrtMemcpy(ptr, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            throw std::runtime_error("host to device copy failed");
        }
    }
    void Download(void* dest)
    {
        if (bytes && aclrtMemcpy(dest, bytes, ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device to host copy failed");
        }
    }
};
template <typename T>
std::vector<T> Matrix(int rows, int cols, int ld, const std::string& fill, uint32_t seed)
{
    if constexpr (std::is_same_v<T, float>) {
        return makeBlasMatrix(rows, cols, ld, fill, seed);
    } else {
        return makeBlasComplexMatrix(rows, cols, ld, parseFill(fill), seed);
    }
}
template <typename T>
T Scalar(const Param& p, const std::string& key)
{
    if constexpr (std::is_same_v<T, float>) {
        return parseFloat(p.Text(key));
    } else {
        return {parseFloat(p.Text(key + "_real")), parseFloat(p.Text(key + "_imag"))};
    }
}
inline bool NullText(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value == "null" || value == "nullptr";
}
template <typename T>
bool NullScalar(const Param& p, const std::string& key)
{
    if constexpr (std::is_same_v<T, float>) {
        return NullText(p.Text(key));
    } else {
        return NullText(p.Text(key + "_real")) || NullText(p.Text(key + "_imag"));
    }
}
inline bool Zero(float x) { return x == 0; }
inline bool Zero(aclblasComplex x) { return x.real == 0 && x.imag == 0; }
inline float Scale(float x, float y) { return x * y; }
inline aclblasComplex Scale(aclblasComplex x, aclblasComplex y)
{
    return {x.real * y.real - x.imag * y.imag, x.real * y.imag + x.imag * y.real};
}
inline CBLAS_TRANSPOSE Trans(aclblasOperation_t op)
{
    return op == ACLBLAS_OP_N ? CblasNoTrans : (op == ACLBLAS_OP_T ? CblasTrans : CblasConjTrans);
}
template <int KIND, typename T>
aclblasStatus_t Invoke(
    aclblasHandle_t h, const Param& p, const T* alpha, const void* a, const void* b, const T* beta, void* c)
{
    auto ta = parseOpTrans(p.Text("transA"));
    auto tb = parseOpTrans(p.Text("transB"));
    int m = p.Int("m");
    int n = p.Int("n");
    int k = p.Int("k");
    int lda = p.Int("lda");
    int ldb = p.Int("ldb");
    int ldc = p.Int("ldc");
    auto at = Dtype(p.Text("Atype"));
    auto bt = Dtype(p.Text("Btype"));
    auto ct = Dtype(p.Text("Ctype"));
    if constexpr (KIND == 0) {
        auto algo = p.Text("algo") == "ACLBLAS_GEMM_DEFAULT" ?
                        ACLBLAS_GEMM_DEFAULT :
                        static_cast<aclblasGemmAlgo_t>(std::stoi(p.Text("algo")));
        return aclblasGemmEx(
            h, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc,
            parseComputeType(p.Text("computeType")), algo);
    } else if constexpr (KIND == 1) {
        return aclblasSgemmEx(h, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc);
    } else {
        return aclblasCgemmEx(h, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc);
    }
}
inline bool SameSpecial(float actual, float golden)
{
    if (std::isfinite(actual) || std::isfinite(golden)) {
        return false;
    }
    return std::isnan(actual) || std::isnan(golden) || std::signbit(actual) == std::signbit(golden);
}
inline void AccumulateFinite(
    float actual, float golden, uint64_t& finite, uint64_t& matched, double& maximum, double& worstGolden,
    double& limit)
{
    ++finite;
    double error = std::abs(static_cast<double>(actual) - golden);
    const double ulp = std::ldexp(1.0, golden == 0.0f ? -149 : std::max(-149, std::ilogb(std::abs(golden)) - 23));
    const double candidateLimit = std::max(0.01, 32 * ulp);
    if (error > maximum || (error == maximum && candidateLimit < limit)) {
        maximum = error;
        worstGolden = golden;
        limit = candidateLimit;
    }
    if (error <= std::ldexp(1.0, -13) * (1.0 + std::abs(static_cast<double>(golden)))) {
        ++matched;
    }
}
template <typename T>
bool CheckPadding(const std::vector<T>& actual, const std::vector<T>& before, size_t pos, int row, int m)
{
    if (row < m) {
        return false;
    }
    EXPECT_EQ(std::memcmp(&actual[pos], &before[pos], sizeof(T)), 0) << "padding " << pos;
    return true;
}
template <typename T>
void Compare(
    const Param& p, const std::vector<T>& actual, const std::vector<T>& golden, const std::vector<T>& before, int m,
    int n, int ldc, bool exact)
{
    constexpr int components = sizeof(T) / sizeof(float);
    uint64_t finite[components] = {};
    uint64_t matched[components] = {};
    uint64_t special = 0;
    uint64_t badSpecial = 0;
    uint64_t badExact = 0;
    double maximum[components] = {};
    double worstGolden[components] = {};
    double limit[components];
    std::fill_n(limit, components, 0.01);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < ldc; ++row) {
            size_t pos = static_cast<size_t>(col) * ldc + row;
            if (CheckPadding(actual, before, pos, row, m)) {
                continue;
            }
            const auto* av = reinterpret_cast<const float*>(&actual[pos]);
            const auto* gv = reinterpret_cast<const float*>(&golden[pos]);
            for (int j = 0; j < components; ++j) {
                if (!std::isfinite(av[j]) || !std::isfinite(gv[j])) {
                    ++special;
                    badSpecial += !SameSpecial(av[j], gv[j]);
                } else {
                    AccumulateFinite(av[j], gv[j], finite[j], matched[j], maximum[j], worstGolden[j], limit[j]);
                }
                if (exact && std::memcmp(av + j, gv + j, sizeof(float))) {
                    ++badExact;
                }
            }
        }
    }
    for (int j = 0; j < components; ++j) {
        double ratio = finite[j] ? static_cast<double>(matched[j]) / finite[j] : 1.0;
        std::cout << "GEMMEX_PRECISION," << p.caseName << "," << j << "," << finite[j] << "," << matched[j] << ","
                  << std::setprecision(12) << maximum[j] << "," << special << "," << badSpecial << "," << badExact
                  << "," << worstGolden[j] << "," << limit[j] << "," << (maximum[j] <= limit[j]) << std::endl;
        EXPECT_GE(ratio, 0.99);
        EXPECT_LE(maximum[j], limit[j]);
    }
    EXPECT_EQ(badSpecial, 0U);
    if (exact) {
        EXPECT_EQ(badExact, 0U);
    }
}
template <int KIND, typename T>
void CheckInvalid(const Param& p, aclblasHandle_t h, aclrtStream stream, const T* ap, const T* bp)
{
    // Invalid calls must return before dereferencing device matrices.
    DeviceBuffer sentinel(64);
    auto status = Invoke<KIND>(
        h, p, ap, p.Text("a_fill") == "NULLPTR" ? nullptr : sentinel.ptr,
        p.Text("b_fill") == "NULLPTR" ? nullptr : sentinel.ptr, bp,
        p.Text("c_fill") == "NULLPTR" ? nullptr : sentinel.ptr);
    ASSERT_EQ(status, p.expectResult);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
}
template <int KIND, typename T>
void RunSamples(
    const Param& p, aclblasHandle_t h, aclrtStream stream, const T* ap, const T* bp, DeviceBuffer& da, DeviceBuffer& db,
    DeviceBuffer& dc, const std::vector<T>& before)
{
    const bool profiling = std::getenv("GEMMEX_PROFILE") != nullptr;
    const int repeats = profiling ? 69 : 1;
    for (int i = 0; i < repeats; ++i) {
        if (i) {
            dc.Upload(before.data());
        }
        const auto start = std::chrono::system_clock::now();
        ASSERT_EQ(Invoke<KIND>(h, p, ap, da.ptr, db.ptr, bp, dc.ptr), ACLBLAS_STATUS_SUCCESS);
        uint32_t lastTask = 0;
        int32_t streamId = 0;
        if (profiling) {
            ASSERT_EQ(aclrtGetThreadLastTaskId(&lastTask), ACL_SUCCESS);
            ASSERT_EQ(aclrtStreamGetId(stream, &streamId), ACL_SUCCESS);
        }
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        const auto end = std::chrono::system_clock::now();
        if (profiling) {
            std::cout << "GEMMEX_SAMPLE," << p.caseName << "," << i << ","
                      << std::chrono::duration_cast<std::chrono::microseconds>(start.time_since_epoch()).count() << ","
                      << std::chrono::duration_cast<std::chrono::microseconds>(end.time_since_epoch()).count() << ","
                      << streamId << "," << lastTask << std::endl;
        }
    }
}
template <int KIND, typename T>
void GenerateGolden(
    const Param& p, const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& before, T alpha, T beta,
    std::vector<T>& golden)
{
    const int m = p.Int("m");
    const int n = p.Int("n");
    const int k = p.Int("k");
    const int lda = p.Int("lda");
    const int ldb = p.Int("ldb");
    const int ldc = p.Int("ldc");
    const auto ta = parseOpTrans(p.Text("transA"));
    const auto tb = parseOpTrans(p.Text("transB"));
    if (Zero(alpha) || k == 0) {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                size_t pos = static_cast<size_t>(col) * ldc + row;
                golden[pos] = Zero(beta) ? T{} : Scale(beta, golden[pos]);
            }
        }
    } else {
        // Netlib computes each output column independently. Partition only N;
        // every dot-product retains the original library and accumulation order.
        auto generate = [&](int first, int count, std::vector<T>& output) {
            const size_t offset = tb == ACLBLAS_OP_N ? static_cast<size_t>(first) * ldb : first;
            if constexpr (KIND == 2) {
                cblas_cgemm(
                    CblasColMajor, Trans(ta), Trans(tb), m, count, k, &alpha, a.data(), lda, b.data() + offset, ldb,
                    &beta, output.data() + static_cast<size_t>(first) * ldc, ldc);
            } else {
                cblas_sgemm(
                    CblasColMajor, Trans(ta), Trans(tb), m, count, k, alpha, a.data(), lda, b.data() + offset, ldb,
                    beta, output.data() + static_cast<size_t>(first) * ldc, ldc);
            }
        };
        const int workers = static_cast<uint64_t>(m) * n * k >= 16777216 ? std::min(n, 16) : 1;
        std::vector<std::thread> threads;
        for (int worker = 0; worker < workers; ++worker) {
            const int first = n * worker / workers;
            const int last = n * (worker + 1) / workers;
            threads.emplace_back([&, first, last] { generate(first, last - first, golden); });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (std::getenv("GEMMEX_CHECK_GOLDEN_SPLIT")) {
            auto serial = before;
            generate(0, n, serial);
            ASSERT_EQ(std::memcmp(serial.data(), golden.data(), serial.size() * sizeof(T)), 0)
                << "Netlib column partition must be bitwise equivalent";
        }
    }
}
template <int KIND>
void Run(const Param& p, aclblasHandle_t handle, aclrtStream stream)
{
    // CANN 9.1 kernel replay restores allocation snapshots on the default
    // stream. Keep those restores ordered with the measured public call.
    // Ordinary correctness tests continue to exercise the fixture's stream.
    if (std::getenv("GEMMEX_PROFILE") && std::getenv("GEMMEX_DEFAULT_STREAM")) {
        ASSERT_EQ(aclblasSetStream(handle, nullptr), ACLBLAS_STATUS_SUCCESS);
        stream = nullptr;
    }
    using T = std::conditional_t<KIND == 2, aclblasComplex, float>;
    const int m = p.Int("m");
    const int n = p.Int("n");
    const int k = p.Int("k");
    const int lda = p.Int("lda");
    const int ldb = p.Int("ldb");
    const int ldc = p.Int("ldc");
    auto ta = parseOpTrans(p.Text("transA"));
    auto tb = parseOpTrans(p.Text("transB"));
    T alpha = Scalar<T>(p, "alpha");
    T beta = Scalar<T>(p, "beta");
    auto h = p.description.find("handle_null") != std::string::npos ? nullptr : handle;
    const T* ap = NullScalar<T>(p, "alpha") ? nullptr : &alpha;
    const T* bp = NullScalar<T>(p, "beta") ? nullptr : &beta;
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        ASSERT_NO_FATAL_FAILURE(CheckInvalid<KIND>(p, h, stream, ap, bp));
        return;
    }
    // A/B are non-null for zero dimensions unless the input explicitly requests nullptr.
    auto a = Matrix<T>(
        std::max(1, ta == ACLBLAS_OP_N ? m : k), std::max(1, ta == ACLBLAS_OP_N ? k : m), std::max(1, lda),
        p.Text("a_fill"), p.randomSeed);
    auto b = Matrix<T>(
        std::max(1, tb == ACLBLAS_OP_N ? k : n), std::max(1, tb == ACLBLAS_OP_N ? n : k), std::max(1, ldb),
        p.Text("b_fill"), p.randomSeed + 1);
    auto c = Matrix<T>(std::max(1, m), std::max(1, n), std::max(1, ldc), p.Text("c_fill"), p.randomSeed + 2);
    const auto before = c;
    DeviceBuffer da(a.size() * sizeof(T));
    DeviceBuffer db(b.size() * sizeof(T));
    DeviceBuffer dc(c.size() * sizeof(T));
    da.Upload(a.data());
    db.Upload(b.data());
    dc.Upload(c.data());
    ASSERT_NO_FATAL_FAILURE(RunSamples<KIND>(p, h, stream, ap, bp, da, db, dc, before));
    dc.Download(c.data());
    // Verify the final output after timing, including performance-only cases.
    // The Netlib work starts after every measured device invocation completes.
    if (m == 0 || n == 0 || c.empty()) {
        return;
    }
    auto golden = before;
    ASSERT_NO_FATAL_FAILURE(GenerateGolden<KIND>(p, a, b, before, alpha, beta, golden));
    Compare(p, c, golden, before, m, n, ldc, Zero(alpha));
}
} // namespace gemm_ex_test
