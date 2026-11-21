/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"
#include "csyr2k_param.h"
#include "csyr2k_golden.h"
#include "csyr2k_npu_wrapper.h"

class Csyr2kArch35Test : public BlasTest<Csyr2kParam> {};

struct Csyr2kHostData {
    std::vector<aclblasComplex> a;
    std::vector<aclblasComplex> b;
    std::vector<aclblasComplex> c;
    std::vector<aclblasComplex> cInput;
    std::vector<aclblasComplex> golden;
    aclblasComplex dummyA{0.0f, 0.0f};
    aclblasComplex dummyB{0.0f, 0.0f};
    aclblasComplex dummyC{0.0f, 0.0f};
};

inline void PrepareCsyr2kHostData(const Csyr2kParam& p, Csyr2kHostData& data)
{
    int rows = (p.trans == ACLBLAS_OP_N) ? p.n : p.k;
    int cols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;
    if (p.expectResult == ACLBLAS_STATUS_SUCCESS && rows > 0 && cols > 0) {
        data.a = makeBlasComplexMatrix(rows, cols, p.lda, p.fillA, p.randomSeed);
        data.b = makeBlasComplexMatrix(rows, cols, p.ldb, p.fillB, p.randomSeed + 1U);
    }
    if (p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0) {
        data.c = makeBlasComplexMatrix(p.n, p.n, p.ldc, p.fillC, p.randomSeed + 2U);
    }
    data.cInput = data.c;
    data.golden = data.c;
}

inline const aclblasComplex* Csyr2kInputPointer(
    bool isNull, const std::vector<aclblasComplex>& values, const aclblasComplex& dummy)
{
    if (isNull)
        return nullptr;
    return values.empty() ? &dummy : values.data();
}

inline aclblasComplex* Csyr2kOutputPointer(bool isNull, std::vector<aclblasComplex>& values, aclblasComplex& dummy)
{
    if (isNull)
        return nullptr;
    return values.empty() ? &dummy : values.data();
}

inline void VerifyCsyr2kOutput(
    const Csyr2kParam& p, const aclblasComplex* actual, const aclblasComplex* golden, const aclblasComplex* input)
{
    std::vector<float> actualReal;
    std::vector<float> actualImag;
    std::vector<float> goldenReal;
    std::vector<float> goldenImag;
    std::vector<float> actualOtherReal;
    std::vector<float> actualOtherImag;
    std::vector<float> inputOtherReal;
    std::vector<float> inputOtherImag;
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.n; ++row) {
            size_t offset = static_cast<size_t>(col) * p.ldc + row;
            bool selected = p.uplo == ACLBLAS_UPPER ? row <= col : row >= col;
            if (selected) {
                actualReal.push_back(actual[offset].real);
                actualImag.push_back(actual[offset].imag);
                goldenReal.push_back(golden[offset].real);
                goldenImag.push_back(golden[offset].imag);
            } else {
                actualOtherReal.push_back(actual[offset].real);
                actualOtherImag.push_back(actual[offset].imag);
                inputOtherReal.push_back(input[offset].real);
                inputOtherImag.push_back(input[offset].imag);
            }
        }
    }

    VerifyConfig precisionReal;
    applyMixedTolerance(precisionReal, ACL_FLOAT, goldenReal.data(), goldenReal.size());
    EXPECT_TRUE(Verifier::verifyVector(
        actualReal.data(), goldenReal.data(), actualReal.size(), 1, precisionReal, p.caseName + "_uplo_real"));

    VerifyConfig precisionImag;
    applyMixedTolerance(precisionImag, ACL_FLOAT, goldenImag.data(), goldenImag.size());
    EXPECT_TRUE(Verifier::verifyVector(
        actualImag.data(), goldenImag.data(), actualImag.size(), 1, precisionImag, p.caseName + "_uplo_imag"));

    if (!actualOtherReal.empty()) {
        VerifyConfig exact;
        exact.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(
            actualOtherReal.data(), inputOtherReal.data(), actualOtherReal.size(), 1, exact,
            p.caseName + "_nonuplo_real"));
        EXPECT_TRUE(Verifier::verifyVector(
            actualOtherImag.data(), inputOtherImag.data(), actualOtherImag.size(), 1, exact,
            p.caseName + "_nonuplo_imag"));
    }
}

TEST_F(Csyr2kArch35Test, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    aclblasComplex data[16]{};
    EXPECT_EQ(
        aclblasCsyr2k(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, data, 4, data, 4, &beta, data, 4),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

struct Csyr2kPerformanceCase {
    int n;
    int k;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    double limitUs;
    uint32_t randomSeed;
};

struct Csyr2kPerformanceData {
    int rows = 0;
    std::vector<aclblasComplex> a;
    std::vector<aclblasComplex> b;
    std::vector<aclblasComplex> c;
    std::vector<aclblasComplex> inputC;
    std::vector<aclblasComplex> golden;
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    Csyr2kDeviceBuffers device;
};

inline void PrepareCsyr2kPerformanceData(const Csyr2kPerformanceCase& p, Csyr2kPerformanceData& data)
{
    data.rows = (p.trans == ACLBLAS_OP_N) ? p.n : p.k;
    int cols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;
    BlasFillMode officialFill("RANDOM_NORM_5_5");
    data.a = makeBlasComplexMatrix(data.rows, cols, data.rows, officialFill, p.randomSeed);
    data.b = makeBlasComplexMatrix(data.rows, cols, data.rows, officialFill, p.randomSeed + 1U);
    data.c = makeBlasComplexMatrix(p.n, p.n, p.n, officialFill, p.randomSeed + 2U);
    data.inputC = data.c;
    data.golden = data.c;
}

inline bool AllocateCsyr2kPerformanceData(Csyr2kPerformanceData& data)
{
    return Csyr2kAllocAndCopy(data.device.alpha, &data.alpha, sizeof(data.alpha)) == ACLBLAS_STATUS_SUCCESS &&
           Csyr2kAllocAndCopy(data.device.a, data.a.data(), data.a.size() * sizeof(aclblasComplex)) ==
               ACLBLAS_STATUS_SUCCESS &&
           Csyr2kAllocAndCopy(data.device.b, data.b.data(), data.b.size() * sizeof(aclblasComplex)) ==
               ACLBLAS_STATUS_SUCCESS &&
           Csyr2kAllocAndCopy(data.device.beta, &data.beta, sizeof(data.beta)) == ACLBLAS_STATUS_SUCCESS &&
           Csyr2kAllocAndCopy(data.device.c, data.c.data(), data.c.size() * sizeof(aclblasComplex)) ==
               ACLBLAS_STATUS_SUCCESS;
}

template <typename Launch>
inline double MeasureCsyr2kPerformance(Launch&& launch)
{
    constexpr int warmup = 5;
    constexpr int iterations = 51;
    for (int i = 0; i < warmup; ++i) {
        if (launch() != ACLBLAS_STATUS_SUCCESS) {
            return -1.0;
        }
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return -1.0;
    }
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (launch() != ACLBLAS_STATUS_SUCCESS) {
            return -1.0;
        }
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return -1.0;
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count() / iterations;
}

inline bool VerifyCsyr2kPerformanceResult(
    aclblasHandle_t handle, const Csyr2kPerformanceCase& p, Csyr2kPerformanceData& data)
{
    // Correctness validation is deliberately outside the timed interval.
    if (aclrtMemcpy(
            data.c.data(), data.c.size() * sizeof(aclblasComplex), data.device.c,
            data.c.size() * sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    if (aclblasCsyr2k_cpu(
            handle, p.uplo, p.trans, p.n, p.k, &data.alpha, data.a.data(), data.rows, data.b.data(), data.rows,
            &data.beta, data.golden.data(), p.n) != ACLBLAS_STATUS_SUCCESS) {
        return false;
    }
    Csyr2kParam verifyParam(csv_map{});
    verifyParam.caseName = "official_random_performance";
    verifyParam.uplo = p.uplo;
    verifyParam.trans = p.trans;
    verifyParam.n = p.n;
    verifyParam.k = p.k;
    verifyParam.ldc = p.n;
    VerifyCsyr2kOutput(verifyParam, data.c.data(), data.golden.data(), data.inputC.data());
    return true;
}

inline double RunCsyr2kPerformanceCase(aclblasHandle_t handle, const Csyr2kPerformanceCase& p)
{
    Csyr2kPerformanceData data;
    PrepareCsyr2kPerformanceData(p, data);
    if (!AllocateCsyr2kPerformanceData(data)) {
        return -1.0;
    }
    auto launch = [&]() {
        return aclblasCsyr2k(
            handle, p.uplo, p.trans, p.n, p.k, static_cast<const aclblasComplex*>(data.device.alpha),
            static_cast<const aclblasComplex*>(data.device.a), data.rows,
            static_cast<const aclblasComplex*>(data.device.b), data.rows,
            static_cast<const aclblasComplex*>(data.device.beta), static_cast<aclblasComplex*>(data.device.c), p.n);
    };
    double averageUs = MeasureCsyr2kPerformance(launch);
    if (averageUs < 0.0 || !VerifyCsyr2kPerformanceResult(handle, p, data)) {
        return -1.0;
    }
    return averageUs;
}

TEST_F(Csyr2kArch35Test, PerformanceRequiredCases)
{
    const char* runPerf = std::getenv("CSYR2K_RUN_PERF");
    if (runPerf == nullptr || std::strcmp(runPerf, "1") != 0) {
        GTEST_SKIP() << "Set CSYR2K_RUN_PERF=1 to run the required >50-sample performance cases";
    }
    const Csyr2kPerformanceCase cases[] = {
        {1024, 1024, ACLBLAS_UPPER, ACLBLAS_OP_N, 758.39, 20261001U},
        {2048, 2048, ACLBLAS_UPPER, ACLBLAS_OP_N, 3961.86, 20261002U},
        {1024, 1024, ACLBLAS_LOWER, ACLBLAS_OP_T, 815.78, 20261003U},
        {2048, 2048, ACLBLAS_LOWER, ACLBLAS_OP_T, 4283.30, 20261004U},
        {4096, 1024, ACLBLAS_UPPER, ACLBLAS_OP_N, 6753.94, 20261005U},
    };
    constexpr size_t caseCount = sizeof(cases) / sizeof(cases[0]);
    const char* selectedCase = std::getenv("CSYR2K_PERF_CASE");
    size_t selectedIndex = caseCount;
    if (selectedCase != nullptr) {
        char* parseEnd = nullptr;
        unsigned long parsedIndex = std::strtoul(selectedCase, &parseEnd, 10);
        ASSERT_NE(parseEnd, selectedCase) << "CSYR2K_PERF_CASE must be in [0, 4]";
        ASSERT_EQ(*parseEnd, '\0') << "CSYR2K_PERF_CASE must be in [0, 4]";
        ASSERT_LT(parsedIndex, caseCount) << "CSYR2K_PERF_CASE must be in [0, 4]";
        selectedIndex = static_cast<size_t>(parsedIndex);
    }
    size_t executedCount = 0;
    for (size_t caseIndex = 0; caseIndex < caseCount; ++caseIndex) {
        if (selectedCase != nullptr && caseIndex != selectedIndex) {
            continue;
        }
        ++executedCount;
        const auto& p = cases[caseIndex];
        double averageUs = RunCsyr2kPerformanceCase(handle_, p);
        std::cout << "[PERF][ascend950] aclblasCsyr2k n=" << p.n << " k=" << p.k << " uplo=" << static_cast<int>(p.uplo)
                  << " trans=" << static_cast<int>(p.trans) << " average_us=" << averageUs << " limit_us=" << p.limitUs
                  << std::endl;
        ASSERT_GT(averageUs, 0.0);
        EXPECT_LE(averageUs, p.limitUs);
    }
    ASSERT_GT(executedCount, 0U);
}

struct Csyr2kDirectStats {
    size_t selectedCount = 0;
    size_t selectedFailures = 0;
    size_t otherFailures = 0;
    size_t paddingFailures = 0;
    float maxAbsError = 0.0f;
};

inline bool Csyr2kDirectComponentMatches(float value, float reference, float error, bool expectExact)
{
    if (std::isnan(reference)) {
        return std::isnan(value);
    }
    if (std::isinf(reference)) {
        return value == reference;
    }
    if (!std::isfinite(value)) {
        return false;
    }
    return expectExact ? value == reference : error <= 1.0e-2f;
}

inline void CheckCsyr2kSelectedValue(
    const std::string& caseName, int row, int col, size_t offset, const std::vector<aclblasComplex>& actual,
    const std::vector<aclblasComplex>& golden, bool expectExact, Csyr2kDirectStats& stats)
{
    ++stats.selectedCount;
    float realError = std::fabs(actual[offset].real - golden[offset].real);
    float imagError = std::fabs(actual[offset].imag - golden[offset].imag);
    bool realMatch = Csyr2kDirectComponentMatches(actual[offset].real, golden[offset].real, realError, expectExact);
    bool imagMatch = Csyr2kDirectComponentMatches(actual[offset].imag, golden[offset].imag, imagError, expectExact);
    if (std::isfinite(realError)) {
        stats.maxAbsError = std::max(stats.maxAbsError, realError);
    }
    if (std::isfinite(imagError)) {
        stats.maxAbsError = std::max(stats.maxAbsError, imagError);
    }
    if (realMatch && imagMatch) {
        return;
    }
    if (stats.selectedFailures == 0) {
        std::cout << "[DIRECT] " << caseName << " first mismatch row=" << row << " col=" << col << " actual=("
                  << actual[offset].real << "," << actual[offset].imag << ") golden=(" << golden[offset].real << ","
                  << golden[offset].imag << ")" << std::endl;
    }
    ++stats.selectedFailures;
}

inline Csyr2kDirectStats CollectCsyr2kDirectStats(
    const std::string& caseName, aclblasFillMode_t uplo, int n, int ldc, const std::vector<aclblasComplex>& actual,
    const std::vector<aclblasComplex>& golden, const std::vector<aclblasComplex>& input, bool expectExact)
{
    Csyr2kDirectStats stats;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            size_t offset = static_cast<size_t>(col) * ldc + row;
            bool selected = uplo == ACLBLAS_UPPER ? row <= col : row >= col;
            if (selected) {
                CheckCsyr2kSelectedValue(caseName, row, col, offset, actual, golden, expectExact, stats);
            } else if (std::memcmp(&actual[offset], &input[offset], sizeof(aclblasComplex)) != 0) {
                ++stats.otherFailures;
            }
        }
        for (int row = n; row < ldc; ++row) {
            size_t offset = static_cast<size_t>(col) * ldc + row;
            if (std::memcmp(&actual[offset], &input[offset], sizeof(aclblasComplex)) != 0) {
                ++stats.paddingFailures;
            }
        }
    }
    return stats;
}

inline void VerifyCsyr2kDirectOutput(
    const std::string& caseName, aclblasFillMode_t uplo, int n, int ldc, const std::vector<aclblasComplex>& actual,
    const std::vector<aclblasComplex>& golden, const std::vector<aclblasComplex>& input, bool expectExact)
{
    Csyr2kDirectStats stats = CollectCsyr2kDirectStats(caseName, uplo, n, ldc, actual, golden, input, expectExact);
    double matchedRatio =
        stats.selectedCount == 0 ? 1.0 : 1.0 - static_cast<double>(stats.selectedFailures) / stats.selectedCount;
    if (expectExact) {
        EXPECT_EQ(stats.selectedFailures, 0U)
            << caseName << " must be numerically exact; maxAbsError=" << stats.maxAbsError;
    } else {
        EXPECT_GE(matchedRatio, 0.99) << caseName << " maxAbsError=" << stats.maxAbsError;
    }
    EXPECT_EQ(stats.otherFailures, 0U) << caseName << " changed the unselected triangle";
    EXPECT_EQ(stats.paddingFailures, 0U) << caseName << " changed C leading-dimension padding";
}

inline aclblasComplex Csyr2kExactGridValue(int row, int col, int salt)
{
    constexpr float grid[] = {-0.5f, -0.375f, -0.25f, -0.125f, 0.0f, 0.125f, 0.25f, 0.375f, 0.5f};
    constexpr int gridSize = sizeof(grid) / sizeof(grid[0]);
    int realIndex = (row * (salt + 1) + col * (salt + 3) + salt) % gridSize;
    int imagIndex = (row * (salt + 4) + col * (salt + 2) + 2 * salt + 1) % gridSize;
    return {grid[realIndex], grid[imagIndex]};
}

inline void RunCsyr2kDirectCorrectnessCase(
    aclblasHandle_t handle, const std::string& caseName, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    int ldPadding, bool useExactGridPattern, bool expectExact, aclblasComplex aValue, aclblasComplex bValue,
    aclblasComplex alpha, aclblasComplex beta, bool poisonLastLogicalComponent = false)
{
    int rows = (trans == ACLBLAS_OP_N) ? n : k;
    int cols = (trans == ACLBLAS_OP_N) ? k : n;
    int lda = rows + ldPadding;
    int ldb = rows + ldPadding + 1;
    int ldc = n + ldPadding + 2;
    float quietNaN = std::numeric_limits<float>::quiet_NaN();
    float infinity = std::numeric_limits<float>::infinity();
    // Poison every leading-dimension padding slot.  Logical values are filled
    // below, so an incorrect compact-stride read cannot accidentally pass.
    std::vector<aclblasComplex> A(static_cast<size_t>(lda) * cols, {quietNaN, infinity});
    std::vector<aclblasComplex> B(static_cast<size_t>(ldb) * cols, {-infinity, quietNaN});
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            A[static_cast<size_t>(col) * lda + row] = useExactGridPattern ? Csyr2kExactGridValue(row, col, 1) : aValue;
            B[static_cast<size_t>(col) * ldb + row] = useExactGridPattern ? Csyr2kExactGridValue(row, col, 5) : bValue;
        }
    }
    if (poisonLastLogicalComponent) {
        // Binary-exact in FP32 but intentionally outside the exact-fast range;
        // an erroneous FP16 conversion changes it by 0.25 and cannot hide
        // behind the normal 99% matched-ratio tolerance.
        A[static_cast<size_t>(cols - 1) * lda + (rows - 1)].real = 1024.25f;
    }
    std::vector<aclblasComplex> C(static_cast<size_t>(ldc) * n, {quietNaN, -infinity});
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            C[static_cast<size_t>(col) * ldc + row] = Csyr2kExactGridValue(row, col, 3);
        }
    }
    std::vector<aclblasComplex> input = C;
    std::vector<aclblasComplex> golden = C;

    ASSERT_EQ(
        aclblasCsyr2k_cpu(handle, uplo, trans, n, k, &alpha, A.data(), lda, B.data(), ldb, &beta, golden.data(), ldc),
        ACLBLAS_STATUS_SUCCESS)
        << caseName;
    ASSERT_EQ(
        aclblasCsyr2k_npu(handle, uplo, trans, n, k, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc),
        ACLBLAS_STATUS_SUCCESS)
        << caseName;
    VerifyCsyr2kDirectOutput(caseName, uplo, n, ldc, C, golden, input, expectExact);
}

TEST_F(Csyr2kArch35Test, ExactFastAndFallbackCorrectnessCases)
{
    constexpr int n = 513;
    constexpr int k = 515;
    float quietNaN = std::numeric_limits<float>::quiet_NaN();
    float infinity = std::numeric_limits<float>::infinity();
    float subnormal = std::numeric_limits<float>::denorm_min();
    // Reuse one handle/workspace in invalid -> valid -> invalid order.  This
    // covers asynchronous flag clearing, partial 64-row/column tiles, padded
    // leading dimensions and all N/T/C routing modes.
    RunCsyr2kDirectCorrectnessCase(
        handle_, "off_grid_fallback_N", ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, 3, false, false, {0.1f, 0.3f}, {-0.2f, 0.4f},
        {1.0f, 0.0f}, {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "exact_grid_fast_N", ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, 4, true, true, {}, {}, {1.0f, 0.0f},
        {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "exact_grid_fast_T", ACLBLAS_LOWER, ACLBLAS_OP_T, n, k, 5, true, true, {}, {}, {1.0f, 0.0f},
        {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "exact_grid_fast_C", ACLBLAS_UPPER, ACLBLAS_OP_C, n, k, 6, true, true, {}, {}, {1.0f, 0.0f},
        {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "exact_grid_complex_combine", ACLBLAS_LOWER, ACLBLAS_OP_N, n, k, 6, true, true, {}, {}, {0.5f, 0.25f},
        {-0.25f, 0.125f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "last_component_off_grid_fallback_N", ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, 3, true, true, {}, {},
        {1.0f, 0.0f}, {0.0f, 0.0f}, true);
    RunCsyr2kDirectCorrectnessCase(
        handle_, "range_fallback_C", ACLBLAS_UPPER, ACLBLAS_OP_C, n, k, 7, false, false, {0.625f, 0.1f},
        {-0.375f, 0.5f}, {1.0f, 0.0f}, {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "subnormal_fallback_N", ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, 2, false, false, {subnormal, 0.0f},
        {0.5f, 0.5f}, {1.0f, 0.0f}, {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "nan_fallback_T", ACLBLAS_LOWER, ACLBLAS_OP_T, n, k, 2, false, false, {quietNaN, 0.0f},
        {0.125f, 0.25f}, {1.0f, 0.0f}, {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "inf_fallback_C", ACLBLAS_UPPER, ACLBLAS_OP_C, n, k, 2, false, false, {infinity, 0.0f}, {0.0f, 0.0f},
        {1.0f, 0.0f}, {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "k4096_exact_boundary_N", ACLBLAS_UPPER, ACLBLAS_OP_N, 512, 4096, 1, true, true, {}, {}, {1.0f, 0.0f},
        {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "k4097_strict_boundary_N", ACLBLAS_LOWER, ACLBLAS_OP_N, 512, 4097, 1, true, true, {}, {}, {1.0f, 0.0f},
        {0.0f, 0.0f});
}

TEST_F(Csyr2kArch35Test, MixEpilogueGuardCorrectnessCases)
{
    constexpr int n = 2048;
    constexpr int k = 2048;
    // These cases deliberately match the q8 MIX launch shape while forcing
    // each guarded route.  Power-of-two inputs keep the strict references
    // binary-exact, so a missed guard cannot hide behind a tolerance.
    RunCsyr2kDirectCorrectnessCase(
        handle_, "mix_q8_range_fallback", ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, 0, false, true, {65536.0f, 0.0f},
        {0.125f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f});
    RunCsyr2kDirectCorrectnessCase(
        handle_, "mix_q8_alpha_fallback", ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, 0, false, true, {0.5f, 0.0f},
        {0.25f, 0.0f}, {0.5f, 0.25f}, {0.0f, 0.0f});
    for (int repeat = 0; repeat < 2; ++repeat) {
        RunCsyr2kDirectCorrectnessCase(
            handle_, "mix_q8_beta_full_combine_" + std::to_string(repeat), ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, 0, false,
            true, {0.5f, 0.0f}, {0.25f, 0.0f}, {1.0f, 0.0f}, {0.5f, -0.25f});
        RunCsyr2kDirectCorrectnessCase(
            handle_, "mix_q8_post_beta_exact_" + std::to_string(repeat), ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, 0, false,
            true, {0.5f, 0.0f}, {0.25f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f});
    }
}

std::string GetCsyr2kCsvPath()
{
    std::string csvPath = ReplaceFileExtension2Csv(__FILE__);
    const char* useFullCsv = std::getenv("CSYR2K_FULL_CSV");
    if (useFullCsv == nullptr || std::strcmp(useFullCsv, "1") != 0) {
        csvPath.insert(csvPath.find_last_of('.'), "_smoke");
    }
    return csvPath;
}

INSTANTIATE_TEST_SUITE_P(
    Csyr2k, Csyr2kArch35Test, ::testing::ValuesIn(GetCasesFromCsv<Csyr2kParam>(GetCsyr2kCsvPath())),
    PrintCaseInfoString<Csyr2kParam>);

TEST_P(Csyr2kArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const char* runCsvPerf = std::getenv("CSYR2K_RUN_CSV_PERF");
    if (runCsvPerf != nullptr && std::strcmp(runCsvPerf, "1") == 0 && p.caseName.rfind("TC_PF_", 0) == 0) {
        Csyr2kPerformanceCase perfCase{p.n, p.k, p.uplo, p.trans, 0.0, p.randomSeed};
        double averageUs = RunCsyr2kPerformanceCase(handle_, perfCase);
        std::cout << "[PERF][csv] case=" << p.caseName << " n=" << p.n << " k=" << p.k
                  << " uplo=" << static_cast<int>(p.uplo) << " trans=" << static_cast<int>(p.trans)
                  << " average_us=" << averageUs << std::endl;
        ASSERT_GT(averageUs, 0.0);
        return;
    }
    Csyr2kHostData data;
    PrepareCsyr2kHostData(p, data);

    const aclblasComplex* alpha = p.nullAlpha ? nullptr : &p.alpha;
    const aclblasComplex* beta = p.nullBeta ? nullptr : &p.beta;
    const aclblasComplex* A = Csyr2kInputPointer(p.nullA, data.a, data.dummyA);
    const aclblasComplex* B = Csyr2kInputPointer(p.nullB, data.b, data.dummyB);
    aclblasComplex* C = Csyr2kOutputPointer(p.nullC, data.c, data.dummyC);

    aclblasStatus_t ret = aclblasCsyr2k_npu(
        Csyr2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k, alpha, A, p.lda, B, p.ldb, beta, C, p.ldc);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS || p.n == 0)
        return;

    aclblasComplex* golden = data.golden.empty() ? &data.dummyC : data.golden.data();
    const aclblasComplex* input = data.cInput.empty() ? &data.dummyC : data.cInput.data();
    aclblasStatus_t goldenRet = aclblasCsyr2k_cpu(
        Csyr2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k, alpha, A, p.lda, B, p.ldb, beta, golden, p.ldc);
    ASSERT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS);
    VerifyCsyr2kOutput(p, C, golden, input);
}
