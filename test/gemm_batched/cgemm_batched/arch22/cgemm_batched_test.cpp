/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cgemm_batched_param.h"
#include "cgemm_batched_golden.h"
#include "cgemm_batched_npu_wrapper.h"

struct CgemmBatchedTestData {
    std::vector<std::vector<aclblasComplex>> aBatch;
    std::vector<std::vector<aclblasComplex>> bBatch;
    std::vector<std::vector<aclblasComplex>> cBatch;
    std::vector<std::vector<aclblasComplex>> cGolden;
};

// Reproducible additional input sets without changing CSVs or fill semantics.
inline uint32_t CgemmBatchedSeedOffset()
{
    static const uint32_t offset = []() {
        const char* value = std::getenv("CGEMM_BATCHED_TEST_SEED_OFFSET");
        if (value == nullptr)
            return uint32_t{0};
        const std::string text(value);
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("CGEMM_BATCHED_TEST_SEED_OFFSET must be a uint32 decimal");
        const auto parsed = std::stoull(text);
        if (parsed > std::numeric_limits<uint32_t>::max())
            throw std::out_of_range("CGEMM_BATCHED_TEST_SEED_OFFSET exceeds uint32");
        return static_cast<uint32_t>(parsed);
    }();
    return offset;
}

// Generate complex array from a float fill mode: use two independent float arrays for real/imag.
inline std::vector<aclblasComplex> makeComplexArray(int64_t size, const BlasFillMode& fill, uint32_t seed)
{
    seed += CgemmBatchedSeedOffset();
    if (fill.method == BlasFillMode::M_NULLPTR || size <= 0) {
        return {};
    }
    auto makePart = [&](uint32_t partSeed) {
        if (fill.method != BlasFillMode::M_GAUSS)
            return makeBlasArray(size, fill, partSeed);
        // GEMM has no conditioning constraint: retain the normal distribution's
        // tails rather than using the framework's triangular-solver clipping.
        std::mt19937 random(partSeed);
        std::uniform_real_distribution<float> mean(-5.0f, 5.0f);
        std::uniform_real_distribution<float> deviation(0.1f, 2.0f);
        const float mu = mean(random);
        const float sigma = deviation(random);
        std::normal_distribution<float> normal(mu, sigma);
        std::vector<float> part(static_cast<size_t>(size));
        for (auto& value : part)
            value = normal(random);
        return part;
    };
    std::vector<float> realPart = makePart(seed);
    std::vector<float> imagPart = makePart(seed + 7919);
    std::vector<aclblasComplex> data(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

inline CgemmBatchedTestData GenerateCgemmBatchedTestData(const CgemmBatchedParam& p)
{
    CgemmBatchedTestData data;
    int safeBatch = std::max(0, p.batchCount);
    int physRowsA = cgemmBatchedPhysRows(p.m, p.k, p.transA);
    int physColsA = cgemmBatchedPhysCols(p.m, p.k, p.transA);
    int physRowsB = cgemmBatchedPhysRows(p.k, p.n, p.transB);
    int physColsB = cgemmBatchedPhysCols(p.k, p.n, p.transB);

    data.aBatch.resize(safeBatch);
    data.bBatch.resize(safeBatch);
    data.cBatch.resize(safeBatch);
    data.cGolden.resize(safeBatch);

    for (int b = 0; b < safeBatch; b++) {
        uint32_t seed = p.randomSeed + static_cast<uint32_t>(b * 3);
        int64_t aSize = static_cast<int64_t>(std::max(1, p.lda)) * std::max(1, physColsA);
        int64_t bSize = static_cast<int64_t>(std::max(1, p.ldb)) * std::max(1, physColsB);
        int64_t cSize = static_cast<int64_t>(std::max(1, p.ldc)) * std::max(1, p.n);
        data.aBatch[b] = makeComplexArray(aSize, p.aFill, seed);
        data.bBatch[b] = makeComplexArray(bSize, p.bFill, seed + 1);
        data.cBatch[b] = makeComplexArray(cSize, p.cFill, seed + 2);
        data.cGolden[b] = data.cBatch[b];
    }
    return data;
}

struct CgemmBatchedPtrArrays {
    std::vector<const aclblasComplex*> aPtrs;
    std::vector<const aclblasComplex*> bPtrs;
    std::vector<aclblasComplex*> cPtrs;
};

class CgemmBatchedTest : public BlasTest<CgemmBatchedParam> {};

TEST_F(CgemmBatchedTest, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCgemmBatched(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, nullptr, 8, nullptr, 8, &beta, nullptr, 8, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(CgemmBatchedTest, KZeroIgnoresNonFiniteAlpha)
{
    aclblasComplex dummy{0.0f, 0.0f};
    const aclblasComplex* aArray[] = {&dummy};
    const aclblasComplex* bArray[] = {&dummy};
    std::vector<aclblasComplex> c = {{1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}, {-7.0f, -8.0f}};
    aclblasComplex* cArray[] = {c.data()};
    const std::vector<aclblasComplex> input = c;
    aclblasComplex alpha{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    aclblasComplex beta{0.5f, -0.25f};

    aclblasStatus_t ret = aclblasCgemmBatched_npu(
        CgemmBatchedTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 2, 2, 0, &alpha, aArray, 2, bArray, 1, &beta, cArray, 2,
        1);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    for (size_t i = 0; i < c.size(); i++) {
        float expectedReal = beta.real * input[i].real - beta.imag * input[i].imag;
        float expectedImag = beta.real * input[i].imag + beta.imag * input[i].real;
        EXPECT_FLOAT_EQ(c[i].real, expectedReal);
        EXPECT_FLOAT_EQ(c[i].imag, expectedImag);
    }
}

struct CgemmBatchedPerfBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    void* aPtrs = nullptr;
    void* bPtrs = nullptr;
    void* cPtrs = nullptr;

    ~CgemmBatchedPerfBuffers()
    {
        if (a != nullptr)
            aclrtFree(a);
        if (b != nullptr)
            aclrtFree(b);
        if (c != nullptr)
            aclrtFree(c);
        if (aPtrs != nullptr)
            aclrtFree(aPtrs);
        if (bPtrs != nullptr)
            aclrtFree(bPtrs);
        if (cPtrs != nullptr)
            aclrtFree(cPtrs);
    }
};

inline void InitializeCgemmBatchedPerfInput(
    void* destination, size_t matrixBytes, const BlasFillMode& fill, uint32_t seed, int batchCount, bool zeros,
    uint64_t& fingerprint)
{
    if (zeros) {
        ASSERT_EQ(aclrtMemset(destination, matrixBytes * batchCount, 0, matrixBytes * batchCount), ACL_SUCCESS);
        return;
    }
    // One matrix at a time bounds host memory. All generation, fingerprinting
    // and H2D copies happen before warmup and outside the timed interval.
    for (int batch = 0; batch < batchCount; ++batch) {
        auto data = makeComplexArray(matrixBytes / sizeof(aclblasComplex), fill, seed + 3U * batch);
        ASSERT_EQ(data.size() * sizeof(aclblasComplex), matrixBytes);
        const auto* bytes = reinterpret_cast<const uint8_t*>(data.data());
        for (size_t i = 0; i < matrixBytes; ++i) {
            fingerprint ^= bytes[i];
            fingerprint *= 1099511628211ULL;
        }
        ASSERT_EQ(
            aclrtMemcpy(
                static_cast<uint8_t*>(destination) + matrixBytes * batch, matrixBytes, data.data(), matrixBytes,
                ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
    }
}

inline void AllocateCgemmBatchedPerfBuffers(
    CgemmBatchedPerfBuffers& buffers, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    int lda, int ldb, int ldc, int batchCount, const CgemmBatchedParam* param = nullptr)
{
    const auto sizes = ComputeCgemmBatchedBufferSizes(transA, transB, m, n, k, lda, ldb, ldc);
    const size_t matrixABytes = sizes.aBytes;
    const size_t matrixBBytes = sizes.bBytes;
    const size_t matrixCBytes = sizes.cBytes;
    const size_t allABytes = matrixABytes * batchCount;
    const size_t allBBytes = matrixBBytes * batchCount;
    const size_t allCBytes = matrixCBytes * batchCount;

    ASSERT_EQ(aclrtMalloc(&buffers.a, allABytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.b, allBBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.c, allCBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    const char* zeroInput = std::getenv("CGEMM_BATCHED_PERF_ZERO_INPUT");
    const bool zeros = zeroInput != nullptr && std::string(zeroInput) == "1";
    const uint32_t baseSeed = param != nullptr ? param->randomSeed : 20261012U;
    uint64_t fingerprint = 14695981039346656037ULL;
    const BlasFillMode defaultFill("RANDOM_NORM_5_5");
    InitializeCgemmBatchedPerfInput(
        buffers.a, matrixABytes, param != nullptr ? param->aFill : defaultFill, baseSeed, batchCount, zeros,
        fingerprint);
    InitializeCgemmBatchedPerfInput(
        buffers.b, matrixBBytes, param != nullptr ? param->bFill : defaultFill, baseSeed + 1U, batchCount, zeros,
        fingerprint);
    InitializeCgemmBatchedPerfInput(
        buffers.c, matrixCBytes, param != nullptr ? param->cFill : defaultFill, baseSeed + 2U, batchCount, zeros,
        fingerprint);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    std::cout << "[CGEMM_BATCHED_INPUT] base_seed=" << baseSeed << " seed_offset=" << CgemmBatchedSeedOffset()
              << " mode=" << (zeros ? "zero" : "csv_fill") << " fnv1a64=" << fingerprint << std::endl;

    std::vector<const aclblasComplex*> aPtrs(batchCount);
    std::vector<const aclblasComplex*> bPtrs(batchCount);
    std::vector<aclblasComplex*> cPtrs(batchCount);
    for (int batch = 0; batch < batchCount; batch++) {
        aPtrs[batch] = reinterpret_cast<const aclblasComplex*>(static_cast<uint8_t*>(buffers.a) + matrixABytes * batch);
        bPtrs[batch] = reinterpret_cast<const aclblasComplex*>(static_cast<uint8_t*>(buffers.b) + matrixBBytes * batch);
        cPtrs[batch] = reinterpret_cast<aclblasComplex*>(static_cast<uint8_t*>(buffers.c) + matrixCBytes * batch);
    }

    const size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    ASSERT_EQ(aclrtMalloc(&buffers.aPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.bPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.cPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(buffers.aPtrs, ptrBytes, aPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(buffers.bPtrs, ptrBytes, bPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(buffers.cPtrs, ptrBytes, cPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
}

inline double RunCgemmBatchedPerformanceCase(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n,
    int k, int lda, int ldb, int ldc, int batchCount, const CgemmBatchedParam* param = nullptr)
{
    constexpr int warmups = 5;
    constexpr int samples = 60;
    size_t freeBefore = 0;
    size_t totalBefore = 0;
    EXPECT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &freeBefore, &totalBefore), ACL_SUCCESS);
    CgemmBatchedPerfBuffers buffers;
    AllocateCgemmBatchedPerfBuffers(buffers, transA, transB, m, n, k, lda, ldb, ldc, batchCount, param);
    if (::testing::Test::HasFatalFailure())
        return std::numeric_limits<double>::infinity();
    const auto* aPtrs = reinterpret_cast<const aclblasComplex* const*>(buffers.aPtrs);
    const auto* bPtrs = reinterpret_cast<const aclblasComplex* const*>(buffers.bPtrs);
    auto* cPtrs = reinterpret_cast<aclblasComplex* const*>(buffers.cPtrs);
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    auto launch = [&]() {
        return aclblasCgemmBatched(
            handle, transA, transB, m, n, k, &alpha, aPtrs, lda, bPtrs, ldb, &beta, cPtrs, ldc, batchCount);
    };

    for (int i = 0; i < warmups; i++) {
        EXPECT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
    }
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    size_t freeAfter = 0;
    size_t totalAfter = 0;
    EXPECT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &freeAfter, &totalAfter), ACL_SUCCESS);
    std::cout << "[CGEMM_BATCHED_MEMORY] m=" << m << " n=" << n << " k=" << k << " batchCount=" << batchCount
              << " device_used_bytes=" << totalAfter - freeAfter
              << " allocation_delta_bytes=" << static_cast<int64_t>(freeBefore) - static_cast<int64_t>(freeAfter)
              << std::endl;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < samples; i++) {
        EXPECT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
    }
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / samples;
}

inline double RunCgemmBatchedPerformanceCase(
    aclblasHandle_t handle, aclrtStream stream, int m, int n, int k, int batchCount)
{
    return RunCgemmBatchedPerformanceCase(handle, stream, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, m, k, m, batchCount);
}

inline char CgemmBatchedOperationChar(aclblasOperation_t operation)
{
    if (operation == ACLBLAS_OP_T)
        return 'T';
    if (operation == ACLBLAS_OP_C)
        return 'C';
    return 'N';
}

TEST_F(CgemmBatchedTest, TC_PF_PerformanceContract)
{
    // Task-book 3.3 bars: average single-call time on Atlas 800T A2 (910B3)
    // must not exceed the benchmark value (warmup, then >50 samples averaged).
    struct PerfCase {
        const char* name;
        int m;
        int n;
        int k;
        int batchCount;
        double benchmarkUs;
    };
    const PerfCase cases[] = {
        {"256x256x256_b32_NN", 256, 256, 256, 32, 342.44},
        {"512x512x512_b16_NN", 512, 512, 512, 16, 1462.32},
        {"1024x1024x1024_b8_NN", 1024, 1024, 1024, 8, 5018.64},
        {"2048x2048x2048_b4_NN", 2048, 2048, 2048, 4, 18032.75},
    };

    for (const auto& perf : cases) {
        const double limitUs = perf.benchmarkUs;
        const double averageUs =
            RunCgemmBatchedPerformanceCase(handle_, stream_, perf.m, perf.n, perf.k, perf.batchCount);
        std::cout << std::fixed << std::setprecision(3) << "[CGEMM_BATCHED_PERF] case=" << perf.name
                  << " soc=" << aclrtGetSocName() << " device_id=" << blas_test_detail::RuntimeDeviceId()
                  << " backend=npu warmups=5 samples=60"
                  << " average_us=" << averageUs << " benchmark_us=" << perf.benchmarkUs << " limit_us=" << limitUs
                  << std::endl;
        EXPECT_LE(averageUs, limitUs) << perf.name;
    }
}

TEST_P(CgemmBatchedTest, CsvPerformance)
{
    const auto& p = GetParam();
    const char* enabled = std::getenv("CGEMM_BATCHED_RUN_ALL_PERF");
    if (enabled == nullptr || std::string(enabled) != "1") {
        GTEST_SKIP() << "set CGEMM_BATCHED_RUN_ALL_PERF=1 to run CSV performance cases";
    }
    if (p.caseName.rfind("TC_PF_", 0) != 0) {
        GTEST_SKIP() << "not a TC_PF case";
    }

    const double averageUs = RunCgemmBatchedPerformanceCase(
        CgemmBatchedTest::handle_, CgemmBatchedTest::stream_, p.transA, p.transB, p.m, p.n, p.k, p.lda, p.ldb, p.ldc,
        p.batchCount, &p);
    std::cout << std::fixed << std::setprecision(3) << "[CGEMM_BATCHED_CSV_PERF] case=" << p.caseName
              << " soc=" << aclrtGetSocName() << " device_id=" << blas_test_detail::RuntimeDeviceId()
              << " backend=npu warmups=5 samples=60"
              << " transA=" << CgemmBatchedOperationChar(p.transA) << " transB=" << CgemmBatchedOperationChar(p.transB)
              << " m=" << p.m << " n=" << p.n << " k=" << p.k << " lda=" << p.lda << " ldb=" << p.ldb
              << " ldc=" << p.ldc << " batchCount=" << p.batchCount << " average_us=" << averageUs << std::endl;
}

// Keep base coverage and every targeted regression in CI. Acceptance runners
// explicitly enable the full CSV, including expensive CPU golden calculations.
inline std::vector<CgemmBatchedParam> GetCgemmBatchedCases()
{
    auto cases = GetCasesFromCsv<CgemmBatchedParam>(ReplaceFileExtension2Csv(__FILE__));
    const char* fullCsv = std::getenv("CGEMM_BATCHED_FULL_CSV");
    const char* performance = std::getenv("CGEMM_BATCHED_RUN_ALL_PERF");
    if ((fullCsv != nullptr && std::string(fullCsv) == "1") ||
        (performance != nullptr && std::string(performance) == "1")) {
        return cases;
    }
    cases.erase(
        std::remove_if(
            cases.begin(), cases.end(),
            [](const CgemmBatchedParam& param) {
                return param.caseName.find("TC_EX_") == 0 || param.caseName.find("TC_PF_") == 0;
            }),
        cases.end());
    return cases;
}

TEST(CgemmBatchedCsvTest, CsvCaseCount)
{
    const auto mainCases = GetCasesFromCsv<CgemmBatchedParam>(ReplaceFileExtension2Csv(__FILE__));
    const auto regression = GetCasesFromCsv<CgemmBatchedParam>(
        std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of('/')) + "/cgemm_batched_regression.csv");
    ASSERT_EQ(mainCases.size(), 1200U);
    ASSERT_EQ(regression.size(), 669U);
    std::set<std::string> names;
    size_t extended = 0, performance = 0;
    for (const auto& c : mainCases) {
        EXPECT_TRUE(names.insert(c.caseName).second) << c.caseName;
        extended += c.caseName.find("TC_EX_") == 0;
        performance += c.caseName.find("TC_PF_") == 0;
    }
    EXPECT_EQ(extended, 755U);
    EXPECT_EQ(performance, 200U);
    EXPECT_EQ(mainCases.size() - extended - performance, 245U);
    for (const auto& c : regression)
        EXPECT_TRUE(names.insert(c.caseName).second) << c.caseName;
}

INSTANTIATE_TEST_SUITE_P(
    CgemmBatched, CgemmBatchedTest, ::testing::ValuesIn(GetCgemmBatchedCases()),
    PrintCaseInfoString<CgemmBatchedParam>);

INSTANTIATE_TEST_SUITE_P(
    CgemmBatchedRegression, CgemmBatchedTest,
    ::testing::ValuesIn(GetCasesFromCsv<CgemmBatchedParam>(
        std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of('/')) + "/cgemm_batched_regression.csv")),
    PrintCaseInfoString<CgemmBatchedParam>);

// Split independent output columns between CPU workers. Each element still uses
// the unmodified Netlib cblas_cgemm reduction; no alternative golden is used.
inline void ComputeCgemmBatchedGolden(
    const CgemmBatchedParam& p, CgemmBatchedTestData& data, int batches, aclblasHandle_t handle)
{
    const int columnsPerJob = 128;
    const int groups = (p.n + columnsPerJob - 1) / columnsPerJob;
    const int jobs = batches * groups;
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (int job = next.fetch_add(1); job < jobs; job = next.fetch_add(1)) {
            const int batch = job / groups;
            const int column = (job % groups) * columnsPerJob;
            const int count = std::min(columnsPerJob, p.n - column);
            const aclblasComplex* a[] = {data.aBatch[batch].data()};
            const aclblasComplex* b[] = {
                data.bBatch[batch].data() + (p.transB == ACLBLAS_OP_N ? static_cast<size_t>(column) * p.ldb : column)};
            aclblasComplex* c[] = {data.cGolden[batch].data() + static_cast<size_t>(column) * p.ldc};
            aclblasCgemmBatched_cpu(
                handle, p.transA, p.transB, p.m, count, p.k, &p.alpha, a, p.lda, b, p.ldb, &p.beta, c, p.ldc, 1);
        }
    };
    const uint64_t work = static_cast<uint64_t>(batches) * p.m * p.n * p.k;
    const int threads = work > 10000000 ? std::min(8, jobs) : 1;
    std::vector<std::thread> workers;
    for (int i = 1; i < threads; ++i)
        workers.emplace_back(worker);
    worker();
    for (auto& thread : workers)
        thread.join();
}

inline void CompareCgemmBatchedComponent(
    const CgemmBatchedParam& p, int b, size_t index, int part, bool exact, float actual, float expected,
    size_t& matched, size_t& hardFailures, double& maxError)
{
    if (exact && std::isfinite(actual) && std::isfinite(expected)) {
        const bool equal = std::memcmp(&actual, &expected, sizeof(float)) == 0;
        matched += equal;
        hardFailures += !equal;
    } else if (!std::isfinite(actual) || !std::isfinite(expected)) {
        const bool equal = (std::isnan(actual) && std::isnan(expected)) || actual == expected;
        if (!equal && hardFailures == 0) {
            std::cout << "[SPECIAL_MISMATCH] case=" << p.caseName << " batch=" << b << " index=" << index
                      << " part=" << part << " actual=" << actual << " expected=" << expected << std::endl;
        }
        matched += equal;
        hardFailures += !equal;
    } else {
        const double diff = std::abs(double(actual) - expected);
        const double magnitude = std::abs(double(expected));
        if (diff > std::max(1e-2, 32.0 * getUlpsAt(magnitude, 23, -126)) && hardFailures == 0) {
            std::cout << "[FINITE_MISMATCH] case=" << p.caseName << " batch=" << b << " index=" << index
                      << " part=" << part << " actual=" << actual << " expected=" << expected << std::endl;
        }
        matched += diff <= std::ldexp(1.0, -16) + std::ldexp(1.0, -10) * magnitude;
        hardFailures += diff > std::max(1e-2, 32.0 * getUlpsAt(magnitude, 23, -126));
        maxError = std::max(maxError, diff);
    }
}

inline void VerifyCgemmBatchedPrecision(
    const CgemmBatchedParam& p, CgemmBatchedTestData& data, CgemmBatchedPtrArrays& ptrs, int safeBatch,
    aclblasHandle_t testHandle)
{
    ComputeCgemmBatchedGolden(p, data, safeBatch, testHandle);
    const bool exact = p.k == 0 || (p.alpha.real == 0.0f && p.alpha.imag == 0.0f);
    size_t totalMatched[2] = {0, 0};
    double minRatio[2] = {1.0, 1.0};
    double maxError[2] = {0.0, 0.0};
    for (int b = 0; b < safeBatch; ++b) {
        ASSERT_NE(ptrs.cPtrs[b], nullptr);
        size_t matched[2] = {0, 0};
        size_t hardFailures[2] = {0, 0};
        size_t paddingFailures = 0;
        for (int col = 0; col < p.n; ++col) {
            for (int row = 0; row < p.ldc; ++row) {
                const size_t index = static_cast<size_t>(col) * p.ldc + row;
                const auto& out = data.cBatch[b][index];
                const auto& gold = data.cGolden[b][index];
                if (row >= p.m) {
                    paddingFailures += std::memcmp(&out, &gold, sizeof(out)) != 0;
                    continue;
                }
                const float actual[] = {out.real, out.imag};
                const float expected[] = {gold.real, gold.imag};
                for (int part = 0; part < 2; ++part) {
                    CompareCgemmBatchedComponent(
                        p, b, index, part, exact, actual[part], expected[part], matched[part], hardFailures[part],
                        maxError[part]);
                }
            }
        }
        EXPECT_EQ(paddingFailures, 0U) << p.caseName << " batch=" << b << " C padding changed";
        for (int part = 0; part < 2; ++part) {
            const double ratio = double(matched[part]) / (static_cast<size_t>(p.m) * p.n);
            minRatio[part] = std::min(minRatio[part], ratio);
            totalMatched[part] += matched[part];
            EXPECT_EQ(hardFailures[part], 0U) << p.caseName << " batch=" << b << " part=" << part;
        }
    }
    for (int part = 0; part < 2; ++part) {
        const double ratio = double(totalMatched[part]) / (static_cast<uint64_t>(safeBatch) * p.m * p.n);
        EXPECT_GE(ratio, exact ? 1.0 : 0.99) << p.caseName << " part=" << part;
    }
    std::cout << "[CGEMM_BATCHED_PRECISION] case=" << p.caseName << " exact=" << exact
              << " matched_ratio_real=" << double(totalMatched[0]) / (static_cast<uint64_t>(safeBatch) * p.m * p.n)
              << " matched_ratio_imag=" << double(totalMatched[1]) / (static_cast<uint64_t>(safeBatch) * p.m * p.n)
              << " min_batch_ratio_real=" << minRatio[0] << " min_batch_ratio_imag=" << minRatio[1]
              << " max_abs_real=" << maxError[0] << " max_abs_imag=" << maxError[1] << std::endl;
}

TEST_P(CgemmBatchedTest, CsvDriven)
{
    const auto& p = GetParam();
    RecordProperty("seed_offset", std::to_string(CgemmBatchedSeedOffset()));
    RecordProperty("base_seed", std::to_string(p.randomSeed));

    aclblasHandle_t testHandle = CgemmBatchedTest::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    int safeBatch = std::max(0, p.batchCount);
    auto data = GenerateCgemmBatchedTestData(p);
    auto ptrs =
        BuildGemmBatchedPtrsTpl<CgemmBatchedTestData, CgemmBatchedParam, CgemmBatchedPtrArrays>(data, p, safeBatch);

    const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const aclblasComplex* betaPtr = p.betaNull ? nullptr : &p.beta;
    const aclblasComplex* const* aPtrArr = (p.aarrayNull || safeBatch == 0) ? nullptr : ptrs.aPtrs.data();
    const aclblasComplex* const* bPtrArr = (p.barrayNull || safeBatch == 0) ? nullptr : ptrs.bPtrs.data();
    aclblasComplex* const* cPtrArr = (p.carrayNull || safeBatch == 0) ? nullptr : ptrs.cPtrs.data();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasCgemmBatched_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtrArr, p.lda, bPtrArr, p.ldb, betaPtr, cPtrArr,
            p.ldc, p.batchCount);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasCgemmBatched_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtrArr, p.lda, bPtrArr, p.ldb, betaPtr, cPtrArr,
        p.ldc, p.batchCount);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || safeBatch == 0 || cPtrArr == nullptr) {
        for (int b = 0; b < safeBatch; ++b) {
            EXPECT_EQ(
                std::memcmp(
                    data.cBatch[b].data(), data.cGolden[b].data(), data.cBatch[b].size() * sizeof(aclblasComplex)),
                0)
                << "no-op modified C";
        }
        return;
    }

    VerifyCgemmBatchedPrecision(p, data, ptrs, safeBatch, testHandle);
}

#include "cgemm_batched_pointer_regression.h"
