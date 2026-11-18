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
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "dtype_cast.h"
#include "hgemmBatched_param.h"
#include "hgemmBatched_golden.h"
#include "hgemmBatched_npu_wrapper.h"

namespace {

using blas_common::FloatToHalf;
using blas_common::HalfToFloat;

struct HgemmBatchedTestData {
    std::vector<std::vector<aclblasHalf>> aBatch;
    std::vector<std::vector<aclblasHalf>> bBatch;
    std::vector<std::vector<aclblasHalf>> cBatch; // device output copied back here
    std::vector<std::vector<float>> aHost;        // FLOAT32 view of A (for golden)
    std::vector<std::vector<float>> bHost;        // FLOAT32 view of B (for golden)
    std::vector<std::vector<aclblasHalf>> cOrig;  // original C (FLOAT16) for golden
};

struct HgemmBatchedPtrArrays {
    std::vector<const aclblasHalf*> aPtrs;
    std::vector<const aclblasHalf*> bPtrs;
    std::vector<aclblasHalf*> cPtrs;
};

inline std::vector<aclblasHalf> MakeHalfBuffer(int64_t count, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<float> f = makeBlasArray(count, fill, seed);
    std::vector<aclblasHalf> h(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        h[i] = FloatToHalf(f[i]);
    }
    return h;
}

inline std::vector<float> ToFloatBuffer(const std::vector<aclblasHalf>& h)
{
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); i++) {
        f[i] = HalfToFloat(h[i]);
    }
    return f;
}

inline HgemmBatchedTestData GenerateHgemmBatchedTestData(const HgemmBatchedParam& p)
{
    HgemmBatchedTestData data;
    int safeBatch = std::max(0, p.batchCount);
    int physColsA = hgemmBatchedPhysCols(p.m, p.k, p.transA);
    int physColsB = hgemmBatchedPhysCols(p.k, p.n, p.transB);

    data.aBatch.resize(safeBatch);
    data.bBatch.resize(safeBatch);
    data.cBatch.resize(safeBatch);
    data.aHost.resize(safeBatch);
    data.bHost.resize(safeBatch);
    data.cOrig.resize(safeBatch);

    for (int b = 0; b < safeBatch; b++) {
        uint32_t seed = p.randomSeed + static_cast<uint32_t>(b * 3);
        int64_t aCount = static_cast<int64_t>(std::max(1, p.lda)) * std::max(1, physColsA);
        int64_t bCount = static_cast<int64_t>(std::max(1, p.ldb)) * std::max(1, physColsB);
        int64_t cCount = static_cast<int64_t>(std::max(1, p.ldc)) * std::max(1, p.n);

        data.aBatch[b] = MakeHalfBuffer(aCount, p.aFill, seed);
        data.bBatch[b] = MakeHalfBuffer(bCount, p.bFill, seed + 1);
        data.cBatch[b] = MakeHalfBuffer(cCount, p.cFill, seed + 2);
        data.cOrig[b] = data.cBatch[b];
        data.aHost[b] = ToFloatBuffer(data.aBatch[b]);
        data.bHost[b] = ToFloatBuffer(data.bBatch[b]);
    }
    return data;
}

inline HgemmBatchedPtrArrays BuildHgemmBatchedPtrs(
    HgemmBatchedTestData& data, const HgemmBatchedParam& p, int safeBatch)
{
    HgemmBatchedPtrArrays ptrs;
    ptrs.aPtrs.resize(safeBatch, nullptr);
    ptrs.bPtrs.resize(safeBatch, nullptr);
    ptrs.cPtrs.resize(safeBatch, nullptr);
    for (int b = 0; b < safeBatch; b++) {
        if (!data.aBatch[b].empty() && !p.aarrayNull) {
            ptrs.aPtrs[b] = data.aBatch[b].data();
        }
        if (!data.bBatch[b].empty() && !p.barrayNull) {
            ptrs.bPtrs[b] = data.bBatch[b].data();
        }
        if (!data.cBatch[b].empty() && !p.carrayNull) {
            ptrs.cPtrs[b] = data.cBatch[b].data();
        }
    }
    return ptrs;
}

// FLOAT16 mixed-tolerance config per task §3.2: rtol = atol = 2^-10,
// matched_ratio >= 0.99, per-element max_abs_error <= max(1e-1, 32 ULP).
inline VerifyConfig MakeFp16MixedConfig()
{
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MIXED_TOLERANCE;
    cfg.mixedAtol = 0.0009765625;            // 2^-10
    cfg.mixedRtol = 0.0009765625;            // 2^-10
    cfg.mixedRequiredMatchedRatio = 0.99;
    cfg.mixedMaxAbsErrorLimit = 1e-1;        // 32 * ULP fallback handled per element
    cfg.mixedMantissaBits = 10;              // FLOAT16 mantissa bits (implicit + explicit = 11, ULP uses 10)
    cfg.mixedEmin = -14;                     // FLOAT16 min normal exponent
    return cfg;
}

// Compare one produced C_i against the FLOAT16 ideal.  Golden is computed in
// FLOAT32 (cblas_sgemm) then rounded to FLOAT16; both actual and golden are
// expanded to FLOAT32 for element-wise comparison.
inline bool VerifyBatch(
    const HgemmBatchedParam& p, HgemmBatchedTestData& data, int b, size_t cCount, int safeBatch)
{
    int origM = p.m;
    int origN = p.n;
    int origLdc = p.ldc;

    std::vector<float> cGolden = ToFloatBuffer(data.cOrig[b]);
    float alpha = p.alpha.isNull ? 0.0f : p.alpha.value;
    float beta = p.beta.isNull ? 0.0f : p.beta.value;
    const float* aPtr = (p.k > 0 && alpha != 0.0f && !data.aHost[b].empty()) ? data.aHost[b].data() : nullptr;
    const float* bPtr = (p.k > 0 && alpha != 0.0f && !data.bHost[b].empty()) ? data.bHost[b].data() : nullptr;

    HgemmBatchedCpuReference(
        p.transA, p.transB, origM, origN, p.k, alpha, aPtr, p.lda, bPtr, p.ldb, beta, cGolden.data(), origLdc);

    // Ideal FLOAT16 result, re-expanded to FLOAT32.
    for (size_t i = 0; i < cGolden.size(); i++) {
        cGolden[i] = HalfToFloat(FloatToHalf(cGolden[i]));
    }

    std::vector<float> cActual = ToFloatBuffer(data.cBatch[b]);

    std::string caseName = p.caseName;
    if (safeBatch > 1) {
        caseName += "_batch" + std::to_string(b);
    }

    VerifyConfig cfg = MakeFp16MixedConfig();
    return Verifier::verifyVector(cActual.data(), cGolden.data(), cCount, 1, cfg, caseName);
}

} // namespace

class HgemmBatchedTest : public BlasTest<HgemmBatchedParam> { };

TEST_F(HgemmBatchedTest, NullHandle)
{
    aclblasHalf one = FloatToHalf(1.0f);
    aclblasHalf zero = FloatToHalf(0.0f);
    aclblasStatus_t ret = aclblasHgemmBatched(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8,
        &one, nullptr, 8, nullptr, 8, &zero, nullptr, 8, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// CSV integrity gate: guards the delivery contract (1000 accuracy + 200
// performance) so an accidental truncation or column drop fails loudly instead
// of silently shrinking coverage. Mirrors the CgemmBatchedCsvTest.CsvCaseCount
// pattern used elsewhere in the repo.
TEST(HgemmBatchedCsvTest, CsvCaseCount)
{
    const auto cases = GetCasesFromCsv<HgemmBatchedParam>(ReplaceFileExtension2Csv(__FILE__));

    std::set<std::string> names;
    size_t accuracySuccess = 0;
    size_t performance = 0;
    size_t nonPerf = 0;
    for (const auto& c : cases) {
        EXPECT_FALSE(c.caseName.empty());
        EXPECT_TRUE(names.insert(c.caseName).second) << c.caseName;
        const bool isPerf = c.caseName.rfind("TC_PF_", 0) == 0;
        if (isPerf) {
            performance++;
        } else {
            nonPerf++;
            if (c.expectResult == ACLBLAS_STATUS_SUCCESS) {
                accuracySuccess++;
            }
        }
    }

    EXPECT_EQ(cases.size(), 1227U);
    EXPECT_EQ(accuracySuccess, 1002U);
    EXPECT_EQ(performance, 200U);
    EXPECT_EQ(nonPerf, 1027U);
}

INSTANTIATE_TEST_SUITE_P(
    HgemmBatched, HgemmBatchedTest,
    ::testing::ValuesIn(GetCasesFromCsv<HgemmBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<HgemmBatchedParam>);

TEST_P(HgemmBatchedTest, CsvDriven)
{
    const auto& p = GetParam();

    aclblasHandle_t testHandle = HgemmBatchedTest::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    int safeBatch = std::max(0, p.batchCount);
    auto data = GenerateHgemmBatchedTestData(p);
    auto ptrs = BuildHgemmBatchedPtrs(data, p, safeBatch);

    aclblasHalf alphaVal = FloatToHalf(p.alpha.value);
    aclblasHalf betaVal = FloatToHalf(p.beta.value);
    const aclblasHalf* alphaPtr = p.alphaNull ? nullptr : &alphaVal;
    const aclblasHalf* betaPtr = p.betaNull ? nullptr : &betaVal;

    const aclblasHalf* const* aArr = (p.aarrayNull || safeBatch == 0) ? nullptr : ptrs.aPtrs.data();
    const aclblasHalf* const* bArr = (p.barrayNull || safeBatch == 0) ? nullptr : ptrs.bPtrs.data();
    aclblasHalf* const* cArr = (p.carrayNull || safeBatch == 0) ? nullptr : ptrs.cPtrs.data();

    aclblasStatus_t ret = aclblasHgemmBatched_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k,
        alphaPtr, aArr, p.lda, bArr, p.ldb, betaPtr, cArr, p.ldc, p.batchCount);

    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "case=" << p.caseName << " desc=" << p.description;

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    // Legal no-ops: nothing to compare.
    if (p.m <= 0 || p.n <= 0 || safeBatch == 0 || cArr == nullptr) {
        return;
    }

    size_t cCount = static_cast<size_t>(std::max(1, p.ldc)) * std::max(1, p.n);
    for (int b = 0; b < safeBatch; b++) {
        if (ptrs.cPtrs[b] == nullptr) {
            continue;
        }
        EXPECT_TRUE(VerifyBatch(p, data, b, cCount, safeBatch)) << "case=" << p.caseName;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Performance: device-resident pointers, warmup 5 + N launches, mean of samples.
// Follows the in-repo batched GEMM performance convention (cgemm_batched).
// ═══════════════════════════════════════════════════════════════════════════════
namespace {

struct HgemmPerfBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    void* aPtrs = nullptr;
    void* bPtrs = nullptr;
    void* cPtrs = nullptr;

    void Free()
    {
        if (a) aclrtFree(a);
        if (b) aclrtFree(b);
        if (c) aclrtFree(c);
        if (aPtrs) aclrtFree(aPtrs);
        if (bPtrs) aclrtFree(bPtrs);
        if (cPtrs) aclrtFree(cPtrs);
    }
};

inline bool AllocHgemmPerfBuffers(
    HgemmPerfBuffers& buf, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, int lda, int ldb, int ldc, int batchCount)
{
    auto sizes = ComputeHgemmBatchedBufferSizes(transA, transB, m, n, k, lda, ldb, ldc);
    const size_t allA = sizes.aBytes * static_cast<size_t>(batchCount);
    const size_t allB = sizes.bBytes * static_cast<size_t>(batchCount);
    const size_t allC = sizes.cBytes * static_cast<size_t>(batchCount);
    if (aclrtMalloc(&buf.a, allA, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;
    if (aclrtMalloc(&buf.b, allB, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;
    if (aclrtMalloc(&buf.c, allC, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;
    if (aclrtMemset(buf.a, allA, 0, allA) != ACL_SUCCESS) return false;
    if (aclrtMemset(buf.b, allB, 0, allB) != ACL_SUCCESS) return false;
    if (aclrtMemset(buf.c, allC, 0, allC) != ACL_SUCCESS) return false;

    std::vector<const aclblasHalf*> aPtrs(batchCount);
    std::vector<const aclblasHalf*> bPtrs(batchCount);
    std::vector<aclblasHalf*> cPtrs(batchCount);
    for (int batch = 0; batch < batchCount; batch++) {
        aPtrs[batch] = reinterpret_cast<const aclblasHalf*>(static_cast<uint8_t*>(buf.a) + sizes.aBytes * batch);
        bPtrs[batch] = reinterpret_cast<const aclblasHalf*>(static_cast<uint8_t*>(buf.b) + sizes.bBytes * batch);
        cPtrs[batch] = reinterpret_cast<aclblasHalf*>(static_cast<uint8_t*>(buf.c) + sizes.cBytes * batch);
    }
    const size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    if (aclrtMalloc(&buf.aPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;
    if (aclrtMalloc(&buf.bPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;
    if (aclrtMalloc(&buf.cPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return false;
    if (aclrtMemcpy(buf.aPtrs, ptrBytes, aPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        return false;
    if (aclrtMemcpy(buf.bPtrs, ptrBytes, bPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        return false;
    if (aclrtMemcpy(buf.cPtrs, ptrBytes, cPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        return false;
    return true;
}

inline double RunHgemmBatchedPerformanceCase(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, int lda, int ldb, int ldc, int batchCount)
{
    constexpr int warmups = 5;
    constexpr int samples = 60;
    HgemmPerfBuffers buf;
    if (!AllocHgemmPerfBuffers(buf, transA, transB, m, n, k, lda, ldb, ldc, batchCount)) {
        buf.Free();
        return -1.0;
    }
    const auto* aPtrs = reinterpret_cast<const aclblasHalf* const*>(buf.aPtrs);
    const auto* bPtrs = reinterpret_cast<const aclblasHalf* const*>(buf.bPtrs);
    auto* cPtrs = reinterpret_cast<aclblasHalf* const*>(buf.cPtrs);
    aclblasHalf alpha = FloatToHalf(1.0f);
    aclblasHalf beta = FloatToHalf(0.0f);
    auto launch = [&]() {
        return aclblasHgemmBatched(
            handle, transA, transB, m, n, k, &alpha, aPtrs, lda, bPtrs, ldb, &beta, cPtrs, ldc, batchCount);
    };

    for (int i = 0; i < warmups; i++) {
        EXPECT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
    }
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < samples; i++) {
        EXPECT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
    }
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    const auto end = std::chrono::steady_clock::now();
    buf.Free();
    return std::chrono::duration<double, std::micro>(end - start).count() / samples;
}

} // namespace

// Task §3.3 typical performance cases (aligned with gpu_baseline.csv base rows).
TEST_F(HgemmBatchedTest, PerformanceContract)
{
    constexpr double performanceRatio = 0.4;
    struct PerfCase {
        const char* name;
        aclblasOperation_t ta;
        aclblasOperation_t tb;
        int m;
        int n;
        int k;
        int batchCount;
        double gpuUs;
    };
    const PerfCase cases[] = {
        {"256x256x256_b16_NN", ACLBLAS_OP_N, ACLBLAS_OP_N, 256, 256, 256, 16, 7.249},
        {"512x512x512_b8_NN", ACLBLAS_OP_N, ACLBLAS_OP_N, 512, 512, 512, 8, 8.647},
        {"1024x1024x1024_b4_TN", ACLBLAS_OP_T, ACLBLAS_OP_N, 1024, 1024, 1024, 4, 26.125},
        {"2048x2048x2048_b2_NT", ACLBLAS_OP_N, ACLBLAS_OP_T, 2048, 2048, 2048, 2, 55.413},
    };

    for (const auto& perf : cases) {
        const double averageUs = RunHgemmBatchedPerformanceCase(
            handle_, stream_, perf.ta, perf.tb, perf.m, perf.n, perf.k, std::max(1, perf.m), std::max(1, perf.k),
            std::max(1, perf.m), perf.batchCount);
        ASSERT_GE(averageUs, 0.0) << perf.name;
        const double limitUs = perf.gpuUs / performanceRatio;
        std::cout << std::fixed << std::setprecision(3) << "[HGEMM_BATCHED_PERF] case=" << perf.name
                  << " device=Ascend-950PR backend=npu warmups=5 samples=60"
                  << " average_us=" << averageUs << " gpu_us=" << perf.gpuUs
                  << " ratio=" << (perf.gpuUs / averageUs) << " limit_us=" << limitUs << std::endl;
        EXPECT_LE(averageUs, limitUs) << perf.name;
    }
}

// CSV-driven performance cases (TC_PF_*).  Opt-in only: set
// HGEMM_BATCHED_RUN_ALL_PERF=1 to execute; otherwise those cases are skipped
// during accuracy runs.
TEST_P(HgemmBatchedTest, CsvPerformance)
{
    const auto& p = GetParam();
    const char* enabled = std::getenv("HGEMM_BATCHED_RUN_ALL_PERF");
    if (enabled == nullptr || std::string(enabled) != "1") {
        GTEST_SKIP() << "set HGEMM_BATCHED_RUN_ALL_PERF=1 to run CSV performance cases";
    }
    if (p.caseName.rfind("TC_PF_", 0) != 0) {
        GTEST_SKIP() << "not a TC_PF case";
    }

    const double averageUs = RunHgemmBatchedPerformanceCase(
        handle_, stream_, p.transA, p.transB, p.m, p.n, p.k, std::max(1, p.lda), std::max(1, p.ldb),
        std::max(1, p.ldc), p.batchCount);
    ASSERT_GE(averageUs, 0.0) << p.caseName;
    std::cout << std::fixed << std::setprecision(3) << "[HGEMM_BATCHED_CSV_PERF] case=" << p.caseName
              << " device=Ascend-950PR backend=npu warmups=5 samples=60"
              << " transA=" << static_cast<int>(p.transA) << " transB=" << static_cast<int>(p.transB)
              << " m=" << p.m << " n=" << p.n << " k=" << p.k << " batchCount=" << p.batchCount
              << " average_us=" << averageUs << std::endl;
}
