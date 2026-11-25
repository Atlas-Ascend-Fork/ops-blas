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
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "npu_event_timer.h"
#include "hgemm_param.h"
#include "hgemm_golden.h"
#include "hgemm_npu_wrapper.h"

class HgemmTest : public BlasTest<HgemmParam> {};

// Build a FLOAT16 (raw bit) column-major matrix from a fill descriptor.
// Generation is streamed in chunks so peak host memory stays at
// ~fp16-buffer size + one small chunk (the 8192^2 perf cases must respect
// the 512MB host budget). The value sequence matches fill.h::makeBlasMatrix
// (same seed, same element order).
static std::vector<aclblasHalf> MakeHalfMatrix(int rows, int cols, int ld, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || rows <= 0 || cols <= 0 || ld <= 0) {
        return {};
    }
    const size_t total = static_cast<size_t>(ld) * cols;
    std::vector<aclblasHalf> out(total);
    if (fill.method == BlasFillMode::M_VALUE) {
        const uint16_t bits = blas_common::FloatToHalf(fill.val1);
        std::fill(out.begin(), out.end(), bits);
        return out;
    }
    // Structured patterns fall back to the reference generator (not used by
    // the hgemm CSV, kept for robustness).
    static const std::map<BlasFillMode::Pattern, bool> structured = {
        {BlasFillMode::P_UPPER, true},
        {BlasFillMode::P_LOWER, true},
        {BlasFillMode::P_DIAG, true},
        {BlasFillMode::P_ILLCOND, true}};
    if (structured.count(fill.pattern)) {
        auto floats = makeBlasMatrix(rows, cols, ld, fill, seed);
        for (size_t i = 0; i < total && i < floats.size(); i++) {
            out[i] = blas_common::FloatToHalf(floats[i]);
        }
        return out;
    }
    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);
    constexpr size_t kChunk = 65536;
    std::vector<float> buf(kChunk);
    for (size_t base = 0; base < total; base += kChunk) {
        size_t cnt = std::min(kChunk, total - base);
        for (size_t j = 0; j < cnt; j++) {
            buf[j] = gen->at(base + j);
        }
        for (size_t j = 0; j < cnt; j++) {
            out[base + j] = blas_common::FloatToHalf(buf[j]);
        }
    }
    return out;
}

static std::vector<float> HalfBitsToFloat(const std::vector<aclblasHalf>& bits)
{
    std::vector<float> out(bits.size());
    for (size_t i = 0; i < bits.size(); i++) {
        out[i] = blas_common::HalfToFloat(bits[i]);
    }
    return out;
}

// Golden path for degenerate shapes (k == 0 or alpha == 0): C = beta * C
// without referencing A/B (mirrors cuBLAS/BLAS semantics where alpha == 0
// never reads the operands). Only the m*n logical region is defined output;
// the ldc padding rows keep their original values.
static void GoldenScaleC(
    std::vector<float>& cGolden, const std::vector<float>& cOrig, float beta, int m, int n, int ldc)
{
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            size_t idx = static_cast<size_t>(row) + static_cast<size_t>(col) * ldc;
            if (beta == 0.0f) {
                cGolden[idx] = 0.0f;
            } else {
                cGolden[idx] = beta * cOrig[idx];
            }
        }
    }
}

// k == 0 collapses an operand's physical extents to a zero dimension, which
// would degenerate MakeHalfMatrix to an empty buffer (data() == nullptr) and
// make the "k == 0 does not reference A/B" cases indistinguishable from the
// explicit nullptr cases. Feed those cases a small legal matrix instead so
// the operator actually receives non-null device pointers.
struct HgemmOperandShape {
    int rows;
    int cols;
    int ld;
};

static HgemmOperandShape OperandShapeFor(int physRows, int physCols, int ld, int k)
{
    constexpr int kDegenerateOperandDim = 8;
    HgemmOperandShape shape{physRows, physCols, ld};
    if (k == 0) {
        if (shape.rows <= 0) {
            shape.rows = kDegenerateOperandDim;
            shape.ld = kDegenerateOperandDim;
        }
        if (shape.cols <= 0) {
            shape.cols = kDegenerateOperandDim;
        }
    }
    return shape;
}

// Golden for the beta == 0 degenerate branch: per BLAS semantics C is not a
// valid input (it may hold NaN/Inf garbage), so the golden must not evaluate
// the beta*C term through cblas (0 * NaN = NaN would poison it). Zero the
// m*n logical region (ldc padding keeps the original fill), then compute
// alpha*op(A)*op(B) on the zeroed buffer while the product term is live.
static void BuildBetaZeroGolden(
    const HgemmParam& p, aclblasHandle_t testHandle, const std::vector<aclblasHalf>& aHalf,
    const std::vector<aclblasHalf>& bHalf, float alphaF, std::vector<float>& cGolden)
{
    for (int col = 0; col < p.n; col++) {
        for (int row = 0; row < p.m; row++) {
            cGolden[static_cast<size_t>(row) + static_cast<size_t>(col) * p.ldc] = 0.0f;
        }
    }
    if (p.k > 0 && alphaF != 0.0f) {
        std::vector<float> aF = HalfBitsToFloat(aHalf);
        std::vector<float> bF = HalfBitsToFloat(bHalf);
        ASSERT_EQ(
            aclblasHgemm_cpu(
                testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaF, aF.data(), p.lda, bF.data(), p.ldb, 0.0f,
                cGolden.data(), p.ldc),
            ACLBLAS_STATUS_SUCCESS);
    }
}

// Golden for the regular path and the k == 0 / alpha == 0 scaling branch.
// The original fill is rebuilt from scratch (the launch already overwrote the
// in-place buffer).
static void BuildReferenceGolden(
    const HgemmParam& p, aclblasHandle_t testHandle, const std::vector<aclblasHalf>& aHalf,
    const std::vector<aclblasHalf>& bHalf, float alphaF, float betaF, std::vector<float>& cGolden)
{
    auto cOrigHalf = MakeHalfMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);
    cGolden = HalfBitsToFloat(cOrigHalf);
    if (p.k == 0 || alphaF == 0.0f) {
        // C = beta * C (A/B not referenced).
        GoldenScaleC(cGolden, cGolden, betaF, p.m, p.n, p.ldc);
        return;
    }
    std::vector<float> aF = HalfBitsToFloat(aHalf);
    std::vector<float> bF = HalfBitsToFloat(bHalf);
    ASSERT_EQ(
        aclblasHgemm_cpu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaF, aF.data(), p.lda, bF.data(), p.ldb, betaF,
            cGolden.data(), p.ldc),
        ACLBLAS_STATUS_SUCCESS);
}

static void VerifyHgemmOutput(
    const HgemmParam& p, const std::vector<float>& cNpuF, std::vector<float>& cGolden, size_t cCount, float alphaF)
{
    HgemmVerifyResult res = VerifyHgemmElements(cNpuF.data(), cGolden.data(), cCount);
    std::cout << "[" << p.caseName << "] matchedRatio=" << res.matchedRatio << ", maxAbsErr=" << res.maxAbsError
              << ", nonFinite=" << res.nonFiniteCount << " (equiv " << res.nonFinitePassCount << ")"
              << ", finiteFail=" << res.finiteFailCount << std::endl;
    EXPECT_TRUE(res.pass) << p.caseName << " mixed-tolerance verification failed";

    // Task spec sec. 3.2 is the binding acceptance criterion (mixed tolerance
    // with the FLOAT16 tier + sec. 3.2.5 special-value rules) - covered by
    // VerifyHgemmElements above. For finite-only cases also run the shared
    // framework verifier with the refreshed FLOAT16 defaults; degenerate
    // alpha/k cases are bit-exact. MERE/MARE CSV columns stay informational.
    bool anyNonFinite = false;
    for (size_t i = 0; i < cCount; i++) {
        if (!std::isfinite(cNpuF[i]) || !std::isfinite(cGolden[i])) {
            anyNonFinite = true;
            break;
        }
    }
    VerifyConfig cfg;
    if (p.k == 0 || alphaF == 0.0f) {
        cfg.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(cNpuF.data(), cGolden.data(), cCount, 1, cfg, p.caseName));
    } else if (!anyNonFinite) {
        applyMixedTolerance(cfg, ACL_FLOAT16, cGolden.data(), cCount);
        EXPECT_TRUE(Verifier::verifyVector(cNpuF.data(), cGolden.data(), cCount, 1, cfg, p.caseName));
    }
}

static void RunHgemmCase(const HgemmParam& p, aclblasHandle_t testHandle)
{
    int physRowsA = (p.transA == ACLBLAS_OP_N) ? p.m : p.k;
    int physColsA = (p.transA == ACLBLAS_OP_N) ? p.k : p.m;
    int physRowsB = (p.transB == ACLBLAS_OP_N) ? p.k : p.n;
    int physColsB = (p.transB == ACLBLAS_OP_N) ? p.n : p.k;
    // Negative-injection rows (k < 0) keep the empty buffers: the operator
    // validates k before touching any operand memory, and OperandShapeFor only
    // substitutes legal extents for the exact k == 0 degenerate cases.
    HgemmOperandShape shapeA = OperandShapeFor(physRowsA, physColsA, p.lda, p.k);
    HgemmOperandShape shapeB = OperandShapeFor(physRowsB, physColsB, p.ldb, p.k);

    auto aHalf = MakeHalfMatrix(shapeA.rows, shapeA.cols, shapeA.ld, p.aFill, p.randomSeed);
    auto bHalf = MakeHalfMatrix(shapeB.rows, shapeB.cols, shapeB.ld, p.bFill, p.randomSeed + 1);
    auto cHalf = MakeHalfMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);

    aclblasHalf alphaH = blas_common::FloatToHalf(p.alphaVal);
    aclblasHalf betaH = blas_common::FloatToHalf(p.betaVal);
    const aclblasHalf* alphaPtr = p.alphaNull ? nullptr : &alphaH;
    const aclblasHalf* betaPtr = p.betaNull ? nullptr : &betaH;
    const aclblasHalf* aPtr = p.aNull ? nullptr : aHalf.data();
    const aclblasHalf* bPtr = p.bNull ? nullptr : bHalf.data();
    aclblasHalf* cPtr = p.cNull ? nullptr : cHalf.data();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasHgemm_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasHgemm_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || cPtr == nullptr)
        return;

    size_t cCount = static_cast<size_t>(p.ldc) * p.n;
    float alphaF = blas_common::HalfToFloat(alphaH);
    float betaF = blas_common::HalfToFloat(betaH);

    std::vector<float> cGolden;
    if (betaF == 0.0f) {
        auto cOrigHalf = MakeHalfMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);
        cGolden = HalfBitsToFloat(cOrigHalf);
        BuildBetaZeroGolden(p, testHandle, aHalf, bHalf, alphaF, cGolden);
    } else {
        BuildReferenceGolden(p, testHandle, aHalf, bHalf, alphaF, betaF, cGolden);
    }
    // Round the FLOAT32 golden to FLOAT16 (round-to-nearest-even), then
    // compare element-wise against the NPU FLOAT16 output (upcast to FLOAT32).
    for (size_t i = 0; i < cGolden.size(); i++) {
        cGolden[i] = blas_common::HalfToFloat(blas_common::FloatToHalf(cGolden[i]));
    }
    std::vector<float> cNpuF = HalfBitsToFloat(cHalf);

    // Task spec sec. 3.2 mixed-tolerance + sec. 3.2.5 special-value verification over
    // the whole ldc*n buffer (padding rows compare original-vs-original, which
    // also asserts the operator never writes beyond the m*n logical matrix).
    VerifyHgemmOutput(p, cNpuF, cGolden, cCount, alphaF);
}

// Performance case: stage device buffers once, warm up, then measure the
// kernel with NPU events (task spec sec. 3.3: warmup 5 + >10 valid samples).
static void MeasureHgemmCase(const HgemmParam& p, aclblasHandle_t testHandle, aclrtStream stream)
{
    int physRowsA = (p.transA == ACLBLAS_OP_N) ? p.m : p.k;
    int physColsA = (p.transA == ACLBLAS_OP_N) ? p.k : p.m;
    int physRowsB = (p.transB == ACLBLAS_OP_N) ? p.k : p.n;
    int physColsB = (p.transB == ACLBLAS_OP_N) ? p.n : p.k;

    auto aHalf = MakeHalfMatrix(physRowsA, physColsA, p.lda, p.aFill, p.randomSeed);
    auto bHalf = MakeHalfMatrix(physRowsB, physColsB, p.ldb, p.bFill, p.randomSeed + 1);
    auto cHalf = MakeHalfMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);

    size_t aBytes = aHalf.size() * sizeof(aclblasHalf);
    size_t bBytes = bHalf.size() * sizeof(aclblasHalf);
    size_t cBytes = cHalf.size() * sizeof(aclblasHalf);

    HgemmDeviceBuf dev;
    ASSERT_EQ(
        HgemmAllocDevice(dev, p.k, aHalf.data(), bHalf.data(), cHalf.data(), aBytes, bBytes, cBytes),
        ACLBLAS_STATUS_SUCCESS);
    void* dInitialC = nullptr;
    ASSERT_EQ(HgemmAllocAndCopy(&dInitialC, cHalf.data(), cBytes), ACLBLAS_STATUS_SUCCESS);

    aclblasHalf alphaH = blas_common::FloatToHalf(p.alphaVal);
    aclblasHalf betaH = blas_common::FloatToHalf(p.betaVal);
    auto resetC = [&]() {
        return aclrtMemcpy(dev.dC, cBytes, dInitialC, cBytes, ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_SUCCESS;
    };
    auto launch = [&]() {
        aclblasStatus_t ret = aclblasHgemm(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, &alphaH, static_cast<const aclblasHalf*>(dev.dA), p.lda,
            static_cast<const aclblasHalf*>(dev.dB), p.ldb, &betaH, static_cast<aclblasHalf*>(dev.dC), p.ldc);
        return ret == ACLBLAS_STATUS_SUCCESS;
    };

    std::vector<double> timings;
    ASSERT_EQ(blas_test::MeasureNpuEvent(stream, p.warmup, p.measurements, resetC, launch, timings), 0);

    HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
    aclrtFree(dInitialC);

    std::cout << std::fixed << std::setprecision(3) << "PERFORMANCE_SUMMARY_JSON={\"case\":\"" << p.caseName
              << "\",\"m\":" << p.m << ",\"n\":" << p.n << ",\"k\":" << p.k
              << ",\"avg_us\":" << blas_test::MeanUs(timings) << ",\"p50_us\":" << blas_test::MedianUs(timings)
              << ",\"warmups\":" << p.warmup << ",\"measurements\":" << p.measurements
              << ",\"timing\":\"npu_event\"}\n";
}

INSTANTIATE_TEST_SUITE_P(
    Hgemm, HgemmTest, ::testing::ValuesIn(GetCasesFromCsv<HgemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<HgemmParam>);

TEST_P(HgemmTest, CsvDriven)
{
    const auto& p = GetParam();

    aclblasHandle_t testHandle = HgemmTest::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    if (p.isPerf) {
        MeasureHgemmCase(p, testHandle, HgemmTest::stream_);
        return;
    }
    RunHgemmCase(p, testHandle);
}

// CSV composition guard: the delivery contract is 1000 accuracy cases
// (980 golden-verified successes + 20 negative validations) plus 200 TC_PF
// performance cases with unique non-empty names. A truncated or malformed
// CSV must fail loudly here instead of silently shrinking the suite.
TEST_F(HgemmTest, CsvCompositionAssert)
{
    auto csvPath = ReplaceFileExtension2Csv(__FILE__);
    std::ifstream ifs(csvPath);
    ASSERT_TRUE(ifs.is_open()) << "cannot open " << csvPath;
    std::string header;
    std::getline(ifs, header);
    std::vector<std::string> names;
    size_t perfCount = 0;
    size_t successCount = 0;
    size_t negativeCount = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto fields = CsvSplitLine(line);
        ASSERT_GE(fields.size(), 16u) << "malformed CSV row: " << line;
        const std::string& name = fields[0];
        ASSERT_FALSE(name.empty()) << "CSV row without case_name";
        names.push_back(name);
        if (name.rfind("TC_PF_", 0) == 0) {
            perfCount++;
        } else if (fields[15].find("SUCCESS") != std::string::npos) {
            successCount++;
        } else {
            negativeCount++;
        }
    }
    ASSERT_EQ(names.size(), 1200u) << "total CSV rows drifted from 1200";
    ASSERT_EQ(successCount, 980u) << "golden-verified accuracy cases drifted from 980";
    ASSERT_EQ(negativeCount, 20u) << "negative validation cases drifted from 20";
    ASSERT_EQ(perfCount, 200u) << "TC_PF performance cases drifted from 200";
    std::sort(names.begin(), names.end());
    for (size_t i = 1; i < names.size(); i++) {
        ASSERT_NE(names[i], names[i - 1]) << "duplicate case_name: " << names[i];
    }
}

// Stream-ordering regression: a GEMM writing C is queued first, then an
// alpha=0/beta=0 Hgemm zeroes C on the same handle stream. After
// synchronization C must be all zeros - the zeroing must not be overtaken by
// the still in-flight GEMM (guards the async memset ordering fix).
TEST_F(HgemmTest, ZeroingOrderedAfterInFlightGemm)
{
    constexpr int m = 64;
    constexpr int n = 48;
    constexpr int k = 32;
    aclblasHandle_t handle = HgemmTest::handle_;
    aclrtStream stream = HgemmTest::stream_;

    auto aHalf = MakeHalfMatrix(m, k, m, parseFill("RANDOM_NORM_5_5"), 606);
    auto bHalf = MakeHalfMatrix(k, n, k, parseFill("RANDOM_NORM_5_5"), 607);
    auto cHalf = MakeHalfMatrix(m, n, m, parseFill("RANDOM_NORM_5_5"), 608);
    aclblasHalf alphaOne = blas_common::FloatToHalf(1.0f);
    aclblasHalf betaZero = blas_common::FloatToHalf(0.0f);

    HgemmDeviceBuf dev;
    size_t aBytes = aHalf.size() * sizeof(aclblasHalf);
    size_t bBytes = bHalf.size() * sizeof(aclblasHalf);
    size_t cBytes = cHalf.size() * sizeof(aclblasHalf);
    ASSERT_EQ(
        HgemmAllocDevice(dev, k, aHalf.data(), bHalf.data(), cHalf.data(), aBytes, bBytes, cBytes),
        ACLBLAS_STATUS_SUCCESS);

    // 1) Regular GEMM queued first (writes non-zero values into C).
    ASSERT_EQ(
        aclblasHgemm(
            handle, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, &alphaOne, static_cast<const aclblasHalf*>(dev.dA), m,
            static_cast<const aclblasHalf*>(dev.dB), k, &betaZero, static_cast<aclblasHalf*>(dev.dC), m),
        ACLBLAS_STATUS_SUCCESS);

    // 2) alpha=0/beta=0 Hgemm zeroes C on the same stream, no host sync in
    //    between: only correct stream ordering guarantees the zero result.
    ASSERT_EQ(
        aclblasHgemm(
            handle, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, &betaZero, static_cast<const aclblasHalf*>(dev.dA), m,
            static_cast<const aclblasHalf*>(dev.dB), k, &betaZero, static_cast<aclblasHalf*>(dev.dC), m),
        ACLBLAS_STATUS_SUCCESS);

    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    std::vector<aclblasHalf> result(cHalf.size());
    ASSERT_EQ(aclrtMemcpy(result.data(), cBytes, dev.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_EQ(result[i], static_cast<aclblasHalf>(0)) << "C[" << i << "] not zeroed: in-flight GEMM overwrote it";
    }
    HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
}
