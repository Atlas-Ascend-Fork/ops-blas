/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file stpsv_test.cpp
 * \brief CSV driven ST for aclblasStpsv on arch22 (Atlas A2/A3).
 *
 * The golden is produced by the reference Netlib implementation (cblas_stpsv) and the
 * result is judged with the FLOAT32 mixed tolerance of the ops-blas precision standard.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "stpsv_param.h"
#include "stpsv_golden.h"
#include "stpsv_npu_wrapper.h"

class TpsvTest : public BlasTest<TpsvParam> {};

namespace {

// Case set = the frozen case table shipped with the task book, plus an optional supplementary
// table. The primary file is never edited so it stays byte-comparable with the reference copy;
// supplementary edge cases (AP carrying Inf / NaN) live in <name>_supplement.csv next to it.
std::vector<TpsvParam> GetStpsvCases()
{
    const std::string primary = ReplaceFileExtension2Csv(__FILE__);
    std::vector<TpsvParam> cases = GetCasesFromCsv<TpsvParam>(primary);

    const size_t dot = primary.rfind('.');
    const std::string extra = ((dot == std::string::npos) ? primary : primary.substr(0, dot)) + "_supplement.csv";
    std::ifstream probe(extra);
    if (probe.good()) {
        probe.close();
        const std::vector<TpsvParam> more = GetCasesFromCsv<TpsvParam>(extra);
        cases.insert(cases.end(), more.begin(), more.end());
    }
    return cases;
}

} // namespace

INSTANTIATE_TEST_SUITE_P(Tpsv, TpsvTest, ::testing::ValuesIn(GetStpsvCases()), PrintCaseInfoString<TpsvParam>);

TEST_F(TpsvTest, NullHandle)
{
    // handle is validated before AP / x, so a null handle wins over null operands.
    aclblasStatus_t ret =
        aclblasStpsv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, nullptr, nullptr, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(TpsvTest, NullHandleBeforeQuickReturn)
{
    // The handle must be validated before the n == 0 quick-return shortcut, so a null handle with
    // n == 0 still reports HANDLE_IS_NULLPTR (matches arch35 and the reference behaviour). This is
    // the case neither NullHandle (n=4) nor TC_ED_245/246 (legal params) covers.
    aclblasStatus_t ret =
        aclblasStpsv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 0, nullptr, nullptr, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(TpsvTest, XSpanReachesUint32Max)
{
    // White-box coverage for the equality boundary of the x-index span guard. The device computes
    // the GM view length as (n-1)*|incx| + 1 in uint32, so a span that lands exactly on UINT32_MAX
    // (4294967295) would wrap xLen to 0 in the kernel. n=4 with incx=1431655765 gives
    // (4-1)*1431655765 = 4294967295 = UINT32_MAX, which is reachable and must be refused up front.
    // The guard rejects at span >= UINT32_MAX (not >), so this exact boundary is refused.
    float scratch = 0.0f;
    const aclblasStatus_t ret =
        aclblasStpsv(handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, &scratch, &scratch, 1431655765);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_NOT_SUPPORTED));
}

TEST_F(TpsvTest, OrderAboveSupportedMaximum)
{
    // The solve keeps the whole solution vector resident in UB, so the supported order is
    // bounded. Anything above that bound has to be refused up front instead of launching a
    // kernel whose UB allocation would overrun. The status is checked before the operands
    // are dereferenced, so a scratch buffer is enough here.
    float scratch = 0.0f;
    const aclblasStatus_t ret =
        aclblasStpsv(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 32769, &scratch, &scratch, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_NOT_SUPPORTED));
}

// issue #819: 框架级用例——dtype_utils.h 的 E4M3FN 编码/解码往返。
// 挂载于 stpsv arch22（本二进制经 fill.h 消费 dtype_utils.h；FP8 用例本体在
// gemm_ex arch35，910B3 无该测试目标）。0x08~0x7E 全部有限位型必须：
// 解码为有限值（无虚假 Inf/NaN）、回编回到同一位型；0x7F 保持 NaN 语义。
TEST_F(TpsvTest, Fp8E4m3fnRoundTripFinitePatterns)
{
    for (uint32_t bits = 0x08; bits <= 0x7E; ++bits) {
        const auto pattern = static_cast<uint8_t>(bits);
        const float decoded = fp8E4m3ToFloat(pattern);
        EXPECT_TRUE(std::isfinite(decoded)) << "pattern 0x" << std::hex << bits << " decoded non-finite";
        EXPECT_EQ(floatToFp8E4m3(decoded), pattern) << "pattern 0x" << std::hex << bits << " round-trip mismatch";
    }
    // 上限 448 可表示（0x7E），不再被编码为 NaN
    EXPECT_EQ(fp8E4m3ToFloat(0x7E), 448.0f);
    EXPECT_EQ(floatToFp8E4m3(448.0f), 0x7E);
    EXPECT_EQ(floatToFp8E4m3(-448.0f), 0xFE);
    // 248 的 RNE 进位路径：舍入到 256（0x78），此前被错误编码为 0x7F=NaN
    EXPECT_EQ(floatToFp8E4m3(248.0f), 0x78);
    EXPECT_EQ(fp8E4m3ToFloat(0x78), 256.0f);
    // 0x7F 为 E4M3FN 唯一正 NaN 位型
    EXPECT_TRUE(std::isnan(fp8E4m3ToFloat(0x7F)));
}

namespace {

// Largest magnitude the fill generator can emit, used to size the diagonal boost.
float FillAmplitude(const BlasFillMode& fill)
{
    float amp = std::max(std::fabs(fill.val1), std::fabs(fill.val2));
    if (!(amp > 0.0f) || std::isinf(amp) || std::isnan(amp)) {
        amp = 1.0f;
    }
    return amp;
}

// Packed AP of length n*(n+1)/2 using the CSV fill token, then conditioned so that the
// single-precision comparison against the cblas golden stays meaningful:
//   NON_UNIT -> strict diagonal dominance,  UNIT -> the whole triangle is scaled down so
//   that the implicit unit diagonal dominates.
std::vector<float> BuildPacked(int n, bool isUpper, const BlasFillMode& fill, uint32_t seed, bool unitDiag)
{
    const size_t apLen = static_cast<size_t>(n) * (n + 1) / 2;
    std::vector<float> ap = makeBlasArray(static_cast<int64_t>(apLen), fill, seed);
    if (ap.size() < apLen) {
        ap.assign(apLen, 0.0f);
    }

    const float amp = FillAmplitude(fill);
    if (unitDiag) {
        const float scale = std::min(1.0f, 1.0f / (amp * static_cast<float>(n)));
        for (size_t i = 0; i < ap.size(); ++i) {
            ap[i] *= scale;
        }
    } else {
        // |off-diagonal| <= amp, so a diagonal of at least amp*n dominates the row.
        const float boost = (amp * std::max(2.0f, static_cast<float>(n))) + 1.0f;
        for (int i = 0; i < n; ++i) {
            const size_t idx = isUpper ? TpsvPackedUpperIdxCpu(i, i) : TpsvPackedLowerIdxCpu(i, i, n);
            const float d = ap[idx];
            const float mag = std::fabs(d) + boost;
            ap[idx] = (d < 0.0f) ? -mag : mag;
        }
    }
    return ap;
}

std::string FormatFloat(float v)
{
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

} // namespace

TEST_P(TpsvTest, CsvDriven)
{
    const auto& p = GetParam();
    const bool isUpper = (p.uplo == ACLBLAS_UPPER);
    const int absIncx = std::abs(p.incx);

    const BlasFillMode apFill(p.apFill);
    const BlasFillMode xFill(p.xFill);
    const bool apNull = (apFill.method == BlasFillMode::M_NULLPTR);
    const bool xNull = (xFill.method == BlasFillMode::M_NULLPTR);

    std::vector<float> ap;
    std::vector<float> xHost;
    if ((p.n > 0) && (p.incx != 0)) {
        if (!apNull) {
            ap = BuildPacked(p.n, isUpper, apFill, p.randomSeed, p.diag == ACLBLAS_UNIT);
        }
        if (!xNull) {
            xHost = makeBlasStrided(p.n, p.incx, xFill, p.randomSeed + 1);
        }
    }

    const float* apPtr = ap.empty() ? nullptr : ap.data();
    float* xPtr = xHost.empty() ? nullptr : xHost.data();

    // The golden has to be produced from the untouched right-hand side: x is solved in place.
    std::vector<float> golden = xHost;

    const aclblasStatus_t ret = aclblasStpsv_npu(TpsvTest::handle_, p.uplo, p.trans, p.diag, p.n, apPtr, xPtr, p.incx);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
            << p.caseName << " expected status " << static_cast<int>(p.expectResult) << " but got "
            << static_cast<int>(ret);
        return;
    }
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS))
        << p.caseName << " unexpected status " << static_cast<int>(ret);

    // n == 0 is a legal quick return: AP and x are not referenced.
    if (p.n <= 0) {
        return;
    }

    aclblasStpsv_cpu(TpsvTest::handle_, p.uplo, p.trans, p.diag, p.n, ap.data(), golden.data(), p.incx);

    const float* outPtr = (p.incx < 0) ? (xPtr + (p.n - 1) * absIncx) : xPtr;
    const float* goldPtr = (p.incx < 0) ? (golden.data() + (p.n - 1) * absIncx) : golden.data();
    const int64_t stride = (p.incx < 0) ? -absIncx : absIncx;

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(p.n));
    const bool ok = Verifier::verifyVector(outPtr, goldPtr, static_cast<size_t>(p.n), stride, cfg, p.caseName);
    if (!ok) {
        ADD_FAILURE() << p.caseName << " x[0]: actual=" << FormatFloat(outPtr[0])
                      << " golden=" << FormatFloat(goldPtr[0]);
    }
    EXPECT_TRUE(ok);
}

namespace {

// Task-book section 3.3 reference budgets (us). Acceptance is a ratio against these.
struct PerfCase {
    int n;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    const char* name;
    double budgetUs;
};

const PerfCase kPerfCases[] = {
    {512, ACLBLAS_LOWER, ACLBLAS_OP_N, "n512_LOWER_N_NON_UNIT", 432.71},
    {1024, ACLBLAS_UPPER, ACLBLAS_OP_N, "n1024_UPPER_N_NON_UNIT", 941.23},
    {2048, ACLBLAS_LOWER, ACLBLAS_OP_T, "n2048_LOWER_T_NON_UNIT", 1690.94},
    {4096, ACLBLAS_UPPER, ACLBLAS_OP_C, "n4096_UPPER_C_NON_UNIT", 5185.20},
};

} // namespace

// Back-to-back kernel timing: the device buffers stay resident and only the operator call is
// inside the timed window, so no host<->device staging cost leaks into the number.
// Sampling follows the task book: warm up first, then average > 50 valid per-call samples.
TEST_F(TpsvTest, PerfBenchmark)
{
    const int iters = 60; // task book 3.3: more than 50 valid samples after warm-up
    const int warmup = 10;
    const BlasFillMode fill("RANDOM_NORM_5_5");

    for (const auto& c : kPerfCases) {
        std::vector<float> ap = BuildPacked(c.n, c.uplo == ACLBLAS_UPPER, fill, 20260001u, false);
        std::vector<float> x = makeBlasStrided(c.n, 1, fill, 20260002u);
        const size_t apBytes = ap.size() * sizeof(float);
        const size_t xBytes = x.size() * sizeof(float);

        void* dAp = nullptr;
        void* dX = nullptr;
        ASSERT_EQ(aclrtMalloc(&dAp, apBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(dAp, apBytes, ap.data(), apBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(dX, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

        for (int i = 0; i < warmup; ++i) {
            aclblasStpsv(
                handle_, c.uplo, c.trans, ACLBLAS_NON_UNIT, c.n, static_cast<const float*>(dAp),
                static_cast<float*>(dX), 1);
        }
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

        std::vector<double> samples;
        samples.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            aclblasStpsv(
                handle_, c.uplo, c.trans, ACLBLAS_NON_UNIT, c.n, static_cast<const float*>(dAp),
                static_cast<float*>(dX), 1);
            ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        std::sort(samples.begin(), samples.end());
        double sum = 0.0;
        for (double s : samples) {
            sum += s;
        }
        const double mean = sum / static_cast<double>(samples.size());
        const double median = samples[samples.size() / 2];
        const double best = samples.front();
        // Acceptance criterion (task book section 3.3): the average per-call time must not
        // exceed the benchmark time itself. The 0.8 factor is already baked into budgetUs:
        // benchmark = A100_gpu_ms / 0.8 (see test_cases/gpu_baseline.csv), i.e. the operator
        // is allowed to be up to 1.25x slower than the raw A100 time. Dividing by 0.8 again
        // here would double-count it.
        std::printf(
            "[PERF] %-26s n=%-5d samples=%-3d mean=%9.2f us median=%9.2f us min=%9.2f us "
            "budget=%9.2f us budget/npu=%6.3f  %s\n",
            c.name, c.n, static_cast<int>(samples.size()), mean, median, best, c.budgetUs, c.budgetUs / mean,
            (mean <= c.budgetUs) ? "PASS" : "FAIL");
        std::fflush(stdout);

        aclrtFree(dAp);
        aclrtFree(dX);
    }
}
