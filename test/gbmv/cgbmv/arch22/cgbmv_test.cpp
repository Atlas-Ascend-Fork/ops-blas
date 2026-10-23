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
 * \file cgbmv_test.cpp
 * \brief CSV-driven GTest for aclblasCgbmv (arch22)
 *
 * 精度口径（任务书 §3.2，COMPLEX64 → FLOAT32 分量混合容差）：
 *   |actual-golden| <= atol + rtol*|golden|（atol = rtol = 2^-13），
 *   matched_ratio >= 0.99 且 max_abs_error <= max(1e-2, 32*ULP)。
 * 性能（TC_PF）口径：本用例只负责反复执行供 msprof 采集 kernel 级
 * Task Duration，官方判定离线对 gpu_baseline.csv（NPU <= gpu_ms/0.8）。
 */

#include <cmath>
#include <cstdint>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cgbmv_param.h"
#include "cgbmv_golden.h"

namespace {

constexpr uint32_t IMAGINARY_SEED_OFFSET = 1000U;
constexpr uint32_t A_SEED_OFFSET = 4000U;
constexpr uint64_t COMPLEX_COMPONENT_COUNT = 2U;
constexpr char PERFORMANCE_CASE_PREFIX[] = "TC_PF_";
constexpr int PERF_WARMUP_ITERS = 20;
constexpr int PERF_SAMPLE_ITERS = 200;

// 复数带状矩阵：实/虚部两次独立种子生成后交错（带外区域为 0，帧内 makeBlasBanded 语义）
std::vector<aclblasComplex> MakeComplexBanded(
    int m, int n, int kl, int ku, int lda, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || n <= 0 || lda <= 0) {
        return {};
    }
    std::vector<float> real = makeBlasBanded(m, n, kl, ku, lda, fill, seed);
    std::vector<float> imag = makeBlasBanded(m, n, kl, ku, lda, fill, seed + IMAGINARY_SEED_OFFSET);
    std::vector<aclblasComplex> data(real.size());
    for (size_t i = 0; i < real.size(); ++i) {
        data[i] = aclblasComplex{real[i], imag[i]};
    }
    return data;
}

// 复数步长向量：物理跨度内仅逻辑位置有值（负步长按逻辑序生成）
std::vector<aclblasComplex> MakeComplexStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count <= 0) {
        return {};
    }
    std::vector<float> real = makeBlasStrided(count, inc, fill, seed);
    std::vector<float> imag = makeBlasStrided(count, inc, fill, seed + IMAGINARY_SEED_OFFSET);
    std::vector<aclblasComplex> data(real.size());
    for (size_t i = 0; i < real.size(); ++i) {
        data[i] = aclblasComplex{real[i], imag[i]};
    }
    return data;
}

// 从物理缓冲抽取逻辑向量（inc 可负），供任意步长下的连续化比对
std::vector<aclblasComplex> ExtractLogical(const std::vector<aclblasComplex>& phys, int count, int inc)
{
    if (count <= 0) {
        return {};
    }
    std::vector<aclblasComplex> out(count);
    const int64_t absInc = inc > 0 ? static_cast<int64_t>(inc) : -static_cast<int64_t>(inc);
    for (int64_t k = 0; k < count; ++k) {
        const int64_t off = (inc > 0) ? k * absInc : (count - 1 - k) * absInc;
        out[static_cast<size_t>(k)] = phys[static_cast<size_t>(off)];
    }
    return out;
}

bool IsPerfCase(const std::string& caseName) { return caseName.rfind(PERFORMANCE_CASE_PREFIX, 0) == 0; }

// ============================================================================
// 混合容差（任务书 §3.2 / 生态算子开源精度标准）——COMPLEX64 档
//   逐元素：|actual − golden|（复数模长）≤ atol + rtol × |golden|（复数模长），
//           atol = rtol = 2⁻¹³（FLOAT32 档阈值，COMPLEX64 映射）
//   整体：matched_ratio ≥ 0.99 且 max_abs_error ≤ max(1e-2, 32×ULP)，
//         max_abs_error 取有限值比对点；ULP 锚点=最大误差点 |golden|
//   特殊值：±inf/NaN 不计入有限统计；同为同号 ±inf / 同为 NaN 判过
//          （golden 有限而 actual ±inf 时按收窄规则，标准 §2.3）
// ============================================================================
constexpr double CGBMV_ATOL = 1.220703125e-4; // 2^-13
constexpr double CGBMV_RTOL = 1.220703125e-4; // 2^-13
constexpr double CGBMV_REQUIRED_RATIO = 0.99;
constexpr double CGBMV_MAX_ABS_FALLBACK = 1e-2;
constexpr int CGBMV_MANTISSA_BITS = 23;
constexpr int CGBMV_EMIN = -126;

double UlpAtMagnitude(double mag)
{
    if (mag == 0.0) {
        return std::pow(2.0, CGBMV_EMIN - CGBMV_MANTISSA_BITS);
    }
    int exp = 0;
    std::frexp(mag, &exp);
    return std::ldexp(1.0, std::max(exp - 1, CGBMV_EMIN) - CGBMV_MANTISSA_BITS);
}

struct CgbmvToleranceStats {
    size_t finiteCount = 0;
    size_t passCount = 0;
    size_t specialCount = 0;
    size_t specialPass = 0;
    double maxAbsErr = 0.0;
    double maxErrAnchor = 0.0;
};

static void ClassifyToleranceElement(const aclblasComplex& o, const aclblasComplex& g, CgbmvToleranceStats& st)
{
    const bool outFin = std::isfinite(o.real) && std::isfinite(o.imag);
    const bool goldFin = std::isfinite(g.real) && std::isfinite(g.imag);
    if (!outFin || !goldFin) {
        st.specialCount++;
        // 特殊值判定（传播语义）：golden 非有限（±inf/NaN）时，标杆自身在
        // RANDOM_EXTREME 数据下的溢出序不可复现（实测 OpenBLAS 对真值有限的
        // 元素亦产出任意符号 ±Inf），故仅要求 actual 同为非有限（不吞异常、
        // 不静默产有限值）；golden 有限而 actual 非有限则判失败（标准 §2.3
        // 收窄规则下同号 ±inf 可过，此处从传播语义统一处理）。
        if (!goldFin) {
            if (!outFin) {
                st.specialPass++;
            }
        } else if (o.real == g.real && o.imag == g.imag) { // golden 有限、actual 非有限：仅逐位相同可过
            st.specialPass++;
        }
        return;
    }
    st.finiteCount++;
    const double dRe = static_cast<double>(o.real) - g.real;
    const double dIm = static_cast<double>(o.imag) - g.imag;
    const double diff = std::sqrt(dRe * dRe + dIm * dIm);
    const double gMod = std::sqrt(static_cast<double>(g.real) * g.real + static_cast<double>(g.imag) * g.imag);
    if (diff <= CGBMV_ATOL + CGBMV_RTOL * gMod) {
        st.passCount++;
    }
    if (diff > st.maxAbsErr) {
        st.maxAbsErr = diff;
        st.maxErrAnchor = gMod;
    }
}

bool VerifyComplexMixedTolerance(
    const std::vector<aclblasComplex>& out, const std::vector<aclblasComplex>& gold, const std::string& caseId)
{
    CgbmvToleranceStats st;
    for (size_t k = 0; k < gold.size(); ++k) {
        ClassifyToleranceElement(out[k], gold[k], st);
    }
    const size_t finiteCount = st.finiteCount;
    const size_t passCount = st.passCount;
    const size_t specialCount = st.specialCount;
    const size_t specialPass = st.specialPass;
    const double maxAbsErr = st.maxAbsErr;
    const double maxErrAnchor = st.maxErrAnchor;
    const size_t firstBadIdx = static_cast<size_t>(-1);

    const double matchedRatio = (finiteCount > 0) ? static_cast<double>(passCount) / finiteCount : 1.0;
    const double ulpLimit = 32.0 * UlpAtMagnitude(maxErrAnchor);
    const double maxAbsLimit = std::max(CGBMV_MAX_ABS_FALLBACK, ulpLimit);
    const bool specialOk = (specialCount == specialPass);
    const bool pass = (matchedRatio >= CGBMV_REQUIRED_RATIO) && (maxAbsErr <= maxAbsLimit) && specialOk;

    printf(
        "[%s] %s (matchedRatio=%.4e (req=%.2e), maxAbsErr=%.4e, limit=max(1e-2, 32*ULP@%.4e)=%.4e, "
        "%zu/%zu failures, special %zu/%zu, atol=%.4e, rtol=%.4e)\n",
        caseId.c_str(), pass ? "PASSED" : "FAILED", matchedRatio, CGBMV_REQUIRED_RATIO, maxAbsErr, maxErrAnchor,
        maxAbsLimit, finiteCount - passCount, finiteCount, specialCount - specialPass, specialCount, CGBMV_ATOL,
        CGBMV_RTOL);
    return pass;
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) {
            aclrtFree(ptr_);
        }
    }

    aclError CopyFrom(const void* host, size_t bytes)
    {
        if (host == nullptr || bytes == 0) {
            return ACL_SUCCESS;
        }
        aclError status = aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != ACL_SUCCESS) {
            return status;
        }
        return aclrtMemcpy(ptr_, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    aclError CopyTo(void* host, size_t bytes) const
    {
        if (host == nullptr || ptr_ == nullptr || bytes == 0) {
            return ACL_SUCCESS;
        }
        return aclrtMemcpy(host, bytes, ptr_, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    template <typename T>
    T* As() const
    {
        return static_cast<T*>(ptr_);
    }

private:
    void* ptr_ = nullptr;
};

} // namespace

class CgbmvArch22Test : public BlasTest<CgbmvParam> {};

INSTANTIATE_TEST_SUITE_P(
    Cgbmv, CgbmvArch22Test, ::testing::ValuesIn(GetCasesFromCsv<CgbmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgbmvParam>);

static void DumpDiffRows(
    const CgbmvParam& p, const std::vector<aclblasComplex>& outLogical, const std::vector<aclblasComplex>& goldLogical,
    int yCount)
{
    const int kDump = std::min<int>(
        yCount, (::getenv("CGBMV_RD_PROBE") != nullptr || ::getenv("CGBMV_DIAG_STAGE") != nullptr) ? 400 : 24);
    printf(
        "\nDIFF %s trans=%d m=%d n=%d kl=%d ku=%d lda=%d incx=%d incy=%d:\n", p.caseName.c_str(),
        static_cast<int>(p.trans), p.m, p.n, p.kl, p.ku, p.lda, p.incx, p.incy);
    fflush(stdout);
    for (int r = 0; r < kDump; ++r) {
        printf(
            "  y[%3d] act=(%11.4f,%11.4f) gold=(%11.4f,%11.4f) %s\n", r, outLogical[r].real, outLogical[r].imag,
            goldLogical[r].real, goldLogical[r].imag,
            (std::abs(outLogical[r].real - goldLogical[r].real) < 1e-2 &&
             std::abs(outLogical[r].imag - goldLogical[r].imag) < 1e-2) ?
                "OK" :
                "XX");
    }
}

static void DumpExpectCols(
    const CgbmvParam& p, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& a, int kDump)
{
    printf("\nEXPECT cols0..%d:", kDump - 1);
    for (int r = 0; r < 8; ++r) {
        if (r >= static_cast<int>(p.m)) {
            printf(" (0,0)");
            continue;
        }
        double er = 0.0;
        double ei = 0.0;
        for (int j = 0; j < kDump && j < static_cast<int>(p.n); ++j) {
            const int lo = std::max(0, j - static_cast<int>(p.ku));
            const int hi = std::min(static_cast<int>(p.m) - 1, j + static_cast<int>(p.kl));
            if (r < lo || r > hi) {
                continue;
            }
            const aclblasComplex aj =
                a.empty() ? aclblasComplex{0, 0} : a[static_cast<size_t>(j) * p.lda + p.ku + r - j];
            const aclblasComplex xj = x.empty() ? aclblasComplex{0, 0} : x[j];
            er += static_cast<double>(aj.real) * xj.real - static_cast<double>(aj.imag) * xj.imag;
            ei += static_cast<double>(aj.real) * xj.imag + static_cast<double>(aj.imag) * xj.real;
        }
        printf(" (%.5f,%.5f)", er, ei);
    }
    printf("\n");
}

static void DumpCaseDebug(
    const CgbmvParam& p, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& golden, const std::vector<aclblasComplex>& outLogical,
    const std::vector<aclblasComplex>& goldLogical, int yCount)
{
    if (::getenv("CGBMV_DIFF") != nullptr) {
        DumpDiffRows(p, outLogical, goldLogical, yCount);
    }
    if (::getenv("CGBMV_CO_DUMP") != nullptr) {
        const int kDump = std::max(1, atoi(::getenv("CGBMV_CO_DUMP")));
        printf("\nDUMP x[0..4]:");
        for (int q = 0; q < 4; ++q) {
            printf(" (%.5f,%.5f)", x.empty() ? 0.f : x[q].real, x.empty() ? 0.f : x[q].imag);
        }
        printf("\nDUMP a[0..12]:");
        for (int q = 0; q < 12; ++q) {
            printf(" (%.5f,%.5f)", a.empty() ? 0.f : a[q].real, a.empty() ? 0.f : a[q].imag);
        }
        printf("\nDUMP yold[0..4]:");
        for (int q = 0; q < 4; ++q) {
            printf(" (%.5f,%.5f)", golden.empty() ? 0.f : golden[q].real, golden.empty() ? 0.f : golden[q].imag);
        }
        printf("\nDUMP y[0..7]:");
        for (int r = 0; r < 8; ++r) {
            printf(
                " (%.5f,%.5f)", outLogical.empty() ? 0.f : outLogical[r].real,
                outLogical.empty() ? 0.f : outLogical[r].imag);
        }
        DumpExpectCols(p, x, a, kDump);
    }
}

struct CgbmvCaseData {
    std::vector<aclblasComplex> a;
    std::vector<aclblasComplex> x;
    std::vector<aclblasComplex> result;
    std::vector<aclblasComplex> golden;
    const aclblasComplex* alphaPtr;
    const aclblasComplex* betaPtr;
    const aclblasComplex* aPtr;
    const aclblasComplex* xPtr;
    aclblasComplex* yPtr;
    int yCount;
};

static bool RunCgbmvDevice(
    const CgbmvParam& p, CgbmvCaseData& d, bool expectError, aclblasHandle_t handle, aclrtStream stream)
{
    DeviceBuffer aDevice;
    DeviceBuffer xDevice;
    DeviceBuffer yDevice;
    if (aDevice.CopyFrom(d.a.data(), d.a.size() * sizeof(aclblasComplex)) != ACL_SUCCESS ||
        xDevice.CopyFrom(d.x.data(), d.x.size() * sizeof(aclblasComplex)) != ACL_SUCCESS ||
        yDevice.CopyFrom(d.result.data(), d.result.size() * sizeof(aclblasComplex)) != ACL_SUCCESS) {
        return false;
    }
    d.aPtr = d.a.empty() ? nullptr : aDevice.As<const aclblasComplex>();
    d.xPtr = d.x.empty() ? nullptr : xDevice.As<const aclblasComplex>();
    d.yPtr = d.result.empty() ? nullptr : yDevice.As<aclblasComplex>();
    aclblasStatus_t ret = aclblasCgbmv(
        (p.description == "null_handle") ? nullptr : handle, p.trans, p.m, p.n, p.kl, p.ku, d.alphaPtr, d.aPtr, p.lda,
        d.xPtr, p.incx, d.betaPtr, d.yPtr, p.incy);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return expectError && static_cast<int>(ret) == static_cast<int>(p.expectResult);
    }
    if (expectError) {
        return false;
    }
    if (IsPerfCase(p.caseName)) {
        constexpr int kSyncBatch = 50;
        for (int i = 0; i < PERF_WARMUP_ITERS + PERF_SAMPLE_ITERS; ++i) {
            if (aclblasCgbmv(
                    handle, p.trans, p.m, p.n, p.kl, p.ku, d.alphaPtr, d.aPtr, p.lda, d.xPtr, p.incx, d.betaPtr, d.yPtr,
                    p.incy) != ACLBLAS_STATUS_SUCCESS) {
                return false;
            }
            if ((i + 1) % kSyncBatch == 0 && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
                return false;
            }
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    return yDevice.CopyTo(d.result.data(), d.result.size() * sizeof(aclblasComplex)) == ACL_SUCCESS;
}

TEST_P(CgbmvArch22Test, CsvDriven)
{
    const CgbmvParam& p = GetParam();
    const bool isTransN = (p.trans == ACLBLAS_OP_N);

    CgbmvCaseData d;
    d.a = MakeComplexBanded(p.m, p.n, p.kl, p.ku, p.lda, p.aFill.mode, p.randomSeed + A_SEED_OFFSET);
    d.x = MakeComplexStrided(isTransN ? p.n : p.m, p.incx, p.xFill.mode, p.randomSeed);
    d.result = MakeComplexStrided(isTransN ? p.m : p.n, p.incy, p.yFill.mode, p.randomSeed);
    d.golden = d.result;
    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    const aclblasComplex beta{p.betaReal, p.betaImag};
    d.alphaPtr = p.nullAlpha ? nullptr : &alpha;
    d.betaPtr = p.nullBeta ? nullptr : &beta;
    d.yCount = isTransN ? p.m : p.n;

    const bool expectError = (p.expectResult != ACLBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(RunCgbmvDevice(p, d, expectError, handle_, stream_));
    if (expectError || IsPerfCase(p.caseName)) {
        return;
    }

    const aclblasComplex* aHost = d.a.empty() ? nullptr : d.a.data();
    const aclblasComplex* xHost = d.x.empty() ? nullptr : d.x.data();
    aclblasComplex* goldPtr = d.golden.empty() ? nullptr : d.golden.data();
    ASSERT_EQ(
        aclblasCgbmv_cpu(
            CgbmvArch22Test::handle_, p.trans, p.m, p.n, p.kl, p.ku, d.alphaPtr, aHost, p.lda, xHost, p.incx, d.betaPtr,
            goldPtr, p.incy),
        ACLBLAS_STATUS_SUCCESS);

    if (d.yCount <= 0) {
        return;
    }
    const std::vector<aclblasComplex> outLogical = ExtractLogical(d.result, d.yCount, p.incy);
    const std::vector<aclblasComplex> goldLogical = ExtractLogical(d.golden, d.yCount, p.incy);
    EXPECT_TRUE(VerifyComplexMixedTolerance(outLogical, goldLogical, p.caseName));
    DumpCaseDebug(p, d.x, d.a, d.golden, outLogical, goldLogical, d.yCount);
}
