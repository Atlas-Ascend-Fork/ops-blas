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
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "npu_event_timer.h"
#include "csv_loader.h"
#include "crot_param.h"
#include "crot_golden.h"
#include "crot_npu_wrapper.h"

class CrotTest : public BlasTest<CrotParam> {};

// ---------------------------------------------------------------- fixed tests
TEST_F(CrotTest, NullHandle)
{
    aclblasComplex x[5] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
    aclblasComplex y[5] = {{6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}};
    aclblasComplex s{0.8f, 0.0f};
    aclblasStatus_t ret = aclblasCrot_npu(nullptr, 5, x, 1, y, 1, 0.6f, s);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CrotTest, NullHandleTakesPrecedenceOverQuickReturn)
{
    EXPECT_EQ(aclblasCrot(nullptr, 0, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CrotTest, QuickReturnSkipsDataPointerValidation)
{
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, 0, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, -1, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_SUCCESS);
}

// ------------------------------------------------------------- CSV data setup
// Build a complex buffer following the same stride layout as makeBlasStrided.
// logical element i lives at (inc > 0) ? i*inc : (n-1-i)*|inc|. Real and imag
// parts are generated independently with seeds `seed` / `seed + 1000`, matching
// fill.h::makeBlasComplexMatrix.
// inc==0: the buffer is sized to n complex elements (as in the srot test), even
// though only element 0 is reused. |inc|==1 also uses n elements.
// NULLPTR fill yields an empty vector -> the test passes a nullptr.
// x and y must NOT share the same seed: with identical seeds the two vectors carry
// the same values, so a bug that swaps or mixes x/y inside the kernel would still
// produce matching golden. The offset keeps both vectors independent.
static constexpr uint32_t kYSeedOffset = 7919U; // prime, distinct from imag offset 1000

static inline std::vector<aclblasComplex> MakeCrotBuffer(int n, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || n <= 0) {
        return {};
    }
    const int64_t absInc = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    std::vector<float> realPart;
    std::vector<float> imagPart;
    if (absInc <= 1) {
        realPart = makeBlasArray(static_cast<int64_t>(n), fill, seed);
        imagPart = makeBlasArray(static_cast<int64_t>(n), fill, seed + 1000U);
    } else {
        realPart = makeBlasStrided(n, inc, fill, seed);
        imagPart = makeBlasStrided(n, inc, fill, seed + 1000U);
    }
    const size_t sz = std::min(realPart.size(), imagPart.size());
    std::vector<aclblasComplex> data(sz);
    for (size_t i = 0; i < sz; i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

static inline int CrotAbsInc(int inc)
{
    const int64_t v = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    if (v == 0)
        return 1; // zero stride reuses element 0
    return static_cast<int>(v);
}

// Scan the physical slots touched by the stride pattern for NaN / Inf.
static inline bool CrotBufferHasSpecial(const std::vector<aclblasComplex>& buf, int n, int inc, bool wantNaN)
{
    if (buf.empty())
        return false;
    const int step = CrotAbsInc(inc);
    for (int i = 0; i < n; i++) {
        const aclblasComplex& v = buf[static_cast<size_t>(i) * static_cast<size_t>(step)];
        const bool hit =
            wantNaN ? (std::isnan(v.real) || std::isnan(v.imag)) : (std::isinf(v.real) || std::isinf(v.imag));
        if (hit)
            return true;
    }
    return false;
}

// NaN-propagation: every output component (real and imag) must be NaN.
static inline bool CrotAllNaNAt(const aclblasComplex* buf, int n, int inc)
{
    if (buf == nullptr)
        return false;
    const int step = CrotAbsInc(inc);
    for (int i = 0; i < n; i++) {
        const aclblasComplex& v = buf[static_cast<size_t>(i) * static_cast<size_t>(step)];
        if (!std::isnan(v.real) || !std::isnan(v.imag))
            return false;
    }
    return true;
}

// Component-wise comparison for inputs containing Inf. A component that is
// non-finite on either side matches only NaN<->NaN or same-sign Inf<->Inf;
// finite components are checked with the FP32 mixed tolerance. This avoids
// false failures from 0*Inf / NaN differences between golden and kernel.
static inline bool CrotVerifyComponentWithSpecials(
    const aclblasComplex* actual, const aclblasComplex* golden, int n, int inc, bool compareImag, double atol,
    double rtol, const std::string& caseId)
{
    const int step = CrotAbsInc(inc);
    size_t failures = 0;
    for (int i = 0; i < n; i++) {
        const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(step);
        const float a = compareImag ? actual[idx].imag : actual[idx].real;
        const float g = compareImag ? golden[idx].imag : golden[idx].real;
        bool ok = false;
        if (std::isnan(a) || std::isnan(g)) {
            ok = std::isnan(a) && std::isnan(g);
        } else if (std::isinf(a) || std::isinf(g)) {
            ok = std::isinf(a) && std::isinf(g) && (std::signbit(a) == std::signbit(g));
        } else {
            const double diff = std::abs(static_cast<double>(a) - static_cast<double>(g));
            ok = diff <= atol + rtol * std::abs(static_cast<double>(g));
        }
        if (!ok) {
            if (failures < 5) {
                std::cout << "[" << caseId << "] mismatch at i=" << i << " actual=" << a << " golden=" << g
                          << std::endl;
            }
            failures++;
        }
    }
    std::cout << "[" << caseId << "] " << (failures == 0 ? "PASSED" : "FAILED") << " (special-value compare, "
              << failures << "/" << n << " failures)" << std::endl;
    return failures == 0;
}

// Compact the strided component into a dense array for Verifier.
static inline void CrotExtractComponent(const aclblasComplex* buf, int n, int inc, bool imag, std::vector<float>& out)
{
    const int step = CrotAbsInc(inc);
    out.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        const aclblasComplex& v = buf[static_cast<size_t>(i) * static_cast<size_t>(step)];
        out[static_cast<size_t>(i)] = imag ? v.imag : v.real;
    }
}

// Host x/y plus the special-value flags recorded before the operator overwrites
// the host buffers in place.
struct CrotHostInputs {
    std::vector<aclblasComplex> xHost;
    std::vector<aclblasComplex> yHost;
    aclblasComplex* xPtr = nullptr;
    aclblasComplex* yPtr = nullptr;
    bool inputHasNaN = false;
    bool inputHasInf = false;
};

static inline void BuildHostInputs(const CrotParam& p, CrotHostInputs& in)
{
    in.xHost = MakeCrotBuffer(p.n, p.incx, p.x, p.randomSeed);
    in.yHost = MakeCrotBuffer(p.n, p.incy, p.y, p.randomSeed + kYSeedOffset);

    in.xPtr = in.xHost.empty() ? nullptr : in.xHost.data();
    in.yPtr = in.yHost.empty() ? nullptr : in.yHost.data();

    // Record input special values before the operator overwrites the hosts.
    in.inputHasNaN =
        CrotBufferHasSpecial(in.xHost, p.n, p.incx, true) || CrotBufferHasSpecial(in.yHost, p.n, p.incy, true);
    in.inputHasInf =
        CrotBufferHasSpecial(in.xHost, p.n, p.incx, false) || CrotBufferHasSpecial(in.yHost, p.n, p.incy, false);
}

static inline aclblasStatus_t ExecuteOperator(
    aclblasHandle_t handle, aclrtStream stream, const CrotParam& p, const CrotHostInputs& in)
{
    const aclblasComplex sVal{p.sReal, p.sImag};
    return aclblasCrot_npu(
        handle, p.n, in.xPtr, p.incx, in.yPtr, p.incy, p.c, sVal, p.csPtrMode, p.cIsNull, p.sIsNull, stream);
}

// NaN inputs must propagate to every output component.
static inline void VerifyNaNPropagation(const CrotParam& p, const CrotHostInputs& in)
{
    const bool allNanX = CrotAllNaNAt(in.xPtr, p.n, p.incx);
    const bool allNanY = CrotAllNaNAt(in.yPtr, p.n, p.incy);
    std::cout << "[" << p.caseName << "] NaN-propagation check: x_allNan=" << allNanX << " y_allNan=" << allNanY
              << std::endl;
    EXPECT_TRUE(allNanX && allNanY);
}

// Inf inputs use the special-value comparator component-wise.
static inline void VerifyInfSpecials(
    const CrotParam& p, const CrotHostInputs& in, const std::vector<aclblasComplex>& goldenX,
    const std::vector<aclblasComplex>& goldenY)
{
    VerifyConfig xCfg;
    applyMixedTolerance(xCfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(p.n));
    VerifyConfig yCfg;
    applyMixedTolerance(yCfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(p.n));

    EXPECT_TRUE(CrotVerifyComponentWithSpecials(
        in.xPtr, goldenX.data(), p.n, p.incx, false, xCfg.mixedAtol, xCfg.mixedRtol, p.caseName + "_x_real"));
    EXPECT_TRUE(CrotVerifyComponentWithSpecials(
        in.xPtr, goldenX.data(), p.n, p.incx, true, xCfg.mixedAtol, xCfg.mixedRtol, p.caseName + "_x_imag"));
    EXPECT_TRUE(CrotVerifyComponentWithSpecials(
        in.yPtr, goldenY.data(), p.n, p.incy, false, yCfg.mixedAtol, yCfg.mixedRtol, p.caseName + "_y_real"));
    EXPECT_TRUE(CrotVerifyComponentWithSpecials(
        in.yPtr, goldenY.data(), p.n, p.incy, true, yCfg.mixedAtol, yCfg.mixedRtol, p.caseName + "_y_imag"));
}

// Regular FP32 mixed tolerance, real and imag compared separately.
static inline void VerifyTolerance(
    const CrotParam& p, const CrotHostInputs& in, const std::vector<aclblasComplex>& goldenX,
    const std::vector<aclblasComplex>& goldenY)
{
    VerifyConfig xCfg;
    applyMixedTolerance(xCfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(p.n));
    VerifyConfig yCfg;
    applyMixedTolerance(yCfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(p.n));

    std::vector<float> outXRe;
    std::vector<float> outXIm;
    std::vector<float> outYRe;
    std::vector<float> outYIm;
    std::vector<float> goldXRe;
    std::vector<float> goldXIm;
    std::vector<float> goldYRe;
    std::vector<float> goldYIm;
    CrotExtractComponent(in.xPtr, p.n, p.incx, false, outXRe);
    CrotExtractComponent(in.xPtr, p.n, p.incx, true, outXIm);
    CrotExtractComponent(in.yPtr, p.n, p.incy, false, outYRe);
    CrotExtractComponent(in.yPtr, p.n, p.incy, true, outYIm);
    CrotExtractComponent(goldenX.data(), p.n, p.incx, false, goldXRe);
    CrotExtractComponent(goldenX.data(), p.n, p.incx, true, goldXIm);
    CrotExtractComponent(goldenY.data(), p.n, p.incy, false, goldYRe);
    CrotExtractComponent(goldenY.data(), p.n, p.incy, true, goldYIm);

    EXPECT_TRUE(Verifier::verifyVector(
        outXRe.data(), goldXRe.data(), static_cast<size_t>(p.n), 1, xCfg, p.caseName + "_x_real"));
    EXPECT_TRUE(Verifier::verifyVector(
        outXIm.data(), goldXIm.data(), static_cast<size_t>(p.n), 1, xCfg, p.caseName + "_x_imag"));
    EXPECT_TRUE(Verifier::verifyVector(
        outYRe.data(), goldYRe.data(), static_cast<size_t>(p.n), 1, yCfg, p.caseName + "_y_real"));
    EXPECT_TRUE(Verifier::verifyVector(
        outYIm.data(), goldYIm.data(), static_cast<size_t>(p.n), 1, yCfg, p.caseName + "_y_imag"));
}

// Recompute the CPU golden from fresh inputs and compare it with the NPU result.
static inline void VerifyResult(aclblasHandle_t handle, const CrotParam& p, const CrotHostInputs& in)
{
    std::vector<aclblasComplex> goldenX = MakeCrotBuffer(p.n, p.incx, p.x, p.randomSeed);
    std::vector<aclblasComplex> goldenY = MakeCrotBuffer(p.n, p.incy, p.y, p.randomSeed + kYSeedOffset);
    const aclblasComplex sVal{p.sReal, p.sImag};
    const float* cArg = p.cIsNull ? nullptr : &p.c;
    const aclblasComplex* sArg = p.sIsNull ? nullptr : &sVal;
    aclblasCrot_cpu(handle, p.n, goldenX.data(), p.incx, goldenY.data(), p.incy, cArg, sArg);

    if (in.inputHasNaN) {
        VerifyNaNPropagation(p, in);
    } else if (in.inputHasInf) {
        VerifyInfSpecials(p, in, goldenX, goldenY);
    } else {
        VerifyTolerance(p, in, goldenX, goldenY);
    }
}

// The operator rejects a strided offset that does not fit in int32 with
// ACLBLAS_STATUS_INVALID_VALUE. The CSV path cannot reach that guard: aclblasCrot_npu
// allocates (n-1)*|inc|+1 elements before calling the operator, so INT32_MIN would need
// ~17 GB and fail allocation first. Call the public API directly with tiny device
// buffers so the guard itself is exercised.
TEST_F(CrotTest, StridedOffsetOverflowRejected)
{
    constexpr int n = 3;
    void* dx = nullptr;
    void* dy = nullptr;
    ASSERT_EQ(aclrtMalloc(&dx, n * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dy, n * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    const float c = 0.6f;
    const aclblasComplex s{0.8f, 0.1f};
    auto* px = static_cast<aclblasComplex*>(dx);
    auto* py = static_cast<aclblasComplex*>(dy);

    // (n-1)*|INT32_MIN| = 2 * 2^31 overflows int32 -> rejected
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, n, px, INT32_MIN, py, 1, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, n, px, 1, py, INT32_MIN, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    // n = 3 with |inc| = INT32_MAX gives 2*INT32_MAX > INT32_MAX -> rejected
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, n, px, INT32_MAX, py, 1, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    // Legal boundary: n = 1 keeps the access span at a single element, so |inc| = INT32_MAX
    // is accepted without requiring a 16 GiB buffer. Using n = 2 here would make the kernel
    // touch x[2*INT32_MAX]; doing that with a small allocation is a deliberate out-of-bounds
    // access, so the full-span boundary is covered by the wrapper-allocated TC_EX_STBIG cases.
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, 1, px, INT32_MAX, py, 1, &c, &s), ACLBLAS_STATUS_SUCCESS);
    // The call is asynchronous: drain the stream before the buffers are released.
    ASSERT_EQ(aclrtSynchronizeStream(CrotTest::stream_), ACL_SUCCESS);

    aclrtFree(dx);
    aclrtFree(dy);
}

TEST_P(CrotTest, CsvDriven)
{
    const auto& p = GetParam();

    // Step 1: host input data (x/y are in-place inputs+outputs).
    CrotHostInputs inputs;
    BuildHostInputs(p, inputs);

    // Step 2: run on NPU. cIsNull/sIsNull force the nullptr negative cases.
    aclblasStatus_t ret = ExecuteOperator(CrotTest::handle_, CrotTest::stream_, p, inputs);

    // Step 3: return code.
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.n <= 0 || inputs.xPtr == nullptr || inputs.yPtr == nullptr)
        return; // no-op / nullptr input: x/y untouched, nothing numeric to check

    // Steps 4-5: CPU golden and numeric verification.
    VerifyResult(CrotTest::handle_, p, inputs);
}

// With a stride larger than one, the elements between two logical slots must keep their
// original values: a wrong index conversion would silently corrupt them, and a precision
// check that only looks at the slots would not notice. Fill the whole span with sentinels,
// rotate the slots, and compare every non-slot position against its sentinel.
TEST_F(CrotTest, GapElementsUntouched)
{
    constexpr int n = 9;
    constexpr int inc = 4;
    constexpr int span = (n - 1) * inc + 1;
    const size_t bytes = static_cast<size_t>(span) * sizeof(aclblasComplex);

    std::vector<aclblasComplex> xHost(static_cast<size_t>(span));
    std::vector<aclblasComplex> yHost(static_cast<size_t>(span));
    for (int i = 0; i < span; ++i) {
        const float v = static_cast<float>(4000 + i);
        xHost[static_cast<size_t>(i)] = aclblasComplex{v, v + 0.5f};
        yHost[static_cast<size_t>(i)] = aclblasComplex{-v, -v - 0.5f};
    }

    void* dx = nullptr;
    void* dy = nullptr;
    ASSERT_EQ(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dx, bytes, xHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dy, bytes, yHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    const float c = 0.6f;
    const aclblasComplex s{0.8f, 0.1f};
    ASSERT_EQ(
        aclblasCrot(
            CrotTest::handle_, n, static_cast<aclblasComplex*>(dx), inc, static_cast<aclblasComplex*>(dy), inc, &c, &s),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(CrotTest::stream_), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(xHost.data(), bytes, dx, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(yHost.data(), bytes, dy, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    aclrtFree(dx);
    aclrtFree(dy);

    for (int i = 0; i < span; ++i) {
        if (i % inc == 0) {
            continue; // logical slot, rotated in place
        }
        const float v = static_cast<float>(4000 + i);
        EXPECT_EQ(xHost[static_cast<size_t>(i)].real, v) << "x gap element written at " << i;
        EXPECT_EQ(yHost[static_cast<size_t>(i)].real, -v) << "y gap element written at " << i;
    }
}

namespace {
struct CrotPerfShape {
    int n;
    double budgetUs;
};

constexpr int CROT_PERF_WARMUP = 10;
constexpr int CROT_PERF_BATCH = 100;
constexpr int CROT_PERF_ROUNDS = 3;

std::vector<aclblasComplex> MakeCrotPerfHostBuffer(size_t count, int realMod, int imagMod)
{
    std::vector<aclblasComplex> host(count);
    for (size_t i = 0; i < count; ++i) {
        host[i] = aclblasComplex{
            static_cast<float>(i % static_cast<size_t>(realMod)) * 0.5f,
            static_cast<float>(i % static_cast<size_t>(imagMod)) * 0.25f};
    }
    return host;
}

// One pair of ACL events brackets CROT_PERF_BATCH back-to-back launches, so the event
// overhead is amortized and the value reflects the kernel itself.
bool RunCrotPerfRounds(
    aclblasHandle_t handle, aclrtStream stream, int n, aclblasComplex* xDev, aclblasComplex* yDev, const float* c,
    const aclblasComplex* s, double& bestUs, double& avgUs)
{
    aclrtEvent startEvent = nullptr;
    aclrtEvent stopEvent = nullptr;
    if (aclrtCreateEvent(&startEvent) != ACL_SUCCESS || aclrtCreateEvent(&stopEvent) != ACL_SUCCESS) {
        return false;
    }
    for (int i = 0; i < CROT_PERF_WARMUP; ++i) {
        if (aclblasCrot(handle, n, xDev, 1, yDev, 1, c, s) != ACLBLAS_STATUS_SUCCESS) {
            aclrtDestroyEvent(startEvent);
            aclrtDestroyEvent(stopEvent);
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        aclrtDestroyEvent(startEvent);
        aclrtDestroyEvent(stopEvent);
        return false;
    }

    std::vector<double> roundUs;
    bool ok = true;
    for (int round = 0; round < CROT_PERF_ROUNDS && ok; ++round) {
        ok = aclrtRecordEvent(startEvent, stream) == ACL_SUCCESS;
        for (int i = 0; i < CROT_PERF_BATCH && ok; ++i) {
            ok = aclblasCrot(handle, n, xDev, 1, yDev, 1, c, s) == ACLBLAS_STATUS_SUCCESS;
        }
        ok = ok && aclrtRecordEvent(stopEvent, stream) == ACL_SUCCESS;
        ok = ok && aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        float elapsedMs = 0.0f;
        ok = ok && aclrtEventElapsedTime(&elapsedMs, startEvent, stopEvent) == ACL_SUCCESS;
        if (ok) {
            roundUs.push_back(static_cast<double>(elapsedMs) * 1000.0 / CROT_PERF_BATCH);
        }
    }
    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(stopEvent);
    if (!ok || roundUs.empty()) {
        return false;
    }
    bestUs = *std::min_element(roundUs.begin(), roundUs.end());
    avgUs = blas_test::MeanUs(roundUs);
    return true;
}
} // namespace

// Performance gate for the three shapes documented in blas/rot/README.md. Each round is
// bracketed by one pair of ACL events around CROT_PERF_BATCH launches, the data stays
// resident on the device (no H2D/D2H, no golden comparison), which matches the acceptance
// metric of kernel average per launch.
TEST_F(CrotTest, PerformanceGate)
{
    const CrotPerfShape shapes[] = {{1048576, 21.96}, {2097152, 43.70}, {4194304, 119.52}};
    const float c = 0.6f;
    const aclblasComplex s{0.8f, 0.1f};

    for (const auto& shape : shapes) {
        const size_t count = static_cast<size_t>(shape.n);
        const size_t bytes = count * sizeof(aclblasComplex);
        void* dx = nullptr;
        void* dy = nullptr;
        ASSERT_EQ(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        auto* px = static_cast<aclblasComplex*>(dx);
        auto* py = static_cast<aclblasComplex*>(dy);
        const std::vector<aclblasComplex> hx = MakeCrotPerfHostBuffer(count, 17, 23);
        const std::vector<aclblasComplex> hy = MakeCrotPerfHostBuffer(count, 29, 31);
        ASSERT_EQ(aclrtMemcpy(dx, bytes, hx.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(dy, bytes, hy.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

        double bestUs = 0.0;
        double avgUs = 0.0;
        const bool ok = RunCrotPerfRounds(CrotTest::handle_, CrotTest::stream_, shape.n, px, py, &c, &s, bestUs, avgUs);
        aclrtFree(dx);
        aclrtFree(dy);
        ASSERT_TRUE(ok) << "event timing failed for n=" << shape.n;

        std::cout << "[PERF] n=" << shape.n << " kernel_best_us=" << bestUs << " kernel_avg_us=" << avgUs
                  << " budget_us=" << shape.budgetUs << std::endl;
        EXPECT_LE(bestUs, shape.budgetUs)
            << "n=" << shape.n << " best per-launch kernel time exceeds the budget in blas/rot/README.md";
    }
}

namespace {
std::vector<aclblasComplex> MakeCrotAliasBuffer(int n, int inc, uint32_t seed)
{
    const int64_t absInc = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    const size_t span = static_cast<size_t>((n - 1) * absInc + 1);
    std::vector<aclblasComplex> data(span);
    for (size_t i = 0; i < span; ++i) {
        const float v = static_cast<float>((i * 37U + seed) % 97U) * 0.1f - 4.0f;
        data[i] = aclblasComplex{v, v * 0.5f + 1.0f};
    }
    return data;
}

// Runs the operator with x and y pointing at the same allocation and compares the whole
// span against the serial CPU golden evaluated on an identical alias.
void CheckCrotAliasedRun(aclblasHandle_t handle, aclrtStream stream, int n, int incx, int incy)
{
    const int maxInc = std::max(std::abs(incx), std::abs(incy));
    std::vector<aclblasComplex> device = MakeCrotAliasBuffer(n, maxInc, 7001U);
    std::vector<aclblasComplex> golden = device;
    const float c = 0.6f;
    const aclblasComplex s{0.8f, 0.1f};
    ASSERT_EQ(aclblasCrot_cpu(handle, n, golden.data(), incx, golden.data(), incy, &c, &s), ACLBLAS_STATUS_SUCCESS);

    const size_t bytes = device.size() * sizeof(aclblasComplex);
    void* dev = nullptr;
    ASSERT_EQ(aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dev, bytes, device.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    auto* ptr = static_cast<aclblasComplex*>(dev);
    ASSERT_EQ(aclblasCrot(handle, n, ptr, incx, ptr, incy, &c, &s), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(device.data(), bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    aclrtFree(dev);

    for (size_t i = 0; i < device.size(); ++i) {
        EXPECT_NEAR(device[i].real, golden[i].real, 1e-5f) << "alias real mismatch at " << i;
        EXPECT_NEAR(device[i].imag, golden[i].imag, 1e-5f) << "alias imag mismatch at " << i;
    }
}
} // namespace

// Aliased input: x and y share one allocation. The CSV path always allocates them separately,
// so these two shapes are the only coverage. Equal strides keep every element independent,
// different strides make two logical slots share a physical element and the host therefore
// routes the call to the single-thread serial path, keeping the result equal to the golden.
TEST_F(CrotTest, AliasedBuffersMatchGolden)
{
    CheckCrotAliasedRun(CrotTest::handle_, CrotTest::stream_, 1024, 2, 2);
    CheckCrotAliasedRun(CrotTest::handle_, CrotTest::stream_, 1024, 1, -1);
}

// Guard the CSV composition while loading. A truncated file, a failed copy step or a
// line-ending change would otherwise shrink the suite silently and still report green.
static std::vector<CrotParam> LoadAndValidateCrotCases(const std::string& csvPath)
{
    constexpr size_t kExpectedTotal = 1206;
    constexpr size_t kExpectedPerfCases = 200;
    std::vector<CrotParam> cases = GetCasesFromCsv<CrotParam>(csvPath);
    if (cases.size() != kExpectedTotal) {
        throw std::runtime_error(
            "aclblasCrot CSV case count mismatch, expected " + std::to_string(kExpectedTotal) + " but got " +
            std::to_string(cases.size()));
    }
    std::unordered_set<std::string> names;
    size_t perfCases = 0;
    for (const auto& item : cases) {
        if (!names.insert(item.caseName).second) {
            throw std::runtime_error("aclblasCrot CSV has a duplicate case_name " + item.caseName);
        }
        if (item.caseName.rfind("TC_PF_", 0) == 0) {
            ++perfCases;
        }
    }
    if (perfCases != kExpectedPerfCases) {
        throw std::runtime_error(
            "aclblasCrot CSV performance case count mismatch, expected " + std::to_string(kExpectedPerfCases) +
            " but got " + std::to_string(perfCases));
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Crot, CrotTest, ::testing::ValuesIn(LoadAndValidateCrotCases(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CrotParam>);
