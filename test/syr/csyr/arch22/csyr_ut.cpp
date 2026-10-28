/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// csyr unit tests (GTest suite csyr_ut): host validation-chain (numeric
// error-code matrix, ordering pins documented inline at the pinning
// tests), quick-return (n==0 / alpha==(0,0)), alpha dual-location
// (host/device pointer branches must agree bitwise), kernel incremental-segment
// (incx != +1 pack path vs streamed self-implemented golden), double-
// buffered batch path (tail blocks, event-slot balance across
// CSYR_BATCH_COLS sweeps, n=4864 limit), and zigzag column-split
// invariants. Scope details live inline at each test; the CSV ST
// suites run from the csyr_test binary.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csyr_fill_ext.h"
#include "csyr_golden.h"
#include "csyr_npu_wrapper.h"
#include "csyr_verify_ext.h"

namespace {

using csyr_verify::CountUntouchedBitMismatch;
using csyr_verify::CountWholeBitMismatch;
using csyr_verify::CsyrToleranceGate;

// Numeric error-code pins (include/cann_ops_blas_common.h)
constexpr int kErrHandleNullptr = 9;  // ACLBLAS_STATUS_HANDLE_IS_NULLPTR
constexpr int kErrInvalidEnum = 10;   // ACLBLAS_STATUS_INVALID_ENUM
constexpr int kErrInvalidValue = 3;   // ACLBLAS_STATUS_INVALID_VALUE

} // namespace

// UT fixture: shares the suite-wide ACL guard/handle from blas_test.h
// (TEST_DEVICE_ID is compile-time injected by build.sh --device=1).
class CsyrUt : public BlasTest<int> {
public:
    static aclblasHandle_t UtHandle()
    {
        return handle_;
    }
};

namespace {

// Device-resident 8-byte alpha scalar (the operator must aclrtMemcpy D2H
// read-back it before the (0,0) decision).
aclblasStatus_t UtMakeDeviceAlpha(float re, float im, void** dAlpha)
{
    aclblasComplex host;
    host.real = re;
    host.imag = im;
    aclError aclRet = aclrtMalloc(dAlpha, sizeof(host), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(*dAlpha, sizeof(host), &host, sizeof(host), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dAlpha);
        *dAlpha = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Streamed golden comparison over the uplo triangle (diagonal included), same
// tolerance semantics as the ST frame (spec gate + MERE/MARE engineering gate).
bool UtGoldenCompareTriangle(const std::string& tag, aclblasFillMode_t uplo, int n, const aclblasComplex& alphaVal,
    const aclblasComplex* xPtr, int incx, const std::vector<aclblasComplex>& aOrig,
    const std::vector<aclblasComplex>& aActual, int lda)
{
    CsyrToleranceGate gateRe;
    CsyrToleranceGate gateIm;
    std::vector<aclblasComplex> colGold(static_cast<size_t>(n));
    for (int j = 0; j < n; j++) {
        int iBegin = 0;
        int iEnd = -1;
        ComputeGoldenCsyrColumn(uplo, n, j, &alphaVal, xPtr, incx, aOrig.data(), lda, colGold.data(), &iBegin, &iEnd);
        for (int i = iBegin; i <= iEnd; i++) {
            const aclblasComplex& act = aActual[static_cast<size_t>(j) * lda + i];
            gateRe.Feed(act.real, colGold[static_cast<size_t>(i - iBegin)].real);
            gateIm.Feed(act.imag, colGold[static_cast<size_t>(i - iBegin)].imag);
        }
    }
    gateRe.Report(tag, "real");
    gateIm.Report(tag, "imag");
    return gateRe.Pass() && gateIm.Pass();
}

struct UtKernelRun {
    aclblasStatus_t status = ACLBLAS_STATUS_INTERNAL_ERROR;
    std::vector<aclblasComplex> aAfter; // host copy of A after the call
    std::vector<aclblasComplex> xHost;  // inputs retained for post-hoc golden checks
    std::vector<aclblasComplex> aHost;  // (pre-update A snapshot)
};

// Drives aclblasCsyr with device-resident x/A buffers (and a device alpha
// scalar), synchronizes, and copies A back - the real NPU kernel path.
// alphaOnHost=false (default) passes a DEVICE-resident alpha (the D2H read-back
// branch); alphaOnHost=true passes a HOST stack alpha (the plain host-read
// branch of the CheckPtrLocation dual contract -- the README example caliber).
void UtRunKernel(aclblasFillMode_t uplo, int n, float alphaRe, float alphaIm,
    const std::vector<aclblasComplex>& xHost, int incx, const std::vector<aclblasComplex>& aHost, int lda,
    UtKernelRun* out, bool alphaOnHost = false)
{
    const size_t xBytes = xHost.size() * sizeof(aclblasComplex);
    const size_t aBytes = aHost.size() * sizeof(aclblasComplex);

    // Device buffers are owned by CsyrDeviceBuf (RAII): a fatal ASSERT_EQ return
    // below still frees every buffer allocated so far.
    CsyrDeviceBuf dAlphaOwner(nullptr, aclrtFree);
    CsyrDeviceBuf dXOwner(nullptr, aclrtFree);
    CsyrDeviceBuf dAOwner(nullptr, aclrtFree);
    aclblasComplex alphaHostScalar;
    alphaHostScalar.real = alphaRe;
    alphaHostScalar.imag = alphaIm;
    if (!alphaOnHost) {
        void* dAlpha = nullptr;
        ASSERT_EQ(UtMakeDeviceAlpha(alphaRe, alphaIm, &dAlpha), ACLBLAS_STATUS_SUCCESS);
        dAlphaOwner.reset(dAlpha);
    }
    if (!xHost.empty()) {
        void* dX = nullptr;
        ASSERT_EQ(CsyrAllocCopyToDevice(&dX, xHost.data(), xBytes), ACLBLAS_STATUS_SUCCESS);
        dXOwner.reset(dX);
    }
    if (!aHost.empty()) {
        void* dA = nullptr;
        ASSERT_EQ(CsyrAllocCopyToDevice(&dA, aHost.data(), aBytes), ACLBLAS_STATUS_SUCCESS);
        dAOwner.reset(dA);
    }

    const aclblasComplex* alphaArg =
        alphaOnHost ? &alphaHostScalar : static_cast<const aclblasComplex*>(dAlphaOwner.get());
    out->status = aclblasCsyr(CsyrUt::UtHandle(), uplo, n, alphaArg,
        static_cast<const aclblasComplex*>(dXOwner.get()), incx, static_cast<aclblasComplex*>(dAOwner.get()), lda);

    if (out->status == ACLBLAS_STATUS_SUCCESS && dAOwner.get() != nullptr) {
        aclrtSynchronizeDevice();
        out->aAfter.resize(aHost.size());
        aclrtMemcpy(out->aAfter.data(), aBytes, dAOwner.get(), aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
}

// A kernel-path case: fill x/A, run on device, golden-compare the uplo triangle
// and snapshot-assert the untouched region (bitwise).
void UtRunKernelCase(const std::string& tag, aclblasFillMode_t uplo, int n, int incx, int lda)
{
    const size_t xLen = static_cast<size_t>(1 + (n - 1) * std::abs(incx));
    const size_t aLen = static_cast<size_t>(lda) * static_cast<size_t>(n);
    std::vector<aclblasComplex> xHost = MakeCsyrComplexArray(static_cast<int64_t>(xLen), "RANDOM_NORM_5_5", 20280001u);
    std::vector<aclblasComplex> aHost = MakeCsyrComplexArray(static_cast<int64_t>(aLen), "RANDOM_NORM_5_5", 20280002u);
    ASSERT_FALSE(xHost.empty());
    ASSERT_EQ(aHost.size(), aLen);

    UtKernelRun run;
    UtRunKernel(uplo, n, 1.0f, -0.5f, xHost, incx, aHost, lda, &run);
    ASSERT_EQ(run.status, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(run.aAfter.size(), aLen);

    aclblasComplex alphaVal;
    alphaVal.real = 1.0f;
    alphaVal.imag = -0.5f;
    EXPECT_TRUE(UtGoldenCompareTriangle(tag, uplo, n, alphaVal, xHost.data(), incx, aHost, run.aAfter, lda))
        << "[" << tag << "] golden triangle mismatch";

    std::vector<csyr_verify::BitMismatchInfo> reports;
    const size_t untouched = CountUntouchedBitMismatch(uplo, n, lda, aHost, run.aAfter, 3, &reports);
    for (const auto& r : reports) {
        std::cout << "[" << tag << "] untouched mismatch (col=" << r.col << ", row=" << r.row << ")" << std::endl;
    }
    EXPECT_EQ(untouched, size_t(0)) << "[" << tag << "] untouched region was written";
}

// Batch-partition driver: same input as
// UtRunKernelCase but retains the host buffers in the result and optionally
// forces a specific batchCols via the CSYR_BATCH_COLS host override (batchCols
// <= 0 runs the production-derived partition). The override is cleared before
// returning so it cannot leak into other cases.
void UtRunKernelBatchCols(aclblasFillMode_t uplo, int n, int incx, int lda, int batchCols, UtKernelRun* out)
{
    if (batchCols > 0) {
        setenv("CSYR_BATCH_COLS", std::to_string(batchCols).c_str(), 1);
    } else {
        unsetenv("CSYR_BATCH_COLS");
    }
    const size_t xLen = static_cast<size_t>(1 + (n - 1) * std::abs(incx));
    const size_t aLen = static_cast<size_t>(lda) * static_cast<size_t>(n);
    std::vector<aclblasComplex> xHost = MakeCsyrComplexArray(static_cast<int64_t>(xLen), "RANDOM_NORM_5_5", 20280011u);
    std::vector<aclblasComplex> aHost = MakeCsyrComplexArray(static_cast<int64_t>(aLen), "RANDOM_NORM_5_5", 20280012u);
    ASSERT_FALSE(xHost.empty());
    ASSERT_EQ(aHost.size(), aLen);

    UtRunKernel(uplo, n, 1.0f, -0.5f, xHost, incx, aHost, lda, out);
    unsetenv("CSYR_BATCH_COLS");
    out->xHost = std::move(xHost);
    out->aHost = std::move(aHost);
}

// Shared body of the event-slot balance sweeps: runs the
// default partition once, then the forced CSYR_BATCH_COLS sweep; every forced
// partition must agree bitwise with the default-partition run and pass the
// streamed golden + untouched gates. `tag` is the golden-report tag prefix
// (per-batchCols suffix appended).
void UtEventSlotBalanceSweep(const std::string& tag, aclblasFillMode_t uplo, int n, int lda)
{
    const aclblasComplex alphaVal{1.0f, -0.5f};

    UtKernelRun base;
    UtRunKernelBatchCols(uplo, n, 1, lda, 0, &base);
    ASSERT_EQ(base.status, ACLBLAS_STATUS_SUCCESS);

    const int sweep[] = {1, 2, 3, 5, 7};
    for (int bc : sweep) {
        UtKernelRun run;
        UtRunKernelBatchCols(uplo, n, 1, lda, bc, &run);
        ASSERT_EQ(run.status, ACLBLAS_STATUS_SUCCESS) << "batchCols=" << bc;
        EXPECT_EQ(CountWholeBitMismatch(base.aAfter, run.aAfter), size_t(0))
            << "[batchCols=" << bc << "] result differs from the default-partition run";
        EXPECT_TRUE(UtGoldenCompareTriangle(tag + std::to_string(bc), uplo, n, alphaVal,
            run.xHost.data(), 1, run.aHost, run.aAfter, lda))
            << "[batchCols=" << bc << "] golden triangle mismatch";
        EXPECT_EQ(CountUntouchedBitMismatch(uplo, n, lda, run.aHost, run.aAfter, 3, nullptr), size_t(0))
            << "[batchCols=" << bc << "] untouched region was written";
    }
}

} // namespace

// ── ① host validation chain: full branch matrix with numeric codes ──

TEST_F(CsyrUt, HandleNullptrReturnsErrorCode9)
{
    std::vector<aclblasComplex> xHost(8);
    std::vector<aclblasComplex> aHost(64);
    const aclblasComplex* alphaPtr = nullptr; // never dereferenced: handle is checked first
    const aclblasStatus_t ret = aclblasCsyr(
        nullptr, ACLBLAS_UPPER, 8, alphaPtr, xHost.data(), 1, aHost.data(), 8);
    EXPECT_EQ(static_cast<int>(ret), kErrHandleNullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsyrUt, InvalidUploReturnsErrorCode10)
{
    const aclblasFillMode_t badModes[] = {
        static_cast<aclblasFillMode_t>(0),   // 0 is not a fill mode
        static_cast<aclblasFillMode_t>(120), // just below ACLBLAS_UPPER(121)
        static_cast<aclblasFillMode_t>(123), // just above ACLBLAS_LOWER(122)
    };
    for (aclblasFillMode_t mode : badModes) {
        const aclblasStatus_t ret = aclblasCsyr(
            CsyrUt::UtHandle(), mode, 8, nullptr, nullptr, 1, nullptr, 8);
        EXPECT_EQ(static_cast<int>(ret), kErrInvalidEnum) << "uplo=" << static_cast<int>(mode);
        EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_ENUM) << "uplo=" << static_cast<int>(mode);
    }
}

TEST_F(CsyrUt, NegativeNReturnsErrorCode3)
{
    const aclblasStatus_t ret = aclblasCsyr(CsyrUt::UtHandle(), ACLBLAS_UPPER, -1, nullptr, nullptr, 1, nullptr, 1);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrUt, IncxZeroReturnsErrorCode3)
{
    const aclblasStatus_t ret = aclblasCsyr(CsyrUt::UtHandle(), ACLBLAS_UPPER, 8, nullptr, nullptr, 0, nullptr, 8);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrUt, LdaBelowMaxOneNReturnsErrorCode3)
{
    // lda < max(1, n): n=8 requires lda >= 8
    const aclblasStatus_t ret = aclblasCsyr(CsyrUt::UtHandle(), ACLBLAS_UPPER, 8, nullptr, nullptr, 1, nullptr, 7);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrUt, NOverMax4864ReturnsErrorCode3)
{
    // n > CSYR_MAX_N(4864): documented out-of-range behavior
    const aclblasStatus_t ret =
        aclblasCsyr(CsyrUt::UtHandle(), ACLBLAS_UPPER, 4865, nullptr, nullptr, 1, nullptr, 4865);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrUt, AlphaNullptrReturnsErrorCode3)
{
    // alpha == nullptr must be rejected before any alpha dereference/read-back
    // (ordering): a crash here means the null check is misordered.
    const aclblasStatus_t ret = aclblasCsyr(CsyrUt::UtHandle(), ACLBLAS_UPPER, 8, nullptr, nullptr, 1, nullptr, 8);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrUt, N0WithLda0ReturnsInvalidValueNotSuccess)
{
    // Ordering pin: the lda check (lda >= max(1, n) = 1) precedes the n==0
    // quick return, so n=0 with lda=0 must NOT succeed.
    const aclblasStatus_t ret = aclblasCsyr(CsyrUt::UtHandle(), ACLBLAS_UPPER, 0, nullptr, nullptr, 1, nullptr, 0);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrUt, N0WithAlphaNullptrReturnsInvalidValueNotSuccess)
{
    // Ordering pin: alpha==nullptr is checked in the scalar group before the
    // n==0 quick return, so n=0 with alpha=nullptr must NOT succeed.
    const aclblasStatus_t ret = aclblasCsyr(CsyrUt::UtHandle(), ACLBLAS_LOWER, 0, nullptr, nullptr, 1, nullptr, 1);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrUt, CheckOrderHandlePrecedesEnumAndScalars)
{
    // handle==nullptr wins over an invalid uplo and a bad scalar group.
    const aclblasStatus_t ret = aclblasCsyr(
        nullptr, static_cast<aclblasFillMode_t>(999), -1, nullptr, nullptr, 0, nullptr, 0);
    EXPECT_EQ(static_cast<int>(ret), kErrHandleNullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsyrUt, CheckOrderEnumPrecedesScalarGroup)
{
    // invalid uplo wins over n<0 / incx==0 / lda<1 / alpha==nullptr.
    const aclblasStatus_t ret = aclblasCsyr(
        CsyrUt::UtHandle(), static_cast<aclblasFillMode_t>(999), -1, nullptr, nullptr, 0, nullptr, 0);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidEnum);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_ENUM);
}

TEST_F(CsyrUt, XNullptrWithNonzeroAlphaReturnsErrorCode3)
{
    // n>0 && alpha != (0,0) with x==nullptr: the alpha D2H read-back happens
    // first (device alpha), then the x/A null check rejects.
    void* dAlpha = nullptr;
    ASSERT_EQ(UtMakeDeviceAlpha(1.0f, 2.0f, &dAlpha), ACLBLAS_STATUS_SUCCESS);
    const CsyrDeviceBuf dAlphaOwner(dAlpha, aclrtFree);
    const aclblasStatus_t ret = aclblasCsyr(
        CsyrUt::UtHandle(), ACLBLAS_UPPER, 8, static_cast<const aclblasComplex*>(dAlphaOwner.get()),
        nullptr, 1, nullptr, 8);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrUt, ANullptrWithNonzeroAlphaReturnsErrorCode3)
{
    void* dAlpha = nullptr;
    ASSERT_EQ(UtMakeDeviceAlpha(1.0f, 2.0f, &dAlpha), ACLBLAS_STATUS_SUCCESS);
    const CsyrDeviceBuf dAlphaOwner(dAlpha, aclrtFree);
    std::vector<aclblasComplex> xHost(8);
    const aclblasStatus_t ret = aclblasCsyr(CsyrUt::UtHandle(), ACLBLAS_UPPER, 8,
        static_cast<const aclblasComplex*>(dAlphaOwner.get()), xHost.data(), 1, nullptr, 8);
    EXPECT_EQ(static_cast<int>(ret), kErrInvalidValue);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// ── ② quick return ──

TEST_F(CsyrUt, QuickReturnN0DoesNotTouchA)
{
    // n==0: SUCCESS, A untouched (whole-storage bitwise), no kernel launched
    // (the host short-circuits before csyr_kernel_do - tiling/malloc/launch
    // are all skipped, so a launched kernel could not have updated A either).
    const size_t aLen = 1;
    std::vector<aclblasComplex> aHost = MakeCsyrComplexArray(1, "RANDOM_NORM_5_5", 20280003u);
    ASSERT_EQ(aHost.size(), aLen);

    UtKernelRun run;
    UtRunKernel(ACLBLAS_LOWER, 0, 1.0f, 1.0f, {}, 1, aHost, 1, &run);
    ASSERT_EQ(run.status, ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(CountWholeBitMismatch(aHost, run.aAfter), size_t(0))
        << "[QuickReturnN0] A was modified on the n==0 quick return";
}

TEST_F(CsyrUt, QuickReturnAlphaZeroDoesNotTouchA)
{
    // alpha==(0,0): the 8-byte D2H read-back short-circuits before any launch;
    // A must remain bitwise identical (both triangles and padding).
    const int n = 16;
    const int lda = 20; // padding columns included in the whole-storage check
    const size_t xLen = static_cast<size_t>(n);
    const size_t aLen = static_cast<size_t>(lda) * static_cast<size_t>(n);
    std::vector<aclblasComplex> xHost = MakeCsyrComplexArray(static_cast<int64_t>(xLen), "RANDOM_NORM_5_5", 20280004u);
    std::vector<aclblasComplex> aHost = MakeCsyrComplexArray(static_cast<int64_t>(aLen), "RANDOM_NORM_5_5", 20280005u);
    ASSERT_EQ(xHost.size(), xLen);
    ASSERT_EQ(aHost.size(), aLen);

    UtKernelRun run;
    UtRunKernel(ACLBLAS_UPPER, n, 0.0f, 0.0f, xHost, 1, aHost, lda, &run);
    ASSERT_EQ(run.status, ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(CountWholeBitMismatch(aHost, run.aAfter), size_t(0))
        << "[QuickReturnAlphaZero] A was modified on the alpha==(0,0) quick return";
}

// ── alpha dual-location contract (Host/Device pointer, CheckPtrLocation branch) ──

TEST_F(CsyrUt, AlphaHostDeviceDualPathUpperLowerN100)
{
    // Alpha dual-location contract: the same inputs driven through a HOST alpha pointer
    // (the README example caliber -- plain host-read branch) and a DEVICE alpha pointer
    // (the 8-byte D2H read-back branch) must both pass the streamed golden + untouched
    // gates and agree BITWISE -- the two read branches must converge on the same 8
    // alpha bytes before any quick-return / tiling decision. Both uplo dispatch bits
    // are exercised (UPPER with lda padding, LOWER exact).
    const aclblasFillMode_t uplos[] = {ACLBLAS_UPPER, ACLBLAS_LOWER};
    for (aclblasFillMode_t uplo : uplos) {
        const int n = 100;
        const int incx = 1;
        const int lda = (uplo == ACLBLAS_UPPER) ? 110 : 100;
        const size_t xLen = static_cast<size_t>(1 + (n - 1) * std::abs(incx));
        const size_t aLen = static_cast<size_t>(lda) * static_cast<size_t>(n);
        std::vector<aclblasComplex> xHost =
            MakeCsyrComplexArray(static_cast<int64_t>(xLen), "RANDOM_NORM_5_5", 20280021u);
        std::vector<aclblasComplex> aHost =
            MakeCsyrComplexArray(static_cast<int64_t>(aLen), "RANDOM_NORM_5_5", 20280022u);
        ASSERT_EQ(aHost.size(), aLen);

        UtKernelRun runHost;
        UtRunKernel(uplo, n, 1.0f, -0.5f, xHost, incx, aHost, lda, &runHost, true);
        ASSERT_EQ(runHost.status, ACLBLAS_STATUS_SUCCESS) << "host-alpha call failed";

        UtKernelRun runDevice;
        UtRunKernel(uplo, n, 1.0f, -0.5f, xHost, incx, aHost, lda, &runDevice, false);
        ASSERT_EQ(runDevice.status, ACLBLAS_STATUS_SUCCESS) << "device-alpha call failed";

        EXPECT_EQ(CountWholeBitMismatch(runHost.aAfter, runDevice.aAfter), size_t(0))
            << "[AlphaDualPath] host-alpha and device-alpha results differ";

        const aclblasComplex alphaVal{1.0f, -0.5f};
        EXPECT_TRUE(UtGoldenCompareTriangle("AlphaHostPath", uplo, n, alphaVal, xHost.data(), incx,
            aHost, runHost.aAfter, lda))
            << "[AlphaDualPath] host-alpha golden triangle mismatch";
        EXPECT_EQ(CountUntouchedBitMismatch(uplo, n, lda, aHost, runHost.aAfter, 3, nullptr), size_t(0))
            << "[AlphaDualPath] host-alpha untouched region was written";
    }
}

// ── ③ kernel incremental segment: incx != +1 device pack + base skeleton ──

TEST_F(CsyrUt, DevicePackIncx2UpperN100)
{
    UtRunKernelCase("DevicePackIncx2UpperN100", ACLBLAS_UPPER, 100, 2, 110);
}

TEST_F(CsyrUt, DevicePackIncxNeg2LowerN100)
{
    UtRunKernelCase("DevicePackIncxNeg2LowerN100", ACLBLAS_LOWER, 100, -2, 100);
}

TEST_F(CsyrUt, DevicePackIncx3UpperN65NonAlignTail)
{
    // n=65: nPad=128 tail block crossed while packing |incx|=3 strides
    UtRunKernelCase("DevicePackIncx3UpperN65NonAlignTail", ACLBLAS_UPPER, 65, 3, 65);
}

TEST_F(CsyrUt, DevicePackIncxNeg3LowerN63)
{
    // negative stride maps logical k -> physical (n-1-k)*3; n=63 < 64 tail block
    UtRunKernelCase("DevicePackIncxNeg3LowerN63", ACLBLAS_LOWER, 63, -3, 70);
}

TEST_F(CsyrUt, DevicePackIncxNeg1UpperN16)
{
    UtRunKernelCase("DevicePackIncxNeg1UpperN16", ACLBLAS_UPPER, 16, -1, 16);
}

TEST_F(CsyrUt, DevicePackIncxNeg1LowerN37)
{
    UtRunKernelCase("DevicePackIncxNeg1LowerN37", ACLBLAS_LOWER, 37, -1, 40);
}

TEST_F(CsyrUt, BaseSkeletonIncx1UpperN100)
{
    // base skeleton core path: incx=+1 MTE2 fast load, UPPER column split
    UtRunKernelCase("BaseSkeletonIncx1UpperN100", ACLBLAS_UPPER, 100, 1, 100);
}

TEST_F(CsyrUt, BaseSkeletonIncx1LowerN100)
{
    // base skeleton core path: LOWER split with the 8-column aligned diagonal mask
    UtRunKernelCase("BaseSkeletonIncx1LowerN100", ACLBLAS_LOWER, 100, 1, 105);
}

// ── ④ double-buffered batch path ──

TEST_F(CsyrUt, Sb21UpperInterleavedTailN67)
{
    // n=67 -> nPad=128 tail block (3 rows past n) on the UPPER interleaved
    // 3-op domain (s' build + w/q'/add chain, no Gather, SoA-free); nb=2
    // zigzag blocks (even+odd). Golden triangle + untouched-region bitwise.
    UtRunKernelCase("Sb21UpperInterleavedTailN67", ACLBLAS_UPPER, 67, 1, 75);
}

TEST_F(CsyrUt, Sb21LowerSoaTailN67)
{
    // n=67 -> nPad=128 tail block on the LOWER SoA domain (rowBegin 8-alignment
    // + headShift Gather base, tail rows beyond n must stay untouched).
    UtRunKernelCase("Sb21LowerSoaTailN67", ACLBLAS_LOWER, 67, 1, 80);
}

TEST_F(CsyrUt, Sb21EventSlotBalanceUpperBatchColsSweep)
{
    // Group-level event-slot balance (1:1 SET/WAIT pairing): the batch partition
    // (which group slot holds which column) must not change ANY result bit.
    // n=192 -> per-core calNum 5/4, default batchCols=61 (single batch). The
    // CSYR_BATCH_COLS sweep forces nb=1..5 partitions: batchCols<=3 makes a
    // group slot REUSED two batches later, which is exactly the window where a
    // misbalanced SET/WAIT pair would let compute(k+2) overwrite batch k's
    // in-flight atomic stores. Every partition must agree bitwise with
    // the default partition and with the streamed golden.
    UtEventSlotBalanceSweep("SweepUpperBC", ACLBLAS_UPPER, 192, 200);
}

TEST_F(CsyrUt, Sb21EventSlotBalanceLowerBatchColsSweep)
{
    // Same event-slot balance sweep on the LOWER SoA domain (rowBegin-aligned
    // compute + headShift Gather + per-column precise atomic store).
    UtEventSlotBalanceSweep("SweepLowerBC", ACLBLAS_LOWER, 192, 192);
}

TEST_F(CsyrUt, Sb21LowerDefaultBatchColsSlotReuseN1024)
{
    // Production 1024L configuration WITHOUT any env override: nPad=1024 ->
    // batchCols=10, per-core calNum=26/25 -> nb=3, so group 0 is naturally
    // REUSED at batch 2 and the triple completion barrier is active.
    // Cross-checked against forced partitions {1,2,3,5,7} for bitwise equality
    // plus golden + untouched gates.
    const aclblasFillMode_t uplo = ACLBLAS_LOWER;
    const int n = 1024;
    const int lda = 1024;
    const aclblasComplex alphaVal{1.0f, -0.5f};

    UtKernelRun base;
    UtRunKernelBatchCols(uplo, n, 1, lda, 0, &base);
    ASSERT_EQ(base.status, ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(UtGoldenCompareTriangle("SlotReuseN1024Default", uplo, n, alphaVal,
        base.xHost.data(), 1, base.aHost, base.aAfter, lda));
    EXPECT_EQ(CountUntouchedBitMismatch(uplo, n, lda, base.aHost, base.aAfter, 3, nullptr), size_t(0));

    const int sweep[] = {1, 2, 3, 5, 7};
    for (int bc : sweep) {
        UtKernelRun run;
        UtRunKernelBatchCols(uplo, n, 1, lda, bc, &run);
        ASSERT_EQ(run.status, ACLBLAS_STATUS_SUCCESS) << "batchCols=" << bc;
        EXPECT_EQ(CountWholeBitMismatch(base.aAfter, run.aAfter), size_t(0))
            << "[batchCols=" << bc << "] result differs from the default-partition run";
    }
}

TEST_F(CsyrUt, Sb21MaxN4864UpperLimitValue)
{
    // Upper-limit value n=4864 on the batchCols=1 path: nPad=4864 -> batchCols=1 ->
    // nb=122 single-column batches per core (max-depth batch loop, event slots
    // alternating on every batch). Streamed golden over the 4864-column UPPER
    // triangle + untouched-region bitwise snapshot (CSV twin: L2_018).
    UtRunKernelCase("Sb21MaxN4864UpperLimitValue", ACLBLAS_UPPER, 4864, 1, 4864);
}

TEST_F(CsyrUt, MaxNBoundaryN4865OverLimitInvalidValue)
{
    // n=4865 = CSYR_MAX_N+1 (documented over-limit behavior) with every
    // OTHER argument valid (device alpha != 0, real device x/A, lda=n, incx=1):
    // the scalar-group check must reject with INVALID_VALUE before the alpha
    // D2H read-back and any launch. All buffers are real device buffers so a
    // missing check degrades to a clean SUCCESS + failed expectation instead of
    // a crash.
    const int n = 4865;
    const int lda = 4865;
    const size_t xLen = static_cast<size_t>(n);
    const size_t aLen = static_cast<size_t>(lda) * static_cast<size_t>(n);
    std::vector<aclblasComplex> xHost = MakeCsyrComplexArray(static_cast<int64_t>(xLen), "RANDOM_NORM_5_5", 20280013u);
    std::vector<aclblasComplex> aHost = MakeCsyrComplexArray(static_cast<int64_t>(aLen), "RANDOM_NORM_5_5", 20280014u);
    ASSERT_EQ(xHost.size(), xLen);
    ASSERT_EQ(aHost.size(), aLen);

    UtKernelRun run;
    UtRunKernel(ACLBLAS_UPPER, n, 1.0f, -0.5f, xHost, 1, aHost, lda, &run);
    EXPECT_EQ(static_cast<int>(run.status), kErrInvalidValue);
    EXPECT_EQ(run.status, ACLBLAS_STATUS_INVALID_VALUE);
}

// ── ⑤ zigzag round-robin column split ──

// The kernel derives its per-core column count from the closed form
// csyr_cal_num_of (csyr_kernel.cpp) inside the by-value tiling. The host twin
// below is an algebraically-identical re-expression of the kernel formula (deliberately
// not token-for-token, to keep the twin an independent derivation); the invariants tested
// here (partition / spread<=1 / equality with the zigzag block loop) gate that formula for
// every dispatch-relevant n. The end-to-end desync gate stays ZigzagKernelEndToEndOddBlocksN81.
constexpr uint32_t CSYR_VEC_CORES = 40;

uint32_t CsyrCalNumOf(uint32_t blockIdx, uint32_t n, uint32_t coreNum)
{
    if (coreNum == 0u) {
        return 0u; // no cores -> no columns; also guards the division below
    }
    const uint32_t full = n / coreNum;
    const uint32_t rem = n - full * coreNum;
    if (rem == 0u) {
        return full;
    }
    // zigzag hot-core range: odd rounds put the remainder on the LAST rem cores,
    // even rounds on the FIRST rem cores (equivalent to the kernel closed form).
    const bool oddRound = (full & 1u) != 0u;
    const bool hot = oddRound ? (blockIdx >= coreNum - rem) : (blockIdx < rem);
    return full + (hot ? 1u : 0u);
}

TEST_F(CsyrUt, ZigzagColumnSplitBalanceAndBijection)
{
    // Structural invariants of the zigzag split for every dispatch-relevant n,
    // both uplo dispatch bits:
    //   - total column count == n (each column owned by exactly one core);
    //   - per-core column-count spread <= 1 (column count is a first-order
    //     critical resource under the per-column fixed stall);
    //   - the closed form (kernel twin csyr_cal_num_of) matches the zigzag
    //     block loop exactly.
    const uint32_t coreNum = CSYR_VEC_CORES;
    const uint32_t ns[] = {1, 39, 40, 41, 79, 80, 81, 100, 127, 128, 129, 1024, 2048, 4096, 4864};
    for (uint32_t n : ns) {
        for (uint32_t uploBit : {0u, 1u}) {
            (void)uploBit; // the zigzag map balances both uplo triangles

            uint32_t total = 0;
            uint32_t minCnt = ~0u;
            uint32_t maxCnt = 0;
            for (uint32_t c = 0; c < coreNum; c++) {
                const uint32_t cnt = CsyrCalNumOf(c, n, coreNum);
                total += cnt;
                minCnt = std::min(minCnt, cnt);
                maxCnt = std::max(maxCnt, cnt);
            }
            EXPECT_EQ(total, n) << "[n=" << n << " uplo=" << uploBit << "] column counts must partition [0, n)";
            EXPECT_LE(maxCnt - minCnt, 1u)
                << "[n=" << n << " uplo=" << uploBit << "] per-core column-count spread exceeds 1";

            const uint32_t nb = (n + coreNum - 1) / coreNum;
            for (uint32_t c = 0; c < coreNum; c++) {
                uint32_t formulaCnt = 0;
                for (uint32_t b = 0; b < nb; b++) {
                    const uint32_t j = b * coreNum + (((b & 1u) != 0u) ? (coreNum - 1u - c) : c);
                    if (j < n) {
                        formulaCnt++;
                    }
                }
                EXPECT_EQ(CsyrCalNumOf(c, n, coreNum), formulaCnt)
                    << "[n=" << n << " uplo=" << uploBit << " c=" << c << "] calNum deviates from the zigzag formula";
            }
        }
    }
}

TEST_F(CsyrUt, ZigzagKernelEndToEndOddBlocksN81)
{
    // n=81 -> nb=3 zigzag blocks (even/odd/even): core 0 owns columns 0, 79, 80
    // -- the odd-block REVERSED assignment is exercised on both compute domains.
    // A host/kernel zigzag desync would
    // double-write or skip columns and fail the golden/untouched gates.
    UtRunKernelCase("ZigzagKernelEndToEndOddBlocksN81U", ACLBLAS_UPPER, 81, 1, 88);
    UtRunKernelCase("ZigzagKernelEndToEndOddBlocksN81L", ACLBLAS_LOWER, 81, 1, 81);
}
