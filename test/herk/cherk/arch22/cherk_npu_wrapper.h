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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <thread>

#include "../cherk_npu_common.h"

// Repeat-window target for performance measurement. Sizing the window by a
// probe keeps short shapes from being dominated by launch jitter while long
// shapes still finish in one window.
constexpr double kCherkPerfWindowMs = 30.0;
constexpr int kCherkPerfMaxIters = 2000;
// The Cube clock drops under sustained load, so an idle gap precedes the timed
// window to keep back-to-back cases from measuring a throttled frequency.
constexpr int kCherkPerfCooldownMs = 300;

// Times a repeat window: one probe call sizes the window, then the average over
// `iters` back-to-back calls is reported. A cooldown precedes both so a throttled
// frequency from the previous case does not leak into this measurement.
template <typename RunFn, typename SyncFn, typename ElapsedFn>
static aclblasStatus_t CherkMeasurePerfWindow(
    RunFn&& runOnce, SyncFn&& syncDevice, ElapsedFn&& elapsedMs, double& measuredMs, int& iters)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(kCherkPerfCooldownMs));
    const auto probeStart = std::chrono::steady_clock::now();
    aclblasStatus_t ret = runOnce();
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (!syncDevice()) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const double probeMs = elapsedMs(probeStart);
    iters =
        (probeMs > 0.0) ? std::max(1, std::min(static_cast<int>(kCherkPerfWindowMs / probeMs), kCherkPerfMaxIters)) : 1;

    std::this_thread::sleep_for(std::chrono::milliseconds(kCherkPerfCooldownMs));
    const auto windowStart = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        ret = runOnce();
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
    }
    if (!syncDevice()) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // std::max above already floors iters at 1; the floor is restated here because
    // the static analyser cannot see through max/min and flags the division.
    if (iters < 1) {
        iters = 1;
    }
    // std::max above already floors iters at 1; the floor is restated here because
    // the static analyser cannot see through max/min and flags the division.
    if (iters < 1) {
        iters = 1;
    }
    measuredMs = elapsedMs(windowStart) / iters;
    return ACLBLAS_STATUS_SUCCESS;
}

// One warm-up call (always) plus, in perfMode, the averaged repeat window.
// measuredMs ends up holding whichever of the two the caller asked for.
template <typename RunFn>
static aclblasStatus_t CherkRunAndTime(RunFn&& runOnce, bool perfMode, double& measuredMs, int& iters)
{
    const auto syncDevice = []() { return aclrtSynchronizeDevice() == ACL_SUCCESS; };
    const auto elapsedMs = [](std::chrono::steady_clock::time_point since) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
    };

    const auto opStart = std::chrono::steady_clock::now();
    const aclblasStatus_t ret = runOnce();
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (!syncDevice()) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    measuredMs = elapsedMs(opStart);
    iters = 1;

    if (perfMode) {
        return CherkMeasurePerfWindow(runOnce, syncDevice, elapsedMs, measuredMs, iters);
    }
    return ret;
}

// NPU wrapper for aclblasCherk. Mirrors aclblasSsyrk_npu with complex-specific
// element sizes (aclblasComplex = 8 bytes). When handle is null or n<=0 it
// forwards directly so the operator's own validation paths are exercised.
//
// nullAlpha handling (§8 遗留问题 3): when nullAlpha=true the wrapper passes
// nullptr for alpha (no device alloc/copy), exercising the API's INVALID_VALUE
// path. Same pattern applies to nullA / nullC.
//
// opMs, when non-null, receives the wall time of the operator call plus device
// synchronisation. Host buffer allocation and H2D/D2H transfers are excluded,
// so it is the cost a caller pays for the operator itself on already-resident
// device data — the quantity the performance baseline is stated against.
//
// perfMode selects how opMs is obtained:
//   false — a single call, which is what an accuracy case wants because C must
//           hold exactly one operator result for the golden comparison.
//   true  — one warm-up call followed by a repeat window whose average is
//           reported. The warm-up matters: the first call on a shape also pays
//           workspace growth and kernel load, which on n=1024 is several times
//           the steady-state cost. C ends up holding the result of the last
//           iteration, so a perfMode call must not be precision-checked.
inline aclblasStatus_t aclblasCherk_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const aclblasComplex* A, int lda, const float* beta, aclblasComplex* C, int ldc, bool nullAlpha = false,
    double* opMs = nullptr, bool perfMode = false, int* perfIters = nullptr)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCherk(handle, uplo, trans, n, k, (nullAlpha ? nullptr : alpha), A, lda, beta, C, ldc);
    }

    const CherkInputLayout inputs = CherkDescribeInputs(trans, n, k, lda, ldc, alpha, nullAlpha);
    CherkDeviceBuffers bufs;
    const aclblasStatus_t upRet = CherkUploadInputs(inputs, A, beta, C, bufs);
    if (upRet != ACLBLAS_STATUS_SUCCESS) {
        return upRet;
    }

    const auto runOnce = [&]() {
        return aclblasCherk(
            handle, uplo, trans, n, k, bufs.dAlpha ? static_cast<const float*>(bufs.dAlpha) : inputs.alpha,
            bufs.dA ? static_cast<const aclblasComplex*>(bufs.dA) : A, lda,
            bufs.dBeta ? static_cast<const float*>(bufs.dBeta) : beta,
            bufs.dC ? static_cast<aclblasComplex*>(bufs.dC) : C, ldc);
    };
    double measuredMs = 0.0;
    int iters = 1;
    const aclblasStatus_t ret = CherkRunAndTime(runOnce, perfMode, measuredMs, iters);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    if (opMs != nullptr) {
        *opMs = measuredMs;
    }
    if (perfIters != nullptr) {
        *perfIters = iters;
    }

    const aclblasStatus_t downRet = CherkDownloadResult(C, bufs.dC, inputs.cBytes);
    if (downRet != ACLBLAS_STATUS_SUCCESS) {
        return downRet;
    }
    return ret;
}
