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

#include <cstddef>

#include "../cherk_npu_common.h"

struct CherkEventPair {
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;

    ~CherkEventPair()
    {
        if (start)
            aclrtDestroyEvent(start);
        if (end)
            aclrtDestroyEvent(end);
    }
};

static aclblasStatus_t CherkWaitForPriorWork(aclrtStream stream)
{
    aclrtEvent event = nullptr;
    if (aclrtCreateEvent(&event) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const aclError recordRet = aclrtRecordEvent(event, stream);
    const aclError waitRet = recordRet == ACL_SUCCESS ? aclrtSynchronizeEvent(event) : recordRet;
    const aclError destroyRet = aclrtDestroyEvent(event);
    return waitRet == ACL_SUCCESS && destroyRet == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS :
                                                                 ACLBLAS_STATUS_EXECUTION_FAILED;
}

// NPU wrapper for aclblasCherk. Mirrors aclblasSsyrk_npu with complex-specific
// element sizes (aclblasComplex = 8 bytes). When handle is null or n<=0 it
// forwards directly so the operator's own validation paths are exercised.
//
// nullAlpha handling (§8 遗留问题 3): when nullAlpha=true the wrapper passes
// nullptr for alpha (no device alloc/copy), exercising the API's INVALID_VALUE
// path. Same pattern applies to nullA / nullC.
inline aclblasStatus_t aclblasCherk_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const aclblasComplex* A, int lda, const float* beta, aclblasComplex* C, int ldc, bool nullAlpha = false)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCherk(handle, uplo, trans, n, k, (nullAlpha ? nullptr : alpha), A, lda, beta, C, ldc);
    }

    const CherkInputLayout inputs = CherkDescribeInputs(trans, n, k, lda, ldc, alpha, nullAlpha);
    CherkDeviceBuffers bufs;
    const aclblasStatus_t st = CherkUploadInputs(inputs, A, beta, C, bufs);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    aclblasStatus_t ret = aclblasCherk(
        handle, uplo, trans, n, k, bufs.dAlpha ? static_cast<const float*>(bufs.dAlpha) : inputs.alpha,
        bufs.dA ? static_cast<const aclblasComplex*>(bufs.dA) : A, lda,
        bufs.dBeta ? static_cast<const float*>(bufs.dBeta) : beta, bufs.dC ? static_cast<aclblasComplex*>(bufs.dC) : C,
        ldc);

    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    aclrtStream stream = nullptr;
    ret = aclblasGetStream(handle, &stream);
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return ret;
    ret = CherkWaitForPriorWork(stream);
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return ret;

    return CherkDownloadResult(C, bufs.dC, inputs.cBytes);
}

// Allocation and host/device copies stay outside the measured interval. The
// measured batch covers the device tasks between two stream-scoped events.
// CPU golden, D2H, result preparation, and verification are not timed.
inline bool IsValidCherkBenchmarkArgs(
    aclblasHandle handle, aclrtStream stream, float* averageUs, int n, int k, const float* alpha,
    const aclblasComplex* A, const float* beta, aclblasComplex* C, int warmup, int iterations, aclblasFillMode_t uplo,
    aclblasOperation_t trans)
{
    return handle != nullptr && stream != nullptr && averageUs != nullptr && warmup >= 0 && iterations > 0 && n > 0 &&
           k >= 0 && alpha != nullptr && beta != nullptr && A != nullptr && C != nullptr &&
           (uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER) && (trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_C);
}

template <typename LaunchFn>
inline aclblasStatus_t RunCherkBenchmarkIterations(LaunchFn&& launch, int count)
{
    for (int i = 0; i < count; ++i) {
        const aclblasStatus_t status = launch();
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t PrepareCherkBenchmarkBuffers(
    CherkDeviceBuffers& bufs, const float* alpha, const aclblasComplex* A, size_t aBytes, const float* beta,
    aclblasComplex* C, size_t cBytes)
{
    aclblasStatus_t status = CherkAllocAndCopy(bufs.dAlpha, alpha, sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = CherkAllocAndCopy(bufs.dA, A, aBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = CherkAllocAndCopy(bufs.dBeta, beta, sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    return CherkAllocAndCopy(bufs.dC, C, cBytes);
}

template <typename LaunchFn>
inline aclblasStatus_t MeasureCherkBenchmark(LaunchFn&& launch, aclrtStream stream, int iterations, float* averageUs)
{
    CherkEventPair events;
    if (aclrtCreateEvent(&events.start) != ACL_SUCCESS || aclrtCreateEvent(&events.end) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtRecordEvent(events.start, stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    aclblasStatus_t status = RunCherkBenchmarkIterations(launch, iterations);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if (aclrtRecordEvent(events.end, stream) != ACL_SUCCESS || aclrtSynchronizeEvent(events.end) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    float elapsedMs = 0.0f;
    if (aclrtEventElapsedTime(&elapsedMs, events.start, events.end) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    *averageUs = elapsedMs * 1000.0f / static_cast<float>(iterations);
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CherkRunBenchmark(
    aclblasHandle handle, aclrtStream stream, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const aclblasComplex* A, int lda, const float* beta, aclblasComplex* C, int ldc, int warmup,
    int iterations, float* averageUs)
{
    if (!IsValidCherkBenchmarkArgs(
            handle, stream, averageUs, n, k, alpha, A, beta, C, warmup, iterations, uplo, trans)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int aCols = trans == ACLBLAS_OP_N ? k : n;
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    CherkDeviceBuffers bufs;
    aclblasStatus_t status = PrepareCherkBenchmarkBuffers(bufs, alpha, A, aBytes, beta, C, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    const auto launch = [&]() {
        return aclblasCherk(
            handle, uplo, trans, n, k, static_cast<const float*>(bufs.dAlpha),
            static_cast<const aclblasComplex*>(bufs.dA), lda, static_cast<const float*>(bufs.dBeta),
            static_cast<aclblasComplex*>(bufs.dC), ldc);
    };
    status = RunCherkBenchmarkIterations(launch, warmup);
    return status == ACLBLAS_STATUS_SUCCESS ? MeasureCherkBenchmark(launch, stream, iterations, averageUs) : status;
}

inline aclblasStatus_t aclblasCherk_npu_benchmark(
    aclblasHandle handle, aclrtStream stream, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const aclblasComplex* A, int lda, const float* beta, aclblasComplex* C, int ldc, int warmup,
    int iterations, float* averageUs)
{
    return CherkRunBenchmark(
        handle, stream, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, warmup, iterations, averageUs);
}
