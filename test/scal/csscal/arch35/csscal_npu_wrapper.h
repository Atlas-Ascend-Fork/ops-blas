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
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

struct CsscalPerfResources {
    void* x = nullptr;
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    ~CsscalPerfResources()
    {
        if (start) aclrtDestroyEvent(start);
        if (end) aclrtDestroyEvent(end);
        if (x) aclrtFree(x);
    }
};

inline aclblasStatus_t RunCsscalPerfSamples(aclblasHandle_t handle, int n,
    const float* alpha, const aclblasComplex* x, int incx, int samples, int warmup,
    aclrtStream stream, CsscalPerfResources& r, size_t bytes,
    std::vector<float>& times, double& hostUs)
{
    for (int i = -warmup; i < samples; ++i) {
        if (aclrtMemcpy(r.x, bytes, x, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        if (i >= 0 && aclrtRecordEvent(r.start, stream) != ACL_SUCCESS) return ACLBLAS_STATUS_INTERNAL_ERROR;
        const auto begin = std::chrono::steady_clock::now();
        auto status = aclblasCsscal(handle, n, alpha, static_cast<aclblasComplex*>(r.x), incx);
        const auto end = std::chrono::steady_clock::now();
        if (status != ACLBLAS_STATUS_SUCCESS) return status;
        if (i < 0) {
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) return ACLBLAS_STATUS_EXECUTION_FAILED;
        } else {
            if (aclrtRecordEvent(r.end, stream) != ACL_SUCCESS || aclrtSynchronizeEvent(r.end) != ACL_SUCCESS)
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            float ms = 0;
            if (aclrtEventElapsedTime(&ms, r.start, r.end) != ACL_SUCCESS) return ACLBLAS_STATUS_INTERNAL_ERROR;
            times.push_back(ms);
            hostUs += std::chrono::duration<double, std::micro>(end - begin).count();
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// samples=60 for performance cases, 1 for accuracy-only cases. Every sample
// starts from the original input; copies and warmups are outside the events.
inline aclblasStatus_t aclblasCsscal_npu(aclblasHandle_t handle, int n,
    const float* alpha, aclblasComplex* x, int incx, int samples = 1)
{
    if (handle == nullptr || n <= 0 || incx <= 0 || x == nullptr || alpha == nullptr)
        return aclblasCsscal(handle, n, alpha, x, incx);
    if (samples < 1) return ACLBLAS_STATUS_INVALID_VALUE;
    const size_t bytes = ((static_cast<size_t>(n) - 1) * static_cast<size_t>(incx) + 1) * sizeof(*x);
    CsscalPerfResources r;
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS) return ACLBLAS_STATUS_INTERNAL_ERROR;
    if (aclrtMalloc(&r.x, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
    if (aclrtCreateEvent(&r.start) != ACL_SUCCESS || aclrtCreateEvent(&r.end) != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    const char* warmupEnv = std::getenv("CSSCAL_PERF_WARMUP");
    const int warmup = samples > 1 ? std::max(1, warmupEnv ? std::atoi(warmupEnv) : 5) : 1;
    std::vector<float> times;
    double hostUs = 0;
    auto status = RunCsscalPerfSamples(handle, n, alpha, x, incx, samples, warmup, stream, r, bytes, times, hostUs);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    double total = 0;
    for (float ms : times) total += ms;
    auto limits = std::minmax_element(times.begin(), times.end());
    // ACL events include stream dispatch/launch gaps. Device-only Task Duration
    // must be obtained from msprof and reported separately.
    printf("[EVENT_TIME] %.8f ms [SAMPLES] %d [WARMUP] %d [MIN_MS] %.8f [MAX_MS] %.8f [HOST_TIME] %.2f us\n",
           total / samples, samples, warmup, *limits.first, *limits.second, hostUs / samples);
    if (aclrtMemcpy(x, bytes, r.x, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    return ACLBLAS_STATUS_SUCCESS;
}
