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

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cgemm3m_param.h"
#include "../cgemm/arch35/cgemm_npu_wrapper.h"

inline int Cgemm3mProfileCount(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed >= 0 && parsed <= 10000 ? static_cast<int>(parsed) : -1;
}

// Device durations come exclusively from msprof. Host time and CPU golden are not sampled.
inline aclblasStatus_t ProfileCgemm3m(
    aclblasHandle_t handle, aclrtStream stream, const Cgemm3mParam& p, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& b, std::vector<aclblasComplex>& c)
{
    const int warmup = Cgemm3mProfileCount("CGEMM3M_WARMUP", 5);
    const int samples = Cgemm3mProfileCount("CGEMM3M_SAMPLES", 10);
    // Reusing C is valid here because every supplied performance case has beta=0.
    if (warmup < 0 || samples <= 0 || p.caseName.find("TC_PF_") != 0 || p.expectResult != ACLBLAS_STATUS_SUCCESS ||
        p.alphaReal != 1.0f || p.alphaImag != 0.0f || p.betaReal != 0.0f || p.betaImag != 0.0f || p.m <= 0 ||
        p.n <= 0 || p.k <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CgemmDeviceBuffers buffers;
    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    const aclblasComplex beta{p.betaReal, p.betaImag};
    aclblasStatus_t status = AllocateCgemmDeviceBuffers(
        buffers, p.k, a.data(), b.data(), c.data(), a.size() * sizeof(alpha), b.size() * sizeof(alpha),
        c.size() * sizeof(alpha));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    for (int iteration = 0; iteration < warmup + samples; ++iteration) {
        status = aclblasCgemm3m(
            handle, p.transA, p.transB, p.m, p.n, p.k, &alpha, static_cast<const aclblasComplex*>(buffers.a), p.lda,
            static_cast<const aclblasComplex*>(buffers.b), p.ldb, &beta, static_cast<aclblasComplex*>(buffers.c),
            p.ldc);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            break;
        }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            status = ACLBLAS_STATUS_INTERNAL_ERROR;
            break;
        }
    }
    FreeCgemmDeviceBuffers(buffers);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        std::cout << "CGEMM3M_PROFILE," << p.caseName << ',' << warmup << ',' << samples << std::endl;
    }
    return status;
}
