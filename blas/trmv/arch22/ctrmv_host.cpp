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
 * \file ctrmv_host.cpp
 * \brief Host side implementation for ctrmv operator
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void ctrmv_kernel_do(
    uint8_t* gm_A, uint8_t* gm_X, uint8_t* gm_uplo, uint8_t* gm_wksp, uint8_t* tilingGm, uint32_t numBlocks,
    void* stream);

constexpr int64_t BASIC_DATA_PROC_CNT = 64;
constexpr uint32_t ELEMENTS_EACH_COMPLEX64 = 2;
constexpr int64_t N0 = 64;

// Tiling data structure - consistent with sip CtrmvTilingData
struct CtrmvTilingData {
    int64_t mode;
    int64_t trans;
    int64_t diag;
    int64_t n;
    int64_t lda;
    int64_t incx;
    int64_t n0;
};

// Tiling calculation - migrated from sip ctrmv_tiling.cpp
CtrmvTilingData CalCtrmvTilingData(int64_t mode, int64_t trans, int64_t diag, int64_t n, int64_t lda, int64_t incx)
{
    CtrmvTilingData tilingData;
    memset(&tilingData, 0, sizeof(CtrmvTilingData));

    tilingData.mode = mode;
    tilingData.trans = trans;
    tilingData.diag = diag;
    tilingData.n = n;
    tilingData.lda = lda;
    tilingData.incx = incx;
    tilingData.n0 = BASIC_DATA_PROC_CNT;

    return tilingData;
}

// Plan: Create uplo mask matrix - migrated from sip BlasCtrmvPlan.cpp
float* CreateCtrmvUploMatrix(int64_t uplo)
{
    int64_t blockSize = N0 * N0;
    // issue #712: 默认 new[] 失败抛出的 std::bad_alloc 会穿越 extern "C" ABI 边界；
    // 改为 nothrow + 判空，由调用方映射为 ACLBLAS_STATUS_ALLOC_FAILED
    float* uploMatrixData = new (std::nothrow) float[blockSize];
    if (uploMatrixData == nullptr) {
        return nullptr;
    }

    float ele = (uplo == 0) ? 0 : 1; // 0 = LOWER, 1 = UPPER

    for (int64_t i = 0; i < N0; ++i) {
        for (int64_t j = 0; j < N0; ++j) {
            if (j < i) {
                *(uploMatrixData + i * N0 + j) = ele;
            } else if (j == i) {
                *(uploMatrixData + i * N0 + j) = 1;
            } else {
                *(uploMatrixData + i * N0 + j) = 1 - ele;
            }
        }
    }

    return uploMatrixData;
}

// Calculate block dim - migrated from sip ctrmv_tiling.cpp
uint32_t CalCtrmvBlockDim(int64_t n, uint32_t coreNum)
{
    int64_t nDupNum = (n - 1) / BASIC_DATA_PROC_CNT + 1;
    int64_t groupDim = nDupNum * nDupNum;

    groupDim = groupDim < coreNum ? groupDim : coreNum;
    if (groupDim == 0) {
        groupDim = 1;
    }
    return static_cast<uint32_t>(groupDim);
}

// issue #710/#779: 入口校验独立成函数，控制 aclblasCtrmv 圈复杂度；
// handle 为空时解引用 h->stream 主机段错误，非法 trans 以 -1 进内核，
// n/lda/指针非法值静默下发设备——对齐 #580 修复后的 strmv 入口校验口径。
aclblasStatus_t ValidateCtrmvArgs(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int lda,
    int incx, const aclblasComplex* A, const aclblasComplex* x)
{
    if (handle == nullptr) {
        LOG_PRINT("aclblasCtrmv: handle is nullptr\n");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        LOG_PRINT("aclblasCtrmv: invalid uplo=%d\n", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        LOG_PRINT("aclblasCtrmv: invalid trans=%d\n", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) {
        LOG_PRINT("aclblasCtrmv: invalid diag=%d\n", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        LOG_PRINT("aclblasCtrmv: invalid n=%d\n", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (lda < std::max(1, n)) {
        LOG_PRINT("aclblasCtrmv: invalid lda=%d, n=%d\n", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        LOG_PRINT("aclblasCtrmv: incx must not be zero\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr || x == nullptr) {
        LOG_PRINT("aclblasCtrmv: A and x must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCtrmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
{
    const aclblasStatus_t entry = ValidateCtrmvArgs(handle, uplo, trans, diag, n, lda, incx, A, x);
    if (entry != ACLBLAS_STATUS_SUCCESS) {
        return entry;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    int64_t uploLocal = (uplo == ACLBLAS_LOWER) ? 0 : 1;
    int64_t transLocal = -1;
    if (trans == ACLBLAS_OP_N) {
        transLocal = 0;
    } else if (trans == ACLBLAS_OP_T) {
        transLocal = 1;
    } else if (trans == ACLBLAS_OP_C) {
        transLocal = 2;
    }
    int64_t diagLocal = (diag == ACLBLAS_NON_UNIT) ? 0 : 1;

    CtrmvTilingData tiling = CalCtrmvTilingData(uploLocal, transLocal, diagLocal, n, lda, incx);

    // issue #716: 核数不得硬编码，不同芯片型号/形态的 AI Core 数不同
    uint32_t coreNum = GetAivCoreCount();
    uint32_t numBlocks = CalCtrmvBlockDim(n, coreNum);

    float* uploMatrixData = CreateCtrmvUploMatrix(uploLocal);
    if (uploMatrixData == nullptr) {
        LOG_PRINT("aclblasCtrmv: alloc uplo matrix failed\n");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    size_t uploByteSize = N0 * N0 * sizeof(float);
    int64_t mTiles = (n + BASIC_DATA_PROC_CNT - 1) / BASIC_DATA_PROC_CNT;
    size_t workspaceSize = mTiles * BASIC_DATA_PROC_CNT * ELEMENTS_EACH_COMPLEX64 * sizeof(float);
    if (workspaceSize < 1024) {
        workspaceSize = 1024;
    }

    uint8_t* uploHost = reinterpret_cast<uint8_t*>(uploMatrixData);

    uint8_t* uploDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    // issue #779: 失败路径用显式 if 块（G.FMT.04 每行仅一条语句，且 clang-format
    // 不会将块内独立语句回并为一行）
    aclError aclRet = aclrtMalloc((void**)&uploDevice, uploByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        delete[] uploMatrixData;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        aclrtFree(uploDevice);
        delete[] uploMatrixData;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    size_t tilingSize = (sizeof(CtrmvTilingData) + 31) / 32 * 32;
    aclRet = aclrtMalloc((void**)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        aclrtFree(workspaceDevice);
        aclrtFree(uploDevice);
        delete[] uploMatrixData;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    std::vector<uint8_t> workspaceHost(workspaceSize, 0);
    aclRet =
        aclrtMemcpy(workspaceDevice, workspaceSize, workspaceHost.data(), workspaceSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet);
        aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice);
        aclrtFree(uploDevice);
        delete[] uploMatrixData;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(uploDevice, uploByteSize, uploHost, uploByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet);
        aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice);
        aclrtFree(uploDevice);
        delete[] uploMatrixData;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CtrmvTilingData), &tiling, sizeof(CtrmvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet);
        aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice);
        aclrtFree(uploDevice);
        delete[] uploMatrixData;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    ctrmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), reinterpret_cast<uint8_t*>(x), uploDevice,
        workspaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
        aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice);
        aclrtFree(uploDevice);
        delete[] uploMatrixData;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(uploDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    delete[] uploMatrixData;

    return ACLBLAS_STATUS_SUCCESS;
}
