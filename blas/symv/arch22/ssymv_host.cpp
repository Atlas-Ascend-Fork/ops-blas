/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <algorithm>
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/host_utils.h"
#include "common/helper/aclblas_handle_internal.h"
#include "ssymv_tiling_data.h"
#include "ssymv_kernel.h"

static SymvTilingData CalSymvTilingData(
    uint32_t totalRows, uint32_t lda, uint32_t vecCoreNum, float alpha, float beta, int64_t incx, int64_t incy,
    uint32_t uplo, uint32_t phase)
{
    SymvTilingData tilingData{};
    tilingData.n = totalRows;
    tilingData.lda = lda;
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = incx;
    tilingData.incy = incy;
    tilingData.uplo = uplo;
    tilingData.phase = phase;

    if (incx == 1 && incy == 1) {
        uint32_t availableCoreNum = vecCoreNum == 0 ? 1U : vecCoreNum;
        if (availableCoreNum > SYMV_MAX_CORE_NUM) {
            availableCoreNum = SYMV_MAX_CORE_NUM;
        }
        tilingData.useCoreNum = std::min(totalRows, availableCoreNum);
    } else {
        tilingData.useCoreNum = 1;
    }
    if (tilingData.useCoreNum == 0) {
        tilingData.useCoreNum = 1;
    }
    return tilingData;
}

aclblasStatus_t aclblasSsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* a, int lda, const float* x,
    int incx, const float* beta, float* y, int incy)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr || a == nullptr || x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    constexpr uint32_t numBlocks = 8;
    float alphaHost = 0.0f;
    float betaHost = 0.0f;
    aclblasStatus_t readSt = ReadScalarFromHostOrDevice(alpha, useStream, alphaHost);
    if (readSt != ACLBLAS_STATUS_SUCCESS) {
        return readSt;
    }
    readSt = ReadScalarFromHostOrDevice(beta, useStream, betaHost);
    if (readSt != ACLBLAS_STATUS_SUCCESS) {
        return readSt;
    }
    const uint8_t* aPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(a));
    const uint8_t* xPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(x));
    uint8_t* yPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(y));

    // y = alpha*A*x + beta*y 拆两次启动（同 stream 顺序执行，天然互斥）：
    // phase 0 单核覆盖写 y = beta*y（beta=0 时即清零，phase 1 必须先于此执行）；
    // phase 1 各核原子累加 alpha*A*x。
    // 消除单启动内 core0 读改写 y 与其余核原子加 y 的跨核竞态
    // （冒烟 n=255 UPPER 实测 diff=(beta-1)*row3 标量贡献）。
    SymvTilingData tilingBeta = CalSymvTilingData(
        static_cast<uint32_t>(n), static_cast<uint32_t>(lda), 1U, 1.0f, betaHost, incx, incy,
        static_cast<uint32_t>(uplo), 0U);
    ssymv_kernel_do(const_cast<uint8_t*>(aPtr), const_cast<uint8_t*>(xPtr), yPtr, nullptr, tilingBeta, 1U, useStream);

    SymvTilingData tilingAxpy = CalSymvTilingData(
        static_cast<uint32_t>(n), static_cast<uint32_t>(lda), numBlocks, alphaHost, 1.0f, incx, incy,
        static_cast<uint32_t>(uplo), 1U);
    ssymv_kernel_do(
        const_cast<uint8_t*>(aPtr), const_cast<uint8_t*>(xPtr), yPtr, nullptr, tilingAxpy, numBlocks, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
