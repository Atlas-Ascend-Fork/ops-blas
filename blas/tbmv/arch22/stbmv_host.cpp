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
#include "common/helper/aclblas_handle_internal.h"
#include "stbmv_tiling_data.h"
#include "stbmv_kernel.h"

static TbmvTilingData CalTbmvTilingData(
    uint32_t totalRows, uint32_t totalDias, uint32_t lda, uint32_t vecCoreNum, int64_t incx, uint32_t uplo,
    uint32_t trans, uint32_t diag)
{
    TbmvTilingData tilingData{};
    tilingData.n = totalRows;
    tilingData.k = totalDias;
    tilingData.lda = lda;
    tilingData.incx = incx;
    tilingData.uplo = uplo;
    tilingData.trans = trans;
    tilingData.diag = diag;

    uint32_t taskCount = totalDias + 1U;
    if (incx == 1) {
        uint32_t availableCoreNum = vecCoreNum;
        if (availableCoreNum == 0) {
            availableCoreNum = 1;
        }
        if (availableCoreNum > TBMV_MAX_CORE_NUM) {
            availableCoreNum = TBMV_MAX_CORE_NUM;
        }
        tilingData.useCoreNum = std::min(taskCount, availableCoreNum);
    } else {
        tilingData.useCoreNum = 1;
    }
    if (tilingData.useCoreNum == 0) {
        tilingData.useCoreNum = 1;
    }
    return tilingData;
}

aclblasStatus_t aclblasStbmv_legacy(
    aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, const float* a,
    const int64_t lda, const float* x, float* y, const int64_t n, const int64_t k, const int64_t incx)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // k/lda 与 n/incx 同为 host 侧契约入参：k<0 或超出 uint32_t 表示域时经
    // static_cast<uint32_t>(k) 回绕（如 0xFFFFFFFF），内核以该值驱动带状循环
    // 产生越界访问；lda<k+1 时 A 按列主序带状布局的行跨度不足以覆盖整个带。
    // arch35 ValidateStbmvParams 与 stbsv/ctbsv arch22 均已校验，此处补齐同口径。
    if (k < 0 || k > static_cast<int64_t>(UINT32_MAX) - 1) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < k + 1) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    uint32_t uploVal = static_cast<uint32_t>(uplo);
    uint32_t transVal = static_cast<uint32_t>(trans);
    uint32_t diagVal = static_cast<uint32_t>(diag);

    if (a == nullptr || x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;

    constexpr uint32_t numBlocks = 8;
    TbmvTilingData tiling = CalTbmvTilingData(
        static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(lda), numBlocks, incx, uploVal,
        transVal, diagVal);

    stbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(a)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<float*>(y)), nullptr, tiling, numBlocks, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
