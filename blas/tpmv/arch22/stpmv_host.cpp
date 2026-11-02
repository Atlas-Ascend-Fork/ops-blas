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
#include "stpmv_tiling_data.h"
#include "stpmv_kernel.h"

static TpmvTilingData CalTilingData(
    uint32_t totalRows, uint32_t vecCoreNum, int64_t incx, uint32_t uplo, uint32_t trans, uint32_t diag)
{
    TpmvTilingData tilingData{};
    tilingData.n = totalRows;
    tilingData.incx = incx;
    tilingData.uplo = uplo;
    tilingData.trans = trans;
    tilingData.diag = diag;

    if (incx == 1) {
        uint32_t availableCoreNum = vecCoreNum;
        if (availableCoreNum == 0) {
            availableCoreNum = 1;
        }
        if (availableCoreNum > TPMV_MAX_CORE_NUM) {
            availableCoreNum = TPMV_MAX_CORE_NUM;
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

aclblasStatus_t aclblasStpmv_legacy(
    aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, int64_t n,
    const float* aPacked, const float* x, float* y, int64_t incx)
{
    // issue #747: 校验顺序对齐 arch35/ctpmv 契约——handle → 枚举 → n<0 拒绝 →
    // n==0 快速返回（仅 n==0 静默成功，n<0 为非法输入）
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aPacked == nullptr || x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    constexpr uint32_t numBlocks = 8;
    uint32_t uploVal = static_cast<uint32_t>(uplo);
    uint32_t transVal = static_cast<uint32_t>(trans);
    TpmvTilingData tiling =
        CalTilingData(static_cast<uint32_t>(n), numBlocks, incx, uploVal, transVal, static_cast<uint32_t>(diag));

    stpmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(aPacked)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<float*>(y)), nullptr, tiling, numBlocks, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
