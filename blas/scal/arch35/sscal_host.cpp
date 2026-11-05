/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sscal_host.cpp
 * \brief Single-precision vector scaling: x = alpha * x
 *        Arch35 (ascend950) host-side implementation.
 *        Dual-path: AIV SIMD for incx==1, SIMT for incx!=1.
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "sscal_tiling_data.h"

void sscal_kernel_do(uint8_t* x, uint8_t* workSpace, const SscalTilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateSscalParams(aclblasHandle_t handle, const float* x, int incx)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSscal", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasSscal", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static SscalTilingData CalSscalTilingDataContiguous(uint32_t totalFloatNum, uint32_t aivCoreNum, float alpha)
{
    SscalTilingData tiling{};
    tiling.totalN = totalFloatNum;
    tiling.incx = 1;

    constexpr uint32_t alignUnit = ELEMENTS_PER_BLOCK;
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSscal", "aivCoreNum is 0, skip tiling calculation");
        return tiling;
    }
    uint32_t rawPerCore = totalFloatNum / aivCoreNum;
    tiling.perCoreN = (rawPerCore / alignUnit) * alignUnit;
    tiling.remainder = totalFloatNum - tiling.perCoreN * aivCoreNum;

    constexpr uint32_t queueCount = 2;
    uint32_t maxElements = UB_SIZE / (queueCount * sizeof(float));
    tiling.tileSize = (maxElements / alignUnit) * alignUnit;

    tiling.alpha = alpha;
    tiling.nthreads = 0;
    tiling.useCoreNum = 0;

    return tiling;
}

static SscalTilingData CalSscalTilingDataStrided(int64_t n, int64_t incx, uint32_t aivCoreNum, float alpha)
{
    SscalTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.incx = incx;
    tiling.alpha = alpha;

    uint32_t useCoreNum = std::min(aivCoreNum, static_cast<uint32_t>(n));
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    if (useCoreNum > SSCAL_MAX_CORE_NUM) {
        useCoreNum = SSCAL_MAX_CORE_NUM;
    }
    tiling.useCoreNum = useCoreNum;

    uint32_t baseCount = static_cast<uint32_t>(n) / useCoreNum;
    uint32_t remain = static_cast<uint32_t>(n) % useCoreNum;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < useCoreNum; i++) {
        tiling.startOffset[i] = offset;
        tiling.calCount[i] = baseCount + (i < remain ? 1 : 0);
        offset += tiling.calCount[i];
    }
    for (uint32_t i = useCoreNum; i < SSCAL_MAX_CORE_NUM; i++) {
        tiling.startOffset[i] = 0;
        tiling.calCount[i] = 0;
    }

    tiling.nthreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), useCoreNum), SIMT_MIN_THREAD_NUM),
        SIMT_MAX_THREAD_NUM);

    tiling.perCoreN = 0;
    tiling.remainder = 0;
    tiling.tileSize = 0;

    return tiling;
}

aclblasStatus_t aclblasSscal(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSscal", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    if (n <= 0 || incx <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (alpha == nullptr) {
        OP_LOGE("aclblasSscal", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasStatus_t status = ValidateSscalParams(handle, x, incx);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // issue #806: SIMT 路径按 (startOffset + i) * stride 计算元素偏移，(n-1)*incx 超出
    // uint32 索引域时回绕读写错误地址（in-place 写，破坏性强）。kernel 侧已统一 uint64
    // 索引（对齐同族 csscal），此处按 srot_host.cpp 的拒绝式守卫口径再做跨度校验。
    const int64_t span = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx);
    if (span > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(
            "aclblasSscal", "strided offset overflow: n=%d incx=%d span=%lld exceeds uint32 index domain", n, incx,
            static_cast<long long>(span));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSscal", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    auto* h = handle;
    uint32_t totalN = static_cast<uint32_t>(n);

    SscalTilingData tiling;
    uint32_t numBlocks;

    if (incx == 1) {
        numBlocks = (totalN < aivCoreNum) ? totalN : aivCoreNum;
        tiling = CalSscalTilingDataContiguous(totalN, numBlocks, *alpha);
    } else {
        numBlocks = std::min(CeilDiv<uint32_t>(totalN, SIMT_MIN_THREAD_NUM), aivCoreNum);
        tiling = CalSscalTilingDataStrided(n, incx, numBlocks, *alpha);
    }

    OP_LOGD(
        "aclblasSscal", "tiling: n=%d incx=%d numBlocks=%u alpha=%.6f", n, incx, numBlocks,
        static_cast<double>(*alpha));

    sscal_kernel_do(reinterpret_cast<uint8_t*>(x), nullptr, tiling, numBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
