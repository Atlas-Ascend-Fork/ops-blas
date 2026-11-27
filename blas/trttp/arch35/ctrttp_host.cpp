/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "ctrttp_kernel.h"
#include "log/log.h"

namespace {

constexpr uint64_t CTRTTP_UB_RESERVE_BYTES = 4096U;

// Parameter contract (task book 2.5): handle -> n < 0 -> uplo -> lda >= max(1, n) -> n == 0
// quick return -> A / AP null check. n == 0 with lda == 0 is therefore invalid, not a no-op.
aclblasStatus_t ValidateCtrttpParams(
    aclblasFillMode_t uplo, int n, const aclblasComplex* A, int lda, const aclblasComplex* AP)
{
    if (n < 0) {
        OP_LOGE("aclblasCtrttp", "n=%d, expected >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) {
        OP_LOGE("aclblasCtrttp", "uplo=%d, expected 121(UPPER) or 122(LOWER)", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int minLda = std::max(1, n);
    if (lda < minLda) {
        OP_LOGE("aclblasCtrttp", "lda=%d, expected >= max(1,n)=%d", lda, minLda);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasCtrttp", "A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (AP == nullptr) {
        OP_LOGE("aclblasCtrttp", "AP must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// UB burst size in complex64 elements: one slot per queue entry, 32-byte aligned, capped at
// the largest transfer the DataCopyPad path is exercised with.
uint32_t CalcTileElems()
{
    uint64_t ubSize = 0;
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform != nullptr) {
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    if (ubSize == 0) {
        ubSize = 248U * 1024U;
    }
    const uint64_t usable = (ubSize > CTRTTP_UB_RESERVE_BYTES) ? (ubSize - CTRTTP_UB_RESERVE_BYTES) : 0;
    uint64_t elems = usable / (static_cast<uint64_t>(CTRTTP_QUEUE_DEPTH) * CTRTTP_BYTES_PER_COMPLEX);
    elems = std::min<uint64_t>(elems, CTRTTP_MAX_CHUNK_ELEMS);
    elems = (elems / CTRTTP_COMPLEX_PER_BLOCK) * CTRTTP_COMPLEX_PER_BLOCK;
    if (elems == 0) {
        elems = CTRTTP_COMPLEX_PER_BLOCK;
    }
    return static_cast<uint32_t>(elems);
}

aclblasStatus_t LaunchCtrttpKernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* A, int lda, aclblasComplex* AP)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCtrttp", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const uint32_t order = static_cast<uint32_t>(n);
    // One column is the finest task unit worth a block; small orders simply use fewer cores.
    uint32_t numBlocks = std::min(aivCoreNum, order);
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    CtrttpTilingData tiling{};
    tiling.n = order;
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = (uplo == ACLBLAS_LOWER) ? CTRTTP_UPLO_LOWER : CTRTTP_UPLO_UPPER;
    tiling.useCoreNum = numBlocks;
    tiling.tileElems = CalcTileElems();
    tiling.totalPacked = static_cast<uint64_t>(order) * (order + 1ULL) / 2ULL;

    OP_LOGD(
        "aclblasCtrttp", "tiling: n=%u lda=%u uplo=%u blocks=%u tileElems=%u totalPacked=%llu", tiling.n, tiling.lda,
        tiling.uplo, tiling.useCoreNum, tiling.tileElems, static_cast<unsigned long long>(tiling.totalPacked));
    OP_LOGI("aclblasCtrttp", "launching kernel: blocks=%u", numBlocks);

    ctrttp_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), reinterpret_cast<uint8_t*>(AP), tiling, numBlocks,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCtrttp(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* A, const int lda,
    aclblasComplex* AP)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCtrttp", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    const aclblasStatus_t status = ValidateCtrttpParams(uplo, n, A, lda, AP);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    return LaunchCtrttpKernel(handle, uplo, n, A, lda, AP);
}
