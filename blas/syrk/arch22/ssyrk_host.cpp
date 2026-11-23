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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "ssyrk_kernel.h"
#include "ssyrk_tiling_data.h"

namespace {

constexpr const char* SSYRK_TAG = "aclblasSsyrk";

aclblasStatus_t ValidateSsyrkParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* beta, const float* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(SSYRK_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE(SSYRK_TAG, "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE(SSYRK_TAG, "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0 || k < 0) {
        OP_LOGE(SSYRK_TAG, "negative dimension n=%d k=%d", n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int minLda = std::max(1, (trans == ACLBLAS_OP_N) ? n : k);
    if (lda < minLda) {
        OP_LOGE(SSYRK_TAG, "invalid lda=%d, expected >= %d", lda, minLda);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, n)) {
        OP_LOGE(SSYRK_TAG, "invalid ldc=%d, expected >= %d", ldc, std::max(1, n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE(SSYRK_TAG, "alpha or beta is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (C == nullptr || (k > 0 && A == nullptr)) {
        OP_LOGE(SSYRK_TAG, "null matrix pointer");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchSsyrkGemm(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t nU, uint32_t kU,
    uint32_t ldaU, uint32_t tempLd, const float* A, uint8_t* tempBuf)
{
    const uint32_t aicCount = GetAicCoreCount();
    if (aicCount == 0U) {
        OP_LOGE(SSYRK_TAG, "GetAicCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    SsyrkGemmTilingData tiling{};
    tiling.n = nU;
    tiling.k = kU;
    tiling.lda = ldaU;
    tiling.tempLd = tempLd;
    tiling.uploMode = (uplo == ACLBLAS_UPPER) ? SSYRK_UPLO_UPPER : SSYRK_UPLO_LOWER;
    tiling.mBlocks = CeilDiv(nU, SSYRK_TILE_M);
    tiling.nBlocks = CeilDiv(nU, SSYRK_TILE_N);

    const bool isTrans = (trans != ACLBLAS_OP_N);
    uint8_t* aPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
    for (uint32_t kBase = 0U; kBase < kU; kBase += SSYRK_SINGLE_K) {
        tiling.kBase = kBase;
        tiling.kCount = std::min(SSYRK_SINGLE_K, kU - kBase);
        tiling.enAtomic = (kBase == 0U) ? 0U : 1U;
        ssyrk_gemm_launch(aicCount, h->stream, aPtr, tempBuf, tiling, isTrans);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t PrepareSsyrkWorkspace(
    _aclblas_handle* h, uint32_t nU, uint32_t tempLd, uint8_t*& tempBuf)
{
    uint64_t tempBytes = 0U;
    if (!CBlas3TryMulU64(tempLd, nU, &tempBytes) ||
        !CBlas3TryMulU64(tempBytes, sizeof(float), &tempBytes) ||
        !CBlas3TryAlignGm(tempBytes, &tempBytes)) {
        OP_LOGE(SSYRK_TAG, "workspace size overflows 64 bits");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclblasStatus_t ret = EnsureDefaultWorkspace(h, static_cast<size_t>(tempBytes));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    tempBuf = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (tempBuf == nullptr) {
        OP_LOGE(SSYRK_TAG, "GetEffectiveWorkspace returned nullptr");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t RunSsyrkNpu(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t nU, uint32_t kU,
    float alphaVal, const float* A, uint32_t ldaU, float betaVal, float* C, uint32_t ldcU)
{
    const bool hasTemp = (kU > 0U && alphaVal != 0.0F);
    if (!hasTemp && betaVal == 1.0F) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t tempLd = SSYRK_TEMP_ALIGN;
    uint8_t* tempBuf = nullptr;
    if (hasTemp) {
        tempLd = CeilAlign<uint32_t>(nU, SSYRK_TEMP_ALIGN);
        aclblasStatus_t ret = PrepareSsyrkWorkspace(h, nU, tempLd, tempBuf);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
        ret = LaunchSsyrkGemm(h, uplo, trans, nU, kU, ldaU, tempLd, A, tempBuf);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
    }

    const uint32_t aivCount = GetAivCoreCount();
    if (aivCount == 0U) {
        OP_LOGE(SSYRK_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint32_t combineCores = std::min(aivCount, nU);

    SsyrkCombineTilingData combineTiling{};
    combineTiling.n = nU;
    combineTiling.ldc = ldcU;
    combineTiling.tempLd = tempLd;
    combineTiling.uploMode = (uplo == ACLBLAS_UPPER) ? SSYRK_UPLO_UPPER : SSYRK_UPLO_LOWER;
    combineTiling.hasTemp = hasTemp ? 1U : 0U;
    combineTiling.alpha = alphaVal;
    combineTiling.beta = betaVal;

    ssyrk_combine_launch(combineCores, h->stream, tempBuf, reinterpret_cast<uint8_t*>(C), combineTiling);
    aclError err = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    if (err != ACL_SUCCESS) {
        OP_LOGE(SSYRK_TAG, "kernel launch failed, err=%d", static_cast<int>(err));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasSsyrk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc)
{
    aclblasStatus_t st = ValidateSsyrkParams(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return st;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);
    float alphaVal = 0.0F;
    float betaVal = 0.0F;
    st = ReadScalarFromHostOrDevice(alpha, h->stream, alphaVal);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = ReadScalarFromHostOrDevice(beta, h->stream, betaVal);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    return RunSsyrkNpu(
        h, uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alphaVal, A,
        static_cast<uint32_t>(lda), betaVal, C, static_cast<uint32_t>(ldc));
}
