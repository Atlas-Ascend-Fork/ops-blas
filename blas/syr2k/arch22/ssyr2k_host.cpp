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
#include "ssyr2k_kernel.h"
#include "ssyr2k_tiling_data.h"

namespace {

constexpr const char* SSYR2K_TAG = "aclblasSsyr2k";

aclblasStatus_t ValidateSsyr2kParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, const float* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(SSYR2K_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE(SSYR2K_TAG, "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE(SSYR2K_TAG, "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0 || k < 0) {
        OP_LOGE(SSYR2K_TAG, "negative dimension n=%d k=%d", n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int minLd = std::max(1, (trans == ACLBLAS_OP_N) ? n : k);
    if (lda < minLd || ldb < minLd) {
        OP_LOGE(SSYR2K_TAG, "invalid lda=%d or ldb=%d, expected >= %d", lda, ldb, minLd);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, n)) {
        OP_LOGE(SSYR2K_TAG, "invalid ldc=%d, expected >= %d", ldc, std::max(1, n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE(SSYR2K_TAG, "alpha or beta is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (C == nullptr || (k > 0 && (A == nullptr || B == nullptr))) {
        OP_LOGE(SSYR2K_TAG, "null matrix pointer");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

void LaunchSsyr2kPair(
    uint32_t aicCount, void* stream, uint8_t* aPtr, uint8_t* bPtr, uint8_t* tempBuf,
    Ssyr2kGemmTilingData tiling, bool isTrans)
{
    const uint32_t origLda = tiling.lda;
    const uint32_t origLdb = tiling.ldb;
    for (uint32_t kBase = 0U; kBase < tiling.k; kBase += SSYR2K_SINGLE_K) {
        tiling.kBase = kBase;
        tiling.kCount = std::min(SSYR2K_SINGLE_K, tiling.k - kBase);
        tiling.lda = origLda;
        tiling.ldb = origLdb;
        tiling.enAtomic = (kBase == 0U) ? 0U : 1U;
        ssyr2k_gemm_launch(aicCount, stream, aPtr, bPtr, tempBuf, tiling, isTrans);

        tiling.lda = origLdb;
        tiling.ldb = origLda;
        tiling.enAtomic = 1U;
        ssyr2k_gemm_launch(aicCount, stream, bPtr, aPtr, tempBuf, tiling, isTrans);
    }
}

aclblasStatus_t PrepareSsyr2kWorkspace(
    _aclblas_handle* h, uint32_t nU, uint32_t tempLd, uint8_t*& tempBuf)
{
    uint64_t tempBytes = 0U;
    if (!CBlas3TryMulU64(tempLd, nU, &tempBytes) ||
        !CBlas3TryMulU64(tempBytes, sizeof(float), &tempBytes) ||
        !CBlas3TryAlignGm(tempBytes, &tempBytes)) {
        OP_LOGE(SSYR2K_TAG, "workspace size overflows 64 bits");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclblasStatus_t ret = EnsureDefaultWorkspace(h, static_cast<size_t>(tempBytes));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    tempBuf = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (tempBuf == nullptr) {
        OP_LOGE(SSYR2K_TAG, "GetEffectiveWorkspace returned nullptr");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchSsyr2kGemm(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t nU, uint32_t kU,
    uint32_t ldaU, uint32_t ldbU, uint32_t tempLd, const float* A, const float* B, uint8_t* tempBuf)
{
    const uint32_t aicCount = GetAicCoreCount();
    if (aicCount == 0U) {
        OP_LOGE(SSYR2K_TAG, "GetAicCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    Ssyr2kGemmTilingData tiling{};
    tiling.n = nU;
    tiling.k = kU;
    tiling.lda = ldaU;
    tiling.ldb = ldbU;
    tiling.tempLd = tempLd;
    tiling.uploMode = (uplo == ACLBLAS_UPPER) ? SSYR2K_UPLO_UPPER : SSYR2K_UPLO_LOWER;
    tiling.mBlocks = CeilDiv(nU, SSYR2K_TILE_M);
    tiling.nBlocks = CeilDiv(nU, SSYR2K_TILE_N);
    LaunchSsyr2kPair(
        aicCount, h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(A)),
        reinterpret_cast<uint8_t*>(const_cast<float*>(B)), tempBuf, tiling, trans != ACLBLAS_OP_N);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t RunSsyr2kNpu(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t nU, uint32_t kU,
    float alphaVal, const float* A, uint32_t ldaU, const float* B, uint32_t ldbU, float betaVal, float* C,
    uint32_t ldcU)
{
    const bool hasTemp = (kU > 0U && alphaVal != 0.0F);
    if (!hasTemp && betaVal == 1.0F) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t tempLd = SSYR2K_TEMP_ALIGN;
    uint8_t* tempBuf = nullptr;
    if (hasTemp) {
        tempLd = CeilAlign<uint32_t>(nU, SSYR2K_TEMP_ALIGN);
        aclblasStatus_t ret = PrepareSsyr2kWorkspace(h, nU, tempLd, tempBuf);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
        ret = LaunchSsyr2kGemm(h, uplo, trans, nU, kU, ldaU, ldbU, tempLd, A, B, tempBuf);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
    }

    const uint32_t aivCount = GetAivCoreCount();
    if (aivCount == 0U) {
        OP_LOGE(SSYR2K_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    Ssyr2kCombineTilingData combineTiling{};
    combineTiling.n = nU;
    combineTiling.ldc = ldcU;
    combineTiling.tempLd = tempLd;
    combineTiling.uploMode = (uplo == ACLBLAS_UPPER) ? SSYR2K_UPLO_UPPER : SSYR2K_UPLO_LOWER;
    combineTiling.hasTemp = hasTemp ? 1U : 0U;
    combineTiling.alpha = alphaVal;
    combineTiling.beta = betaVal;

    ssyr2k_combine_launch(std::min(aivCount, nU), h->stream, tempBuf, reinterpret_cast<uint8_t*>(C), combineTiling);
    aclError err = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    if (err != ACL_SUCCESS) {
        OP_LOGE(SSYR2K_TAG, "kernel launch failed, err=%d", static_cast<int>(err));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasSsyr2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    aclblasStatus_t st = ValidateSsyr2kParams(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

    return RunSsyr2kNpu(
        h, uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alphaVal, A,
        static_cast<uint32_t>(lda), B, static_cast<uint32_t>(ldb), betaVal, C, static_cast<uint32_t>(ldc));
}
