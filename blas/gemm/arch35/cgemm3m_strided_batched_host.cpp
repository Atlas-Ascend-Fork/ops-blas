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
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemm_host_common.h"
#include "cgemm3m_strided_batched_tiling_data.h"
namespace {
constexpr const char* OP = "aclblasCgemm3mStridedBatched";
constexpr uint32_t DIRECT_THREADS = 256;
constexpr uint64_t DIRECT_OUTPUT_LIMIT = 1024;
constexpr uint32_t DIRECT_REDUCTION_LIMIT = 256;
bool ValidOperation(aclblasOperation_t operation)
{
    return operation == ACLBLAS_OP_N || operation == ACLBLAS_OP_T || operation == ACLBLAS_OP_C;
}
bool InvalidPointers(
    const aclblasComplex* alpha, const aclblasComplex* a, const aclblasComplex* b, const aclblasComplex* beta,
    const aclblasComplex* c, int m, int n, int k, int batch)
{
    if (alpha == nullptr || beta == nullptr) {
        return true;
    }
    if (m == 0 || n == 0 || batch == 0) {
        return false;
    }
    if (c == nullptr) {
        return true;
    }
    const bool multiply = k > 0 && (alpha->real != 0 || alpha->imag != 0);
    return multiply && (a == nullptr || b == nullptr);
}
aclblasStatus_t Validate(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb,
    const aclblasComplex* beta, aclblasComplex* c, int ldc, int batch)
{
    if (handle == nullptr) {
        OP_LOGE(OP, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!ValidOperation(ta) || !ValidOperation(tb)) {
        OP_LOGE(OP, "invalid operation: transa=%d transb=%d", ta, tb);
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0 || batch < 0) {
        OP_LOGE(OP, "invalid shape: m=%d n=%d k=%d batch=%d", m, n, k, batch);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, ta == ACLBLAS_OP_N ? m : k) || ldb < std::max(1, tb == ACLBLAS_OP_N ? k : n) ||
        ldc < std::max(1, m)) {
        OP_LOGE(OP, "invalid leading dimensions: m=%d n=%d k=%d lda=%d ldb=%d ldc=%d", m, n, k, lda, ldb, ldc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (InvalidPointers(alpha, a, b, beta, c, m, n, k, batch)) {
        OP_LOGE(OP, "invalid scalar or matrix pointer: m=%d n=%d k=%d batch=%d", m, n, k, batch);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
aclblasStatus_t LaunchDirect(
    aclblasHandle_t handle, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c,
    const Cgemm3mSbTiling& t)
{
    uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        OP_LOGE(OP, "GetAivCoreCount returned zero");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint64_t elements = uint64_t(t.m) * t.n * (t.strideC == 0 ? 1 : t.batch);
    uint32_t blocks = std::min<uint64_t>(cores, (elements + DIRECT_THREADS - 1) / DIRECT_THREADS);
    cgemm3m_sb_direct_do(blocks, handle->stream, a, b, c, t);
    return ACLBLAS_STATUS_SUCCESS;
}
aclblasStatus_t LaunchBatches(
    aclblasHandle_t handle, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c,
    const Cgemm3mSbTiling& t)
{
    const aclblasComplex alpha{t.ar, t.ai}, beta{t.br, t.bi};
    const auto ta = t.transa == 0 ? ACLBLAS_OP_N : t.transa == 1 ? ACLBLAS_OP_T : ACLBLAS_OP_C;
    const auto tb = t.transb == 0 ? ACLBLAS_OP_N : t.transb == 1 ? ACLBLAS_OP_T : ACLBLAS_OP_C;
    const bool repair = t.strideC == 0 && (t.br < 0 || t.bi != 0);
    for (uint32_t batch = 0; batch < t.batch; ++batch) {
        const bool multiply = t.k != 0 && (t.ar != 0 || t.ai != 0);
        const auto* ap = multiply ? a + int64_t(batch) * t.strideA : a;
        const auto* bp = multiply ? b + int64_t(batch) * t.strideB : b;
        auto* cp = c + int64_t(batch) * t.strideC;
        auto status =
            repair ?
                aclblasCgemmStridedBatched(
                    handle, ta, tb, t.m, t.n, t.k, &alpha, ap, t.lda, 0, bp, t.ldb, 0, &beta, cp, t.ldc, 0, 1) :
                LaunchCgemmThreeProducts(handle, ta, tb, t.m, t.n, t.k, alpha, ap, t.lda, bp, t.ldb, beta, cp, t.ldc);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            OP_LOGE(OP, "3M batch=%u failed: m=%u n=%u k=%u status=%d", batch, t.m, t.n, t.k, status);
            return status;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
bool UseOnchipKernel(const Cgemm3mSbTiling& t)
{
    const int m = t.m, n = t.n, k = t.k;
    return (k <= 128 && std::max(m, n) <= 128) ||
           (k >= 512 && ((std::min(m, n) > 32 && std::min(m, n) <= 64) || (k <= 768 && std::min(m, n) <= 128)));
}
aclblasStatus_t LaunchUnitBatches(
    aclblasHandle_t handle, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c,
    const Cgemm3mSbTiling& t)
{
    const int m = t.m, n = t.n, k = t.k;
    const auto ta = t.transa == 0 ? ACLBLAS_OP_N : t.transa == 1 ? ACLBLAS_OP_T : ACLBLAS_OP_C;
    const auto tb = t.transb == 0 ? ACLBLAS_OP_N : t.transb == 1 ? ACLBLAS_OP_T : ACLBLAS_OP_C;
    if (m <= 32 && tb != ACLBLAS_OP_N && k <= 128) {
        uint32_t cores = GetAivCoreCount();
        if (cores == 0) {
            OP_LOGE(OP, "GetAivCoreCount returned zero");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        const uint64_t tiles = uint64_t((m + CGEMM3M_PANEL_ROWS - 1) / CGEMM3M_PANEL_ROWS) *
                               ((n + CGEMM3M_PANEL_COLS - 1) / CGEMM3M_PANEL_COLS) * t.batch;
        cgemm3m_sb_panel_do(std::min<uint64_t>(cores, tiles), handle->stream, a, b, c, t);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (UseOnchipKernel(t)) {
        return LaunchCgemm3mOnchipBatches(
            handle, ta, tb, m, n, k, a, t.lda, t.strideA, b, t.ldb, t.strideB, c, t.ldc, t.strideC, t.batch);
    }
    if (t.batch <= 2 && k >= 512 && k <= 1024 && std::min(m, n) <= 256 && std::max(m, n) <= 1024) {
        return LaunchBatches(handle, a, b, c, t);
    }
    return LaunchCgemm3mUnitBatches(
        handle, ta, tb, m, n, k, a, t.lda, t.strideA, b, t.ldb, t.strideB, c, t.ldc, t.strideC, t.batch);
}
bool UseDirectKernel(const Cgemm3mSbTiling& t, bool unit)
{
    const bool tiny = t.m <= 8 && t.n <= 8 && t.k <= 8;
    const bool small = uint64_t(t.m) * t.n <= DIRECT_OUTPUT_LIMIT && t.k <= DIRECT_REDUCTION_LIMIT;
    return small &&
           ((!unit && t.k <= 128) || (unit && (tiny || uint64_t(t.m) * t.n * t.batch <= 65536 / std::max(t.k, 1U))));
}
bool HasProduct(const Cgemm3mSbTiling& t) { return t.k > 0 && (t.ar != 0 || t.ai != 0); }
bool IndependentOutput(const Cgemm3mSbTiling& t)
{
    const int64_t outputSpan = (int64_t(t.n) - 1) * t.ldc + t.m;
    return t.batch == 1 || t.strideC >= outputSpan || t.strideC <= -outputSpan;
}
aclblasStatus_t DispatchBatches(
    aclblasHandle_t handle, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c,
    const Cgemm3mSbTiling& t)
{
    const bool multiply = HasProduct(t);
    if (!multiply && t.br == 1 && t.bi == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const bool unit = t.ar == 1 && t.ai == 0 && t.br == 0 && t.bi == 0;
    const bool direct = UseDirectKernel(t, unit);
    const bool independent = IndependentOutput(t);
    if (!independent && t.strideC != 0) {
        return LaunchBatches(handle, a, b, c, t);
    }
    if (multiply && unit && independent && !direct) {
        return LaunchUnitBatches(handle, a, b, c, t);
    }
    // A reused C with a rotating/sign-changing beta amplifies cancellation across batches.
    if (multiply && t.strideC == 0 && (t.br < 0 || t.bi != 0)) {
        return LaunchBatches(handle, a, b, c, t);
    }
    if (!multiply || direct) {
        return LaunchDirect(handle, a, b, c, t);
    }
    return LaunchBatches(handle, a, b, c, t);
}
} // namespace
extern "C" aclblasStatus_t aclblasCgemm3mStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* a, int lda, int64_t strideA, const aclblasComplex* b, int ldb,
    int64_t strideB, const aclblasComplex* beta, aclblasComplex* c, int ldc, int64_t strideC, int batchCount)
{
    auto status = Validate(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batchCount);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0 || batchCount == 0) {
        return status;
    }
    Cgemm3mSbTiling t{
        uint32_t(m),
        uint32_t(n),
        uint32_t(k),
        uint32_t(lda),
        uint32_t(ldb),
        uint32_t(ldc),
        uint32_t(
            transa == ACLBLAS_OP_N ? 0 :
            transa == ACLBLAS_OP_T ? 1 :
                                     2),
        uint32_t(
            transb == ACLBLAS_OP_N ? 0 :
            transb == ACLBLAS_OP_T ? 1 :
                                     2),
        uint32_t(batchCount),
        strideA,
        strideB,
        strideC,
        alpha->real,
        alpha->imag,
        beta->real,
        beta->imag};
    return DispatchBatches(handle, a, b, c, t);
}
