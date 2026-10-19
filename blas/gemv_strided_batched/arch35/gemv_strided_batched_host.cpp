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
#include <climits>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "gemv_strided_batched_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

static constexpr const char* K_OP_NAME = "aclblasGemvStridedBatched";

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void gemv_strided_batched_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR y,
                                     const GemvStridedBatchedTilingData& tiling,
                                     uint32_t numBlocks, void *stream);

// ============================================================
// Buffer layout helpers
// ============================================================
enum class GemvType { S, HSH, HSS, TST, TSS };

struct GemvBufCfg {
    uint32_t szIn;
    uint32_t szY;
    uint32_t tn;
    bool hasExtra;
};

static constexpr GemvBufCfg kGmvCfg_F32 = {sizeof(float),    sizeof(float),    1, false};
static constexpr GemvBufCfg kGmvCfg_HSH = {sizeof(uint16_t), sizeof(uint16_t), 2, true};
static constexpr GemvBufCfg kGmvCfg_HSS = {sizeof(uint16_t), sizeof(float),    2, true};
static constexpr GemvBufCfg kGmvCfg_TST = {sizeof(uint16_t), sizeof(uint16_t), 2, true};
static constexpr GemvBufCfg kGmvCfg_TSS = {sizeof(uint16_t), sizeof(float),    2, true};

struct GemvBufLayout { uint32_t inA, inx, inY, matTmp, vecTmp; uint32_t total; };

static GemvBufLayout ComputeBufferLayout(GemvType type, uint32_t mRow, uint32_t nCol)
{
    constexpr uint32_t MTE_ALIGN_BYTES = 256u;
    auto round256 = [](uint64_t x) -> uint32_t {
        return static_cast<uint32_t>((x + MTE_ALIGN_BYTES - 1u) & ~(MTE_ALIGN_BYTES - 1u));
    };
    GemvBufCfg c = (type == GemvType::HSH) ? kGmvCfg_HSH
                 : (type == GemvType::HSS) ? kGmvCfg_HSS
                 : (type == GemvType::TST) ? kGmvCfg_TST
                 : (type == GemvType::TSS) ? kGmvCfg_TSS : kGmvCfg_F32;
    constexpr uint32_t BN = GEMV_STRIDED_BATCHED_BN_NORMAL;
    constexpr uint32_t MIN_VEC_ELEMENTS = 64u;
    // issue #495: 转置 RegBase 点积路径每行按整 VL（64 个 FP32 = 256B）加载且
    // load 本身不带掩码，尾行读取会越过 inA 子缓冲区。这里为 inA 末尾预留一个
    // VL 的 staging 余量，保证 load 范围始终落在已分配的子缓冲区内。
    constexpr uint32_t INA_VL_PAD_BYTES = 256u;
    uint32_t vecE = (nCol < MIN_VEC_ELEMENTS) ? MIN_VEC_ELEMENTS : nCol;
    GemvBufLayout lay = {};
    lay.inA    = round256(static_cast<uint64_t>(BN) * mRow * nCol * c.szIn + INA_VL_PAD_BYTES);
    lay.inx    = round256(static_cast<uint64_t>(BN) * vecE * c.szIn);
    lay.inY    = round256(static_cast<uint64_t>(BN) * mRow * c.szY);
    lay.matTmp = round256(static_cast<uint64_t>(c.tn) * mRow * nCol * sizeof(float));
    lay.vecTmp = round256(static_cast<uint64_t>(2u) * mRow * sizeof(float));
    if (c.hasExtra) {
        lay.vecTmp = round256(static_cast<uint64_t>(2u) * mRow * sizeof(float)
                            + static_cast<uint64_t>(nCol) * sizeof(float));
        uint32_t outBufSize = (type == GemvType::HSH || type == GemvType::TST) ? lay.inY : 0;
        lay.total  = lay.inA + lay.inx + lay.inY + outBufSize + lay.matTmp + lay.vecTmp
                   + round256(static_cast<uint64_t>(mRow) * nCol * sizeof(uint16_t))
                   + round256(static_cast<uint64_t>(vecE) * sizeof(uint16_t));
        return lay;
    }
    lay.total = lay.inA + lay.inx + lay.inY + lay.matTmp + lay.vecTmp;
    return lay;
}

static inline uint32_t ComputeUBPerBatch(GemvType type, uint32_t mRow, uint32_t nCol)
    { return ComputeBufferLayout(type, mRow, nCol).total; }

static inline uint32_t AlignDown(uint32_t x, uint32_t a)
{ return (a == 0) ? 0 : (x - (x % a)); }

static inline uint32_t AlignUp(uint32_t x, uint32_t a)
{ return (a == 0) ? 0 : AlignDown(x + a - 1, a); }


template <typename F>
static uint32_t BinaryMaxRowsCols(uint32_t UB, uint32_t hi, F&& f)
{
    uint32_t lo = 0;
    while (lo < hi) {
        uint32_t mid = (lo + hi + 1) / 2;
        if (f(mid) <= UB) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

static void CalTilingTranspose(GemvType type, uint32_t dotDim, uint32_t outDim,
                                uint32_t &dotTile, uint32_t &outTile)
{
    const uint32_t UB = GEMV_STRIDED_BATCHED_UBUF_SIZE;
    uint32_t elemSize = (type == GemvType::S) ? sizeof(float) : sizeof(uint16_t);
    uint32_t align = 32u / elemSize, outMin = 1;
    auto ubFunc = [type](uint32_t r, uint32_t c) { return ComputeUBPerBatch(type, r, c); };
    constexpr uint32_t MAX_SEARCH_DIM = 65536u;
    uint32_t dot_aligned = AlignUp(dotDim, align);
    uint32_t outFitForFullDot = BinaryMaxRowsCols(UB, outDim > 0 ? outDim : MAX_SEARCH_DIM,
        [&](uint32_t r) { return ubFunc(r, dot_aligned); });
    if (outFitForFullDot >= outMin) {
        dotTile = dot_aligned;
    } else {
        uint32_t dotTileMax = BinaryMaxRowsCols(UB, MAX_SEARCH_DIM,
            [&](uint32_t c) { return ubFunc(outMin, c); });
        dotTile = AlignDown((dotTileMax < align) ? align : dotTileMax, align);
        if (dotTile < align) dotTile = align;
        if (dotTile > dot_aligned) dotTile = dot_aligned;
        outFitForFullDot = BinaryMaxRowsCols(UB, outDim > 0 ? outDim : MAX_SEARCH_DIM,
            [&](uint32_t r) { return ubFunc(r, dotTile); });
    }
    uint32_t outFit = AlignDown(outFitForFullDot, outMin);
    if (outFit < outMin) outFit = outMin;
    outTile = (outDim < outFit) ? outDim : outFit;
}

static void CalTilingParams(bool isTrans, bool isFp32, uint32_t m, uint32_t n,
                             uint32_t &dotTile, uint32_t &outTile)
{
    if (!isTrans) {
        dotTile = 0; outTile = m;
        return;
    }
    CalTilingTranspose(isFp32 ? GemvType::S : GemvType::HSH, m, n, dotTile, outTile);
}

static void CalTilingParamsHss(bool isTrans, uint32_t m, uint32_t n,
                                uint32_t &dotTile, uint32_t &outTile)
{
    if (!isTrans) { dotTile = 0; outTile = m; return; }
    CalTilingTranspose(GemvType::HSS, m, n, dotTile, outTile);
}

// ============================================================
// Tiling data calculation
// ============================================================
static GemvStridedBatchedTilingData CalTilingData(
    uint32_t batchCount, uint32_t m, uint32_t n,
    uint32_t dtype, uint32_t trans,
    float alpha, float beta,
    uint32_t usedCoreNum, uint32_t maxPerCore,
    int32_t lda, int32_t incx, int32_t incy,
    int64_t strideA, int64_t stridex, int64_t stridey)
{
    GemvStridedBatchedTilingData td = {};
    td.dtype = dtype;
    td.trans = trans;
    td.alpha = alpha;  td.beta  = beta;
    td.m = m;          td.n = n;

    uint32_t dotTile = 0, outTile = 0;
    if (dtype == 2 || dtype == 4)
        CalTilingParamsHss(trans != 0, m, n, dotTile, outTile);
    else if (dtype == 0 || dtype == 3)
        CalTilingParams(trans != 0, false, m, n, dotTile, outTile);
    else
        CalTilingParams(trans != 0, true, m, n, dotTile, outTile);
    td.dotTile = dotTile;  td.outTile = outTile;

    auto bufType = (dtype == 2) ? GemvType::HSS
                 : (dtype == 3) ? GemvType::TST
                 : (dtype == 4) ? GemvType::TSS
                 : (dtype != 0) ? GemvType::S : GemvType::HSH;
    auto lay = ComputeBufferLayout(bufType, outTile, dotTile);
    td.bufInA = lay.inA;  td.bufInx = lay.inx;  td.bufInY = lay.inY;
    td.bufMatTmp = lay.matTmp;  td.bufVecTmp = lay.vecTmp;

    td.lda = lda;  td.incx = incx;  td.incy = incy;
    td.strideA = strideA;  td.stridex = stridex;  td.stridey = stridey;

    td.outSize = (trans != 0) ? n : m;
    td.dotSize = (trans != 0) ? m : n;

    td.usedCoreNum = usedCoreNum;
    td.batchCount = batchCount;  td.batchPerCore = maxPerCore;
    td.batchTail = batchCount - static_cast<uint32_t>(static_cast<uint64_t>(usedCoreNum - 1) * maxPerCore);

    uint64_t workPerCore = static_cast<uint64_t>(maxPerCore) * static_cast<uint64_t>(td.outSize);
    uint32_t numThreads;
    if (workPerCore >= static_cast<uint64_t>(SIMT_MAX_THREAD_NUM)) {
        numThreads = SIMT_MAX_THREAD_NUM;
    } else {
        numThreads = CeilAlign(static_cast<uint32_t>(workPerCore), SIMT_MIN_THREAD_NUM);
        if (numThreads > SIMT_MAX_THREAD_NUM) numThreads = SIMT_MAX_THREAD_NUM;
        if (numThreads < SIMT_MIN_THREAD_NUM) numThreads = SIMT_MIN_THREAD_NUM;
    }
    td.numThreads = numThreads;

    return td;
}

// ============================================================
// Parameter validation
// ============================================================
static aclblasStatus_t ValidateGemvStridedBatchedParams(
    aclblasOperation_t trans, int m, int n,
    int lda, int64_t strideA, int incx, int incy, int64_t stridex, int64_t stridey,
    const float* alpha, const float* beta,
    const void* A, const void* x, const void* y)
{
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE(K_OP_NAME, "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        lda >= std::max(1, m), OP_LOGE(K_OP_NAME, "invalid lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        strideA > 0, OP_LOGE(K_OP_NAME, "strideA must be positive, got %lld", static_cast<long long>(strideA));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE(K_OP_NAME, "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE(K_OP_NAME, "incx must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE(K_OP_NAME, "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != INT_MIN, OP_LOGE(K_OP_NAME, "incy must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        stridex > 0, OP_LOGE(K_OP_NAME, "stridex must be positive, got %lld", static_cast<long long>(stridex));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        stridey > 0, OP_LOGE(K_OP_NAME, "stridey must be positive, got %lld", static_cast<long long>(stridey));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE(K_OP_NAME, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE(K_OP_NAME, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (m > 0 && n > 0) {
        CHECK_RET(A != nullptr, OP_LOGE(K_OP_NAME, "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(x != nullptr, OP_LOGE(K_OP_NAME, "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(y != nullptr, OP_LOGE(K_OP_NAME, "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================
// Launch kernel: pass tiling by value
// ============================================================
template <typename T_IN, typename T_OUT>
static aclblasStatus_t LaunchKernel(
    _aclblas_handle* h, const T_IN* A, const T_IN* x, T_OUT* y,
    const GemvStridedBatchedTilingData& tiling, uint32_t usedCoreNum)
{
    gemv_strided_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<T_IN*>(A)),
        reinterpret_cast<uint8_t*>(const_cast<T_IN*>(x)),
        reinterpret_cast<uint8_t*>(y),
        tiling, usedCoreNum, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================
// Template implementation
// ============================================================
template <typename T_IN, typename T_OUT>
static aclblasStatus_t GemvStridedBatchedImpl(
    aclblasHandle_t    handle,
    aclblasOperation_t trans,
    int                m,
    int                n,
    const float*       alpha,
    const T_IN*        A,
    int                lda,
    int64_t            strideA,
    const T_IN*        x,
    int                incx,
    int64_t            stridex,
    const float*       beta,
    T_OUT*             y,
    int                incy,
    int64_t            stridey,
    int                batchCount,
    uint32_t           dtype)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE(K_OP_NAME, "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(m >= 0, OP_LOGE(K_OP_NAME, "invalid m=%d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE(K_OP_NAME, "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        batchCount >= 0, OP_LOGE(K_OP_NAME, "invalid batchCount=%d", batchCount); return ACLBLAS_STATUS_INVALID_VALUE);

    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateGemvStridedBatchedParams(
        trans, m, n, lda, strideA, incx, incy, stridex, stridey, alpha, beta, A, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE(K_OP_NAME, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t batchCountU = static_cast<uint32_t>(batchCount);
    uint32_t maxPerCore  = (batchCountU + coreNum - 1) / coreNum;
    uint32_t usedCoreNum = (maxPerCore > 0) ? ((batchCountU + maxPerCore - 1) / maxPerCore) : 0;
    uint32_t transUint   = (trans == ACLBLAS_OP_N) ? 0 : 1;

    GemvStridedBatchedTilingData tiling = CalTilingData(
        batchCountU, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        dtype, transUint, *alpha, *beta,
        usedCoreNum, maxPerCore,
        static_cast<int32_t>(lda), static_cast<int32_t>(incx), static_cast<int32_t>(incy),
        strideA, stridex, stridey);

    OP_LOGD(K_OP_NAME,
        "tiling: dtype=%u m=%u n=%u trans=%u usedCoreNum=%u batchPerCore=%u numThreads=%u "
        "strideA=%lld stridex=%lld stridey=%lld",
        tiling.dtype, tiling.m, tiling.n, tiling.trans, tiling.usedCoreNum,
        tiling.batchPerCore, tiling.numThreads,
        static_cast<long long>(tiling.strideA), static_cast<long long>(tiling.stridex),
        static_cast<long long>(tiling.stridey));
    OP_LOGI(K_OP_NAME, "launching kernel: blocks=%u, cores=%u", usedCoreNum, coreNum);

    return LaunchKernel<T_IN, T_OUT>(h, A, x, y, tiling, usedCoreNum);
}

// ============================================================
// Host API entry points
// ============================================================
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float* A, int lda, int64_t strideA, const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount) {
    return GemvStridedBatchedImpl<float, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 1);
}

aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount) {
    return GemvStridedBatchedImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 0);
}

aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount) {
    return GemvStridedBatchedImpl<uint16_t, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 2);
}

aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount) {
    return GemvStridedBatchedImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 3);
}

aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount) {
    return GemvStridedBatchedImpl<uint16_t, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 4);
}
