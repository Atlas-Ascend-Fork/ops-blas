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
 * \file gemv_batched_host.cpp
 * \brief Host-side API for batched real GEMV: y[i] = alpha * op(A[i]) * x[i] + beta * y[i]
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "log/log.h"
#include "gemv_batched_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void gemv_batched_kernel_do(uint8_t* A, uint8_t* x, uint8_t* y,
                            const GemvBatchedTilingData& tiling,
                            uint32_t numBlocks, void *stream);

// Per-batch UB usage with all buffers 256B-aligned.
// VEC_SCOPE loads 64 floats ⇒ inxBuf widened to VL when nCol<64.
enum class GemvType { S, HSH, HSS, TST, TSS };

struct GemvBufCfg {
    uint32_t szIn;   // sizeof(A/x element)
    uint32_t szY;    // sizeof(y element)
    uint32_t tn;     // matTmpBuf copies (1=reuse, 2=separate)
    bool hasExtra;   // matPreBuf + vecPreBuf needed
};

static constexpr GemvBufCfg kGmvCfg_F32 = {sizeof(float),      sizeof(float),      1, false};
static constexpr GemvBufCfg kGmvCfg_HSH = {sizeof(uint16_t),   sizeof(uint16_t),   2, true};
static constexpr GemvBufCfg kGmvCfg_HSS = {sizeof(uint16_t),   sizeof(float),      2, true};
static constexpr GemvBufCfg kGmvCfg_TST = {sizeof(uint16_t),   sizeof(uint16_t),   2, true};
static constexpr GemvBufCfg kGmvCfg_TSS = {sizeof(uint16_t),   sizeof(float),      2, true};

// issue #495: one vector-length (64 FP32 = 256B) staging pad appended to inA for
// the transposed RegBase path, whose per-row DataCopy reads a full VL unmasked.
static constexpr uint32_t GEMV_INA_VL_PAD_BYTES = 256;

struct GemvBufLayout { uint32_t inA, inx, inY, out, matTmp, vecTmp; uint32_t total; };

static GemvBufLayout ComputeBufferLayout(GemvType type, uint32_t mRow, uint32_t nCol)
{
    auto round256 = [](uint32_t x) { return (x + 255u) & ~255u; };
    GemvBufCfg c = (type == GemvType::HSH) ? kGmvCfg_HSH
                 : (type == GemvType::HSS) ? kGmvCfg_HSS
                 : (type == GemvType::TST) ? kGmvCfg_TST
                 : (type == GemvType::TSS) ? kGmvCfg_TSS : kGmvCfg_F32;
    uint32_t vecE = (nCol < 64u) ? 64u : nCol;
    GemvBufLayout lay = {};
    // issue #495: 转置 RegBase 点积路径每行按整 VL（64 个 FP32 = 256B）加载且
    // load 本身不带掩码，尾行读取会越过 inA 子缓冲区。这里为 inA 末尾预留一个
    // VL 的 staging 余量，保证 load 范围始终落在已分配的子缓冲区内。
    lay.inA    = round256(GEMV_BATCHED_BN_NORMAL * mRow * nCol * c.szIn + GEMV_INA_VL_PAD_BYTES);
    lay.inx    = round256(GEMV_BATCHED_BN_NORMAL * vecE * c.szIn);
    lay.inY    = round256(GEMV_BATCHED_BN_NORMAL * mRow * c.szY);
    lay.out    = lay.inY;
    lay.matTmp = round256(c.tn * mRow * nCol * sizeof(float));
    lay.vecTmp = round256(2u * mRow * sizeof(float));
    if (c.hasExtra) {
        lay.vecTmp = round256(2u * mRow * sizeof(float) + nCol * sizeof(float));
        lay.matTmp = round256(lay.matTmp); // already rounded, no-op but explicit
        lay.total  = lay.inA + lay.inx + lay.inY + lay.out + lay.matTmp + lay.vecTmp
                   + round256(mRow * nCol * sizeof(uint16_t))   // matPreBuf
                   + round256(vecE * sizeof(uint16_t));          // vecPreBuf
        return lay;
    }
    lay.total = lay.inA + lay.inx + lay.inY + lay.out + lay.matTmp + lay.vecTmp;
    return lay;
}

static inline uint32_t ComputeUBPerBatch(GemvType type, uint32_t mRow, uint32_t nCol)
    { return ComputeBufferLayout(type, mRow, nCol).total; }

// Helpers
static inline uint32_t AlignDown(uint32_t x, uint32_t a) { return (a == 0) ? 0 : (x - (x % a)); }

// Binary-search max values of `arg` (row count or col count) s.t. f(arg) <= UB.
// `f` takes a single uint32_t and returns total bytes per batch.
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

// Transpose AIV tiling: 优先切点积轴（m）→dotTile，再切输出轴（n）→outTile
static void CalTilingTranspose(GemvType type, uint32_t dotDim, uint32_t outDim,
                                uint32_t &batchGroupSize, uint32_t &dotTile, uint32_t &outTile)
{
    const uint32_t UB = GEMV_BATCHED_UBUF_SIZE;
    uint32_t elemSize = (type == GemvType::S) ? sizeof(float) : sizeof(uint16_t);
    uint32_t align = 32u / elemSize, outMin = 1;
    auto ubFunc = [type](uint32_t r, uint32_t c) { return ComputeUBPerBatch(type, r, c); };
    uint32_t dotAligned = AlignDown(dotDim + align - 1, align);
    uint32_t outFitForFullDot = BinaryMaxRowsCols(UB, outDim > 0 ? outDim : 65536u,
        [&](uint32_t r) { return ubFunc(r, dotAligned); });
    if (outFitForFullDot >= outMin) {
        dotTile = dotAligned;
    } else {
        uint32_t dotTileMax = BinaryMaxRowsCols(UB, 65536u,
            [&](uint32_t c) { return ubFunc(outMin, c); });
        dotTile = AlignDown((dotTileMax < align) ? align : dotTileMax, align);
        if (dotTile < align) dotTile = align;
        if (dotTile > dotAligned) dotTile = dotAligned;
        outFitForFullDot = BinaryMaxRowsCols(UB, outDim > 0 ? outDim : 65536u,
            [&](uint32_t r) { return ubFunc(r, dotTile); });
    }
    uint32_t outFit = AlignDown(outFitForFullDot, outMin);
    if (outFit < outMin) outFit = outMin;
    outTile = (outDim < outFit) ? outDim : outFit;
    uint32_t perData = ubFunc(outTile, dotTile);
    batchGroupSize = (perData > 0) ? (UB / perData) : 1;
    while (batchGroupSize < 1 && dotTile >= align + align) {
        dotTile -= align;
        perData = ubFunc(outTile, dotTile);
        if (perData > 0) batchGroupSize = UB / perData; else break;
    }
    if (batchGroupSize < 1) batchGroupSize = 1;
}

static void CalTilingParams(bool isTrans, bool isFp32, uint32_t m, uint32_t n,
                             uint32_t &batchGroupSize, uint32_t &dotTile, uint32_t &outTile)
{
    if (!isTrans) {
        // Normal: SIMT, no AIV tiling
        dotTile = 0; outTile = m; batchGroupSize = 1;
        return;
    }

    // Transpose (column-major): AIV tiling，优先切点积轴（m）→dotTile，再切输出轴（n）→outTile
    CalTilingTranspose(isFp32 ? GemvType::S : GemvType::HSH, m, n, batchGroupSize, dotTile, outTile);
}

// HSS tiling (half A/x, float y)
static void CalTilingParamsHss(bool isTrans, uint32_t m, uint32_t n,
                                uint32_t &batchGroupSize, uint32_t &dotTile, uint32_t &outTile)
{
    if (!isTrans) { dotTile = 0; outTile = m; batchGroupSize = 1; return; }
    CalTilingTranspose(GemvType::HSS, m, n, batchGroupSize, dotTile, outTile);
}

static void LogTilingParams(const GemvBatchedTilingData &td)
{
    uint32_t ubTotal = td.bufInA + td.bufInx + td.bufInY + td.bufOut + td.bufMatTmp + td.bufVecTmp;
    OP_LOGD("aclblasGemvBatched",
            "tiling: dtype=%u trans=%u alpha=%.3f beta=%.3f m=%u n=%u lda=%d incx=%d incy=%d "
            "outSize=%u dotSize=%u outTile=%u dotTile=%u batchCount=%u batchGroupSize=%u "
            "coreNum=%u usedCoreNum=%u batchPerCore=%u batchTail=%u "
            "ub[inA=%u inx=%u inY=%u out=%u mat=%u vec=%u total=%u (%.1f%%)]",
            td.dtype, td.trans, td.alpha, td.beta, td.m, td.n, td.lda, td.incx, td.incy,
            td.outSize, td.dotSize, td.outTile, td.dotTile, td.batchCount, td.batchGroupSize,
            td.coreNum, td.usedCoreNum, td.batchPerCore, td.batchTail,
            td.bufInA, td.bufInx, td.bufInY, td.bufOut, td.bufMatTmp, td.bufVecTmp,
            ubTotal, 100.0f * ubTotal / GEMV_BATCHED_UBUF_SIZE);
}

static GemvBatchedTilingData CalTilingData(uint32_t batchCount, uint32_t m, uint32_t n,
                                            uint32_t dtype, uint32_t trans,
                                            float alpha, float beta,
                                            uint32_t coreNum, uint32_t usedCoreNum,
                                            uint32_t maxPerCore,
                                            int32_t lda, int32_t incx, int32_t incy)
{
    GemvBatchedTilingData td = {};
    td.dtype = dtype;  td.trans = trans;
    td.alpha = alpha;  td.beta  = beta;
    td.m = m;          td.n = n;

    uint32_t batchGroupSize = 0, dotTile = 0, outTile = 0;
    if (dtype == 2 || dtype == 4)  // HSS/TSS: mixed in/out
        CalTilingParamsHss(trans != 0, m, n, batchGroupSize, dotTile, outTile);
    else if (dtype == 0 || dtype == 3)  // HSH/TST: same in/out
        CalTilingParams(trans != 0, false, m, n, batchGroupSize, dotTile, outTile);
    else
        CalTilingParams(trans != 0, true, m, n, batchGroupSize, dotTile, outTile);

    td.batchGroupSize = batchGroupSize;  td.dotTile = dotTile;  td.outTile = outTile;

    auto bufType = (dtype == 2) ? GemvType::HSS
                 : (dtype == 3) ? GemvType::TST
                 : (dtype == 4) ? GemvType::TSS
                 : (dtype != 0) ? GemvType::S : GemvType::HSH;
    auto lay = ComputeBufferLayout(bufType, outTile, dotTile);
    td.bufInA = lay.inA;  td.bufInx = lay.inx;  td.bufInY = lay.inY;
    td.bufOut = lay.out;  td.bufMatTmp = lay.matTmp;  td.bufVecTmp = lay.vecTmp;
    td.lda = (int32_t)lda;  td.incx = (int32_t)incx;  td.incy = (int32_t)incy;

    // 操作维度：Normal-输出=m点积=n，Transpose-输出=n点积=m
    td.outSize = (trans != 0) ? n : m;
    td.dotSize = (trans != 0) ? m : n;

    td.coreNum = coreNum;  td.usedCoreNum = usedCoreNum;
    td.batchCount = batchCount;  td.batchPerCore = maxPerCore;
    td.batchTail = batchCount - (usedCoreNum - 1) * maxPerCore;

    LogTilingParams(td);
    return td;
}


template <typename T_IN, typename T_OUT>
static aclblasStatus_t GemvBatchedImpl(aclblasHandle_t handle, aclblasOperation_t trans,
                                       int m, int n,
                                       const float *alpha, const T_IN *const Aarray[], int lda,
                                       const T_IN *const xarray[], int incx,
                                       const float *beta, T_OUT *const yarray[], int incy,
                                       int batchCount, uint32_t dtype)
{
    if (handle == nullptr)  return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (m <= 0 || n <= 0)  return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < (m > 0 ? m : 1)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0 || batchCount <= 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (Aarray == nullptr || xarray == nullptr || yarray == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr)            return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T) return ACLBLAS_STATUS_INVALID_ENUM;
    auto* h = handle;
    aclrtStream useStream = h->stream;
    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasGemvBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t maxPerCore  = (batchCount + coreNum - 1) / coreNum;
    uint32_t usedCoreNum = (maxPerCore > 0) ? ((batchCount + maxPerCore - 1) / maxPerCore) : 0;
    uint32_t transUint   = (trans == ACLBLAS_OP_N) ? 0 : 1;
    GemvBatchedTilingData tiling = CalTilingData((uint32_t)batchCount, (uint32_t)m, (uint32_t)n,
        dtype, transUint, *alpha, *beta, coreNum, usedCoreNum, maxPerCore,
        (int32_t)lda, (int32_t)incx, (int32_t)incy);

    gemv_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<T_IN**>(Aarray)),
        reinterpret_cast<uint8_t*>(const_cast<T_IN**>(xarray)),
        reinterpret_cast<uint8_t*>(const_cast<T_OUT**>(yarray)),
        tiling, usedCoreNum, useStream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans,
                                     int m, int n,
                                     const float *alpha, const float *const Aarray[], int lda,
                                     const float *const xarray[], int incx,
                                     const float *beta, float *const yarray[], int incy,
                                     int batchCount)
{
    return GemvBatchedImpl<float, float>(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 1);
}

aclblasStatus_t aclblasHSHgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans,
                                       int m, int n,
                                       const float *alpha, const uint16_t *const Aarray[], int lda,
                                       const uint16_t *const xarray[], int incx,
                                       const float *beta, uint16_t *const yarray[], int incy,
                                       int batchCount)
{
    return GemvBatchedImpl<uint16_t, uint16_t>(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 0);
}

aclblasStatus_t aclblasHSSgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans,
                                       int m, int n,
                                       const float *alpha, const uint16_t *const Aarray[], int lda,
                                       const uint16_t *const xarray[], int incx,
                                       const float *beta, float *const yarray[], int incy,
                                       int batchCount)
{
    return GemvBatchedImpl<uint16_t, float>(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 2);
}

aclblasStatus_t aclblasTSTgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans,
                                       int m, int n,
                                       const float *alpha, const uint16_t *const Aarray[], int lda,
                                       const uint16_t *const xarray[], int incx,
                                       const float *beta, uint16_t *const yarray[], int incy,
                                       int batchCount)
{
    return GemvBatchedImpl<uint16_t, uint16_t>(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 3);
}

aclblasStatus_t aclblasTSSgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans,
                                       int m, int n,
                                       const float *alpha, const uint16_t *const Aarray[], int lda,
                                       const uint16_t *const xarray[], int incx,
                                       const float *beta, float *const yarray[], int incy,
                                       int batchCount)
{
    return GemvBatchedImpl<uint16_t, float>(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 4);
}
