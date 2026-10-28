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
 * \file csyr2_tiling.h
 * \brief Host/Kernel 共用 tiling。列轮转由 device 用 blockIdx + usedCoreNum 计算。
 */
#pragma once

#ifndef CSYR2_TILING_H
#define CSYR2_TILING_H

#include <cstdint>
#include "cann_ops_blas_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* BUFFER_NUM 恒为 2。xCost/yCost/tmpCost 与占用式分母从 design §3.2.1 第 6 节抄录。 */
static const uint32_t CSYR2_BUFFER_NUM = 2;
static const uint32_t CSYR2_X_COST = 8;
static const uint32_t CSYR2_Y_COST = 8;
static const uint32_t CSYR2_TMP_COST = 16;
/* device 实际每 tileRow：aIn 16 + aOut 16 + aPlane 8 + xyPlane 16 + xyStage 8 + tmp 16 + idx 8 = 88B */
static const uint32_t CSYR2_UB_BYTES_PER_TILEROW = 88;
/* InitBuffer 的固定增量：tiling 64 + xyPlane 256 + xyStage 512 + tmp 768。 */
static const uint32_t CSYR2_UB_FIXED_BYTES = 1856;
/* 窗口路径不分配 aPlane/xyStage：tiling 64 + xyPlane 256 + tmp 768。 */
static const uint32_t CSYR2_WINDOW_UB_FIXED_BYTES = 1344;
static const uint32_t CSYR2_OVERHEAD_BYTES = CSYR2_UB_FIXED_BYTES;
static const uint32_t CSYR2_ALIGN_ELEMS = 4; /* 32B / 8B */
/* usedCoreNum 上限。列轮转在 device 用 blockIdx + usedCoreNum 计算。 */
#define CSYR2_MAX_CORES 64

/*! tilingKey：0/1 为 UPPER/LOWER 单位正步长，2/3 为非单位正步长，4/5 为单位正步长窗口路径。 */
static const uint32_t CSYR2_TILING_KEY_UPPER_UNIT = 0;
static const uint32_t CSYR2_TILING_KEY_LOWER_UNIT = 1;
static const uint32_t CSYR2_TILING_KEY_UPPER_STRIDED = 2;
static const uint32_t CSYR2_TILING_KEY_LOWER_STRIDED = 3;
static const uint32_t CSYR2_TILING_KEY_UPPER_WINDOWED = 4;
static const uint32_t CSYR2_TILING_KEY_LOWER_WINDOWED = 5;

typedef struct Csyr2TilingData {
    int32_t n;
    int32_t lda;
    int32_t incx;
    int32_t incy;
    // Keep the 64-byte tiling layout; vector starts are derived in 64 bits.
    int32_t reservedOffsets[2];
    int32_t uplo;
    float alphaReal;
    float alphaImag;
    uint32_t usedCoreNum;
    uint32_t tileRow;
    uint32_t bufferNum;
    uint32_t xyReuse;
    uint32_t reserved[3];
} Csyr2TilingData;

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus

/* Pure tiling helpers are shared by host and device callers. */
#if defined(__CCE__) || defined(__CCE_AICORE__)
#define CSYR2_HOST_DEVICE __attribute__((cce_aicore, cce_host)) inline
#else
#define CSYR2_HOST_DEVICE inline
#endif
/* 核列区间由 n / uplo / usedCoreNum / GetBlockIdx() 闭式计算，不另存均匀列切字段。 */

CSYR2_HOST_DEVICE int64_t Csyr2Kx0(int32_t n, int32_t incx)
{
    return (incx > 0) ? 0 : ((1LL - n) * incx);
}

CSYR2_HOST_DEVICE int64_t Csyr2Ky0(int32_t n, int32_t incy) { return Csyr2Kx0(n, incy); }

CSYR2_HOST_DEVICE int64_t Csyr2VectorOffset(int32_t n, int32_t inc, int32_t index)
{
    return Csyr2Kx0(n, inc) + static_cast<int64_t>(index) * inc;
}

// Shared by host dispatch and device allocation; retain the existing window layout.
CSYR2_HOST_DEVICE int32_t Csyr2SelectWindowBatchCount(
    int32_t n, int32_t tileRow, uint32_t ilvBytes, uint32_t planeBytes, bool upper)
{
    if (tileRow >= n) {
        const int32_t bTop = upper ? 2 : 4;
        for (int32_t candB = bTop; candB >= 2; candB >>= 1) {
            const uint32_t total =
                CSYR2_WINDOW_UB_FIXED_BYTES + ilvBytes * (5u + 2u * static_cast<uint32_t>(candB)) + planeBytes * 2u;
            if (total <= 192u * 1024u) {
                return candB;
            }
        }
    }
    return 1;
}

inline int32_t Csyr2AlignDown4(int32_t v)
{
    if (v < 0) {
        return 0;
    }
    return (v / static_cast<int32_t>(CSYR2_ALIGN_ELEMS)) * static_cast<int32_t>(CSYR2_ALIGN_ELEMS);
}

CSYR2_HOST_DEVICE int64_t Csyr2PrefixUpper(int32_t j)
{
    /* S(j) = j*(j+1)/2，列 0..j-1 累计。 */
    const int64_t jj = static_cast<int64_t>(j);
    return jj * (jj + 1) / 2;
}

CSYR2_HOST_DEVICE int64_t Csyr2PrefixLower(int32_t j, int32_t n)
{
    /* S(j) = j*(2n-j+1)/2 */
    const int64_t jj = static_cast<int64_t>(j);
    const int64_t nn = static_cast<int64_t>(n);
    return jj * (2 * nn - jj + 1) / 2;
}

/*!
 * min { j | S(j+1) > T }，缺省/夹紧到 [0, n]。
 * UPPER 反函数初值 (-1+sqrt(1+8T))/2，LOWER 同理二次根，再校正到满足上式的整数，
 * 保证连续列段、核间无重叠无空洞。
 */
CSYR2_HOST_DEVICE int32_t Csyr2InvertPrefixGt(int32_t n, int32_t uplo, int64_t t)
{
    if (n <= 0) {
        return 0;
    }
    if (t < 0) {
        return 0;
    }
    int32_t lo = 0;
    int32_t hi = n;
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        const int64_t sNext = (uplo == ACLBLAS_UPPER) ? Csyr2PrefixUpper(mid + 1) : Csyr2PrefixLower(mid + 1, n);
        if (sNext > t) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

CSYR2_HOST_DEVICE void Csyr2ColumnRange(
    int32_t n, int32_t uplo, uint32_t usedCoreNum, uint32_t blockIdx, int32_t* jBegin, int32_t* jEnd)
{
    if (jBegin == nullptr || jEnd == nullptr) {
        return;
    }
    if (n <= 0 || usedCoreNum == 0 || blockIdx >= usedCoreNum) {
        *jBegin = n;
        *jEnd = n;
        return;
    }
    const int64_t total = static_cast<int64_t>(n) * (static_cast<int64_t>(n) + 1) / 2;
    const int64_t t0 = static_cast<int64_t>(blockIdx) * total / static_cast<int64_t>(usedCoreNum);
    const int64_t t1 = static_cast<int64_t>(blockIdx + 1) * total / static_cast<int64_t>(usedCoreNum);
    /* jBegin = min { j | S(j+1) > T0 }；非末核 jEnd 用同一反函数作用于 T1，末核为 n。 */
    *jBegin = Csyr2InvertPrefixGt(n, uplo, t0);
    if (blockIdx + 1 >= usedCoreNum) {
        *jEnd = n;
    } else {
        *jEnd = Csyr2InvertPrefixGt(n, uplo, t1);
    }
}

inline int32_t Csyr2ComputeTileRow(uint64_t ubBytes, uint32_t overhead, int32_t n)
{
    const int64_t denom = static_cast<int64_t>(CSYR2_UB_BYTES_PER_TILEROW);
    int64_t numer = static_cast<int64_t>(ubBytes) - static_cast<int64_t>(overhead);
    const auto fits = [ubBytes](int32_t rows) {
        constexpr uint64_t kVecFloats = 64;
        const uint64_t ilvFloats = ((static_cast<uint64_t>(rows) * 2 + kVecFloats - 1) / kVecFloats) * kVecFloats;
        const uint64_t planeFloats = ((static_cast<uint64_t>(rows) + kVecFloats - 1) / kVecFloats) * kVecFloats;
        const uint64_t ilvBytes = ilvFloats * sizeof(float);
        const uint64_t planeBytes = planeFloats * sizeof(float);
        const uint64_t tmpFloats = (ilvFloats * 2 > planeFloats * 3) ? ilvFloats * 2 : planeFloats * 3;
        const uint64_t occupied =
            static_cast<uint64_t>(CSYR2_UB_FIXED_BYTES) + ilvBytes * 6u + planeBytes * 6u + tmpFloats * sizeof(float);
        return occupied <= ubBytes;
    };
    if (numer < 0) {
        numer = 0;
    }
    int64_t raw = (denom > 0) ? (numer / denom) : 0;
    if (raw > 0x7fffffffLL) {
        raw = 0x7fffffffLL;
    }
    int32_t tileRow = Csyr2AlignDown4(static_cast<int32_t>(raw));
    if (n > 0 && tileRow > n) {
        tileRow = n;
    }
    while (tileRow > 1 && !fits(tileRow)) {
        const int32_t aligned = Csyr2AlignDown4(tileRow - 1);
        tileRow = (aligned > 0) ? aligned : (tileRow - 1);
    }
    if (!fits(tileRow)) {
        return 1;
    }
    if (tileRow == 0) {
        tileRow = 1;
    }
    return tileRow;
}

inline uint32_t Csyr2ComputeXyReuse(
    uint64_t ubBytes, uint32_t overhead, int32_t n, int32_t uplo, uint32_t usedCoreNum, int32_t tileRow)
{
    int32_t worstUnion = 0;
    for (uint32_t c = 0; c < usedCoreNum; ++c) {
        int32_t jBegin = 0;
        int32_t jEnd = 0;
        Csyr2ColumnRange(n, uplo, usedCoreNum, c, &jBegin, &jEnd);
        const int32_t unionLen = (uplo == ACLBLAS_UPPER) ? jEnd : (n - jBegin);
        if (unionLen > worstUnion) {
            worstUnion = unionLen;
        }
    }
    const int64_t occ = static_cast<int64_t>(overhead) +
                        static_cast<int64_t>(CSYR2_BUFFER_NUM) * static_cast<int64_t>(tileRow) * 8 +
                        static_cast<int64_t>(worstUnion) * (CSYR2_X_COST + CSYR2_Y_COST) +
                        static_cast<int64_t>(tileRow) * CSYR2_TMP_COST;
    return (occ > static_cast<int64_t>(ubBytes)) ? 0U : 1U;
}

inline uint32_t Csyr2SelectTilingKey(aclblasFillMode_t uplo, int incx, int incy)
{
    const bool unitPos = (incx == 1 && incy == 1);
    const bool upper = (uplo == ACLBLAS_UPPER);
    if (upper && unitPos) {
        return CSYR2_TILING_KEY_UPPER_UNIT;
    }
    if (!upper && unitPos) {
        return CSYR2_TILING_KEY_LOWER_UNIT;
    }
    if (upper) {
        return CSYR2_TILING_KEY_UPPER_STRIDED;
    }
    return CSYR2_TILING_KEY_LOWER_STRIDED;
}

CSYR2_HOST_DEVICE bool Csyr2NeedDataCopyPad(int32_t len)
{
    /* DataCopy 对齐写死 32B。len*8 非 32 倍数（含 len=1,2,3 与滑窗尾块）走 DataCopyPad。 */
    const int64_t bytes = static_cast<int64_t>(len) * 8;
    return (bytes % 32) != 0;
}

#if defined(__CCE_AICORE__)
#include "kernel_operator.h" /* __aicore__ 由此提供 */

/* Preserve the device helper names while sharing the pure arithmetic. */
namespace Csyr2Dev {

__aicore__ inline int64_t PrefixUpper(int32_t j) { return Csyr2PrefixUpper(j); }

__aicore__ inline int64_t PrefixLower(int32_t j, int32_t n) { return Csyr2PrefixLower(j, n); }

__aicore__ inline int32_t InvertPrefixGt(int32_t n, int32_t uplo, int64_t t) { return Csyr2InvertPrefixGt(n, uplo, t); }

__aicore__ inline void ColumnRange(
    int32_t n, int32_t uplo, uint32_t usedCoreNum, uint32_t blockIdx, int32_t* jBegin, int32_t* jEnd)
{
    Csyr2ColumnRange(n, uplo, usedCoreNum, blockIdx, jBegin, jEnd);
}

__aicore__ inline bool NeedDataCopyPad(int32_t len) { return Csyr2NeedDataCopyPad(len); }

} /* namespace Csyr2Dev */
#endif /* __CCE_AICORE__ */

#undef CSYR2_HOST_DEVICE

#endif /* __cplusplus */

#endif /* CSYR2_TILING_H */
