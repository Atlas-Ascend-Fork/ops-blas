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
 * \file ctpmv_kernel_simt.cpp
 * \brief SIMT VF kernel for single-precision complex triangular packed matrix-vector multiply.
 *
 * v3 架构（性能版）：
 *  - trans=N：列交错分区 + 行块(RB=8)寄存器累加 + P 份部分和 + reduce。
 *  - trans=T/C：行组(GRP=8)交错分 block × 列捆(CB=8)分线程 × 部分和 + reduce。
 *  workspace 布局：[P×n 部分和 | n 最终 y]；部分和为全量覆写，无需清零。
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "ctpmv_tiling_data.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

// Disable FMA contraction to match CPU BLAS results (separate mul + add).
#pragma clang fp contract(off)

__simt_callee__ inline uint64_t CtpmvColStartUpper(uint32_t j)
{
    uint64_t j64 = static_cast<uint64_t>(j);
    return j64 * (j64 + 1ULL) / 2ULL;
}

__simt_callee__ inline uint64_t CtpmvColStartLower(uint32_t j, uint32_t n)
{
    uint64_t n64 = static_cast<uint64_t>(n);
    uint64_t j64 = static_cast<uint64_t>(j);
    return j64 * (2ULL * n64 - j64 + 1ULL) / 2ULL;
}

__simt_callee__ inline uint32_t CtpmvInterleavedIdx(uint32_t k, uint32_t n)
{
    uint32_t half = k >> 1;
    if ((k & 1u) == 0u) {
        return half;
    }
    return n - 1u - half;
}

__simt_callee__ inline void CtpmvCfma(float aR, float aI, float xR, float xI, float& sumR, float& sumI)
{
    float pr = aR * xR - aI * xI;
    float pi = aR * xI + aI * xR;
    sumR = sumR + pr;
    sumI = sumI + pi;
}

// Diagonal element: UNIT diag contributes x itself, otherwise a full complex fma.
template <bool UNIT>
__simt_callee__ inline void CtpmvCfmaDiag(float aR, float aI, float xR, float xI, float& sumR, float& sumI)
{
    if constexpr (UNIT) {
        sumR = sumR + xR;
        sumI = sumI + xI;
    } else {
        CtpmvCfma(aR, aI, xR, xI, sumR, sumI);
    }
}

// T/C path tail columns (n % CB): one thread per tail column, row-group strided partial sums.
template <bool UPPER, bool CONJ, bool UNIT, uint32_t CB>
__simt_callee__ __aicore__ inline void CtpmvTcTail(
    uint32_t n, uint32_t parts, uint32_t bid, uint32_t fullBundles, uint32_t tailCount, uint32_t numGroups,
    uint32_t grp, uint64_t absIncx, int64_t incx, __gm__ const float* apGm, __gm__ const float* xGm,
    __gm__ float* yPart)
{
    for (uint32_t t = threadIdx.x; t < tailCount; t += blockDim.x) {
        const uint32_t j = fullBundles * CB + t;
        uint64_t colStart = UPPER ? CtpmvColStartUpper(j) : CtpmvColStartLower(j, n);
        float sumR = 0.0f;
        float sumI = 0.0f;
        for (uint32_t g = bid; g < numGroups; g += parts) {
            const uint32_t gr0 = g * grp;
            const uint32_t gr1 = (gr0 + grp < n) ? (gr0 + grp) : n;
            for (uint32_t r = gr0; r < gr1; ++r) {
                // UPPER 取 r<=j, LOWER 取 r>=j(对角单列); 编译期方向, 单次比较
                if (r != j && ((r > j) == UPPER)) {
                    continue;
                }
                uint64_t xOff =
                    (incx >= 0) ? (static_cast<uint64_t>(r) * absIncx) : (static_cast<uint64_t>(n - 1u - r) * absIncx);
                float aR;
                float aI;
                if constexpr (UPPER) {
                    aR = apGm[2ULL * (colStart + r)];
                    aI = apGm[2ULL * (colStart + r) + 1ULL];
                } else {
                    aR = apGm[2ULL * (colStart + (r - j))];
                    aI = apGm[2ULL * (colStart + (r - j)) + 1ULL];
                }
                if constexpr (CONJ) {
                    aI = -aI;
                }
                if (r == j) {
                    CtpmvCfmaDiag<UNIT>(aR, aI, xGm[2ULL * xOff], xGm[2ULL * xOff + 1ULL], sumR, sumI);
                } else {
                    CtpmvCfma(aR, aI, xGm[2ULL * xOff], xGm[2ULL * xOff + 1ULL], sumR, sumI);
                }
            }
        }
        yPart[2ULL * static_cast<uint64_t>(j)] = sumR;
        yPart[2ULL * static_cast<uint64_t>(j) + 1ULL] = sumI;
    }
}

// T/C path one bundle (CB columns): row-group strided partial sums into accR/accI.
template <bool UPPER, bool CONJ, bool UNIT, uint32_t CB>
__simt_callee__ __aicore__ inline void CtpmvTcBundleAcc(
    uint32_t n, uint32_t parts, uint32_t bid, uint32_t numGroups, uint32_t grp, uint32_t j0, uint64_t cs0,
    uint64_t absIncx, int64_t incx, __gm__ const float* apGm, __gm__ const float* xGm, float* accR, float* accI)
{
    for (uint32_t g = bid; g < numGroups; g += parts) {
        const uint32_t gr0 = g * grp;
        const uint32_t gr1 = (gr0 + grp < n) ? (gr0 + grp) : n;
        for (uint32_t r = gr0; r < gr1; ++r) {
            uint64_t xOff =
                (incx >= 0) ? (static_cast<uint64_t>(r) * absIncx) : (static_cast<uint64_t>(n - 1u - r) * absIncx);
            float xR = xGm[2ULL * xOff];
            float xI = xGm[2ULL * xOff + 1ULL];
#pragma unroll
            for (uint32_t c = 0; c < CB; ++c) {
                const uint32_t j = j0 + c;
                // UPPER 取 r<=j, LOWER 取 r>=j(对角单列); 编译期方向, 单次比较
                if (r != j && ((r > j) == UPPER)) {
                    continue;
                }
                uint64_t off;
                if constexpr (UPPER) {
                    off = cs0 + static_cast<uint64_t>(c) * (j0 + 1u) + static_cast<uint64_t>(c) * (c - 1u) / 2u + r;
                } else {
                    off = cs0 + static_cast<uint64_t>(c) * (n - j0) - static_cast<uint64_t>(c) * (c - 1u) / 2u +
                          (r - j0 - c);
                }
                float aR = apGm[2ULL * off];
                float aI = apGm[2ULL * off + 1ULL];
                if constexpr (CONJ) {
                    aI = -aI;
                }
                if (r == j) {
                    CtpmvCfmaDiag<UNIT>(aR, aI, xR, xI, accR[c], accI[c]);
                } else {
                    CtpmvCfma(aR, aI, xR, xI, accR[c], accI[c]);
                }
            }
        }
    }
}

template <bool UPPER, int TRANS_MODE, bool UNIT, uint32_t CB = 8u>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvTcRowChunk(
    uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ const float* xGm, __gm__ float* ypGm)
{
    const uint32_t parts = gridDim.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t fullBundles = n / CB;
    const uint32_t tailCount = n - fullBundles * CB;
    uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : (static_cast<uint64_t>(~incx) + 1ULL);
    constexpr bool CONJ = (TRANS_MODE == 2);
    __gm__ float* yPart = ypGm + 2ULL * static_cast<uint64_t>(bid) * n;

    constexpr uint32_t GRP = 8u;
    const uint32_t numGroups = (n + GRP - 1u) / GRP;

    for (uint32_t kk = threadIdx.x; kk < fullBundles; kk += blockDim.x) {
        const uint32_t j0 = CtpmvInterleavedIdx(kk, fullBundles) * CB;
        const uint64_t cs0 = UPPER ? CtpmvColStartUpper(j0) : CtpmvColStartLower(j0, n);

        float accR[CB];
        float accI[CB];
#pragma unroll
        for (uint32_t c = 0; c < CB; ++c) {
            accR[c] = 0.0f;
            accI[c] = 0.0f;
        }

        CtpmvTcBundleAcc<UPPER, CONJ, UNIT, CB>(
            n, parts, bid, numGroups, GRP, j0, cs0, absIncx, incx, apGm, xGm, accR, accI);

#pragma unroll
        for (uint32_t c = 0; c < CB; ++c) {
            yPart[2ULL * static_cast<uint64_t>(j0 + c)] = accR[c];
            yPart[2ULL * static_cast<uint64_t>(j0 + c) + 1ULL] = accI[c];
        }
    }

    CtpmvTcTail<UPPER, CONJ, UNIT, CB>(
        n, parts, bid, fullBundles, tailCount, numGroups, GRP, absIncx, incx, apGm, xGm, yPart);
}

// v5.3 试 RB=16 判负: 950 实测 N kernel 慢 ~35%(case1 42.7→58.4us, 寄存器压力),
// 累加器深度 8 是甜点——回退保留注释存证
constexpr uint32_t CTPMV_RB = 8;

// Return this column's start before advancing the per-row-block recurrence.
template <bool UPPER>
__simt_callee__ inline uint64_t CtpmvNColumnStart(
    uint32_t j, uint32_t n, uint32_t parts, uint64_t& colStart, bool& csValid)
{
    if (!csValid) {
        colStart = UPPER ? CtpmvColStartUpper(j) : CtpmvColStartLower(j, n);
        csValid = true;
    }
    const uint64_t csThis = colStart;
    if (j + parts < n) {
        if constexpr (UPPER) {
            colStart += static_cast<uint64_t>(parts) * j + static_cast<uint64_t>(parts) * (parts + 1u) / 2u;
        } else {
            const uint64_t delta = static_cast<uint64_t>(parts) * (2ULL * n + 1ULL) - 2ULL * parts * j -
                                   static_cast<uint64_t>(parts) * parts;
            colStart += delta / 2ULL;
        }
    }
    return csThis;
}

__simt_callee__ inline uint64_t CtpmvNXOffset(uint32_t j, uint32_t n, int64_t incx, uint64_t absIncx)
{
    const uint32_t logical = (incx >= 0) ? j : (n - 1u - j);
    return static_cast<uint64_t>(logical) * absIncx;
}

template <bool UPPER, bool UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvNRowBlock(
    uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ const float* xGm, __gm__ float* ypGm)
{
    const uint32_t parts = gridDim.x;
    const uint32_t bid = blockIdx.x;
    uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : (static_cast<uint64_t>(~incx) + 1ULL);

    const uint32_t numRowBlocks = (n + CTPMV_RB - 1u) / CTPMV_RB;
    __gm__ float* yPart = ypGm + 2ULL * static_cast<uint64_t>(bid) * n;

    for (uint32_t rb = threadIdx.x; rb < numRowBlocks; rb += blockDim.x) {
        const uint32_t r0 = rb * CTPMV_RB;
        const uint32_t r1 = (r0 + CTPMV_RB < n) ? (r0 + CTPMV_RB) : n;

        float accR[CTPMV_RB] = {};
        float accI[CTPMV_RB] = {};

        uint64_t colStart = 0;
        bool csValid = false;
        for (uint32_t j = bid; j < n; j += parts) {
            if constexpr (UPPER) {
                if (j < r0) {
                    continue;
                }
            } else {
                if (j >= r1) {
                    break;
                }
            }
            const uint64_t xOff = CtpmvNXOffset(j, n, incx, absIncx);
            float xJR = xGm[2ULL * xOff];
            float xJI = xGm[2ULL * xOff + 1ULL];

            const uint64_t csThis = CtpmvNColumnStart<UPPER>(j, n, parts, colStart, csValid);
            const uint32_t rBegin = UPPER ? r0 : ((r0 > j) ? r0 : j);
            const uint32_t rEnd = UPPER ? ((j + 1u < r1) ? (j + 1u) : r1) : r1;
            const uint32_t rOff = UPPER ? r0 : (rBegin - j);
            const __gm__ float* pA = apGm + 2ULL * (csThis + static_cast<uint64_t>(rOff));
            for (uint32_t r = rBegin; r < rEnd; ++r, pA += 2ULL) {
                // UNIT diagonal stays in the VF and never evaluates an AP load.
                if (UNIT && r == j) {
                    accR[r - r0] = accR[r - r0] + xJR;
                    accI[r - r0] = accI[r - r0] + xJI;
                } else {
                    CtpmvCfma(pA[0], pA[1], xJR, xJI, accR[r - r0], accI[r - r0]);
                }
            }
        }

        for (uint32_t r = r0; r < r1; ++r) {
            yPart[2ULL * static_cast<uint64_t>(r)] = accR[r - r0];
            yPart[2ULL * static_cast<uint64_t>(r) + 1ULL] = accI[r - r0];
        }
    }
}

__simt_vf__ __aicore__
LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvReduceSimt(uint32_t n, uint32_t parts, __gm__ float* wsGm)
{
    for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        float sumR = 0.0f;
        float sumI = 0.0f;
        for (uint32_t p = 0; p < parts; ++p) {
            __gm__ const float* src = wsGm + 2ULL * static_cast<uint64_t>(p) * n + 2ULL * static_cast<uint64_t>(r);
            float pr = src[0];
            float pi = src[1];
            sumR = sumR + pr;
            sumI = sumI + pi;
        }
        __gm__ float* dst = wsGm + 2ULL * static_cast<uint64_t>(parts) * n + 2ULL * static_cast<uint64_t>(r);
        dst[0] = sumR;
        dst[1] = sumI;
    }
}

// 12 组合按 trans 拆成三个薄入口(N/T/C), uplo×diag 四分支; trans 判断在 host 侧 _do 完成。
// 模板实例与启动参数与拆分前逐字节一致——仅分发链位置变化。

__global__ __aicore__ void ctpmv_simt_n_kernel(GM_ADDR aP, GM_ADDR x, GM_ADDR y, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* apGm = reinterpret_cast<__gm__ const float*>(aP);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            asc_vf_call<CtpmvNRowBlock<false, false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        } else {
            asc_vf_call<CtpmvNRowBlock<false, true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        }
    } else {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            asc_vf_call<CtpmvNRowBlock<true, false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        } else {
            asc_vf_call<CtpmvNRowBlock<true, true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        }
    }
}

__global__ __aicore__ void ctpmv_simt_t_kernel(GM_ADDR aP, GM_ADDR x, GM_ADDR y, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* apGm = reinterpret_cast<__gm__ const float*>(aP);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            asc_vf_call<CtpmvTcRowChunk<false, 1, false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        } else {
            asc_vf_call<CtpmvTcRowChunk<false, 1, true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        }
    } else {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            asc_vf_call<CtpmvTcRowChunk<true, 1, false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        } else {
            asc_vf_call<CtpmvTcRowChunk<true, 1, true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        }
    }
}

__global__ __aicore__ void ctpmv_simt_c_kernel(GM_ADDR aP, GM_ADDR x, GM_ADDR y, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* apGm = reinterpret_cast<__gm__ const float*>(aP);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            asc_vf_call<CtpmvTcRowChunk<false, 2, false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        } else {
            asc_vf_call<CtpmvTcRowChunk<false, 2, true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        }
    } else {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            asc_vf_call<CtpmvTcRowChunk<true, 2, false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        } else {
            asc_vf_call<CtpmvTcRowChunk<true, 2, true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
        }
    }
}

__global__ __aicore__ void ctpmv_reduce_kernel(GM_ADDR ws, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* wsGm = reinterpret_cast<__gm__ float*>(ws);
    asc_vf_call<CtpmvReduceSimt>(dim3{tiling.numThreads, 1, 1}, tiling.n, AscendC::GetBlockNum(), wsGm);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvScatterSimt(
    uint32_t n, int64_t incx, __gm__ float* dstGm, __gm__ const float* srcGm)
{
    uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : (static_cast<uint64_t>(~incx) + 1ULL);
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        uint64_t dstOff =
            (incx >= 0) ? (static_cast<uint64_t>(i) * absIncx) : (static_cast<uint64_t>(n - 1u - i) * absIncx);
        dstGm[2ULL * dstOff] = srcGm[2ULL * static_cast<uint64_t>(i)];
        dstGm[2ULL * dstOff + 1ULL] = srcGm[2ULL * static_cast<uint64_t>(i) + 1ULL];
    }
}

__global__ __aicore__ void ctpmv_scatter_kernel(GM_ADDR dst, GM_ADDR src, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* dstGm = reinterpret_cast<__gm__ float*>(dst);
    auto* srcGm = reinterpret_cast<__gm__ const float*>(src);
    asc_vf_call<CtpmvScatterSimt>(dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, dstGm, srcGm);
}

void ctpmv_arch35_kernel_do(
    GM_ADDR aP, GM_ADDR x, GM_ADDR y, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    if (tiling.trans == ACLBLAS_OP_T) {
        ctpmv_simt_t_kernel<<<numBlocks, nullptr, stream>>>(aP, x, y, tiling);
    } else if (tiling.trans == ACLBLAS_OP_C) {
        ctpmv_simt_c_kernel<<<numBlocks, nullptr, stream>>>(aP, x, y, tiling);
    } else {
        ctpmv_simt_n_kernel<<<numBlocks, nullptr, stream>>>(aP, x, y, tiling);
    }
}

void ctpmv_arch35_reduce_do(GM_ADDR ws, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctpmv_reduce_kernel<<<numBlocks, nullptr, stream>>>(ws, tiling);
}

// ==========================================================================
//  v5.0 两遍分组归约：pass1 每 block 连续归约一段行(7行)到 mid[g]，pass2 归约
//  mid(8行)到 y。读模式从每元素 56 次 4KB 跳跃变为段内连续读，跳数 56→7+8。
//  主 kernel(yPart 行数 = parts) 与布局兼容：mid 8 行 + y 1 行在 parts 行之后。
// ==========================================================================

constexpr uint32_t CTPMV_REDUCE_MID_ROWS = 8;

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvReduceGroupSimt(
    uint32_t n, uint32_t rows, uint32_t rowBase, __gm__ const float* srcGm, __gm__ float* dstRowGm)
{
    // 每线程连续列段（段长 n/blockDim，段内地址连续，预取友好）
    uint32_t colsPer = (n + blockDim.x - 1u) / blockDim.x;
    if (colsPer == 0u) {
        colsPer = 1u;
    }
    uint32_t c0 = threadIdx.x * colsPer;
    uint32_t c1 = c0 + colsPer;
    if (c1 > n) {
        c1 = n;
    }
    for (uint32_t c = c0; c < c1; ++c) {
        float sR = 0.0f;
        float sI = 0.0f;
        for (uint32_t p = 0; p < rows; ++p) {
            __gm__ const float* s =
                srcGm + 2ULL * static_cast<uint64_t>(rowBase + p) * n + 2ULL * static_cast<uint64_t>(c);
            sR = sR + s[0];
            sI = sI + s[1];
        }
        dstRowGm[2ULL * static_cast<uint64_t>(c)] = sR;
        dstRowGm[2ULL * static_cast<uint64_t>(c) + 1ULL] = sI;
    }
}

__global__ __aicore__ void ctpmv_reduce_group_kernel(
    GM_ADDR src, GM_ADDR dst, CtpmvTilingData tiling, uint32_t parts, uint32_t rowsPer)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* srcGm = reinterpret_cast<__gm__ float*>(src);
    auto* dstGm = reinterpret_cast<__gm__ float*>(dst);
    uint32_t rowBase = AscendC::GetBlockIdx() * rowsPer;
    // parts<8 时高位 block 的 rowBase 可能 >= parts，无符号减法须防下溢（否则读越界行污染 mid）
    uint32_t rows = (parts > rowBase) ? (parts - rowBase) : 0u;
    if (rows > rowsPer) {
        rows = rowsPer;
    }
    asc_vf_call<CtpmvReduceGroupSimt>(
        dim3{tiling.numThreads, 1, 1}, tiling.n, rows, rowBase, srcGm,
        dstGm + 2ULL * static_cast<uint64_t>(AscendC::GetBlockIdx()) * tiling.n);
}

__simt_callee__ inline bool CtpmvColRange(uint32_t n, uint32_t& c0, uint32_t& c1)
{
    uint32_t colsPerBlock = (n + gridDim.x - 1u) / gridDim.x;
    uint32_t b0 = blockIdx.x * colsPerBlock;
    uint32_t b1 = b0 + colsPerBlock;
    if (b1 > n) {
        b1 = n;
    }
    if (b1 <= b0) {
        return false;
    }
    uint32_t colsPer = (b1 - b0 + blockDim.x - 1u) / blockDim.x;
    if (colsPer == 0u) {
        colsPer = 1u;
    }
    c0 = b0 + threadIdx.x * colsPer;
    c1 = c0 + colsPer;
    if (c1 > b1) {
        c1 = b1;
    }
    return true;
}

__simt_callee__ inline void CtpmvReduceRowMajorColumn(
    uint32_t n, uint32_t rows, uint32_t c, __gm__ const float* srcGm, float& sR, float& sI)
{
    for (uint32_t p = 0; p < rows; ++p) {
        __gm__ const float* s = srcGm + 2ULL * static_cast<uint64_t>(p) * n + 2ULL * static_cast<uint64_t>(c);
        sR = sR + s[0];
        sI = sI + s[1];
    }
}

// v5.17: nsmall2 配套短归约，行循环 4 批装载(u32 索引)。
__simt_callee__ inline void CtpmvReduceTransposedColumn4(
    uint32_t rows, uint32_t c, __gm__ const float* srcGm, float& sR, float& sI)
{
    // v5.18: 转置布局——列 c 的 rows 个部分和在 srcGm + 2*c*rows 起连续
    const __gm__ float* base = srcGm + 2ULL * static_cast<uint64_t>(c) * rows;
    uint32_t p = 0u;
    const uint32_t pMain = rows & ~3u;
    for (; p < pMain; p += 4u) {
        const __gm__ float* s0 = base + 2u * p;
        const __gm__ float* s1 = s0 + 2u;
        const __gm__ float* s2 = s0 + 4u;
        const __gm__ float* s3 = s0 + 6u;
        const float r0 = s0[0];
        const float i0 = s0[1];
        const float r1 = s1[0];
        const float i1 = s1[1];
        const float r2 = s2[0];
        const float i2 = s2[1];
        const float r3 = s3[0];
        const float i3 = s3[1];
        sR = sR + r0 + r1 + r2 + r3;
        sI = sI + i0 + i1 + i2 + i3;
    }
    for (; p < rows; ++p) {
        const __gm__ float* s = base + 2u * p;
        sR = sR + s[0];
        sI = sI + s[1];
    }
}

// v5.3: pass2 改列分割多 block——每 block 连续负责一段列, 消掉单 block 归约 8 行的串行段
// (N 路径近失 case 差 0.2-4us, 此处省 2-4us 直接翻盘)
template <bool TRANSPOSED>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvReduceColumnsSimt(
    uint32_t n, uint32_t rows, __gm__ const float* srcGm, __gm__ float* dstGm)
{
    uint32_t c0;
    uint32_t c1;
    if (!CtpmvColRange(n, c0, c1)) {
        return;
    }
    for (uint32_t c = c0; c < c1; ++c) {
        float sR = 0.0f;
        float sI = 0.0f;
        if constexpr (TRANSPOSED) {
            CtpmvReduceTransposedColumn4(rows, c, srcGm, sR, sI);
        } else {
            CtpmvReduceRowMajorColumn(n, rows, c, srcGm, sR, sI);
        }
        dstGm[2ULL * static_cast<uint64_t>(c)] = sR;
        dstGm[2ULL * static_cast<uint64_t>(c) + 1ULL] = sI;
    }
}

__global__ __aicore__ void ctpmv_reduce_final_kernel(GM_ADDR src, GM_ADDR dst, CtpmvTilingData tiling, uint32_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* srcGm = reinterpret_cast<__gm__ float*>(src);
    auto* dstGm = reinterpret_cast<__gm__ float*>(dst);
    asc_vf_call<CtpmvReduceColumnsSimt<false>>(dim3{tiling.numThreads, 1, 1}, tiling.n, rows, srcGm, dstGm);
}

// v5.6: parts>8 改调单遍 vec 归约(DMA 分块装载+向量 Add, 见 ctpmv_kernel_vec.cpp)
void ctpmv_arch35_reduceall_vec_do(
    GM_ADDR src, GM_ADDR dst, const CtpmvTilingData& tiling, uint32_t parts, void* stream);

void ctpmv_arch35_reduce2_do(
    GM_ADDR ws, GM_ADDR mid, GM_ADDR ySlot, const CtpmvTilingData& tiling, uint32_t parts, void* stream)
{
    // v5.7: n<=512 保护性走已验证的两遍 SIMT 归约(v5.6 实测 rv 在小 n 反慢, case1 曾翻车);
    // n>512 走 4 深装载批的单遍 vec 归约
    if (tiling.n <= 512u) {
        uint32_t rowsPer = (parts + CTPMV_REDUCE_MID_ROWS - 1u) / CTPMV_REDUCE_MID_ROWS;
        ctpmv_reduce_group_kernel<<<CTPMV_REDUCE_MID_ROWS, nullptr, stream>>>(ws, mid, tiling, parts, rowsPer);
        ctpmv_reduce_final_kernel<<<CTPMV_REDUCE_MID_ROWS, nullptr, stream>>>(
            mid, ySlot, tiling, CTPMV_REDUCE_MID_ROWS);
        return;
    }
    ctpmv_arch35_reduceall_vec_do(ws, ySlot, tiling, parts, stream);
}

// ==========================================================================
//  v5.5: 微小尺寸专用 kernel（trans=N, n<=8）——主 simt kernel 含 12 组合模板分发,
//  代码体积使固定成本 ~8us；朴素单代码路径(行和+Netlib 顺序)把固定成本压到 ~3us。
//  y 按 incx 无关的紧凑顺序写到 workspace 行 0, 由 host 侧 memcpy/scatter 回写。
// ==========================================================================

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvTinySimt(
    uint32_t n, uint32_t uplo, uint32_t trans, uint32_t diag, int64_t incx, __gm__ const float* apGm,
    __gm__ const float* xGm, __gm__ float* yGm)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    (void)trans; // 仅 trans=N 由 host 门保证
    const bool upper = (uplo == static_cast<uint32_t>(ACLBLAS_UPPER));
    const bool unit = (diag == static_cast<uint32_t>(ACLBLAS_UNIT));
    uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : (static_cast<uint64_t>(~incx) + 1ULL);
    for (uint32_t i = 0; i < n; ++i) {
        float sR = 0.0f;
        float sI = 0.0f;
        for (uint32_t j = 0; j < n; ++j) {
            const bool inTri = upper ? (j >= i) : (j <= i);
            if (!inTri) {
                continue;
            }
            uint64_t off;
            if (upper) {
                off = static_cast<uint64_t>(j) * (static_cast<uint64_t>(j) + 1ULL) / 2ULL + i;
            } else {
                off = static_cast<uint64_t>(j) * (2ULL * static_cast<uint64_t>(n) - static_cast<uint64_t>(j) + 1ULL) /
                          2ULL +
                      (i - j);
            }
            const float aR = apGm[2ULL * off];
            const float aI = apGm[2ULL * off + 1ULL];
            const uint64_t xOff =
                (incx >= 0) ? (static_cast<uint64_t>(j) * absIncx) : (static_cast<uint64_t>(n - 1u - j) * absIncx);
            const float xR = xGm[2ULL * xOff];
            const float xI = xGm[2ULL * xOff + 1ULL];
            if (i == j && unit) {
                sR = sR + xR;
                sI = sI + xI;
            } else {
                sR = sR + aR * xR - aI * xI;
                sI = sI + aR * xI + aI * xR;
            }
        }
        yGm[2ULL * static_cast<uint64_t>(i)] = sR;
        yGm[2ULL * static_cast<uint64_t>(i) + 1ULL] = sI;
    }
}

__global__ __aicore__ void ctpmv_tiny_kernel(GM_ADDR aP, GM_ADDR x, GM_ADDR y, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* apGm = reinterpret_cast<__gm__ const float*>(aP);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<CtpmvTinySimt>(
        dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.uplo, tiling.trans, tiling.diag, tiling.incx, apGm, xGm, yGm);
}

void ctpmv_arch35_tiny_do(GM_ADDR aP, GM_ADDR x, GM_ADDR y, const CtpmvTilingData& tiling, void* stream)
{
    ctpmv_tiny_kernel<<<1, nullptr, stream>>>(aP, x, y, tiling);
}

// ==========================================================================
//  v5.6: N 中小尺寸(8<n<=128)行独占直出——block b 独占 8 行, 线程 t 串行走 j 直接算
//  终值 y, 无部分和无归约(部分和+两遍归约的固定开销在小 n 占比过高)。
//  y 紧凑写到 workspace 行 0, host 侧 memcpy/scatter 回写。
// ==========================================================================

// v5.9: 全 u32 索引——v5.8 实测 aiv_vec_ratio=0.985, 64 位乘除打满向量 ALU(51us/64行的根因);
// n<=128 时所有偏移 < 2^17, u32 乘+移位即可
__simt_callee__ inline uint32_t NsmallPackIdx(uint32_t j, uint32_t row, uint32_t n, bool upper)
{
    if (upper) {
        return ((j * (j + 1u)) >> 1) + row;
    }
    return ((j * (2u * n - j + 1u)) >> 1) + (row - j);
}

__simt_callee__ inline uint32_t NsmallXIdx(uint32_t j, int64_t incx, uint64_t absIncx, uint32_t n)
{
    const uint32_t s = (incx >= 0) ? j : (n - 1u - j);
    return static_cast<uint32_t>(static_cast<uint64_t>(s) * absIncx);
}

struct CtpmvNSmallContext {
    uint32_t n;
    uint32_t row;
    bool upper;
    int64_t incx;
    uint64_t absIncx;
};

// IndexT preserves the original 2ULL (nsmall) / 2u (nsmall2) arithmetic width.
template <typename IndexT>
__simt_callee__ inline void CtpmvNSmallAcc4(
    const CtpmvNSmallContext& ctx, uint32_t j, __gm__ const float* apGm, __gm__ const float* xGm, float& sR, float& sI)
{
    const uint32_t o0 = NsmallPackIdx(j, ctx.row, ctx.n, ctx.upper);
    const uint32_t o1 = NsmallPackIdx(j + 1u, ctx.row, ctx.n, ctx.upper);
    const uint32_t o2 = NsmallPackIdx(j + 2u, ctx.row, ctx.n, ctx.upper);
    const uint32_t o3 = NsmallPackIdx(j + 3u, ctx.row, ctx.n, ctx.upper);
    const uint32_t x0 = NsmallXIdx(j, ctx.incx, ctx.absIncx, ctx.n);
    const uint32_t x1 = NsmallXIdx(j + 1u, ctx.incx, ctx.absIncx, ctx.n);
    const uint32_t x2 = NsmallXIdx(j + 2u, ctx.incx, ctx.absIncx, ctx.n);
    const uint32_t x3 = NsmallXIdx(j + 3u, ctx.incx, ctx.absIncx, ctx.n);
    const float a0R = apGm[IndexT{2} * o0];
    const float a0I = apGm[IndexT{2} * o0 + IndexT{1}];
    const float a1R = apGm[IndexT{2} * o1];
    const float a1I = apGm[IndexT{2} * o1 + IndexT{1}];
    const float a2R = apGm[IndexT{2} * o2];
    const float a2I = apGm[IndexT{2} * o2 + IndexT{1}];
    const float a3R = apGm[IndexT{2} * o3];
    const float a3I = apGm[IndexT{2} * o3 + IndexT{1}];
    const float x0R = xGm[IndexT{2} * x0];
    const float x0I = xGm[IndexT{2} * x0 + IndexT{1}];
    const float x1R = xGm[IndexT{2} * x1];
    const float x1I = xGm[IndexT{2} * x1 + IndexT{1}];
    const float x2R = xGm[IndexT{2} * x2];
    const float x2I = xGm[IndexT{2} * x2 + IndexT{1}];
    const float x3R = xGm[IndexT{2} * x3];
    const float x3I = xGm[IndexT{2} * x3 + IndexT{1}];
    sR = sR + a0R * x0R - a0I * x0I;
    sI = sI + a0R * x0I + a0I * x0R;
    sR = sR + a1R * x1R - a1I * x1I;
    sI = sI + a1R * x1I + a1I * x1R;
    sR = sR + a2R * x2R - a2I * x2I;
    sI = sI + a2R * x2I + a2I * x2R;
    sR = sR + a3R * x3R - a3I * x3I;
    sI = sI + a3R * x3I + a3I * x3R;
}

template <typename IndexT>
__simt_callee__ inline void CtpmvNSmallAccRange(
    const CtpmvNSmallContext& ctx, uint32_t jLo, uint32_t jHi, __gm__ const float* apGm, __gm__ const float* xGm,
    float& sR, float& sI)
{
    uint32_t j = jLo;
    const uint32_t jMain = jLo + ((jHi - jLo) & ~3u);
    for (; j < jMain; j += 4u) {
        CtpmvNSmallAcc4<IndexT>(ctx, j, apGm, xGm, sR, sI);
    }
    for (; j < jHi; ++j) {
        const uint32_t off = NsmallPackIdx(j, ctx.row, ctx.n, ctx.upper);
        const float aR = apGm[2u * off];
        const float aI = apGm[2u * off + 1u];
        const uint32_t xo = NsmallXIdx(j, ctx.incx, ctx.absIncx, ctx.n);
        const float xR = xGm[2u * xo];
        const float xI = xGm[2u * xo + 1u];
        sR = sR + aR * xR - aI * xI;
        sI = sI + aR * xI + aI * xR;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvNSmallSimt(
    uint32_t n, uint32_t uplo, uint32_t diag, int64_t incx, __gm__ const float* apGm, __gm__ const float* xGm,
    __gm__ float* yGm)
{
    const uint32_t row = blockIdx.x * 8u + threadIdx.x;
    if (threadIdx.x >= 8u || row >= n) {
        return;
    }
    const bool upper = (uplo == static_cast<uint32_t>(ACLBLAS_UPPER));
    const bool unit = (diag == static_cast<uint32_t>(ACLBLAS_UNIT));
    uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : (static_cast<uint64_t>(~incx) + 1ULL);
    // v5.8: 对角剥离——UNIT 时 x[row] 单独加, 主循环纯 FMA 无分支;
    // 4 批显式标量装载(弃 v5.7 数组版——溢出 local 反慢)
    const uint32_t jLo = upper ? (unit ? row + 1u : row) : 0u;
    const uint32_t jHi = upper ? n : (unit ? row : row + 1u);
    float sR = 0.0f;
    float sI = 0.0f;
    if (unit) {
        const uint64_t xo =
            (incx >= 0) ? (static_cast<uint64_t>(row) * absIncx) : (static_cast<uint64_t>(n - 1u - row) * absIncx);
        sR = xGm[2ULL * xo];
        sI = xGm[2ULL * xo + 1ULL];
    }
    const CtpmvNSmallContext ctx{n, row, upper, incx, absIncx};
    CtpmvNSmallAccRange<decltype(2ULL)>(ctx, jLo, jHi, apGm, xGm, sR, sI);
    yGm[2ULL * static_cast<uint64_t>(row)] = sR;
    yGm[2ULL * static_cast<uint64_t>(row) + 1ULL] = sI;
}

__global__ __aicore__ void ctpmv_nsmall_kernel(GM_ADDR aP, GM_ADDR x, GM_ADDR y, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* apGm = reinterpret_cast<__gm__ const float*>(aP);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<CtpmvNSmallSimt>(
        dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.uplo, tiling.diag, tiling.incx, apGm, xGm, yGm);
}

void ctpmv_arch35_nsmall_do(GM_ADDR aP, GM_ADDR x, GM_ADDR y, const CtpmvTilingData& tiling, void* stream)
{
    const uint32_t blocks = (tiling.n + 7u) / 8u;
    ctpmv_nsmall_kernel<<<blocks, nullptr, stream>>>(aP, x, y, tiling);
}

// ==========================================================================
//  v5.12: N-AXPY 残余行预置 kernel(preswap 模式)——槽位装载只覆盖 8 对齐段,
//  残余行(UPPER 列尾 / LOWER 头尾)在此以纯 GM 标量累加进 gap 数组(宿主已清零),
//  naxpy kernel 末尾 SoA 拆装+向量加。跨 kernel 边界保序, 零标量 UB 写。
// ==========================================================================

__simt_callee__ inline uint64_t NPRepLowerIdx(uint32_t r, uint32_t c, uint32_t n)
{
    const uint64_t jj = c;
    const uint64_t nn = n;
    return jj * (2ULL * nn - jj + 1ULL) / 2ULL + (r - jj);
}

__simt_callee__ inline uint64_t NPRepUpperIdx(uint32_t r, uint32_t c)
{
    const uint64_t jj = c;
    return jj * (jj + 1ULL) / 2ULL + r;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvNPrepSimt(
    uint32_t n, uint32_t uplo, uint32_t diag, __gm__ const float* apGm, __gm__ const float* xGm, __gm__ float* gapGm)
{
    if (blockIdx.x != 0u) {
        return;
    }
    const uint32_t c = threadIdx.x;
    if (c >= n) {
        return;
    }
    const bool unit = (diag == static_cast<uint32_t>(ACLBLAS_UNIT));
    const float xR = xGm[2ULL * c];
    const float xI = xGm[2ULL * c + 1ULL];
    const bool upper = (uplo == static_cast<uint32_t>(ACLBLAS_UPPER));
    if (upper) {
        const uint32_t dR = unit ? c : (c + 1u);
        const uint32_t a8 = dR & ~7u;
        for (uint32_t r = a8; r < dR && r < n; ++r) {
            const uint64_t off = NPRepUpperIdx(r, c);
            const float aR = apGm[2ULL * off];
            const float aI = apGm[2ULL * off + 1ULL];
            gapGm[2ULL * r] = gapGm[2ULL * r] + aR * xR - aI * xI;
            gapGm[2ULL * r + 1ULL] = gapGm[2ULL * r + 1ULL] + aR * xI + aI * xR;
        }
    } else {
        const uint32_t startRow = unit ? (c + 1u) : c;
        const uint32_t ds = (startRow + 7u) & ~7u;
        for (uint32_t r = startRow; r < ds && r < n; ++r) {
            const uint64_t off = NPRepLowerIdx(r, c, n);
            const float aR = apGm[2ULL * off];
            const float aI = apGm[2ULL * off + 1ULL];
            gapGm[2ULL * r] = gapGm[2ULL * r] + aR * xR - aI * xI;
            gapGm[2ULL * r + 1ULL] = gapGm[2ULL * r + 1ULL] + aR * xI + aI * xR;
        }
        if (ds < n) {
            const uint32_t tailCnt = (n - ds) & ~7u;
            const uint32_t t8 = ds + tailCnt;
            for (uint32_t r = t8; r < n; ++r) {
                const uint64_t off = NPRepLowerIdx(r, c, n);
                const float aR = apGm[2ULL * off];
                const float aI = apGm[2ULL * off + 1ULL];
                gapGm[2ULL * r] = gapGm[2ULL * r] + aR * xR - aI * xI;
                gapGm[2ULL * r + 1ULL] = gapGm[2ULL * r + 1ULL] + aR * xI + aI * xR;
            }
        }
    }
}

__global__ __aicore__ void ctpmv_ngap_kernel(GM_ADDR aP, GM_ADDR x, GM_ADDR gap, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* apGm = reinterpret_cast<__gm__ const float*>(aP);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* gapGm = reinterpret_cast<__gm__ float*>(gap);
    asc_vf_call<CtpmvNPrepSimt>(dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.uplo, tiling.diag, apGm, xGm, gapGm);
}

void ctpmv_arch35_ngap_do(GM_ADDR aP, GM_ADDR x, GM_ADDR gap, const CtpmvTilingData& tiling, void* stream)
{
    ctpmv_ngap_kernel<<<1, nullptr, stream>>>(aP, x, gap, tiling);
}

// ==========================================================================
//  v5.15: N 中尺寸(33<=n<=128)分行分块——8 块各认领每行的一段 j 列, 部分和写
//  GM ws[8][n], 再由已验证的 reduce_final 归约 8 行出 y。nsmall 单线程整行
//  的串行装载链是 n=64/128 超 2.2x 的根因(行63 要串行走 63 项 x ~500ns)。
//  全程纯 GM 标量, 不涉 strided 装载与标量 UB 写。
// ==========================================================================

constexpr uint32_t CTPMV_NSMALL2_CHUNKS = 32; // v5.18: 16->32, 每线程装载压到2项(每线程GM标量装载~500ns串行是硬墙)

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvNSmall2Simt(
    uint32_t n, uint32_t uplo, uint32_t diag, int64_t incx, __gm__ const float* apGm, __gm__ const float* xGm,
    __gm__ float* wsGm)
{
    const uint32_t row = threadIdx.x; // 单 block 线程宽覆盖 n<=128
    if (row >= n) {
        return;
    }
    const uint32_t chunk = blockIdx.x;
    const bool upper = (uplo == static_cast<uint32_t>(ACLBLAS_UPPER));
    const bool unit = (diag == static_cast<uint32_t>(ACLBLAS_UNIT));
    uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : (static_cast<uint64_t>(~incx) + 1ULL);
    const uint32_t jLo = upper ? (unit ? row + 1u : row) : 0u;
    const uint32_t jHi = upper ? n : (unit ? row : row + 1u);
    const uint32_t total = jHi - jLo;
    const uint32_t chunkLen = (total + CTPMV_NSMALL2_CHUNKS - 1u) / CTPMV_NSMALL2_CHUNKS;
    const uint32_t jStart = jLo + chunk * chunkLen;
    __gm__ float* out = wsGm + 2ULL * (static_cast<uint64_t>(row) * CTPMV_NSMALL2_CHUNKS + chunk); // 转置布局
    float sR = 0.0f;
    float sI = 0.0f;
    // UNIT 对角 x[row] 只由块 0 承担(jLo/jHi 已排除对角); 必须先于空块判断——
    // UPPER+UNIT 末行/LOWER+UNIT 首行的 total=0, 对角是其唯一贡献(sim 抓出的 bug)
    if (unit && chunk == 0u) {
        const uint64_t xo = NsmallXIdx(row, incx, absIncx, n);
        sR = xGm[2u * xo];
        sI = xGm[2u * xo + 1u];
    }
    if (jStart >= jHi) {
        out[0] = sR; // 空块: 写对角贡献(块0)或零
        out[1] = sI;
        return;
    }
    const uint32_t jEnd = (jStart + chunkLen < jHi) ? (jStart + chunkLen) : jHi;
    const CtpmvNSmallContext ctx{n, row, upper, incx, absIncx};
    CtpmvNSmallAccRange<decltype(2u)>(ctx, jStart, jEnd, apGm, xGm, sR, sI);
    out[0] = sR;
    out[1] = sI;
}

__global__ __aicore__ void ctpmv_nsmall2_kernel(GM_ADDR aP, GM_ADDR x, GM_ADDR y, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* apGm = reinterpret_cast<__gm__ const float*>(aP);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<CtpmvNSmall2Simt>(
        dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.uplo, tiling.diag, tiling.incx, apGm, xGm, yGm);
}

__global__ __aicore__ void ctpmv_reduce4_kernel(GM_ADDR src, GM_ADDR dst, CtpmvTilingData tiling, uint32_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* srcGm = reinterpret_cast<__gm__ float*>(src);
    auto* dstGm = reinterpret_cast<__gm__ float*>(dst);
    asc_vf_call<CtpmvReduceColumnsSimt<true>>(dim3{tiling.numThreads, 1, 1}, tiling.n, rows, srcGm, dstGm);
}

void ctpmv_arch35_reduce4_do(GM_ADDR src, GM_ADDR dst, const CtpmvTilingData& tiling, uint32_t rows, void* stream)
{
    ctpmv_reduce4_kernel<<<8, nullptr, stream>>>(src, dst, tiling, rows);
}

void ctpmv_arch35_nsmall2_do(GM_ADDR aP, GM_ADDR x, GM_ADDR y, const CtpmvTilingData& tiling, void* stream)
{
    ctpmv_nsmall2_kernel<<<CTPMV_NSMALL2_CHUNKS, nullptr, stream>>>(aP, x, y, tiling);
}

void ctpmv_arch35_scatter_do(GM_ADDR dst, GM_ADDR src, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctpmv_scatter_kernel<<<numBlocks, nullptr, stream>>>(dst, src, tiling);
}

// ==========================================================================
//  Preswap kernel（v4.5 配套）：把 x 的符号变体预计算到 GM scratch
//    T: scrA=[xR,-xI] scrB=[xI, xR]   C: scrA=[xR, xI] scrB=[xI,-xR]
//  vec kernel 经 DMA 装载使用——消灭"标量写UB→向量读"的跨单元数据流。
// ==========================================================================

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpmvPreswapSimt(
    uint32_t n, uint32_t conj, __gm__ const float* srcGm, __gm__ float* aGm, __gm__ float* bGm)
{
    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
        float xR = srcGm[2ULL * k];
        float xI = srcGm[2ULL * k + 1ULL];
        if (conj != 0U) {
            aGm[2ULL * k] = xR;
            aGm[2ULL * k + 1ULL] = xI;
            bGm[2ULL * k] = xI;
            bGm[2ULL * k + 1ULL] = -xR;
        } else {
            aGm[2ULL * k] = xR;
            aGm[2ULL * k + 1ULL] = -xI;
            bGm[2ULL * k] = xI;
            bGm[2ULL * k + 1ULL] = xR;
        }
    }
}

__global__ __aicore__ void ctpmv_preswap_kernel(GM_ADDR snap, GM_ADDR scrA, GM_ADDR scrB, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* s = reinterpret_cast<__gm__ const float*>(snap);
    auto* a = reinterpret_cast<__gm__ float*>(scrA);
    auto* b = reinterpret_cast<__gm__ float*>(scrB);
    asc_vf_call<CtpmvPreswapSimt>(
        dim3{tiling.numThreads, 1, 1}, tiling.n, (tiling.trans == ACLBLAS_OP_C) ? 1U : 0U, s, a, b);
}

void ctpmv_arch35_preswap_do(
    GM_ADDR snap, GM_ADDR scrA, GM_ADDR scrB, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctpmv_preswap_kernel<<<numBlocks, nullptr, stream>>>(snap, scrA, scrB, tiling);
}
