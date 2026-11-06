/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl_rt.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "chpmv_kernel.h"

__simt_callee__ __aicore__ inline aclblasComplex ChpmvMul(aclblasComplex a, aclblasComplex b)
{
    // 显式乘以再加负零，在复数加减前保留各乘积的 FP32 舍入。
    const float rr = fmaf(a.real, b.real, -0.0f);
    const float ii = fmaf(a.imag, b.imag, -0.0f);
    const float ri = fmaf(a.real, b.imag, -0.0f);
    const float ir = fmaf(a.imag, b.real, -0.0f);
    return {rr - ii, ri + ir};
}

__simt_callee__ __aicore__ inline void ChpmvAdd(aclblasComplex& a, aclblasComplex b)
{
    a.real = a.real + b.real;
    a.imag = a.imag + b.imag;
}

__simt_callee__ __aicore__ inline aclblasComplex ChpmvLoad(__gm__ const float* p, int64_t i)
{
    return {p[2 * i], p[2 * i + 1]};
}

__simt_callee__ __aicore__ inline aclblasComplex ChpmvInitialOutput(
    __gm__ const float* y, int64_t index, aclblasComplex beta, bool betaZero, bool betaOne)
{
    if (betaZero) {
        return {0.0f, 0.0f};
    }
    const aclblasComplex old = ChpmvLoad(y, index);
    return betaOne ? old : ChpmvMul(beta, old);
}

__simt_callee__ __aicore__ inline int64_t ChpmvVectorIndex(uint32_t i, uint32_t n, int64_t inc)
{
    return inc > 0 ? static_cast<int64_t>(i) * inc : static_cast<int64_t>(n - 1U - i) * (-inc);
}

template <bool UPPER>
__simt_callee__ __aicore__ inline int64_t ChpmvColumnStart(uint32_t j, uint32_t n)
{
    const int64_t col = static_cast<int64_t>(j);
    if constexpr (UPPER) {
        return col * (col + 1) / 2;
    }
    return col * (2 * static_cast<int64_t>(n) - col + 1) / 2;
}

template <bool UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHPMV_THREADS) inline void ChpmvOrdered(
    __gm__ const float* ap, __gm__ const float* x, __gm__ float* y, ChpmvTilingData t)
{
    const bool alphaZero = t.alpha.real == 0.0f && t.alpha.imag == 0.0f;
    const bool betaZero = t.beta.real == 0.0f && t.beta.imag == 0.0f;
    const bool betaOne = t.beta.real == 1.0f && t.beta.imag == 0.0f;
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < t.n; row += gridDim.x * blockDim.x) {
        const int64_t yi = ChpmvVectorIndex(row, t.n, t.incy);
        aclblasComplex out = ChpmvInitialOutput(y, yi, t.beta, betaZero, betaOne);
        if (!alphaZero) {
            const int64_t own = ChpmvColumnStart<UPPER>(row, t.n);
            aclblasComplex sum{0.0f, 0.0f};
            // 每行保持 Netlib 的有序更新；共轭侧单独顺序规约。
            for (uint32_t col = 0; col < row; ++col) {
                const aclblasComplex xv = ChpmvLoad(x, ChpmvVectorIndex(col, t.n, t.incx));
                if constexpr (UPPER) {
                    aclblasComplex av = ChpmvLoad(ap, own + col);
                    av.imag = -av.imag;
                    ChpmvAdd(sum, ChpmvMul(av, xv));
                } else {
                    const aclblasComplex av = ChpmvLoad(ap, ChpmvColumnStart<false>(col, t.n) + row - col);
                    ChpmvAdd(out, ChpmvMul(ChpmvMul(t.alpha, xv), av));
                }
            }
            const aclblasComplex temp = ChpmvMul(t.alpha, ChpmvLoad(x, ChpmvVectorIndex(row, t.n, t.incx)));
            const int64_t diag = UPPER ? own + row : own;
            // 对角虚部不参与读取表达式，避免 0*Inf/NaN 污染结果。
            const float diagonal = ap[2 * diag];
            ChpmvAdd(out, {fmaf(temp.real, diagonal, -0.0f), fmaf(temp.imag, diagonal, -0.0f)});
            if constexpr (UPPER) {
                ChpmvAdd(out, ChpmvMul(t.alpha, sum));
            }
            for (uint32_t col = row + 1; col < t.n; ++col) {
                const aclblasComplex xv = ChpmvLoad(x, ChpmvVectorIndex(col, t.n, t.incx));
                if constexpr (UPPER) {
                    const aclblasComplex av = ChpmvLoad(ap, ChpmvColumnStart<true>(col, t.n) + row);
                    ChpmvAdd(out, ChpmvMul(ChpmvMul(t.alpha, xv), av));
                } else {
                    aclblasComplex av = ChpmvLoad(ap, own + col - row);
                    av.imag = -av.imag;
                    ChpmvAdd(sum, ChpmvMul(av, xv));
                }
            }
            if constexpr (!UPPER) {
                ChpmvAdd(out, ChpmvMul(t.alpha, sum));
            }
        }
        y[2 * yi] = out.real;
        y[2 * yi + 1] = out.imag;
    }
}

template <bool UPPER, bool CONJUGATE>
__simt_callee__ __aicore__ inline void ChpmvGroupSide(
    __gm__ const float* ap, __gm__ const float* x, const ChpmvTilingData& t, uint32_t row, int64_t own, uint32_t begin,
    uint32_t end, aclblasComplex& acc)
{
    const uint32_t lane = threadIdx.x % CHPMV_GROUP_SIZE;
    for (uint32_t base = begin; base < end; base += CHPMV_GROUP_SIZE) {
        const uint32_t col = base + lane;
        aclblasComplex product{0.0f, 0.0f};
        if (col < end) {
            const aclblasComplex xv = ChpmvLoad(x, ChpmvVectorIndex(col, t.n, t.incx));
            if constexpr (CONJUGATE) {
                const int64_t at = UPPER ? own + col : own + col - row;
                aclblasComplex av = ChpmvLoad(ap, at);
                av.imag = -av.imag;
                product = ChpmvMul(av, xv);
            } else {
                const int64_t at = ChpmvColumnStart<UPPER>(col, t.n) + row - (UPPER ? 0U : col);
                product = ChpmvMul(ChpmvMul(t.alpha, xv), ChpmvLoad(ap, at));
            }
        }
        // 小组并行形成乘积，再严格按列顺序累加；完整段不逐项判断尾部。
        const uint32_t count = end - base;
        if (count >= CHPMV_GROUP_SIZE) {
#pragma unroll
            for (uint32_t k = 0; k < CHPMV_GROUP_SIZE; ++k) {
                ChpmvAdd(
                    acc, {asc_shfl(product.real, k, CHPMV_GROUP_SIZE), asc_shfl(product.imag, k, CHPMV_GROUP_SIZE)});
            }
        } else {
            for (uint32_t k = 0; k < count; ++k) {
                ChpmvAdd(
                    acc, {asc_shfl(product.real, k, CHPMV_GROUP_SIZE), asc_shfl(product.imag, k, CHPMV_GROUP_SIZE)});
            }
        }
    }
}

template <bool UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHPMV_THREADS) inline void ChpmvGroupOrdered(
    __gm__ const float* ap, __gm__ const float* x, __gm__ float* y, ChpmvTilingData t)
{
    const uint32_t lane = threadIdx.x % CHPMV_GROUP_SIZE;
    const uint32_t groups = blockDim.x / CHPMV_GROUP_SIZE;
    const bool alphaZero = t.alpha.real == 0.0f && t.alpha.imag == 0.0f;
    const bool betaZero = t.beta.real == 0.0f && t.beta.imag == 0.0f;
    const bool betaOne = t.beta.real == 1.0f && t.beta.imag == 0.0f;
    for (uint32_t row = blockIdx.x * groups + threadIdx.x / CHPMV_GROUP_SIZE; row < t.n; row += gridDim.x * groups) {
        const int64_t yi = ChpmvVectorIndex(row, t.n, t.incy);
        aclblasComplex out{0.0f, 0.0f};
        if (!betaZero) {
            const aclblasComplex old = ChpmvLoad(y, yi);
            out = betaOne ? old : ChpmvMul(t.beta, old);
        }
        if (!alphaZero) {
            const int64_t own = ChpmvColumnStart<UPPER>(row, t.n);
            aclblasComplex sum{0.0f, 0.0f};
            if constexpr (UPPER) {
                ChpmvGroupSide<true, true>(ap, x, t, row, own, 0, row, sum);
            } else {
                ChpmvGroupSide<false, false>(ap, x, t, row, own, 0, row, out);
            }
            const aclblasComplex temp = ChpmvMul(t.alpha, ChpmvLoad(x, ChpmvVectorIndex(row, t.n, t.incx)));
            const float diagonal = ap[2 * (UPPER ? own + row : own)];
            ChpmvAdd(out, {fmaf(temp.real, diagonal, -0.0f), fmaf(temp.imag, diagonal, -0.0f)});
            if constexpr (UPPER) {
                ChpmvAdd(out, ChpmvMul(t.alpha, sum));
                ChpmvGroupSide<true, false>(ap, x, t, row, own, row + 1U, t.n, out);
            } else {
                ChpmvGroupSide<false, true>(ap, x, t, row, own, row + 1U, t.n, sum);
                ChpmvAdd(out, ChpmvMul(t.alpha, sum));
            }
        }
        // 一个小组独占一行，只有组内 lane 0 写入两个分量。
        if (lane == 0U) {
            y[2 * yi] = out.real;
            y[2 * yi + 1] = out.imag;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CHPMV_THREADS) inline void ChpmvStageInitial(
    __gm__ const float* y, __ubuf__ float* acc, uint32_t row0, ChpmvTilingData t)
{
    const uint32_t lane = threadIdx.x;
    if (lane < CHPMV_TILE_ROWS) {
        aclblasComplex value{0.0f, 0.0f};
        if (row0 + lane < t.n && (t.beta.real != 0.0f || t.beta.imag != 0.0f)) {
            value = ChpmvLoad(y, ChpmvVectorIndex(row0 + lane, t.n, t.incy));
            if (t.beta.real != 1.0f || t.beta.imag != 0.0f)
                value = ChpmvMul(t.beta, value);
        }
        acc[2U * lane] = value.real;
        acc[2U * lane + 1U] = value.imag;
        acc[2U * CHPMV_TILE_ROWS + 2U * lane] = 0.0f;
        acc[2U * CHPMV_TILE_ROWS + 2U * lane + 1U] = 0.0f;
    }
}

template <bool UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHPMV_STAGE_THREADS) inline void ChpmvStageTile(
    __gm__ const float* ap, __gm__ const float* x, __ubuf__ float* staged, __ubuf__ float* xTile, uint32_t row0,
    uint32_t col0, uint32_t cols, ChpmvTilingData t)
{
    const uint32_t tid = threadIdx.x;
    // 共轭区域沿 packed 列连续读取，再写为逐列 SIMD 布局。
    const bool direct = UPPER ? col0 > row0 : col0 < row0;
    const bool rowMajor = !direct;
    for (uint32_t item = tid; item < CHPMV_TILE_ROWS * CHPMV_TILE_COLS; item += blockDim.x) {
        const uint32_t lr = rowMajor ? item / CHPMV_TILE_COLS : item % CHPMV_TILE_ROWS;
        const uint32_t lc = rowMajor ? item % CHPMV_TILE_COLS : item / CHPMV_TILE_ROWS;
        const uint32_t row = row0 + lr, col = col0 + lc;
        aclblasComplex value{0.0f, 0.0f};
        if (row < t.n && lc < cols) {
            int64_t at;
            bool conjugate;
            if constexpr (UPPER) {
                conjugate = row > col;
                at = conjugate ? ChpmvColumnStart<true>(row, t.n) + col : ChpmvColumnStart<true>(col, t.n) + row;
            } else {
                conjugate = row < col;
                at = conjugate ? ChpmvColumnStart<false>(row, t.n) + col - row :
                                 ChpmvColumnStart<false>(col, t.n) + row - col;
            }
            // 对角只读实部。设备原缓冲区的对角虚部不进入任何计算。
            value = row == col ? aclblasComplex{ap[2 * at], 0.0f} : ChpmvLoad(ap, at);
            if (conjugate)
                value.imag = -value.imag;
        }
        const uint32_t dest = 2U * (lc * CHPMV_STAGE_PITCH + lr);
        staged[dest] = value.real;
        staged[dest + 1U] = value.imag;
    }
    if (tid < CHPMV_TILE_COLS) {
        aclblasComplex raw{0.0f, 0.0f}, scaled{0.0f, 0.0f};
        if (tid < cols) {
            raw = ChpmvLoad(x, ChpmvVectorIndex(col0 + tid, t.n, t.incx));
            scaled = ChpmvMul(t.alpha, raw);
        }
        xTile[4U * tid] = raw.real;
        xTile[4U * tid + 1U] = raw.imag;
        xTile[4U * tid + 2U] = scaled.real;
        xTile[4U * tid + 3U] = scaled.imag;
    }
}

__simd_vf__ inline void ChpmvAccumulateTile(
    uint32_t cols, uint32_t direct, __ubuf__ float* matrix, __ubuf__ float* xTile, __ubuf__ float* acc)
{
    using namespace AscendC;
    Reg::RegTensor<float> ar, ai, xr, xi, outR, outI, productR, productI, tmp;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(outR, outI, acc);
    for (uint16_t col = 0; col < static_cast<uint16_t>(cols); ++col) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, matrix + col * 2U * CHPMV_STAGE_PITCH);
        const uint32_t offset = 4U * col + (direct ? 2U : 0U);
        Reg::Duplicate(xr, xTile[offset], all);
        Reg::Duplicate(xi, xTile[offset + 1U], all);
        // 四个乘积先分别舍入；不把整个复数乘法收缩成 FMA。
        Reg::Mul(productR, ar, xr, all);
        Reg::Mul(tmp, ai, xi, all);
        Reg::Sub(productR, productR, tmp, all);
        if (direct) {
            Reg::Mul(productI, xr, ai, all);
            Reg::Mul(tmp, xi, ar, all);
        } else {
            Reg::Mul(productI, ar, xi, all);
            Reg::Mul(tmp, ai, xr, all);
        }
        Reg::Add(productI, productI, tmp, all);
        Reg::Add(outR, outR, productR, all);
        Reg::Add(outI, outI, productI, all);
    }
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(acc, outR, outI, all);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <bool UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHPMV_THREADS) inline void ChpmvDiagonalTile(
    __gm__ const float* ap, __gm__ const float* x, __ubuf__ float* acc, uint32_t row0, uint32_t cols, ChpmvTilingData t)
{
    const uint32_t lane = threadIdx.x % CHPMV_GROUP_SIZE;
    const uint32_t groups = blockDim.x / CHPMV_GROUP_SIZE;
    // 每个 warp 独占一行，协作形成乘积后按原列顺序累加。
    for (uint32_t local = threadIdx.x / CHPMV_GROUP_SIZE; local < cols; local += groups) {
        const uint32_t row = row0 + local;
        aclblasComplex out{acc[2U * local], acc[2U * local + 1U]};
        aclblasComplex sum{acc[2U * CHPMV_TILE_ROWS + 2U * local], acc[2U * CHPMV_TILE_ROWS + 2U * local + 1U]};
        const int64_t own = ChpmvColumnStart<UPPER>(row, t.n);
        if constexpr (UPPER) {
            ChpmvGroupSide<true, true>(ap, x, t, row, own, row0, row, sum);
        } else {
            ChpmvGroupSide<false, false>(ap, x, t, row, own, row0, row, out);
        }
        const aclblasComplex scaled = ChpmvMul(t.alpha, ChpmvLoad(x, ChpmvVectorIndex(row, t.n, t.incx)));
        const float diagonal = ap[2 * (UPPER ? own + row : own)];
        ChpmvAdd(out, {fmaf(scaled.real, diagonal, -0.0f), fmaf(scaled.imag, diagonal, -0.0f)});
        if constexpr (UPPER) {
            ChpmvAdd(out, ChpmvMul(t.alpha, sum));
            ChpmvGroupSide<true, false>(ap, x, t, row, own, row + 1U, row0 + cols, out);
        } else {
            ChpmvGroupSide<false, true>(ap, x, t, row, own, row + 1U, row0 + cols, sum);
        }
        if (lane == 0U) {
            acc[2U * local] = out.real;
            acc[2U * local + 1U] = out.imag;
            acc[2U * CHPMV_TILE_ROWS + 2U * local] = sum.real;
            acc[2U * CHPMV_TILE_ROWS + 2U * local + 1U] = sum.imag;
        }
    }
}

template <bool UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHPMV_THREADS) inline void ChpmvStoreRows(
    __gm__ float* y, __ubuf__ const float* acc, uint32_t row0, ChpmvTilingData t)
{
    const uint32_t lane = threadIdx.x;
    if (lane < CHPMV_TILE_ROWS && row0 + lane < t.n) {
        const int64_t at = ChpmvVectorIndex(row0 + lane, t.n, t.incy);
        aclblasComplex out{acc[2U * lane], acc[2U * lane + 1U]};
        if constexpr (!UPPER) {
            const aclblasComplex sum{acc[2U * CHPMV_TILE_ROWS + 2U * lane], acc[2U * CHPMV_TILE_ROWS + 2U * lane + 1U]};
            ChpmvAdd(out, ChpmvMul(t.alpha, sum));
        }
        y[2 * at] = out.real;
        y[2 * at + 1] = out.imag;
    }
}

template <bool UPPER>
__aicore__ inline void ChpmvSimdRun(GM_ADDR ap, GM_ADDR x, GM_ADDR y, const ChpmvTilingData& t, __ubuf__ float* arena)
{
    using namespace AscendC;
    auto* aUb = arena;
    auto* xUb = arena + CHPMV_STAGE_BYTES / sizeof(float);
    auto* accUb = xUb + CHPMV_X_BYTES / sizeof(float);
    auto* apGm = reinterpret_cast<__gm__ const float*>(ap);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    for (uint32_t row0 = GetBlockIdx() * CHPMV_TILE_ROWS; row0 < t.n; row0 += GetBlockNum() * CHPMV_TILE_ROWS) {
        const uint32_t rows = t.n - row0 < CHPMV_TILE_ROWS ? t.n - row0 : CHPMV_TILE_ROWS;
        asc_vf_call<ChpmvStageInitial>(dim3{CHPMV_THREADS, 1, 1}, yGm, accUb, row0, t);
        // SIMT 和 SIMD VF 同属 Vector 流水，仅同步该流水并保留 UB 访存屏障。
        PipeBarrier<PIPE_V>();
        DataSyncBarrier<MemDsbT::UB>();
        // 批量形成非对角乘积，遇到本行块的对角区域时提前截断。
        for (uint32_t col0 = 0; col0 < t.n;) {
            const uint32_t limit = col0 == row0 ? CHPMV_TILE_ROWS : CHPMV_TILE_COLS;
            uint32_t cols = t.n - col0 < limit ? t.n - col0 : limit;
            if (col0 < row0 && cols > row0 - col0)
                cols = row0 - col0;
            if (col0 == row0) {
                asc_vf_call<ChpmvDiagonalTile<UPPER>>(dim3{CHPMV_THREADS, 1, 1}, apGm, xGm, accUb, row0, cols, t);
            } else {
                asc_vf_call<ChpmvStageTile<UPPER>>(
                    dim3{CHPMV_STAGE_THREADS, 1, 1}, apGm, xGm, aUb, xUb, row0, col0, cols, t);
                // SIMT 暂存完成后，SIMD 才能读取 aUb 和 xUb。
                PipeBarrier<PIPE_V>();
                DataSyncBarrier<MemDsbT::UB>();
                const bool direct = UPPER ? col0 > row0 : col0 < row0;
                asc_vf_call<ChpmvAccumulateTile>(
                    cols, static_cast<uint32_t>(direct), aUb, xUb, accUb + (direct ? 0U : 2U * CHPMV_TILE_ROWS));
            }
            // 本列块累加完成后，下一块才能复用暂存区并更新累加区。
            PipeBarrier<PIPE_V>();
            DataSyncBarrier<MemDsbT::UB>();
            col0 += cols;
        }
        asc_vf_call<ChpmvStoreRows<UPPER>>(dim3{CHPMV_THREADS, 1, 1}, yGm, accUb, row0, t);
        // 写回 VF 完成后，下一行块才能重新初始化同一累加区。
        PipeBarrier<PIPE_V>();
        DataSyncBarrier<MemDsbT::UB>();
    }
}

__global__ __aicore__ void chpmv_kernel(GM_ADDR ap, GM_ADDR x, GM_ADDR y, const ChpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 显式声明跨 SIMT/SIMD 共用的静态 UB，确保编译元数据包含该内存区域。
    __ubuf__ __align__(32) float arena[(CHPMV_STAGE_BYTES + CHPMV_X_BYTES + CHPMV_ACC_BYTES) / sizeof(float)];
    if (tiling.strategy == 2 && tiling.uplo == ACLBLAS_UPPER) {
        ChpmvSimdRun<true>(ap, x, y, tiling, arena);
    } else if (tiling.strategy == 2) {
        ChpmvSimdRun<false>(ap, x, y, tiling, arena);
    } else if (tiling.strategy != 0 && tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<ChpmvGroupOrdered<true>>(
            dim3{tiling.threads, 1, 1}, reinterpret_cast<__gm__ const float*>(ap),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ float*>(y), tiling);
    } else if (tiling.strategy != 0) {
        asc_vf_call<ChpmvGroupOrdered<false>>(
            dim3{tiling.threads, 1, 1}, reinterpret_cast<__gm__ const float*>(ap),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ float*>(y), tiling);
    } else if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<ChpmvOrdered<true>>(
            dim3{tiling.threads, 1, 1}, reinterpret_cast<__gm__ const float*>(ap),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ float*>(y), tiling);
    } else {
        asc_vf_call<ChpmvOrdered<false>>(
            dim3{tiling.threads, 1, 1}, reinterpret_cast<__gm__ const float*>(ap),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ float*>(y), tiling);
    }
}

aclError chpmv_kernel_do(GM_ADDR ap, GM_ADDR x, GM_ADDR y, const ChpmvTilingData& tiling, uint32_t blocks, void* stream)
{
    chpmv_kernel<<<blocks, nullptr, stream>>>(ap, x, y, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
