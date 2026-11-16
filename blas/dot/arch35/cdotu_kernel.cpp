/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/kernel_utils.h"
#include "cann_ops_blas_common.h"
#include "cdotu_tiling_data.h"

using namespace AscendC;

// ---------------------------------------------------------------------------
// Optimized SIMT cdotu kernel for Ascend 950PR (dav-3510).
//
// Architecture: pure SIMT. The complex payload is interleaved (re,im per element),
// so a SIMD DeInterleave is scalar-bound and cancels vectorization gains - SIMT
// threads with per-lane FMA chains win. Two compute paths selected at compile time:
//
//   CONTIGUOUS : incx==incy==1 -> element addresses are consecutive float2 slots.
//                The float stream is reinterpreted as float4 -- one 16B load packs
//                2 complex elements, halving GM load count and giving a fully
//                coalesced 512B/warp transaction.
//   STRIDED    : any other stride (|inc|!=1 or negative) -> float2 loads with
//                per-element stride arithmetic (Netlib negative-stride reversal).
//
// Reduction is three-level:
//   L1  thread  : per-thread FMA accumulator over grid-stride elements.
//   L2  warp    : asc_reduce_add collapses each 32-thread warp in one step, then
//                 one warp reduces the per-warp partials.
//   L3  core    : block 0 Kahan-summs the per-core partials (compensates the
//                 cross-core sum-order error).
//
// Key optimizations vs. baseline:
//   LAUNCH_BOUND 1024             32 regs/thread (was 16), unblocks 4x-issue FMA
//   ubuf arrays  64 (=max warps)  1/32 UB footprint vs. per-thread slot arrays
//   vectorize    CONTIGUOUS=float4 (2 complex/load), STRIDED=float2 (#pragma unroll 2)
//   DUAL_ACC     n>=3M            two independent accumulator sets (the two
//                                 complex halves of each float4 feed separate
//                                 chains) halve the serial FMA dependency length.
// ---------------------------------------------------------------------------

// Max warps per SIMT block (1024 threads / 32 = 32 warps); padded to 64 for
// alignment safety. This is the upper bound for ubuf partial-sum arrays.
constexpr uint32_t CDOTU_MAX_WARPS = 64;

// Resolve the float2 offset of complex index `idx` for a vector with stride `inc`.
// Negative stride: Netlib semantics -> 0-based (n-1-idx)*|inc|.
__simt_callee__ __aicore__ inline uint64_t CdotuOffset(uint32_t idx, int32_t inc, uint32_t incAbs, uint32_t nU)
{
    return (inc > 0) ? static_cast<uint64_t>(idx) * static_cast<uint64_t>(inc) :
                       static_cast<uint64_t>(nU - 1 - idx) * static_cast<uint64_t>(incAbs);
}

// Simplified warp-level reduction: asc_reduce_add collapses each warp in
// hardware, then a single warp collects and reduces the per-warp partials.
__simt_callee__ __aicore__ inline void CdotuWarpReduce(
    __ubuf__ float* ubReal, __ubuf__ float* ubImag, float real, float imag, __gm__ float* partialRealOut,
    __gm__ float* partialImagOut)
{
    // Step 1: hardware warp reduction (one instruction per warp)
    float warpReal = asc_reduce_add(real);
    float warpImag = asc_reduce_add(imag);
    if ((threadIdx.x & 31u) == 0) {
        ubReal[threadIdx.x >> 5] = warpReal;
        ubImag[threadIdx.x >> 5] = warpImag;
    }
    asc_syncthreads();

    // Step 2: one warp reduces the per-warp partials
    if (threadIdx.x < 32u) {
        uint32_t numWarps = blockDim.x >> 5;
        float vr = (threadIdx.x < numWarps) ? ubReal[threadIdx.x] : 0.0f;
        float vi = (threadIdx.x < numWarps) ? ubImag[threadIdx.x] : 0.0f;

        // Hardware reduce within the first warp
        float sr = asc_reduce_add(vr);
        float si = asc_reduce_add(vi);

        if (threadIdx.x == 0) {
            // host 侧 nthreads 封顶 1024（tiling_data 注释 128..1024），numWarps = blockDim.x >> 5
            // 恒不超过首个 warp 的硬件归约容量 32，ubReal/ubImag 仅前 numWarps 项有效；
            // 若未来将每块线程数提升到 2048（numWarps 达 CDOTU_MAX_WARPS），
            // 需在此恢复对 32 号之后 warp 部分和的显式累加
            partialRealOut[0] = sr;
            partialImagOut[0] = si;
        }
    }
}

// -- CONTIGUOUS path: float4 vectorized load (plain __gm__ subscript) --------
// incx==incy==1 -> complex elements are consecutive. Reinterpret the float
// stream as float4: one float4 (16B) packs 2 consecutive complex elements
//   xq.xy = complex(2q)   (real=x, imag=y)
//   xq.zw = complex(2q+1) (real=z, imag=w)
// A warp of 32 threads reads 32x16B = 512B contiguous -> fully coalesced, and
// halves the number of GM load instructions vs the float2 path.
//
// Host guarantees startIdx is even (pair-aligned partition), so base4 =
// startIdx/2 lands the core on a 16B float4 boundary. calNum may be odd only on
// the last core carrying the leftover complex; that single element is handled
// by a float2 tail on thread 0.
//
// DUAL_ACC mode: 2 independent accumulator sets (the two complex halves of each
// float4 feed separate chains) to break the serial FMA dependency.
template <bool DUAL_ACC>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void CdotuSimtComputeContig(
    uint32_t calNum, uint32_t startIdx, int32_t n, __gm__ const float* xf, __gm__ const float* yf,
    __gm__ float* partialRealOut, __gm__ float* partialImagOut)
{
    (void)n;
    if (calNum == 0) {
        return;
    }

    __ubuf__ float ubReal[CDOTU_MAX_WARPS];
    __ubuf__ float ubImag[CDOTU_MAX_WARPS];
    float real0 = 0.0f, imag0 = 0.0f;
    float real1 = 0.0f, imag1 = 0.0f;

    // float4 view; base4 is the float4 index of this core's first pair.
    // const_cast: reinterpret to a mutable float4 view for plain subscript loads
    // (read-only; the kernel never writes through xf4/yf4).
    __gm__ float4* xf4 = const_cast<__gm__ float4*>(reinterpret_cast<const __gm__ float4*>(xf));
    __gm__ float4* yf4 = const_cast<__gm__ float4*>(reinterpret_cast<const __gm__ float4*>(yf));
    uint32_t base4 = startIdx >> 1; // startIdx is even (host guarantee)
    uint32_t numQuad = calNum >> 1; // full complex pairs handled by float4
    uint32_t step = blockDim.x;

    for (uint32_t q = threadIdx.x; q < numQuad; q += step) {
        float4 xq = xf4[base4 + q];
        float4 yq = yf4[base4 + q];
        // complex A = xq.xy * yq.xy ; complex B = xq.zw * yq.zw
        if constexpr (DUAL_ACC) {
            real0 += xq.x * yq.x - xq.y * yq.y;
            imag0 += xq.x * yq.y + xq.y * yq.x;
            real1 += xq.z * yq.z - xq.w * yq.w;
            imag1 += xq.z * yq.w + xq.w * yq.z;
        } else {
            real0 += (xq.x * yq.x - xq.y * yq.y) + (xq.z * yq.z - xq.w * yq.w);
            imag0 += (xq.x * yq.y + xq.y * yq.x) + (xq.z * yq.w + xq.w * yq.z);
        }
    }

    float real = DUAL_ACC ? real0 + real1 : real0;
    float imag = DUAL_ACC ? imag0 + imag1 : imag0;

    // Tail: leftover odd complex (only when calNum is odd, at most one). Thread 0
    // reads it via float2 to avoid a misaligned float4 straddling the boundary.
    if ((calNum & 1u) != 0u && threadIdx.x == 0u) {
        const __gm__ float2* xf2 = reinterpret_cast<const __gm__ float2*>(xf);
        const __gm__ float2* yf2 = reinterpret_cast<const __gm__ float2*>(yf);
        uint32_t idx = startIdx + calNum - 1u;
        float2 xc = xf2[idx];
        float2 yc = yf2[idx];
        real += xc.x * yc.x - xc.y * yc.y;
        imag += xc.x * yc.y + xc.y * yc.x;
    }

    CdotuWarpReduce(ubReal, ubImag, real, imag, partialRealOut, partialImagOut);
}

// -- STRIDED path: 2x unroll ------------------------------------------------
// Non-contiguous offsets require per-element stride arithmetic; unrolling
// beyond 2x yields diminishing returns (the offset calculation dominates).
template <bool DUAL_ACC>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void CdotuSimtComputeStrided(
    uint32_t calNum, uint32_t startIdx, int32_t n, int32_t incx, int32_t incy, __gm__ const float* xf,
    __gm__ const float* yf, __gm__ float* partialRealOut, __gm__ float* partialImagOut)
{
    if (calNum == 0) {
        return;
    }

    __ubuf__ float ubReal[CDOTU_MAX_WARPS];
    __ubuf__ float ubImag[CDOTU_MAX_WARPS];
    float real0 = 0.0f, imag0 = 0.0f;
    float real1 = 0.0f, imag1 = 0.0f;

    const __gm__ float2* xf2 = reinterpret_cast<const __gm__ float2*>(xf);
    const __gm__ float2* yf2 = reinterpret_cast<const __gm__ float2*>(yf);
    uint32_t nU = static_cast<uint32_t>(n);
    uint32_t step = blockDim.x;
    uint32_t i = threadIdx.x;
    uint32_t incxAbs = (incx >= 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-static_cast<int64_t>(incx));
    uint32_t incyAbs = (incy >= 0) ? static_cast<uint32_t>(incy) : static_cast<uint32_t>(-static_cast<int64_t>(incy));

#pragma unroll 2
    for (; i + step < calNum; i += 2 * step) {
        uint32_t idx = startIdx + i;
        uint32_t idx1 = idx + step;
        uint64_t xo = CdotuOffset(idx, incx, incxAbs, nU);
        uint64_t yo = CdotuOffset(idx, incy, incyAbs, nU);
        uint64_t xo1 = CdotuOffset(idx1, incx, incxAbs, nU);
        uint64_t yo1 = CdotuOffset(idx1, incy, incyAbs, nU);
        float2 xc0 = xf2[xo];
        float2 yc0 = yf2[yo];
        float2 xc1 = xf2[xo1];
        float2 yc1 = yf2[yo1];

        if constexpr (DUAL_ACC) {
            real0 += xc0.x * yc0.x - xc0.y * yc0.y;
            imag0 += xc0.x * yc0.y + xc0.y * yc0.x;
            real1 += xc1.x * yc1.x - xc1.y * yc1.y;
            imag1 += xc1.x * yc1.y + xc1.y * yc1.x;
        } else {
            real0 += xc0.x * yc0.x - xc0.y * yc0.y + xc1.x * yc1.x - xc1.y * yc1.y;
            imag0 += xc0.x * yc0.y + xc0.y * yc0.x + xc1.x * yc1.y + xc1.y * yc1.x;
        }
    }

    // Tail: at most 1 element per thread
    for (; i < calNum; i += step) {
        uint32_t idx = startIdx + i;
        uint64_t xo = CdotuOffset(idx, incx, incxAbs, nU);
        uint64_t yo = CdotuOffset(idx, incy, incyAbs, nU);
        float2 xc = xf2[xo];
        float2 yc = yf2[yo];
        real0 += xc.x * yc.x - xc.y * yc.y;
        imag0 += xc.x * yc.y + xc.y * yc.x;
    }

    float real = DUAL_ACC ? real0 + real1 : real0;
    float imag = DUAL_ACC ? imag0 + imag1 : imag0;

    CdotuWarpReduce(ubReal, ubImag, real, imag, partialRealOut, partialImagOut);
}

// Dispatch helper: one asc_vf_call per template instantiation.
// Splits contiguous (float4 load, 2 complex/load) and strided (float2 load,
// #pragma unroll 2) paths at compile time; DUAL_ACC selects single vs dual
// accumulator chains.
template <bool DUAL_ACC>
__aicore__ inline void CdotuDispatchSimt(
    uint32_t calNum, uint32_t startIdx, int32_t n, int32_t incx, int32_t incy, GM_ADDR inX, GM_ADDR inY,
    __gm__ float* ws, uint32_t useCoreNum, uint32_t nthreads, bool contiguous)
{
    if (contiguous) {
        asc_vf_call<CdotuSimtComputeContig<DUAL_ACC>>(
            dim3{nthreads, 1, 1}, calNum, startIdx, n, reinterpret_cast<__gm__ const float*>(inX),
            reinterpret_cast<__gm__ const float*>(inY), ws, ws + useCoreNum);
    } else {
        asc_vf_call<CdotuSimtComputeStrided<DUAL_ACC>>(
            dim3{nthreads, 1, 1}, calNum, startIdx, n, incx, incy, reinterpret_cast<__gm__ const float*>(inX),
            reinterpret_cast<__gm__ const float*>(inY), ws, ws + useCoreNum);
    }
}

// Combine the per-core partials (Kahan-compensated to keep the final sum order
// error below the float32 accumulation limit).
__aicore__ inline void CdotuCombinePartials(__gm__ const float* ws, uint32_t useCoreNum, __gm__ float* res)
{
    float real = 0.0f;
    float realC = 0.0f;
    float imag = 0.0f;
    float imagC = 0.0f;
    for (uint32_t i = 0; i < useCoreNum; i++) {
        float yr = ws[i] - realC;
        float tr = real + yr;
        realC = (tr - real) - yr;
        real = tr;
        float yi = ws[useCoreNum + i] - imagC;
        float ti = imag + yi;
        imagC = (ti - imag) - yi;
        imag = ti;
    }
    res[0] = real;
    res[1] = imag;
}

extern "C" __global__ __aicore__ void cdotu_simt_kernel(
    GM_ADDR inX, GM_ADDR inY, GM_ADDR result, GM_ADDR workSpace, CdotuTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockIdx = GetBlockIdx();
    __gm__ float* ws = reinterpret_cast<__gm__ float*>(workSpace);

    // Even-aligned partition recomputed per core from n / useCoreNum / blockIdx
    // (no per-core arrays in tiling, matching sdot/dotex):
    //   base    = floor(n / useCoreNum) rounded down to even
    //   extra   = n - base * useCoreNum          (0 <= extra < 2 * useCoreNum)
    //   calNum  = base, +2 for the first extra/2 cores (one float4 pair each),
    //             +1 for the last core when extra is odd (lone complex tail)
    //   startIdx = even prefix sum of calNum over cores [0, blockIdx)
    // Invariants: every startIdx is even (float4 boundary), every core except the
    // last has even calNum, the only odd calNum is the last core's single leftover.
    uint32_t nU = static_cast<uint32_t>(tdata.n);
    uint32_t base = (nU / tdata.useCoreNum) & ~1u;
    uint32_t extra = nU - base * tdata.useCoreNum;
    uint32_t givePairs = extra >> 1;
    uint32_t calNum = base + (static_cast<uint32_t>(blockIdx) < givePairs ? 2u : 0u);
    if ((extra & 1u) != 0u && static_cast<uint32_t>(blockIdx) == tdata.useCoreNum - 1u) {
        calNum += 1u;
    }
    // startIdx = base * blockIdx + 2 * min(blockIdx, givePairs); all even.
    uint32_t startIdx =
        base * static_cast<uint32_t>(blockIdx) +
        2u * (static_cast<uint32_t>(blockIdx) < givePairs ? static_cast<uint32_t>(blockIdx) : givePairs);

    // calNum==0 cores must still zero their partial slots: CdotuCombinePartials
    // sums ws[0..useCoreNum) unconditionally, and a pair-aligned partition can
    // leave trailing cores idle. Reading an unwritten slot returned garbage that
    // corrupted the final dot (real wrong / imag right, and flaky single-run
    // pass/fail depending on stale workspace residue).
    // Note: with useCoreNum capped at n/2 on the host, every core carries >=2
    // elements, but this guard is retained as defensive zero-init.
    if (calNum > 0) {
        bool contiguous = (tdata.incx == 1 && tdata.incy == 1);
        if (tdata.useDualAcc) {
            CdotuDispatchSimt<true>(
                calNum, startIdx, tdata.n, tdata.incx, tdata.incy, inX, inY, ws + blockIdx, tdata.useCoreNum,
                tdata.nthreads, contiguous);
        } else {
            CdotuDispatchSimt<false>(
                calNum, startIdx, tdata.n, tdata.incx, tdata.incy, inX, inY, ws + blockIdx, tdata.useCoreNum,
                tdata.nthreads, contiguous);
        }
    } else {
        ws[blockIdx] = 0.0f;
        ws[tdata.useCoreNum + blockIdx] = 0.0f;
    }

    // Cross-core barrier: the same SIMT full-core sync pattern used by dotex/snrm2_ex.
    // A pure SIMT scalar GM store does not go through the MTE3 pipe, so
    // CrossCoreSetFlag<0, PIPE_MTE3> (the SIMD per-core DMA sync primitive) cannot sync
    // on it - under Release, block0 would read stale values (Output=0 / shifted lanes).
    // SyncAll makes every core's workspace write visible to block0 before block0 runs
    // the Kahan reduction.
    SyncAll();
    if (blockIdx == 0) {
        CdotuCombinePartials(ws, tdata.useCoreNum, reinterpret_cast<__gm__ float*>(result));
    }
}

void cdotu_kernel_do(
    GM_ADDR inX, GM_ADDR inY, GM_ADDR result, GM_ADDR workSpace, const CdotuTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);

    // Single SIMT kernel handles both contiguous (incx=incy=1) and strided paths
    // (including negative strides) and performs the cross-core reduce internally.
    // Measured fastest on Ascend 950PR (vectorized AIV + DeInterleave is scalar-bound).
    cdotu_simt_kernel<<<numBlocks, nullptr, aclStream>>>(inX, inY, result, workSpace, tiling);
}