/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// aclblasCrot device implementation (arch35 / ascend950).
// Two kernel entries, selected by tilingKey on the host.
//   crot_aiv_kernel  (tilingKey 0): contiguous, incx == incy == 1 -> SIMD membase.
//                    The complex stream is one run of floats, so whole UB tiles
//                    move with DataCopyPad and the rotation is 20 Vector ops.
//   crot_simt_kernel (tilingKey 1): every other stride (including 0 and negative)
//                    -> SIMT grid-stride, one complex element per thread.
// Rotation applied by both paths.
//     x'[k] = c*x[k] + s*y[j]
//     y'[j] = c*y[j] - conj(s)*x[k]      (original, pre-rotation x[k])
// In-place safety: both paths hold the original x and y until both results are
// formed - SIMT in registers, SIMD in UB - so y's update always sees the
// pre-rotation x.
// Zero stride: the task book (section 2.1.4) requires zero strides to be accepted and
// evaluated with reference order semantics, so the N
// iterations act serially on the same element (each iteration consumes the value
// written by the previous one). Only block0/thread0 executes the loop; every
// other thread returns immediately. A parallel grid-stride loop would let
// several threads read the same original element and race on the store,
// diverging from the reference. This mirrors the srot kernel's R2 fallback.

#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "crot_kernel.h"

using namespace AscendC;

// ==========================================================================
// Rotation math shared by both paths.
// Every product is stored through a volatile temporary so the compiler cannot
// contract multiply+add into an FMA. The host golden uses the same sequence,
// which makes the two sides bit-identical even for the extreme scalar values
// where catastrophic cancellation magnifies a single fused-rounding difference
// far beyond the FP32 tolerance.
// ==========================================================================
__simt_callee__ __aicore__ inline void CrotRotate(
    float c, float sr, float si, float xr, float xi, float yr, float yi, float& newXr, float& newXi, float& newYr,
    float& newYi)
{
    volatile float srYr = sr * yr;
    volatile float siYi = si * yi;
    volatile float srYi = sr * yi;
    volatile float siYr = si * yr;
    volatile float srXr = sr * xr;
    volatile float siXi = si * xi;
    volatile float srXi = sr * xi;
    volatile float siXr = si * xr;
    volatile float cXr = c * xr;
    volatile float cXi = c * xi;
    volatile float cYr = c * yr;
    volatile float cYi = c * yi;
    volatile float xRePart = srYr - siYi;
    volatile float xImPart = srYi + siYr;
    volatile float yRePart = srXr + siXi;
    volatile float yImPart = srXi - siXr;
    newXr = cXr + xRePart;
    newXi = cXi + xImPart;
    newYr = cYr - yRePart;
    newYi = cYi - yImPart;
}

// ==========================================================================
// Contiguous path: SIMD (AIV) membase.
// incx==incy==1 makes the complex stream one contiguous run of floats, so the
// whole rotation is done with MTE2/MTE3 bulk transfers instead of per-thread GM
// accesses: DataCopyPad GM->UB, DeInterleave to split the interleaved AoS payload
// into pure real / pure imag vectors, 20 scalar-broadcast Vector ops, Interleave
// back to AoS, DataCopyPad UB->GM.
// Rounding: every product is a separate Muls, and every sum a separate Add/Sub,
// exactly mirroring the per-product sequencing of CrotRotate (no FMA contraction),
// so the vector path is bit-identical to the SIMT path and to the host golden.
// In-place safety: xr/xi/yr/yi all live in UB before any result is stored, so
// y' is computed from the pre-rotation x.
// ==========================================================================
class CrotKernel {
public:
    __aicore__ inline CrotKernel() {}
    __aicore__ inline void Init(
        __gm__ float* x, __gm__ float* y, __gm__ float* cPtr, __gm__ float* sPtr, const CrotTilingData& tiling,
        TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(uint32_t elementOffset, uint32_t elementCount, bool firstTile);
    __aicore__ inline void ComputeRotation(int32_t count);

    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    TBuf<TPosition::VECIN> aosXBuf_; // interleaved x tile (GM -> UB, reused as new-x staging)
    TBuf<TPosition::VECIN> aosYBuf_; // interleaved y tile (GM -> UB, reused as new-y staging)
    TBuf<TPosition::VECCALC> xrBuf_;
    TBuf<TPosition::VECCALC> xiBuf_;
    TBuf<TPosition::VECCALC> yrBuf_;
    TBuf<TPosition::VECCALC> yiBuf_;
    TBuf<TPosition::VECCALC> nxrBuf_;
    TBuf<TPosition::VECCALC> nxiBuf_;
    TBuf<TPosition::VECCALC> nyrBuf_;
    TBuf<TPosition::VECCALC> nyiBuf_;
    TBuf<TPosition::VECCALC> t0Buf_; // scratch: s*y / conj(s)*x products
    TBuf<TPosition::VECCALC> t1Buf_; // scratch
    float cosValue_;
    float sinRe_;
    float sinIm_;
    uint32_t myOffset_;
    uint32_t myCount_;
};

__aicore__ inline void CrotKernel::Init(
    __gm__ float* x, __gm__ float* y, __gm__ float* cPtr, __gm__ float* sPtr, const CrotTilingData& tiling, TPipe* pipe)
{
    cosValue_ = tiling.cIsDevice ? *cPtr : tiling.cosValue;
    sinRe_ = tiling.sIsDevice ? sPtr[0] : tiling.sinRe;
    sinIm_ = tiling.sIsDevice ? sPtr[1] : tiling.sinIm;

    xGM_.SetGlobalBuffer(x, static_cast<uint64_t>(tiling.totalN) * 2U);
    yGM_.SetGlobalBuffer(y, static_cast<uint64_t>(tiling.totalN) * 2U);

    // Even split: base perCoreN elements per core, the front `remainder` cores
    // take one extra. Ranges butt exactly, no gap/overlap.
    uint32_t blockIdx = GetBlockIdx();
    uint32_t perCore = tiling.perCoreN;
    myOffset_ = blockIdx * perCore;
    if (blockIdx < tiling.remainder) {
        myOffset_ += blockIdx;
        perCore += 1U;
    } else {
        myOffset_ += tiling.remainder;
    }
    myCount_ = perCore;

    constexpr uint32_t tileBytes = CROT_TILE_COMPLEX * sizeof(float);
    pipe->InitBuffer(aosXBuf_, 2U * tileBytes);
    pipe->InitBuffer(aosYBuf_, 2U * tileBytes);
    pipe->InitBuffer(xrBuf_, tileBytes);
    pipe->InitBuffer(xiBuf_, tileBytes);
    pipe->InitBuffer(yrBuf_, tileBytes);
    pipe->InitBuffer(yiBuf_, tileBytes);
    pipe->InitBuffer(nxrBuf_, tileBytes);
    pipe->InitBuffer(nxiBuf_, tileBytes);
    pipe->InitBuffer(nyrBuf_, tileBytes);
    pipe->InitBuffer(nyiBuf_, tileBytes);
    pipe->InitBuffer(t0Buf_, tileBytes);
    pipe->InitBuffer(t1Buf_, tileBytes);
}

__aicore__ inline void CrotKernel::ProcessTile(uint32_t elementOffset, uint32_t elementCount, bool firstTile)
{
    uint32_t floatOffset = elementOffset * 2U;
    uint32_t floatCount = elementCount * 2U;
    // Vector/permute ops run on a 32B-rounded element count: DeInterleave and
    // Interleave require even counts and every UB address they touch must be
    // 32B aligned, which also makes the outY offset (2*opCount floats) legal.
    // The padded tail is UB garbage that is computed and then discarded by the
    // DataCopyPad stores, which only move floatCount floats.
    uint32_t opCount = (elementCount + 7U) & ~7U;
    int32_t count = static_cast<int32_t>(opCount);

    LocalTensor<float> aosX = aosXBuf_.Get<float>();
    LocalTensor<float> aosY = aosYBuf_.Get<float>();
    // The interleaved results are staged back into the (now consumed) AoS input
    // buffers: DeInterleave finished reading them before Interleave writes.
    LocalTensor<float> outX = aosX;
    LocalTensor<float> outY = aosY;
    LocalTensor<float> xr = xrBuf_.Get<float>();
    LocalTensor<float> xi = xiBuf_.Get<float>();
    LocalTensor<float> yr = yrBuf_.Get<float>();
    LocalTensor<float> yi = yiBuf_.Get<float>();
    LocalTensor<float> nxr = nxrBuf_.Get<float>();
    LocalTensor<float> nxi = nxiBuf_.Get<float>();
    LocalTensor<float> nyr = nyrBuf_.Get<float>();
    LocalTensor<float> nyi = nyiBuf_.Get<float>();

    // The in/out staging buffers are reused every tile: the previous tile's MTE3
    // store must have landed before this tile's MTE2 load overwrites them.
    if (!firstTile) {
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(floatCount * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    DataCopyPad(aosX, xGM_[floatOffset], copyParams, padParams);
    DataCopyPad(aosY, yGM_[floatOffset], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    // DeInterleave's last argument is the SOURCE float count (2 * opCount floats),
    // while Interleave's is the per-side element count (opCount). Both derive from
    // the same 32B-rounded opCount, so the GM-side DataCopyPad stays floatCount.
    DeInterleave(xr, xi, aosX, 2 * count);
    DeInterleave(yr, yi, aosY, 2 * count);
    PipeBarrier<PIPE_V>();

    ComputeRotation(count);

    Interleave(outX, outX[count], nxr, nxi, count);
    Interleave(outY, outY[count], nyr, nyi, count);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    DataCopyPad(xGM_[floatOffset], outX, copyParams);
    DataCopyPad(yGM_[floatOffset], outY, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(0);
}

// Rotate the de-interleaved UB vectors in place. Every product is a separate
// Muls and every sum a separate Add/Sub (no FMA contraction), matching
// CrotRotate so the vector path stays bit-identical to the golden.
__aicore__ inline void CrotKernel::ComputeRotation(int32_t count)
{
    LocalTensor<float> xr = xrBuf_.Get<float>();
    LocalTensor<float> xi = xiBuf_.Get<float>();
    LocalTensor<float> yr = yrBuf_.Get<float>();
    LocalTensor<float> yi = yiBuf_.Get<float>();
    LocalTensor<float> nxr = nxrBuf_.Get<float>();
    LocalTensor<float> nxi = nxiBuf_.Get<float>();
    LocalTensor<float> nyr = nyrBuf_.Get<float>();
    LocalTensor<float> nyi = nyiBuf_.Get<float>();
    LocalTensor<float> t0 = t0Buf_.Get<float>();
    LocalTensor<float> t1 = t1Buf_.Get<float>();

    // newXr = c*xr + (sr*yr - si*yi)
    Muls(t0, yr, sinRe_, count);
    Muls(t1, yi, sinIm_, count);
    Sub(t0, t0, t1, count);
    Muls(nxr, xr, cosValue_, count);
    Add(nxr, nxr, t0, count);
    // newXi = c*xi + (sr*yi + si*yr)
    Muls(t1, yi, sinRe_, count);
    Muls(t0, yr, sinIm_, count);
    Add(t1, t1, t0, count);
    Muls(nxi, xi, cosValue_, count);
    Add(nxi, nxi, t1, count);
    // newYr = c*yr - (sr*xr + si*xi)
    Muls(t0, xr, sinRe_, count);
    Muls(t1, xi, sinIm_, count);
    Add(t0, t0, t1, count);
    Muls(nyr, yr, cosValue_, count);
    Sub(nyr, nyr, t0, count);
    // newYi = c*yi - (sr*xi - si*xr)
    Muls(t1, xi, sinRe_, count);
    Muls(t0, xr, sinIm_, count);
    Sub(t1, t1, t0, count);
    Muls(nyi, yi, cosValue_, count);
    Sub(nyi, nyi, t1, count);
}

__aicore__ inline void CrotKernel::Process()
{
    if (myCount_ == 0U) {
        return;
    }

    uint32_t tileLoop = myCount_ / CROT_TILE_COMPLEX;
    uint32_t tileTail = myCount_ % CROT_TILE_COMPLEX;
    uint32_t curOffset = myOffset_;
    bool firstTile = true;
    for (uint32_t i = 0; i < tileLoop; i++) {
        ProcessTile(curOffset, CROT_TILE_COMPLEX, firstTile);
        firstTile = false;
        curOffset += CROT_TILE_COMPLEX;
    }
    if (tileTail > 0U) {
        ProcessTile(curOffset, tileTail, firstTile);
    }
    // Drain the last tile's store so the MTE3->MTE2 flag pair stays balanced.
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

// ==========================================================================
// SIMT compute: one grid-stride loop over the logical element range.
// ==========================================================================

// Load one complex element from x and y, rotate it, and store both results
// back to GM. x and y are read before either result is written, so y' always
// sees the pre-rotation x. Shared by the zero-stride serial path and the
// parallel grid-stride path.
__simt_callee__ __aicore__ inline void CrotRotateAtGm(
    __gm__ float* xGm, __gm__ float* yGm, uint32_t xIdx, uint32_t yIdx, float c, float sr, float si)
{
    float xr = xGm[2U * xIdx];
    float xi = xGm[2U * xIdx + 1U];
    float yr = yGm[2U * yIdx];
    float yi = yGm[2U * yIdx + 1U];
    float newXr;
    float newXi;
    float newYr;
    float newYi;
    CrotRotate(c, sr, si, xr, xi, yr, yi, newXr, newXi, newYr, newYi);
    xGm[2U * xIdx] = newXr;
    xGm[2U * xIdx + 1U] = newXi;
    yGm[2U * yIdx] = newYr;
    yGm[2U * yIdx + 1U] = newYi;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void crot_simt_compute(
    uint32_t n, uint32_t absIncX, uint32_t absIncY, int32_t negX, int32_t negY, int32_t zeroIncX, int32_t zeroIncY,
    int32_t serialOnly, float c, float sr, float si, __gm__ float* xGm, __gm__ float* yGm)
{
    // ---- Serial path: zero strides and aliased buffers ----
    // absInc is coerced to 1 by the caller for the zero side; jx/jy force the element index
    // to 0, so the same address is read/written N times in order. The same single-thread
    // walk is used when the host detected an aliased buffer (x == y with different strides),
    // where parallel iterations would otherwise race on the shared elements.
    if ((serialOnly != 0) || (zeroIncX != 0) || (zeroIncY != 0)) {
        if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
            return;
        }
        for (uint32_t i = 0; i < n; i++) {
            uint32_t jx = (zeroIncX != 0) ? 0U : ((negX != 0) ? (n - 1 - i) : i);
            uint32_t jy = (zeroIncY != 0) ? 0U : ((negY != 0) ? (n - 1 - i) : i);
            CrotRotateAtGm(xGm, yGm, jx * absIncX, jy * absIncY, c, sr, si);
        }
        return;
    }

    uint32_t tidGlobal = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t gridStride = block_num * blockDim.x;

    // ---- Parallel grid-stride path ----
    for (uint32_t i = tidGlobal; i < n; i += gridStride) {
        uint32_t jx = (negX != 0) ? (n - 1 - i) : i;
        uint32_t jy = (negY != 0) ? (n - 1 - i) : i;
        // Load original x and y first: y' must use the pre-rotation x.
        CrotRotateAtGm(xGm, yGm, jx * absIncX, jy * absIncY, c, sr, si);
    }
}

// ==========================================================================
// Kernel entries — two separate binaries so SIMD (TPipe/DataCopy) and SIMT
// (asc_vf_call) code never coexist in one kernel image.
// ==========================================================================

// Contiguous path (incx == incy == 1): SIMD membase.
extern "C" __global__ __aicore__ void crot_aiv_kernel(
    __gm__ float* x, __gm__ float* y, __gm__ float* cPtr, __gm__ float* sPtr, CrotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CrotKernel op;
    op.Init(x, y, cPtr, sPtr, tiling, &pipe);
    op.Process();
}

// Strided path (any other stride): SIMT.
extern "C" __global__ __aicore__ void crot_simt_kernel(
    __gm__ float* x, __gm__ float* y, __gm__ float* cPtr, __gm__ float* sPtr, CrotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ float* xGm = x;
    __gm__ float* yGm = y;
    uint32_t n = tiling.totalN;
    int32_t incx = tiling.incx;
    int32_t incy = tiling.incy;
    int32_t zeroIncX = (incx == 0) ? 1 : 0;
    int32_t zeroIncY = (incy == 0) ? 1 : 0;
    // |inc| via int64 to avoid signed-negation UB when inc == INT32_MIN.
    uint32_t absIncX = static_cast<uint32_t>(incx < 0 ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx));
    uint32_t absIncY = static_cast<uint32_t>(incy < 0 ? -static_cast<int64_t>(incy) : static_cast<int64_t>(incy));
    if (absIncX == 0) {
        absIncX = 1; // inc==0 reuses element 0; a single stride covers it
    }
    if (absIncY == 0) {
        absIncY = 1;
    }
    int32_t negX = (incx < 0) ? 1 : 0;
    int32_t negY = (incy < 0) ? 1 : 0;
    int32_t serialOnly = (tiling.serialOnly != 0U) ? 1 : 0;

    // c/s source: device pointer -> read scalar from GM; host pointer -> tiling scalar.
    float c = tiling.cIsDevice ? *cPtr : tiling.cosValue;
    float sr = tiling.sIsDevice ? sPtr[0] : tiling.sinRe;
    float si = tiling.sIsDevice ? sPtr[1] : tiling.sinIm;

    asc_vf_call<crot_simt_compute>(
        dim3{tiling.nthreads, 1, 1}, n, absIncX, absIncY, negX, negY, zeroIncX, zeroIncY, serialOnly, c, sr, si, xGm,
        yGm);
}

// ==========================================================================
// Host-side kernel launcher
// ==========================================================================
void crot_kernel_do(
    float* x, float* y, float* cPtr, float* sPtr, uint32_t numBlocks, const CrotTilingData& tiling, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    if (tiling.tilingKey == 0U) {
        crot_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(x, y, cPtr, sPtr, tiling);
    } else {
        crot_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, y, cPtr, sPtr, tiling);
    }
}
