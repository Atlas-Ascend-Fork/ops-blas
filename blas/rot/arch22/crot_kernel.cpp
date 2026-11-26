/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// aclblasCrot device implementation (arch22 / Atlas A2/A3, ASCEND_V220).
//
// Complex Givens plane rotation with a complex sine s:
//     x'[k] = c*x[k] + s*y[j]
//     y'[j] = c*y[j] - conj(s)*x[k]     (pre-rotation x[k])
//
// Two code paths, selected by the host tiling:
//   incx == 1 && incy == 1  -> contiguous SIMD (AIV) path: bulk GM<->UB transfers,
//                              vreducev2 de-interleave into planar real/imag, 20 vector
//                              ops, AscendC::Gather re-interleave, bulk store. A short
//                              tail (< one tile) is handled by the scalar path.
//   otherwise (|inc|>1 or negative) -> strided scalar path: per-element scalar GM loads
//                              staged into 32B-aligned UB slots, rotated, written back with
//                              one 8-byte DataCopyPad burst per element.
//
// Zero stride and INT32_MIN are rejected on the host (arch22 contract, issue #827), so the
// kernel never needs a zero-stride serial path. Negative strides resolve the physical index
// as (totalN - 1 - logical) * |inc| per Netlib semantics.
//
// In-place safety: both paths hold the original x and y until both results are formed
// (SIMD keeps them in planar UB, strided keeps them in staging slots), so y' always sees
// the pre-rotation x.

#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/compute/simd.h"
#include "crot_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t CROT_MASK_CHUNKS = 2U;
constexpr uint32_t CROT_COMPONENTS = 2U;                      // float components per complex
constexpr uint32_t CROT_PLANAR_FLOATS = CROT_TILE_COMPLEX;    // real (or imag) floats per tile
constexpr uint32_t CROT_TILE_FLOATS = 2U * CROT_TILE_COMPLEX; // interleaved floats per tile

// Fixed UB byte layout for the contiguous path (single-buffered, one tile at a time).
// All addresses are 32B-aligned. Total 72 KiB, well under the 910B3 192 KiB UB budget.
// NXI = NXR + PLANAR_FLOATS floats, NYI = NYR + PLANAR_FLOATS floats, so each real/imag
// pair is contiguous and the Gather re-interleave can use relative element offsets.
constexpr uint32_t CROT_UB_XAOS = 0U; // interleaved x tile / x output staging
constexpr uint32_t CROT_UB_YAOS = CROT_UB_XAOS + CROT_TILE_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_XR = CROT_UB_YAOS + CROT_TILE_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_XI = CROT_UB_XR + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_YR = CROT_UB_XI + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_YI = CROT_UB_YR + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_NXR = CROT_UB_YI + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_NXI = CROT_UB_NXR + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_NYR = CROT_UB_NXI + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_NYI = CROT_UB_NYR + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_T0 = CROT_UB_NYI + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_T1 = CROT_UB_T0 + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_T2 = CROT_UB_T1 + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_T3 = CROT_UB_T2 + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_MASK_X = CROT_UB_T3 + CROT_PLANAR_FLOATS * sizeof(float);
constexpr uint32_t CROT_UB_XAOS2 = CROT_UB_MASK_X + CROT_TILE_FLOATS * sizeof(uint32_t);
constexpr uint32_t CROT_UB_YAOS2 = CROT_UB_XAOS2 + CROT_TILE_FLOATS * sizeof(float);
} // namespace

// ==========================================================================
// Rotation math shared by both paths.
// Every product is stored through a volatile temporary so the compiler cannot
// contract multiply+add into an FMA. The host golden (crot_golden.h) uses the same
// sequence, which makes the two sides bit-identical even for the extreme scalar
// values where catastrophic cancellation magnifies a single fused-rounding
// difference far beyond the FP32 tolerance.
//   x' re = c*xr + (sr*yr - si*yi)
//   x' im = c*xi + (sr*yi + si*yr)
//   y' re = c*yr - (sr*xr + si*xi)
//   y' im = c*yi - (sr*xi - si*xr)
// ==========================================================================
__aicore__ __inline__ void CrotRotateScalar(
    float c, float sr, float si, float xr, float xi, float yr, float yi, float& nxr, float& nxi, float& nyr, float& nyi)
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
    nxr = cXr + xRePart;
    nxi = cXi + xImPart;
    nyr = cYr - yRePart;
    nyi = cYi - yImPart;
}

// Physical element index of logical index i for stride inc; negative strides start from
// the physical vector tail per Netlib semantics.
__aicore__ __inline__ int64_t CrotStridedPhysIndex(int64_t logical, int32_t inc, int64_t total)
{
    return (inc > 0) ? logical * inc : (total - 1 - logical) * (-static_cast<int64_t>(inc));
}

// Rotate one strided element into its 32B-aligned UB staging slot.
__aicore__ __inline__ void CrotStridedStageOne(
    GlobalTensor<float> gmX, GlobalTensor<float> gmY, int64_t logical, int32_t incx, int32_t incy, int64_t total,
    float cosValue, float sinRe, float sinIm, LocalTensor<float> slotX, LocalTensor<float> slotY, uint32_t offset)
{
    const int64_t xPhys = CrotStridedPhysIndex(logical, incx, total);
    const int64_t yPhys = CrotStridedPhysIndex(logical, incy, total);
    const uint64_t xF = static_cast<uint64_t>(xPhys) * CROT_COMPONENTS;
    const uint64_t yF = static_cast<uint64_t>(yPhys) * CROT_COMPONENTS;
    const float xr = gmX.GetValue(xF);
    const float xi = gmX.GetValue(xF + 1);
    const float yr = gmY.GetValue(yF);
    const float yi = gmY.GetValue(yF + 1);
    float nxr;
    float nxi;
    float nyr;
    float nyi;
    CrotRotateScalar(cosValue, sinRe, sinIm, xr, xi, yr, yi, nxr, nxi, nyr, nyi);
    slotX.SetValue(offset, nxr);
    slotX.SetValue(offset + 1, nxi);
    slotY.SetValue(offset, nyr);
    slotY.SetValue(offset + 1, nyi);
}

// Write one pass of staged results back with an 8-byte burst per element.
__aicore__ __inline__ void CrotStridedStorePass(
    GlobalTensor<float> gmX, GlobalTensor<float> gmY, int64_t firstLogical, int64_t passElems, int32_t incx,
    int32_t incy, int64_t total, LocalTensor<float> slotX, LocalTensor<float> slotY)
{
    AscendC::DataCopyExtParams outParams{1, CROT_COMPONENTS * 4, 0, 0, 0};
    for (int64_t i = 0; i < passElems; i++) {
        const int64_t logical = firstLogical + i;
        const int64_t xPhys = CrotStridedPhysIndex(logical, incx, total);
        const int64_t yPhys = CrotStridedPhysIndex(logical, incy, total);
        const uint32_t offset = static_cast<uint32_t>(i) * CROT_STRIDED_SLOT;
        DataCopyPad(gmX[static_cast<uint64_t>(xPhys) * CROT_COMPONENTS], slotX[offset], outParams);
        DataCopyPad(gmY[static_cast<uint64_t>(yPhys) * CROT_COMPONENTS], slotY[offset], outParams);
    }
}

// ==========================================================================
// Strided path: non-unit / negative strides. Old values are read with scalar GM loads,
// rotated into 32B-aligned UB slots and written back with one 8-byte DataCopyPad burst
// per element.
// ==========================================================================
__aicore__ __inline__ void crot_process_strided_aiv(
    GlobalTensor<float> gmX, GlobalTensor<float> gmY, int32_t elementCount, int32_t incx, int32_t incy, float cosValue,
    float sinRe, float sinIm)
{
    const int64_t total = elementCount / static_cast<int32_t>(CROT_COMPONENTS);
    const int16_t aivId = AscendC::GetBlockIdx();
    int16_t aivNum = AscendC::GetBlockNum();
    if (aivNum == 0) {
        aivNum = 1;
    }

    int64_t elemsPerCore = total / aivNum;
    const int64_t elemsRemain = total % aivNum;
    int64_t startElem = static_cast<int64_t>(aivId) * elemsPerCore;
    if (aivId < elemsRemain) {
        startElem += aivId;
        elemsPerCore += 1;
    } else {
        startElem += elemsRemain;
    }

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    LocalTensor<float> slotX = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    LocalTensor<float> slotY = buf.GetBuffer<BufferType::ASCEND_UB, float>(32 * 1024);

    for (int64_t base = 0; base < elemsPerCore; base += CROT_STRIDED_ELEMS_PER_PASS) {
        const int64_t rest = elemsPerCore - base;
        const int64_t passElems = rest < CROT_STRIDED_ELEMS_PER_PASS ? rest : CROT_STRIDED_ELEMS_PER_PASS;
        const int64_t firstLogical = startElem + base;
        for (int64_t i = 0; i < passElems; i++) {
            CrotStridedStageOne(
                gmX, gmY, firstLogical + i, incx, incy, total, cosValue, sinRe, sinIm, slotX, slotY,
                static_cast<uint32_t>(i) * CROT_STRIDED_SLOT);
        }
        // Scalar UB writes above do not ride the V pipe, so a full barrier is required
        // before MTE3 reads the staging slots.
        PIPE_BARRIER(ALL);
        CrotStridedStorePass(gmX, gmY, firstLogical, passElems, incx, incy, total, slotX, slotY);
        PIPE_BARRIER(ALL);
    }
}

// ==========================================================================
// Contiguous path: SIMD planar rotation over full CROT_TILE_COMPLEX tiles.
// vreducev2 splits the interleaved AoS tile into pure real / pure imag vectors; the four
// result planes are computed with muls_v/add_v/sub_v (no FMA contraction), then
// AscendC::Gather re-interleaves them back into the AoS output staging buffers.
// ==========================================================================

// UB tensor bundle, so the tiling helpers keep short signatures.
struct CrotTileUb {
    LocalTensor<float> xaos;
    LocalTensor<float> yaos;
    LocalTensor<float> xaos2;
    LocalTensor<float> yaos2;
    LocalTensor<float> xr;
    LocalTensor<float> xi;
    LocalTensor<float> yr;
    LocalTensor<float> yi;
    LocalTensor<float> nxr;
    LocalTensor<float> nxi;
    LocalTensor<float> nyr;
    LocalTensor<float> nyi;
    LocalTensor<float> t0;
    LocalTensor<float> t1;
    LocalTensor<float> t2;
    LocalTensor<float> t3;
    LocalTensor<uint32_t> maskX;
};

__aicore__ __inline__ CrotTileUb CrotMakeTileUb()
{
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    CrotTileUb ub;
    ub.xaos = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_XAOS);
    ub.yaos = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_YAOS);
    ub.xaos2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_XAOS2);
    ub.yaos2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_YAOS2);
    ub.xr = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_XR);
    ub.xi = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_XI);
    ub.yr = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_YR);
    ub.yi = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_YI);
    ub.nxr = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_NXR);
    ub.nxi = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_NXI);
    ub.nyr = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_NYR);
    ub.nyi = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_NYI);
    ub.t0 = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_T0);
    ub.t1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_T1);
    ub.t2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_T2);
    ub.t3 = buf.GetBuffer<BufferType::ASCEND_UB, float>(CROT_UB_T3);
    ub.maskX = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(CROT_UB_MASK_X);
    return ub;
}

// Re-interleave offset table, built with scalar writes -- the only construction that has
// ever measured 1005/1005. A GM-loaded table and a vector-built table both failed the whole
// contiguous path in ways that were never explained, so keep the proven form.
// The two 32-bit offsets of one complex pair are packed into a single 64-bit scalar store
// (table[2i] = low word, table[2i+1] = high word), halving the store count to planarFloats.
// Only 1/CROT_MASK_CHUNKS of the table is built: the offsets of chunk k are the base table's
// offsets plus k * chunkFloats bytes, which is exactly what Gather's srcBaseAddr adds.
__aicore__ __inline__ void CrotBuildMaskTable(LocalTensor<uint32_t> maskX, uint32_t planarFloats)
{
    LocalTensor<uint64_t> maskU64 = maskX.ReinterpretCast<uint64_t>();
    for (uint32_t i = 0; i < planarFloats / CROT_MASK_CHUNKS; i++) {
        const uint64_t lo = static_cast<uint64_t>(i * sizeof(float));
        const uint64_t hi = static_cast<uint64_t>((planarFloats + i) * sizeof(float));
        maskU64.SetValue(i, (hi << 32) | lo);
    }
    PIPE_BARRIER(ALL);
}

// Load one tile of x and y into the slot buffers. Only the store issued two slots back is
// waited on, so the store of tile t-1 overlaps the load of tile t.
__aicore__ __inline__ void CrotTileCopyIn(
    GlobalTensor<float> gmX, GlobalTensor<float> gmY, LocalTensor<float> xa, LocalTensor<float> ya, int64_t floatOffset,
    int32_t currentFloats, int32_t slot)
{
    // Event ids must stay compile-time constants: a runtime value defeats the pipeline
    // model and leaves the WAR dependency between the two slots unordered.
    if (slot == 0) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    } else {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    const uint32_t copyBytes = static_cast<uint32_t>(currentFloats) * static_cast<uint32_t>(sizeof(float));
    AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    DataCopyPad(xa, gmX[static_cast<uint64_t>(floatOffset)], copyParams, padParams);
    DataCopyPad(ya, gmY[static_cast<uint64_t>(floatOffset)], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

// Split the interleaved AoS tile into the four planar vectors.
__aicore__ __inline__ void CrotTileDeinterleave(
    CrotTileUb& ub, LocalTensor<float> xa, LocalTensor<float> ya, int32_t opFloats)
{
    // The planar intermediates (xr..nyi, t0..t3) are shared by both slots, so the previous
    // tile's V work (compute + Gather) must be fully drained before this tile's vreducev2
    // overwrites them. Without this barrier the deferred MTE3 wait lets two tiles' vector
    // work overlap and corrupt each other.
    PIPE_BARRIER(V);
    const uint16_t reduceRepeat = static_cast<uint16_t>(opFloats / 64);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(ub.xr.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(xa.GetPhyAddr()),
        nullptr, reduceRepeat, 1, 1, 8, 8);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(ub.xi.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(xa.GetPhyAddr()),
        nullptr, reduceRepeat, 1, 2, 8, 8);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(ub.yr.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(ya.GetPhyAddr()),
        nullptr, reduceRepeat, 1, 1, 8, 8);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(ub.yi.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(ya.GetPhyAddr()),
        nullptr, reduceRepeat, 1, 2, 8, 8);
    PIPE_BARRIER(V);
}

// Compute the four result planes from the input planes (two rounds of two independent
// rotations, sharing the t0..t3 scratch pair).
__aicore__ __inline__ void CrotTileRotatePlanes(
    CrotTileUb& ub, uint8_t repeat, float cosValue, float sinRe, float sinIm)
{
    // group 1: newXr (t0,t1) and newXi (t2,t3) are independent -> no barrier between.
    muls_v<ArchType::ASCEND_V220, float>(ub.nxr, ub.xr, cosValue, repeat, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.t0, ub.yr, sinRe, repeat, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.t1, ub.yi, sinIm, repeat, 1, 1, 8, 8);
    sub_v<ArchType::ASCEND_V220, float>(ub.t0, ub.t0, ub.t1, repeat, 1, 1, 1, 8, 8, 8);
    add_v<ArchType::ASCEND_V220, float>(ub.nxr, ub.nxr, ub.t0, repeat, 1, 1, 1, 8, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.nxi, ub.xi, cosValue, repeat, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.t2, ub.yi, sinRe, repeat, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.t3, ub.yr, sinIm, repeat, 1, 1, 8, 8);
    add_v<ArchType::ASCEND_V220, float>(ub.t2, ub.t2, ub.t3, repeat, 1, 1, 1, 8, 8, 8);
    add_v<ArchType::ASCEND_V220, float>(ub.nxi, ub.nxi, ub.t2, repeat, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);

    // group 2: newYr and newYi reuse the same scratch pair; independent of each other.
    muls_v<ArchType::ASCEND_V220, float>(ub.nyr, ub.yr, cosValue, repeat, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.t0, ub.xr, sinRe, repeat, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.t1, ub.xi, sinIm, repeat, 1, 1, 8, 8);
    add_v<ArchType::ASCEND_V220, float>(ub.t0, ub.t0, ub.t1, repeat, 1, 1, 1, 8, 8, 8);
    sub_v<ArchType::ASCEND_V220, float>(ub.nyr, ub.nyr, ub.t0, repeat, 1, 1, 1, 8, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.nyi, ub.yi, cosValue, repeat, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.t2, ub.xi, sinRe, repeat, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(ub.t3, ub.xr, sinIm, repeat, 1, 1, 8, 8);
    sub_v<ArchType::ASCEND_V220, float>(ub.t2, ub.t2, ub.t3, repeat, 1, 1, 1, 8, 8, 8);
    sub_v<ArchType::ASCEND_V220, float>(ub.nyi, ub.nyi, ub.t2, repeat, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
}

// Re-interleave the result planes back into the AoS staging buffers, chunk by chunk.
__aicore__ __inline__ void CrotTileReinterleave(
    CrotTileUb& ub, LocalTensor<float> xa, LocalTensor<float> ya, int32_t opFloats)
{
    constexpr int32_t chunkFloats = static_cast<int32_t>(2U * CROT_PLANAR_FLOATS / CROT_MASK_CHUNKS);
    for (uint32_t ch = 0; ch < CROT_MASK_CHUNKS; ++ch) {
        const int32_t base = static_cast<int32_t>(ch) * chunkFloats;
        if (base >= opFloats) {
            break;
        }
        const int32_t n = (opFloats - base) < chunkFloats ? (opFloats - base) : chunkFloats;
        // table[J] for J = chunk + base is table[chunk] + 2*C*ch bytes (the byte offsets grow
        // by 4 per PAIR of entries, i.e. 2 per entry), which is what srcBaseAddr adds.
        const uint32_t byteShift = 2U * static_cast<uint32_t>(base);
        AscendC::Gather(xa[base], ub.nxr, ub.maskX, byteShift, static_cast<uint32_t>(n));
        AscendC::Gather(ya[base], ub.nyr, ub.maskX, byteShift, static_cast<uint32_t>(n));
    }
    PIPE_BARRIER(V);
}

// Store one tile back to GM and release the slot for the load two tiles later.
__aicore__ __inline__ void CrotTileCopyOut(
    GlobalTensor<float> gmX, GlobalTensor<float> gmY, LocalTensor<float> xa, LocalTensor<float> ya, int64_t floatOffset,
    int32_t currentFloats, int32_t slot)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const uint32_t copyBytes = static_cast<uint32_t>(currentFloats) * static_cast<uint32_t>(sizeof(float));
    AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
    DataCopyPad(gmX[static_cast<uint64_t>(floatOffset)], xa, copyParams);
    DataCopyPad(gmY[static_cast<uint64_t>(floatOffset)], ya, copyParams);
    if (slot == 0) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

// Every element is handled by the SIMD path: a short final tile is padded up to a 64-complex
// (128-float) multiple for the vector/reduce ops, but only the valid elements are written
// back. The previous per-element scalar tail was ~100x slower per element and dominated the
// measured time for large n.
__aicore__ __inline__ void crot_process_contiguous_aiv(
    GlobalTensor<float> gmX, GlobalTensor<float> gmY, int32_t startIndex, int32_t elementPerCore, float cosValue,
    float sinRe, float sinIm)
{
    constexpr uint32_t planarFloats = CROT_PLANAR_FLOATS;
    CrotTileUb ub = CrotMakeTileUb();
    CrotBuildMaskTable(ub.maskX, planarFloats);

    const int32_t tileCap = static_cast<int32_t>(planarFloats);
    const int32_t numTiles = (elementPerCore + tileCap - 1) / tileCap;

    // Both MTE3->MTE2 flags start armed so the first "buffer free" wait on each slot returns
    // immediately (same pattern as the arch22 csrot pipeline).
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int32_t t = 0; t < numTiles; ++t) {
        const int32_t tileStart = t * tileCap;
        const int32_t remainElems = elementPerCore - tileStart;
        const int32_t currentElems = remainElems < tileCap ? remainElems : tileCap;
        const int32_t currentFloats = currentElems * static_cast<int32_t>(CROT_COMPONENTS);
        const int32_t opCount = ((currentElems + 63) / 64) * 64;
        const int32_t opFloats = opCount * static_cast<int32_t>(CROT_COMPONENTS);
        // int64 so a future call site cannot truncate the float index; the host rejects
        // n > INT32_MAX/2, so (startIndex + tileStart) itself still fits in int32.
        const int64_t floatOffset =
            static_cast<int64_t>(startIndex + tileStart) * static_cast<int64_t>(CROT_COMPONENTS);

        const int32_t slot = t & 1;
        LocalTensor<float> xa = (slot == 0) ? ub.xaos : ub.xaos2;
        LocalTensor<float> ya = (slot == 0) ? ub.yaos : ub.yaos2;

        CrotTileCopyIn(gmX, gmY, xa, ya, floatOffset, currentFloats, slot);
        CrotTileDeinterleave(ub, xa, ya, opFloats);
        CrotTileRotatePlanes(ub, static_cast<uint8_t>(opCount / 64), cosValue, sinRe, sinIm);
        CrotTileReinterleave(ub, xa, ya, opFloats);
        CrotTileCopyOut(gmX, gmY, xa, ya, floatOffset, currentFloats, slot);
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

// ==========================================================================
// Kernel entry
// ==========================================================================
extern "C" __global__ __aicore__ __vector__ void crot_kernel(
    GM_ADDR gm_x, GM_ADDR gm_y, int32_t elementCount, int32_t incx, int32_t incy, float cosValue, float sinRe,
    float sinIm)
{
    AscendC::SetMaskNorm();
    AscendC::SetAtomicNone();
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    const int16_t aivId = AscendC::GetBlockIdx();
    int16_t aivNum = AscendC::GetBlockNum();
    if (aivNum == 0) {
        aivNum = 1;
    }

    AscendC::GlobalTensor<float> gmX;
    AscendC::GlobalTensor<float> gmY;

    gmX.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gm_x));
    gmY.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gm_y));

    if (incx != 1 || incy != 1) {
        crot_process_strided_aiv(gmX, gmY, elementCount, incx, incy, cosValue, sinRe, sinIm);
        return;
    }

    const int32_t totalComplex = elementCount / static_cast<int32_t>(CROT_COMPONENTS);
    int32_t elementPerCore = totalComplex / aivNum;
    const int32_t elementRemain = totalComplex % aivNum;
    int32_t startIndex = aivId * elementPerCore;
    if (aivId < elementRemain) {
        startIndex += aivId;
        elementPerCore += 1;
    } else {
        startIndex += elementRemain;
    }
    if (elementPerCore <= 0) {
        return;
    }
    crot_process_contiguous_aiv(gmX, gmY, startIndex, elementPerCore, cosValue, sinRe, sinIm);
}

// ==========================================================================
// Host-side kernel launcher
// ==========================================================================
aclError crot_kernel_do(
    GM_ADDR x, GM_ADDR y, int32_t elementCount, int32_t incx, int32_t incy, float cosValue, float sinRe, float sinIm,
    uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    crot_kernel<<<numBlocks, nullptr, stream>>>(x, y, elementCount, incx, incy, cosValue, sinRe, sinIm);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
