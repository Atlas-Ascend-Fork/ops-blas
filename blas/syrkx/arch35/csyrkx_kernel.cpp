/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/helper/devkit_version_compat.h"
#if ASC_DEVKIT_GE_9_1
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "c_api/asc_simd.h"
#include "csyrkx_kernel.h"

namespace {
constexpr uint32_t THREADS = 256;

__simt_callee__ __aicore__ inline void AccumulateExactTriangle(
    __gm__ const float* a, __gm__ const float* b, const CsyrkxTilingData& t, uint32_t row, uint32_t col, float& real,
    float& imag)
{
#pragma clang fp contract(off)
    for (uint32_t l = 0; l < t.k; ++l) {
        const uint64_t ap =
            2 * (t.trans ? static_cast<uint64_t>(row) * t.lda + l : static_cast<uint64_t>(l) * t.lda + row);
        const uint64_t bp =
            2 * (t.trans ? static_cast<uint64_t>(col) * t.ldb + l : static_cast<uint64_t>(l) * t.ldb + col);
        const float xr = a[ap], xi = a[ap + 1];
        float yr = b[bp], yi = b[bp + 1];
        // Netlib's N path scales B before its inner accumulation;
        // T scales the accumulated dot product instead.
        if (!t.trans) {
            // Reference N skips a zero B entry before alpha*B.
            if (yr == 0.0f && yi == 0.0f) {
                continue;
            }
            // Copy only finite B at alpha==1. With nonfinite B,
            // zero-times-infinity is part of the reference result.
            if (!(t.ar == 1.0f && t.ai == 0.0f && isfinite(yr) && isfinite(yi))) {
                volatile float rr = t.ar * yr;
                volatile float ii = t.ai * yi;
                volatile float ri = t.ar * yi;
                volatile float ir = t.ai * yr;
                yr = rr - ii;
                yi = ri + ir;
            }
        }
        // The backend can contract despite contract(off). Volatile
        // materializes all four products before complex cancellation.
        volatile float rr = xr * yr;
        volatile float ii = xi * yi;
        volatile float ri = xr * yi;
        volatile float ir = xi * yr;
        real += rr - ii;
        imag += ri + ir;
    }
}

__simt_callee__ __aicore__ inline void ExactTriangle(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, const CsyrkxTilingData& t, uint32_t row,
    uint32_t col, uint64_t out, float& real, float& imag, bool& betaApplied)
{
#pragma clang fp contract(off)
    real = 0.0f;
    imag = 0.0f;
    if (!t.trans) {
        // N initializes C before adding terms. Moving beta*C to
        // the end can create spurious overflow despite finite C.
        betaApplied = true;
        if (!t.betaZero) {
            const float cr = c[out], ci = c[out + 1];
            if (t.br == 1.0f && t.bi == 0.0f) {
                real = cr;
                imag = ci;
            } else {
                volatile float rr = t.br * cr;
                volatile float ii = t.bi * ci;
                volatile float ri = t.br * ci;
                volatile float ir = t.bi * cr;
                real = rr - ii;
                imag = ri + ir;
            }
        }
    }
    AccumulateExactTriangle(a, b, t, row, col, real, imag);
    if (t.trans) {
        volatile float rr = t.ar * real;
        volatile float ii = t.ai * imag;
        volatile float ri = t.ar * imag;
        volatile float ir = t.ai * real;
        real = rr - ii;
        imag = ri + ir;
    }
}

__simt_callee__ __aicore__ inline void CombineTriangleProducts(
    __gm__ const float* p0, __gm__ const float* p1, __gm__ const float* p2, __gm__ const float* p3,
    const CsyrkxTilingData& t, uint32_t row, uint32_t col, float& real, float& imag)
{
    const uint64_t offset = static_cast<uint64_t>(col) * t.productLd + row;
    for (uint32_t part = 0; part < t.parts; ++part) {
        const uint64_t pos = offset + part * t.partStride;
        real += p0[pos] - p1[pos];
        imag += p2[pos] + p3[pos];
    }
    const float productReal = real;
    // Match the explicit SIMD Mul/Sub/Add predicate exactly. The
    // SIMT backend may otherwise fuse even with contract(off).
    volatile float rr = t.ar * productReal;
    volatile float ii = t.ai * imag;
    volatile float ri = t.ar * imag;
    volatile float ir = t.ai * productReal;
    real = rr - ii;
    imag = ri + ir;
}

__simt_callee__ __aicore__ inline void FinishTriangleBeta(
    __gm__ float* c, const CsyrkxTilingData& t, uint64_t out, bool betaApplied, float& real, float& imag)
{
    if (!t.betaZero && !betaApplied) {
        const float cr = c[out], ci = c[out + 1];
        if (!t.trans && t.br == 1.0f && t.bi == 0.0f) {
            // Netlib N skips beta multiplication at one; in particular,
            // do not contaminate an independent component with 0*Inf.
            real += cr;
            imag += ci;
        } else {
            volatile float rr = t.br * cr;
            volatile float ii = t.bi * ci;
            volatile float ri = t.br * ci;
            volatile float ir = t.bi * cr;
            real += rr - ii;
            imag += ri + ir;
        }
    }
}

// One SIMT lane owns an output element. No atomics or cross-block reductions.
template <bool Direct, bool RepairOnly = false, bool ForceExact = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) void UpdateTriangle(
    __gm__ const float* a, __gm__ const float* b, __gm__ const float* p0, __gm__ const float* p1,
    __gm__ const float* p2, __gm__ const float* p3, __gm__ float* c, CsyrkxTilingData t, uint32_t block,
    uint32_t blocks, uint64_t firstIndex, uint32_t count)
{
    // Repair owns one completed SIMD chunk; ForceExact owns an untouched
    // chunk. Direct keeps global strided ownership. No cross-block barrier.
    constexpr bool Scoped = RepairOnly || ForceExact;
    const uint64_t total = Scoped ? firstIndex + count : static_cast<uint64_t>(t.n) * t.n;
    const uint64_t step = Scoped ? blockDim.x : static_cast<uint64_t>(blocks) * blockDim.x;
    const uint64_t first = Scoped ? firstIndex : static_cast<uint64_t>(block) * blockDim.x;
    for (uint64_t index = first + threadIdx.x; index < total; index += step) {
        const uint32_t col = index / t.n;
        const uint32_t row = index - static_cast<uint64_t>(col) * t.n;
        if ((t.upper && row > col) || (!t.upper && row < col)) {
            continue;
        }
        const uint64_t out = 2 * (static_cast<uint64_t>(col) * t.ldc + row);
        float real = 0.0f;
        float imag = 0.0f;
        bool betaApplied = false;
        if (!t.scaleOnly) {
            // Direct and risk-flagged chunks must run the serial complex
            // recurrence even when the four real partial sums appear finite.
            if constexpr (!Direct && !ForceExact) {
                CombineTriangleProducts(p0, p1, p2, p3, t, row, col, real, imag);
                if (RepairOnly && isfinite(real) && isfinite(imag)) {
                    continue; // SIMD already wrote this lane, including beta*C.
                }
            }
            // Forced exact chunks have not written C at all. RepairOnly is
            // reserved for bounded-input chunks whose SIMD product is nonfinite.
            if (Direct || ForceExact || !isfinite(real) || !isfinite(imag)) {
                ExactTriangle(a, b, c, t, row, col, out, real, imag, betaApplied);
            }
        }
        FinishTriangleBeta(c, t, out, betaApplied, real, imag);
        c[out] = real;
        c[out + 1] = imag;
    }
}

// Eight real product planes, old complex C, complex output, and risk/control
// storage. Only selected rows are copied; no opposite-triangle/padding RMW.
constexpr uint32_t EPILOGUE_CHUNK = 2048;
constexpr uint32_t VECTOR_FLOATS = 64;
constexpr uint32_t OLD_C_OFFSET = 8 * EPILOGUE_CHUNK;
constexpr uint32_t OUTPUT_OFFSET = 10 * EPILOGUE_CHUNK;
constexpr uint32_t CONTROL_OFFSET = 12 * EPILOGUE_CHUNK;
constexpr uint32_t RISK_CONTROL_WORDS = 1024;
constexpr uint32_t EPILOGUE_UB_FLOATS = CONTROL_OFFSET + RISK_CONTROL_WORDS;
constexpr uint32_t FP32_EXPONENT_MASK = 0x7f800000u;
constexpr uint32_t FP32_ABSOLUTE_MASK = 0x7fffffffu;
constexpr uint32_t SAFE_MAGNITUDE_BITS = 0x49800000u; // exactly 2^20
static_assert(EPILOGUE_UB_FLOATS * sizeof(float) <= 192 * 1024, "Epilogue UB budget exceeded");

// Queue one chunk's four Cube product planes into a UB plane slot. The one-part
// layout has two disjoint slots so the next chunk's GM->UB copy overlaps the
// current chunk's reduction; the two-part layout uses the whole plane area.
__aicore__ inline void CopyProductChunk(
    AscendC::GlobalTensor<float> (&products)[4], AscendC::LocalTensor<float>& buffer, const CsyrkxTilingData& t,
    uint32_t col, uint32_t row, uint32_t count, uint32_t slotBase, const AscendC::DataCopyPadExtParams<float>& pad)
{
    using namespace AscendC;
    const uint64_t index = static_cast<uint64_t>(col) * t.productLd + row;
    const DataCopyExtParams planeCopy{1, count * uint32_t(sizeof(float)), 0, 0, 0};
    for (uint32_t part = 0; part < t.parts; ++part) {
        for (uint32_t product = 0; product < 4; ++product) {
            DataCopyPad(
                buffer[slotBase + (4 * part + product) * EPILOGUE_CHUNK],
                products[product][index + part * t.partStride], planeCopy, pad);
        }
    }
}

// Every producer initializes its whole slot. Mask the unused lanes in the
// final batch so a previously loaded slot cannot affect the reduction.
__simd_vf__ inline void ReduceInputRisk(__ubuf__ uint32_t* control, uint32_t count)
{
    namespace V = AscendC::MicroAPI;
    auto full = V::CreateMask<uint32_t, V::MaskPattern::ALL>();
    uint32_t remaining = count;
    V::RegTensor<uint32_t> values, zero, largest;
    V::Duplicate(zero, uint32_t(0));
    V::Duplicate(largest, uint32_t(0));
    for (uint32_t offset = 0; offset < count; offset += VECTOR_FLOATS) {
        auto mask = V::UpdateMask<uint32_t>(remaining);
        V::LoadAlign(values, control + offset);
        V::Select(values, values, zero, mask);
        V::Max(largest, largest, values, full);
    }
    V::Reduce<V::ReduceType::MAX>(largest, largest, full);
    V::StoreAlign(control, largest, full);
}

// Load producer metadata through DMA, not scalar cached GM reads. This also
// avoids observing stale flags when the same workspace is reused by a repeat.
__aicore__ inline bool InputRisk(GM_ADDR riskFlags, uint32_t blocks, AscendC::LocalTensor<uint32_t> control)
{
    using namespace AscendC;
    GlobalTensor<uint32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(riskFlags));
    const DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
    bool risk = false;
    if (blocks == 0) {
        return false;
    }
    constexpr uint32_t slotsPerLoad = RISK_CONTROL_WORDS / CSYRKX_RISK_SLOT_WORDS;
    // LoadAlign reads a full register even for the last partial batch. Ensure
    // every UB lane is initialized before the first DMA and SIMD reduction.
    Duplicate(control, uint32_t(0), RISK_CONTROL_WORDS);
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    for (uint32_t first = 0; first < blocks; first += slotsPerLoad) {
        const uint32_t count = blocks - first < slotsPerLoad ? blocks - first : slotsPerLoad;
        const DataCopyExtParams copy{1, count * CSYRKX_RISK_SLOT_WORDS * uint32_t(sizeof(uint32_t)), 0, 0, 0};
        DataCopyPad(control, flags[first * CSYRKX_RISK_SLOT_WORDS], copy, pad);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ReduceInputRisk(reinterpret_cast<__ubuf__ uint32_t*>(control.GetPhyAddr()), count * CSYRKX_RISK_SLOT_WORDS);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        risk = (control.GetValue(0) != 0) || risk;
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
    return risk;
}

// Inspect original selected C before ANY output is written. A risky C chunk
// needs every lane recomputed, not just lanes whose alpha*product is nonfinite.
__simd_vf__ inline void BoundOldC(__ubuf__ float* buffer, uint32_t count)
{
    namespace V = AscendC::MicroAPI;
    auto full = V::CreateMask<uint32_t, V::MaskPattern::ALL>();
    V::RegTensor<uint32_t> absoluteMask, zero, largest;
    V::Duplicate(absoluteMask, FP32_ABSOLUTE_MASK);
    V::Duplicate(zero, uint32_t(0));
    V::Duplicate(largest, uint32_t(0));
    uint32_t remaining = 2 * count;
    for (uint32_t offset = 0; offset < 2 * count; offset += VECTOR_FLOATS) {
        auto mask = V::UpdateMask<uint32_t>(remaining);
        V::RegTensor<uint32_t> value;
        V::LoadAlign(value, reinterpret_cast<__ubuf__ uint32_t*>(buffer + OLD_C_OFFSET + offset));
        V::And(value, value, absoluteMask, mask);
        V::Select(value, value, zero, mask);
        V::Max(largest, largest, value, full);
    }
    V::Reduce<V::ReduceType::MAX>(largest, largest, full);
    V::StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(buffer + CONTROL_OFFSET), largest, full);
}

template <typename Register, typename Mask>
__simd_callee__ inline void ApplyEpilogueBeta(
    __ubuf__ float* buffer, uint32_t offset, float br, float bi, bool betaOneN, Register& tr, Register& ti, Mask mask,
    Mask finite)
{
    namespace V = AscendC::MicroAPI;
    V::RegTensor<float> cr, ci, rr, ii;
    V::LoadAlign<float, V::LoadDist::DIST_DINTLV_B32>(cr, ci, buffer + OLD_C_OFFSET + 2 * offset);
    if (betaOneN) {
        // Netlib N copies beta==1 C; multiplying 0*Inf would corrupt
        // its other component. T/C deliberately retain multiplication.
        V::Add(tr, tr, cr, mask);
        V::Add(ti, ti, ci, mask);
    } else {
        V::Muls(rr, cr, br, mask);
        V::Muls(ii, ci, bi, mask);
        V::Sub(rr, rr, ii, mask);
        V::Add(tr, tr, rr, mask);
        V::Muls(rr, ci, br, mask);
        V::Muls(ii, cr, bi, mask);
        V::Add(rr, rr, ii, mask);
        V::Add(ti, ti, rr, mask);
    }
    // Repair must still see the ORIGINAL C for affected lanes. Finite
    // lanes are already final and the cold SIMT pass skips them.
    V::Select(tr, tr, cr, finite);
    V::Select(ti, ti, ci, finite);
}

template <uint32_t Parts, bool BetaZero>
__simd_vf__ inline void BuildEpilogueChunk(
    __ubuf__ float* planes, __ubuf__ float* buffer, uint32_t count, float ar, float ai, float br, float bi,
    bool betaOneN)
{
#pragma clang fp contract(off)
    namespace V = AscendC::MicroAPI;
    auto full = V::CreateMask<float, V::MaskPattern::ALL>();
    V::RegTensor<uint32_t> exponentMask, zero, largest;
    V::Duplicate(exponentMask, FP32_EXPONENT_MASK);
    V::Duplicate(zero, uint32_t(0));
    V::Duplicate(largest, uint32_t(0));
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += VECTOR_FLOATS) {
        auto mask = V::UpdateMask<float>(remaining);
        V::RegTensor<float> real, imag, x, y, tr, ti, temporary;
        V::Duplicate(real, 0.0f);
        V::Duplicate(imag, 0.0f);
        // Keep (rr0-ii0)+(rr1-ii1), not (rr0+rr1)-(ii0+ii1).
        for (uint32_t part = 0; part < Parts; ++part) {
            const uint32_t base = 4 * part * EPILOGUE_CHUNK + offset;
            V::LoadAlign(x, planes + base);
            V::LoadAlign(y, planes + base + EPILOGUE_CHUNK);
            V::Sub(temporary, x, y, mask);
            V::Add(real, real, temporary, mask);
            V::LoadAlign(x, planes + base + 2 * EPILOGUE_CHUNK);
            V::LoadAlign(y, planes + base + 3 * EPILOGUE_CHUNK);
            V::Add(temporary, x, y, mask);
            V::Add(imag, imag, temporary, mask);
        }
        V::Muls(tr, real, ar, mask);
        V::Muls(temporary, imag, ai, mask);
        V::Sub(tr, tr, temporary, mask);
        V::Muls(ti, imag, ar, mask);
        V::Muls(temporary, real, ai, mask);
        V::Add(ti, ti, temporary, mask);

        // Exponent bits detect both NaN and infinity before beta*C, exactly
        // where the original repair predicate is evaluated. Inactive tail
        // lanes contribute zero to the block-local control reduction.
        V::RegTensor<uint32_t> er, ei;
        V::And(er, reinterpret_cast<V::RegTensor<uint32_t>&>(tr), exponentMask, mask);
        V::And(ei, reinterpret_cast<V::RegTensor<uint32_t>&>(ti), exponentMask, mask);
        V::Max(er, er, ei, mask);
        V::MaskReg finite;
        V::Compares<uint32_t, AscendC::CMPMODE::NE>(finite, er, FP32_EXPONENT_MASK, mask);
        V::Select(er, er, zero, mask);
        V::Max(largest, largest, er, full);
        if constexpr (!BetaZero) {
            ApplyEpilogueBeta(buffer, offset, br, bi, betaOneN, tr, ti, mask, finite);
        }
        V::RegTensor<float> lo, hi;
        V::Interleave(lo, hi, tr, ti);
        V::StoreAlign(buffer + OUTPUT_OFFSET + 2 * offset, lo, full);
        V::StoreAlign(buffer + OUTPUT_OFFSET + 2 * offset + VECTOR_FLOATS, hi, full);
    }
    V::Reduce<V::ReduceType::MAX>(largest, largest, full);
    V::StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(buffer + CONTROL_OFFSET), largest, full);
}

// Selected-row length of one triangle chunk; never spans the opposite triangle.
__aicore__ inline uint32_t EpilogueCount(const CsyrkxTilingData& t, uint32_t col, uint32_t row)
{
    const uint32_t end = t.upper ? col + 1 : t.n;
    const uint32_t left = end - row;
    return left < EPILOGUE_CHUNK ? left : EPILOGUE_CHUNK;
}

struct EpilogueChunkContext {
    GM_ADDR a;
    GM_ADDR b;
    GM_ADDR p0;
    GM_ADDR p1;
    GM_ADDR p2;
    GM_ADDR p3;
    GM_ADDR c;
    const CsyrkxTilingData& t;
    AscendC::GlobalTensor<float>& output;
    AscendC::GlobalTensor<float> (&products)[4];
    AscendC::LocalTensor<float>& buffer;
    AscendC::LocalTensor<uint32_t>& control;
    __ubuf__ float* ub;
};

__aicore__ inline void DispatchEpilogueChunk(
    __ubuf__ float* ub, const CsyrkxTilingData& t, uint32_t slotBase, uint32_t count, bool betaOneN)
{
    if (t.parts == 1) {
        if (t.betaZero) {
            BuildEpilogueChunk<1, true>(ub + slotBase, ub, count, t.ar, t.ai, t.br, t.bi, betaOneN);
        } else {
            BuildEpilogueChunk<1, false>(ub + slotBase, ub, count, t.ar, t.ai, t.br, t.bi, betaOneN);
        }
    } else {
        if (t.betaZero) {
            BuildEpilogueChunk<2, true>(ub, ub, count, t.ar, t.ai, t.br, t.bi, betaOneN);
        } else {
            BuildEpilogueChunk<2, false>(ub, ub, count, t.ar, t.ai, t.br, t.bi, betaOneN);
        }
    }
}

__aicore__ inline void ExecuteEpilogueChunk(
    EpilogueChunkContext& ctx, uint32_t col, uint32_t row, uint32_t count, uint64_t out, uint32_t slotBase,
    bool betaOneN, bool forceExact, const AscendC::DataCopyExtParams& complexCopy)
{
    using namespace AscendC;
    const auto& t = ctx.t;
    auto& output = ctx.output;
    auto& buffer = ctx.buffer;
    auto& control = ctx.control;
    auto* ub = ctx.ub;
    GM_ADDR a = ctx.a;
    GM_ADDR b = ctx.b;
    GM_ADDR p0 = ctx.p0;
    GM_ADDR p1 = ctx.p1;
    GM_ADDR p2 = ctx.p2;
    GM_ADDR p3 = ctx.p3;
    GM_ADDR c = ctx.c;
    if (forceExact) {
        // No SIMD output was written: every lane still sees ORIGINAL
        // C. A finite Cube result is not evidence of a safe recurrence.
        // betaZero never reads C, including this forced-exact branch.
        asc_vf_call<UpdateTriangle<false, false, true>>(
            dim3{THREADS, 1, 1}, reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(b),
            reinterpret_cast<__gm__ const float*>(p0), reinterpret_cast<__gm__ const float*>(p1),
            reinterpret_cast<__gm__ const float*>(p2), reinterpret_cast<__gm__ const float*>(p3),
            reinterpret_cast<__gm__ float*>(c), t, 0, 1, static_cast<uint64_t>(col) * t.n + row, count);
    } else {
        DispatchEpilogueChunk(ub, t, slotBase, count, betaOneN);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(output[out], buffer[OUTPUT_OFFSET], complexCopy);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        const bool repair = control.GetValue(0) == FP32_EXPONENT_MASK;
        // Complete this precise store before SIMT can read its preserved C,
        // and before the next chunk may reuse the UB output region.
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
        if (repair) {
            asc_vf_call<UpdateTriangle<false, true>>(
                dim3{THREADS, 1, 1}, reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(b),
                reinterpret_cast<__gm__ const float*>(p0), reinterpret_cast<__gm__ const float*>(p1),
                reinterpret_cast<__gm__ const float*>(p2), reinterpret_cast<__gm__ const float*>(p3),
                reinterpret_cast<__gm__ float*>(c), t, 0, 1, static_cast<uint64_t>(col) * t.n + row, count);
        }
    }
}

__aicore__ inline bool NextEpilogueChunk(
    const CsyrkxTilingData& t, uint32_t col, uint32_t row, uint32_t& nextCol, uint32_t& nextRow)
{
    using namespace AscendC;
    const uint32_t end = t.upper ? col + 1 : t.n;
    // Enumerate the following chunk before waiting: one SetFlag per slot is
    // always matched by its WaitFlag before that slot id is set again.
    nextCol = col;
    nextRow = row;
    bool haveNext = false;
    if (row + EPILOGUE_CHUNK < end) {
        nextRow = row + EPILOGUE_CHUNK;
        haveNext = true;
    } else if (col + GetBlockNum() < t.n) {
        nextCol = col + GetBlockNum();
        nextRow = t.upper ? 0 : nextCol;
        haveNext = true;
    }
    return haveNext;
}

__aicore__ inline bool EpilogueNeedsExact(
    EpilogueChunkContext& ctx, bool inputRisk, uint32_t count, uint64_t out,
    const AscendC::DataCopyExtParams& complexCopy, const AscendC::DataCopyPadExtParams<float>& pad)
{
    using namespace AscendC;
    const auto& t = ctx.t;
    auto& output = ctx.output;
    auto& buffer = ctx.buffer;
    auto& control = ctx.control;
    auto* ub = ctx.ub;
    bool forceExact = inputRisk;
    if (!forceExact && !t.betaZero) {
        // Old C uses its own MTE2->V flag id: the two product slots keep
        // ids 0/1, so a pending product prefetch can never be consumed here.
        DataCopyPad(buffer[OLD_C_OFFSET], output[out], complexCopy, pad);
        SetFlag<HardEvent::MTE2_V>(2);
        WaitFlag<HardEvent::MTE2_V>(2);
        BoundOldC(ub, count);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        forceExact = control.GetValue(0) > SAFE_MAGNITUDE_BITS;
    }
    return forceExact;
}

__aicore__ inline void PrepareEpilogueProducts(
    EpilogueChunkContext& ctx, const AscendC::DataCopyPadExtParams<float>& pad, bool pipelined, bool haveNext,
    uint32_t col, uint32_t row, uint32_t count, uint32_t nextCol, uint32_t nextRow, uint32_t slotBase,
    uint32_t nextBase)
{
    using namespace AscendC;
    auto& products = ctx.products;
    auto& buffer = ctx.buffer;
    const auto& t = ctx.t;
    if (pipelined && haveNext) {
        CopyProductChunk(products, buffer, t, nextCol, nextRow, EpilogueCount(t, nextCol, nextRow), nextBase, pad);
        SetFlag<HardEvent::MTE2_V>(nextBase == 0 ? 0 : 1);
    }
    if (pipelined) {
        WaitFlag<HardEvent::MTE2_V>(slotBase == 0 ? 0 : 1);
    } else {
        CopyProductChunk(products, buffer, t, col, row, count, slotBase, pad);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }
}

__aicore__ inline void RunVectorEpilogue(
    GM_ADDR a, GM_ADDR b, GM_ADDR p0, GM_ADDR p1, GM_ADDR p2, GM_ADDR p3, GM_ADDR riskFlags, GM_ADDR c,
    const CsyrkxTilingData& t)
{
    using namespace AscendC;
    GlobalTensor<float> products[4];
    products[0].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p0));
    products[1].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p1));
    products[2].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p2));
    products[3].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p3));
    GlobalTensor<float> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    LocalTensor<float> buffer(TPosition::VECCALC, 0, EPILOGUE_UB_FLOATS);
    LocalTensor<uint32_t> control(TPosition::VECCALC, CONTROL_OFFSET * sizeof(float), RISK_CONTROL_WORDS);
    auto* ub = reinterpret_cast<__ubuf__ float*>(buffer.GetPhyAddr());
    const bool betaOneN = !t.trans && t.br == 1.0f && t.bi == 0.0f;
    const bool inputRisk = InputRisk(riskFlags, t.riskBlocks, control);
    const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    EpilogueChunkContext ctx{a, b, p0, p1, p2, p3, c, t, output, products, buffer, control, ub};
    // A one-part chunk owns four planes, so the plane area holds two disjoint
    // slots: the next chunk's GM->UB copy is in flight while the current chunk
    // is reduced. Two-part chunks need all eight planes and issue once, exactly
    // as before. Old C, output and control offsets are shared and unchanged.
    const bool pipelined = t.parts == 1;
    uint32_t col = GetBlockIdx();
    if (col >= t.n)
        return;
    uint32_t row = t.upper ? 0 : col;
    uint32_t slotBase = 0;
    if (pipelined) {
        CopyProductChunk(products, buffer, t, col, row, EpilogueCount(t, col, row), slotBase, pad);
        SetFlag<HardEvent::MTE2_V>(0);
    }
    for (;;) {
        const uint32_t count = EpilogueCount(t, col, row);
        const uint64_t out = 2 * (static_cast<uint64_t>(col) * t.ldc + row);
        const DataCopyExtParams complexCopy{1, 2 * count * uint32_t(sizeof(float)), 0, 0, 0};
        uint32_t nextCol;
        uint32_t nextRow;
        const bool haveNext = NextEpilogueChunk(t, col, row, nextCol, nextRow);
        const uint32_t nextBase = slotBase == 0 ? 4 * EPILOGUE_CHUNK : 0;
        PrepareEpilogueProducts(ctx, pad, pipelined, haveNext, col, row, count, nextCol, nextRow, slotBase, nextBase);
        const bool forceExact = EpilogueNeedsExact(ctx, inputRisk, count, out, complexCopy, pad);
        ExecuteEpilogueChunk(ctx, col, row, count, out, slotBase, betaOneN, forceExact, complexCopy);
        // A slot can be reused two iterations later. Wait for both its MTE2
        // writes and any V reads before another MTE2 write, including the
        // ForceExact path where the product planes are not consumed by V.
        PipeBarrier<PIPE_MTE2>();
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        if (!haveNext) {
            break;
        }
        col = nextCol;
        row = nextRow;
        if (pipelined) {
            slotBase = nextBase;
        }
    }
}

extern "C" __global__ __aicore__ void csyrkx_epilogue_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR p0, GM_ADDR p1, GM_ADDR p2, GM_ADDR p3, GM_ADDR riskFlags, GM_ADDR c,
    CsyrkxTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunVectorEpilogue(a, b, p0, p1, p2, p3, riskFlags, c, t);
}

extern "C" __global__ __aicore__ void csyrkx_direct_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CsyrkxTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<UpdateTriangle<true>>(
        dim3{THREADS, 1, 1}, reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(b),
        static_cast<__gm__ const float*>(nullptr), static_cast<__gm__ const float*>(nullptr),
        static_cast<__gm__ const float*>(nullptr), static_cast<__gm__ const float*>(nullptr),
        reinterpret_cast<__gm__ float*>(c), t, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), 0, 0);
}
} // namespace

void csyrkx_epilogue_do(
    uint32_t blocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* p0, uint8_t* p1, uint8_t* p2,
    uint8_t* p3, const uint8_t* riskFlags, uint8_t* c, const CsyrkxTilingData& t)
{
    csyrkx_epilogue_kernel<<<blocks, nullptr, stream>>>(
        const_cast<uint8_t*>(a), const_cast<uint8_t*>(b), p0, p1, p2, p3, const_cast<uint8_t*>(riskFlags), c, t);
}

void csyrkx_direct_do(
    uint32_t blocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* c, const CsyrkxTilingData& t)
{
    csyrkx_direct_kernel<<<blocks, nullptr, stream>>>(const_cast<uint8_t*>(a), const_cast<uint8_t*>(b), c, t);
}
#endif
