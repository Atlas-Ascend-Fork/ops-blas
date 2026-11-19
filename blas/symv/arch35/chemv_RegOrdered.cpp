/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#include "chemv_tiling_data.h"

#ifdef __clang__
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif

// Keep the two FP32 products rounded before the complex add/subtract. The
// temporary barriers are intentional: device precision is checked against the
// reference BLAS grouping, including cancellation-sensitive output components.
__simt_callee__ __aicore__ inline float ChemvMulSub(float a, float b, float c, float d)
{
    const float first = ::fmaf(a, b, 0.0f);
    const float second = ::fmaf(c, d, 0.0f);
    return first - second;
}

__simt_callee__ __aicore__ inline float ChemvMulAdd(float a, float b, float c, float d)
{
    const float first = ::fmaf(a, b, 0.0f);
    const float second = ::fmaf(c, d, 0.0f);
    return first + second;
}

__simt_callee__ __aicore__ inline void ChemvDirectRow(
    uint32_t row, uint32_t lda, uint32_t directBegin, uint32_t directEnd, int64_t xStart, int64_t incx, float ar,
    float ai, __gm__ const float* a, __gm__ const float* x, float& real, float& imag)
{
    for (uint32_t col = directBegin; col < directEnd; ++col) {
        const int64_t xi = 2 * (xStart + static_cast<int64_t>(col) * incx);
        const float xr = x[xi];
        const float ximag = x[xi + 1];
        const float tr = ChemvMulSub(ar, xr, ai, ximag);
        const float ti = ChemvMulAdd(ar, ximag, ai, xr);
        const int64_t index = 2 * (static_cast<int64_t>(col) * lda + row);
        const float mr = a[index];
        const float mi = a[index + 1];
        real += ChemvMulSub(tr, mr, ti, mi);
        imag += ChemvMulAdd(tr, mi, ti, mr);
    }
}

template <bool UPPER>
__simt_callee__ __aicore__ inline void ChemvAlphaRow(
    uint32_t row, uint32_t n, uint32_t lda, int64_t xStart, int64_t incx, float ar, float ai, __gm__ const float* a,
    __gm__ const float* x, float& real, float& imag)
{
    // Preserve the reference BLAS grouping per output row. Moving alpha
    // outside a full dot product changes cancellation error materially.
    // Each thread owns y[row]; no atomics or cross-core reduction.
    const uint32_t directBegin = UPPER ? row + 1 : 0;
    const uint32_t directEnd = UPPER ? n : row;
    if constexpr (!UPPER) {
        ChemvDirectRow(row, lda, directBegin, directEnd, xStart, incx, ar, ai, a, x, real, imag);
    }
    const int64_t selfX = 2 * (xStart + static_cast<int64_t>(row) * incx);
    const float selfXReal = x[selfX];
    const float selfXImag = x[selfX + 1];
    const float selfReal = ChemvMulSub(ar, selfXReal, ai, selfXImag);
    const float selfImag = ChemvMulAdd(ar, selfXImag, ai, selfXReal);
    // Only the real component of the diagonal is loaded.
    const float diagonal = a[2 * (static_cast<int64_t>(row) * lda + row)];
    real += selfReal * diagonal;
    imag += selfImag * diagonal;
    float sumReal = 0.0f;
    float sumImag = 0.0f;
    const uint32_t mirrorBegin = UPPER ? 0 : row + 1;
    const uint32_t mirrorEnd = UPPER ? row : n;
    for (uint32_t col = mirrorBegin; col < mirrorEnd; ++col) {
        const int64_t index = 2 * (static_cast<int64_t>(row) * lda + col);
        const int64_t xi = 2 * (xStart + static_cast<int64_t>(col) * incx);
        const float mr = a[index];
        const float mi = a[index + 1];
        const float xr = x[xi];
        const float ximag = x[xi + 1];
        sumReal += ChemvMulAdd(mr, xr, mi, ximag);
        sumImag += ChemvMulSub(mr, ximag, mi, xr);
    }
    real += ChemvMulSub(ar, sumReal, ai, sumImag);
    imag += ChemvMulAdd(ar, sumImag, ai, sumReal);
    if constexpr (UPPER) {
        ChemvDirectRow(row, lda, directBegin, directEnd, xStart, incx, ar, ai, a, x, real, imag);
    }
}

template <bool UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(128) inline void ChemvCompute(
    uint32_t n, uint32_t lda, int64_t incx, int64_t incy, float ar, float ai, float br, float bi, __gm__ const float* a,
    __gm__ const float* x, __gm__ float* y)
{
    const int64_t xStart = incx < 0 ? (1 - static_cast<int64_t>(n)) * incx : 0;
    const int64_t yStart = incy < 0 ? (1 - static_cast<int64_t>(n)) * incy : 0;
    const bool alphaZero = ar == 0.0f && ai == 0.0f;
    const bool betaZero = br == 0.0f && bi == 0.0f;
    const bool betaOne = br == 1.0f && bi == 0.0f;
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        float real = 0.0f;
        float imag = 0.0f;
        const int64_t yi = 2 * (yStart + static_cast<int64_t>(row) * incy);
        if (!betaZero) {
            const float yr = y[yi];
            const float yimag = y[yi + 1];
            if (betaOne) {
                real += yr;
                imag += yimag;
            } else {
                real += ChemvMulSub(br, yr, bi, yimag);
                imag += ChemvMulAdd(br, yimag, bi, yr);
            }
        }
        if (!alphaZero) {
            ChemvAlphaRow<UPPER>(row, n, lda, xStart, incx, ar, ai, a, x, real, imag);
        }
        y[yi] = real;
        y[yi + 1] = imag;
    }
}

using namespace AscendC;
namespace {
// CHEMV_REG_ROWS comes from chemv_tiling_data.h: the shared definition keeps the
// kernel register tile and the host rowsPerBlock bound from drifting apart.
constexpr uint32_t CHEMV_REG_COLS = 64U;
constexpr uint32_t CHEMV_REG_PITCH = 136U;
constexpr uint32_t CHEMV_REG_MATRIX_SLOTS = CHEMV_REG_ROWS > CHEMV_REG_COLS ? CHEMV_REG_ROWS : CHEMV_REG_COLS;
static_assert(CHEMV_REG_COLS > 0U && CHEMV_REG_COLS <= 64U, "x buffer and mirror pitch limit");
static_assert(CHEMV_REG_PITCH >= 2U * CHEMV_REG_COLS, "mirror run must fit its UB slot");

// All operations are separate FP32 register instructions, without horizontal reduction.
__simd_callee__ inline void ChemvRegAlphaOne(
    Reg::RegTensor<float>& tr, Reg::RegTensor<float>& ti, Reg::RegTensor<float>& xr, Reg::RegTensor<float>& xi,
    Reg::MaskReg& mask)
{
    Reg::RegTensor<float> tmp;
    Reg::Muls(tr, xr, 1.0f, mask);
    Reg::Muls(tmp, xi, 0.0f, mask);
    Reg::Sub(tr, tr, tmp, mask);
    Reg::Muls(ti, xi, 1.0f, mask);
    Reg::Muls(tmp, xr, 0.0f, mask);
    Reg::Add(ti, ti, tmp, mask);
}

// One SIMD lane per input column computes the exact same alpha1 grouping once.
// Raw x stays separate for the Hermitian mirror phase.
__simd_vf__ inline void ChemvRegPrefactorVf(__ubuf__ float* rawX, __ubuf__ float* factoredX, uint32_t colCount)
{
    using namespace AscendC::Reg;
    RegTensor<float> xr, xi, tr, ti;
    MaskReg active = UpdateMask<float>(colCount);
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(xr, xi, rawX);
    ChemvRegAlphaOne(tr, ti, xr, xi, active);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(factoredX, tr, ti, all);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_callee__ inline void ChemvRegMirrorOffsets(
    Reg::RegTensor<int32_t>& lanes, Reg::RegTensor<int32_t>& pairBase0, Reg::RegTensor<int32_t>& pairBase1,
    Reg::MaskReg& packedMask0, Reg::MaskReg& packedMask1, Reg::MaskReg& all, uint32_t originalRows)
{
    using namespace AscendC::Reg;
    RegTensor<int32_t> halves;
    RegTensor<int32_t> odd;
    // One register holds 32 adjacent complex pairs, not 64 separated real words.
    ShiftRights(halves, lanes, static_cast<int16_t>(1), all);
    Muls(odd, halves, 2, all);
    Sub(odd, lanes, odd, all);
    Muls(pairBase0, halves, static_cast<int32_t>(CHEMV_REG_PITCH), all);
    Add(pairBase0, pairBase0, odd, all);
    Adds(pairBase1, pairBase0, static_cast<int32_t>(32U * CHEMV_REG_PITCH), all);
    uint32_t valid0 = originalRows < 32U ? 2U * originalRows : 64U;
    uint32_t valid1 = originalRows > 32U ? 2U * (originalRows - 32U) : 0U;
    packedMask0 = UpdateMask<float>(valid0);
    packedMask1 = UpdateMask<float>(valid1);
}

template <bool UPPER, bool MIRROR, bool FULL>
__simd_callee__ inline void ChemvRegLoadColumn(
    Reg::RegTensor<float>& ar, Reg::RegTensor<float>& ai, Reg::RegTensor<int32_t>& offsets,
    Reg::RegTensor<int32_t>& lanes, Reg::RegTensor<int32_t>& starts, Reg::RegTensor<int32_t>& pairBase0,
    Reg::RegTensor<int32_t>& pairBase1, Reg::MaskReg& packedMask0, Reg::MaskReg& packedMask1, Reg::MaskReg& active,
    Reg::MaskReg& all, __ubuf__ float* a, uint16_t c, uint32_t col, uint32_t rowBase)
{
    using namespace AscendC::Reg;
    RegTensor<float> packed0, packed1;
    if constexpr (MIRROR) {
        if constexpr (FULL) {
            Adds(offsets, pairBase0, static_cast<int32_t>(2U * c), all);
            Gather(packed0, a, reinterpret_cast<RegTensor<uint32_t>&>(offsets), packedMask0);
            Adds(offsets, pairBase1, static_cast<int32_t>(2U * c), all);
            Gather(packed1, a, reinterpret_cast<RegTensor<uint32_t>&>(offsets), packedMask1);
            DeInterleave<float>(ar, ai, packed0, packed1);
        } else {
            Muls(offsets, lanes, static_cast<int32_t>(CHEMV_REG_PITCH), all);
            if constexpr (!UPPER) {
                RegTensor<int32_t> shift;
                Muls(shift, starts, 2, all);
                Sub(offsets, offsets, shift, all);
            }
            Adds(offsets, offsets, static_cast<int32_t>(2U * c), all);
            Gather(ar, a, reinterpret_cast<RegTensor<uint32_t>&>(offsets), active);
            Adds(offsets, offsets, 1, all);
            Gather(ai, a, reinterpret_cast<RegTensor<uint32_t>&>(offsets), active);
        }
    } else {
        const uint32_t start = !FULL && !UPPER && col >= rowBase ? col + 1U - rowBase : 0U;
        if (start == 0U) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(ar, ai, a + c * CHEMV_REG_PITCH);
        } else {
            Muls(offsets, lanes, 2, all);
            Adds(offsets, offsets, -static_cast<int32_t>(2U * start), all);
            Gather(ar, a + c * CHEMV_REG_PITCH, reinterpret_cast<RegTensor<uint32_t>&>(offsets), active);
            Adds(offsets, offsets, 1, all);
            Gather(ai, a + c * CHEMV_REG_PITCH, reinterpret_cast<RegTensor<uint32_t>&>(offsets), active);
        }
    }
}

template <bool FULL>
__simd_callee__ inline void ChemvRegAddContribution(
    Reg::RegTensor<float>& value, Reg::RegTensor<float>& contribution, Reg::MaskReg& active)
{
    using namespace AscendC::Reg;
    if constexpr (FULL) {
        Add(value, value, contribution, active);
    } else {
        RegTensor<float> next;
        Add(next, value, contribution, active);
        Select(value, next, value, active);
    }
}

template <bool MIRROR, bool FULL>
__simd_callee__ inline void ChemvRegAccumulateColumn(
    Reg::RegTensor<float>& real, Reg::RegTensor<float>& imag, Reg::RegTensor<float>& ar, Reg::RegTensor<float>& ai,
    __ubuf__ float* x, uint16_t c, Reg::MaskReg& active)
{
    using namespace AscendC::Reg;
    RegTensor<float> xr, xi, tr, ti, p, q, contribution;
    if constexpr (MIRROR) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(xr, x + 2U * c);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(xi, x + 2U * c + 1U);
        Mul(p, ar, xr, active);
        Mul(q, ai, xi, active);
        Add(contribution, p, q, active);
        ChemvRegAddContribution<FULL>(real, contribution, active);
        Mul(p, ar, xi, active);
        Mul(q, ai, xr, active);
        Sub(contribution, p, q, active);
        ChemvRegAddContribution<FULL>(imag, contribution, active);
    } else {
        // x points to prefactored values; raw x is preserved for MIRROR.
        LoadAlign<float, LoadDist::DIST_BRC_B32>(tr, x + 2U * c);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(ti, x + 2U * c + 1U);
        Mul(p, tr, ar, active);
        Mul(q, ti, ai, active);
        Sub(contribution, p, q, active);
        ChemvRegAddContribution<FULL>(real, contribution, active);
        Mul(p, tr, ai, active);
        Mul(q, ti, ar, active);
        Add(contribution, p, q, active);
        ChemvRegAddContribution<FULL>(imag, contribution, active);
    }
}

template <bool UPPER, bool MIRROR, bool FULL>
__simd_vf__ inline void ChemvRegColumnsVf(
    __ubuf__ float* a, __ubuf__ float* x, __ubuf__ float* state, uint32_t rowBase, uint32_t rowCount, uint32_t colBase,
    uint32_t colCount)
{
    using namespace AscendC::Reg;
    RegTensor<float> real, imag, ar, ai;
    RegTensor<int32_t> rows, lanes, offsets, starts, zero, mirrorBase;
    RegTensor<int32_t> pairBase0, pairBase1;
    MaskReg packedMask0, packedMask1;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    const uint32_t originalRows = rowCount;
    MaskReg rowMask = UpdateMask<float>(rowCount);
    MaskReg active, positive;
    Arange(rows, static_cast<int32_t>(rowBase));
    Arange(lanes, 0);
    Duplicate(zero, 0, all);
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(real, imag, state);
    if constexpr (MIRROR && !UPPER && !FULL) {
        Adds(starts, rows, 1 - static_cast<int32_t>(colBase), all);
        CompareScalar<int32_t, CMPMODE::GT>(positive, starts, 0, all);
        Select(starts, starts, zero, positive);
    }
    if constexpr (MIRROR && FULL) {
        ChemvRegMirrorOffsets(lanes, pairBase0, pairBase1, packedMask0, packedMask1, all, originalRows);
    }
    for (uint16_t c = 0U; c < colCount; ++c) {
        const uint32_t col = colBase + c;
        if constexpr (FULL) {
            active = rowMask;
        } else if constexpr (UPPER != MIRROR) {
            CompareScalar<int32_t, CMPMODE::LT>(active, rows, static_cast<int32_t>(col), rowMask);
        } else {
            CompareScalar<int32_t, CMPMODE::GT>(active, rows, static_cast<int32_t>(col), rowMask);
        }
        ChemvRegLoadColumn<UPPER, MIRROR, FULL>(
            ar, ai, offsets, lanes, starts, pairBase0, pairBase1, packedMask0, packedMask1, active, all, a, c, col,
            rowBase);
        ChemvRegAccumulateColumn<MIRROR, FULL>(real, imag, ar, ai, x, c, active);
    }
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(state, real, imag, all);
    // Match cgemv_nslab's carry-state visibility without a full pipeline drain.
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void ChemvRegSpecialVf(
    __ubuf__ float* state, __ubuf__ float* mirror, __ubuf__ float* xRow, __ubuf__ float* diagonal, uint32_t rowCount,
    bool doDiagonal)
{
    using namespace AscendC::Reg;
    RegTensor<float> real, imag, xr, xi, tr, ti, diag, p, next;
    RegTensor<int32_t> offsets;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg active = UpdateMask<float>(rowCount);
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(real, imag, state);
    if (doDiagonal) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(xr, xi, xRow);
        Arange(offsets, 0);
        Muls(offsets, offsets, 8, all);
        Gather(diag, diagonal, reinterpret_cast<RegTensor<uint32_t>&>(offsets), active);
        ChemvRegAlphaOne(tr, ti, xr, xi, active);
        Mul(p, tr, diag, active);
        Add(next, real, p, active);
        Select(real, next, real, active);
        Mul(p, ti, diag, active);
        Add(next, imag, p, active);
        Select(imag, next, imag, active);
    } else {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(xr, xi, mirror);
        ChemvRegAlphaOne(tr, ti, xr, xi, active);
        Add(next, real, tr, active);
        Select(real, next, real, active);
        Add(next, imag, ti, active);
        Select(imag, next, imag, active);
    }
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(state, real, imag, all);
}

__simd_vf__ inline void ChemvRegCopyVf(__ubuf__ float* source, __ubuf__ float* target)
{
    using namespace AscendC::Reg;
    RegTensor<float> real, imag;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(real, imag, source);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(target, real, imag, all);
}

template <bool UPPER>
class chemv_RegOrdered {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const ChemvTilingData& t, TPipe* pipe)
    {
        t_ = t;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(t.n) * t.lda * 2U);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(t.n) * 2U);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint64_t>(t.n) * 2U);
        pipe->InitBuffer(aQueue_, 3, CHEMV_REG_MATRIX_SLOTS * CHEMV_REG_PITCH * sizeof(float));
        pipe->InitBuffer(xQueue_, 3, 256U * sizeof(float));
        pipe->InitBuffer(diagQueue_, 1, CHEMV_REG_ROWS * 8U * sizeof(float));
        pipe->InitBuffer(xRowQueue_, 1, 128U * sizeof(float));
        pipe->InitBuffer(outQueue_, 1, 128U * sizeof(float));
        pipe->InitBuffer(stateBuf_, 128U * sizeof(float));
        pipe->InitBuffer(mirrorBuf_, 128U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint64_t cores = GetBlockNum();
        const uint64_t core = GetBlockIdx();
        if (cores == 0U || core >= cores) {
            return;
        }
        const uint32_t rowTile = t_.rowsPerBlock;
        // Fail-silent safety net: the host gates rowsPerBlock to the supported set
        // before tiling (device code has no host-side log channel). If this guard
        // ever trips, check the shared CHEMV_REG_ROWS contract in chemv_tiling_data.h.
        if (!CHEMV_ROWS_PER_BLOCK_VALID(rowTile)) {
            return;
        }
        const uint64_t groups = (static_cast<uint64_t>(t_.n) + rowTile - 1U) / rowTile;
        const uint64_t q = groups / cores;
        const uint64_t r = groups % cores;
        uint64_t first;
        uint64_t count;
        if constexpr (UPPER) {
            // Small upper rows contain fewer mirrored (Gather) columns.
            first = core * q + (core < r ? core : r);
            count = q + (core < r ? 1U : 0U);
        } else {
            // Large lower rows contain fewer mirrored (Gather) columns.
            const uint64_t cut = cores - r;
            first = core * q + (core > cut ? core - cut : 0U);
            count = q + (core >= cut ? 1U : 0U);
        }
        for (uint64_t ordinal = 0U; ordinal < count; ++ordinal) {
            const uint64_t group = SelectGroup(ordinal, first, q, r, core, cores, groups);
            ProcessGroup(group, rowTile);
        }
    }

private:
    __aicore__ inline void ProcessGroup(uint64_t group, uint32_t rowTile)
    {
        const uint32_t base = static_cast<uint32_t>(group * rowTile);
        const uint32_t rows = t_.n - base < rowTile ? t_.n - base : rowTile;
        auto state = stateBuf_.Get<float>();
        auto mirror = mirrorBuf_.Get<float>();
        Duplicate(state, 0.0f, 128U);
        Duplicate(mirror, 0.0f, 128U);
        PipeBarrier<PIPE_V>();
        if constexpr (!UPPER) {
            Columns<false>(base, rows, state);
        }
        Diagonal(base, rows, state);
        Columns<true>(base, rows, mirror);
        asc_vf_call<ChemvRegSpecialVf>(Ptr(state), Ptr(mirror), Ptr(state), Ptr(state), rows, false);
        PipeBarrier<PIPE_V>();
        if constexpr (UPPER) {
            Columns<false>(base, rows, state);
        }
        auto out = outQueue_.AllocTensor<float>();
        asc_vf_call<ChemvRegCopyVf>(Ptr(state), Ptr(out));
        outQueue_.EnQue(out);
        out = outQueue_.DeQue<float>();
        DataCopyExtParams params{1U, rows * 2U * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(y_[static_cast<uint64_t>(base) * 2U], out, params);
        outQueue_.FreeTensor(out);
    }

    __aicore__ inline uint64_t SelectGroup(
        uint64_t ordinal, uint64_t first, uint64_t q, uint64_t r, uint64_t core, uint64_t cores, uint64_t groups)
    {
        uint64_t group = first + ordinal;
        if (q == 1U && r > 1U) {
            if constexpr (UPPER) {
                if (core < r) {
                    group = ordinal == 0U ? core : 2U * r - 1U - core;
                }
            } else {
                const uint64_t cut = cores - r;
                if (core >= cut) {
                    const uint64_t j = core - cut;
                    group = ordinal == 0U ? groups - 2U * r + j : groups - 1U - j;
                }
            }
        }
        return group;
    }

    __aicore__ inline __ubuf__ float* Ptr(LocalTensor<float>& v)
    {
        return reinterpret_cast<__ubuf__ float*>(v.GetPhyAddr());
    }

    // Each logical source run is strict stored triangle, packed at an aligned UB slot.
    __aicore__ inline void CopyRun(LocalTensor<float>& dst, uint32_t dstOffset, uint64_t srcOffset, uint32_t floats)
    {
        if (floats == 0U) {
            return;
        }
        DataCopyExtParams params{1U, floats * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const uint32_t padding = ((floats + 7U) & ~7U) - floats;
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(padding), 0.0f};
        DataCopyPad(dst[dstOffset], a_[srcOffset], params, pad);
    }

    template <bool MIRROR>
    __aicore__ inline void PrefetchColumns(uint32_t base, uint32_t rows, uint32_t cb, uint32_t cc)
    {
        auto matrix = aQueue_.AllocTensor<float>();
        auto xv = xQueue_.AllocTensor<float>();
        const bool fullStoredRectangle = (UPPER != MIRROR) ? cb >= base + rows : cb + cc <= base;
        if (fullStoredRectangle) {
            const uint32_t copyRows = MIRROR ? cc : rows;
            const uint32_t copyCols = MIRROR ? rows : cc;
            const uint32_t alignedFloats = (2U * copyRows + 7U) & ~7U;
            DataCopyExtParams mp{
                static_cast<uint16_t>(copyCols), copyRows * 2U * static_cast<uint32_t>(sizeof(float)),
                static_cast<int64_t>(t_.lda - copyRows) * 2L * static_cast<int64_t>(sizeof(float)),
                static_cast<int64_t>((CHEMV_REG_PITCH - alignedFloats) / 8U), 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(alignedFloats - 2U * copyRows), 0.0f};
            const uint64_t offset = MIRROR ? 2U * (static_cast<uint64_t>(base) * t_.lda + cb) :
                                             2U * (static_cast<uint64_t>(cb) * t_.lda + base);
            DataCopyPad(matrix, a_[offset], mp, pad);
        } else if constexpr (MIRROR) {
            for (uint32_t r = 0U; r < rows; ++r) {
                const uint32_t row = base + r;
                const uint32_t lo = UPPER ? cb : (cb > row + 1U ? cb : row + 1U);
                const uint32_t hi = UPPER ? (cb + cc < row ? cb + cc : row) : cb + cc;
                if (hi > lo) {
                    CopyRun(
                        matrix, r * CHEMV_REG_PITCH, 2U * (static_cast<uint64_t>(row) * t_.lda + lo), 2U * (hi - lo));
                }
            }
        } else {
            for (uint32_t c = 0U; c < cc; ++c) {
                const uint32_t col = cb + c;
                const uint32_t lo = UPPER ? base : (base > col + 1U ? base : col + 1U);
                const uint32_t hi = UPPER ? (base + rows < col ? base + rows : col) : base + rows;
                if (hi > lo) {
                    CopyRun(
                        matrix, c * CHEMV_REG_PITCH, 2U * (static_cast<uint64_t>(col) * t_.lda + lo), 2U * (hi - lo));
                }
            }
        }
        DataCopyExtParams xp{1U, cc * 2U * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const uint32_t padCount = ((cc * 2U + 7U) & ~7U) - cc * 2U;
        DataCopyPadExtParams<float> xpad{true, 0, static_cast<uint8_t>(padCount), 0.0f};
        DataCopyPad(xv, x_[static_cast<uint64_t>(cb) * 2U], xp, xpad);
        aQueue_.EnQue(matrix);
        xQueue_.EnQue(xv);
    }

    template <bool MIRROR>
    __aicore__ inline void ConsumeColumns(
        uint32_t base, uint32_t rows, uint32_t cb, uint32_t cc, LocalTensor<float>& state)
    {
        auto matrix = aQueue_.DeQue<float>();
        auto xv = xQueue_.DeQue<float>();
        const bool fullStoredRectangle = (UPPER != MIRROR) ? cb >= base + rows : cb + cc <= base;
        __ubuf__ float* selectedX = Ptr(xv);
        if constexpr (!MIRROR) {
            // Two128-float planes: raw complex x then exact alpha*x.
            asc_vf_call<ChemvRegPrefactorVf>(Ptr(xv), Ptr(xv) + 128U, cc);
            selectedX = Ptr(xv) + 128U;
        }
        if (fullStoredRectangle) {
            asc_vf_call<ChemvRegColumnsVf<UPPER, MIRROR, true>>(Ptr(matrix), selectedX, Ptr(state), base, rows, cb, cc);
        } else {
            asc_vf_call<ChemvRegColumnsVf<UPPER, MIRROR, false>>(
                Ptr(matrix), selectedX, Ptr(state), base, rows, cb, cc);
        }
        // Queue FreeTensor records V_MTE2 completion for this physical slot.
        // A later AllocTensor must wait before MTE2 can overwrite it.
        aQueue_.FreeTensor(matrix);
        xQueue_.FreeTensor(xv);
    }

    template <bool MIRROR>
    __aicore__ inline void Columns(uint32_t base, uint32_t rows, LocalTensor<float>& state)
    {
        const uint32_t begin = (UPPER != MIRROR) ? base + 1U : 0U;
        const uint32_t end = (UPPER != MIRROR) ? t_.n : base + rows - 1U;
        if (begin >= end) {
            return;
        }
        bool alignSuffix = false;
        if constexpr (UPPER != MIRROR) {
            const uint64_t originalTiles = (static_cast<uint64_t>(end - begin) + CHEMV_REG_COLS - 1U) / CHEMV_REG_COLS;
            const uint64_t alignedTiles =
                static_cast<uint64_t>((end - 1U) / CHEMV_REG_COLS) - begin / CHEMV_REG_COLS + 1U;
            alignSuffix = alignedTiles <= originalTiles;
        }
        const uint32_t firstWidth = alignSuffix ? CHEMV_REG_COLS - begin % CHEMV_REG_COLS : CHEMV_REG_COLS;
        const uint32_t firstCount = end - begin < firstWidth ? end - begin : firstWidth;
        PrefetchColumns<MIRROR>(base, rows, begin, firstCount);
        // Track the actual next tile, since an aligned first tile may be short.
        uint32_t prefetchCB = begin + firstCount;
        if (prefetchCB < end) {
            const uint32_t count = end - prefetchCB < CHEMV_REG_COLS ? end - prefetchCB : CHEMV_REG_COLS;
            PrefetchColumns<MIRROR>(base, rows, prefetchCB, count);
            prefetchCB += count;
        }
        uint32_t cb = begin;
        uint32_t cc = firstCount;
        while (cb < end) {
            if (prefetchCB < end) {
                const uint32_t count = end - prefetchCB < CHEMV_REG_COLS ? end - prefetchCB : CHEMV_REG_COLS;
                // At most three FIFO tiles are ready/in-flight before consumption.
                PrefetchColumns<MIRROR>(base, rows, prefetchCB, count);
                prefetchCB += count;
            }
            // The same tile stream drives producer and consumer; state order is unchanged.
            ConsumeColumns<MIRROR>(base, rows, cb, cc, state);
            cb += cc;
            cc = end - cb < CHEMV_REG_COLS ? end - cb : CHEMV_REG_COLS;
        }
        // One conservative drain per reference phase, not one per input tile.
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Diagonal(uint32_t base, uint32_t rows, LocalTensor<float>& state)
    {
        auto diag = diagQueue_.AllocTensor<float>();
        auto xr = xRowQueue_.AllocTensor<float>();
        // Each source block is exactly one real FP32 diagonal component.
        // Source stride skips the imaginary component; padding is UB-only.
        DataCopyExtParams diagonalParams{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(float)),
            (static_cast<int64_t>(t_.lda) + 1L) * 2L * static_cast<int64_t>(sizeof(float)) -
                static_cast<int64_t>(sizeof(float)),
            0, 0};
        DataCopyPadExtParams<float> diagonalPad{true, 0, 7, 0.0f};
        DataCopyPad(diag, a_[2U * (static_cast<uint64_t>(base) * t_.lda + base)], diagonalParams, diagonalPad);
        DataCopyExtParams xp{1U, rows * 2U * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const uint32_t padCount = ((rows * 2U + 7U) & ~7U) - rows * 2U;
        DataCopyPadExtParams<float> xpad{true, 0, static_cast<uint8_t>(padCount), 0.0f};
        DataCopyPad(xr, x_[static_cast<uint64_t>(base) * 2U], xp, xpad);
        diagQueue_.EnQue(diag);
        xRowQueue_.EnQue(xr);
        diag = diagQueue_.DeQue<float>();
        xr = xRowQueue_.DeQue<float>();
        asc_vf_call<ChemvRegSpecialVf>(Ptr(state), Ptr(state), Ptr(xr), Ptr(diag), rows, true);
        PipeBarrier<PIPE_V>();
        diagQueue_.FreeTensor(diag);
        xRowQueue_.FreeTensor(xr);
    }
    ChemvTilingData t_{};
    GlobalTensor<float> a_, x_, y_;
    TQue<TPosition::VECIN, 3> aQueue_, xQueue_;
    TQue<TPosition::VECIN, 1> diagQueue_, xRowQueue_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> stateBuf_, mirrorBuf_;
};
} // namespace

__global__ __aicore__ void chemv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, const ChemvTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const bool fast = t.alphaReal == 1.0f && t.alphaImag == 0.0f && t.betaReal == 0.0f && t.betaImag == 0.0f &&
                      t.incx == 1 && t.incy == 1;
    if (fast) {
        TPipe pipe;
        if (t.uplo == ACLBLAS_UPPER) {
            chemv_RegOrdered<true> op;
            op.Init(a, x, y, t, &pipe);
            op.Process();
        } else {
            chemv_RegOrdered<false> op;
            op.Init(a, x, y, t, &pipe);
            op.Process();
        }
        return;
    }
    const dim3 threadDim = {t.nthreads, 1, 1};
    if (t.uplo == ACLBLAS_UPPER) {
        asc_vf_call<ChemvCompute<true>>(
            threadDim, t.n, t.lda, t.incx, t.incy, t.alphaReal, t.alphaImag, t.betaReal, t.betaImag,
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(y));
    } else {
        asc_vf_call<ChemvCompute<false>>(
            threadDim, t.n, t.lda, t.incx, t.incy, t.alphaReal, t.alphaImag, t.betaReal, t.betaImag,
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(y));
    }
}
void chemv_kernel_do(GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t blocks, const ChemvTilingData& t, void* stream)
{
    chemv_kernel<<<blocks, nullptr, stream>>>(a, x, y, t);
}
