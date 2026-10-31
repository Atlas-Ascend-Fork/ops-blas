/*
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
#include "ctrsv_kernel.h"
#include "ctrsv_complex.h"

// Logical panel/cache sizes, not physical device core counts.
constexpr uint32_t CTRSV_CACHE = 4096;
constexpr uint32_t CTRSV_PANEL_SIZE = 64;
constexpr uint32_t CTRSV_UNIT_PANEL_SIZE = 32;
constexpr uint32_t CTRSV_SIMT_THREADS = 64;
constexpr uint32_t CTRSV_CLASSIFY_THREADS = 32;
constexpr uint32_t CTRSV_VECTOR_FLOATS = 64;
constexpr uint32_t CTRSV_UPDATE_ROWS = 128;
constexpr uint32_t CTRSV_COMPLEX_BYTES = 2 * sizeof(float);
constexpr uint32_t CTRSV_SCALE_BYTES = sizeof(float);
constexpr uint32_t CTRSV_COMPLEX_ALIGNMENT = 4;
constexpr uint32_t CTRSV_CACHE_FLOATS_PER_ROW = 3;
constexpr uint32_t CTRSV_PANEL_DATA_FLOATS = 2 * CTRSV_PANEL_SIZE * CTRSV_PANEL_SIZE;
constexpr uint32_t CTRSV_PANEL_NAN_FLAG = 0x40000000;
constexpr float CTRSV_DIAGONAL_SCALE_THRESHOLD = 0x1p120f;
constexpr float CTRSV_INVERSE_SOLUTION_SCALE = 0x1p-24f;
constexpr float CTRSV_FAST_DIVISION_MIN = 0x1p-50f;
constexpr float CTRSV_FAST_DIVISION_MAX = 0x1p50f;
constexpr float CTRSV_VECTOR_DIAGONAL_MIN = 16.0f;
constexpr float CTRSV_VECTOR_DIAGONAL_MAX = 0x1p20f;
constexpr float CTRSV_VECTOR_OFF_DIAGONAL_MAX = 5.0f;
constexpr float CTRSV_VECTOR_RHS_MAX = 1024.0f;
// 64x64 complex panel plus one 64-float classification row.
constexpr uint32_t CTRSV_PACKED_PANEL_FLOATS = 8256;
// Launch UB budgets include explicit TBuf storage and SIMT shared/spill headroom.
constexpr uint32_t CTRSV_SOLVE_UB_BYTES = 208 * 1024;
constexpr uint32_t CTRSV_PACK_UB_BYTES = 96 * 1024;
constexpr uint32_t CTRSV_TRANSFER_UB_BYTES = 64 * 1024;
// Update tile: 128 rows, 64 panel columns plus 4 complex padding elements;
// an extra 512 bytes hold classification flags and alignment padding.
constexpr uint32_t CTRSV_UPDATE_BUFFER_BYTES = 2 * 68 * 128 * sizeof(float) + 512;

__simt_callee__ inline void SolveGeneralPivot(
    __gm__ const float* a, __gm__ float* x, __ubuf__ float* shared, CtrsvTilingData t, uint32_t j, bool trans)
{
    if (threadIdx.x == 0) {
        auto pivot = LoadComplex(x, t.start + int64_t(j) * t.incx);
        float inverse = 1.0f;
        if (t.diag != ACLBLAS_UNIT && (trans || pivot.real != 0 || pivot.imag != 0)) {
            auto diagonal = LoadComplex(a, uint64_t(j) * t.lda + j);
            if (t.trans == ACLBLAS_OP_C) {
                diagonal.imag = -diagonal.imag;
            }
            bool scaled = fabsf(diagonal.real) > CTRSV_DIAGONAL_SCALE_THRESHOLD ||
                          fabsf(diagonal.imag) > CTRSV_DIAGONAL_SCALE_THRESHOLD;
            pivot = Divide(pivot, diagonal, scaled ? CTRSV_SOLUTION_SCALE : 1.0f);
            inverse = scaled ? CTRSV_INVERSE_SOLUTION_SCALE : 1.0f;
        }
        shared[0] = pivot.real;
        shared[1] = pivot.imag;
        shared[2] = inverse;
        StoreComplex(x, t.start + int64_t(j) * t.incx, {pivot.real * inverse, pivot.imag * inverse});
    }
}

__simt_callee__ inline void UpdateGeneralRows(
    __gm__ const float* a, __gm__ float* x, __ubuf__ float* shared, CtrsvTilingData t, uint32_t q, uint32_t j,
    bool trans, bool forward)
{
    ComplexValue pivot{shared[0], shared[1]};
    if (trans || pivot.real != 0 || pivot.imag != 0) {
        for (uint32_t r = q + 1 + threadIdx.x; r < t.n; r += blockDim.x) {
            uint32_t i = forward ? r : t.n - 1 - r;
            uint64_t offset = t.start + int64_t(i) * t.incx;
            auto coefficient = LoadComplex(a, trans ? uint64_t(i) * t.lda + j : uint64_t(j) * t.lda + i);
            if (t.trans == ACLBLAS_OP_C) {
                coefficient.imag = -coefficient.imag;
            }
            if (isfinite(pivot.real) && isfinite(pivot.imag)) {
                coefficient.real *= shared[2];
                coefficient.imag *= shared[2];
            }
            auto value = LoadComplex(x, offset);
            SubtractProduct(value, trans ? coefficient : pivot, trans ? pivot : coefficient);
            StoreComplex(x, offset, value);
        }
    }
}

// The general path keeps only the current pivot in UB. It has no shape cap.
__simt_vf__ __aicore__
LAUNCH_BOUND(CTRSV_SIMT_THREADS) void SolveGeneral(__gm__ const float* a, __gm__ float* x, CtrsvTilingData t)
{
    __ubuf__ float shared[3];
    const bool trans = t.trans != ACLBLAS_OP_N;
    const bool forward = (t.uplo == ACLBLAS_UPPER) == trans;
    asc_dcci_entire(x);
    asc_dcci_entire(const_cast<__gm__ float*>(a));
    asc_syncthreads();
    for (uint32_t q = 0; q < t.n; ++q) {
        uint32_t j = forward ? q : t.n - 1 - q;
        SolveGeneralPivot(a, x, shared, t, j, trans);
        asc_syncthreads();
        UpdateGeneralRows(a, x, shared, t, q, j, trans, forward);
        asc_threadfence();
        asc_syncthreads();
        asc_dcci_entire(x);
        asc_syncthreads();
    }
}

__simt_callee__ inline bool FastDivisionComponent(float value)
{
    const float magnitude = fabsf(value);
    return magnitude == 0.0f || (magnitude >= CTRSV_FAST_DIVISION_MIN && magnitude < CTRSV_FAST_DIVISION_MAX);
}

__simt_callee__ inline ComplexValue FastDivide(ComplexValue a, ComplexValue b)
{
    const float maximum = fmaxf(fabsf(b.real), fabsf(b.imag));
    if (maximum > CTRSV_FAST_DIVISION_MIN && FastDivisionComponent(b.real) && FastDivisionComponent(b.imag) &&
        FastDivisionComponent(a.real) && FastDivisionComponent(a.imag)) {
        const float inverse = 1.0f / (b.real * b.real + b.imag * b.imag);
        return {(a.real * b.real + a.imag * b.imag) * inverse, (a.imag * b.real - a.real * b.imag) * inverse};
    }
    return Divide(a, b, 1.0f);
}

__simt_vf__ __aicore__
LAUNCH_BOUND(CTRSV_SIMT_THREADS) void CopyX(__gm__ float* x, __ubuf__ float* cache, CtrsvTilingData t, bool output)
{
    if (!output) {
        asc_dcci_entire(x);
        asc_syncthreads();
    }
    for (uint32_t i = threadIdx.x; i < t.n; i += blockDim.x) {
        if (output) {
            StoreComplex(
                x, t.start + int64_t(i) * t.incx,
                {cache[2 * i] * cache[2 * CTRSV_CACHE + i], cache[2 * i + 1] * cache[2 * CTRSV_CACHE + i]});
        } else {
            auto value = LoadComplex(x, t.start + int64_t(i) * t.incx);
            cache[2 * i] = value.real;
            cache[2 * i + 1] = value.imag;
            cache[2 * CTRSV_CACHE + i] = 1.0f;
        }
    }
    if (output) {
        asc_threadfence();
        asc_syncthreads();
        asc_dcci_entire(x);
    }
}

__simt_callee__ inline void SolveCachedPivot(
    __ubuf__ float* a, __ubuf__ float* x, CtrsvTilingData t, uint32_t j, uint32_t low, bool trans)
{
    if (threadIdx.x == 0 && t.diag != ACLBLAS_UNIT) {
        ComplexValue value{x[2 * j], x[2 * j + 1]};
        if (trans || value.real != 0 || value.imag != 0) {
            uint32_t diagonal = 2 * ((j - low) * CTRSV_PANEL_SIZE + ((trans || t.uplo == ACLBLAS_UPPER) ? j - low : 0));
            ComplexValue d{a[diagonal], a[diagonal + 1]};
            if (t.trans == ACLBLAS_OP_C) {
                d.imag = -d.imag;
            }
            bool scaled =
                fabsf(d.real) > CTRSV_DIAGONAL_SCALE_THRESHOLD || fabsf(d.imag) > CTRSV_DIAGONAL_SCALE_THRESHOLD;
            value = scaled ? Divide(value, d, CTRSV_SOLUTION_SCALE) : FastDivide(value, d);
            x[2 * CTRSV_CACHE + j] = scaled ? CTRSV_INVERSE_SOLUTION_SCALE : 1.0f;
            x[2 * j] = value.real;
            x[2 * j + 1] = value.imag;
        }
    }
}

__simt_callee__ inline void UpdateCachedRows(
    __ubuf__ float* a, __ubuf__ float* x, CtrsvTilingData t, uint32_t first, uint32_t count, uint32_t q, uint32_t j,
    uint32_t low, bool trans, bool forward)
{
    ComplexValue pivot{x[2 * j], x[2 * j + 1]};
    if (trans || pivot.real != 0 || pivot.imag != 0) {
        for (uint32_t r = q + 1 + threadIdx.x; r < count; r += blockDim.x) {
            uint32_t i = forward ? first + r : t.n - 1 - first - r;
            uint32_t ar = (trans ? j : i) - low, ac = (trans ? i : j) - low;
            uint32_t offset =
                trans ?
                    2 * (ar * CTRSV_PANEL_SIZE + ac) :
                    2 * (ac * CTRSV_PANEL_SIZE + ar - (t.uplo == ACLBLAS_UPPER ? 0 : ac + (t.diag == ACLBLAS_UNIT)));
            ComplexValue c{a[offset], a[offset + 1]};
            if (t.trans == ACLBLAS_OP_C) {
                c.imag = -c.imag;
            }
            if (isfinite(pivot.real) && isfinite(pivot.imag)) {
                c.real *= x[2 * CTRSV_CACHE + j];
                c.imag *= x[2 * CTRSV_CACHE + j];
            }
            ComplexValue value{x[2 * i], x[2 * i + 1]};
            SubtractProduct(value, trans ? c : pivot, trans ? pivot : c);
            x[2 * i] = value.real;
            x[2 * i + 1] = value.imag;
        }
    }
}

// Solve dependencies inside one diagonal panel before updating disjoint rows.
__simt_vf__ __aicore__ LAUNCH_BOUND(CTRSV_SIMT_THREADS) void SolvePanel(
    __ubuf__ float* a, __ubuf__ float* x, CtrsvTilingData t, uint32_t first, uint32_t count)
{
    const bool trans = t.trans != ACLBLAS_OP_N;
    const bool forward = (t.uplo == ACLBLAS_UPPER) == trans;
    const uint32_t low = forward ? first : t.n - first - count;
    for (uint32_t q = 0; q < count; ++q) {
        uint32_t j = forward ? first + q : t.n - 1 - first - q;
        SolveCachedPivot(a, x, t, j, low, trans);
        asc_syncthreads();
        UpdateCachedRows(a, x, t, first, count, q, j, low, trans, forward);
        asc_syncthreads();
    }
    if (threadIdx.x == 0) {
        uint32_t nanFlag = 0;
        for (uint32_t i = low; i < low + count; ++i) {
            if (isnan(x[2 * i]) && isnan(x[2 * i + 1])) {
                nanFlag = CTRSV_PANEL_NAN_FLAG;
            }
        }
        reinterpret_cast<__ubuf__ uint32_t*>(a + 2 * CTRSV_PANEL_SIZE * CTRSV_PANEL_SIZE)[0] = nanFlag;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CTRSV_SIMT_THREADS) void UpdatePanel(
    __ubuf__ float* a, __ubuf__ float* x, CtrsvTilingData t, uint32_t first, uint32_t count, uint32_t row,
    uint32_t rows, uint32_t pitch, __ubuf__ float* backup, bool repair)
{
    const bool trans = t.trans != ACLBLAS_OP_N;
    const bool forward = (t.uplo == ACLBLAS_UPPER) == trans;
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        uint32_t i = row + r;
        ComplexValue value{x[2 * i], x[2 * i + 1]};
        if (repair) {
            if (isfinite(value.real) && isfinite(value.imag)) {
                continue;
            }
            value = {backup[2 * r], backup[2 * r + 1]};
        }
        if (isnan(value.real) && isnan(value.imag)) {
            continue;
        }
        for (uint32_t q = 0; q < count; ++q) {
            uint32_t j = forward ? first + q : t.n - 1 - first - q;
            uint32_t column = forward ? q : count - 1 - q;
            uint32_t off = 2 * (trans ? r * pitch + column : column * pitch + r);
            ComplexValue c{a[off], a[off + 1]}, pivot{x[2 * j], x[2 * j + 1]};
            if (t.trans == ACLBLAS_OP_C) {
                c.imag = -c.imag;
            }
            if (isfinite(pivot.real) && isfinite(pivot.imag)) {
                c.real *= x[2 * CTRSV_CACHE + j];
                c.imag *= x[2 * CTRSV_CACHE + j];
            }
            if (trans || pivot.real != 0 || pivot.imag != 0) {
                SubtractProduct(value, trans ? c : pivot, trans ? pivot : c);
            }
        }
        x[2 * i] = value.real;
        x[2 * i + 1] = value.imag;
    }
}

__simd_callee__ inline void LoadUpdateCoefficient(
    __ubuf__ float* a, AscendC::Reg::RegTensor<float>& ar, AscendC::Reg::RegTensor<float>& ai,
    AscendC::Reg::MaskReg& mask, uint32_t r, uint32_t c, uint32_t pitch, bool trans, bool conjugate)
{
    using namespace AscendC;
    if (trans) {
        Reg::RegTensor<uint32_t> offsets;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(offsets), int32_t(r));
        Reg::Muls(offsets, offsets, uint32_t(2 * pitch), mask);
        Reg::Adds(offsets, offsets, uint32_t(2 * c), mask);
        Reg::Gather(ar, a, offsets, mask);
        Reg::Adds(offsets, offsets, uint32_t(1), mask);
        Reg::Gather(ai, a, offsets, mask);
        if (conjugate) {
            Reg::Muls(ai, ai, -1.0f, mask);
        }
    } else {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, a + 2 * (c * pitch + r));
    }
}

__simd_vf__ inline void UpdatePanelVector(
    __ubuf__ float* a, __ubuf__ float* x, __ubuf__ float* backup, uint32_t n, uint32_t first, uint32_t count,
    uint32_t row, uint32_t rows, uint32_t pitch, bool forward, bool trans, bool conjugate)
{
    using namespace AscendC;
    Reg::RegTensor<float> vr, vi, ar, ai, pr, pi, u, v, w, scale, inter0, inter1;
    for (uint16_t r = 0; r < rows; r += CTRSV_VECTOR_FLOATS) {
        uint32_t valid = rows - r;
        auto mask = Reg::UpdateMask<float>(valid);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vr, vi, x + 2 * (row + r));
        uint32_t floats = 2 * ((rows - r) < CTRSV_VECTOR_FLOATS ? rows - r : CTRSV_VECTOR_FLOATS);
        auto store0 = Reg::UpdateMask<float>(floats);
        auto store1 = Reg::UpdateMask<float>(floats);
        Reg::Interleave(inter0, inter1, vr, vi);
        Reg::StoreAlign(backup + 2 * r, inter0, store0);
        Reg::StoreAlign(backup + 2 * r + CTRSV_VECTOR_FLOATS, inter1, store1);
        for (uint16_t q = 0; q < count; ++q) {
            uint32_t j = forward ? first + q : n - 1 - first - q;
            uint32_t c = forward ? q : count - 1 - q;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(pr, x + 2 * j);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(pi, x + 2 * j + 1);
            LoadUpdateCoefficient(a, ar, ai, mask, r, c, pitch, trans, conjugate);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(scale, x + 2 * CTRSV_CACHE + j);
            Reg::Mul(ar, ar, scale, mask);
            Reg::Mul(ai, ai, scale, mask);
            Reg::Mul(u, pr, ar, mask);
            Reg::Mul(v, pi, ai, mask);
            Reg::Sub(w, u, v, mask);
            Reg::Sub(vr, vr, w, mask);
            Reg::Mul(u, pr, ai, mask);
            Reg::Mul(v, pi, ar, mask);
            Reg::Add(w, u, v, mask);
            Reg::Sub(vi, vi, w, mask);
        }
        Reg::Interleave(inter0, inter1, vr, vi);
        Reg::StoreAlign(x + 2 * (row + r), inter0, store0);
        Reg::StoreAlign(x + 2 * (row + r) + CTRSV_VECTOR_FLOATS, inter1, store1);
    }
}

__simt_callee__ inline uint32_t ClassifyCoefficients(
    __gm__ const float* gm, __ubuf__ float* a, CtrsvTilingData t, uint32_t low, uint32_t count)
{
    uint32_t flag = 0;
    for (uint32_t index = threadIdx.x; index < count * CTRSV_PANEL_SIZE; index += blockDim.x) {
        uint32_t c = index / CTRSV_PANEL_SIZE, r = index % CTRSV_PANEL_SIZE;
        uint32_t begin = t.uplo == ACLBLAS_UPPER ? 0 : c + (t.diag == ACLBLAS_UNIT);
        uint32_t end = t.uplo == ACLBLAS_UPPER ? c + (t.diag != ACLBLAS_UNIT) : count;
        if (r < begin || r >= end) {
            continue;
        }
        uint32_t off =
            t.trans != ACLBLAS_OP_N ? 2 * (r * CTRSV_PANEL_SIZE + c) : 2 * (c * CTRSV_PANEL_SIZE + r - begin);
        auto value = LoadComplex(gm, uint64_t(low + c) * t.lda + low + r);
        float ar = value.real, ai = value.imag;
        a[off] = ar;
        a[off + 1] = ai;
        if (!isfinite(ar) || !isfinite(ai)) {
            flag |= 1;
        }
        if (r == c) {
            if (fmaxf(fabsf(ar), fabsf(ai)) < CTRSV_VECTOR_DIAGONAL_MIN || fabsf(ar) > CTRSV_VECTOR_DIAGONAL_MAX ||
                fabsf(ai) > CTRSV_VECTOR_DIAGONAL_MAX) {
                flag |= 1;
            }
        } else if (fabsf(ar) > CTRSV_VECTOR_OFF_DIAGONAL_MAX || fabsf(ai) > CTRSV_VECTOR_OFF_DIAGONAL_MAX) {
            flag |= 1;
        }
    }
    return flag;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CTRSV_SIMT_THREADS) void ClassifyPanel(
    __gm__ const float* gm, __ubuf__ float* a, __ubuf__ float* x, __ubuf__ uint32_t* flags, CtrsvTilingData t,
    uint32_t low, uint32_t count)
{
    asc_dcci_entire(const_cast<__gm__ float*>(gm));
    asc_syncthreads();
    uint32_t flag = t.diag == ACLBLAS_UNIT && count > CTRSV_UNIT_PANEL_SIZE ? 1 : 0;
    if (threadIdx.x < count) {
        uint32_t c = threadIdx.x;
        float xr = x[2 * (low + c)], xi = x[2 * (low + c) + 1];
        if (!(isnan(xr) && isnan(xi))) {
            flag |= 2;
        }
        if (!isfinite(xr) || !isfinite(xi) || fabsf(xr) > CTRSV_VECTOR_RHS_MAX || fabsf(xi) > CTRSV_VECTOR_RHS_MAX) {
            flag |= 1;
        }
    }
    flag |= ClassifyCoefficients(gm, a, t, low, count);
    flags[threadIdx.x] = flag;
    asc_syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t i = 1; i < CTRSV_PANEL_SIZE; ++i) {
            flag |= flags[i];
        }
        flags[0] = flag;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CTRSV_SIMT_THREADS) void FillRemainingNan(
    __ubuf__ float* x, CtrsvTilingData t, uint32_t first, bool forward)
{
    for (uint32_t q = first + threadIdx.x; q < t.n; q += blockDim.x) {
        uint32_t i = forward ? q : t.n - 1 - q;
        x[2 * i] = __builtin_nanf("");
        x[2 * i + 1] = __builtin_nanf("");
    }
}

// Helpers retain the original FP32 operation order; implicit FMA is disabled.
__simd_callee__ inline void DivideVectorPivot(
    AscendC::Reg::RegTensor<float>& pr, AscendC::Reg::RegTensor<float>& pi, __ubuf__ float* a, uint32_t offset,
    bool conjugate, AscendC::Reg::MaskReg& all)
{
    using namespace AscendC;
    Reg::RegTensor<float> ar, ai, u, v, w, den;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ar, a + offset);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ai, a + offset + 1);
    if (conjugate) {
        Reg::Muls(ai, ai, -1.0f, all);
    }
    Reg::Mul(u, ar, ar, all);
    Reg::Mul(v, ai, ai, all);
    Reg::Add(den, u, v, all);
    Reg::Mul(u, pr, ar, all);
    Reg::Mul(v, pi, ai, all);
    Reg::Add(w, u, v, all);
    Reg::Mul(u, pi, ar, all);
    Reg::Mul(v, pr, ai, all);
    Reg::Sub(v, u, v, all);
    Reg::Div(pr, w, den, all);
    Reg::Div(pi, v, den, all);
}

__simd_callee__ inline void UpdateVectorPivot(
    __ubuf__ float* a, AscendC::Reg::RegTensor<float>& xr, AscendC::Reg::RegTensor<float>& xi,
    AscendC::Reg::RegTensor<float>& pr, AscendC::Reg::RegTensor<float>& pi, AscendC::Reg::RegTensor<uint32_t>& index,
    AscendC::Reg::MaskReg& active, AscendC::Reg::MaskReg& all, uint32_t j, bool upper, bool trans, bool conjugate,
    bool unit)
{
    using namespace AscendC;
    Reg::RegTensor<float> ar, ai, u, v, w;
    uint32_t base = upper ? 0 : j + uint32_t(unit);
    Reg::RegTensor<uint32_t> offsets;
    Reg::Muls(offsets, index, uint32_t(2), all);
    Reg::Adds(offsets, offsets, uint32_t(2 * (j * CTRSV_PANEL_SIZE - (trans ? 0 : base))), all);
    Reg::Gather(ar, a, offsets, active);
    Reg::Adds(offsets, offsets, uint32_t(1), all);
    Reg::Gather(ai, a, offsets, active);
    if (conjugate) {
        Reg::Muls(ai, ai, -1.0f, active);
    }
    Reg::Mul(u, pr, ar, active);
    Reg::Mul(v, pi, ai, active);
    Reg::Sub(w, u, v, active);
    Reg::Sub(u, xr, w, active);
    Reg::Select(xr, u, xr, active);
    Reg::Mul(u, pr, ai, active);
    Reg::Mul(v, pi, ar, active);
    Reg::Add(w, u, v, active);
    Reg::Sub(u, xi, w, active);
    Reg::Select(xi, u, xi, active);
}

__simd_callee__ inline void StorePanelNanFlag(
    __ubuf__ float* a, AscendC::Reg::RegTensor<float>& xr, AscendC::Reg::RegTensor<float>& xi,
    AscendC::Reg::MaskReg& all)
{
    using namespace AscendC;
    Reg::RegTensor<float> u, v, w;
    Reg::MaskReg nanR, nanI, both;
    Reg::Compare<float, CMPMODE::NE>(nanR, xr, xr, all);
    Reg::Compare<float, CMPMODE::NE>(nanI, xi, xi, all);
    Reg::And(both, nanR, nanI, all);
    Reg::Duplicate(u, 4.0f);
    Reg::Duplicate(v, 0.0f);
    Reg::Select(u, u, v, both);
    Reg::Reduce<Reg::ReduceType::MAX>(w, u, all);
    uint32_t one = 1;
    auto flagMask = Reg::UpdateMask<float>(one);
    Reg::StoreAlign(a + CTRSV_PANEL_DATA_FLOATS, w, flagMask);
}

__simd_callee__ inline void StorePanelVector(
    __ubuf__ float* x, AscendC::Reg::RegTensor<float>& xr, AscendC::Reg::RegTensor<float>& xi, uint32_t low,
    uint32_t count)
{
    using namespace AscendC;
    Reg::RegTensor<float> first, second;
    Reg::Interleave(first, second, xr, xi);
    uint32_t floats = 2 * count;
    auto firstMask = Reg::UpdateMask<float>(floats);
    auto secondMask = Reg::UpdateMask<float>(floats);
    Reg::StoreAlign(x + 2 * low, first, firstMask);
    Reg::StoreAlign(x + 2 * low + CTRSV_VECTOR_FLOATS, second, secondMask);
}

template <bool UNIT>
__simd_vf__ inline void SolvePanelVectorN(
    __ubuf__ float* a, __ubuf__ float* x, uint32_t low, uint32_t count, bool upper, bool trans, bool conjugate)
{
    using namespace AscendC;
    constexpr bool unit = UNIT;
    Reg::RegTensor<float> xr, xi, pr, pi;
    uint32_t len = count;
    auto all = Reg::UpdateMask<float>(len);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(xr, xi, x + 2 * low);
    Reg::RegTensor<uint32_t> index, pivotIndex;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
    for (uint16_t q = 0; q < count; ++q) {
        const bool forward = upper == trans;
        uint32_t j = forward ? q : count - 1 - q;
        Reg::Duplicate(pivotIndex, j);
        if constexpr (UNIT) {
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(pr, x + 2 * (low + j));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(pi, x + 2 * (low + j) + 1);
        } else {
            Reg::Gather(pr, xr, pivotIndex);
            Reg::Gather(pi, xi, pivotIndex);
        }
        Reg::MaskReg pivotMask, active;
        Reg::Compares<uint32_t, CMPMODE::EQ>(pivotMask, index, j, all);
        if (!unit) {
            const uint32_t offset = 2 * (j * CTRSV_PANEL_SIZE + ((trans || upper) ? j : 0));
            DivideVectorPivot(pr, pi, a, offset, conjugate, all);
            Reg::Select(xr, pr, xr, pivotMask);
            Reg::Select(xi, pi, xi, pivotMask);
        }
        if (forward) {
            Reg::Compares<uint32_t, CMPMODE::GT>(active, index, j, all);
        } else {
            Reg::Compares<uint32_t, CMPMODE::LT>(active, index, j, all);
        }
        UpdateVectorPivot(a, xr, xi, pr, pi, index, active, all, j, upper, trans, conjugate, unit);
        if constexpr (UNIT) {
            StorePanelVector(x, xr, xi, low, count);
        }
    }
    StorePanelVector(x, xr, xi, low, count);
    StorePanelNanFlag(a, xr, xi, all);
}

__aicore__ inline void DispatchPanel(
    __ubuf__ float* ap, __ubuf__ float* xp, CtrsvTilingData t, uint32_t first, uint32_t count, uint32_t low,
    uint32_t flags)
{
    const bool trans = t.trans != ACLBLAS_OP_N;
    if ((flags & 1) == 0 && low % CTRSV_COMPLEX_ALIGNMENT == 0) {
        if (t.diag == ACLBLAS_UNIT) {
            asc_vf_call<SolvePanelVectorN<true>>(
                ap, xp, low, count, t.uplo == ACLBLAS_UPPER, trans, t.trans == ACLBLAS_OP_C);
        } else {
            asc_vf_call<SolvePanelVectorN<false>>(
                ap, xp, low, count, t.uplo == ACLBLAS_UPPER, trans, t.trans == ACLBLAS_OP_C);
        }
    } else {
        asc_vf_call<SolvePanel>(dim3{CTRSV_SIMT_THREADS, 1, 1}, ap, xp, t, first, count);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void UpdateBlockedRows(
    AscendC::GlobalTensor<float>& ag, AscendC::LocalTensor<float>& al, __ubuf__ float* xp, __ubuf__ float* bp,
    CtrsvTilingData t, uint32_t first, uint32_t count, uint32_t low)
{
    constexpr uint32_t tile = CTRSV_UPDATE_ROWS;
    const bool trans = t.trans != ACLBLAS_OP_N;
    const bool forward = (t.uplo == ACLBLAS_UPPER) == trans;
    auto* ap = reinterpret_cast<__ubuf__ float*>(al.GetPhyAddr());
    uint32_t begin = forward ? first + count : 0, end = forward ? t.n : t.n - first - count;
    for (uint32_t row = begin; row < end; row += tile) {
        uint32_t rows = end - row < tile ? end - row : tile;
        uint32_t length = trans ? count : rows, blocks = trans ? rows : count;
        uint32_t pitch = (length + CTRSV_COMPLEX_ALIGNMENT - 1) / CTRSV_COMPLEX_ALIGNMENT * CTRSV_COMPLEX_ALIGNMENT +
                         (trans ? CTRSV_COMPLEX_ALIGNMENT : 0);
        AscendC::DataCopyExtParams cp(
            blocks, CTRSV_COMPLEX_BYTES * length, uint64_t(CTRSV_COMPLEX_BYTES) * (t.lda - length), trans ? 1U : 0U, 0);
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        uint64_t offset = trans ? uint64_t(row) * t.lda + low : uint64_t(low) * t.lda + row;
        AscendC::DataCopyPad(al, ag[2 * offset], cp, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        const bool vectorUpdate = row % CTRSV_COMPLEX_ALIGNMENT == 0;
        if (vectorUpdate) {
            asc_vf_call<UpdatePanelVector>(
                ap, xp, bp, t.n, first, count, row, rows, pitch, forward, trans, t.trans == ACLBLAS_OP_C);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        asc_vf_call<UpdatePanel>(
            dim3{CTRSV_SIMT_THREADS, 1, 1}, ap, xp, t, first, count, row, rows, pitch, bp, vectorUpdate);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void BlockedSolve(GM_ADDR a, GM_ADDR x, CtrsvTilingData t)
{
    const uint32_t panel = t.diag == ACLBLAS_UNIT ? CTRSV_UNIT_PANEL_SIZE : CTRSV_PANEL_SIZE;
    constexpr uint32_t tile = CTRSV_UPDATE_ROWS;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ab, xb, backup;
    pipe.InitBuffer(ab, CTRSV_UPDATE_BUFFER_BYTES);
    pipe.InitBuffer(xb, CTRSV_CACHE_FLOATS_PER_ROW * CTRSV_CACHE * sizeof(float));
    pipe.InitBuffer(backup, 2 * tile * sizeof(float));
    auto al = ab.Get<float>(), xl = xb.Get<float>();
    auto* ap = reinterpret_cast<__ubuf__ float*>(al.GetPhyAddr());
    auto* xp = reinterpret_cast<__ubuf__ float*>(xl.GetPhyAddr());
    auto* bp = reinterpret_cast<__ubuf__ float*>(backup.Get<float>().GetPhyAddr());
    auto* agp = reinterpret_cast<__gm__ const float*>(a);
    AscendC::GlobalTensor<float> ag;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    asc_vf_call<CopyX>(dim3{CTRSV_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(x), xp, t, false);
    AscendC::PipeBarrier<PIPE_ALL>();
    const bool trans = t.trans != ACLBLAS_OP_N;
    const bool forward = (t.uplo == ACLBLAS_UPPER) == trans;
    for (uint32_t first = 0, count = 0; first < t.n; first += count) {
        count = forward ? (t.n - first < panel ? t.n - first : panel) : (t.n - first - 1) % panel + 1;
        uint32_t low = forward ? first : t.n - first - count;
        auto flagTensor = al[2 * CTRSV_PANEL_SIZE * CTRSV_PANEL_SIZE].ReinterpretCast<uint32_t>();
        auto* fp = reinterpret_cast<__ubuf__ uint32_t*>(flagTensor.GetPhyAddr());
        asc_vf_call<ClassifyPanel>(dim3{CTRSV_SIMT_THREADS, 1, 1}, agp, ap, xp, fp, t, low, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        uint32_t flags = flagTensor.GetValue(0);
        if ((flags & 2) == 0) {
            asc_vf_call<FillRemainingNan>(dim3{CTRSV_SIMT_THREADS, 1, 1}, xp, t, first, forward);
            AscendC::PipeBarrier<PIPE_ALL>();
            break;
        }
        DispatchPanel(ap, xp, t, first, count, low, flags);
        if ((flagTensor.GetValue(0) & CTRSV_PANEL_NAN_FLAG) != 0) {
            asc_vf_call<FillRemainingNan>(dim3{CTRSV_SIMT_THREADS, 1, 1}, xp, t, first + count, forward);
            AscendC::PipeBarrier<PIPE_ALL>();
            break;
        }
        UpdateBlockedRows(ag, al, xp, bp, t, first, count, low);
    }
    asc_vf_call<CopyX>(dim3{CTRSV_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(x), xp, t, true);
}

__global__ __aicore__ void ctrsv_kernel(GM_ADDR a, GM_ADDR x, CtrsvTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BlockedSolve(a, x, t);
}

__global__ __aicore__ void ctrsv_fallback(GM_ADDR a, GM_ADDR x, CtrsvTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<SolveGeneral>(
        dim3{CTRSV_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ float*>(x),
        t);
}

__global__ __aicore__ void ctrsv_transfer(GM_ADDR x, GM_ADDR work, CtrsvTilingData t, bool output)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, CTRSV_CACHE_FLOATS_PER_ROW * CTRSV_CACHE * sizeof(float));
    auto local = buffer.Get<float>();
    auto* ptr = reinterpret_cast<__ubuf__ float*>(local.GetPhyAddr());
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work));
    DataCopyExtParams values(1, CTRSV_COMPLEX_BYTES * t.n, 0, 0, 0), scales(1, CTRSV_SCALE_BYTES * t.n, 0, 0, 0);
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    if (output) {
        DataCopyPad(local, gm, values, pad);
        DataCopyPad(local[2 * CTRSV_CACHE], gm[2 * CTRSV_CACHE], scales, pad);
        PipeBarrier<PIPE_ALL>();
        asc_vf_call<CopyX>(dim3{CTRSV_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(x), ptr, t, true);
    } else {
        asc_vf_call<CopyX>(dim3{CTRSV_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(x), ptr, t, false);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(gm, local, values);
        DataCopyPad(gm[2 * CTRSV_CACHE], local[2 * CTRSV_CACHE], scales);
    }
    PipeBarrier<PIPE_ALL>();
}

__global__ __aicore__ void ctrsv_prepack(GM_ADDR a, GM_ADDR work, CtrsvTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    TPipe pipe;
    TBuf<TPosition::VECCALC> ab, xb;
    pipe.InitBuffer(ab, CTRSV_PACKED_PANEL_FLOATS * sizeof(float));
    pipe.InitBuffer(xb, CTRSV_CACHE_FLOATS_PER_ROW * CTRSV_CACHE * sizeof(float));
    auto al = ab.Get<float>(), xl = xb.Get<float>();
    Duplicate(al, 0.0f, CTRSV_PACKED_PANEL_FLOATS);
    Duplicate(xl, 0.0f, CTRSV_CACHE_FLOATS_PER_ROW * CTRSV_CACHE);
    PipeBarrier<PIPE_ALL>();
    auto flags = al[CTRSV_PANEL_DATA_FLOATS].ReinterpretCast<uint32_t>();
    asc_vf_call<ClassifyPanel>(
        dim3{CTRSV_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ const float*>(a),
        reinterpret_cast<__ubuf__ float*>(al.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(xl.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(flags.GetPhyAddr()), t, GetBlockIdx() * CTRSV_PANEL_SIZE,
        t.n - GetBlockIdx() * CTRSV_PANEL_SIZE < CTRSV_PANEL_SIZE ? t.n - GetBlockIdx() * CTRSV_PANEL_SIZE :
                                                                    CTRSV_PANEL_SIZE);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> wg;
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work));
    DataCopy(
        wg[CTRSV_CACHE_FLOATS_PER_ROW * CTRSV_CACHE + CTRSV_PANEL_SIZE + GetBlockIdx() * CTRSV_PACKED_PANEL_FLOATS], al,
        CTRSV_PACKED_PANEL_FLOATS);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CTRSV_SIMT_THREADS) void ClassifyX(
    __ubuf__ float* x, __ubuf__ uint32_t* flags, uint32_t low, uint32_t count)
{
    uint32_t flag = threadIdx.x == 0 ? flags[0] & 1 : 0;
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        float xr = x[2 * (low + i)], xi = x[2 * (low + i) + 1];
        if (!isfinite(xr) || !isfinite(xi) || fabsf(xr) > CTRSV_VECTOR_RHS_MAX || fabsf(xi) > CTRSV_VECTOR_RHS_MAX) {
            flag |= 1;
        }
    }
    uint32_t any = AscendC::Simt::WarpAnySync(flag != 0);
    if (threadIdx.x == 0) {
        flags[0] = any;
    }
}

__aicore__ inline void SolveStagePanel(
    AscendC::GlobalTensor<float>& wg, AscendC::LocalTensor<float>& al, AscendC::LocalTensor<float>& xl,
    CtrsvTilingData t, uint32_t first, uint32_t count, uint32_t low)
{
    using namespace AscendC;
    auto* ap = reinterpret_cast<__ubuf__ float*>(al.GetPhyAddr());
    auto* xp = reinterpret_cast<__ubuf__ float*>(xl.GetPhyAddr());
    DataCopyExtParams panelCopy(1, CTRSV_COMPLEX_BYTES * count, 0, 0, 0),
        scaleCopy(1, CTRSV_SCALE_BYTES * count, 0, 0, 0);
    auto flags = al[2 * CTRSV_PANEL_SIZE * CTRSV_PANEL_SIZE].ReinterpretCast<uint32_t>();
    DataCopy(
        al,
        wg[CTRSV_CACHE_FLOATS_PER_ROW * CTRSV_CACHE + CTRSV_PANEL_SIZE +
           (low / CTRSV_PANEL_SIZE) * CTRSV_PACKED_PANEL_FLOATS],
        CTRSV_PACKED_PANEL_FLOATS);
    PipeBarrier<PIPE_ALL>();
    asc_vf_call<ClassifyX>(
        dim3{CTRSV_CLASSIFY_THREADS, 1, 1}, xp, reinterpret_cast<__ubuf__ uint32_t*>(flags.GetPhyAddr()), low, count);
    PipeBarrier<PIPE_ALL>();
    DispatchPanel(ap, xp, t, first, count, low, flags.GetValue(0));
    DataCopyPad(wg[2 * low], xl[2 * low], panelCopy);
    DataCopyPad(wg[2 * CTRSV_CACHE + low], xl[2 * CTRSV_CACHE + low], scaleCopy);
}

__aicore__ inline void UpdateStageRows(
    AscendC::GlobalTensor<float>& wg, AscendC::GlobalTensor<float>& ag, AscendC::LocalTensor<float>& al,
    AscendC::LocalTensor<float>& xl, __ubuf__ float* bp, CtrsvTilingData t, uint32_t first, uint32_t count,
    uint32_t low, uint32_t row, uint32_t end)
{
    using namespace AscendC;
    auto* ap = reinterpret_cast<__ubuf__ float*>(al.GetPhyAddr());
    auto* xp = reinterpret_cast<__ubuf__ float*>(xl.GetPhyAddr());
    const bool trans = t.trans != ACLBLAS_OP_N;
    const bool forward = (t.uplo == ACLBLAS_UPPER) == trans;
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    uint32_t rows = end - row, length = trans ? count : rows, blocks = trans ? rows : count;
    uint32_t pitch = (length + CTRSV_COMPLEX_ALIGNMENT - 1) / CTRSV_COMPLEX_ALIGNMENT * CTRSV_COMPLEX_ALIGNMENT +
                     (trans ? CTRSV_COMPLEX_ALIGNMENT : 0);
    DataCopyExtParams own(1, CTRSV_COMPLEX_BYTES * rows, 0, 0, 0),
        cp(blocks, CTRSV_COMPLEX_BYTES * length, uint64_t(CTRSV_COMPLEX_BYTES) * (t.lda - length), trans ? 1U : 0U, 0);
    DataCopyPad(xl[2 * row], wg[2 * row], own, pad);
    uint64_t offset = trans ? uint64_t(row) * t.lda + low : uint64_t(low) * t.lda + row;
    DataCopyPad(al, ag[2 * offset], cp, pad);
    PipeBarrier<PIPE_ALL>();
    bool vector = row % CTRSV_COMPLEX_ALIGNMENT == 0;
    if (vector) {
        asc_vf_call<UpdatePanelVector>(
            ap, xp, bp, t.n, first, count, row, rows, pitch, forward, trans, t.trans == ACLBLAS_OP_C);
        PipeBarrier<PIPE_ALL>();
    }
    asc_vf_call<UpdatePanel>(dim3{CTRSV_SIMT_THREADS, 1, 1}, ap, xp, t, first, count, row, rows, pitch, bp, vector);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(wg[2 * row], xl[2 * row], own);
}

__global__ __aicore__ void ctrsv_stage(GM_ADDR a, GM_ADDR work, CtrsvTilingData t, uint32_t first, bool update)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    const bool trans = t.trans != ACLBLAS_OP_N, forward = (t.uplo == ACLBLAS_UPPER) == trans;
    uint32_t count = forward ? (t.n - first < CTRSV_PANEL_SIZE ? t.n - first : CTRSV_PANEL_SIZE) :
                               (t.n - first - 1) % CTRSV_PANEL_SIZE + 1;
    uint32_t low = forward ? first : t.n - first - count;
    uint32_t row = GetBlockIdx() * CTRSV_UPDATE_ROWS,
             end = row + CTRSV_UPDATE_ROWS < t.n ? row + CTRSV_UPDATE_ROWS : t.n;
    if (update) {
        if (forward) {
            row = row > first + count ? row : first + count;
        } else {
            end = end < t.n - first - count ? end : t.n - first - count;
        }
        if (row >= end) {
            return;
        }
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> ab, xb, bb;
    pipe.InitBuffer(ab, CTRSV_UPDATE_BUFFER_BYTES);
    pipe.InitBuffer(xb, CTRSV_CACHE_FLOATS_PER_ROW * CTRSV_CACHE * sizeof(float));
    pipe.InitBuffer(bb, 2 * CTRSV_UPDATE_ROWS * sizeof(float));
    auto al = ab.Get<float>(), xl = xb.Get<float>();
    auto* bp = reinterpret_cast<__ubuf__ float*>(bb.Get<float>().GetPhyAddr());
    GlobalTensor<float> wg, ag;
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work));
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyExtParams panelCopy(1, CTRSV_COMPLEX_BYTES * count, 0, 0, 0),
        scaleCopy(1, CTRSV_SCALE_BYTES * count, 0, 0, 0);
    DataCopyPad(xl[2 * low], wg[2 * low], panelCopy, pad);
    DataCopyPad(xl[2 * CTRSV_CACHE + low], wg[2 * CTRSV_CACHE + low], scaleCopy, pad);
    PipeBarrier<PIPE_ALL>();
    if (!update) {
        SolveStagePanel(wg, al, xl, t, first, count, low);
    } else {
        UpdateStageRows(wg, ag, al, xl, bp, t, first, count, low, row, end);
    }
}

void ctrsv_kernel_do(uint8_t* a, uint8_t* x, CtrsvTilingData t, void* stream, void* workspace)
{
    if (workspace != nullptr && t.n >= CTRSV_UPDATE_ROWS && t.n <= CTRSV_CACHE && t.diag != ACLBLAS_UNIT) {
        auto* work = reinterpret_cast<uint8_t*>(workspace);
        ctrsv_prepack<<<(t.n + CTRSV_PANEL_SIZE - 1) / CTRSV_PANEL_SIZE, CTRSV_PACK_UB_BYTES, stream>>>(a, work, t);
        ctrsv_transfer<<<1, CTRSV_TRANSFER_UB_BYTES, stream>>>(x, work, t, false);
        bool forward = (t.uplo == ACLBLAS_UPPER) == (t.trans != ACLBLAS_OP_N);
        for (uint32_t first = 0, count = 0; first < t.n; first += count) {
            count = forward ? (t.n - first < CTRSV_PANEL_SIZE ? t.n - first : CTRSV_PANEL_SIZE) :
                              (t.n - first - 1) % CTRSV_PANEL_SIZE + 1;
            ctrsv_stage<<<1, CTRSV_SOLVE_UB_BYTES, stream>>>(a, work, t, first, false);
            if (first + count < t.n) {
                ctrsv_stage<<<(t.n + CTRSV_UPDATE_ROWS - 1) / CTRSV_UPDATE_ROWS, CTRSV_SOLVE_UB_BYTES, stream>>>(
                    a, work, t, first, true);
            }
        }
        ctrsv_transfer<<<1, CTRSV_TRANSFER_UB_BYTES, stream>>>(x, work, t, true);
        return;
    }
    if (t.n <= CTRSV_CACHE) {
        ctrsv_kernel<<<1, CTRSV_SOLVE_UB_BYTES, stream>>>(a, x, t);
    } else {
        ctrsv_fallback<<<1, CTRSV_SOLVE_UB_BYTES, stream>>>(a, x, t);
    }
}
