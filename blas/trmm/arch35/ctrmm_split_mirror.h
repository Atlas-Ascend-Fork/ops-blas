/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

/*!
 * \file ctrmm_split_mirror.h
 * \brief SIMT A-side mirror element/compute (extracted verbatim from
 *        ctrmm_kernel.cpp so the split/mirror helpers can be instantiated
 *        WITHOUT altering the production instantiations' code generation).
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/kernel_utils.h"

#include "ctrmm_tiling_data.h"

// Same alias the production translation unit uses.
namespace te = AscendC::Te;

template <bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ inline void CtrmmMirrorElement(
    const __gm__ float* __restrict aGm, int64_t lda64, uint32_t row, uint32_t col, float& vr, float& vi)
{
    vr = 0.0f;
    vi = 0.0f;
    bool inTriangle;
    int64_t aIdx;
    if constexpr (TRANS_IS_T) {
        // op(A)(row,col) = A(col,row) (+conj)
        inTriangle = UPLO_IS_UPPER ? (col <= row) : (col >= row);
        aIdx = static_cast<int64_t>(row) * lda64 + static_cast<int64_t>(col);
    } else {
        inTriangle = UPLO_IS_UPPER ? (row <= col) : (row >= col);
        aIdx = static_cast<int64_t>(col) * lda64 + static_cast<int64_t>(row);
    }
    if (inTriangle) {
        vr = aGm[2 * aIdx];
        vi = aGm[2 * aIdx + 1];
        if constexpr (CONJ) {
            vi = -vi;
        }
    }
    if constexpr (DIAG_IS_UNIT) {
        if (row == col) {
            vr = 1.0f;
            vi = 0.0f;
        }
    }
}
template <bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtrmmSplitMirrorCompute(
    uint32_t dimA, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, const __gm__ float* __restrict aGm,
    const __gm__ float* __restrict bGm, __gm__ float* __restrict arGm, __gm__ float* __restrict aiGm,
    __gm__ float* __restrict brGm, __gm__ float* __restrict biGm, __gm__ float* __restrict asumGm,
    __gm__ float* __restrict bsumGm, uint32_t rowStartA, uint32_t rowEndA, uint32_t rowStartB, uint32_t rowEndB,
    bool diagUnit, uint32_t skipB)
{
    const int64_t lda64 = static_cast<int64_t>(lda);
    const int64_t ldb64 = static_cast<int64_t>(ldb);
    if (diagUnit) {
        // A-mirror rows, UNIT instantiation.
        for (uint32_t row = rowStartA + threadIdx.x; row < rowEndA; row += blockDim.x) {
            for (uint32_t col = 0; col < dimA; ++col) {
                float vr;
                float vi;
                CtrmmMirrorElement<UPLO_IS_UPPER, TRANS_IS_T, CONJ, true>(aGm, lda64, row, col, vr, vi);
                const int64_t o = static_cast<int64_t>(col) * lda64 + static_cast<int64_t>(row);
                arGm[o] = vr;
                aiGm[o] = vi;
                asumGm[o] = vr + vi;
            }
        }
    } else {
        for (uint32_t row = rowStartA + threadIdx.x; row < rowEndA; row += blockDim.x) {
            for (uint32_t col = 0; col < dimA; ++col) {
                float vr;
                float vi;
                CtrmmMirrorElement<UPLO_IS_UPPER, TRANS_IS_T, CONJ, false>(aGm, lda64, row, col, vr, vi);
                const int64_t o = static_cast<int64_t>(col) * lda64 + static_cast<int64_t>(row);
                arGm[o] = vr;
                aiGm[o] = vi;
                asumGm[o] = vr + vi;
            }
        }
    }
    // The B-side split is always handled by the tensor_api kernel (the host
    // sets skipB=1 unconditionally), so this function covers the A-side only.
    (void)m;
    (void)skipB;
    (void)rowStartB;
    (void)rowEndB;
    (void)n;
    (void)ldb64;
    (void)bGm;
    (void)brGm;
    (void)biGm;
    (void)bsumGm;
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_T>
__aicore__ inline void CtrmmDispatchSplitConj(
    bool conj, bool diagUnit, uint32_t dimA, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb,
    const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* arGm, __gm__ float* aiGm, __gm__ float* brGm,
    __gm__ float* biGm, __gm__ float* asumGm, __gm__ float* bsumGm, uint32_t rowStartA, uint32_t rowEndA,
    uint32_t rowStartB, uint32_t rowEndB, uint32_t nthreads, uint32_t skipB)
{
    if (conj) {
        asc_vf_call<CtrmmSplitMirrorCompute<UPLO_IS_UPPER, TRANS_IS_T, true>>(
            dim3{nthreads, 1, 1}, dimA, m, n, lda, ldb, aGm, bGm, arGm, aiGm, brGm, biGm, asumGm, bsumGm, rowStartA,
            rowEndA, rowStartB, rowEndB, diagUnit, skipB);
    } else {
        asc_vf_call<CtrmmSplitMirrorCompute<UPLO_IS_UPPER, TRANS_IS_T, false>>(
            dim3{nthreads, 1, 1}, dimA, m, n, lda, ldb, aGm, bGm, arGm, aiGm, brGm, biGm, asumGm, bsumGm, rowStartA,
            rowEndA, rowStartB, rowEndB, diagUnit, skipB);
    }
}

template <bool UPLO_IS_UPPER>
__aicore__ inline void CtrmmDispatchSplitTrans(
    uint32_t transMode, bool diagUnit, uint32_t dimA, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb,
    const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* arGm, __gm__ float* aiGm, __gm__ float* brGm,
    __gm__ float* biGm, __gm__ float* asumGm, __gm__ float* bsumGm, uint32_t rowStartA, uint32_t rowEndA,
    uint32_t rowStartB, uint32_t rowEndB, uint32_t nthreads, uint32_t skipB)
{
    if (transMode == ACLBLAS_OP_N) {
        CtrmmDispatchSplitConj<UPLO_IS_UPPER, false>(
            false, diagUnit, dimA, m, n, lda, ldb, aGm, bGm, arGm, aiGm, brGm, biGm, asumGm, bsumGm, rowStartA, rowEndA,
            rowStartB, rowEndB, nthreads, skipB);
    } else if (transMode == ACLBLAS_OP_T) {
        CtrmmDispatchSplitConj<UPLO_IS_UPPER, true>(
            false, diagUnit, dimA, m, n, lda, ldb, aGm, bGm, arGm, aiGm, brGm, biGm, asumGm, bsumGm, rowStartA, rowEndA,
            rowStartB, rowEndB, nthreads, skipB);
    } else {
        CtrmmDispatchSplitConj<UPLO_IS_UPPER, true>(
            true, diagUnit, dimA, m, n, lda, ldb, aGm, bGm, arGm, aiGm, brGm, biGm, asumGm, bsumGm, rowStartA, rowEndA,
            rowStartB, rowEndB, nthreads, skipB);
    }
}
