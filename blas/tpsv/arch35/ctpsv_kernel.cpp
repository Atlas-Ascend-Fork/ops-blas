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
 * \file ctpsv_kernel.cpp
 * \brief Single-precision complex triangular packed solver kernel (scalar path, n < 128).
 *
 * For n >= 128, ctpsv_kernel_simt.cpp provides a SIMT VF parallelized kernel.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "ctpsv_tiling_data.h"
#include "common/helper/kernel_constant.h"
#include "ctpsv_kernel_utils.h"

using namespace AscendC;

// ==========================================================================
//  Complex (float32 pairs) helpers
// ==========================================================================

__aicore__ inline float CtpsvAbsf(float v) { return (v < 0.0f) ? -v : v; }

__aicore__ inline void CtpsvCmul(float ar, float ai, float br, float bi, float& cr, float& ci)
{
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

// Complex division following the Smith algorithm used by libgcc's __divsc3, so that
// the kernel result matches the complex division performed by the reference BLAS golden.
// Defined by the shared CTPSV_DEFINE_CDIV macro (see ctpsv_kernel_utils.h).
CTPSV_DEFINE_CDIV(__aicore__, CtpsvCdiv, CtpsvAbsf)

// ==========================================================================
//  Kernel class — parameterized on uplo / trans / diag
// ==========================================================================

enum class CtpsvUplo { UPPER, LOWER };
enum class CtpsvTrans { NO_TRANS, TRANS, CONJ_TRANS };
enum class CtpsvDiag { UNIT, NON_UNIT };

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
class CtpsvKernel {
public:
    __aicore__ inline CtpsvKernel() {}
    __aicore__ inline void Init(CtpsvTilingData tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t XOffset(uint32_t idx);
    __aicore__ inline uint32_t ApIdxDiag(uint32_t row);
    __aicore__ inline uint32_t ApIdxOffDiag(uint32_t row, uint32_t col);
    __aicore__ inline void ComputeRow(uint32_t row);
    __aicore__ inline void Accumulate(uint32_t row, uint32_t jBegin, uint32_t jEnd, float& sumReal, float& sumImag);

    GlobalTensor<float> apGM;
    GlobalTensor<float> xGM;

    static constexpr bool kForward = (UPLO == CtpsvUplo::LOWER && TRANS == CtpsvTrans::NO_TRANS) ||
                                     (UPLO == CtpsvUplo::UPPER && TRANS != CtpsvTrans::NO_TRANS);
    static constexpr bool kConj = (TRANS == CtpsvTrans::CONJ_TRANS);

    uint32_t n;
    int64_t incx;
};

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline uint32_t CtpsvKernel<UPLO, TRANS, DIAG>::XOffset(uint32_t idx)
{
    if (incx >= 0) {
        return idx * static_cast<uint32_t>(incx);
    } else {
        return (n - 1 - idx) * static_cast<uint32_t>(-incx);
    }
}

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline uint32_t CtpsvKernel<UPLO, TRANS, DIAG>::ApIdxDiag(uint32_t row)
{
    if constexpr (UPLO == CtpsvUplo::LOWER) {
        return CtpsvPackedLowerIdx(row, row, n);
    } else {
        return CtpsvPackedUpperIdx(row, row);
    }
}

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline uint32_t CtpsvKernel<UPLO, TRANS, DIAG>::ApIdxOffDiag(uint32_t row, uint32_t col)
{
    if constexpr (UPLO == CtpsvUplo::LOWER) {
        if constexpr (TRANS == CtpsvTrans::NO_TRANS) {
            return CtpsvPackedLowerIdx(row, col, n);
        } else {
            return CtpsvPackedLowerIdx(col, row, n);
        }
    } else {
        if constexpr (TRANS == CtpsvTrans::NO_TRANS) {
            return CtpsvPackedUpperIdx(row, col);
        } else {
            return CtpsvPackedUpperIdx(col, row);
        }
    }
}

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline void CtpsvKernel<UPLO, TRANS, DIAG>::Init(CtpsvTilingData tiling)
{
    this->n = tiling.n;
    this->incx = tiling.incx;

    uint32_t apCount = n * (n + 1) / 2;
    uint32_t absIncx = static_cast<uint32_t>(incx >= 0 ? incx : -incx);
    uint32_t xCount = (n > 0) ? (absIncx * (n - 1) + 1) : 0;
    // Complex64 occupies two consecutive float32 entries in global memory.
    apGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.ap), apCount * 2U);
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x), xCount * 2U);
}

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline void CtpsvKernel<UPLO, TRANS, DIAG>::Accumulate(
    uint32_t row, uint32_t jBegin, uint32_t jEnd, float& sumReal, float& sumImag)
{
    for (uint32_t j = jBegin; j < jEnd; ++j) {
        uint32_t apIdx = ApIdxOffDiag(row, j);
        float aReal = apGM.GetValue(2U * apIdx);
        float aImag = apGM.GetValue(2U * apIdx + 1U);
        if constexpr (kConj) {
            aImag = -aImag;
        }
        uint32_t xjOff = XOffset(j);
        float xReal = xGM.GetValue(2U * xjOff);
        float xImag = xGM.GetValue(2U * xjOff + 1U);
        float pReal;
        float pImag;
        CtpsvCmul(aReal, aImag, xReal, xImag, pReal, pImag);
        sumReal -= pReal;
        sumImag -= pImag;
    }
}

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline void CtpsvKernel<UPLO, TRANS, DIAG>::ComputeRow(uint32_t row)
{
    uint32_t xOff = XOffset(row);
    float sumReal = xGM.GetValue(2U * xOff);
    float sumImag = xGM.GetValue(2U * xOff + 1U);

    if constexpr (kForward) {
        Accumulate(row, 0, row, sumReal, sumImag);
    } else {
        Accumulate(row, row + 1, n, sumReal, sumImag);
    }

    if constexpr (DIAG == CtpsvDiag::NON_UNIT) {
        uint32_t dIdx = ApIdxDiag(row);
        float dReal = apGM.GetValue(2U * dIdx);
        float dImag = apGM.GetValue(2U * dIdx + 1U);
        if constexpr (kConj) {
            dImag = -dImag;
        }
        float outReal;
        float outImag;
        CtpsvCdiv(sumReal, sumImag, dReal, dImag, outReal, outImag);
        sumReal = outReal;
        sumImag = outImag;
    }

    xGM.SetValue(2U * xOff, sumReal);
    xGM.SetValue(2U * xOff + 1U, sumImag);
}

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline void CtpsvKernel<UPLO, TRANS, DIAG>::Process()
{
    if constexpr (kForward) {
        for (uint32_t i = 0; i < n; ++i) {
            ComputeRow(i);
        }
    } else {
        for (uint32_t i = n; i-- > 0;) {
            ComputeRow(i);
        }
    }
}

// ==========================================================================
//  Kernel entry points (12 combinations)
// ==========================================================================

#define DEFINE_CTPSV_KERNEL(uplo, trans, diag, name)                         \
    __global__ __aicore__ void name(CtpsvTilingData tiling)                  \
    {                                                                        \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                      \
        CtpsvKernel<CtpsvUplo::uplo, CtpsvTrans::trans, CtpsvDiag::diag> op; \
        op.Init(tiling);                                                     \
        op.Process();                                                        \
    }

DEFINE_CTPSV_KERNEL(LOWER, NO_TRANS, NON_UNIT, ctpsv_kernel_lower_no_trans_non_unit)
DEFINE_CTPSV_KERNEL(LOWER, NO_TRANS, UNIT, ctpsv_kernel_lower_no_trans_unit)
DEFINE_CTPSV_KERNEL(UPPER, NO_TRANS, NON_UNIT, ctpsv_kernel_upper_no_trans_non_unit)
DEFINE_CTPSV_KERNEL(UPPER, NO_TRANS, UNIT, ctpsv_kernel_upper_no_trans_unit)
DEFINE_CTPSV_KERNEL(LOWER, TRANS, NON_UNIT, ctpsv_kernel_lower_trans_non_unit)
DEFINE_CTPSV_KERNEL(LOWER, TRANS, UNIT, ctpsv_kernel_lower_trans_unit)
DEFINE_CTPSV_KERNEL(UPPER, TRANS, NON_UNIT, ctpsv_kernel_upper_trans_non_unit)
DEFINE_CTPSV_KERNEL(UPPER, TRANS, UNIT, ctpsv_kernel_upper_trans_unit)
DEFINE_CTPSV_KERNEL(LOWER, CONJ_TRANS, NON_UNIT, ctpsv_kernel_lower_conj_trans_non_unit)
DEFINE_CTPSV_KERNEL(LOWER, CONJ_TRANS, UNIT, ctpsv_kernel_lower_conj_trans_unit)
DEFINE_CTPSV_KERNEL(UPPER, CONJ_TRANS, NON_UNIT, ctpsv_kernel_upper_conj_trans_non_unit)
DEFINE_CTPSV_KERNEL(UPPER, CONJ_TRANS, UNIT, ctpsv_kernel_upper_conj_trans_unit)

#undef DEFINE_CTPSV_KERNEL

// ==========================================================================
//  Kernel dispatcher
// ==========================================================================

void ctpsv_simt_kernel_do(const CtpsvTilingData& tiling, void* stream);

#define CTPSV_LAUNCH(uploTag, transTag)                                                     \
    do {                                                                                    \
        if (tiling.diag == ACLBLAS_NON_UNIT) {                                              \
            ctpsv_kernel_##uploTag##_##transTag##_non_unit<<<1, nullptr, stream>>>(tiling); \
        } else {                                                                            \
            ctpsv_kernel_##uploTag##_##transTag##_unit<<<1, nullptr, stream>>>(tiling);     \
        }                                                                                   \
    } while (0)

void ctpsv_kernel_do(const CtpsvTilingData& tiling, void* stream)
{
    // SIMT path: dispatch to kernel in ctpsv_kernel_simt.cpp for n >= 128
    if (tiling.numThreads > 0) {
        ctpsv_simt_kernel_do(tiling, stream);
        return;
    }

    // Scalar path: n < 128
    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            CTPSV_LAUNCH(lower, no_trans);
        } else if (tiling.trans == ACLBLAS_OP_T) {
            CTPSV_LAUNCH(lower, trans);
        } else {
            CTPSV_LAUNCH(lower, conj_trans);
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            CTPSV_LAUNCH(upper, no_trans);
        } else if (tiling.trans == ACLBLAS_OP_T) {
            CTPSV_LAUNCH(upper, trans);
        } else {
            CTPSV_LAUNCH(upper, conj_trans);
        }
    }
}

#undef CTPSV_LAUNCH
