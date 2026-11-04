/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file stpsv_kernel.cpp
 * \brief Single-precision triangular packed solver kernel (scalar path, n < 128).
 *
 * For n >= 128, stpsv_kernel_simt.cpp provides a SIMT VF parallelized kernel.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "stpsv_tiling_data.h"
#include "common/helper/kernel_constant.h"
#include "stpsv_kernel_utils.h"

using namespace AscendC;

// ==========================================================================
//  Kernel class — parameterized on uplo / trans / diag
// ==========================================================================

enum class TpsvUplo { UPPER, LOWER };
enum class TpsvTrans { NO_TRANS, TRANS };
enum class TpsvDiag { UNIT, NON_UNIT };

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
class StpsvKernel {
public:
    __aicore__ inline StpsvKernel() {}
    __aicore__ inline void Init(StpsvTilingData tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessForward();
    __aicore__ inline void ProcessBackward();
    __aicore__ inline float GetDiag(uint32_t row);
    __aicore__ inline float GetElemOffDiag(uint32_t row, uint32_t col);
    __aicore__ inline uint32_t XOffset(uint32_t idx);
    __aicore__ inline void CopyIn(uint32_t row);
    __aicore__ inline void Compute(uint32_t row);
    __aicore__ inline void CopyOut(uint32_t row);

    TPipe pipe;

    GlobalTensor<float> apGM;
    GlobalTensor<float> xGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xOutQueue;

    static constexpr bool kForward = (UPLO == TpsvUplo::LOWER && TRANS == TpsvTrans::NO_TRANS) ||
                                     (UPLO == TpsvUplo::UPPER && TRANS == TpsvTrans::TRANS);

    uint32_t n;
    int64_t incx;
};

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline uint32_t StpsvKernel<UPLO, TRANS, DIAG>::XOffset(uint32_t idx)
{
    if (incx >= 0) {
        return idx * static_cast<uint32_t>(incx);
    } else {
        return (n - 1 - idx) * static_cast<uint32_t>(-incx);
    }
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline void StpsvKernel<UPLO, TRANS, DIAG>::Init(StpsvTilingData tiling)
{
    this->n = tiling.n;
    this->incx = tiling.incx;

    uint32_t apCount = n * (n + 1) / 2;
    // Host ValidateTpsvParams rejects (n-1)*|incx| >= UINT32_MAX before launch, so absIncx*(n-1)+1
    // and the per-element offsets below cannot wrap in uint32; the host guard is the only line of
    // defense for the uint32 arithmetic here.
    uint32_t absIncx = static_cast<uint32_t>(incx >= 0 ? incx : -incx);
    uint32_t xCount = (n > 0) ? (absIncx * (n - 1) + 1) : 0;
    apGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.ap), apCount);
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x), xCount);

    pipe.InitBuffer(xInQueue, BUFFER_NUM, sizeof(float));
    pipe.InitBuffer(xOutQueue, BUFFER_NUM, sizeof(float));
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline float StpsvKernel<UPLO, TRANS, DIAG>::GetDiag(uint32_t row)
{
    if constexpr (UPLO == TpsvUplo::LOWER) {
        return apGM.GetValue(TpsvPackedLowerIdx(row, row, n));
    } else {
        return apGM.GetValue(TpsvPackedUpperIdx(row, row));
    }
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline float StpsvKernel<UPLO, TRANS, DIAG>::GetElemOffDiag(uint32_t row, uint32_t col)
{
    if constexpr (UPLO == TpsvUplo::LOWER) {
        if constexpr (TRANS == TpsvTrans::NO_TRANS) {
            return apGM.GetValue(TpsvPackedLowerIdx(row, col, n));
        } else {
            return apGM.GetValue(TpsvPackedLowerIdx(col, row, n));
        }
    } else {
        if constexpr (TRANS == TpsvTrans::NO_TRANS) {
            return apGM.GetValue(TpsvPackedUpperIdx(row, col));
        } else {
            return apGM.GetValue(TpsvPackedUpperIdx(col, row));
        }
    }
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline void StpsvKernel<UPLO, TRANS, DIAG>::CopyIn(uint32_t row)
{
    LocalTensor<float> xLocal = xInQueue.AllocTensor<float>();
    xLocal.SetValue(0, xGM.GetValue(XOffset(row)));
    xInQueue.EnQue<float>(xLocal);
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline void StpsvKernel<UPLO, TRANS, DIAG>::Compute(uint32_t row)
{
    LocalTensor<float> xLocal = xInQueue.DeQue<float>();
    float sum = xLocal.GetValue(0);

    if constexpr (kForward) {
        for (uint32_t j = 0; j < row; ++j) {
            sum -= GetElemOffDiag(row, j) * xGM.GetValue(XOffset(j));
        }
        if constexpr (DIAG == TpsvDiag::NON_UNIT) {
            sum = sum / GetDiag(row);
        }
    } else {
        for (uint32_t j = row + 1; j < n; ++j) {
            sum -= GetElemOffDiag(row, j) * xGM.GetValue(XOffset(j));
        }
        if constexpr (DIAG == TpsvDiag::NON_UNIT) {
            sum = sum / GetDiag(row);
        }
    }

    xLocal.SetValue(0, sum);
    xOutQueue.EnQue<float>(xLocal);
    xInQueue.FreeTensor(xLocal);
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline void StpsvKernel<UPLO, TRANS, DIAG>::CopyOut(uint32_t row)
{
    LocalTensor<float> xLocal = xOutQueue.DeQue<float>();
    xGM.SetValue(XOffset(row), xLocal.GetValue(0));
    xOutQueue.FreeTensor(xLocal);
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline void StpsvKernel<UPLO, TRANS, DIAG>::ProcessForward()
{
    for (uint32_t i = 0; i < n; ++i) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline void StpsvKernel<UPLO, TRANS, DIAG>::ProcessBackward()
{
    for (uint32_t i = n; i-- > 0;) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

template <TpsvUplo UPLO, TpsvTrans TRANS, TpsvDiag DIAG>
__aicore__ inline void StpsvKernel<UPLO, TRANS, DIAG>::Process()
{
    if constexpr (kForward) {
        ProcessForward();
    } else {
        ProcessBackward();
    }
}

// ==========================================================================
//  Kernel entry points (8 combinations)
// ==========================================================================

#define DEFINE_TPSV_KERNEL(uplo, trans, diag, name)                       \
    __global__ __aicore__ void name(StpsvTilingData tiling)               \
    {                                                                     \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                   \
        StpsvKernel<TpsvUplo::uplo, TpsvTrans::trans, TpsvDiag::diag> op; \
        op.Init(tiling);                                                  \
        op.Process();                                                     \
    }

DEFINE_TPSV_KERNEL(LOWER, NO_TRANS, NON_UNIT, stpsv_kernel_lower_no_trans_non_unit)
DEFINE_TPSV_KERNEL(LOWER, NO_TRANS, UNIT, stpsv_kernel_lower_no_trans_unit)
DEFINE_TPSV_KERNEL(UPPER, NO_TRANS, NON_UNIT, stpsv_kernel_upper_no_trans_non_unit)
DEFINE_TPSV_KERNEL(UPPER, NO_TRANS, UNIT, stpsv_kernel_upper_no_trans_unit)
DEFINE_TPSV_KERNEL(LOWER, TRANS, NON_UNIT, stpsv_kernel_lower_trans_non_unit)
DEFINE_TPSV_KERNEL(LOWER, TRANS, UNIT, stpsv_kernel_lower_trans_unit)
DEFINE_TPSV_KERNEL(UPPER, TRANS, NON_UNIT, stpsv_kernel_upper_trans_non_unit)
DEFINE_TPSV_KERNEL(UPPER, TRANS, UNIT, stpsv_kernel_upper_trans_unit)

#undef DEFINE_TPSV_KERNEL

// ==========================================================================
//  Kernel dispatcher
// ==========================================================================

void stpsv_simt_kernel_do(const StpsvTilingData& tiling, void* stream);

void stpsv_kernel_do(const StpsvTilingData& tiling, void* stream)
{
    // SIMT path: dispatch to kernel in stpsv_kernel_simt.cpp for n >= 128
    if (tiling.numThreads > 0) {
        stpsv_simt_kernel_do(tiling, stream);
        return;
    }

    // Scalar path: n < 128
    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stpsv_kernel_lower_no_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stpsv_kernel_lower_no_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stpsv_kernel_lower_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stpsv_kernel_lower_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stpsv_kernel_upper_no_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stpsv_kernel_upper_no_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stpsv_kernel_upper_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stpsv_kernel_upper_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        }
    }
}
