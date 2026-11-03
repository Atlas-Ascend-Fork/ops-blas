/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRSV_KERNEL_H
#define STRSV_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t MAX_UB_ELEMENTS = UB_SIZE / BYTENUM_PER_FLOAT32;
constexpr uint32_t MAX_CORE_NUM = 40;

enum class UploMode { UPPER, LOWER };
enum class TransMode { NO_TRANS, TRANS };
enum class DiagMode { UNIT, NON_UNIT };

struct StrsvTilingDataDevice {
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    uint32_t startRow[MAX_CORE_NUM];
    uint32_t rowCount[MAX_CORE_NUM];
};

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
class StrsvKernel {
public:
    __aicore__ inline StrsvKernel() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void ProcessLower(int64_t rowStart, int64_t rowCount);
    __aicore__ inline void ProcessUpper(int64_t rowStart, int64_t rowCount);
    __aicore__ inline void CopyIn(int64_t row);
    __aicore__ inline void Compute(int64_t row);
    __aicore__ inline void CopyOut(int64_t row);

    TPipe pipe;

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue;

    int32_t blockIdx;
    int32_t blockNum;
    int64_t n;
    int64_t lda;

    uint32_t rowsPerCore;
    uint32_t rowStart;
    uint32_t rowCount;
};

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
__aicore__ inline void StrsvKernel<T, UPLO, TRANS, DIAG>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tiling = reinterpret_cast<__gm__ StrsvTilingDataDevice*>(tilingGm);

    this->n = tiling->n;
    this->lda = tiling->lda;
    this->blockNum = tiling->useCoreNum;
    this->rowStart = tiling->startRow[this->blockIdx];
    this->rowCount = tiling->rowCount[this->blockIdx];
}

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
__aicore__ inline void StrsvKernel<T, UPLO, TRANS, DIAG>::Init(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    this->blockNum = GetBlockNum();
    this->blockIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A), static_cast<uint32_t>(n * lda));
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), static_cast<uint32_t>(n));

    uint32_t maxRowElements = static_cast<uint32_t>(n);
    pipe.InitBuffer(aQueue, BUFFER_NUM, maxRowElements * sizeof(T));
    pipe.InitBuffer(xQueue, BUFFER_NUM, sizeof(T));
}

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
__aicore__ inline void StrsvKernel<T, UPLO, TRANS, DIAG>::CopyIn(int64_t row)
{
    LocalTensor<T> aLocal = aQueue.AllocTensor<T>();
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();

    for (int64_t col = 0; col < n; ++col) {
        int64_t aOffset;
        if constexpr (TRANS == TransMode::TRANS) {
            aOffset = row * lda + col;
        } else {
            aOffset = col * lda + row;
        }
        aLocal.SetValue(col, aGM.GetValue(aOffset));
    }
    xLocal.SetValue(0, xGM.GetValue(row));

    aQueue.EnQue<T>(aLocal);
    xQueue.EnQue<T>(xLocal);
}

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
__aicore__ inline void StrsvKernel<T, UPLO, TRANS, DIAG>::Compute(int64_t row)
{
    LocalTensor<T> aLocal = aQueue.DeQue<T>();
    LocalTensor<T> xLocal = xQueue.DeQue<T>();

    T sum = xLocal.GetValue(0);

    if constexpr (UPLO == UploMode::LOWER) {
        for (int64_t col = 0; col < row; ++col) {
            T aVal = aLocal.GetValue(col);
            T xVal = xGM.GetValue(col);
            sum -= aVal * xVal;
        }
        if constexpr (DIAG == DiagMode::NON_UNIT) {
            T diagVal = aLocal.GetValue(row);
            sum = sum / diagVal;
        }
    } else {
        for (int64_t col = row + 1; col < n; ++col) {
            T aVal = aLocal.GetValue(col);
            T xVal = xGM.GetValue(col);
            sum -= aVal * xVal;
        }
        if constexpr (DIAG == DiagMode::NON_UNIT) {
            T diagVal = aLocal.GetValue(row);
            sum = sum / diagVal;
        }
    }

    xLocal.SetValue(0, sum);
    xQueue.EnQue<T>(xLocal);
    aQueue.FreeTensor(aLocal);
}

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
__aicore__ inline void StrsvKernel<T, UPLO, TRANS, DIAG>::CopyOut(int64_t row)
{
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    xGM.SetValue(row, xLocal.GetValue(0));
    xQueue.FreeTensor(xLocal);
}

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
__aicore__ inline void StrsvKernel<T, UPLO, TRANS, DIAG>::ProcessLower(int64_t rowStart, int64_t rowCount)
{
    for (int64_t i = 0; i < rowCount; ++i) {
        int64_t globalRow = rowStart + i;
        CopyIn(globalRow);
        Compute(globalRow);
        CopyOut(globalRow);
    }
}

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
__aicore__ inline void StrsvKernel<T, UPLO, TRANS, DIAG>::ProcessUpper(int64_t rowStart, int64_t rowCount)
{
    for (int64_t i = rowCount - 1; i >= 0; --i) {
        int64_t globalRow = rowStart + i;
        CopyIn(globalRow);
        Compute(globalRow);
        CopyOut(globalRow);
    }
}

template <typename T, UploMode UPLO, TransMode TRANS, DiagMode DIAG>
__aicore__ inline void StrsvKernel<T, UPLO, TRANS, DIAG>::Process()
{
    if (rowCount == 0) {
        return;
    }

    if constexpr (UPLO == UploMode::LOWER) {
        ProcessLower(rowStart, rowCount);
    } else {
        ProcessUpper(rowStart, rowCount);
    }
}

__global__ __aicore__ void strsv_kernel_lower_no_trans_non_unit(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StrsvKernel<float, UploMode::LOWER, TransMode::NO_TRANS, DiagMode::NON_UNIT> op;
    op.Init(A, x, tilingGm);
    op.Process();
}

__global__ __aicore__ void strsv_kernel_lower_no_trans_unit(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StrsvKernel<float, UploMode::LOWER, TransMode::NO_TRANS, DiagMode::UNIT> op;
    op.Init(A, x, tilingGm);
    op.Process();
}

__global__ __aicore__ void strsv_kernel_upper_no_trans_non_unit(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StrsvKernel<float, UploMode::UPPER, TransMode::NO_TRANS, DiagMode::NON_UNIT> op;
    op.Init(A, x, tilingGm);
    op.Process();
}

__global__ __aicore__ void strsv_kernel_upper_no_trans_unit(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StrsvKernel<float, UploMode::UPPER, TransMode::NO_TRANS, DiagMode::UNIT> op;
    op.Init(A, x, tilingGm);
    op.Process();
}

__global__ __aicore__ void strsv_kernel_lower_trans_non_unit(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StrsvKernel<float, UploMode::LOWER, TransMode::TRANS, DiagMode::NON_UNIT> op;
    op.Init(A, x, tilingGm);
    op.Process();
}

__global__ __aicore__ void strsv_kernel_lower_trans_unit(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StrsvKernel<float, UploMode::LOWER, TransMode::TRANS, DiagMode::UNIT> op;
    op.Init(A, x, tilingGm);
    op.Process();
}

__global__ __aicore__ void strsv_kernel_upper_trans_non_unit(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StrsvKernel<float, UploMode::UPPER, TransMode::TRANS, DiagMode::NON_UNIT> op;
    op.Init(A, x, tilingGm);
    op.Process();
}

__global__ __aicore__ void strsv_kernel_upper_trans_unit(GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    StrsvKernel<float, UploMode::UPPER, TransMode::TRANS, DiagMode::UNIT> op;
    op.Init(A, x, tilingGm);
    op.Process();
}

void strsv_kernel_do(
    GM_ADDR A, GM_ADDR x, GM_ADDR tilingGm, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int lda, uint32_t numBlocks, void* stream)
{
    if (uplo == ACLBLAS_LOWER) {
        if (trans == ACLBLAS_OP_N) {
            if (diag == ACLBLAS_NON_UNIT) {
                strsv_kernel_lower_no_trans_non_unit<<<numBlocks, nullptr, stream>>>(A, x, tilingGm);
            } else {
                strsv_kernel_lower_no_trans_unit<<<numBlocks, nullptr, stream>>>(A, x, tilingGm);
            }
        } else {
            if (diag == ACLBLAS_NON_UNIT) {
                strsv_kernel_lower_trans_non_unit<<<numBlocks, nullptr, stream>>>(A, x, tilingGm);
            } else {
                strsv_kernel_lower_trans_unit<<<numBlocks, nullptr, stream>>>(A, x, tilingGm);
            }
        }
    } else {
        if (trans == ACLBLAS_OP_N) {
            if (diag == ACLBLAS_NON_UNIT) {
                strsv_kernel_upper_no_trans_non_unit<<<numBlocks, nullptr, stream>>>(A, x, tilingGm);
            } else {
                strsv_kernel_upper_no_trans_unit<<<numBlocks, nullptr, stream>>>(A, x, tilingGm);
            }
        } else {
            if (diag == ACLBLAS_NON_UNIT) {
                strsv_kernel_upper_trans_non_unit<<<numBlocks, nullptr, stream>>>(A, x, tilingGm);
            } else {
                strsv_kernel_upper_trans_unit<<<numBlocks, nullptr, stream>>>(A, x, tilingGm);
            }
        }
    }
}

#endif // STRSV_KERNEL_H
