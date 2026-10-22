/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STPTTR_KERNEL_H
#define STPTTR_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "stpttr_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTES_PER_FLOAT = 4;
constexpr uint32_t ELEMENTS_PER_BLOCK = 8;
constexpr uint32_t MAX_CHUNK_SIZE = 28672;

template <typename T>
class TpttrAIV {
public:
    __aicore__ inline TpttrAIV(TPipe& pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR aPacked, GM_ADDR aFull, const TpttrTilingData& tiling);
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    TpttrTilingData tiling_;

    GlobalTensor<T> aPGM;
    GlobalTensor<T> aGM;
    TBuf<TPosition::VECIN> copyBuf;

    uint32_t startCol_;
    uint32_t colCount_;

    __aicore__ inline uint32_t CalcColLen(uint32_t col) const;
    __aicore__ inline uint64_t CalcSrcOffset(uint32_t col) const;
    __aicore__ inline uint64_t CalcDstOffset(uint32_t col) const;
    __aicore__ inline void CopyColumn(uint32_t col, int32_t eventIdM2M3, int32_t eventIdM3M2);
};

template <typename T>
__aicore__ inline void TpttrAIV<T>::Init(GM_ADDR aPacked, GM_ADDR aFull, const TpttrTilingData& tiling)
{
    tiling_ = tiling;
    aPGM.SetGlobalBuffer((__gm__ T*)aPacked);
    aGM.SetGlobalBuffer((__gm__ T*)aFull);
    pipe_.InitBuffer(copyBuf, MAX_CHUNK_SIZE * sizeof(T));

    uint32_t uN = static_cast<uint32_t>(tiling_.n);
    uint32_t baseCols = uN / tiling_.useCoreNum;
    uint32_t remainCols = uN % tiling_.useCoreNum;
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx < remainCols) {
        startCol_ = blockIdx * (baseCols + 1);
        colCount_ = baseCols + 1;
    } else {
        startCol_ = remainCols * (baseCols + 1) + (blockIdx - remainCols) * baseCols;
        colCount_ = baseCols;
    }
}

template <typename T>
__aicore__ inline uint32_t TpttrAIV<T>::CalcColLen(uint32_t col) const
{
    return (tiling_.uplo == 0) ? (tiling_.n - col) : (col + 1);
}

template <typename T>
__aicore__ inline uint64_t TpttrAIV<T>::CalcSrcOffset(uint32_t col) const
{
    if (tiling_.uplo == 0) {
        const uint64_t uCol = static_cast<uint64_t>(col);
        const uint64_t uN = static_cast<uint64_t>(tiling_.n);
        return (uCol * (2ULL * uN - uCol + 1ULL)) / 2ULL;
    } else {
        const uint64_t uCol = static_cast<uint64_t>(col);
        return (uCol * (uCol + 1ULL)) / 2ULL;
    }
}

template <typename T>
__aicore__ inline uint64_t TpttrAIV<T>::CalcDstOffset(uint32_t col) const
{
    const uint64_t uCol = static_cast<uint64_t>(col);
    const uint64_t uLda = static_cast<uint64_t>(tiling_.lda);
    return (tiling_.uplo == 0) ? (uCol * uLda + uCol) : (uCol * uLda);
}

template <typename T>
__aicore__ inline void TpttrAIV<T>::CopyColumn(uint32_t col, int32_t eventIdM2M3, int32_t eventIdM3M2)
{
    uint32_t colLen = CalcColLen(col);
    uint64_t srcOff = CalcSrcOffset(col);
    uint64_t dstOff = CalcDstOffset(col);

    uint32_t processed = 0;
    while (processed < colLen) {
        uint32_t chunkSize = colLen - processed;
        if (chunkSize > MAX_CHUNK_SIZE) {
            chunkSize = MAX_CHUNK_SIZE;
        }

        uint8_t paddingNum = 0;
        uint32_t remainder = chunkSize % ELEMENTS_PER_BLOCK;
        if (remainder != 0) {
            paddingNum = static_cast<uint8_t>(ELEMENTS_PER_BLOCK - remainder);
        }

        uint32_t blockBytes = chunkSize * sizeof(T);
        DataCopyExtParams cp{1, blockBytes, 0, 0, 0};
        DataCopyPadExtParams<T> pp{true, 0, paddingNum, static_cast<T>(0)};

        LocalTensor<T> ub = copyBuf.Get<T>();
        DataCopyPad(ub, aPGM[srcOff + processed], cp, pp);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdM2M3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdM2M3);

        DataCopyPad(aGM[dstOff + processed], ub, cp);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdM3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdM3M2);

        processed += chunkSize;
    }
}

template <typename T>
__aicore__ inline void TpttrAIV<T>::Process()
{
    if (colCount_ == 0) {
        return;
    }
    int32_t eventIdM2M3 = static_cast<int32_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    int32_t eventIdM3M2 = static_cast<int32_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    for (uint32_t c = 0; c < colCount_; c++) {
        CopyColumn(startCol_ + c, eventIdM2M3, eventIdM3M2);
    }
}

__global__ __aicore__ void stpttr_kernel(GM_ADDR aPacked, GM_ADDR aFull, GM_ADDR workSpace, TpttrTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TpttrAIV<float> op(pipe);
    op.Init(aPacked, aFull, tiling);
    op.Process();
}

void stpttr_kernel_do(
    GM_ADDR aPacked, GM_ADDR aFull, GM_ADDR workSpace, const TpttrTilingData& tiling, uint32_t numBlocks, void* stream)
{
    stpttr_kernel<<<numBlocks, nullptr, stream>>>(aPacked, aFull, workSpace, tiling);
}

#endif // STPTTR_KERNEL_H
