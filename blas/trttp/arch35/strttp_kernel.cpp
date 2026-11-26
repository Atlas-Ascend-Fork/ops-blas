/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRTTP_KERNEL_H
#define STRTTP_KERNEL_H

#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "strttp_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTES_PER_FLOAT = 4;
constexpr uint32_t ELEMENTS_PER_BLOCK = 8;
constexpr uint32_t MAX_CHUNK_SIZE = 28672;

template <typename T>
class TrttpAIV {
public:
    __aicore__ inline TrttpAIV(TPipe& pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR aFull, GM_ADDR aPacked, const TrttpTilingData& tiling);
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    TrttpTilingData tiling_;

    GlobalTensor<T> aGM;
    GlobalTensor<T> apGM;
    TBuf<TPosition::VECIN> copyBuf;

    uint32_t startCol_;
    uint32_t colCount_;

    __aicore__ inline uint32_t CalcColLen(uint32_t col) const;
    __aicore__ inline uint32_t CalcSrcOffset(uint32_t col) const;
    __aicore__ inline uint32_t CalcDstOffset(uint32_t col) const;
    __aicore__ inline void CopyColumn(uint32_t col, int32_t eventIdM2M3, int32_t eventIdM3M2);
};

template <typename T>
__aicore__ inline void TrttpAIV<T>::Init(GM_ADDR aFull, GM_ADDR aPacked, const TrttpTilingData& tiling)
{
    tiling_ = tiling;
    aGM.SetGlobalBuffer((__gm__ T*)aFull);
    apGM.SetGlobalBuffer((__gm__ T*)aPacked);
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
__aicore__ inline uint32_t TrttpAIV<T>::CalcColLen(uint32_t col) const
{
    return (tiling_.uplo == 0) ? (tiling_.n - col) : (col + 1);
}

template <typename T>
__aicore__ inline uint32_t TrttpAIV<T>::CalcSrcOffset(uint32_t col) const
{
    return (tiling_.uplo == 0) ? (col * tiling_.lda + col) : (col * tiling_.lda);
}

template <typename T>
__aicore__ inline uint32_t TrttpAIV<T>::CalcDstOffset(uint32_t col) const
{
    if (tiling_.uplo == 0) {
        return static_cast<uint32_t>((static_cast<uint64_t>(col) * (2ULL * tiling_.n - col + 1)) / 2);
    } else {
        return static_cast<uint32_t>((static_cast<uint64_t>(col) * (col + 1)) / 2);
    }
}

template <typename T>
__aicore__ inline void TrttpAIV<T>::CopyColumn(uint32_t col, int32_t eventIdM2M3, int32_t eventIdM3M2)
{
    uint32_t colLen = CalcColLen(col);
    uint32_t srcOff = CalcSrcOffset(col);
    uint32_t dstOff = CalcDstOffset(col);

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
        DataCopyPad(ub, aGM[srcOff + processed], cp, pp);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdM2M3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdM2M3);

        DataCopyPad(apGM[dstOff + processed], ub, cp);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdM3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdM3M2);

        processed += chunkSize;
    }
}

template <typename T>
__aicore__ inline void TrttpAIV<T>::Process()
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

__global__ __aicore__ void strttp_kernel(GM_ADDR aFull, GM_ADDR aPacked, TrttpTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TrttpAIV<float> op(pipe);
    op.Init(aFull, aPacked, tiling);
    op.Process();
}

aclError strttp_kernel_do(
    GM_ADDR aFull, GM_ADDR aPacked, const TrttpTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    strttp_kernel<<<numBlocks, nullptr, stream>>>(aFull, aPacked, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

#endif // STRTTP_KERNEL_H
