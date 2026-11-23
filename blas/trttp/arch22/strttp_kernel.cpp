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
 * \file strttp_kernel.cpp
 * \brief Device-side Ascend C kernel for aclblasStrttp (arch22 / dav-2201).
 */

#include <cstdint>
#include "kernel_operator.h"
#include "strttp_tiling_data.h"
#include "strttp_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t STRTTP_ELEMENTS_PER_BLOCK = 8U;
constexpr uint32_t STRTTP_MAX_CHUNK_SIZE = 28672U;

class StrttpKernel {
public:
    __aicore__ inline StrttpKernel() = default;

    __aicore__ inline void Init(GM_ADDR aFull, GM_ADDR aPacked, const StrttpTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        if (tiling_.useCoreNum == 0U || tiling_.n == 0U) {
            colCount_ = 0U;
            return;
        }
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= tiling_.useCoreNum) {
            colCount_ = 0U;
            return;
        }
        aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aFull));
        apGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aPacked));
        pipe_->InitBuffer(copyBuf_, STRTTP_MAX_CHUNK_SIZE * sizeof(float));

        const uint32_t baseCols = tiling_.n / tiling_.useCoreNum;
        const uint32_t remainCols = tiling_.n % tiling_.useCoreNum;
        if (blockIdx < remainCols) {
            startCol_ = blockIdx * (baseCols + 1U);
            colCount_ = baseCols + 1U;
        } else {
            startCol_ = remainCols * (baseCols + 1U) + (blockIdx - remainCols) * baseCols;
            colCount_ = baseCols;
        }
    }

    __aicore__ inline void Process()
    {
        if (colCount_ == 0U) {
            return;
        }
        const event_t eidM2M3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE3));
        const event_t eidM3M2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        for (uint32_t c = 0U; c < colCount_; ++c) {
            CopyColumn(startCol_ + c, eidM2M3, eidM3M2);
        }
    }

private:
    __aicore__ inline uint32_t CalcColLen(uint32_t col) const
    {
        return (tiling_.uplo == 0U) ? (tiling_.n - col) : (col + 1U);
    }

    __aicore__ inline uint64_t CalcSrcOffset(uint32_t col) const
    {
        const uint64_t uCol = static_cast<uint64_t>(col);
        const uint64_t uLda = static_cast<uint64_t>(tiling_.lda);
        return (tiling_.uplo == 0U) ? (uCol * uLda + uCol) : (uCol * uLda);
    }

    __aicore__ inline uint64_t CalcDstOffset(uint32_t col) const
    {
        const uint64_t uCol = static_cast<uint64_t>(col);
        if (tiling_.uplo == 0U) {
            const uint64_t uN = static_cast<uint64_t>(tiling_.n);
            return (uCol * (2ULL * uN - uCol + 1ULL)) / 2ULL;
        }
        return (uCol * (uCol + 1ULL)) / 2ULL;
    }

    __aicore__ inline void CopyColumn(uint32_t col, event_t eidM2M3, event_t eidM3M2)
    {
        const uint32_t colLen = CalcColLen(col);
        const uint64_t srcOff = CalcSrcOffset(col);
        const uint64_t dstOff = CalcDstOffset(col);

        uint32_t processed = 0U;
        while (processed < colLen) {
            uint32_t chunkSize = colLen - processed;
            if (chunkSize > STRTTP_MAX_CHUNK_SIZE) {
                chunkSize = STRTTP_MAX_CHUNK_SIZE;
            }
            const uint32_t remainder = chunkSize % STRTTP_ELEMENTS_PER_BLOCK;
            const uint8_t padNum =
                (remainder != 0U) ? static_cast<uint8_t>(STRTTP_ELEMENTS_PER_BLOCK - remainder) : 0U;
            const uint32_t blockBytes = chunkSize * static_cast<uint32_t>(sizeof(float));
            DataCopyExtParams cp{1U, blockBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pp{true, 0U, padNum, 0.0F};

            LocalTensor<float> ub = copyBuf_.Get<float>();
            DataCopyPad(ub, aGM_[srcOff + processed], cp, pp);
            SetFlag<HardEvent::MTE2_MTE3>(eidM2M3);
            WaitFlag<HardEvent::MTE2_MTE3>(eidM2M3);

            DataCopyPad(apGM_[dstOff + processed], ub, cp);
            SetFlag<HardEvent::MTE3_MTE2>(eidM3M2);
            WaitFlag<HardEvent::MTE3_MTE2>(eidM3M2);

            processed += chunkSize;
        }
    }

    TPipe* pipe_ = nullptr;
    StrttpTilingData tiling_{};
    GlobalTensor<float> aGM_;
    GlobalTensor<float> apGM_;
    TBuf<TPosition::VECIN> copyBuf_;
    uint32_t startCol_ = 0U;
    uint32_t colCount_ = 0U;
};

} // namespace

extern "C" __global__ __aicore__ void strttp_kernel_impl(GM_ADDR aFull, GM_ADDR aPacked, StrttpTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    StrttpKernel op;
    op.Init(aFull, aPacked, tiling, &pipe);
    op.Process();
}

void strttp_kernel_do(uint8_t* a, uint8_t* ap, const StrttpTilingData& tiling, uint32_t numBlocks, void* stream)
{
    strttp_kernel_impl<<<numBlocks, nullptr, stream>>>(a, ap, tiling);
}
