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
 * \file stpttr_kernel.cpp
 * \brief Device-side Ascend C kernel for aclblasStpttr (arch22 / dav-2201).
 */

#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "stpttr_tiling_data.h"
#include "stpttr_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t STPTTR_ELEMENTS_PER_BLOCK = 8U;
constexpr uint32_t STPTTR_MAX_CHUNK_SIZE = 28672U;

class StpttrKernel {
public:
    __aicore__ inline StpttrKernel() = default;

    __aicore__ inline void Init(GM_ADDR aPacked, GM_ADDR aFull, const StpttrTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        if (tiling_.useCoreNum == 0U || tiling_.n <= 0) {
            colCount_ = 0U;
            return;
        }
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= tiling_.useCoreNum) {
            colCount_ = 0U;
            return;
        }
        aPGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aPacked));
        aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aFull));
        pipe_->InitBuffer(copyBuf_, STPTTR_MAX_CHUNK_SIZE * sizeof(float));

        const uint32_t uN = static_cast<uint32_t>(tiling_.n);
        const uint32_t baseCols = uN / tiling_.useCoreNum;
        const uint32_t remainCols = uN % tiling_.useCoreNum;
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
        return (tiling_.uplo == 0) ? (static_cast<uint32_t>(tiling_.n) - col) : (col + 1U);
    }

    __aicore__ inline uint64_t CalcSrcOffset(uint32_t col) const
    {
        const uint64_t uCol = static_cast<uint64_t>(col);
        if (tiling_.uplo == 0) {
            const uint64_t uN = static_cast<uint64_t>(tiling_.n);
            return (uCol * (2ULL * uN - uCol + 1ULL)) / 2ULL;
        }
        return (uCol * (uCol + 1ULL)) / 2ULL;
    }

    __aicore__ inline uint64_t CalcDstOffset(uint32_t col) const
    {
        const uint64_t uCol = static_cast<uint64_t>(col);
        const uint64_t uLda = static_cast<uint64_t>(tiling_.lda);
        return (tiling_.uplo == 0) ? (uCol * uLda + uCol) : (uCol * uLda);
    }

    __aicore__ inline void CopyColumn(uint32_t col, event_t eidM2M3, event_t eidM3M2)
    {
        const uint32_t colLen = CalcColLen(col);
        const uint64_t srcOff = CalcSrcOffset(col);
        const uint64_t dstOff = CalcDstOffset(col);

        uint32_t processed = 0U;
        while (processed < colLen) {
            uint32_t chunkSize = colLen - processed;
            if (chunkSize > STPTTR_MAX_CHUNK_SIZE) {
                chunkSize = STPTTR_MAX_CHUNK_SIZE;
            }
            const uint32_t remainder = chunkSize % STPTTR_ELEMENTS_PER_BLOCK;
            const uint8_t padNum = (remainder != 0U) ? static_cast<uint8_t>(STPTTR_ELEMENTS_PER_BLOCK - remainder) : 0U;
            const uint32_t blockBytes = chunkSize * static_cast<uint32_t>(sizeof(float));
            DataCopyExtParams cp{1U, blockBytes, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pp{true, 0U, padNum, 0.0F};

            LocalTensor<float> ub = copyBuf_.Get<float>();
            DataCopyPad(ub, aPGM_[srcOff + processed], cp, pp);
            SetFlag<HardEvent::MTE2_MTE3>(eidM2M3);
            WaitFlag<HardEvent::MTE2_MTE3>(eidM2M3);

            DataCopyPad(aGM_[dstOff + processed], ub, cp);
            SetFlag<HardEvent::MTE3_MTE2>(eidM3M2);
            WaitFlag<HardEvent::MTE3_MTE2>(eidM3M2);

            processed += chunkSize;
        }
    }

    TPipe* pipe_ = nullptr;
    StpttrTilingData tiling_{};
    GlobalTensor<float> aPGM_;
    GlobalTensor<float> aGM_;
    TBuf<TPosition::VECIN> copyBuf_;
    uint32_t startCol_ = 0U;
    uint32_t colCount_ = 0U;
};

} // namespace

extern "C" __global__ __aicore__ void stpttr_kernel_impl(GM_ADDR aPacked, GM_ADDR aFull, StpttrTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    StpttrKernel op;
    op.Init(aPacked, aFull, tiling, &pipe);
    op.Process();
}

aclError stpttr_kernel_do(
    uint8_t* aPacked, uint8_t* aFull, const StpttrTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    stpttr_kernel_impl<<<numBlocks, nullptr, stream>>>(aPacked, aFull, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
