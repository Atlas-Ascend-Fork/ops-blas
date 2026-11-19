/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "saxpy_kernel.h"
#include "saxpy_tiling_data.h"

using namespace AscendC;

namespace {

class SaxpyAIV {
public:
    __aicore__ inline SaxpyAIV() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SaxpyTilingData& tiling, TPipe* pipeIn)
    {
        pipe_ = pipeIn;
        totalN_ = tiling.totalN;
        alpha_ = tiling.alpha;
        tileElems_ = (tiling.tileSize == 0U || tiling.tileSize > SAXPY_TILE_ELEMS) ? SAXPY_TILE_ELEMS : tiling.tileSize;
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(totalN_));
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint64_t>(totalN_));

        const uint32_t blockIdx = GetBlockIdx();
        start_ = tiling.perBlockN * blockIdx;
        count_ = tiling.perBlockN;
        if (blockIdx + 1U == tiling.activeBlocks) {
            count_ += tiling.remainder;
        }

        pipe_->InitBuffer(xQueue_, SAXPY_BUFFER_NUM, SAXPY_TILE_BYTES);
        pipe_->InitBuffer(yQueue_, SAXPY_BUFFER_NUM, SAXPY_TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        uint32_t done = 0U;
        while (done < count_) {
            const uint32_t todo = (count_ - done > tileElems_) ? tileElems_ : (count_ - done);
            const uint32_t aligned = todo - (todo % SAXPY_ELEMS_PER_BLOCK);
            const uint32_t tail = todo - aligned;
            const uint32_t gmOffset = start_ + done;

            LocalTensor<float> xLocal = xQueue_.AllocTensor<float>();
            LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();

            if (aligned > 0U) {
                DataCopy(xLocal, xGM_[gmOffset], aligned);
                DataCopy(yLocal, yGM_[gmOffset], aligned);
            }
            if (tail > 0U) {
                DataCopyExtParams copyParams{1U, static_cast<uint32_t>(tail * SAXPY_ELEM_BYTES), 0U, 0U, 0U};
                DataCopyPadExtParams<float> padParams{true, static_cast<uint8_t>(0U), static_cast<uint8_t>(0U), 0.0F};
                DataCopyPad(xLocal[aligned], xGM_[gmOffset + aligned], copyParams, padParams);
                DataCopyPad(yLocal[aligned], yGM_[gmOffset + aligned], copyParams, padParams);
            }

            xQueue_.EnQue<float>(xLocal);
            yQueue_.EnQue<float>(yLocal);
            xLocal = xQueue_.DeQue<float>();
            yLocal = yQueue_.DeQue<float>();

            Axpy(yLocal, xLocal, alpha_, static_cast<int32_t>(todo));

            const int32_t eventVToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
            if (aligned > 0U) {
                DataCopy(yGM_[gmOffset], yLocal, aligned);
            }
            if (tail > 0U) {
                DataCopyExtParams copyParams{1U, static_cast<uint32_t>(tail * SAXPY_ELEM_BYTES), 0U, 0U, 0U};
                DataCopyPad(yGM_[gmOffset + aligned], yLocal[aligned], copyParams);
            }

            xQueue_.FreeTensor(xLocal);
            yQueue_.FreeTensor(yLocal);
            const int32_t eventMte3ToMte2 =
                static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventMte3ToMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventMte3ToMte2);

            done += todo;
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, SAXPY_BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, SAXPY_BUFFER_NUM> yQueue_;
    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    uint32_t totalN_ = 0U;
    uint32_t start_ = 0U;
    uint32_t count_ = 0U;
    uint32_t tileElems_ = SAXPY_TILE_ELEMS;
    float alpha_ = 0.0F;
};

constexpr uint32_t SAXPY_STRIDED_BLOCK_FLOATS = 8U;
constexpr uint32_t SAXPY_STRIDED_TILE_ELEMS = 2048U;
constexpr uint32_t SAXPY_STRIDED_UB_X_OFFSET = 0U;
constexpr uint32_t SAXPY_STRIDED_UB_Y_OFFSET = SAXPY_STRIDED_TILE_ELEMS * 32U;
constexpr uint32_t SAXPY_STRIDED_UB_BYTES = 2U * SAXPY_STRIDED_TILE_ELEMS * 32U;
constexpr uint32_t SAXPY_STRIDED_REPEAT_ELEMS = 8U;
constexpr uint32_t SAXPY_STRIDED_MAX_REPEAT = 255U;
constexpr uint32_t SAXPY_STRIDED_COMPUTE_CHUNK = SAXPY_STRIDED_REPEAT_ELEMS * SAXPY_STRIDED_MAX_REPEAT;
constexpr uint32_t SAXPY_STRIDED_MAX_BURST = 2048U;
constexpr uint32_t SAXPY_STRIDED_SPAN_ELEMS = SAXPY_STRIDED_TILE_ELEMS * 8U;
static_assert(SAXPY_STRIDED_UB_BYTES <= SAXPY_UB_BYTES, "saxpy strided buffers must fit into UB");

class SaxpyStridedAIV {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SaxpyTilingData& tiling)
    {
        xGm_ = reinterpret_cast<__gm__ float*>(x);
        yGm_ = reinterpret_cast<__gm__ float*>(y);
        ubX_ = reinterpret_cast<__ubuf__ float*>((uintptr_t)SAXPY_STRIDED_UB_X_OFFSET);
        ubY_ = reinterpret_cast<__ubuf__ float*>((uintptr_t)SAXPY_STRIDED_UB_Y_OFFSET);
        totalN_ = static_cast<int64_t>(tiling.totalN);
        incx_ = tiling.incx;
        incy_ = tiling.incy;
        alpha_ = tiling.alpha;

        const uint32_t blockIdx = GetBlockIdx();
        start_ = tiling.perBlockN * blockIdx + ((blockIdx < tiling.remainder) ? blockIdx : tiling.remainder);
        count_ = tiling.perBlockN + ((blockIdx < tiling.remainder) ? 1U : 0U);

        absIncx_ = static_cast<uint32_t>(AbsIncrement(incx_));
        absIncy_ = static_cast<uint32_t>(AbsIncrement(incy_));
        vectorPath_ = ((incx_ > 0) && (incy_ > 0)) || ((incx_ < 0) && (incy_ < 0));
    }

    __aicore__ inline void Process()
    {
        if (vectorPath_) {
            ProcessBlocked();
        } else {
            ProcessSpan();
        }
    }

private:
    __aicore__ inline void ProcessBlocked()
    {
        uint32_t done = 0U;
        while (done < count_) {
            const uint32_t todo =
                (count_ - done > SAXPY_STRIDED_TILE_ELEMS) ? SAXPY_STRIDED_TILE_ELEMS : (count_ - done);
            for (uint32_t begin = 0U; begin < todo; begin += SAXPY_STRIDED_MAX_BURST) {
                const uint32_t batch =
                    (todo - begin > SAXPY_STRIDED_MAX_BURST) ? SAXPY_STRIDED_MAX_BURST : (todo - begin);
                const uint32_t logical = BlockedLogicalIndex(start_ + done, todo, begin, incx_);
                const uint64_t xIndex = SpanOffset(incx_, absIncx_, logical, totalN_);
                const uint64_t yIndex = SpanOffset(incy_, absIncy_, logical, totalN_);
                asc_copy_gm2ub_align(
                    ubX_ + begin * SAXPY_STRIDED_BLOCK_FLOATS, xGm_ + xIndex, static_cast<uint16_t>(batch),
                    SAXPY_ELEM_BYTES, 0U, 0U, (absIncx_ - 1U) * SAXPY_ELEM_BYTES, 0U);
                asc_copy_gm2ub_align(
                    ubY_ + begin * SAXPY_STRIDED_BLOCK_FLOATS, yGm_ + yIndex, static_cast<uint16_t>(batch),
                    SAXPY_ELEM_BYTES, 0U, 0U, (absIncy_ - 1U) * SAXPY_ELEM_BYTES, 0U);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            for (uint32_t begin = 0U; begin < todo; begin += SAXPY_STRIDED_COMPUTE_CHUNK) {
                const uint32_t count = (todo - begin > SAXPY_STRIDED_COMPUTE_CHUNK) ? SAXPY_STRIDED_COMPUTE_CHUNK
                                                                                   : (todo - begin);
                const uint32_t repeat = (count + SAXPY_STRIDED_REPEAT_ELEMS - 1U) / SAXPY_STRIDED_REPEAT_ELEMS;
                const uint32_t offset = begin * SAXPY_STRIDED_BLOCK_FLOATS;
                vaxpy(ubY_ + offset, ubX_ + offset, alpha_, static_cast<uint8_t>(repeat), 1, 1, 8, 8);
            }
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            for (uint32_t begin = 0U; begin < todo; begin += SAXPY_STRIDED_MAX_BURST) {
                const uint32_t batch =
                    (todo - begin > SAXPY_STRIDED_MAX_BURST) ? SAXPY_STRIDED_MAX_BURST : (todo - begin);
                const uint32_t logical = BlockedLogicalIndex(start_ + done, todo, begin, incy_);
                const uint64_t yIndex = SpanOffset(incy_, absIncy_, logical, totalN_);
                asc_copy_ub2gm_align(
                    yGm_ + yIndex, ubY_ + begin * SAXPY_STRIDED_BLOCK_FLOATS, static_cast<uint16_t>(batch),
                    SAXPY_ELEM_BYTES, 0U, 0U, 0U, (absIncy_ - 1U) * SAXPY_ELEM_BYTES);
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

            done += todo;
        }
    }

    __aicore__ inline void ProcessSpan()
    {
        const uint32_t maxInc = (absIncx_ > absIncy_) ? absIncx_ : absIncy_;
        const uint32_t tileCount = (SAXPY_STRIDED_SPAN_ELEMS - 1U) / maxInc + 1U;
        uint32_t done = 0U;
        while (done < count_) {
            const uint32_t todo = (count_ - done > tileCount) ? tileCount : (count_ - done);
            const uint32_t logicalStart = start_ + done;
            const uint32_t spanStartX = SpanStart(incx_, absIncx_, logicalStart, todo, totalN_);
            const uint32_t spanStartY = SpanStart(incy_, absIncy_, logicalStart, todo, totalN_);
            const uint32_t spanX = (todo - 1U) * absIncx_ + 1U;
            const uint32_t spanY = (todo - 1U) * absIncy_ + 1U;

            asc_copy_gm2ub_align(ubX_, xGm_ + spanStartX, 1U, spanX * SAXPY_ELEM_BYTES, 0U, 0U, 0U, 0U);
            asc_copy_gm2ub_align(ubY_, yGm_ + spanStartY, 1U, spanY * SAXPY_ELEM_BYTES, 0U, 0U, 0U, 0U);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

            for (uint32_t i = 0U; i < todo; ++i) {
                ubY_[SpanIndex(i, todo, absIncy_, incy_)] =
                    alpha_ * ubX_[SpanIndex(i, todo, absIncx_, incx_)] + ubY_[SpanIndex(i, todo, absIncy_, incy_)];
            }

            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            asc_copy_ub2gm_align(yGm_ + spanStartY, ubY_, 1U, spanY * SAXPY_ELEM_BYTES, 0U, 0U, 0U, 0U);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

            done += todo;
        }
    }

    __aicore__ static int64_t AbsIncrement(int64_t increment) { return (increment > 0) ? increment : -increment; }

    __aicore__ static uint64_t SpanOffset(int64_t inc, uint32_t absInc, uint32_t logicalIndex, int64_t totalN)
    {
        if (inc > 0) {
            return static_cast<uint64_t>(logicalIndex) * absInc;
        }
        return static_cast<uint64_t>(totalN - 1 - static_cast<int64_t>(logicalIndex)) * absInc;
    }

    __aicore__ static uint32_t SpanStart(
        int64_t inc, uint32_t absInc, uint32_t logicalStart, uint32_t todo, int64_t totalN)
    {
        if (inc > 0) {
            return static_cast<uint32_t>(static_cast<uint64_t>(logicalStart) * absInc);
        }
        const int64_t tail = totalN - static_cast<int64_t>(logicalStart) - static_cast<int64_t>(todo);
        return static_cast<uint32_t>(static_cast<uint64_t>(tail) * absInc);
    }

    __aicore__ static uint32_t SpanIndex(uint32_t i, uint32_t todo, uint32_t absInc, int64_t inc)
    {
        return (inc > 0) ? (i * absInc) : ((todo - 1U - i) * absInc);
    }

    __aicore__ static uint32_t BlockedLogicalIndex(uint32_t logicalStart, uint32_t todo, uint32_t begin, int64_t inc)
    {
        return (inc > 0) ? (logicalStart + begin) : (logicalStart + todo - 1U - begin);
    }

    __gm__ float* xGm_ = nullptr;
    __gm__ float* yGm_ = nullptr;
    __ubuf__ float* ubX_ = nullptr;
    __ubuf__ float* ubY_ = nullptr;
    int64_t totalN_ = 0;
    int64_t incx_ = 1;
    int64_t incy_ = 1;
    uint32_t absIncx_ = 1U;
    uint32_t absIncy_ = 1U;
    uint32_t start_ = 0U;
    uint32_t count_ = 0U;
    bool vectorPath_ = true;
    float alpha_ = 0.0F;
};

} // namespace

__global__ __aicore__ __vector__ void saxpy_kernel(GM_ADDR x, GM_ADDR y, SaxpyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (GetBlockIdx() >= tiling.activeBlocks) {
        return;
    }

    if (tiling.mode == 0U) {
        TPipe pipe;
        SaxpyAIV op;
        op.Init(x, y, tiling, &pipe);
        op.Process();
    } else {
        SaxpyStridedAIV op;
        op.Init(x, y, tiling);
        op.Process();
    }
}

void saxpy_kernel_do(uint8_t* x, uint8_t* y, const SaxpyTilingData& tiling, uint32_t numBlocks, void* stream)
{
    saxpy_kernel<<<numBlocks, nullptr, stream>>>((GM_ADDR)(uintptr_t)x, (GM_ADDR)(uintptr_t)y, tiling);
}
