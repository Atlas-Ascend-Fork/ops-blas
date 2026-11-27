/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ctrttp_kernel.h"

#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t CTRTTP_BUFFER_NUM = CTRTTP_QUEUE_DEPTH;
constexpr uint32_t CTRTTP_LANES_PER_BLOCK = 32U / sizeof(uint32_t);

/**
 * @brief Packs the uplo triangle of a column-major lda x n complex64 matrix.
 *
 * Column j owns two contiguous runs: A[j*lda + j, n) -> AP[...] for LOWER and
 * A[j*lda, j + 1) -> AP[...] for UPPER, so one column is one contiguous DataCopyPad pair.
 * Blocks take columns round-robin (j = blockIdx, blockIdx + useCoreNum, ...): every block
 * then handles the same number of columns and, because the lengths of those columns are
 * spread uniformly over 1..n, nearly the same byte count as well. Balancing on columns
 * instead of bytes also equalizes the number of DMA bursts per block, which matters because
 * a packed prefix made only of short columns would otherwise spend its time in DMA setup.
 *
 * Bursts are issued into a CTRTTP_QUEUE_DEPTH-deep queue before being drained, so several
 * MTE2 reads stay in flight while earlier columns are already being written back.
 */
class CtrttpKernel {
public:
    __aicore__ inline CtrttpKernel() = default;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR packed, const CtrttpTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    struct PendingBurst {
        uint64_t dstLane;
        uint32_t bytes;
    };

    __aicore__ inline uint64_t ColumnStart(uint32_t col) const;
    __aicore__ inline uint32_t ColumnLength(uint32_t col) const;
    __aicore__ inline void IssueRun(uint64_t srcElem, uint64_t dstElem, uint32_t count);
    __aicore__ inline void IssueBurst(uint64_t srcLane, uint64_t dstLane, uint32_t elems);
    __aicore__ inline void DrainBurst();
    __aicore__ inline void Flush();

    TPipe* pipe_ = nullptr;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, CTRTTP_BUFFER_NUM> copyQue_;
    GlobalTensor<uint32_t> aGm_;
    GlobalTensor<uint32_t> apGm_;
    CtrttpTilingData tiling_{};
    PendingBurst pending_[CTRTTP_BUFFER_NUM]{};
    uint32_t pendingHead_ = 0;
    uint32_t pendingCount_ = 0;
};

__aicore__ inline void CtrttpKernel::Init(GM_ADDR a, GM_ADDR packed, const CtrttpTilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    tiling_ = tiling;

    const uint64_t aLanes = static_cast<uint64_t>(tiling_.lda) * tiling_.n * CTRTTP_LANES_PER_COMPLEX;
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a), aLanes);
    apGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(packed), tiling_.totalPacked * CTRTTP_LANES_PER_COMPLEX);
    pipe_->InitBuffer(copyQue_, CTRTTP_BUFFER_NUM, tiling_.tileElems * CTRTTP_BYTES_PER_COMPLEX);
}

__aicore__ inline uint64_t CtrttpKernel::ColumnStart(uint32_t col) const
{
    if (tiling_.uplo == CTRTTP_UPLO_UPPER) {
        return static_cast<uint64_t>(col) * (col + 1ULL) / 2ULL;
    }
    return static_cast<uint64_t>(col) * (2ULL * tiling_.n - col + 1ULL) / 2ULL;
}

__aicore__ inline uint32_t CtrttpKernel::ColumnLength(uint32_t col) const
{
    return (tiling_.uplo == CTRTTP_UPLO_UPPER) ? (col + 1U) : (tiling_.n - col);
}

__aicore__ inline void CtrttpKernel::DrainBurst()
{
    LocalTensor<uint32_t> packed = copyQue_.DeQue<uint32_t>();
    const PendingBurst& burst = pending_[pendingHead_];
    const DataCopyExtParams copyOut{1U, burst.bytes, 0U, 0U, 0U};
    DataCopyPad(apGm_[burst.dstLane], packed, copyOut);
    copyQue_.FreeTensor(packed);
    pendingHead_ = (pendingHead_ + 1U) % CTRTTP_BUFFER_NUM;
    --pendingCount_;
}

__aicore__ inline void CtrttpKernel::IssueBurst(uint64_t srcLane, uint64_t dstLane, uint32_t elems)
{
    if (pendingCount_ == CTRTTP_BUFFER_NUM) {
        DrainBurst();
    }

    const uint32_t lanes = elems * CTRTTP_LANES_PER_COMPLEX;
    const uint32_t bytes = lanes * static_cast<uint32_t>(sizeof(uint32_t));
    const uint32_t tailLanes = lanes % CTRTTP_LANES_PER_BLOCK;
    const uint8_t rightPadding = static_cast<uint8_t>((CTRTTP_LANES_PER_BLOCK - tailLanes) % CTRTTP_LANES_PER_BLOCK);
    const DataCopyExtParams copyIn{1U, bytes, 0U, 0U, 0U};
    const DataCopyPadExtParams<uint32_t> padIn{true, 0U, rightPadding, 0U};

    LocalTensor<uint32_t> tile = copyQue_.AllocTensor<uint32_t>();
    DataCopyPad(tile, aGm_[srcLane], copyIn, padIn);
    copyQue_.EnQue<uint32_t>(tile);

    pending_[(pendingHead_ + pendingCount_) % CTRTTP_BUFFER_NUM] = PendingBurst{dstLane, bytes};
    ++pendingCount_;
}

__aicore__ inline void CtrttpKernel::IssueRun(uint64_t srcElem, uint64_t dstElem, uint32_t count)
{
    uint32_t done = 0;
    while (done < count) {
        uint32_t chunk = count - done;
        if (chunk > tiling_.tileElems) {
            chunk = tiling_.tileElems;
        }
        const uint64_t laneOffset = static_cast<uint64_t>(done) * CTRTTP_LANES_PER_COMPLEX;
        IssueBurst(srcElem * CTRTTP_LANES_PER_COMPLEX + laneOffset, dstElem * CTRTTP_LANES_PER_COMPLEX + laneOffset,
                   chunk);
        done += chunk;
    }
}

__aicore__ inline void CtrttpKernel::Flush()
{
    while (pendingCount_ > 0) {
        DrainBurst();
    }
}

__aicore__ inline void CtrttpKernel::Process()
{
    const uint32_t stride = tiling_.useCoreNum;
    for (uint32_t col = static_cast<uint32_t>(GetBlockIdx()); col < tiling_.n; col += stride) {
        // LOWER packs A(j..n-1, j): the first packed element of column j is row j.
        const uint64_t rowBias = (tiling_.uplo == CTRTTP_UPLO_LOWER) ? col : 0ULL;
        const uint64_t srcElem = static_cast<uint64_t>(col) * tiling_.lda + rowBias;
        IssueRun(srcElem, ColumnStart(col), ColumnLength(col));
    }
    Flush();
}

} // namespace

__global__ __aicore__ void ctrttp_kernel(GM_ADDR a, GM_ADDR packed, CtrttpTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CtrttpKernel op;
    op.Init(a, packed, tiling, &pipe);
    op.Process();
}

void ctrttp_kernel_do(uint8_t* A, uint8_t* AP, const CtrttpTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctrttp_kernel<<<numBlocks, nullptr, stream>>>(A, AP, tiling);
}
