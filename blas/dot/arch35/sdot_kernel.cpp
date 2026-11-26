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
#include "acl/acl.h"
#include "kernel_operator.h"
#include "common/helper/kernel_constant.h"
#include "sdot_tiling_data.h"

using namespace AscendC;

constexpr uint32_t ELEMENTS_PER_BLOCK = 8;
// ReduceSum 内部暂存空间，按最大归约量 chunk≤11616 代入 API 公式得下限 184，取 256 留余量。
constexpr uint32_t SHARED_TMP_ELEMENTS = 256;
// chunk 的 UB 分配倍数：xQueue(2×) + yQueue(2×) + product(1×)
constexpr uint32_t CHUNK_MULTIPLIER = 5;

// SAFETY_MARGIN 为 TPipe/TQue 运行时元数据预留（字节）。下界取决于 TPipe 内部实现，
// 偏小则 UB 溢出→运行时异常；偏大则 chunk 变小→性能下降。
// 当前取 20480 为保守值，对应 chunk=11608，与理论最大 11616 仅差 8 个 float。
//
// 给定 SAFETY_MARGIN，chunk 推导（UB_SIZE = 248×1024 = 253952，单位 float）：
//   availFloats = (UB_SIZE - SAFETY_MARGIN) / sizeof(float)
//   chunk ≤ (availFloats - 1(哨兵) - 256(SHARED_TMP) - 64(ACC)) / CHUNK_MULTIPLIER
// 代入 SAFETY_MARGIN=20480：availFloats=58368，chunk≤(58368-321)/5=11609，对齐8得 11608。
constexpr uint32_t SAFETY_MARGIN = 20480;
// MAX_CHUNK_SIZE = ceil8((availFloats - 320) / 5)：58048/5=11609.6，取下一个 8 倍数得 11616。
// 代码中整数除法实际产出 11608，此值大于实际产出，仅作兜底上限。
constexpr uint32_t MAX_CHUNK_SIZE = 11616;
// 累加器：实际只需 1 个 float 存部分和，取 64 对齐 vector 单元一次操作最大宽度（256B/4=64）。
constexpr uint32_t ACCUMULATOR_FLOATS = 64;

template <typename T>
class SdotKernel {
public:
    __aicore__ inline SdotKernel() {}
    __aicore__ inline void Init(
        TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, SdotTilingData tiling);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_, yQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<T> xGM_, yGM_, resultGM_, workspaceGM_;
    int64_t n_, incx_, incy_;
    uint32_t useCoreNum_;
    int64_t incxAbs_, incyAbs_;
    bool incxPos_, incyPos_;
    uint32_t coreIdx_, startIdx_, calCount_, chunkSize_, accOffset_;

    __aicore__ inline void SyncMTEToV();
    __aicore__ inline void SyncVToMTE();
    __aicore__ inline void ProcessContiguous();
    __aicore__ inline void ProcessStrided();
};

template <typename T>
__aicore__ inline void SdotKernel<T>::Init(
    TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, SdotTilingData tiling)
{
    pipe_ = pipe;
    n_ = tiling.n;
    incx_ = tiling.incx;
    incy_ = tiling.incy;
    useCoreNum_ = tiling.useCoreNum;
    coreIdx_ = GetBlockIdx();

    uint32_t baseCount = static_cast<uint32_t>(n_) / useCoreNum_;
    uint32_t remain = static_cast<uint32_t>(n_) % useCoreNum_;
    startIdx_ = coreIdx_ * baseCount + (coreIdx_ < remain ? coreIdx_ : remain);
    calCount_ = baseCount + (coreIdx_ < remain ? 1 : 0);

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    resultGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(result), 1);
    workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workSpace), useCoreNum_);

    incxAbs_ = (incx_ >= 0) ? incx_ : -incx_;
    incyAbs_ = (incy_ >= 0) ? incy_ : -incy_;
    incxPos_ = (incx_ >= 0);
    incyPos_ = (incy_ >= 0);

    uint32_t availFloats = (UB_SIZE - SAFETY_MARGIN) / sizeof(T);
    uint32_t fixedCost = 1 + SHARED_TMP_ELEMENTS + ACCUMULATOR_FLOATS;
    uint32_t rawChunk = (availFloats > fixedCost) ? (availFloats - fixedCost) / CHUNK_MULTIPLIER : 0;
    rawChunk = (rawChunk / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    chunkSize_ = (rawChunk > MAX_CHUNK_SIZE) ? MAX_CHUNK_SIZE : rawChunk;

    pipe_->InitBuffer(xQueue_, BUFFER_NUM, chunkSize_ * sizeof(T));
    pipe_->InitBuffer(yQueue_, BUFFER_NUM, chunkSize_ * sizeof(T));

    uint32_t alignedChunk = ((chunkSize_ + 1 + 7) / 8) * 8;
    accOffset_ = ((alignedChunk + SHARED_TMP_ELEMENTS + 7) / 8) * 8;
    pipe_->InitBuffer(tmpBuf_, (accOffset_ + ACCUMULATOR_FLOATS) * sizeof(T));
}

template <typename T>
__aicore__ inline void SdotKernel<T>::SyncMTEToV()
{
    int32_t eid = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eid);
    WaitFlag<HardEvent::MTE3_V>(eid);
}

template <typename T>
__aicore__ inline void SdotKernel<T>::SyncVToMTE()
{
    int32_t eid = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eid);
    WaitFlag<HardEvent::V_MTE3>(eid);
}

template <typename T>
__aicore__ inline void SdotKernel<T>::ProcessContiguous()
{
    uint32_t remaining = calCount_;
    uint32_t offset = startIdx_;
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};

    LocalTensor<T> product = tmpBuf_.Get<T>();
    LocalTensor<T> acc = product[accOffset_];
    // 使用 3 参数 Duplicate(dst, scalar, count) 填充 count 个元素；切勿使用 6 参数 mask 重载，
    // 后者的 mask 参数控制 block 数量（每个 block 8 个 float），容易与元素数量混淆导致溢出。
    Duplicate<T>(acc, static_cast<T>(0), ACCUMULATOR_FLOATS);
    PipeBarrier<PIPE_V>();

    while (remaining > 0) {
        uint32_t cur = (remaining < chunkSize_) ? remaining : chunkSize_;
        DataCopyExtParams cp{1, cur * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

        LocalTensor<T> xB = xQueue_.AllocTensor<T>();
        LocalTensor<T> yB = yQueue_.AllocTensor<T>();
        DataCopyPad(xB, xGM_[offset], cp, pp);
        DataCopyPad(yB, yGM_[offset], cp, pp);
        xQueue_.EnQue<T>(xB);
        yQueue_.EnQue<T>(yB);
        SyncMTEToV();

        LocalTensor<T> xC = xQueue_.DeQue<T>();
        LocalTensor<T> yC = yQueue_.DeQue<T>();

        LocalTensor<T> prd = tmpBuf_.Get<T>();
        uint32_t alignedCur = ((cur + 1 + 7) / 8) * 8;
        LocalTensor<T> sharedTmp = prd[alignedCur];

        Mul<T>(prd, xC, yC, static_cast<int32_t>(cur));
        // dst与src为同一tensor，inplace归约：对prd中cur个元素求和，结果存入prd[0]
        ReduceSum<T>(prd, prd, sharedTmp, static_cast<int32_t>(cur));
        PipeBarrier<PIPE_V>();

        Add<T>(acc, acc, prd, 1);
        PipeBarrier<PIPE_V>();

        xQueue_.FreeTensor(xC);
        yQueue_.FreeTensor(yC);
        remaining -= cur;
        offset += cur;
    }
    SyncVToMTE();
    DataCopyPad(workspaceGM_[coreIdx_], acc, {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
}

template <typename T>
__aicore__ inline void SdotKernel<T>::ProcessStrided()
{
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyExtParams cp1{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

    LocalTensor<T> acc = tmpBuf_.Get<T>()[accOffset_];
    Duplicate<T>(acc, static_cast<T>(0), ACCUMULATOR_FLOATS);
    PipeBarrier<PIPE_V>();

    uint32_t nv = static_cast<uint32_t>(n_);
    for (uint32_t i = 0; i < calCount_; i++) {
        int64_t idx = static_cast<int64_t>(startIdx_ + i);
        int64_t xo = incxPos_ ? idx * incxAbs_ : (static_cast<int64_t>(nv) - 1 - idx) * incxAbs_;
        int64_t yo = incyPos_ ? idx * incyAbs_ : (static_cast<int64_t>(nv) - 1 - idx) * incyAbs_;

        LocalTensor<T> xE = xQueue_.AllocTensor<T>();
        LocalTensor<T> yE = yQueue_.AllocTensor<T>();
        DataCopyPad(xE, xGM_[xo], cp1, pp);
        DataCopyPad(yE, yGM_[yo], cp1, pp);
        xQueue_.EnQue<T>(xE);
        yQueue_.EnQue<T>(yE);
        SyncMTEToV();
        xE = xQueue_.DeQue<T>();
        yE = yQueue_.DeQue<T>();
        Mul<T>(xE, xE, yE, 1);
        Add<T>(acc, acc, xE, 1);
        PipeBarrier<PIPE_V>();
        xQueue_.FreeTensor(xE);
        yQueue_.FreeTensor(yE);
    }
    SyncVToMTE();
    DataCopyPad(workspaceGM_[coreIdx_], acc, {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
}

template <typename T>
__aicore__ inline void SdotKernel<T>::Process()
{
    if (calCount_ == 0) {
        LocalTensor<T> z = tmpBuf_.Get<T>();
        Duplicate<T>(z, static_cast<T>(0), ACCUMULATOR_FLOATS);
        PipeBarrier<PIPE_V>();
        SyncVToMTE();
        DataCopyPad(workspaceGM_[coreIdx_], z, {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
    } else if (incxPos_ && incyPos_ && incxAbs_ == 1 && incyAbs_ == 1) {
        ProcessContiguous();
    } else {
        ProcessStrided();
    }

    // 各分支（ProcessContiguous/ProcessStrided/calCount_==0）均已在其写出 workspace 的
    // DataCopyPad 前完成 V→MTE 同步，此处汇合点无新的向量写，无需重复同步
    AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(0);
    AscendC::CrossCoreWaitFlag<0, PIPE_MTE3>(0);

    if (coreIdx_ == 0) {
        LocalTensor<T> ws = tmpBuf_.Get<T>();
        uint32_t cpLen = static_cast<uint32_t>(useCoreNum_ * sizeof(T));
        DataCopyPad(ws, workspaceGM_[0], {1, cpLen, 0, 0, 0}, {false, 0, 0, 0});
        SyncMTEToV();

        uint32_t wa = ((useCoreNum_ + 7) / 8) * 8;
        ReduceSum<T>(ws, ws, ws[wa], static_cast<int32_t>(useCoreNum_));
        PipeBarrier<PIPE_V>();

        SyncVToMTE();
        DataCopyPad(resultGM_[0], ws, {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
    }
}

__global__ __aicore__ void sdot_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, SdotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SdotKernel<float> op;
    op.Init(&pipe, x, y, result, workSpace, tiling);
    op.Process();
}

aclError sdot_kernel_do(
    GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, uint32_t numBlocks, const SdotTilingData& tiling,
    void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    sdot_kernel<<<numBlocks, nullptr, stream>>>(x, y, result, workSpace, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
