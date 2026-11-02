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
 * \file dotex_kernel.cpp
 * \brief Extended dot product: result = sum(x[i*incx] * y[i*incy])
 *        incx==incy → SIMD,  incx!=incy → SIMT grid-stride.
 *        Input: half / float, compute in FP32, result matches input.
 *        Arch35 (ascend950) kernel-side implementation.
 */

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "dotex_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;

__aicore__ inline void SyncMteToV(TPipe& p)
{
    int32_t eid = static_cast<int32_t>(p.FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eid);
    WaitFlag<HardEvent::MTE3_V>(eid);
}
__aicore__ inline void SyncVToMte(TPipe& p)
{
    int32_t eid = static_cast<int32_t>(p.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eid);
    WaitFlag<HardEvent::V_MTE3>(eid);
}

// ==========================================================================
// SIMT VF path — grid-stride for incx != incy
// ==========================================================================
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void DotexSimt(
    uint32_t n, int32_t incx, int32_t incy, int64_t kx, int64_t ky, __gm__ float* wsGm, __gm__ T* xGm, __gm__ T* yGm,
    __gm__ float* finalWsGm, uint32_t scratchBase)
{
    float acc = 0.0f;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        int64_t xi = kx + static_cast<int64_t>(idx) * incx;
        int64_t yi = ky + static_cast<int64_t>(idx) * incy;
        if constexpr (std::is_same_v<T, float>) {
            acc += xGm[xi] * yGm[yi];
        } else {
            acc += static_cast<float>(xGm[xi]) * static_cast<float>(yGm[yi]);
        }
    }
    // workspace 两区分离：final 区 [0, useCoreNum) 存 per-core 部分和（final
    // kernel 读取），scratch 区 [useCoreNum, useCoreNum*(1+numThreads)) 存
    // per-thread 中间值。两区必须隔离——若重叠，blockDim>1 时 block0 的
    // thread t 写 wsGm[t] 与 block t 的 thread0 写 finalWsGm[t] 是同一地址，
    // 跨核 WAR 竞争会把 per-thread partial 混入最终求和（结果偏大）。
    wsGm[scratchBase + blockIdx.x * blockDim.x + threadIdx.x] = acc;
    asc_syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < blockDim.x; ++t) {
            sum += wsGm[scratchBase + blockIdx.x * blockDim.x + t];
        }
        finalWsGm[blockIdx.x] = sum;
        // 保证本核 GM store 对后续 final kernel（kernel 边界）可见。
        asc_threadfence();
    }
}

// ==========================================================================
// SIMD path — for incx == incy
// ==========================================================================
template <typename T>
class DotexKernel {
public:
    __aicore__ inline DotexKernel() {}
    __aicore__ inline void Init(
        TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR result, const DotexTilingData& tilingData, GM_ADDR workSpace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const DotexTilingData& tilingData);
    __aicore__ inline void CopyIn(int32_t paddedOffset, int32_t paddedCount);
    __aicore__ inline void Compute(int32_t paddedCount);
    __aicore__ inline void CopyOutPartial();
    __aicore__ inline void CopyInWorkspace(uint32_t count);
    __aicore__ inline void ComputeFinalReduce(uint32_t count);
    __aicore__ inline void CopyOutFinalResult();

    TPipe* pipe_;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;
    GlobalTensor<T> resultGM;
    GlobalTensor<float> workspaceGM;

    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> yQueue;

    TBuf<TPosition::VECCALC> xTmpBuf;
    TBuf<TPosition::VECCALC> yTmpBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> resTmpBuf;

    uint32_t blockIdx;
    uint32_t blockNum;

    int32_t startLogicalIdx;
    int32_t calCount;
    int32_t tileSize;
    int32_t tileNum;
    int32_t remainderCount;
};

template <typename T>
__aicore__ inline void DotexKernel<T>::ParseTilingData(const DotexTilingData& tilingData)
{
    blockIdx = GetBlockIdx();
    blockNum = GetBlockNum();

    int n = tilingData.n;
    int32_t useCore = static_cast<int32_t>(tilingData.useCoreNum);
    int32_t baseCount = n / useCore;
    int32_t remain = n % useCore;
    int32_t coreIdx = static_cast<int32_t>(blockIdx);

    if (coreIdx < remain) {
        calCount = baseCount + 1;
        startLogicalIdx = coreIdx * calCount;
    } else {
        calCount = baseCount;
        startLogicalIdx = coreIdx * calCount + remain;
    }
}

template <typename T>
__aicore__ inline void DotexKernel<T>::Init(
    TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR result, const DotexTilingData& tilingData, GM_ADDR workSpace)
{
    pipe_ = pipe;
    ParseTilingData(tilingData);

    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    resultGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(result), 1);
    workspaceGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), tilingData.useCoreNum);

    if (calCount == 0)
        return;

    constexpr int32_t kBlock = static_cast<int32_t>(32 / sizeof(float));
    constexpr int32_t kRepeat = static_cast<int32_t>(256 / sizeof(float));
    int32_t repeatTimes = (calCount + kRepeat - 1) / kRepeat;
    if (repeatTimes < 1)
        repeatTimes = 1;
    int32_t tmpDataCount = repeatTimes;
    int32_t reduceTmpElems = ((tmpDataCount + kBlock - 1) / kBlock) * kBlock;
    uint32_t reduceTmpBytes = static_cast<uint32_t>(reduceTmpElems) * sizeof(float);

    // 固定开销（不随 tile 增长）
    uint32_t fixedBytes = reduceTmpBytes  // ReduceSum 暂存空间
                          + sizeof(float) // 累加器 resTmp
                          + BLOCK_SIZE;   // TPipe 安全边界 32B
    uint32_t avail = (UB_SIZE > fixedBytes) ? (UB_SIZE - fixedBytes) : 0;

    if (avail <= 0)
        return;
    // 均分的份数：输入个数*doubleBuffer + x需要Cast（无论是否需要cast，这个都需要存mul的中间值） + y是否需要Cast
    // 权重 = xQueue + yQueue(各 BUFFER_NUM 份 T) + xTmpBuf(1 份 float) + yTmpBuf(FP16/BF16 才需要)
    uint32_t ubBufferWeight =
        BUFFER_NUM * 2 * sizeof(T) + sizeof(float) + (std::is_same_v<T, float> ? 0 : sizeof(float));

    uint32_t inputQueueUb =
        static_cast<uint32_t>(sizeof(T) / static_cast<float>(ubBufferWeight) * avail) / BLOCK_SIZE * BLOCK_SIZE;
    uint32_t castUb =
        static_cast<uint32_t>(sizeof(float) / static_cast<float>(ubBufferWeight) * avail) / BLOCK_SIZE * BLOCK_SIZE;

    tileSize = static_cast<int32_t>(inputQueueUb / sizeof(T));
    tileNum = calCount / tileSize;
    remainderCount = calCount % tileSize;

    pipe_->InitBuffer(xQueue, BUFFER_NUM, inputQueueUb);
    pipe_->InitBuffer(yQueue, BUFFER_NUM, inputQueueUb);
    pipe_->InitBuffer(xTmpBuf, castUb);
    if constexpr (!std::is_same_v<T, float>) { // FP32: 只需 1 个 float 占位
        pipe_->InitBuffer(yTmpBuf, castUb);    // FP16/BF16: Cast y→float
    }
    pipe_->InitBuffer(reduceTmpBuf, reduceTmpBytes);
    pipe_->InitBuffer(resTmpBuf, sizeof(float));
}

// ---------------------------------------------------------------------------
// CopyIn
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void DotexKernel<T>::CopyIn(int32_t paddedOffset, int32_t paddedCount)
{
    uint32_t nbytes = static_cast<uint32_t>(paddedCount * static_cast<int32_t>(sizeof(T)));
    DataCopyExtParams ext{1, nbytes, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();

    DataCopyPad(xLocal, xGM[paddedOffset], ext, padParams);
    xQueue.EnQue(xLocal);
    DataCopyPad(yLocal, yGM[paddedOffset], ext, padParams);
    yQueue.EnQue(yLocal);
}

// ---------------------------------------------------------------------------
// Compute：count 个元素的 Mul + ReduceSum → 累加到 resTmp
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void DotexKernel<T>::Compute(int32_t count)
{
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> yLocal = yQueue.DeQue<T>();

    LocalTensor<float> xTmp = xTmpBuf.Get<float>();
    LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
    LocalTensor<float> resTmp = resTmpBuf.Get<float>();

    if constexpr (std::is_same_v<T, float>) {
        Mul(xTmp, xLocal, yLocal, count);
    } else {
        LocalTensor<float> yTmp = yTmpBuf.Get<float>();
        Cast(xTmp, xLocal, RoundMode::CAST_NONE, count);
        Cast(yTmp, yLocal, RoundMode::CAST_NONE, count);
        Mul(xTmp, xTmp, yTmp, count);
    }

    ReduceSum<float>(xTmp, xTmp, reduceTmp, count);
    Add(resTmp, resTmp, xTmp, 1);

    xQueue.FreeTensor(xLocal);
    yQueue.FreeTensor(yLocal);
}

// ---------------------------------------------------------------------------
// CopyOutPartial
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void DotexKernel<T>::CopyOutPartial()
{
    DataCopyExtParams ext{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};

    LocalTensor<float> resTmp = resTmpBuf.Get<float>();
    DataCopyPad(workspaceGM[blockIdx], resTmp, ext);
}

// ---------------------------------------------------------------------------
// CopyInWorkspace (parameterised by element count)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void DotexKernel<T>::CopyInWorkspace(uint32_t count)
{
    uint32_t bytes = static_cast<uint32_t>(count * sizeof(float));

    LocalTensor<float> workSpaceBuf = xTmpBuf.Get<float>();
    DataCopyExtParams ext{1, bytes, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(workSpaceBuf, workspaceGM[0], ext, padParams);
}

// ---------------------------------------------------------------------------
// ComputeFinalReduce (parameterised by element count)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void DotexKernel<T>::ComputeFinalReduce(uint32_t count)
{
    LocalTensor<float> workSpaceBuf = xTmpBuf.Get<float>();
    LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
    ReduceSum<float>(workSpaceBuf, workSpaceBuf, reduceTmp, static_cast<int32_t>(count));
}

// ---------------------------------------------------------------------------
// CopyOutFinalResult
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void DotexKernel<T>::CopyOutFinalResult()
{
    LocalTensor<float> wsBuf = xTmpBuf.Get<float>();

    DataCopyExtParams ext{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(resultGM[0], wsBuf, ext);
    } else {
        LocalTensor<T> resT = yTmpBuf.Get<T>();
        Cast(resT, wsBuf, RoundMode::CAST_ROUND, 1);
        DataCopyPad(resultGM[0], resT, ext);
    }
}

// ---------------------------------------------------------------------------
// Process (Simd path)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void DotexKernel<T>::Process()
{
    if (calCount == 0)
        return;

    LocalTensor<float> resTmp = resTmpBuf.Get<float>();
    Duplicate<float>(resTmp, 0.0f, 1);

    for (int32_t i = 0; i < tileNum; ++i) {
        int32_t offset = startLogicalIdx + i * tileSize;
        CopyIn(offset, tileSize);
        Compute(tileSize);
    }

    if (remainderCount > 0) {
        int32_t offset = startLogicalIdx + tileNum * tileSize;
        CopyIn(offset, remainderCount);
        Compute(remainderCount);
    }

    SyncVToMte(*pipe_);
    CopyOutPartial();
}

// ==========================================================================
// SIMT dispatch helper
// ==========================================================================
template <typename T>
__aicore__ inline void LaunchDotexSimt(const DotexTilingData& t, GM_ADDR workspace, GM_ADDR x, GM_ADDR y)
{
    // workspace 布局（host 侧按 useCoreNum*(numThreads+1) floats 分配）：
    //   [0, useCoreNum)                     per-core 部分和（final kernel 读取）
    //   [useCoreNum, useCoreNum*(1+numThreads))  per-thread scratch
    asc_vf_call<DotexSimt<T>>(
        dim3{t.numThreads, 1, 1}, static_cast<uint32_t>(t.n), static_cast<int32_t>(t.incx),
        static_cast<int32_t>(t.incy), t.kx, t.ky, reinterpret_cast<__gm__ float*>(workspace),
        reinterpret_cast<__gm__ T*>(x), reinterpret_cast<__gm__ T*>(y), reinterpret_cast<__gm__ float*>(workspace),
        t.useCoreNum);
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void DotexSimtFinalWrite(
    uint32_t blockNum, __gm__ float* wsGm, __gm__ T* resultGm)
{
    if (blockIdx.x != 0) {
        return;
    }
    float sum = 0.0f;
    for (uint32_t b = 0; b < blockNum; ++b) {
        sum += wsGm[b];
    }
    volatile float ordered = sum;
    if constexpr (std::is_same_v<T, float>) {
        resultGm[0] = ordered;
    } else {
        resultGm[0] = static_cast<T>(ordered);
    }
}

// ==========================================================================
// SIMD entry (incx == incy, |inc| == 1)
// ==========================================================================
__aicore__ inline void DoDotexSimd(const DotexTilingData& t, GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workspace)
{
    TPipe pipe;
    if (t.srcType == DOTEX_XTYPE_FP32) {
        DotexKernel<float> op;
        op.Init(&pipe, x, y, result, t, workspace);
        op.Process();
    } else if (t.srcType == DOTEX_XTYPE_BF16) {
        DotexKernel<bfloat16_t> op;
        op.Init(&pipe, x, y, result, t, workspace);
        op.Process();
    } else {
        DotexKernel<half> op;
        op.Init(&pipe, x, y, result, t, workspace);
        op.Process();
    }
}

// ==========================================================================
// SIMT entry (incx != incy, or |inc| > 1)
// ==========================================================================
__aicore__ inline void DoDotexSimt(const DotexTilingData& t, GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workspace)
{
    if (t.srcType == DOTEX_XTYPE_FP32) {
        LaunchDotexSimt<float>(t, workspace, x, y);
    } else if (t.srcType == DOTEX_XTYPE_BF16) {
        LaunchDotexSimt<bfloat16_t>(t, workspace, x, y);
    } else {
        LaunchDotexSimt<half>(t, workspace, x, y);
    }
}

// ==========================================================================
// Final reduce entry — 单核读 workspace 求和并写 result；与 partial kernel
// 在同一 stream 上串行下发，跨核可见性由 kernel 边界保证。单 kernel 内
// SyncAll 方案已证实不可行：CopyOutPartial（MTE3 异步写）/DotexSimt（VF
// GM store）与 SyncAll 后 block0 的 MTE2/标量读竞争，产生 0/部分和/stale 值。
// ==========================================================================
template <typename T>
__global__ __aicore__ void dotex_final_reduce_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR result, const DotexTilingData t, GM_ADDR workspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)x;
    (void)y;
    // final kernel 固定单核下发，部分和个数取 tiling.useCoreNum（partial kernel
    // 的实际核数），不能用 GetBlockNum()（此处恒为 1）。求和走 VF 标量读
    // （与 SIMT final 完全同构）：TBuf + DataCopyPad 的 MTE2 读需要 MTE2_V
    // 事件同步，实测（MTE3_V 误配）会读到未落盘/残留数据。
    asc_vf_call<DotexSimtFinalWrite<T>>(
        dim3{1, 1, 1}, t.useCoreNum, reinterpret_cast<__gm__ float*>(workspace), reinterpret_cast<__gm__ T*>(result));
}

// ==========================================================================
// SIMT final write entry — SIMT 路径的 per-core 部分和已由 DotexSimt 写好
// （含 asc_threadfence），单核标量求和写出即可。
// ==========================================================================
// ==========================================================================
// Kernel entry — dual path router
// ==========================================================================
extern "C" __global__ __aicore__ void dotex_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR result, const DotexTilingData t, GM_ADDR workspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int absInc = t.incx > 0 ? t.incx : -t.incx;
    if (t.incx == t.incy && absInc == 1) {
        DoDotexSimd(t, x, y, result, workspace);
    } else {
        DoDotexSimt(t, x, y, result, workspace);
    }
}

void dotex_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* result, const DotexTilingData& tilingData, uint8_t* workSpace, uint32_t numBlocks,
    void* stream)
{
    // 两段式串行下发（同 stream，kernel 边界保证跨核可见性）：
    //   1) partial kernel：各核写 per-core 部分和到 workspace[0..blockNum)
    //   2) final kernel  ：单核读 workspace 归约写 result
    // 单 kernel 内 SyncAll + block0 归约的旧方案在 MTE3 异步写 / VF GM store
    // 场景下与 SyncAll 后的读取存在竞争，产生 0 / 部分和 / stale 值，已废弃。
    dotex_kernel<<<numBlocks, nullptr, stream>>>(x, y, result, tilingData, workSpace);

    // SIMD / SIMT 两路的 per-core 部分和布局一致（workspace[0..useCoreNum)），
    // final 归约统一走 dotex_final_reduce_kernel（VF 标量读求和）。
    if (tilingData.srcType == DOTEX_XTYPE_FP32) {
        dotex_final_reduce_kernel<float><<<1, nullptr, stream>>>(x, y, result, tilingData, workSpace);
    } else if (tilingData.srcType == DOTEX_XTYPE_BF16) {
        dotex_final_reduce_kernel<bfloat16_t><<<1, nullptr, stream>>>(x, y, result, tilingData, workSpace);
    } else {
        dotex_final_reduce_kernel<half><<<1, nullptr, stream>>>(x, y, result, tilingData, workSpace);
    }
}
