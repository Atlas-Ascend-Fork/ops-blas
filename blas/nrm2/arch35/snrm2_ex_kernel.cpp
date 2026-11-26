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
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "common/helper/kernel_constant.h"
#include "snrm2_ex_tiling_data.h"

using namespace AscendC;

namespace {

constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
// 所有 UB 分配和 MTE 搬运必须 32B 对齐。
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t ELEMENTS_PER_BLOCK = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32; // 8
// ReduceSum/ReduceMax Level-1 单元：256B repeat = 64 个 FP32 元素。
constexpr uint32_t REDUCE_REPEAT_BYTES = 256;
constexpr uint32_t ELEMENTS_PER_REPEAT = REDUCE_REPEAT_BYTES / BYTENUM_PER_FLOAT32; // 64

// ===========================================================================
// AIV 路径（incx == 1）：SIMD membase 两遍缩放归约。
//
// Pass 1：scale_local = 各 tile 的 max(|x|) 之最大值
// Pass 2：若 scale_local > 0，ssq_local = 各 tile 的 sum((x/scale)^2) 之和
//
// 每个 core 的 (scale_local, ssq_local) 写入 workspace[blockIdx*2 .. +1]；
// 阶段 2（core 0）用缩放合并公式跨核归约。
// ===========================================================================
template <typename T_in>
class Snrm2ExAIV {
public:
    __aicore__ inline Snrm2ExAIV() {}
    __aicore__ inline void Init(
        TPipe* pipe, GM_ADDR xGM, GM_ADDR wsGM, uint32_t blockIdx, uint32_t useCoreNum, const Snrm2ExTilingData& tdata);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t dataCount);
    __aicore__ inline float ComputeMax(uint32_t dataCount);
    __aicore__ inline float ComputeScaledSsq(uint32_t dataCount, float scale);
    __aicore__ inline void WriteWorkspace(float scaleLocal, float ssqLocal);
    __aicore__ float Pass1ComputeMax();
    __aicore__ float Pass2ComputeSsq(float scaleLocal);

    TPipe* pipe_ = nullptr;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TBuf<TPosition::VECCALC> absBuf_;     // |x|（FP32），供 ReduceMax 使用
    TBuf<TPosition::VECCALC> computeBuf_; // 仅 FP16：Cast FP16->FP32 + 缩放平方计算
    TBuf<TPosition::VECCALC> workBuf_;    // ReduceMax/ReduceSum 共享临时空间
    TBuf<TPosition::VECCALC> outBuf_;     // 归约标量输出（用于 GetValue）

    GlobalTensor<T_in> xGM_;
    GlobalTensor<float> wsGM_;

    uint32_t blockIdx_ = 0;
    uint32_t useCoreNum_ = 0;
    uint32_t calNum_ = 0;
    uint32_t startOffset_ = 0;
    uint32_t maxDataCount_ = 0;
};

template <typename T_in>
__aicore__ inline void Snrm2ExAIV<T_in>::Init(
    TPipe* pipe, GM_ADDR xGM, GM_ADDR wsGM, uint32_t blockIdx, uint32_t useCoreNum, const Snrm2ExTilingData& tdata)
{
    pipe_ = pipe;
    blockIdx_ = blockIdx;
    useCoreNum_ = useCoreNum;

    calNum_ = tdata.batchPerCore + (blockIdx_ < tdata.remain ? 1 : 0);
    startOffset_ = blockIdx_ * tdata.batchPerCore + (blockIdx_ < tdata.remain ? blockIdx_ : tdata.remain);
    maxDataCount_ = tdata.maxDataCount;

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T_in*>(xGM), static_cast<uint64_t>(tdata.n));
    wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsGM), static_cast<uint64_t>(useCoreNum_) * 2);

    // UB 布局（设计 1.3.A 2.3）。FP32：inQueue(8M) + absBuf(4M) = 12M。
    // FP16：inQueue(4M) + computeBuf(4M) + absBuf(4M) = 12M。两者在 M=17920 时均不超过预算。
    pipe_->InitBuffer(inQueue_, BUFFER_NUM, maxDataCount_ * sizeof(T_in));
    pipe_->InitBuffer(absBuf_, maxDataCount_ * sizeof(float));
    if constexpr (IsSameType<T_in, half>::value) {
        pipe_->InitBuffer(computeBuf_, maxDataCount_ * sizeof(float));
    }

    // workBuf：ceil(maxDataCount / ELEMENTS_PER_REPEAT) 对齐到 ELEMENTS_PER_BLOCK，按 FP32 字节数计，
    // 额外加一个 32B block 用于 MTE 对齐 padding。
    uint32_t level1Rep = (maxDataCount_ + ELEMENTS_PER_REPEAT - 1) / ELEMENTS_PER_REPEAT;
    uint32_t level1Align = ((level1Rep + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    pipe_->InitBuffer(workBuf_, level1Align * sizeof(float) + UB_BYTENUM_PER_BLOCK);
    // outBuf：ReduceMax/ReduceSum 产生标量；MTE 搬运至少需要 32B。
    pipe_->InitBuffer(outBuf_, UB_BYTENUM_PER_BLOCK);
}

template <typename T_in>
__aicore__ inline void Snrm2ExAIV<T_in>::CopyIn(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<T_in> inLocal = inQueue_.AllocTensor<T_in>();
    DataCopy(inLocal, xGM_[offset], dataCount);
    inQueue_.EnQue<T_in>(inLocal);
}

template <typename T_in>
__aicore__ inline void Snrm2ExAIV<T_in>::CopyInPad(uint32_t offset, uint32_t dataCount)
{
    constexpr uint32_t elemsPerBlock = UB_BYTENUM_PER_BLOCK / sizeof(T_in);
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataCount * sizeof(T_in)), 0, 0};
    uint8_t paddingNum = static_cast<uint8_t>(elemsPerBlock - dataCount % elemsPerBlock);
    DataCopyPadParams padParams{true, 0, paddingNum, 0};

    LocalTensor<T_in> inLocal = inQueue_.AllocTensor<T_in>();
    DataCopyPad(inLocal, xGM_[offset], copyParams, padParams);
    inQueue_.EnQue<T_in>(inLocal);
}

template <typename T_in>
__aicore__ inline float Snrm2ExAIV<T_in>::ComputeMax(uint32_t dataCount)
{
    LocalTensor<T_in> inLocal = inQueue_.DeQue<T_in>();
    LocalTensor<float> absLocal = absBuf_.Get<float>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> outLocal = outBuf_.Get<float>();

    if constexpr (IsSameType<T_in, half>::value) {
        LocalTensor<float> computeLocal = computeBuf_.Get<float>();
        Cast(computeLocal, inLocal, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Abs(absLocal, computeLocal, dataCount);
    } else {
        Abs(absLocal, inLocal, dataCount);
    }
    PipeBarrier<PIPE_V>();
    ReduceMax(outLocal, absLocal, workLocal, static_cast<int32_t>(dataCount), false);

    // V -> S 同步：在标量读取归约结果前。
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evt);
    WaitFlag<HardEvent::V_S>(evt);
    float tileMax = outLocal.GetValue(0);

    inQueue_.FreeTensor(inLocal);
    return tileMax;
}

template <typename T_in>
__aicore__ inline float Snrm2ExAIV<T_in>::ComputeScaledSsq(uint32_t dataCount, float scale)
{
    LocalTensor<T_in> inLocal = inQueue_.DeQue<T_in>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> outLocal = outBuf_.Get<float>();

    if constexpr (IsSameType<T_in, half>::value) {
        LocalTensor<float> computeLocal = computeBuf_.Get<float>();
        Cast(computeLocal, inLocal, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Divs(computeLocal, computeLocal, scale, dataCount);
        PipeBarrier<PIPE_V>();
        Mul(computeLocal, computeLocal, computeLocal, dataCount);
        PipeBarrier<PIPE_V>();
        ReduceSum(outLocal, computeLocal, workLocal, static_cast<int32_t>(dataCount));
    } else {
        // inLocal 是 FP32；原地复用为计算缓冲区。
        LocalTensor<float> computeLocal = inLocal.template ReinterpretCast<float>();
        Divs(computeLocal, computeLocal, scale, dataCount);
        PipeBarrier<PIPE_V>();
        Mul(computeLocal, computeLocal, computeLocal, dataCount);
        PipeBarrier<PIPE_V>();
        ReduceSum(outLocal, computeLocal, workLocal, static_cast<int32_t>(dataCount));
    }

    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evt);
    WaitFlag<HardEvent::V_S>(evt);
    float tileSsq = outLocal.GetValue(0);

    inQueue_.FreeTensor(inLocal);
    return tileSsq;
}

template <typename T_in>
__aicore__ inline void Snrm2ExAIV<T_in>::WriteWorkspace(float scaleLocal, float ssqLocal)
{
    LocalTensor<float> outLocal = outBuf_.Get<float>();
    outLocal.SetValue(0, scaleLocal);
    outLocal.SetValue(1, ssqLocal);

    // S -> MTE3 同步：在将标量拷出到 GM 前。
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evt);
    WaitFlag<HardEvent::S_MTE3>(evt);

    DataCopyParams copyParams{1, static_cast<uint16_t>(2 * sizeof(float)), 0, 0};
    DataCopyPad(wsGM_[blockIdx_ * 2], outLocal, copyParams);
}

template <typename T_in>
__aicore__ float Snrm2ExAIV<T_in>::Pass1ComputeMax()
{
    constexpr uint32_t maxCopyPadNum = (UINT16_MAX + 1) / sizeof(T_in);

    float scaleLocal = 0.0f;
    uint32_t repeatTimes = calNum_ / maxDataCount_;
    uint32_t remainNum = calNum_ % maxDataCount_;
    uint32_t currOffset = startOffset_;

    for (uint32_t i = 0; i < repeatTimes; i++) {
        CopyIn(currOffset, maxDataCount_);
        float tileMax = ComputeMax(maxDataCount_);
        if (tileMax > scaleLocal) {
            scaleLocal = tileMax;
        }
        currOffset += maxDataCount_;
    }
    if (remainNum > 0) {
        if (remainNum >= maxCopyPadNum) {
            CopyIn(currOffset, maxCopyPadNum);
            float tileMax = ComputeMax(maxCopyPadNum);
            if (tileMax > scaleLocal) {
                scaleLocal = tileMax;
            }
            currOffset += maxCopyPadNum;
            remainNum -= maxCopyPadNum;
        }
        if (remainNum > 0) {
            CopyInPad(currOffset, remainNum);
            uint32_t alignedCount = (remainNum + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
            float tileMax = ComputeMax(alignedCount);
            if (tileMax > scaleLocal) {
                scaleLocal = tileMax;
            }
        }
    }
    return scaleLocal;
}

template <typename T_in>
__aicore__ float Snrm2ExAIV<T_in>::Pass2ComputeSsq(float scaleLocal)
{
    constexpr uint32_t maxCopyPadNum = (UINT16_MAX + 1) / sizeof(T_in);

    float ssqLocal = 0.0f;
    if (scaleLocal <= 0.0f) {
        return ssqLocal;
    }

    uint32_t repTimes2 = calNum_ / maxDataCount_;
    uint32_t remNum2 = calNum_ % maxDataCount_;
    uint32_t currOffset = startOffset_;

    for (uint32_t i = 0; i < repTimes2; i++) {
        CopyIn(currOffset, maxDataCount_);
        ssqLocal += ComputeScaledSsq(maxDataCount_, scaleLocal);
        currOffset += maxDataCount_;
    }
    if (remNum2 > 0) {
        if (remNum2 >= maxCopyPadNum) {
            CopyIn(currOffset, maxCopyPadNum);
            ssqLocal += ComputeScaledSsq(maxCopyPadNum, scaleLocal);
            currOffset += maxCopyPadNum;
            remNum2 -= maxCopyPadNum;
        }
        if (remNum2 > 0) {
            CopyInPad(currOffset, remNum2);
            uint32_t alignedCount = (remNum2 + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
            ssqLocal += ComputeScaledSsq(alignedCount, scaleLocal);
        }
    }
    return ssqLocal;
}

template <typename T_in>
__aicore__ inline void Snrm2ExAIV<T_in>::Process()
{
    float scaleLocal = Pass1ComputeMax();
    float ssqLocal = Pass2ComputeSsq(scaleLocal);
    WriteWorkspace(scaleLocal, ssqLocal);
}

// ===========================================================================
// SIMT 路径（incx != 1）：GM 直接访问 + 线程级增量缩放范数。
// 每个线程流式处理其跨步元素，通过 LAPACK DLASSQ 更新维护 (scale, ssq)；
// thread 0 合并所有线程的 (scale, ssq) 对，并将该 core 的
// (scale_local, ssq_local) 写入 workspace[blkIdx*2 .. +1]。
// ===========================================================================
__simt_callee__ inline void Snrm2ExSimtMerge(
    const __ubuf__ float* sBuf, const __ubuf__ float* qBuf, uint32_t blockDimX, __gm__ float* wsOut)
{
    float gScale = 0.0f;
    float gSsq = 0.0f;
    for (uint32_t i = 0; i < blockDimX; i++) {
        float s = sBuf[i];
        float q = qBuf[i];
        if (s == 0.0f) {
            continue;
        }
        if (gScale == 0.0f) {
            gScale = s;
            gSsq = q;
        } else if (gScale >= s) {
            float r = s / gScale;
            gSsq += q * r * r;
        } else {
            float r = gScale / s;
            gSsq = q + gSsq * r * r;
            gScale = s;
        }
    }
    wsOut[0] = gScale;
    wsOut[1] = gSsq;
}

template <typename T_in>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Snrm2ExSimt(
    uint32_t calNum, uint32_t elemStart, int64_t incx, uint32_t n, __gm__ const T_in* xGm, __gm__ float* wsOut)
{
    constexpr bool IN_FP16 = IsSameType<T_in, half>::value;
    int64_t absInc = (incx > 0) ? incx : -incx;

    float tScale = 0.0f;
    float tSsq = 0.0f;
    for (uint32_t t = threadIdx.x; t < calNum; t += blockDim.x) {
        uint32_t elemIdx = elemStart + t;
        uint64_t gmIdx = (incx > 0) ? static_cast<uint64_t>(elemIdx) * static_cast<uint64_t>(absInc) :
                                      static_cast<uint64_t>(n - 1 - elemIdx) * static_cast<uint64_t>(absInc);
        float xv;
        if constexpr (IN_FP16) {
            xv = __half2float(xGm[gmIdx]);
        } else {
            xv = static_cast<float>(xGm[gmIdx]);
        }
        float ax = (xv >= 0.0f) ? xv : -xv;
        // 增量缩放平方和（可结合，溢出安全）。
        if (ax != 0.0f) {
            if (tScale < ax) {
                float ratio = tScale / ax;
                tSsq = 1.0f + tSsq * ratio * ratio;
                tScale = ax;
            } else {
                float ratio = ax / tScale;
                tSsq = tSsq + ratio * ratio;
            }
        }
    }

    __ubuf__ float sBuf[SIMT_MAX_THREAD_NUM];
    __ubuf__ float qBuf[SIMT_MAX_THREAD_NUM];
    sBuf[threadIdx.x] = tScale;
    qBuf[threadIdx.x] = tSsq;
    asc_syncthreads();

    if (threadIdx.x == 0) {
        Snrm2ExSimtMerge(sBuf, qBuf, blockDim.x, wsOut);
    }
}

// ===========================================================================
// 阶段 2（core 0）：将各 core 的 (scale_i, ssq_i) 合并为最终结果。
//
//   global_scale = max(scale_i)
//   global_ssq   = sum_i (scale_i / global_scale)^2 * ssq_i
//   result       = global_scale * sqrt(global_ssq)
//
// useCoreNum <= 72 => 最多 144 个 FP32 值，属于小规模标量归约，
// 用 GetValue 即可完成（R1 允许小规模标量归约使用 GetValue）。
// 最终的 sqrt/mul 用向量算子完成，避免使用标量数学库。
// ===========================================================================
class Snrm2ExFinal {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR wsGM, GM_ADDR resultGM, uint32_t useCoreNum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void ComputeResult();
    __aicore__ inline void CopyOut();

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TBuf<TPosition::VECCALC> outBuf_;

    GlobalTensor<float> wsGM_;
    GlobalTensor<float> outGM_;

    uint32_t useCoreNum_ = 0;
    uint32_t count_ = 0;
    uint32_t paddedCount_ = 0;
    float globalScale_ = 0.0f;
    float globalSsq_ = 0.0f;
};

__aicore__ inline void Snrm2ExFinal::Init(TPipe* pipe, GM_ADDR wsGM, GM_ADDR resultGM, uint32_t useCoreNum)
{
    pipe_ = pipe;
    useCoreNum_ = useCoreNum;
    count_ = useCoreNum_ * 2;
    paddedCount_ = (count_ + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
    if (paddedCount_ < ELEMENTS_PER_BLOCK) {
        paddedCount_ = ELEMENTS_PER_BLOCK;
    }

    wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsGM), paddedCount_);
    outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(resultGM), 1);

    pipe_->InitBuffer(inQueue_, 1, paddedCount_ * sizeof(float));
    pipe_->InitBuffer(outBuf_, UB_BYTENUM_PER_BLOCK);
}

__aicore__ inline void Snrm2ExFinal::CopyIn()
{
    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
    if (count_ % ELEMENTS_PER_BLOCK != 0) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(count_ * sizeof(float)), 0, 0};
        uint8_t paddingNum = static_cast<uint8_t>(paddedCount_ - count_);
        DataCopyPadParams padParams{true, 0, paddingNum, 0};
        DataCopyPad(inLocal, wsGM_, copyParams, padParams);
    } else {
        DataCopy(inLocal, wsGM_, count_);
    }
    inQueue_.EnQue<float>(inLocal);
}

__aicore__ inline void Snrm2ExFinal::ComputeResult()
{
    LocalTensor<float> inLocal = inQueue_.DeQue<float>();

    // DeQue 同步 MTE2→V，GetValue 走 S pipe，需显式同步 MTE2→S。
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evt);
    WaitFlag<HardEvent::MTE2_S>(evt);

    float globalScale = 0.0f;
    for (uint32_t i = 0; i < useCoreNum_; i++) {
        float s = inLocal.GetValue(i * 2);
        if (s > globalScale) {
            globalScale = s;
        }
    }

    float globalSsq = 0.0f;
    if (globalScale > 0.0f) {
        for (uint32_t i = 0; i < useCoreNum_; i++) {
            float s = inLocal.GetValue(i * 2);
            float q = inLocal.GetValue(i * 2 + 1);
            float ratio = s / globalScale;
            globalSsq += q * ratio * ratio;
        }
    }

    inQueue_.FreeTensor(inLocal);
    globalScale_ = globalScale;
    globalSsq_ = globalSsq;
}

__aicore__ inline void Snrm2ExFinal::CopyOut()
{
    // result = globalScale_ * sqrt(globalSsq_)，用向量算子计算。
    // 仅 outLocal[0] 有意义；[1..7] 为垃圾数据但不拷出。
    LocalTensor<float> outLocal = outBuf_.Get<float>();
    outLocal.SetValue(0, globalSsq_);

    // S -> V：SetValue（S）写 UB，Sqrt（V）读 UB。
    event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evtSV);
    WaitFlag<HardEvent::S_V>(evtSV);

    Sqrt(outLocal, outLocal, static_cast<int32_t>(ELEMENTS_PER_BLOCK));
    Muls(outLocal, outLocal, globalScale_, static_cast<int32_t>(ELEMENTS_PER_BLOCK));

    // V -> MTE3：向量结果必须在 DataCopyPad 读 UB 前完成。
    event_t evtVM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evtVM);
    WaitFlag<HardEvent::V_MTE3>(evtVM);

    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
    DataCopyPad(outGM_, outLocal, copyParams);
}

__aicore__ inline void Snrm2ExFinal::Process()
{
    if (useCoreNum_ == 0) {
        globalScale_ = 0.0f;
        globalSsq_ = 0.0f;
        CopyOut();
        return;
    }
    CopyIn();
    ComputeResult();
    CopyOut();
}

} // namespace

__global__ __aicore__ void snrm2_ex_kernel(GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, Snrm2ExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t blkIdx = GetBlockIdx();
    uint32_t useCoreNum = tiling.useCoreNum;
    bool active = (blkIdx < useCoreNum);

    if (active) {
        uint32_t calNum = tiling.batchPerCore + (blkIdx < tiling.remain ? 1 : 0);
        uint32_t elemStart = blkIdx * tiling.batchPerCore + (blkIdx < tiling.remain ? blkIdx : tiling.remain);

        if (tiling.incx == 1) {
            // R3：TPipe 在 kernel 入口创建，不作为成员变量。
            TPipe pipe;
            if (tiling.xtype == 0u) {
                Snrm2ExAIV<float> op;
                op.Init(&pipe, x, workSpace, blkIdx, useCoreNum, tiling);
                op.Process();
            } else {
                Snrm2ExAIV<half> op;
                op.Init(&pipe, x, workSpace, blkIdx, useCoreNum, tiling);
                op.Process();
            }
        } else {
            __gm__ float* wsGm = reinterpret_cast<__gm__ float*>(workSpace);
            if (tiling.xtype == 0u) {
                asc_vf_call<Snrm2ExSimt<float>>(
                    dim3{tiling.nthreads, 1, 1}, calNum, elemStart, tiling.incx, static_cast<uint32_t>(tiling.n),
                    reinterpret_cast<__gm__ const float*>(x), wsGm + blkIdx * 2);
            } else {
                asc_vf_call<Snrm2ExSimt<half>>(
                    dim3{tiling.nthreads, 1, 1}, calNum, elemStart, tiling.incx, static_cast<uint32_t>(tiling.n),
                    reinterpret_cast<__gm__ const half*>(x), wsGm + blkIdx * 2);
            }
        }
    }

    // SyncAll：所有启动的 core（含非活跃 core）都必须到达。
    SyncAll();

    // 阶段 2：仅 core 0 合并并写出最终结果。
    if (blkIdx == 0) {
        TPipe pipe;
        Snrm2ExFinal op;
        op.Init(&pipe, workSpace, result, useCoreNum);
        op.Process();
    }
}

aclError snrm2_ex_kernel_do(
    GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, const Snrm2ExTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    snrm2_ex_kernel<<<numBlocks, nullptr, aclStream>>>(x, result, workSpace, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
