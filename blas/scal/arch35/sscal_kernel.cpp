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
#include <cstdlib>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "sscal_tiling_data.h"

using namespace AscendC;

class SscalAIV {
public:
    __aicore__ inline SscalAIV() {}
    __aicore__ inline void Init(GM_ADDR x, const SscalTilingData& tdata, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const SscalTilingData& tdata);
    __aicore__ inline void SingleIteration(uint32_t curOffset, uint32_t dataCount);

    TPipe* pipe_;
    GlobalTensor<float> xGM_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    SscalTilingData tiling_;
    uint32_t blockIdx_;
    uint32_t myOffset_;
    uint32_t myCount_;
};

__aicore__ inline void SscalAIV::ParseTilingData(const SscalTilingData& tdata)
{
    tiling_.totalN = tdata.totalN;
    tiling_.perCoreN = tdata.perCoreN;
    tiling_.remainder = tdata.remainder;
    tiling_.tileSize = tdata.tileSize;
    tiling_.alpha = tdata.alpha;
}

__aicore__ inline void SscalAIV::Init(GM_ADDR x, const SscalTilingData& tdata, TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    ParseTilingData(tdata);

    myOffset_ = blockIdx_ * tiling_.perCoreN;
    myCount_ = tiling_.perCoreN;
    if (blockIdx_ == GetBlockNum() - 1) {
        myCount_ += tiling_.remainder;
    }

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling_.totalN);

    uint32_t bufSize = tiling_.tileSize * sizeof(float);
    pipe_->InitBuffer(inQueue_, 1, bufSize);
    pipe_->InitBuffer(outQueue_, 1, bufSize);
}

__aicore__ inline void SscalAIV::SingleIteration(uint32_t curOffset, uint32_t dataCount)
{
    uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();

    if (alignedCount > 0) {
        DataCopy(inLocal, xGM_[curOffset], alignedCount);
    }

    if (tailCount > 0) {
        uint8_t paddingNum = ELEMENTS_PER_BLOCK - tailCount;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0.0f};
        DataCopyPad(inLocal[alignedCount], xGM_[curOffset + alignedCount], copyParams, padParams);
    }

    inQueue_.EnQue<float>(inLocal);

    LocalTensor<float> vData = inQueue_.DeQue<float>();
    LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

    Muls(outLocal, vData, tiling_.alpha, static_cast<int32_t>(dataCount));

    inQueue_.FreeTensor(vData);
    outQueue_.EnQue<float>(outLocal);

    LocalTensor<float> writeData = outQueue_.DeQue<float>();

    if (alignedCount > 0) {
        DataCopy(xGM_[curOffset], writeData, alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(xGM_[curOffset + alignedCount], writeData[alignedCount], copyParams);
    }

    outQueue_.FreeTensor(writeData);
}

__aicore__ inline void SscalAIV::Process()
{
    if (myCount_ == 0) {
        return;
    }

    uint32_t tileLoop = myCount_ / tiling_.tileSize;
    uint32_t tileTail = myCount_ % tiling_.tileSize;
    uint32_t curOffset = myOffset_;

    for (uint32_t i = 0; i < tileLoop; i++) {
        SingleIteration(curOffset, tiling_.tileSize);
        curOffset += tiling_.tileSize;
    }

    if (tileTail > 0) {
        SingleIteration(curOffset, tileTail);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SscalSimtCompute(
    uint32_t calNum, uint32_t startOffset, uint64_t stride, float alpha, __gm__ float* xGm)
{
    if (calNum == 0) {
        return;
    }

    for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
        // issue #806: 索引与 stride 统一 uint64 域（对齐同族 csscal 的 uint64 索引写法），
        // (n-1)*incx 超出 uint32 时 uint32 乘法回绕会读写错误地址（in-place 写，破坏性强）。
        const uint64_t idx = (static_cast<uint64_t>(startOffset) + i) * stride;
        xGm[idx] = alpha * xGm[idx];
    }
}

__global__ __aicore__ void sscal_aiv_kernel(GM_ADDR x, GM_ADDR workSpace, SscalTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SscalAIV op;
    op.Init(x, tdata, &pipe);
    op.Process();
}

__global__ __aicore__ void sscal_simt_kernel(GM_ADDR x, SscalTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockIdx = GetBlockIdx();
    uint32_t calNum = tdata.calCount[blockIdx];

    if (calNum > 0) {
        // issue #806: tdata.incx 为 int64 域，经 uint32 窄化会在步长超出 uint32 时符号/量值
        // 双重失真；直接以 uint64 传入 SIMT 计算核。
        const uint64_t stride = static_cast<uint64_t>(tdata.incx);
        asc_vf_call<SscalSimtCompute>(
            dim3{tdata.nthreads, 1, 1}, calNum, tdata.startOffset[blockIdx], stride, tdata.alpha,
            reinterpret_cast<__gm__ float*>(x));
    }
}

void sscal_kernel_do(GM_ADDR x, GM_ADDR workSpace, const SscalTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);

    if (tiling.incx == 1) {
        sscal_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(x, nullptr, tiling);
    } else {
        sscal_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, tiling);
    }
}