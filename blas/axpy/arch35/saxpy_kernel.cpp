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
#include "saxpy_tiling_data.h"

using namespace AscendC;

class SaxpyAIV {
public:
    __aicore__ inline SaxpyAIV() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint32_t totalN, uint32_t perCoreN, uint32_t remainder, uint32_t tileSize, float alpha,
        TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SingleIteration(uint32_t curOffset, uint32_t dataCount);

    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    TBuf<TPosition::VECIN> xBuf_;
    TBuf<TPosition::VECIN> yBuf_;
    float alpha_;
    uint32_t myOffset_;
    uint32_t myCount_;
    uint32_t tileSize_;
};

__aicore__ inline void SaxpyAIV::Init(
    GM_ADDR x, GM_ADDR y, uint32_t totalN, uint32_t perCoreN, uint32_t remainder, uint32_t tileSize, float alpha,
    TPipe* pipe)
{
    alpha_ = alpha;
    tileSize_ = tileSize;

    uint32_t blockIdx = GetBlockIdx();
    myOffset_ = blockIdx * perCoreN;
    myCount_ = perCoreN;
    if (blockIdx == GetBlockNum() - 1) {
        myCount_ += remainder;
    }

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), totalN);
    yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), totalN);

    uint32_t bufSize = tileSize * sizeof(float);
    pipe->InitBuffer(xBuf_, bufSize);
    pipe->InitBuffer(yBuf_, bufSize);
}

__aicore__ inline void SaxpyAIV::SingleIteration(uint32_t curOffset, uint32_t dataCount)
{
    uint32_t alignedCount = (dataCount / SAXPY_ELEMENTS_PER_BLOCK) * SAXPY_ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    LocalTensor<float> xLocal = xBuf_.Get<float>();
    LocalTensor<float> yLocal = yBuf_.Get<float>();

    if (alignedCount > 0) {
        DataCopy(xLocal, xGM_[curOffset], alignedCount);
        DataCopy(yLocal, yGM_[curOffset], alignedCount);
    }

    if (tailCount > 0) {
        uint8_t paddingNum = SAXPY_ELEMENTS_PER_BLOCK - tailCount;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0.0f};
        DataCopyPad(xLocal[alignedCount], xGM_[curOffset + alignedCount], copyParams, padParams);
        DataCopyPad(yLocal[alignedCount], yGM_[curOffset + alignedCount], copyParams, padParams);
    }

    Axpy(yLocal, xLocal, alpha_, static_cast<int32_t>(dataCount));

    if (alignedCount > 0) {
        DataCopy(yGM_[curOffset], yLocal, alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(yGM_[curOffset + alignedCount], yLocal[alignedCount], copyParams);
    }
}

__aicore__ inline void SaxpyAIV::Process()
{
    if (myCount_ == 0) {
        return;
    }

    uint32_t tileSize = tileSize_;
    uint32_t tileLoop = myCount_ / tileSize;
    uint32_t tileTail = myCount_ % tileSize;
    uint32_t curOffset = myOffset_;

    for (uint32_t i = 0; i < tileLoop; i++) {
        LocalTensor<float> xLocal = xBuf_.Get<float>();
        LocalTensor<float> yLocal = yBuf_.Get<float>();

        DataCopy(xLocal, xGM_[curOffset], tileSize);
        DataCopy(yLocal, yGM_[curOffset], tileSize);

        Axpy(yLocal, xLocal, alpha_, static_cast<int32_t>(tileSize));

        DataCopy(yGM_[curOffset], yLocal, tileSize);

        curOffset += tileSize;
    }

    if (tileTail > 0) {
        SingleIteration(curOffset, tileTail);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void saxpy_simt_compute(
    uint32_t calNum, uint32_t startOffset, uint32_t totalN, int32_t incx, int32_t incy, float alpha, __gm__ float* xGm,
    __gm__ float* yGm)
{
    if (calNum == 0) {
        return;
    }

    uint64_t absIncx = (incx < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(incx)) : static_cast<uint64_t>(incx);
    uint64_t absIncy = (incy < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(incy)) : static_cast<uint64_t>(incy);

    for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
        // issue #483: 保留步长符号，负步长按 BLAS 语义从 (n-1)*|inc| 反向遍历
        uint64_t logical = static_cast<uint64_t>(startOffset) + i;
        uint64_t xElement = (incx < 0) ? (static_cast<uint64_t>(totalN) - 1U - logical) * absIncx : logical * absIncx;
        uint64_t yElement = (incy < 0) ? (static_cast<uint64_t>(totalN) - 1U - logical) * absIncy : logical * absIncy;
        yGm[yElement] = alpha * xGm[xElement] + yGm[yElement];
    }
}

__global__ __aicore__ void saxpy_aiv_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SaxpyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    SaxpyAIV op;
    op.Init(x, y, tiling.totalN, tiling.perCoreN, tiling.remainder, tiling.tileSize, tiling.alpha, &pipe);
    op.Process();
}

__global__ __aicore__ void saxpy_simt_kernel(GM_ADDR x, GM_ADDR y, SaxpyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockIdx = GetBlockIdx();
    uint32_t calNum = tiling.calCount[blockIdx];

    if (calNum > 0) {
        asc_vf_call<saxpy_simt_compute>(
            dim3{tiling.nthreads, 1, 1}, calNum, tiling.startOffset[blockIdx], tiling.totalN,
            static_cast<int32_t>(tiling.incx), static_cast<int32_t>(tiling.incy), tiling.alpha,
            reinterpret_cast<__gm__ float*>(x), reinterpret_cast<__gm__ float*>(y));
    }
}

void saxpy_kernel_do(
    GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SaxpyTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);

    if (tiling.incx == 1 && tiling.incy == 1) {
        saxpy_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(x, y, nullptr, tiling);
    } else {
        saxpy_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, y, tiling);
    }
}
