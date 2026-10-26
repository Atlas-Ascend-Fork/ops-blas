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
 * \file ssyrkx_kernel.cpp
 * \brief SSYRKX Kernel implementation for ascend950 (DAV_3510)
 *        Phase 1: GEMM kernel   (AIC-only, tensor_api) - computes temp_T = op(B) * op(A)^T
 *                Swaps left/right vs. syrkx formula (op(A)*op(B)^T) so that fixpipe output is
 *                the transpose of the desired temp. Uses DNExt/NDExt for on-the-fly transpose
 *                during GM→L1 (no separate transpose kernel).
 *        Phase 2: Scale kernel  (AIV-only, SIMD) - C = alpha * temp + beta * C
 *                Reads temp_T with swapped (iBase,jBase) coordinates. Since
 *                temp_T[j][i] = (op(B)*op(A)^T)[j][i] = (op(A)*op(B)^T)[i][j] = temp[i][j],
 *                the DataCopyPad produces column-major UB layout matching cInUb — no Gather
 *                transpose needed, and no symmetry assumption required.
 */

#include <cstdint>

#include "ssyrkx_tiling_data.h"
#include "common/helper/syrk_gemm_arch35.h"
#include "common/helper/syrk_scale_arch35.h"
#include "ssyrkx_kernel.h"

using namespace AscendC;

constexpr int64_t L1_SIZE = static_cast<int64_t>(SYRKX_ARCH35_L1_SIZE_BYTES);

extern "C" __global__ __aicore__ void ssyrkx_gemm_kernel(
    GM_ADDR gmB, GM_ADDR gmA, GM_ADDR gmTemp,
    const SsyrkxGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    auto gmTempTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmTemp)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.tempRowStride));

    auto gmBPtr = te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmB));
    auto gmAPtr = te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA));

    if (tiling.isTransN != 0) {
        auto gmLeftTensor = te::MakeTensor(gmBPtr, te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.ldb, tiling.k));
        auto gmRightTensor = te::MakeTensor(gmAPtr, te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.k, tiling.lda));
        SyrkGemmKernelImpl<decltype(gmLeftTensor), decltype(gmRightTensor), decltype(gmTempTensor), SsyrkxGemmTilingData>(
            gmLeftTensor, gmRightTensor, gmTempTensor, tiling,
            SYRKX_ARCH35_BASE_K, SYRKX_ARCH35_L1_BUF_NUM, L1_SIZE);
    } else {
        auto gmLeftTensor = te::MakeTensor(gmBPtr, te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.ldb));
        auto gmRightTensor = te::MakeTensor(gmAPtr, te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.lda, tiling.n));
        SyrkGemmKernelImpl<decltype(gmLeftTensor), decltype(gmRightTensor), decltype(gmTempTensor), SsyrkxGemmTilingData>(
            gmLeftTensor, gmRightTensor, gmTempTensor, tiling,
            SYRKX_ARCH35_BASE_K, SYRKX_ARCH35_L1_BUF_NUM, L1_SIZE);
    }
}

void ssyrkx_gemm_kernel_do(
    GM_ADDR gmB, GM_ADDR gmA, GM_ADDR gmTemp,
    const SsyrkxGemmTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssyrkx_gemm_kernel<<<numBlocks, nullptr, stream>>>(
        gmB, gmA, gmTemp, tiling);
}

// ================================================================================
// Phase 2: Scale Kernel (AIV-only, SIMD)
// C = alpha * temp + beta * C (column-major C, row-major temp_T)
// Only updates the uplo triangle (UPPER: j>=i, LOWER: j<=i).
// k=0 also only updates uplo triangle (no temp read, C=beta*C).
// Processes 64×64 blocks; skips blocks entirely outside uplo triangle.
//
// Transpose elimination via swapped-coordinate read:
//   temp_T = op(B) * op(A)^T, so temp_T[j][i] = (op(A)*op(B)^T)[i][j] = temp[i][j].
//   Instead of reading temp_T[iBase..iBase+rows][jBase..jBase+cols] (row-major → needs
//   Gather transpose), we read temp_T[jBase..jBase+cols][iBase..iBase+rows] with swapped
//   coordinates. By the identity above, temp_T[jBase+c][iBase+r] = temp[iBase+r][jBase+c],
//   which is exactly the value we need. The DataCopyPad reads `cols` rows of `rows` elements
//   each, producing a column-major UB layout that matches cInUb — no Gather transpose needed.
// ================================================================================

constexpr uint32_t SYRKX_SCALE_UB_FLOATS = SYRKX_ARCH35_SCALE_BLOCK * SYRKX_ARCH35_SCALE_BLOCK;
// Queue depth of the scale-kernel TQue buffers; 2 enables double buffering.
constexpr uint32_t SYRKX_SCALE_QUE_DEPTH = 2;

class SsyrkxScaleAIV {
public:
    __aicore__ inline explicit SsyrkxScaleAIV(TPipe& pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR gmTemp, GM_ADDR gmC, const SsyrkxScaleTilingData& tiling);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBlock(uint32_t iBase, uint32_t jBase,
        uint32_t rows, uint32_t cols);

private:
    TPipe& pipe_;
    const SsyrkxScaleTilingData* tiling_;
    GlobalTensor<float> tempGM_;
    GlobalTensor<float> cGM_;
    TQue<TPosition::VECIN, 1> tempQue_;
    TQue<TPosition::VECIN, 1> cInQue_;
    TQue<TPosition::VECOUT, 1> cOutQue_;
    uint32_t rowStart_;
    uint32_t rowEnd_;
};

__aicore__ inline void SsyrkxScaleAIV::Init(
    GM_ADDR gmTemp, GM_ADDR gmC, const SsyrkxScaleTilingData& tiling)
{
    // Kernel parameters stay valid for the whole core lifetime, so storing the
    // address (instead of copying the struct) is safe.
    tiling_ = &tiling;
    tempGM_.SetGlobalBuffer((__gm__ float*)gmTemp);
    cGM_.SetGlobalBuffer((__gm__ float*)gmC);

    // Each queue buffers one SCALE_BLOCK x SCALE_BLOCK tile of floats:
    //   tempQue_ - VECIN, enqueues temp-matrix tiles (GM -> UB)
    //   cInQue_  - VECIN, enqueues the original C tiles (GM -> UB)
    //   cOutQue_ - VECOUT, dequeues the scaled result tiles (UB -> GM)
    // Depth SYRKX_SCALE_QUE_DEPTH = 2 gives every queue double buffering so
    // that MTE transfers overlap with vector computation.
    pipe_.InitBuffer(tempQue_, SYRKX_SCALE_QUE_DEPTH, SYRKX_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cInQue_, SYRKX_SCALE_QUE_DEPTH, SYRKX_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cOutQue_, SYRKX_SCALE_QUE_DEPTH, SYRKX_SCALE_UB_FLOATS * sizeof(float));

    uint32_t blockIdx = GetBlockIdx();
    rowStart_ = blockIdx * tiling_->rowsPerCore;
    rowEnd_ = Min<uint32_t>(rowStart_ + tiling_->rowsPerCore, tiling_->n);
}

__aicore__ inline void SsyrkxScaleAIV::ProcessBlock(
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    float alpha = tiling_->alphaVal;
    float beta = tiling_->betaVal;
    bool skipTemp = (tiling_->isAlphaZero || tiling_->isKZero);
    bool isBetaZero = tiling_->isBetaZero;

    uint32_t ldc = tiling_->ldc;
    uint32_t tempRowStride = tiling_->tempRowStride;
    uint32_t ubColStride = RoundUp<uint32_t>(rows, SYRKX_ARCH35_ELEMENTS_PER_BLOCK);

    LocalTensor<float> cInUb = cInQue_.AllocTensor<float>();

    auto blockCount = static_cast<uint16_t>(cols);
    auto blockLen = static_cast<uint32_t>(rows * sizeof(float));
    int64_t cSrcStride = 0;
    int64_t cGmOffset = 0;
    SyrkLoadCBlock(cGM_, cInUb, iBase, jBase, rows, cols, ldc, cSrcStride, cGmOffset);
    cInQue_.EnQue(cInUb);

    LocalTensor<float> tempUb = tempQue_.AllocTensor<float>();

    // skipTemp (alpha == 0 or k == 0) is a rare degenerate case; hint the
    // compiler so the common path (reading temp) stays preferred.
    if (!__builtin_expect(skipTemp, 0)) {
        int64_t tempSrcStride = static_cast<int64_t>(static_cast<int64_t>(tempRowStride) - rows) * sizeof(float);
        DataCopyExtParams tempCopyParams{blockCount, blockLen, tempSrcStride, 0, 0};
        DataCopyPadExtParams<float> tempPadParams{true, 0, 0, 0.0f};
        int64_t tempGmOffset = static_cast<int64_t>(jBase) * tempRowStride + iBase;
        DataCopyPad(tempUb, tempGM_[tempGmOffset], tempCopyParams, tempPadParams);
    }
    tempQue_.EnQue(tempUb);

    cInUb = cInQue_.DeQue<float>();
    tempUb = tempQue_.DeQue<float>();

    LocalTensor<float> cUb = cOutQue_.AllocTensor<float>();

    SyrkScaleComputeResult(tiling_->uploMode, iBase, jBase, rows, cols, ubColStride,
        cUb, cInUb, tempUb, skipTemp, isBetaZero, alpha, beta);

    cOutQue_.EnQue(cUb);
    cInQue_.FreeTensor(cInUb);
    tempQue_.FreeTensor(tempUb);

    cUb = cOutQue_.DeQue<float>();

    DataCopyExtParams cOutCopyParams{blockCount, blockLen, 0, cSrcStride, 0};
    DataCopyPad(cGM_[cGmOffset], cUb, cOutCopyParams);
    cOutQue_.FreeTensor(cUb);
}

__aicore__ inline void SsyrkxScaleAIV::Process()
{
    SyrkScaleProcess(*this, rowStart_, rowEnd_, tiling_->n, tiling_->uploMode, SYRKX_ARCH35_SCALE_BLOCK);
}

extern "C" __global__ __aicore__ void ssyrkx_scale_kernel(
    GM_ADDR gmTemp, GM_ADDR gmC,
    const SsyrkxScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SsyrkxScaleAIV op(pipe);
    op.Init(gmTemp, gmC, tiling);
    op.Process();
}

void ssyrkx_scale_kernel_do(
    GM_ADDR gmTemp, GM_ADDR gmC,
    const SsyrkxScaleTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssyrkx_scale_kernel<<<numBlocks, nullptr, stream>>>(
        gmTemp, gmC, tiling);
}
