/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file ssyrk_kernel.cpp
 * \brief SSYRK Kernel implementation for ascend950 (DAV_3510)
 *        Phase 1: GEMM kernel   (AIC-only, tensor_api) - computes temp = op(A) * op(A)^T
 *                Uses DNExtLayoutPtn for on-the-fly transpose during GM→L1 (no separate transpose kernel).
 *        Phase 2: Scale kernel  (AIV-only, SIMD) - C = alpha * temp + beta * C
 *                Exploits temp symmetry: reads temp with swapped (iBase,jBase) to get column-major
 *                layout without Gather transpose.
 */

#include <cstdint>

#include "ssyrk_tiling_data.h"
#include "common/helper/syrk_gemm_arch35.h"
#include "common/helper/syrk_scale_arch35.h"
#include "ssyrk_kernel.h"

using namespace AscendC;

constexpr int64_t L1_SIZE = static_cast<int64_t>(SYRK_ARCH35_L1_SIZE_BYTES);

extern "C" __global__ __aicore__ void syrk_gemm_kernel(
    GM_ADDR gmA, GM_ADDR gmTemp,
    const SyrkGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    auto gmTempTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmTemp)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.tempRowStride));

    if (tiling.isTransN != 0) {
        // trans=N: left=DNExt(lda,K) reads A[n][k] from column-major; right=NDExt(K,lda) reads A^T
        auto gmLeftTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA)),
            te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.lda, tiling.k));
        auto gmRightTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA)),
            te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.k, tiling.lda));
        SyrkGemmKernelImpl<decltype(gmLeftTensor), decltype(gmRightTensor), decltype(gmTempTensor), SyrkGemmTilingData>(
            gmLeftTensor, gmRightTensor, gmTempTensor, tiling,
            SYRK_ARCH35_BASE_K, SYRK_ARCH35_L1_BUF_NUM, L1_SIZE);
    } else {
        // trans=T: left=NDExt(N,lda) reads A^T; right=DNExt(lda,N) reads A[k][n]
        auto gmLeftTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA)),
            te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.lda));
        auto gmRightTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmA)),
            te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.lda, tiling.n));
        SyrkGemmKernelImpl<decltype(gmLeftTensor), decltype(gmRightTensor), decltype(gmTempTensor), SyrkGemmTilingData>(
            gmLeftTensor, gmRightTensor, gmTempTensor, tiling,
            SYRK_ARCH35_BASE_K, SYRK_ARCH35_L1_BUF_NUM, L1_SIZE);
    }
}

void syrk_gemm_kernel_do(
    GM_ADDR gmA, GM_ADDR gmTemp,
    const SyrkGemmTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    syrk_gemm_kernel<<<numBlocks, nullptr, stream>>>(
        gmA, gmTemp, tiling);
}

// ================================================================================
// Phase 2: Scale Kernel (AIV-only, SIMD)
// C = alpha * temp + beta * C (column-major C, row-major temp)
// Only updates the uplo triangle (UPPER: j>=i, LOWER: j<=i).
// k=0 also only updates uplo triangle (no temp read, C=beta*C).
// Processes 64×64 blocks; skips blocks entirely outside uplo triangle.
//
// Transpose elimination via temp symmetry:
//   temp = A * A^T is symmetric, so temp[i][j] = temp[j][i].
//   Instead of reading temp[iBase..iBase+rows][jBase..jBase+cols] (row-major → needs
//   Gather transpose), we read temp[jBase..jBase+cols][iBase..iBase+rows] with swapped
//   coordinates. By symmetry, temp[jBase+c][iBase+r] = temp[iBase+r][jBase+c], which is
//   exactly the value we need. The DataCopyPad reads `cols` rows of `rows` elements each,
//   producing a column-major UB layout that matches cInUb — no Gather transpose needed.
// ================================================================================

constexpr uint32_t SYRK_SCALE_BLOCK = SYRK_ARCH35_SCALE_BLOCK;
constexpr uint32_t SYRK_SCALE_UB_FLOATS = SYRK_SCALE_BLOCK * SYRK_SCALE_BLOCK;

class SyrkScaleAIV {
public:
    __aicore__ inline explicit SyrkScaleAIV(TPipe& pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR gmTemp, GM_ADDR gmC, const SyrkScaleTilingData& tiling);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBlock(uint32_t iBase, uint32_t jBase,
        uint32_t rows, uint32_t cols);

private:
    TPipe& pipe_;
    SyrkScaleTilingData tiling_;
    GlobalTensor<float> tempGM_;
    GlobalTensor<float> cGM_;
    TBuf<TPosition::VECIN> tempBuf_;
    TBuf<TPosition::VECIN> cInBuf_;
    TBuf<TPosition::VECOUT> cOutBuf_;
    uint32_t rowStart_;
    uint32_t rowEnd_;
};

__aicore__ inline void SyrkScaleAIV::Init(
    GM_ADDR gmTemp, GM_ADDR gmC, const SyrkScaleTilingData& tiling)
{
    tiling_ = tiling;
    tempGM_.SetGlobalBuffer((__gm__ float*)gmTemp);
    cGM_.SetGlobalBuffer((__gm__ float*)gmC);

    pipe_.InitBuffer(tempBuf_, SYRK_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cInBuf_, SYRK_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cOutBuf_, SYRK_SCALE_UB_FLOATS * sizeof(float));

    uint32_t blockIdx = GetBlockIdx();
    rowStart_ = blockIdx * tiling_.rowsPerCore;
    rowEnd_ = Min<uint32_t>(rowStart_ + tiling_.rowsPerCore, tiling_.n);
}

__aicore__ inline void SyrkScaleAIV::ProcessBlock(
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    float alpha = tiling_.alphaVal;
    float beta = tiling_.betaVal;
    bool skipTemp = (tiling_.isAlphaZero || tiling_.isKZero);
    bool isBetaZero = tiling_.isBetaZero;

    uint32_t ldc = tiling_.ldc;
    uint32_t tempRowStride = tiling_.tempRowStride;
    uint32_t ubColStride = RoundUp<uint32_t>(rows, SYRK_ARCH35_ELEMENTS_PER_BLOCK);

    LocalTensor<float> cUb = cOutBuf_.Get<float>();
    LocalTensor<float> cInUb = cInBuf_.Get<float>();

    SyrkCrossBlockWarAcquire();

    int64_t cSrcStride = 0;
    int64_t cGmOffset = 0;
    SyrkLoadCBlock(cGM_, cInUb, iBase, jBase, rows, cols, ldc, cSrcStride, cGmOffset);

    LocalTensor<float> tempUb = tempBuf_.Get<float>();

    if (!skipTemp) {
        // Read temp with swapped coordinates: rows [jBase, jBase+cols), cols [iBase, iBase+rows).
        // By temp symmetry: temp[jBase+c][iBase+r] = temp[iBase+r][jBase+c] — correct value,
        // and the DataCopyPad produces column-major UB layout matching cInUb. No transpose needed.
        int64_t tempSrcStride = static_cast<int64_t>(tempRowStride - rows) * sizeof(float);
        DataCopyExtParams cpTemp{static_cast<uint16_t>(cols),
            static_cast<uint32_t>(rows * sizeof(float)),
            tempSrcStride, 0, 0};
        DataCopyPadExtParams<float> ppTemp{true, 0, 0, 0.0f};
        int64_t tempGmOffset = static_cast<int64_t>(jBase) * tempRowStride + iBase;
        DataCopyPad(tempUb, tempGM_[tempGmOffset], cpTemp, ppTemp);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

    SyrkScaleComputeResult(tiling_.uploMode, iBase, jBase, rows, cols, ubColStride,
        cUb, cInUb, tempUb, skipTemp, isBetaZero, alpha, beta);

    SyrkStoreCBlockAndSync(cGM_, cUb, cGmOffset, cSrcStride, rows, cols);
}

__aicore__ inline void SyrkScaleAIV::Process()
{
    // The pre-loop release primes block 0's acquire (the V pipe is empty, so
    // it fires immediately); each later acquire is released by the previous
    // block's compute-end release. The post-loop acquire consumes the final
    // block's release, keeping the Set/Wait counts balanced (a residual Set
    // without a matching Wait is undefined on Ascend 950).
    SyrkCrossBlockWarRelease();
    SyrkScaleProcess(*this, rowStart_, rowEnd_, tiling_.n, tiling_.uploMode, SYRK_SCALE_BLOCK);
    SyrkCrossBlockWarAcquire();
}

extern "C" __global__ __aicore__ void syrk_scale_kernel(
    GM_ADDR gmTemp, GM_ADDR gmC,
    const SyrkScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SyrkScaleAIV op(pipe);
    op.Init(gmTemp, gmC, tiling);
    op.Process();
}

void syrk_scale_kernel_do(
    GM_ADDR gmTemp, GM_ADDR gmC,
    const SyrkScaleTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    syrk_scale_kernel<<<numBlocks, nullptr, stream>>>(
        gmTemp, gmC, tiling);
}
