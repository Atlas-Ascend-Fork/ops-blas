/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file ssyr2k_kernel.cpp
 * \brief SSYR2K Kernel implementation for ascend950 (DAV_3510)
 *        Phase 1: GEMM kernel   (AIC-only, tensor_api) - computes temp1 = op(A) * op(B)^T
 *                                                         and temp2 = op(B) * op(A)^T
 *                Uses DNExtLayoutPtn for on-the-fly transpose during GM->L1 (no separate transpose kernel).
 *        Phase 2: Scale kernel  (AIV-only, SIMD) - C = alpha * (temp1 + temp2) + beta * C
 *                Exploits temp1+temp2 symmetry: reads both with swapped (iBase,jBase) to get
 *                column-major layout without Gather transpose.
 */

#include <cstdint>

#include "ssyr2k_tiling_data.h"
#include "common/helper/syrk_gemm_arch35.h"
#include "common/helper/syrk_scale_arch35.h"
#include "ssyr2k_kernel.h"

using namespace AscendC;

constexpr int64_t L1_SIZE = static_cast<int64_t>(SSYR2K_ARCH35_L1_SIZE_BYTES);

extern "C" __global__ __aicore__ void ssyr2k_gemm_kernel(
    GM_ADDR gmLeft, GM_ADDR gmRight, GM_ADDR gmTemp,
    const Ssyr2kGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    auto gmTempTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmTemp)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.tempRowStride));

    if (tiling.isTransN != 0) {
        // trans=N: left=DNExt(leftLd,K) right=NDExt(K,rightLd)
        auto gmLeftTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmLeft)),
            te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.leftLd, tiling.k));
        auto gmRightTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmRight)),
            te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.k, tiling.rightLd));
        SyrkGemmKernelImpl<decltype(gmLeftTensor), decltype(gmRightTensor), decltype(gmTempTensor), Ssyr2kGemmTilingData>(
            gmLeftTensor, gmRightTensor, gmTempTensor, tiling,
            SSYR2K_ARCH35_BASE_K, SSYR2K_ARCH35_L1_BUF_NUM, L1_SIZE);
    } else {
        // trans=T: left=NDExt(N,leftLd) right=DNExt(rightLd,N)
        auto gmLeftTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmLeft)),
            te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.leftLd));
        auto gmRightTensor = te::MakeTensor(
            te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmRight)),
            te::MakeFrameLayout<te::DNExtLayoutPtn>(tiling.rightLd, tiling.n));
        SyrkGemmKernelImpl<decltype(gmLeftTensor), decltype(gmRightTensor), decltype(gmTempTensor), Ssyr2kGemmTilingData>(
            gmLeftTensor, gmRightTensor, gmTempTensor, tiling,
            SSYR2K_ARCH35_BASE_K, SSYR2K_ARCH35_L1_BUF_NUM, L1_SIZE);
    }
}

void ssyr2k_gemm_kernel_do(
    GM_ADDR gmLeft, GM_ADDR gmRight, GM_ADDR gmTemp,
    const Ssyr2kGemmTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssyr2k_gemm_kernel<<<numBlocks, nullptr, stream>>>(
        gmLeft, gmRight, gmTemp, tiling);
}

// ================================================================================
// Phase 2: Scale Kernel (AIV-only, SIMD)
// C = alpha * (temp1 + temp2) + beta * C (column-major C, row-major temp1/temp2)
// Only updates the uplo triangle (UPPER: j>=i, LOWER: j<=i).
// k=0 or alpha=0 also only updates uplo triangle (no temp read, C=beta*C).
// Processes 64x64 blocks; skips blocks entirely outside uplo triangle.
//
// Transpose elimination via temp1+temp2 symmetry:
//   temp1 + temp2 = op(A)*op(B)^T + op(B)*op(A)^T is symmetric, so
//   (temp1+temp2)[i][j] = (temp1+temp2)[j][i].
//   Instead of reading temp[iBase..iBase+rows][jBase..jBase+cols] (row-major -> needs
//   Gather transpose), we read temp[jBase..jBase+cols][iBase..iBase+rows] with swapped
//   coordinates. By symmetry, (temp1+temp2)[jBase+c][iBase+r] = (temp1+temp2)[iBase+r][jBase+c],
//   which is exactly the value we need. We read temp1 and temp2 separately with swapped
//   coordinates, add them in UB, and the result is in column-major layout matching cInUb.
// ================================================================================

constexpr uint32_t SSYR2K_SCALE_BLOCK = SSYR2K_ARCH35_SCALE_BLOCK;
constexpr uint32_t SSYR2K_SCALE_UB_FLOATS = SSYR2K_SCALE_BLOCK * SSYR2K_SCALE_BLOCK;

class Ssyr2kScaleAIV {
public:
    __aicore__ inline explicit Ssyr2kScaleAIV(TPipe& pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC, const Ssyr2kScaleTilingData& tiling);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBlock(uint32_t iBase, uint32_t jBase,
        uint32_t rows, uint32_t cols);

private:
    TPipe& pipe_;
    Ssyr2kScaleTilingData tiling_;
    GlobalTensor<float> temp1GM_;
    GlobalTensor<float> temp2GM_;
    GlobalTensor<float> cGM_;
    TBuf<TPosition::VECIN> temp1Buf_;
    TBuf<TPosition::VECIN> temp2Buf_;
    TBuf<TPosition::VECIN> cInBuf_;
    TBuf<TPosition::VECOUT> cOutBuf_;
    uint32_t rowStart_;
    uint32_t rowEnd_;
};

__aicore__ inline void Ssyr2kScaleAIV::Init(
    GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC, const Ssyr2kScaleTilingData& tiling)
{
    tiling_ = tiling;
    temp1GM_.SetGlobalBuffer((__gm__ float*)gmTemp1);
    temp2GM_.SetGlobalBuffer((__gm__ float*)gmTemp2);
    cGM_.SetGlobalBuffer((__gm__ float*)gmC);

    pipe_.InitBuffer(temp1Buf_, SSYR2K_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(temp2Buf_, SSYR2K_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cInBuf_, SSYR2K_SCALE_UB_FLOATS * sizeof(float));
    pipe_.InitBuffer(cOutBuf_, SSYR2K_SCALE_UB_FLOATS * sizeof(float));

    uint32_t blockIdx = GetBlockIdx();
    rowStart_ = blockIdx * tiling_.rowsPerCore;
    rowEnd_ = Min<uint32_t>(rowStart_ + tiling_.rowsPerCore, tiling_.n);
}

__aicore__ inline void Ssyr2kScaleAIV::ProcessBlock(
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    float alpha = tiling_.alphaVal;
    float beta = tiling_.betaVal;
    bool skipTemp = (tiling_.isAlphaZero || tiling_.isKZero);
    bool isBetaZero = tiling_.isBetaZero;

    uint32_t ldc = tiling_.ldc;
    uint32_t tempRowStride = tiling_.tempRowStride;
    uint32_t ubColStride = RoundUp<uint32_t>(rows, SSYR2K_ARCH35_ELEMENTS_PER_BLOCK);

    LocalTensor<float> cUb = cOutBuf_.Get<float>();
    LocalTensor<float> cInUb = cInBuf_.Get<float>();

    SyrkCrossBlockWarAcquire();

    int64_t cSrcStride = 0;
    int64_t cGmOffset = 0;
    SyrkLoadCBlock(cGM_, cInUb, iBase, jBase, rows, cols, ldc, cSrcStride, cGmOffset);

    LocalTensor<float> temp1Ub = temp1Buf_.Get<float>();

    if (!skipTemp) {
        // Read temp1 and temp2 with swapped coordinates: rows [jBase, jBase+cols), cols [iBase, iBase+rows).
        // By (temp1+temp2) symmetry: (temp1+temp2)[jBase+c][iBase+r] = (temp1+temp2)[iBase+r][jBase+c]
        // -- correct value, and DataCopyPad produces column-major UB layout matching cInUb.
        LocalTensor<float> temp2Ub = temp2Buf_.Get<float>();
        int64_t tempSrcStride = static_cast<int64_t>(tempRowStride - rows) * sizeof(float);
        DataCopyExtParams cpTemp{static_cast<uint16_t>(cols),
            static_cast<uint32_t>(rows * sizeof(float)),
            tempSrcStride, 0, 0};
        DataCopyPadExtParams<float> ppTemp{true, 0, 0, 0.0f};
        int64_t tempGmOffset = static_cast<int64_t>(jBase) * tempRowStride + iBase;
        DataCopyPad(temp1Ub, temp1GM_[tempGmOffset], cpTemp, ppTemp);
        DataCopyPad(temp2Ub, temp2GM_[tempGmOffset], cpTemp, ppTemp);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        // temp1Ub = temp1 + temp2 (in-place add, full block)
        Add(temp1Ub, temp1Ub, temp2Ub, static_cast<int32_t>(ubColStride * cols));
    } else {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    }

    SyrkScaleComputeResult(tiling_.uploMode, iBase, jBase, rows, cols, ubColStride,
        cUb, cInUb, temp1Ub, skipTemp, isBetaZero, alpha, beta);

    SyrkStoreCBlockAndSync(cGM_, cUb, cGmOffset, cSrcStride, rows, cols);
}

__aicore__ inline void Ssyr2kScaleAIV::Process()
{
    // The pre-loop release primes block 0's acquire (the V pipe is empty, so
    // it fires immediately); each later acquire is released by the previous
    // block's compute-end release. The post-loop acquire consumes the final
    // block's release, keeping the Set/Wait counts balanced (a residual Set
    // without a matching Wait is undefined on Ascend 950).
    SyrkCrossBlockWarRelease();
    SyrkScaleProcess(*this, rowStart_, rowEnd_, tiling_.n, tiling_.uploMode, SSYR2K_SCALE_BLOCK);
    SyrkCrossBlockWarAcquire();
}

extern "C" __global__ __aicore__ void ssyr2k_scale_kernel(
    GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC,
    const Ssyr2kScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    Ssyr2kScaleAIV op(pipe);
    op.Init(gmTemp1, gmTemp2, gmC, tiling);
    op.Process();
}

void ssyr2k_scale_kernel_do(
    GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC,
    const Ssyr2kScaleTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssyr2k_scale_kernel<<<numBlocks, nullptr, stream>>>(
        gmTemp1, gmTemp2, gmC, tiling);
}
