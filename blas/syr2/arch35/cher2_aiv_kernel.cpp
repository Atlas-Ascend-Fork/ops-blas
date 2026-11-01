/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "cher2_tiling_data.h"

using namespace AscendC;

#pragma clang fp contract(off)

namespace {
// One extra register makes every masked 64-lane UB load fit its own plane,
// including a LOWER subview beginning near the end of a 4096-element vector.
constexpr uint32_t CAPACITY = 4096 + 64;

class Cher2AivKernel {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR a, const Cher2TilingData& t)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        // Every plane has a fixed 32B-aligned origin. AoS is reused only after
        // its preceding DMA/vector consumer has completed.
        pipe_.InitBuffer(storage_, 10U * CAPACITY * sizeof(float));
        auto storage = storage_.Get<float>();
        auto xr = storage[0];
        auto xi = storage[CAPACITY];
        auto yr = storage[2U * CAPACITY];
        auto yi = storage[3U * CAPACITY];
        auto aos = storage[4U * CAPACITY];
        auto pr = storage[6U * CAPACITY];
        auto pi = storage[7U * CAPACITY];
        auto qr = storage[8U * CAPACITY];
        auto qi = storage[9U * CAPACITY];

        const bool upper = t.uplo == ACLBLAS_UPPER;
        // A core's LOWER columns have the same residue modulo eight when
        // the grid is aligned. Shift the initial GM load by that residue;
        // every later SoA subview then starts on a 32B boundary.
        const uint32_t shift = upper ? 0U : GetBlockIdx() % 8U;
        {
            const uint32_t remaining = t.n - shift;
            const uint32_t paddedN = (remaining + 7U) / 8U * 8U;
            LoadSplit(xGm_, 2ULL * shift, remaining, paddedN, aos, xr, xi);
            LoadSplit(yGm_, 2ULL * shift, remaining, paddedN, aos, yr, yi);
            ScaleComplex(yr, yi, pr, pi, t.alphaReal, t.alphaImag, paddedN, true);
            ScaleComplex(xr, xi, qr, qi, t.alphaReal, t.alphaImag, paddedN, false);
        }
        ProcessColumns(t, upper, shift, aos, xr, xi, yr, yi, pr, pi, qr, qi);
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void ProcessColumns(
        const Cher2TilingData& t, bool upper, uint32_t shift, LocalTensor<float> aos, LocalTensor<float> xr,
        LocalTensor<float> xi, LocalTensor<float> yr, LocalTensor<float> yi, LocalTensor<float> pr,
        LocalTensor<float> pi, LocalTensor<float> qr, LocalTensor<float> qi)
    {
        for (uint32_t col = GetBlockIdx(); col < t.n; col += GetBlockNum()) {
            const uint32_t begin = upper ? 0U : col;
            const uint32_t count = upper ? col + 1U : t.n - col;
            const uint32_t padded = (count + 7U) / 8U * 8U;
            const uint64_t offset = 2ULL * (static_cast<uint64_t>(col) * t.lda + begin);
            DataCopyExtParams copy{1, count * 8U, 0, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
            DataCopyPad(aos, aGm_[offset], copy, pad);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            const uint32_t columnLane = col - shift;
            const uint32_t rowOffset = !upper ? col - shift : 0U;
            const float xcR = xr.GetValue(columnLane), xcI = xi.GetValue(columnLane);
            const float ycR = yr.GetValue(columnLane), ycI = yi.GetValue(columnLane);
            const uint32_t diagonal = upper ? col : 0U;
            const float oldDiagonal = aos.GetValue(2U * diagonal);
            const bool nonzero = xcR != 0.0f || xcI != 0.0f || ycR != 0.0f || ycI != 0.0f;
            if (nonzero) {
                Compute(
                    aos, xr[rowOffset], xi[rowOffset], yr[rowOffset], yi[rowOffset], pr[columnLane], pi[columnLane],
                    qr[columnLane], qi[columnLane], count);
            }
            PipeBarrier<PIPE_ALL>();
            const float t1R = t.alphaReal * ycR + t.alphaImag * ycI;
            const float t1I = t.alphaImag * ycR - t.alphaReal * ycI;
            const float t2R = t.alphaReal * xcR - t.alphaImag * xcI;
            const float t2I = -(t.alphaReal * xcI + t.alphaImag * xcR);
            const float update = (xcR * t1R - xcI * t1I) + (ycR * t2R - ycI * t2I);
            aos.SetValue(2U * diagonal, nonzero ? oldDiagonal + update : oldDiagonal);
            aos.SetValue(2U * diagonal + 1U, 0.0f);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(aGm_[offset], aos, copy);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline void ScaleComplex(
        LocalTensor<float> real, LocalTensor<float> imag, LocalTensor<float> outReal, LocalTensor<float> outImag,
        float alphaReal, float alphaImag, uint32_t count, bool conjugateInput)
    {
        auto* realAddr = reinterpret_cast<__ubuf__ float*>(real.GetPhyAddr());
        auto* imagAddr = reinterpret_cast<__ubuf__ float*>(imag.GetPhyAddr());
        auto* outRealAddr = reinterpret_cast<__ubuf__ float*>(outReal.GetPhyAddr());
        auto* outImagAddr = reinterpret_cast<__ubuf__ float*>(outImag.GetPhyAddr());
        const uint16_t repeats = static_cast<uint16_t>((count + 63U) / 64U);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> realReg, imagReg, left, right, result;
            uint32_t remaining = count;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                auto mask = MicroAPI::UpdateMask<float>(remaining);
                const uint32_t offset = static_cast<uint32_t>(repeat) * 64U;
                MicroAPI::DataCopy(realReg, realAddr + offset);
                MicroAPI::DataCopy(imagReg, imagAddr + offset);
                MicroAPI::Muls(left, realReg, alphaReal, mask);
                MicroAPI::Muls(right, imagReg, alphaImag, mask);
                if (conjugateInput) {
                    MicroAPI::Add(result, left, right, mask);
                } else {
                    MicroAPI::Sub(result, left, right, mask);
                }
                MicroAPI::DataCopy(outRealAddr + offset, result, mask);
                MicroAPI::Muls(left, imagReg, alphaReal, mask);
                MicroAPI::Muls(right, realReg, alphaImag, mask);
                if (conjugateInput) {
                    MicroAPI::Sub(result, right, left, mask);
                } else {
                    MicroAPI::Add(result, left, right, mask);
                    MicroAPI::Muls(result, result, -1.0f, mask);
                }
                MicroAPI::DataCopy(outImagAddr + offset, result, mask);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadSplit(
        GlobalTensor<float>& gm, uint64_t offset, uint32_t count, uint32_t padded, LocalTensor<float> aos,
        LocalTensor<float> real, LocalTensor<float> imag)
    {
        // Clear the entire calculation tail in UB, never overread GM to
        // obtain alignment. This also covers LOWER's final one-element column.
        Duplicate(aos, 0.0f, padded * 2U);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copy{1, count * 8U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
        DataCopyPad(aos, gm[offset], copy, pad);
        PipeBarrier<PIPE_ALL>();
        DeInterleave(real, imag, aos, static_cast<int32_t>(padded * 2U));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Compute(
        LocalTensor<float> aos, LocalTensor<float> xr, LocalTensor<float> xi, LocalTensor<float> yr,
        LocalTensor<float> yi, LocalTensor<float> pr, LocalTensor<float> pi, LocalTensor<float> qr,
        LocalTensor<float> qi, uint32_t count)
    {
        auto* aosAddr = reinterpret_cast<__ubuf__ float*>(aos.GetPhyAddr());
        auto* xrAddr = reinterpret_cast<__ubuf__ float*>(xr.GetPhyAddr());
        auto* xiAddr = reinterpret_cast<__ubuf__ float*>(xi.GetPhyAddr());
        auto* yrAddr = reinterpret_cast<__ubuf__ float*>(yr.GetPhyAddr());
        auto* yiAddr = reinterpret_cast<__ubuf__ float*>(yi.GetPhyAddr());
        auto* prAddr = reinterpret_cast<__ubuf__ float*>(pr.GetPhyAddr());
        auto* piAddr = reinterpret_cast<__ubuf__ float*>(pi.GetPhyAddr());
        auto* qrAddr = reinterpret_cast<__ubuf__ float*>(qr.GetPhyAddr());
        auto* qiAddr = reinterpret_cast<__ubuf__ float*>(qi.GetPhyAddr());
        ComputeVector(aosAddr, xrAddr, xiAddr, yrAddr, yiAddr, prAddr, piAddr, qrAddr, qiAddr, count);
    }

    __aicore__ inline void ComputeVector(
        __ubuf__ float* aosAddr, __ubuf__ float* xrAddr, __ubuf__ float* xiAddr, __ubuf__ float* yrAddr,
        __ubuf__ float* yiAddr, __ubuf__ float* prAddr, __ubuf__ float* piAddr, __ubuf__ float* qrAddr,
        __ubuf__ float* qiAddr, uint32_t count)
    {
        const uint16_t repeats = static_cast<uint16_t>((count + 63U) / 64U);
        // Keep intermediates in vector registers rather than invoking a
        // separate vector function and round-tripping UB for each operation.
        // Separate multiply and Add/Sub preserve the non-FMA FP32 sequence.
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> vr, vi, vxr, vxi, vyr, vyi, left, right, a0, a1;
            MicroAPI::RegTensor<float> vpr, vpi, vqr, vqi;
            auto all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            // B32 broadcast loads accept the scalar coefficient address;
            // avoid four scalar-pipeline UB reads for every column.
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vpr, prAddr);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vpi, piAddr);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vqr, qrAddr);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vqi, qiAddr);
            uint32_t remaining = count;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                auto mask = MicroAPI::UpdateMask<float>(remaining);
                const uint32_t offset = static_cast<uint32_t>(repeat) * 64U;
                MicroAPI::DataCopy(a0, aosAddr + 2U * offset);
                MicroAPI::DataCopy(a1, aosAddr + 2U * offset + 64U);
                MicroAPI::DeInterleave(vr, vi, a0, a1);
                MicroAPI::DataCopy(vxr, xrAddr + offset);
                MicroAPI::DataCopy(vxi, xiAddr + offset);
                MicroAPI::DataCopy(vyr, yrAddr + offset);
                MicroAPI::DataCopy(vyi, yiAddr + offset);
                MicroAPI::Mul(left, vxr, vpr, mask);
                MicroAPI::Mul(right, vxi, vpi, mask);
                MicroAPI::Sub(left, left, right, mask);
                MicroAPI::Add(vr, vr, left, mask);
                MicroAPI::Mul(left, vxr, vpi, mask);
                MicroAPI::Mul(right, vxi, vpr, mask);
                MicroAPI::Add(left, left, right, mask);
                MicroAPI::Add(vi, vi, left, mask);
                MicroAPI::Mul(left, vyr, vqr, mask);
                MicroAPI::Mul(right, vyi, vqi, mask);
                MicroAPI::Sub(left, left, right, mask);
                MicroAPI::Add(vr, vr, left, mask);
                MicroAPI::Mul(left, vyr, vqi, mask);
                MicroAPI::Mul(right, vyi, vqr, mask);
                MicroAPI::Add(left, left, right, mask);
                MicroAPI::Add(vi, vi, left, mask);
                MicroAPI::Interleave(a0, a1, vr, vi);
                MicroAPI::DataCopy(aosAddr + 2U * offset, a0, all);
                MicroAPI::DataCopy(aosAddr + 2U * offset + 64U, a1, all);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> storage_;
    GlobalTensor<float> xGm_, yGm_, aGm_;
};
} // namespace

__global__ __aicore__ void cher2_aiv_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR a, const Cher2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2AivKernel kernel;
    kernel.Run(x, y, a, tiling);
}

void cher2_aiv_kernel_do(
    GM_ADDR x, GM_ADDR y, GM_ADDR a, const Cher2TilingData& tiling, uint32_t numBlocks, void* stream)
{
    cher2_aiv_kernel<<<numBlocks, nullptr, stream>>>(x, y, a, tiling);
}
