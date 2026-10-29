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
#include "cgemm_strided_batched_kernel.h"

// ---- direct implementation ----

namespace cgemm_sb {
using AscendC::Add;
using AscendC::Adds;
using AscendC::CreateVecIndex;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::Duplicate;
using AscendC::Gather;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::PipeBarrier;
using AscendC::ShiftRight;
using AscendC::Sub;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;

class DirectUb {
public:
    __aicore__ inline void Init(TPipe& pipe, const CgemmSbTiling& tiling)
    {
        t_ = tiling;
        pipe.InitBuffer(work_, 25U * kWidth * sizeof(float));
        pipe.InitBuffer(indices_, 8U * kWidth * sizeof(uint32_t));
        auto ix = indices_.Get<int32_t>();
        CreateVecIndex(ix, 0, kWidth);
        Muls(ix, ix, 8, kWidth);
        Adds(ix[kWidth], ix, 4, kWidth);
        Muls(ix[2U * kWidth], ix, 4, kWidth);
        Adds(ix[3U * kWidth], ix[2U * kWidth], 4, kWidth);
        Duplicate(ix[4U * kWidth], 0, kWidth);
        Duplicate(ix[5U * kWidth], 4, kWidth);
        auto output = ix[6U * kWidth];
        CreateVecIndex(output, 0, 2U * kWidth);
        ShiftRight(output, output, 1, 2U * kWidth);
        Muls(output, output, 4, 2U * kWidth);
        uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
        Adds(output, output, static_cast<int32_t>(kWidth * sizeof(float)), odd, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        const bool product = t_.alphaReal != 0.0f || t_.alphaImag != 0.0f;
        if (product) {
            ag_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
            bg_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        }
        cg_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
        const uint64_t rowTiles = (static_cast<uint64_t>(t_.m) + kWidth - 1U) / kWidth;
        for (uint64_t tile = GetBlockIdx(); tile < rowTiles * t_.n; tile += GetBlockNum()) {
            const uint32_t row = static_cast<uint32_t>(tile % rowTiles) * kWidth;
            const uint32_t col = static_cast<uint32_t>(tile / rowTiles);
            const uint32_t rows = t_.m - row < kWidth ? t_.m - row : kWidth;
            Duplicate(F(kSumReal), 0.0f, kWidth);
            Duplicate(F(kSumImag), 0.0f, kWidth);
            if (product) {
                for (uint32_t kk = 0; kk < t_.k; ++kk) {
                    LoadOperands(row, col, kk, rows);
                    Accumulate(rows);
                }
            }
            Store(row, col, rows);
        }
    }

private:
    static constexpr uint32_t kWidth = 32U;
    static constexpr uint32_t kInputA = 0U, kInputB = 8U, kInputC = 9U;
    static constexpr uint32_t kAr = 11U, kAi = 12U, kBr = 13U, kBi = 14U;
    static constexpr uint32_t kTempReal = 15U, kTempImag = 16U, kSumReal = 17U, kSumImag = 18U;
    static constexpr uint32_t kOutReal = 19U, kOutImag = 20U, kOldReal = 21U, kOldImag = 22U, kOutput = 23U;
    CgemmSbTiling t_{};
    TBuf<TPosition::VECCALC> work_, indices_;
    GlobalTensor<float> ag_, bg_, cg_;

    __aicore__ inline LocalTensor<float> F(uint32_t slot, uint32_t count = kWidth)
    {
        return work_.GetWithOffset<float>(count, slot * kWidth * sizeof(float));
    }

    __aicore__ inline LocalTensor<uint32_t> I(uint32_t slot, uint32_t count = kWidth)
    {
        return indices_.GetWithOffset<uint32_t>(count, slot * kWidth * sizeof(uint32_t));
    }

    __aicore__ inline void LoadA(uint32_t row, uint32_t kk, uint32_t rows)
    {
        auto input = F(kInputA, kWidth * 8U);
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        if (t_.transA == ACLBLAS_OP_N) {
            DataCopyExtParams copy{1U, rows * 8U, 0U, 0U, 0U};
            DataCopyPad(input, ag_[(static_cast<uint64_t>(kk) * t_.lda + row) * 2U], copy, pad);
        } else if (t_.lda - 1U <= 0xffffffffU / 8U) {
            DataCopyExtParams copy{static_cast<uint16_t>(rows), 8U, (t_.lda - 1U) * 8U, 0U, 0U};
            DataCopyPad(input, ag_[(static_cast<uint64_t>(row) * t_.lda + kk) * 2U], copy, pad);
        } else {
            // A DMA stride is 32-bit; form very large column addresses in 64-bit.
            DataCopyExtParams copy{1U, 8U, 0U, 0U, 0U};
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopyPad(input[i * 8U], ag_[(static_cast<uint64_t>(row + i) * t_.lda + kk) * 2U], copy, pad);
            }
        }
    }

    __aicore__ inline void LoadOperands(uint32_t row, uint32_t col, uint32_t kk, uint32_t rows)
    {
        PipeBarrier<PIPE_ALL>();
        LoadA(row, kk, rows);
        const uint64_t bi = t_.transB == ACLBLAS_OP_N ? static_cast<uint64_t>(col) * t_.ldb + kk :
                                                        static_cast<uint64_t>(kk) * t_.ldb + col;
        DataCopyExtParams copy{1U, 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(F(kInputB), bg_[bi * 2U], copy, pad);
        PipeBarrier<PIPE_ALL>();
        const uint32_t index = t_.transA == ACLBLAS_OP_N ? 0U : 2U;
        Gather(F(kAr), F(kInputA, kWidth * 8U), I(index), 0U, rows);
        Gather(F(kAi), F(kInputA, kWidth * 8U), I(index + 1U), 0U, rows);
        Gather(F(kBr), F(kInputB), I(4U), 0U, rows);
        Gather(F(kBi), F(kInputB), I(5U), 0U, rows);
        if (t_.transA == ACLBLAS_OP_C) {
            Muls(F(kAi), F(kAi), -1.0f, rows);
        }
        if (t_.transB == ACLBLAS_OP_C) {
            Muls(F(kBi), F(kBi), -1.0f, rows);
        }
    }

    __aicore__ inline void Accumulate(uint32_t rows)
    {
        Mul(F(kTempReal), F(kAr), F(kBr), rows);
        Mul(F(kTempImag), F(kAi), F(kBi), rows);
        Sub(F(kTempReal), F(kTempReal), F(kTempImag), rows);
        Add(F(kSumReal), F(kSumReal), F(kTempReal), rows);
        Mul(F(kTempReal), F(kAr), F(kBi), rows);
        Mul(F(kTempImag), F(kAi), F(kBr), rows);
        Add(F(kTempReal), F(kTempReal), F(kTempImag), rows);
        Add(F(kSumImag), F(kSumImag), F(kTempReal), rows);
    }

    __aicore__ inline void Scale(
        uint32_t real, uint32_t imag, uint32_t outReal, uint32_t outImag, float sr, float si, uint32_t rows)
    {
        // Skip zero components, matching BLAS handling of zero times Inf/NaN.
        if (si == 0.0f) {
            Muls(F(outReal), F(real), sr, rows);
            Muls(F(outImag), F(imag), sr, rows);
        } else if (sr == 0.0f) {
            Muls(F(outReal), F(imag), -si, rows);
            Muls(F(outImag), F(real), si, rows);
        } else {
            Muls(F(outReal), F(real), sr, rows);
            Muls(F(kTempReal), F(imag), -si, rows);
            Add(F(outReal), F(outReal), F(kTempReal), rows);
            Muls(F(outImag), F(imag), sr, rows);
            Muls(F(kTempReal), F(real), si, rows);
            Add(F(outImag), F(outImag), F(kTempReal), rows);
        }
    }

    __aicore__ inline void Store(uint32_t row, uint32_t col, uint32_t rows)
    {
        const uint64_t offset = (static_cast<uint64_t>(col) * t_.ldc + row) * 2U;
        Scale(kSumReal, kSumImag, kOutReal, kOutImag, t_.alphaReal, t_.alphaImag, rows);
        if (t_.betaReal != 0.0f || t_.betaImag != 0.0f) {
            DataCopyExtParams copy{1U, rows * 8U, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(F(kInputC, 2U * kWidth), cg_[offset], copy, pad);
            PipeBarrier<PIPE_ALL>();
            Gather(F(kOldReal), F(kInputC, 2U * kWidth), I(0U), 0U, rows);
            Gather(F(kOldImag), F(kInputC, 2U * kWidth), I(1U), 0U, rows);
            Scale(kOldReal, kOldImag, kAr, kAi, t_.betaReal, t_.betaImag, rows);
            Add(F(kOutReal), F(kOutReal), F(kAr), rows);
            Add(F(kOutImag), F(kOutImag), F(kAi), rows);
        }
        Gather(F(kOutput, 2U * kWidth), F(kOutReal, 2U * kWidth), I(6U, 2U * kWidth), 0U, rows * 2U);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copy{1U, rows * 8U, 0U, 0U, 0U};
        DataCopyPad(cg_[offset], F(kOutput, 2U * kWidth), copy);
        // Complete DMA before a following batch can read overlapping C.
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace cgemm_sb

using namespace AscendC;

extern "C" __global__ __aicore__ void cgemm_sb_direct_batched_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    cgemm_sb::DirectUb compute;
    compute.Init(pipe, t);
    const bool alphaZero = t.alphaReal == 0.0f && t.alphaImag == 0.0f;
    for (uint32_t batch = 0; batch < t.batchCount; ++batch) {
        GM_ADDR ai = alphaZero ? nullptr : a + static_cast<int64_t>(batch) * t.strideA * 8;
        GM_ADDR bi = alphaZero ? nullptr : b + static_cast<int64_t>(batch) * t.strideB * 8;
        GM_ADDR ci = c + static_cast<int64_t>(batch) * t.strideC * 8;
        // Overlapping C uses one core, preserving the previous batch's beta input.
        compute.Process(ai, bi, ci);
    }
}

void cgemm_sb_direct_batched_launch(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmSbTiling& tiling, uint32_t blocks, void* stream)
{
    cgemm_sb_direct_batched_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
}
