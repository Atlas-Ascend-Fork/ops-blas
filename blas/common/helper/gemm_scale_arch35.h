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
 * \file gemm_scale_arch35.h
 * \brief Shared FP32 column-tiled epilogue helpers for the arch35 GEMM family.
 *
 * The in-place scale kernel (C = beta * C, with an optional complex multiply)
 * and the surrounding "walk the output columns in tiles" loop are identical in
 * gemm_kernel.cpp and csymm_cube_kernel.cpp, which tripped the duplicate-code
 * scan. Both kernels now instantiate the class below; the per-operator
 * __global__ entry points stay in their own translation units because they
 * export different kernel symbols.
 */

#pragma once

#include <cstdint>
#include "kernel_operator.h"

// Column partition: each block owns a contiguous [startCol, endCol) slice of the
// n output columns.
__aicore__ inline void PartitionColsByBlock(
    int32_t n, int32_t& colsPerCore, int32_t& startCol, int32_t& endCol)
{
    const uint32_t blockNum = AscendC::GetBlockNum();
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    colsPerCore = (n + static_cast<int32_t>(blockNum) - 1) / static_cast<int32_t>(blockNum);
    startCol = blockIdx * colsPerCore;
    endCol = startCol + colsPerCore;
    if (endCol > n) endCol = n;
    if (startCol >= n) {
        startCol = 0;
        endCol = 0;
    }
}

// Walk every (column, row-tile) work item of a column-partitioned element-wise
// kernel. Op must expose ProcessTile(int32_t col, int32_t rowOffset, int32_t count).
template <typename Op>
__aicore__ inline void ForEachColumnTile(
    Op& op, int32_t startCol, int32_t endCol, int32_t elemPerCol, int32_t tileSize)
{
    for (int32_t col = startCol; col < endCol; col++) {
        int32_t rowOffset = 0;
        while (rowOffset < elemPerCol) {
            int32_t count = elemPerCol - rowOffset;
            if (count > tileSize) count = tileSize;
            op.ProcessTile(col, rowOffset, count);
            rowOffset += count;
        }
    }
}

namespace gemm_scale {
// Targeted using-declarations (not a blanket `using namespace`) to avoid leaking
// the entire AscendC symbol set into every translation unit that includes this
// shared header. Only the symbols actually used below are imported here.
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Muls;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

constexpr int32_t SCALE_TILE_SIZE = 256;
constexpr int32_t SCALE_BUF_NUM = 2;

class ScaleFp32Kernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc,
        float betaReal, float betaImag, int32_t isComplex, TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = m;
        n_ = n;
        ldc_ = ldc;
        betaReal_ = betaReal;
        betaImag_ = betaImag;
        isComplex_ = isComplex;
        elemPerCol_ = isComplex ? (m * 2) : m;
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cInOut),
            static_cast<uint64_t>(ldc) * n * (isComplex ? 2 : 1));
        PartitionColsByBlock(n_, colsPerCore_, startCol_, endCol_);
    }

    __aicore__ inline void Process()
    {
        if (startCol_ >= endCol_) return;
        pipe_->InitBuffer(inQue_, SCALE_BUF_NUM, SCALE_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(outQue_, SCALE_BUF_NUM, SCALE_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(calcBuf_, SCALE_TILE_SIZE * sizeof(float));
        ForEachColumnTile(*this, startCol_, endCol_, elemPerCol_, SCALE_TILE_SIZE);
    }

    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t nbytes = static_cast<uint32_t>(count) * sizeof(float);
        DataCopyExtParams ext{1, nbytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

        uint64_t cOffset = static_cast<uint64_t>(col) * ldc_ * (isComplex_ ? 2 : 1) + rowOffset;
        LocalTensor<float> inTile = inQue_.AllocTensor<float>();
        DataCopyPad(inTile, cGlobal_[cOffset], ext, padParams);
        inQue_.EnQue(inTile);
        LocalTensor<float> inLocal = inQue_.DeQue<float>();

        LocalTensor<float> outTile = outQue_.AllocTensor<float>();
        if (isComplex_ && betaImag_ != 0.0f) {
            // Complex multiply: (br+bi*i)*(cr+ci*i) = (br*cr-bi*ci) + (br*ci+bi*cr)*i
            // count is even (2*m), process real/imag interleaved
            int32_t halfCount = count / 2;
            LocalTensor<float> tmp = calcBuf_.Get<float>();
            // tmp = bi * input
            Muls(tmp, inLocal, betaImag_, count);
            // out[even] = br*in[even] - bi*in[odd] = br*in[even] - tmp[odd]
            // out[odd]  = br*in[odd] + bi*in[even] = br*in[odd] + tmp[even]
            Muls(outTile, inLocal, betaReal_, count);
            for (int32_t i = 0; i < halfCount; i++) {
                int32_t re = i * 2;
                int32_t im = re + 1;
                outTile.SetValue(re, outTile.GetValue(re) - tmp.GetValue(im));
                outTile.SetValue(im, outTile.GetValue(im) + tmp.GetValue(re));
            }
        } else {
            Muls(outTile, inLocal, betaReal_, count);
        }
        outQue_.EnQue(outTile);
        LocalTensor<float> outLocal = outQue_.DeQue<float>();
        DataCopyPad(cGlobal_[cOffset], outLocal, ext);
        outQue_.FreeTensor(outLocal);
        inQue_.FreeTensor(inLocal);
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, SCALE_BUF_NUM> inQue_;
    TQue<QuePosition::VECOUT, SCALE_BUF_NUM> outQue_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    GlobalTensor<float> cGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int32_t elemPerCol_ = 0;
    float betaReal_ = 1.0f;
    float betaImag_ = 0.0f;
    int32_t isComplex_ = 0;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
    int32_t colsPerCore_ = 0;
};

} // namespace gemm_scale
