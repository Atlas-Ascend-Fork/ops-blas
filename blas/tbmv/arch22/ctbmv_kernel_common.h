/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>

#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "common/compute/simd.h"
#include "common/iterator/iterator.h"
#include "common/helper/kernel_utils.h"
#include "common/memory/mem.h"
#include "ctbmv_kernel.h"
#include "ctbmv_tiling_data.h"

namespace Ctbmv {
using AscendC::Abs;
using AscendC::Add;
using AscendC::BlockReduceSum;
using AscendC::Brcb;
using AscendC::CacheLine;
using AscendC::CMPMODE;
using AscendC::CompareScalar;
using AscendC::DataCacheCleanAndInvalid;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::Duplicate;
using AscendC::Gather;
using AscendC::GatherMask;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::PipeBarrier;
using AscendC::ResetMask;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicNone;
using AscendC::SetMaskNorm;
using AscendC::SetVectorMask;
using AscendC::Sub;
using AscendC::TPosition;
using AscendC::WholeReduceSum;

constexpr uint32_t CTBMV_FAST_TILE_COMPLEX = 1024U;
constexpr uint32_t CTBMV_COMPACT_FAST_MAX_K = 32U;
constexpr uint32_t CTBMV_COMPACT_FAST_MAX_LDA = CTBMV_COMPACT_FAST_MAX_K + 1U;
constexpr uint32_t CTBMV_COMPACT_GATHER_ALIGNMENT = 32U;
constexpr uint32_t CTBMV_COMPACT_PADDED_COMPLEX_FLOATS = CTBMV_COMPACT_GATHER_ALIGNMENT / sizeof(float);
constexpr uint32_t CTBMV_PANEL_TILE = 128U;
constexpr uint32_t CTBMV_PANEL_MAX_X = CTBMV_PANEL_TILE + CTBMV_COMPACT_FAST_MAX_K;
constexpr uint32_t CTBMV_PANEL_X_VIEWS = 8U;
constexpr uint32_t CTBMV_PANEL_GATHER_PHASES = 8U;
constexpr uint32_t CTBMV_PANEL_WINDOW_WORDS = CTBMV_PANEL_X_VIEWS * CTBMV_PANEL_MAX_X;
constexpr uint32_t CTBMV_PANEL_BASE_WORDS = CTBMV_PANEL_TILE * (2U + CTBMV_PANEL_GATHER_PHASES);
constexpr uint32_t CTBMV_PANEL_TABLE_WORDS = CTBMV_PANEL_BASE_WORDS + 2U * CTBMV_PANEL_WINDOW_WORDS;
constexpr uint32_t CTBMV_PANEL_MAX_A_FLOATS = CTBMV_PANEL_MAX_X * CTBMV_COMPACT_FAST_MAX_LDA * 2U;

class CtbmvShape {
protected:
    __aicore__ inline void GetRowRange(uint32_t row, uint32_t& colBegin, uint32_t& colEnd) const
    {
        if (trans_ == ACLBLAS_OP_N) {
            if (uplo_ == ACLBLAS_UPPER) {
                colBegin = row;
                colEnd = Min(n_ - 1U, row + k_);
            } else {
                colBegin = SaturatingSub(row, k_);
                colEnd = row;
            }
        } else if (uplo_ == ACLBLAS_UPPER) {
            colBegin = SaturatingSub(row, k_);
            colEnd = row;
        } else {
            colBegin = row;
            colEnd = Min(n_ - 1U, row + k_);
        }
    }

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
    __aicore__ inline uint32_t SaturatingSub(uint32_t a, uint32_t b) const { return a > b ? a - b : 0U; }
    __aicore__ inline void InitShape(const CtbmvTilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        n_ = tiling.n;
        k_ = tiling.k;
        lda_ = tiling.lda;
        useCoreNum_ = tiling.useCoreNum;
        rowsPerCore_ = tiling.rowsPerCore;
        uplo_ = tiling.uplo;
        trans_ = tiling.trans;
        diag_ = tiling.diag;
    }

    uint32_t blockIdx_ = 0U;
    uint32_t n_ = 0U;
    uint32_t k_ = 0U;
    uint32_t lda_ = 0U;
    uint32_t useCoreNum_ = 0U;
    uint32_t rowsPerCore_ = 0U;
    uint32_t uplo_ = ACLBLAS_UPPER;
    uint32_t trans_ = ACLBLAS_OP_N;
    uint32_t diag_ = ACLBLAS_NON_UNIT;
};

#if __DAV_C220_VEC__
class CtbmvVectorBuffers : public CtbmvShape {
protected:
    LocalTensor<float> aAos;
    LocalTensor<float> xAos;
    LocalTensor<float> aReal;
    LocalTensor<float> aImag;
    LocalTensor<float> xReal;
    LocalTensor<float> xImag;
    LocalTensor<float> outReal;
    LocalTensor<float> outImag;
    LocalTensor<float> tmpReal;
    LocalTensor<float> tmpImag;
};

__aicore__ inline void SplitComplexVector(
    LocalTensor<float> real, LocalTensor<float> imag, LocalTensor<float> aos, uint32_t count)
{
    const uint32_t repeats = (count * 2U + 63U) / 64U;
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(real.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(aos.GetPhyAddr()),
        nullptr, repeats, 1, 1, 8, 8);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(imag.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(aos.GetPhyAddr()),
        nullptr, repeats, 1, 2, 8, 8);
}
#endif

} // namespace Ctbmv
