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

#include "kernel_operator.h"
#include "ctrsv_kernel_utils.h"
#include "common/helper/kernel_utils.h"

namespace ctrsv_layout {
constexpr uint32_t COMPLEX_FLOATS = 2U;

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

template <bool UNIT>
__aicore__ inline bool NeedsOrderedInput(
    AscendC::LocalTensor<float> matrix, AscendC::LocalTensor<float> vector, uint32_t n, uint32_t leading, bool upper)
{
    SET_FLAG(V, S, EVENT_ID0);
    WAIT_FLAG(V, S, EVENT_ID0);
    bool ordered = false;
    for (uint32_t row = 0; row < n; ++row) {
        ordered |= ctrsv_netlib::NeedsSoftwareProduct({vector.GetValue(row * 2U), vector.GetValue(row * 2U + 1U)});
        for (uint32_t col = 0; col < n; ++col) {
            if ((upper ? row > col : row < col) || (UNIT && row == col))
                continue;
            const uint32_t offset = (col * leading + row) * 2U;
            const ctrsv_netlib::Complex a{matrix.GetValue(offset), matrix.GetValue(offset + 1U)};
            ordered |= row == col ? ctrsv_netlib::NeedsFullRangeDiagonal(a) : ctrsv_netlib::NeedsSoftwareProduct(a);
        }
    }
    SET_FLAG(S, V, EVENT_ID0);
    WAIT_FLAG(S, V, EVENT_ID0);
    return ordered;
}

template <uint32_t STRIDE>
__aicore__ inline void CtrsvIndexRamp64(AscendC::LocalTensor<int32_t> indices)
{
    // Binary lane masks keep index generation on Vector without TPipe/GM.
    AscendC::Duplicate(indices, int32_t{0}, 64U);
    AscendC::PipeBarrier<PIPE_V>();
    constexpr uint64_t masks[] = {0xaaaaaaaaaaaaaaaaULL, 0xccccccccccccccccULL, 0xf0f0f0f0f0f0f0f0ULL,
                                  0xff00ff00ff00ff00ULL, 0xffff0000ffff0000ULL, 0xffffffff00000000ULL};
    for (uint32_t bit = 0; bit < 6U; ++bit) {
        uint64_t mask[2] = {masks[bit], 0U};
        AscendC::Adds(
            indices, indices, static_cast<int32_t>((1U << bit) * STRIDE), mask, 1U,
            AscendC::UnaryRepeatParams{1U, 1U, 8U, 8U});
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void SplitComplex(
    AscendC::LocalTensor<float> real, AscendC::LocalTensor<float> imag, AscendC::LocalTensor<float> interleaved,
    uint32_t complexCount)
{
    if (complexCount == 0U)
        return;
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>(0U, ~0ULL);
    // Keep each issue below the instruction's uint8 repeat limit.
    constexpr uint32_t maxComplexPerIssue = 128U * 32U;
    for (uint32_t offset = 0; offset < complexCount; offset += maxComplexPerIssue) {
        const uint32_t count = MinU32(maxComplexPerIssue, complexCount - offset);
        const uint8_t repeats = static_cast<uint8_t>((count + 31U) / 32U);
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(real[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(interleaved[offset * COMPLEX_FLOATS].GetPhyAddr()), nullptr, repeats,
            1, 1, 8, 8);
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(imag[offset].GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(interleaved[offset * COMPLEX_FLOATS].GetPhyAddr()), nullptr, repeats,
            1, 2, 8, 8);
    }
    AscendC::PipeBarrier<PIPE_V>();
}
template <uint32_t WIDTH>
__aicore__ inline void CtrsvTransposePlane(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src)
{
    for (uint32_t row = 0; row < WIDTH; row += 16U) {
        uint64_t sources[AscendC::NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t targets[AscendC::NCHW_CONV_ADDR_LIST_SIZE];
#pragma unroll
        for (uint32_t lane = 0; lane < 16U; ++lane) {
            sources[lane] = reinterpret_cast<uint64_t>(src[(row + lane) * WIDTH].GetPhyAddr());
        }
#pragma unroll
        for (uint32_t lane = 0; lane < 8U; ++lane) {
            targets[2U * lane] = reinterpret_cast<uint64_t>(dst[lane * WIDTH + row].GetPhyAddr());
            targets[2U * lane + 1U] = reinterpret_cast<uint64_t>(dst[lane * WIDTH + row + 8U].GetPhyAddr());
        }
        AscendC::TransDataTo5HD<float>(
            targets, sources, AscendC::TransDataTo5HDParams(false, false, WIDTH / 8U, WIDTH, 1U));
    }
}
} // namespace ctrsv_layout
