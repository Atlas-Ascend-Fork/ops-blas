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
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/compute/simd.h"
#include <type_traits>

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub_float(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0, // sid
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub_int(
    AscendC::LocalTensor<uint32_t> dst, AscendC::GlobalTensor<uint32_t> src, uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(uint32_t);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
        dst, src,
        0, // sid
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_ub2gm(
    AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src, uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0, // sid
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_separate_c64(
    AscendC::LocalTensor<float> ub_in, AscendC::LocalTensor<float> ub_sep, AscendC::GlobalTensor<float> x, uint32_t len)
{
    uint32_t repeatTime = (len + 63) / 64; // 3
    uint32_t real_offset = 0;
    uint32_t imag_offset = 32 * 1024 / sizeof(float) / 2;

    AscendC::LocalTensor<float> ub_out_real = ub_sep;
    AscendC::LocalTensor<float> ub_out_imag = ub_sep[imag_offset];

    copy_vec_gm2ub_float(ub_in, x, len);

    PIPE_BARRIER(ALL);

    // vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(ub_out_real.GetPhyAddr()),  // dst in UB
    //           reinterpret_cast<__ubuf__ uint32_t *>(ub_in.GetPhyAddr()),        // src in UB
    //           nullptr, repeatTime,
    //           1,      // src0BlockStride
    //           1,      // patternMode, 101010???0
    //           8, 8);  // src0RepeatStride, src1RepeatStride

    // vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(ub_out_imag.GetPhyAddr()),  // dst in UB
    //           reinterpret_cast<__ubuf__ uint32_t *>(ub_in.GetPhyAddr()),        // src in UB
    //           nullptr, repeatTime,
    //           1,      // src0BlockStride
    //           2,      // patternMode, 010101???1
    //           8, 8);  // src0RepeatStride, src1RepeatStride

    uint32_t mask = 0;
    uint64_t rsvdCnt = 0;
    AscendC::GatherMask<float>(
        ub_out_real, ub_in, 1, false, mask, {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    AscendC::GatherMask<float>(
        ub_out_imag, ub_in, 2, false, mask, {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);
}

__aicore__ __inline__ __attribute__((always_inline)) void compute(
    AscendC::GlobalTensor<float> gm_x, AscendC::GlobalTensor<float> gm_y, AscendC::LocalTensor<float> ub_y,
    AscendC::LocalTensor<uint32_t> gatherOffset, AscendC::LocalTensor<float> ub_calc,
    AscendC::LocalTensor<float> ub_out, AscendC::GlobalTensor<float> A, float alphaReal, float alphaImag, int32_t x_id,
    int64_t calNum, // ????????
    int64_t offset, uint32_t event_id)
{
    uint32_t real_offset = 0;
    uint32_t imag_offset = 32 * 1024 / sizeof(float) / 2;
    AscendC::LocalTensor<float> ub_in_real = ub_y;
    AscendC::LocalTensor<float> ub_in_imag = ub_y[imag_offset];

    float calc_x_imag = gm_x.GetValue(x_id * 2) * alphaImag + gm_x.GetValue(x_id * 2 + 1) * alphaReal;
    float calc_x_real = gm_x.GetValue(x_id * 2) * alphaReal - gm_x.GetValue(x_id * 2 + 1) * alphaImag;

    uint64_t repeatTimes = (calNum * 2 + 63) / 64;
    uint64_t computeRepeat = (calNum + 63) / 64;

    // R * R
    muls_v<ArchType::ASCEND_V220, float>(ub_calc, ub_in_real, calc_x_real, computeRepeat, 1, 1, 8, 8);

    // R * I
    muls_v<ArchType::ASCEND_V220, float>(ub_calc[imag_offset], ub_in_real, calc_x_imag, computeRepeat, 1, 1, 8, 8);

    // I * I
    muls_v<ArchType::ASCEND_V220, float>(ub_out, ub_in_imag, calc_x_imag, computeRepeat, 1, 1, 8, 8);

    PIPE_BARRIER(V);
    // R * R + I * I
    add_v<ArchType::ASCEND_V220, float>(ub_calc, ub_calc, ub_out, computeRepeat, 1, 1, 1, 8, 8, 8);

    // I * R
    muls_v<ArchType::ASCEND_V220, float>(ub_out[imag_offset], ub_in_imag, calc_x_real, computeRepeat, 1, 1, 8, 8);

    PIPE_BARRIER(V);
    // R * I - I * R
    sub_v<ArchType::ASCEND_V220, float>(
        ub_calc[imag_offset], ub_calc[imag_offset], ub_out[imag_offset], computeRepeat, 1, 1, 1, 8, 8, 8);

    PIPE_BARRIER(V);

    AscendC::Gather(ub_out, ub_calc, gatherOffset, (uint32_t)0, repeatTimes * 64);

    SET_FLAG(V, MTE3, event_id);
    WAIT_FLAG(V, MTE3, event_id);

    copy_vec_ub2gm(A[offset], ub_out, calNum * 2);
}

__aicore__ __inline__ __attribute__((always_inline)) void process(
    AscendC::GlobalTensor<float> x, AscendC::GlobalTensor<float> y, AscendC::GlobalTensor<uint32_t> gatherOffset,
    AscendC::GlobalTensor<float> A, float alphaReal, float alphaImag, uint32_t m, uint32_t n, uint64_t x_offset,
    uint64_t x_count_num)
{
    // ub 192kb
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ub_in = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<uint32_t> ub_gather_offset = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(32 * 1024);
    AscendC::LocalTensor<float> ub_out_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(64 * 1024);
    AscendC::LocalTensor<float> ub_out_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(96 * 1024);
    AscendC::LocalTensor<float> ub_calc_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(128 * 1024);
    AscendC::LocalTensor<float> ub_calc_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(160 * 1024);

    uint32_t max_data_count = 4 * 1024;

    // copyIn gather offset
    copy_vec_gm2ub_int(ub_gather_offset, gatherOffset, max_data_count * 2);
    PIPE_BARRIER(ALL);

    // tiling
    uint64_t row_block_num;
    uint64_t row_remain_num;

    uint32_t ping_flag = 1;

    row_block_num = n / max_data_count;
    row_remain_num = n % max_data_count;

    if (row_remain_num > 0)
        row_block_num++;

    uint64_t curr_compute_row_num = max_data_count;

    // compute
    for (uint64_t row_block_idx = 0; row_block_idx < row_block_num; row_block_idx++) {
        if (row_block_idx == row_block_num - 1 && row_remain_num > 0)
            curr_compute_row_num = row_remain_num;
        else
            curr_compute_row_num = max_data_count;

        PIPE_BARRIER(ALL);
        copy_separate_c64(ub_out_ping, ub_in, y[row_block_idx * max_data_count * 2], curr_compute_row_num * 2);
        PIPE_BARRIER(ALL);

        uint64_t calNum = curr_compute_row_num;
        uint64_t offset = row_block_idx * max_data_count;

        SET_FLAG(MTE3, V, EVENT_ID0);
        SET_FLAG(MTE3, V, EVENT_ID2);

        for (uint64_t i = x_offset; i < x_offset + x_count_num; i++) {
            AscendC::LocalTensor<float> ub_out = ping_flag ? ub_out_ping : ub_out_pong;
            AscendC::LocalTensor<float> ub_calc = ping_flag ? ub_calc_ping : ub_calc_pong;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID2;

            WAIT_FLAG(MTE3, V, event_id);

            compute(
                x, y, ub_in, ub_gather_offset, ub_calc, ub_out, A, alphaReal, alphaImag, i, calNum,
                i * n * 2 + offset * 2, event_id);

            SET_FLAG(MTE3, V, event_id);

            ping_flag = 1 - ping_flag;
        }

        WAIT_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID2);
    }

    PIPE_BARRIER(ALL);
}
