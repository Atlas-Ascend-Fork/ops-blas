/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSWAP_KERNEL_H
#define SSWAP_KERNEL_H

#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/compute/simd.h"
#ifdef __CCE_KT_TEST__
#undef __aicore__
#define __aicore__
#else
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#endif

constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t TOTAL_VEC_CORE_NUM = 40;

#ifdef __DAV_C220_VEC__
__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 8 - len % 8;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, float>(
        dst, src, 0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
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
        dst, src, 0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void sswap_single_iteration_aiv(
    AscendC::GlobalTensor<float> gm_x, AscendC::GlobalTensor<float> gm_y, AscendC::LocalTensor<float> ub_x,
    AscendC::LocalTensor<float> ub_y, uint32_t len, uint32_t event_id)
{
    copy_vec_gm2ub(ub_x, gm_x, len);
    copy_vec_gm2ub(ub_y, gm_y, len);

    SET_FLAG(MTE2, MTE3, event_id);
    WAIT_FLAG(MTE2, MTE3, event_id);

    copy_vec_ub2gm(gm_x, ub_y, len);
    copy_vec_ub2gm(gm_y, ub_x, len);
}

__aicore__ __inline__ __attribute__((always_inline)) void sswap_process_aiv(
    AscendC::GlobalTensor<float> gm_x, AscendC::GlobalTensor<float> gm_y, uint32_t offset, uint32_t compute_num)
{
    AscendC::SetAtomicType<float>();
    SetAtomicnone();

    uint32_t ubSpacePerVar = 20;
    uint32_t maxDataCount = ubSpacePerVar * 1024 / 8;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ub_x_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> ub_x_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(ubSpacePerVar * 1024);
    AscendC::LocalTensor<float> ub_y_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(ubSpacePerVar * 2 * 1024);
    AscendC::LocalTensor<float> ub_y_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(ubSpacePerVar * 3 * 1024);

    uint32_t repeatTimes = compute_num / maxDataCount;
    uint32_t remainNum = compute_num % maxDataCount;

    uint32_t ping_flag = 1;

    uint32_t curr_offset = offset;

    if (repeatTimes > 0) {
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);

        for (uint32_t i = 0; i < repeatTimes; i++) {
            AscendC::LocalTensor<float> ub_x = ping_flag ? ub_x_ping : ub_x_pong;
            AscendC::LocalTensor<float> ub_y = ping_flag ? ub_y_ping : ub_y_pong;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            WAIT_FLAG(MTE3, MTE2, event_id);

            sswap_single_iteration_aiv(gm_x[curr_offset], gm_y[curr_offset], ub_x, ub_y, maxDataCount, event_id);

            curr_offset += maxDataCount;

            SET_FLAG(MTE3, MTE2, event_id);
            ping_flag = 1 - ping_flag;
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    }

    if (remainNum > 0) {
        auto ub_x = ping_flag ? ub_x_ping : ub_x_pong;
        auto ub_y = ping_flag ? ub_y_ping : ub_y_pong;
        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
        SET_FLAG(MTE3, MTE2, event_id);
        WAIT_FLAG(MTE3, MTE2, event_id);

        sswap_single_iteration_aiv(gm_x[curr_offset], gm_y[curr_offset], ub_x, ub_y, remainNum, event_id);
    }
    PIPE_BARRIER(ALL);

    SetAtomicnone();
}

// Strided swap: elements are read with scalar GM loads and staged into 32B-aligned
// UB slots; each element is written back with one 4-byte DataCopyPad burst from its
// slot, so arbitrary positive and negative strides are honoured (UB-side offsets stay
// 32B-aligned, which the MTE datapath requires). Negative strides start from the
// physical vector tail per Netlib semantics (same convention as cswap_strided_aiv
// and arch35 cswap_strided_kernel).
constexpr uint32_t SSWAP_STRIDED_ELEMS_PER_PASS = 1024;
constexpr uint32_t SSWAP_STRIDED_SLOT = 8; // per-element 32B-aligned UB slot (1 data float + padding)

__aicore__ __inline__ void sswap_process_strided_aiv(
    AscendC::GlobalTensor<float> gm_x, AscendC::GlobalTensor<float> gm_y, uint32_t offset, uint32_t compute_num,
    uint32_t n, int32_t incx, int32_t incy)
{
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ub_x = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> ub_y = buf.GetBuffer<BufferType::ASCEND_UB, float>(32 * 1024);

    const int64_t total = static_cast<int64_t>(n);
    AscendC::DataCopyExtParams outParams{1, BYTENUM_PER_FLOAT32, 0, 0, 0};

    for (uint64_t base = 0; base < compute_num; base += SSWAP_STRIDED_ELEMS_PER_PASS) {
        const uint32_t passElems = static_cast<uint32_t>(
            (compute_num - base) < SSWAP_STRIDED_ELEMS_PER_PASS ? (compute_num - base) : SSWAP_STRIDED_ELEMS_PER_PASS);
        for (uint32_t i = 0; i < passElems; i++) {
            const int64_t logical = static_cast<int64_t>(offset + base + i);
            const int64_t xPhys = (incx > 0) ? logical * incx : (total - 1 - logical) * (-static_cast<int64_t>(incx));
            const int64_t yPhys = (incy > 0) ? logical * incy : (total - 1 - logical) * (-static_cast<int64_t>(incy));
            const uint32_t slot = i * SSWAP_STRIDED_SLOT;
            // cross-stage: ub_x holds the old y element, ub_y the old x element
            ub_x.SetValue(slot, gm_y.GetValue(static_cast<uint64_t>(yPhys)));
            ub_y.SetValue(slot, gm_x.GetValue(static_cast<uint64_t>(xPhys)));
        }
        // Scalar UB writes above do not ride the V pipe, so a full barrier is
        // required before MTE3 reads the staging slots.
        PIPE_BARRIER(ALL);
        for (uint32_t i = 0; i < passElems; i++) {
            const int64_t logical = static_cast<int64_t>(offset + base + i);
            const int64_t xPhys = (incx > 0) ? logical * incx : (total - 1 - logical) * (-static_cast<int64_t>(incx));
            const int64_t yPhys = (incy > 0) ? logical * incy : (total - 1 - logical) * (-static_cast<int64_t>(incy));
            AscendC::DataCopyPad(gm_x[static_cast<uint64_t>(xPhys)], ub_x[i * SSWAP_STRIDED_SLOT], outParams);
            AscendC::DataCopyPad(gm_y[static_cast<uint64_t>(yPhys)], ub_y[i * SSWAP_STRIDED_SLOT], outParams);
        }
        PIPE_BARRIER(ALL);
    }
}
#endif

extern "C" __global__ __aicore__ void sswap(GM_ADDR x, GM_ADDR y, GM_ADDR tiling_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto core_idx = AscendC::GetBlockIdx();
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t*>(tiling_gm);

    uint32_t n = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf));
    uint32_t coreNum = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 4));
    uint32_t offset = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 8 + 4 * core_idx));
    uint32_t cal_num = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 8 + TOTAL_VEC_CORE_NUM * 4 + 4 * core_idx));
    int32_t incx = (*(__gm__ int32_t*)((__gm__ uint8_t*)tiling_buf + 8 + TOTAL_VEC_CORE_NUM * 8));
    int32_t incy = (*(__gm__ int32_t*)((__gm__ uint8_t*)tiling_buf + 8 + TOTAL_VEC_CORE_NUM * 8 + 4));

    AscendC::GlobalTensor<float> x_tensor;
    AscendC::GlobalTensor<float> y_tensor;
    x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    y_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

#ifdef __DAV_C220_VEC__
    if (incx == 1 && incy == 1) {
        sswap_process_aiv(x_tensor, y_tensor, offset, cal_num);
    } else {
        sswap_process_strided_aiv(x_tensor, y_tensor, offset, cal_num, n, incx, incy);
    }
#endif
}

void sswap_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    sswap<<<numBlocks, nullptr, stream>>>(x, y, tilingGm);
}

#endif