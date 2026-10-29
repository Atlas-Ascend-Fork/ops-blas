/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSROT_KERNEL_H
#define CSROT_KERNEL_H

#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/compute/simd.h"
using namespace AscendC;

constexpr uint32_t CSROT_COMPONENTS = 2;
constexpr uint32_t CSROT_STRIDED_ELEMS_PER_PASS = 1024;
constexpr uint32_t CSROT_STRIDED_SLOT = 8; // per-element 32B-aligned UB slot (2 data floats + padding)

// Physical element index of logical index i for stride inc; negative strides
// start from the physical vector tail per Netlib semantics.
__aicore__ __inline__ int64_t CsrotStridedPhysIndex(int64_t logical, int32_t inc, int64_t total)
{
    return (inc > 0) ? logical * inc : (total - 1 - logical) * (-static_cast<int64_t>(inc));
}

// Write one strided pass back from the 32B-aligned staging slots (8-byte
// DataCopyPad burst per complex element; UB-side offsets stay 32B-aligned,
// which the MTE datapath requires).
__aicore__ __inline__ void csrot_strided_store_pass(
    AscendC::GlobalTensor<float> gm_x, AscendC::GlobalTensor<float> gm_y, AscendC::LocalTensor<float> buf_temp1,
    AscendC::LocalTensor<float> buf_temp2, int64_t startElem, int64_t base, int64_t passElems, int64_t total,
    int32_t incx, int32_t incy)
{
    DataCopyExtParams outParams{1, CSROT_COMPONENTS * 4, 0, 0, 0};
    for (int64_t i = 0; i < passElems; i++) {
        const int64_t logical = startElem + base + i;
        const int64_t xPhys = CsrotStridedPhysIndex(logical, incx, total);
        const int64_t yPhys = CsrotStridedPhysIndex(logical, incy, total);
        DataCopyPad(
            gm_x[static_cast<uint64_t>(xPhys) * CSROT_COMPONENTS],
            buf_temp1[static_cast<uint32_t>(i) * CSROT_STRIDED_SLOT], outParams);
        DataCopyPad(
            gm_y[static_cast<uint64_t>(yPhys) * CSROT_COMPONENTS],
            buf_temp2[static_cast<uint32_t>(i) * CSROT_STRIDED_SLOT], outParams);
    }
}

// Strided rotation for non-unit strides: old values are read with scalar GM
// loads, rotated into 32B-aligned UB slots and written back with one 8-byte
// DataCopyPad burst per element. Negative strides start from the physical
// vector tail per Netlib semantics.
__aicore__ __inline__ void csrot_process_strided_aiv(
    AscendC::GlobalTensor<float> gm_x, AscendC::GlobalTensor<float> gm_y, int32_t elementCount, float cosValue,
    float sinValue, int32_t incx, int32_t incy)
{
    const float negaSinValue = 0 - sinValue;
    const int64_t total = elementCount / CSROT_COMPONENTS;
    const int16_t aiv_id = AscendC::GetBlockIdx();
    int16_t aiv_num = AscendC::GetBlockNum();
    if (aiv_num == 0) {
        aiv_num = 1;
    }

    int64_t elemsPerCore = total / aiv_num;
    const int64_t elemsRemain = total % aiv_num;
    int64_t startElem = static_cast<int64_t>(aiv_id) * elemsPerCore;
    if (aiv_id < elemsRemain) {
        startElem += aiv_id;
        elemsPerCore += 1;
    } else {
        startElem += elemsRemain;
    }

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> buf_temp1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> buf_temp2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(32 * 1024);

    for (int64_t base = 0; base < elemsPerCore; base += CSROT_STRIDED_ELEMS_PER_PASS) {
        const int64_t passElems =
            (elemsPerCore - base) < CSROT_STRIDED_ELEMS_PER_PASS ? (elemsPerCore - base) : CSROT_STRIDED_ELEMS_PER_PASS;
        for (int64_t i = 0; i < passElems; i++) {
            const int64_t logical = startElem + base + i;
            const int64_t xPhys = CsrotStridedPhysIndex(logical, incx, total);
            const int64_t yPhys = CsrotStridedPhysIndex(logical, incy, total);
            const uint64_t xF = static_cast<uint64_t>(xPhys) * CSROT_COMPONENTS;
            const uint64_t yF = static_cast<uint64_t>(yPhys) * CSROT_COMPONENTS;
            const float xr = gm_x.GetValue(xF);
            const float xi = gm_x.GetValue(xF + 1);
            const float yr = gm_y.GetValue(yF);
            const float yi = gm_y.GetValue(yF + 1);
            const uint32_t slot = static_cast<uint32_t>(i) * CSROT_STRIDED_SLOT;
            buf_temp1.SetValue(slot, cosValue * xr + sinValue * yr);
            buf_temp1.SetValue(slot + 1, cosValue * xi + sinValue * yi);
            buf_temp2.SetValue(slot, negaSinValue * xr + cosValue * yr);
            buf_temp2.SetValue(slot + 1, negaSinValue * xi + cosValue * yi);
        }
        // Scalar UB writes above do not ride the V pipe, so a full barrier is
        // required before MTE3 reads the staging slots.
        PIPE_BARRIER(ALL);
        csrot_strided_store_pass(gm_x, gm_y, buf_temp1, buf_temp2, startElem, base, passElems, total, incx, incy);
        PIPE_BARRIER(ALL);
    }
}

// CopyIn phase of the contiguous pipeline: stage the per-core float ranges of
// x and y into the ping-pong UB buffers.
__aicore__ __inline__ void csrot_contiguous_copy_in(
    AscendC::GlobalTensor<float> current_gm_x, AscendC::GlobalTensor<float> current_gm_y,
    AscendC::LocalTensor<float> buf_x, AscendC::LocalTensor<float> buf_y, int32_t currentElementSingleCount,
    event_t event_id)
{
    WAIT_FLAG(MTE3, MTE2, event_id);
    // CopyIn
    gm_to_ub_align<ArchType::ASCEND_V220, float>(
        buf_x,                                     // __ubuf__ void *dst,
        current_gm_x,                              // __gm__ void *src,
        0,                                         // uint8_t sid,
        1,                                         // uint16_t nBurst,
        currentElementSingleCount * sizeof(float), // uint32_t lenBurst,
        0,                                         // uint8_t leftPaddingNum,
        0,                                         // uint8_t rightPaddingNum,
        0,                                         // uint32_t srcGap,
        0);                                        // uint32_t dstGap

    SET_FLAG(MTE2, V, EVENT_ID0);

    gm_to_ub_align<ArchType::ASCEND_V220, float>(
        buf_y,                                     // __ubuf__ void *dst,
        current_gm_y,                              // __gm__ void *src,
        0,                                         // uint8_t sid,
        1,                                         // uint16_t nBurst,
        currentElementSingleCount * sizeof(float), // uint32_t lenBurst,
        0,                                         // uint8_t leftPaddingNum,
        0,                                         // uint8_t rightPaddingNum,
        0,                                         // uint32_t srcGap,
        0);                                        // uint32_t dstGap

    SET_FLAG(MTE2, V, EVENT_ID1);
}

// Compute phase of the contiguous pipeline: x' = c*x + s*y and y' = c*y - s*x,
// component-wise on the interleaved float lanes.
__aicore__ __inline__ void csrot_contiguous_compute(
    AscendC::LocalTensor<float> buf_x, AscendC::LocalTensor<float> buf_y, AscendC::LocalTensor<float> buf_temp1,
    AscendC::LocalTensor<float> buf_temp2, int32_t currentRepeatCount, float cosValue, float negaSinValue,
    float sinValue)
{
    WAIT_FLAG(MTE2, V, EVENT_ID0);

    // Compute
    muls_v<ArchType::ASCEND_V220, float>(
        buf_temp1,          // __ubuf__ float *dst,
        buf_x,              // __ubuf__ float *src0,
        cosValue,           // float src1,
        currentRepeatCount, // uint8_t repeat,
        1,                  // uint16_t dstBlockStride,
        1,                  // uint16_t srcBlockStride,
        8,                  // uint8_t dstRepeatStride,
        8);                 // uint8_t srcRepeatStride
    muls_v<ArchType::ASCEND_V220, float>(
        buf_temp2,          // __ubuf__ float *dst,
        buf_x,              // __ubuf__ float *src0,
        negaSinValue,       // float src1,
        currentRepeatCount, // uint8_t repeat,
        1,                  // uint16_t dstBlockStride,
        1,                  // uint16_t srcBlockStride,
        8,                  // uint8_t dstRepeatStride,
        8);                 // uint8_t srcRepeatStride

    PIPE_BARRIER(V);

    WAIT_FLAG(MTE2, V, EVENT_ID1);

    AscendC::Axpy(buf_temp1, buf_y, sinValue, currentRepeatCount * 64);
    AscendC::Axpy(buf_temp2, buf_y, cosValue, currentRepeatCount * 64);
}

// CopyOut phase of the contiguous pipeline: write the rotated ranges back to GM.
__aicore__ __inline__ void csrot_contiguous_copy_out(
    AscendC::GlobalTensor<float> current_gm_x, AscendC::GlobalTensor<float> current_gm_y,
    AscendC::LocalTensor<float> buf_temp1, AscendC::LocalTensor<float> buf_temp2, int32_t currentElementSingleCount,
    event_t event_id)
{
    SET_FLAG(V, MTE3, event_id);
    WAIT_FLAG(V, MTE3, event_id);

    // CopyOut
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        current_gm_x,                              // __gm__ void *dst,
        buf_temp1,                                 // __ubuf__ void *src,
        0,                                         // uint8_t sid,
        1,                                         // uint16_t nBurst,
        currentElementSingleCount * sizeof(float), // uint32_t lenBurst,
        0,                                         // uint8_t leftPaddingNum,
        0,                                         // uint8_t rightPaddingNum,
        0,                                         // uint32_t srcGap,
        0);                                        // uint32_t dstGap

    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        current_gm_y,                              // __gm__ void *dst,
        buf_temp2,                                 // __ubuf__ void *src,
        0,                                         // uint8_t sid,
        1,                                         // uint16_t nBurst,
        currentElementSingleCount * sizeof(float), // uint32_t lenBurst,
        0,                                         // uint8_t leftPaddingNum,
        0,                                         // uint8_t rightPaddingNum,
        0,                                         // uint32_t srcGap,
        0);                                        // uint32_t dstGap

    SET_FLAG(MTE3, MTE2, event_id);
}

// Unit-stride pipeline: per-core contiguous float ranges processed through
// double-buffered CopyIn -> Compute -> CopyOut phases.
__aicore__ __inline__ void csrot_process_contiguous_aiv(
    AscendC::GlobalTensor<float> gm_x_tensor, AscendC::GlobalTensor<float> gm_y_tensor, int32_t startIndex,
    int32_t elementPerCore, float cosValue, float sinValue)
{
    const float negaSinValue = 0 - sinValue;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> buf0_x = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);              // 24 KB
    AscendC::LocalTensor<float> buf0_y = buf.GetBuffer<BufferType::ASCEND_UB, float>(24 * 1024);      // 24 KB
    AscendC::LocalTensor<float> buf0_temp1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(48 * 1024);  // 24 KB
    AscendC::LocalTensor<float> buf0_temp2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(72 * 1024);  // 24 KB
    AscendC::LocalTensor<float> buf1_x = buf.GetBuffer<BufferType::ASCEND_UB, float>(96 * 1024);      // 24 KB
    AscendC::LocalTensor<float> buf1_y = buf.GetBuffer<BufferType::ASCEND_UB, float>(120 * 1024);     // 24 KB
    AscendC::LocalTensor<float> buf1_temp1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(144 * 1024); // 24 KB
    AscendC::LocalTensor<float> buf1_temp2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(168 * 1024); // 24 KB

    const int32_t maxElementSingleCount = 24 * 1024 / sizeof(float);
    int32_t loop_count = elementPerCore / maxElementSingleCount;
    const int32_t loop_remain = elementPerCore % maxElementSingleCount;
    if (loop_remain > 0) {
        loop_count += 1;
    }

    bool flag = 0;

    AscendC::GlobalTensor<float> current_gm_x = gm_x_tensor[startIndex];
    AscendC::GlobalTensor<float> current_gm_y = gm_y_tensor[startIndex];

    int32_t currentElementSingleCount = maxElementSingleCount;

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
        currentElementSingleCount =
            (loop_idx == loop_count - 1 && loop_remain > 0) ? loop_remain : maxElementSingleCount;
        int32_t currentRepeatCount = (currentElementSingleCount + 63) / 64;

        auto buf_x = flag ? buf0_x : buf1_x;
        auto buf_y = flag ? buf0_y : buf1_y;
        auto buf_temp1 = flag ? buf0_temp1 : buf1_temp1;
        auto buf_temp2 = flag ? buf0_temp2 : buf1_temp2;

        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;

        csrot_contiguous_copy_in(current_gm_x, current_gm_y, buf_x, buf_y, currentElementSingleCount, event_id);
        csrot_contiguous_compute(
            buf_x, buf_y, buf_temp1, buf_temp2, currentRepeatCount, cosValue, negaSinValue, sinValue);
        csrot_contiguous_copy_out(
            current_gm_x, current_gm_y, buf_temp1, buf_temp2, currentElementSingleCount, event_id);

        current_gm_x = current_gm_x[maxElementSingleCount];
        current_gm_y = current_gm_y[maxElementSingleCount];
        flag = 1 - flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

__global__ __aicore__ void csrot_kernel(GM_ADDR gm_x, GM_ADDR gm_y, GM_ADDR gm_workspace, GM_ADDR tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // Init
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();

    const int32_t elementCount = *reinterpret_cast<__gm__ int32_t*>(tiling_para_gm);
    float cosValue = *reinterpret_cast<__gm__ float*>(tiling_para_gm + 4);
    float sinValue = *reinterpret_cast<__gm__ float*>(tiling_para_gm + 8);
    const int32_t incx = *reinterpret_cast<__gm__ int32_t*>(tiling_para_gm + 12);
    const int32_t incy = *reinterpret_cast<__gm__ int32_t*>(tiling_para_gm + 16);

    const int16_t aiv_id = AscendC::GetBlockIdx();
    int16_t aiv_num = AscendC::GetBlockNum();
    if (aiv_num == 0) {
        aiv_num = 1;
    }

    int32_t elementPerCore = elementCount / aiv_num;
    const int32_t elementRemain = elementCount % aiv_num;

    int32_t startIndex = aiv_id * elementPerCore;
    if (aiv_id < elementRemain) {
        startIndex += aiv_id;
        elementPerCore += 1;
    } else {
        startIndex += elementRemain;
    }

    AscendC::GlobalTensor<float> gm_x_tensor;
    AscendC::GlobalTensor<float> gm_y_tensor;
    gm_x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gm_x));
    gm_y_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gm_y));

    if (incx != 1 || incy != 1) {
        csrot_process_strided_aiv(gm_x_tensor, gm_y_tensor, elementCount, cosValue, sinValue, incx, incy);
        return;
    }
    csrot_process_contiguous_aiv(gm_x_tensor, gm_y_tensor, startIndex, elementPerCore, cosValue, sinValue);
}

void csrot_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    csrot_kernel<<<numBlocks, nullptr, stream>>>(x, y, workSpace, tilingGm);
}

#endif // CSROT_KERNEL_H
