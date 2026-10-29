/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef __CCE_KT_TEST__
#undef __aicore__
#define __aicore__
#else
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#endif

#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/compute/simd.h"
constexpr int MAX_LENG_PER_UB_PROC = 6144;
constexpr int COMPLEX_DATA_NUM_PER_BLOCK = 4;
constexpr int COMPLEX_DATA_NUM_PER_REPEAT = 32;
constexpr int FLOAT_DATA_NUM_PER_REPEAT = 64;

struct UB_FOR_CACL {
    AscendC::LocalTensor<float> buf_for_load_vector_fp32;
    AscendC::LocalTensor<float> buf_for_real_part_fp32;
    AscendC::LocalTensor<float> buf_for_imag_part_fp32;
    AscendC::LocalTensor<uint32_t> buf_for_mask_fp32;

    AscendC::LocalTensor<float> buf_for_imag_real_fp32;
    AscendC::LocalTensor<float> buf_for_imag_imag_fp32;

    AscendC::LocalTensor<float> buf_for_final_real_fp32;
    AscendC::LocalTensor<float> buf_for_final_imag_fp32;
    AscendC::LocalTensor<float> buf_for_final_complex_fp32;

    AscendC::LocalTensor<float> pp_buf_for_load_vector_fp32;
    AscendC::LocalTensor<float> pp_buf_for_load_vector_fp32_0;
    AscendC::LocalTensor<float> pp_buf_for_load_vector_fp32_1;
    AscendC::LocalTensor<float> pp_buf_for_real_real_fp32;
    AscendC::LocalTensor<float> pp_buf_for_real_real_fp32_0;
    AscendC::LocalTensor<float> pp_buf_for_real_real_fp32_1;
    AscendC::LocalTensor<float> pp_buf_for_real_imag_fp32;
    AscendC::LocalTensor<float> pp_buf_for_real_imag_fp32_0;
    AscendC::LocalTensor<float> pp_buf_for_real_imag_fp32_1;
    AscendC::LocalTensor<float> pp_buf_for_final_complex_fp32;
    AscendC::LocalTensor<float> pp_buf_for_final_complex_fp32_0;
    AscendC::LocalTensor<float> pp_buf_for_final_complex_fp32_1;
};

struct LocalInfo {
    int32_t cur_core_id;
    int32_t start_pos;
    int32_t element_count;
};

struct Param {
    int32_t data_offset;
    int32_t data_count;
};

#ifdef __DAV_C220_VEC__

__aicore__ __inline__ void allocate_ubuf(UB_FOR_CACL* ub_buf)
{
    int32_t offset = 0;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    ub_buf->buf_for_mask_fp32 = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(offset);
    offset += MAX_LENG_PER_UB_PROC * 2 * 4;

    ub_buf->buf_for_load_vector_fp32 = buf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += MAX_LENG_PER_UB_PROC * 8 * 2;

    ub_buf->buf_for_real_part_fp32 = buf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += MAX_LENG_PER_UB_PROC * 4;

    ub_buf->buf_for_imag_part_fp32 = buf.GetBuffer<BufferType::ASCEND_UB, float>(offset);

    ub_buf->pp_buf_for_load_vector_fp32_0 = ub_buf->buf_for_load_vector_fp32;
    ub_buf->pp_buf_for_load_vector_fp32_1 = ub_buf->buf_for_load_vector_fp32[MAX_LENG_PER_UB_PROC * 2];

    ub_buf->buf_for_imag_real_fp32 = ub_buf->buf_for_real_part_fp32;
    ub_buf->buf_for_imag_imag_fp32 = ub_buf->buf_for_imag_part_fp32;
    ub_buf->buf_for_final_imag_fp32 = ub_buf->buf_for_real_part_fp32;
    ub_buf->buf_for_final_real_fp32 = ub_buf->buf_for_imag_part_fp32;

    ub_buf->pp_buf_for_real_real_fp32_0 = ub_buf->pp_buf_for_load_vector_fp32_0;
    ub_buf->pp_buf_for_real_real_fp32_1 = ub_buf->pp_buf_for_load_vector_fp32_1;

    ub_buf->pp_buf_for_real_imag_fp32_0 = ub_buf->pp_buf_for_load_vector_fp32_0[MAX_LENG_PER_UB_PROC];
    ub_buf->pp_buf_for_real_imag_fp32_1 = ub_buf->pp_buf_for_load_vector_fp32_1[MAX_LENG_PER_UB_PROC];

    ub_buf->pp_buf_for_final_complex_fp32_0 = ub_buf->pp_buf_for_load_vector_fp32_0;
    ub_buf->pp_buf_for_final_complex_fp32_1 = ub_buf->pp_buf_for_load_vector_fp32_1;
}

__aicore__ __inline__ void get_local_info(LocalInfo* local_info, int vector_len)
{
    local_info->cur_core_id = AscendC::GetBlockIdx();

    // 纵深防御：空任务（host 侧已拦截 n<=0）时所有核置 start_pos=-1，
    // 避免 max_cores=0 下的 0/0 除零
    if (vector_len <= 0) {
        local_info->start_pos = -1;
        local_info->element_count = 0;
        return;
    }

    int32_t max_cores = vector_len / COMPLEX_DATA_NUM_PER_BLOCK;

    int32_t total_cores = AscendC::GetBlockNum();
    int32_t cores_num = total_cores > max_cores ? max_cores : total_cores;
    if (cores_num == 0) {
        cores_num = 1;
    }

    int32_t elnum_per_core = 0;
    if (cores_num != 0) {
        elnum_per_core = (vector_len / cores_num + 3) / 4 * 4;
        cores_num = vector_len / elnum_per_core;
    }

    int32_t tail = (vector_len - elnum_per_core * cores_num + 3) / 4 * 4;
    bool tail_only = false;
    if (tail > 0 && cores_num < total_cores) {
        cores_num += 1;
        tail_only = true;
    }

    if (local_info->cur_core_id >= cores_num) {
        local_info->start_pos = -1;
    } else {
        local_info->start_pos = local_info->cur_core_id * elnum_per_core;
        local_info->element_count = elnum_per_core;

        if (local_info->cur_core_id == cores_num - 1) {
            if (tail_only == false) {
                local_info->element_count += tail;
            } else {
                local_info->element_count = tail;
            }
        }
    }
}

__aicore__ __inline__ void _cacl(
    Param param, int32_t ping_pong_flag, UB_FOR_CACL ub_buf, float alpha_real, float alpha_imag,
    AscendC::GlobalTensor<float> vector_x, AscendC::GlobalTensor<float> vector_y,
    AscendC::GlobalTensor<float> workspace, int32_t loops)
{
    auto event_id = ping_pong_flag == 0 ? EVENT_ID0 : EVENT_ID1;

    WAIT_FLAG(MTE3, MTE2, event_id);

    ub_buf.pp_buf_for_load_vector_fp32 =
        (ping_pong_flag == 0) ? ub_buf.pp_buf_for_load_vector_fp32_0 : ub_buf.pp_buf_for_load_vector_fp32_1;
    ub_buf.pp_buf_for_real_real_fp32 =
        (ping_pong_flag == 0) ? ub_buf.pp_buf_for_real_real_fp32_0 : ub_buf.pp_buf_for_real_real_fp32_1;
    ub_buf.pp_buf_for_real_imag_fp32 =
        (ping_pong_flag == 0) ? ub_buf.pp_buf_for_real_imag_fp32_0 : ub_buf.pp_buf_for_real_imag_fp32_1;
    ub_buf.pp_buf_for_final_complex_fp32 =
        (ping_pong_flag == 0) ? ub_buf.pp_buf_for_final_complex_fp32_0 : ub_buf.pp_buf_for_final_complex_fp32_1;

    gm_to_ub<ArchType::ASCEND_V220, float>(
        ub_buf.pp_buf_for_load_vector_fp32, vector_x[param.data_offset * 2], 0, 1,
        param.data_count / COMPLEX_DATA_NUM_PER_BLOCK, 0, 0);

    SET_FLAG(MTE2, V, event_id);
    WAIT_FLAG(MTE2, V, event_id);

    int32_t repeat_times_complex = param.data_count / COMPLEX_DATA_NUM_PER_REPEAT;
    int32_t repeat_tail_complex = param.data_count % COMPLEX_DATA_NUM_PER_REPEAT;
    int32_t all_repeat_times_complex = repeat_times_complex + (repeat_tail_complex > 0 ? 1 : 0);

    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(ub_buf.buf_for_real_part_fp32.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(ub_buf.pp_buf_for_load_vector_fp32.GetPhyAddr()), nullptr,
        all_repeat_times_complex, 1, 1, 8, 8);

    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(ub_buf.buf_for_imag_part_fp32.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(ub_buf.pp_buf_for_load_vector_fp32.GetPhyAddr()), nullptr,
        all_repeat_times_complex, 1, 2, 8, 8);

    PIPE_BARRIER(V);

    int32_t repeat_times = param.data_count / FLOAT_DATA_NUM_PER_REPEAT;
    int32_t repeat_tail = param.data_count % FLOAT_DATA_NUM_PER_REPEAT;
    int32_t all_repeat_times = repeat_times + (repeat_tail > 0 ? 1 : 0);

    muls_v<ArchType::ASCEND_V220, float>(
        ub_buf.pp_buf_for_real_real_fp32, ub_buf.buf_for_real_part_fp32, alpha_real, all_repeat_times, 1, 1, 8, 8);
    PIPE_BARRIER(V);

    muls_v<ArchType::ASCEND_V220, float>(
        ub_buf.pp_buf_for_real_imag_fp32, ub_buf.buf_for_real_part_fp32, alpha_imag, all_repeat_times, 1, 1, 8, 8);
    PIPE_BARRIER(V);

    muls_v<ArchType::ASCEND_V220, float>(
        ub_buf.buf_for_imag_real_fp32, ub_buf.buf_for_imag_part_fp32, alpha_real, all_repeat_times, 1, 1, 8, 8);
    PIPE_BARRIER(V);

    muls_v<ArchType::ASCEND_V220, float>(
        ub_buf.buf_for_imag_imag_fp32, ub_buf.buf_for_imag_part_fp32, (-alpha_imag), all_repeat_times, 1, 1, 8, 8);
    PIPE_BARRIER(V);

    add_v<ArchType::ASCEND_V220, float>(
        ub_buf.buf_for_final_imag_fp32, ub_buf.pp_buf_for_real_imag_fp32, ub_buf.buf_for_imag_real_fp32,
        all_repeat_times, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);

    add_v<ArchType::ASCEND_V220, float>(
        ub_buf.buf_for_final_real_fp32, ub_buf.pp_buf_for_real_real_fp32, ub_buf.buf_for_imag_imag_fp32,
        all_repeat_times, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);

    vgather(
        reinterpret_cast<__ubuf__ uint32_t*>((ub_buf.pp_buf_for_final_complex_fp32).GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(ub_buf.buf_for_mask_fp32.GetPhyAddr()), 0, 8, all_repeat_times_complex);
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> mask_addr = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::Gather(
        ub_buf.pp_buf_for_final_complex_fp32, mask_addr, ub_buf.buf_for_mask_fp32, (uint32_t)(0),
        all_repeat_times_complex * 64);
    PIPE_BARRIER(V);

    SET_FLAG(V, MTE3, event_id);
    WAIT_FLAG(V, MTE3, event_id);

    ub_to_gm<ArchType::ASCEND_V220, float>(
        vector_y[param.data_offset * 2], ub_buf.pp_buf_for_final_complex_fp32, 0, 1,
        param.data_count / COMPLEX_DATA_NUM_PER_BLOCK, 0, 0);

    SET_FLAG(MTE3, MTE2, event_id);
}

__aicore__ __inline__ void process_cacl(
    LocalInfo local_info, UB_FOR_CACL ub_buf, float alpha_real, float alpha_imag, AscendC::GlobalTensor<float> vector_x,
    AscendC::GlobalTensor<float> vector_y, AscendC::GlobalTensor<float> workspace)
{
    int32_t cur_data_per_proc = MAX_LENG_PER_UB_PROC;

    if (local_info.element_count <= 256) {
        Param pp_param;

        pp_param.data_offset = local_info.start_pos;
        pp_param.data_count = local_info.element_count;

        SET_FLAG(MTE3, MTE2, EVENT_ID0);

        _cacl(pp_param, 0, ub_buf, alpha_real, alpha_imag, vector_x, vector_y, workspace, 0);

        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    } else if (local_info.element_count <= cur_data_per_proc * 2) {
        Param pp_param[2];

        pp_param[0].data_offset = local_info.start_pos;
        pp_param[0].data_count = (local_info.element_count / 2 + 3) / 4 * 4;

        pp_param[1].data_offset = local_info.start_pos + pp_param[0].data_count;
        pp_param[1].data_count = local_info.element_count - pp_param[0].data_count;

        int32_t ping_pong_flag = 0;

        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        for (int32_t loop_time = 0; loop_time < 2; loop_time++) {
            _cacl(
                pp_param[ping_pong_flag], ping_pong_flag, ub_buf, alpha_real, alpha_imag, vector_x, vector_y, workspace,
                loop_time);

            ping_pong_flag = 1 - ping_pong_flag;
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    } else {
        Param pp_param[2];

        int32_t ping_pong_times = local_info.element_count / cur_data_per_proc;
        int32_t ping_pong_tail = local_info.element_count % cur_data_per_proc;

        if (ping_pong_tail > 0) {
            ping_pong_times += 1;
        }

        pp_param[0].data_offset = local_info.start_pos;
        pp_param[0].data_count = cur_data_per_proc;

        pp_param[1].data_offset = local_info.start_pos + cur_data_per_proc;
        pp_param[1].data_count = cur_data_per_proc;

        int32_t ping_pong_flag = 0;

        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        for (int32_t loop = 0; loop < ping_pong_times; loop++) {
            if (loop == ping_pong_times - 1 && ping_pong_tail > 0) {
                pp_param[ping_pong_flag].data_count = ping_pong_tail;

                _cacl(
                    pp_param[ping_pong_flag], ping_pong_flag, ub_buf, alpha_real, alpha_imag, vector_x, vector_y,
                    workspace, loop);

                continue;
            }

            _cacl(
                pp_param[ping_pong_flag], ping_pong_flag, ub_buf, alpha_real, alpha_imag, vector_x, vector_y, workspace,
                loop);

            pp_param[ping_pong_flag].data_offset += cur_data_per_proc * 2;

            ping_pong_flag = 1 - ping_pong_flag;
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    }
}

// 尾部残部（非 4 复数块粒度）标量处理：GM 标量读 + 32B 对齐 UB 槽 + 8 字节
// DataCopyPad 写回（!720 已验证模式），由 0 号核在块管线完成后追加执行。
__aicore__ __inline__ void cscal_process_tail_scalar(
    AscendC::GlobalTensor<float> gm_x, int32_t startElem, int32_t count, float alpha_r, float alpha_i)
{
    if (count <= 0) {
        return;
    }
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> slot = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::DataCopyExtParams outParams{1, 8, 0, 0, 0};
    for (int32_t i = 0; i < count; i++) {
        const uint64_t f = static_cast<uint64_t>(startElem + i) * 2;
        const float xr = gm_x.GetValue(f);
        const float xi = gm_x.GetValue(f + 1);
        slot.SetValue(0, alpha_r * xr - alpha_i * xi);
        slot.SetValue(1, alpha_r * xi + alpha_i * xr);
        // UB 标量写不经 V 管道，与 MTE3 读之间需全屏障
        PIPE_BARRIER(ALL);
        DataCopyPad(gm_x[f], slot, outParams);
        PIPE_BARRIER(ALL);
    }
}

// 非连续路径（incx != 1）：按逻辑下标步进寻址（x[(start+i)*incx]），每元素独立
// 复数缩放；GM 标量读 + 32B 对齐 UB 槽批量旋转 + 8 字节 DataCopyPad 写回。
__aicore__ __inline__ void cscal_process_strided_aiv(
    AscendC::GlobalTensor<float> gm_x, int32_t n, int32_t incx, float alpha_r, float alpha_i)
{
    const int16_t aiv_id = AscendC::GetBlockIdx();
    int16_t aiv_num = AscendC::GetBlockNum();
    if (aiv_num == 0) {
        aiv_num = 1;
    }
    int32_t elemsPerCore = n / aiv_num;
    const int32_t elemsRemain = n % aiv_num;
    int32_t startElem = aiv_id * elemsPerCore;
    if (aiv_id < elemsRemain) {
        startElem += aiv_id;
        elemsPerCore += 1;
    } else {
        startElem += elemsRemain;
    }
    if (elemsPerCore <= 0) {
        return;
    }

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> slot = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::DataCopyExtParams outParams{1, 8, 0, 0, 0};
    constexpr int32_t CSCAL_STRIDED_ELEMS_PER_PASS = 1024;
    for (int32_t base = 0; base < elemsPerCore; base += CSCAL_STRIDED_ELEMS_PER_PASS) {
        const int32_t passElems =
            (elemsPerCore - base) < CSCAL_STRIDED_ELEMS_PER_PASS ? (elemsPerCore - base) : CSCAL_STRIDED_ELEMS_PER_PASS;
        for (int32_t i = 0; i < passElems; i++) {
            const uint64_t f = static_cast<uint64_t>(startElem + base + i) * incx * 2;
            const float xr = gm_x.GetValue(f);
            const float xi = gm_x.GetValue(f + 1);
            slot.SetValue(static_cast<uint32_t>(i) * 2, alpha_r * xr - alpha_i * xi);
            slot.SetValue(static_cast<uint32_t>(i) * 2 + 1, alpha_r * xi + alpha_i * xr);
        }
        // UB 标量写不经 V 管道，与 MTE3 读之间需全屏障
        PIPE_BARRIER(ALL);
        for (int32_t i = 0; i < passElems; i++) {
            const uint64_t f = static_cast<uint64_t>(startElem + base + i) * incx * 2;
            DataCopyPad(gm_x[f], slot[static_cast<uint32_t>(i) * 2], outParams);
        }
        PIPE_BARRIER(ALL);
    }
}

__aicore__ __inline__ void cscal_aiv(GM_ADDR x, GM_ADDR maskBuf_device, GM_ADDR workspace, GM_ADDR tiling_para_gm)
{
    AscendC::SetMaskNorm();
    SetAtomicnone();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    auto tiling_para = reinterpret_cast<__gm__ uint8_t*>(tiling_para_gm);

    int32_t vector_len = (*(__gm__ int32_t*)((__gm__ uint8_t*)tiling_para));
    int32_t incx = (*(__gm__ int32_t*)((__gm__ uint8_t*)tiling_para + 4));
    float alpha_r = (*(__gm__ float*)((__gm__ uint8_t*)tiling_para + 8));
    float alpha_i = (*(__gm__ float*)((__gm__ uint8_t*)tiling_para + 12));

    AscendC::GlobalTensor<float> x_tensor;
    AscendC::GlobalTensor<float> workspace_tensor;
    AscendC::GlobalTensor<uint32_t> maskBuf_device_tensor;

    x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    workspace_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    maskBuf_device_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(maskBuf_device));

    if (incx != 1) {
        cscal_process_strided_aiv(x_tensor, vector_len, incx, alpha_r, alpha_i);
        return;
    }

    // 连续路径：4 复数块对齐部分走原块管线（划分对 4 倍数长度无越界），
    // 残部（n % 4，最多 3 个复数）由 0 号核在主处理后追加标量处理
    const int32_t aligned_len = vector_len / COMPLEX_DATA_NUM_PER_BLOCK * COMPLEX_DATA_NUM_PER_BLOCK;

    UB_FOR_CACL ub_buf;
    allocate_ubuf(&ub_buf);

    gm_to_ub<ArchType::ASCEND_V220, uint32_t>(
        ub_buf.buf_for_mask_fp32, maskBuf_device_tensor, 0, 1, MAX_LENG_PER_UB_PROC / COMPLEX_DATA_NUM_PER_BLOCK, 0, 0);

    SET_FLAG(MTE2, V, EVENT_ID0);
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID1);

    LocalInfo local_info;
    get_local_info(&local_info, aligned_len);

    if (local_info.start_pos != -1) {
        process_cacl(local_info, ub_buf, alpha_r, alpha_i, x_tensor, x_tensor, workspace_tensor);
    }
    if (AscendC::GetBlockIdx() == 0) {
        cscal_process_tail_scalar(x_tensor, aligned_len, vector_len - aligned_len, alpha_r, alpha_i);
    }
}

#endif

extern "C" __global__ __aicore__ __vector__ void cscal(
    GM_ADDR x, GM_ADDR maskBuf_device, GM_ADDR workspace, GM_ADDR tiling_para_gm)
{
#ifdef __DAV_C220_VEC__
    cscal_aiv(x, maskBuf_device, workspace, tiling_para_gm);
#endif
}

void cscal_kernel_do(GM_ADDR x, GM_ADDR maskBuf, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    cscal<<<numBlocks, nullptr, stream>>>(x, maskBuf, workSpace, tilingGm);
}