/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

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
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 32 / sizeof(float);
constexpr int64_t VEC_MIN_SIZE = 256 / sizeof(float);
constexpr int64_t NUM_ELE_PERBLOCK = 32 / sizeof(float);
constexpr int64_t UINT16_STRIDE_LIMIT = 65536;

#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#define MAXALIGN_WITH(a, b) ((((a) + (b) - 1) / (b)) * (b))
#define ROUND(num, paddingNum) ((((num) + (paddingNum) - 1) / (paddingNum) * (paddingNum)))

#if __DAV_C220_VEC__
__aicore__ __inline__ __attribute__((always_inline)) void matrix_gm2ubuf(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    int64_t mActual, int64_t nActual, size_t srcStride, size_t dstStride)
{
    int64_t mEound = ROUND(mActual, NUM_ELE_PERBLOCK);
    if (mActual % NUM_ELE_PERBLOCK == 0 && srcStride % NUM_ELE_PERBLOCK == 0 && srcStride < UINT16_STRIDE_LIMIT) {
        gm_to_ub<ArchType::ASCEND_V220, float>(
            dst, src, 0, nActual, mEound / NUM_ELE_PERBLOCK, (srcStride - mEound) / NUM_ELE_PERBLOCK,
            (dstStride - mEound) / NUM_ELE_PERBLOCK);
    } else if (mActual % NUM_ELE_PERBLOCK == 0 && srcStride * NUM_ELE_PERBLOCK < UINT16_STRIDE_LIMIT) {
        int C0_SIZE_loop = nActual / NUM_ELE_PERBLOCK;
        int C0_SIZE_remain = nActual % NUM_ELE_PERBLOCK;
        if (C0_SIZE_loop > 0) {
            for (int i = 0; i < NUM_ELE_PERBLOCK; i++) {
                gm_to_ub<ArchType::ASCEND_V220, float>(
                    dst[i * dstStride], src[i * srcStride], 0, C0_SIZE_loop, mEound / NUM_ELE_PERBLOCK,
                    (srcStride * NUM_ELE_PERBLOCK - mEound) / NUM_ELE_PERBLOCK,
                    (dstStride * NUM_ELE_PERBLOCK - mEound) / NUM_ELE_PERBLOCK);
            }
        }
        for (int i = 0; i < C0_SIZE_remain; i++) {
            gm_to_ub<ArchType::ASCEND_V220, float>(
                dst[C0_SIZE_loop * NUM_ELE_PERBLOCK * dstStride + i * dstStride],
                src[C0_SIZE_loop * NUM_ELE_PERBLOCK * srcStride + i * srcStride],
                0, 1, mEound / NUM_ELE_PERBLOCK, 0, 0);
        }
    } else {
        for (int i = 0; i < nActual; i++) {
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                dst[i * dstStride], src[i * srcStride], 0, 1, mActual * sizeof(float), 0, 0, 0, 0);
        }
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void split_CVector(
    AscendC::LocalTensor<float> dst_r,
    AscendC::LocalTensor<float> dst_i,
    AscendC::LocalTensor<float> src,
    int64_t num_complex)
{
    int64_t num_repeats = (num_complex * 2 + 63) / 64;

    vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(dst_r.GetPhyAddr()),
              reinterpret_cast<__ubuf__ uint32_t *>(src.GetPhyAddr()),
              nullptr,
              num_repeats,
              1,
              1,
              8, 8);

    vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(dst_i.GetPhyAddr()),
              reinterpret_cast<__ubuf__ uint32_t *>(src.GetPhyAddr()),
              nullptr,
              num_repeats,
              1,
              2,
              8, 8);
}

__aicore__ __inline__ __attribute__((always_inline)) void cvec_gm2ubuf(
    AscendC::LocalTensor<float> dst_r,
    AscendC::LocalTensor<float> dst_i,
    AscendC::LocalTensor<float> tmpbuf,
    AscendC::GlobalTensor<float> src,
    int64_t X,
    int64_t ldx)
{
    if (ldx == 1) {
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            tmpbuf,
            src,
            0,
            1,
            X * 2 * sizeof(float),
            0, 0, 0, 0);

        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        split_CVector(dst_r, dst_i, tmpbuf, X);
    } else {
        int64_t temp_size = MIN(BLOCK_SIZE, ldx * 2);
        matrix_gm2ubuf(
            tmpbuf,
            src,
            temp_size,
            X,
            ldx * 2,
            BLOCK_SIZE);
        SET_FLAG(MTE2, S, EVENT_ID0);

        WAIT_FLAG(MTE2, S, EVENT_ID0);
        for (int64_t i = 0; i < X; i++) {
            dst_r[i] = tmpbuf[i * BLOCK_SIZE];
            dst_i[i] = tmpbuf[i * BLOCK_SIZE + 1];
        }
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void cvec_ubuf2gm(
    AscendC::LocalTensor<float> Ybase_r,
    AscendC::LocalTensor<float> Ybase_i,
    AscendC::LocalTensor<uint32_t> maskBuf,
    AscendC::LocalTensor<float> tmpbuf,
    AscendC::GlobalTensor<float> y_gmptr,
    int64_t szVert_real, int64_t incy, bool atomicAdd)
{
    if (atomicAdd) {
        AscendC::SetAtomicAdd<float>();
    }

    if (incy == 1) {
        int64_t nRepeats = (szVert_real * sizeof(float) * 2 + 255) / 256;
        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<float> mask_addr = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
        AscendC::Gather(tmpbuf, mask_addr, maskBuf, (uint32_t)0, nRepeats * 64);
        SET_FLAG(V, MTE3, EVENT_ID0);

        WAIT_FLAG(V, MTE3, EVENT_ID0);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            y_gmptr, tmpbuf, 0, 1, szVert_real * 2 * sizeof(float), 0, 0, 0, 0);
    } else {
        for (int64_t i = 0; i < szVert_real; i++) {
            tmpbuf[i * BLOCK_SIZE] = Ybase_r[i];
            tmpbuf[i * BLOCK_SIZE + 1] = Ybase_i[i];
        }
        SET_FLAG(S, MTE3, EVENT_ID0);

        WAIT_FLAG(S, MTE3, EVENT_ID0);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            y_gmptr, tmpbuf, 0, szVert_real, 2 * sizeof(float), 0, 0, 0, (2 * incy - 2) * sizeof(float));
    }

    SetAtomicnone();
}

__aicore__ __inline__ __attribute__((always_inline)) void mulSComplex(
    AscendC::LocalTensor<float> src_r,
    AscendC::LocalTensor<float> src_i,
    AscendC::LocalTensor<float> tmp_r,
    AscendC::LocalTensor<float> tmp_i,
    int64_t n_blocks, aclblasComplex beta)
{
    muls_v<ArchType::ASCEND_V220, float>(tmp_r, src_r, beta.real, (n_blocks + 7) / 8, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(tmp_i, src_i, beta.imag, (n_blocks + 7) / 8, 1, 1, 8, 8);
    PIPE_BARRIER(V);

    sub_v<ArchType::ASCEND_V220, float>(tmp_r, tmp_r, tmp_i, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
    muls_v<ArchType::ASCEND_V220, float>(tmp_i, src_r, beta.imag, (n_blocks + 7) / 8, 1, 1, 8, 8);
    PIPE_BARRIER(V);

    ub_to_ub<ArchType::ASCEND_V220, float>(src_r, tmp_r, 0, 1, n_blocks, 0, 0);
    PIPE_BARRIER(V);

    muls_v<ArchType::ASCEND_V220, float>(tmp_r, src_i, beta.real, (n_blocks + 7) / 8, 1, 1, 8, 8);
    PIPE_BARRIER(V);

    add_v<ArchType::ASCEND_V220, float>(tmp_i, tmp_i, tmp_r, (n_blocks + 7) / 8, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);

    ub_to_ub<ArchType::ASCEND_V220, float>(src_i, tmp_i, 0, 1, n_blocks, 0, 0);
}
#endif

