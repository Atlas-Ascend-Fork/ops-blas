/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "ascblasCgemv_utils.h"

#define MAXALIGN_WITH(a, b) (((a) + (b)-1) / (b)) * (b)

extern "C" __global__ __aicore__ __vector__ void cgemv_do_trans(
    __gm__ uint8_t *__restrict__ d_A, __gm__ uint8_t *__restrict__ d_x,
    __gm__ uint8_t *__restrict__ d_y_in, __gm__ uint8_t *__restrict__ maskBuf_device,
    __gm__ uint8_t *__restrict__ d_y, __gm__ uint8_t *__restrict__ workspace,
    __gm__ uint8_t *__restrict__ tiling_para_gm)
{
#if __DAV_C220_VEC__
    SetAtomicnone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    AscendC::GlobalTensor<float> d_A_tensor;
    AscendC::GlobalTensor<float> d_x_tensor;
    AscendC::GlobalTensor<float> d_y_tensor;
    AscendC::GlobalTensor<uint32_t> maskBuf_device_tensor;

    d_A_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_A));
    d_x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_x));
    d_y_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_y));
    maskBuf_device_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(maskBuf_device));

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_para_gm);

    int64_t _transA = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf));
    int64_t M = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 8));
    int64_t N = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 16));
    int64_t lda = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 24));
    int64_t incx = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 32));
    int64_t incy = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 40));
    int64_t sectionDim = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 48));
    float alpha_r = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 56));
    float alpha_i = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 60));
    float beta_r = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 64));
    float beta_i = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 68));

    PIPE_BARRIER(ALL);

    aclblasComplex alpha{alpha_r, alpha_i};
    aclblasComplex beta{beta_r, beta_i};
    const bool conjugateA = (_transA == ACLBLAS_OP_C);

    uint64_t UB_NOW = 0;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> Ybase_r = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 4 * 1024;
    AscendC::LocalTensor<float> Ybase_i = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 4 * 1024;
    AscendC::LocalTensor<float> Ytemp = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 8 * 1024;

    AscendC::LocalTensor<float> Abase_r = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 32 * 1024;
    AscendC::LocalTensor<float> Abase_i = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 32 * 1024;

    AscendC::LocalTensor<float> Xbase_r = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 4 * 1024;
    AscendC::LocalTensor<float> Xbase_i = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 4 * 1024;
    AscendC::LocalTensor<float> Xtemp = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 8 * 1024;

    AscendC::LocalTensor<float> negXbase_i = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 4 * 1024;

    AscendC::LocalTensor<uint32_t> maskBuf = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_NOW);
    UB_NOW += 8 * 1024;

    AscendC::LocalTensor<float> tmpbuf_r = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 8 * 1024;
    AscendC::LocalTensor<float> tmpbuf_i = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 8 * 1024;

    AscendC::LocalTensor<float> tmpbuf = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 64 * 1024;

    int64_t szHorz = 512;
    int64_t szVert = 16;

    int64_t szX = szHorz;
    int64_t szY = szVert;

    int64_t numVert = (N + szVert - 1) / szVert;
    int64_t numHorz = (M + szHorz - 1) / szHorz;

    int64_t idxHorz, idxVert;
    int64_t szHorz_real, szHorz_real_pad;
    int64_t szVert_real;

    int64_t numSlide, idxSlide;
    int64_t slide_offset = 0;

    AscendC::GlobalTensor<float> y_gmptr = d_y_tensor;
    AscendC::GlobalTensor<float> x_gmptr = d_x_tensor;
    AscendC::GlobalTensor<float> A_gmptr = d_A_tensor;

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    for (idxVert = 0; idxVert < numVert; idxVert++) {
        int64_t aiv_id = AscendC::GetBlockIdx();
        int64_t aiv_num = AscendC::GetBlockNum();

        if (idxVert % aiv_num != aiv_id)
            continue;

        szVert_real = (idxVert == numVert - 1) ? N - idxVert * szVert : szVert;

        y_gmptr = d_y_tensor[idxVert * szVert * 2 * incy];

        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        cvec_gm2ubuf(Ybase_r, Ybase_i, Ytemp, y_gmptr, szVert_real, incy);

        PIPE_BARRIER(ALL);
        if (incy == 1) {
            PIPE_BARRIER(V);
        } else {
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
        }
        mulSComplex(Ybase_r, Ybase_i, tmpbuf_r, tmpbuf_i, MAXALIGN_WITH(szVert_real, BLOCK_SIZE) / BLOCK_SIZE, beta);

        PIPE_BARRIER(V);

        SET_FLAG(V, MTE2, EVENT_ID0);
        for (idxHorz = 0; idxHorz < numHorz; idxHorz++) {
            szHorz_real = (idxHorz == numHorz - 1) ? M - idxHorz * szHorz : szHorz;
            szHorz_real_pad = MAXALIGN_WITH(szHorz_real, VEC_MIN_SIZE);
            x_gmptr = d_x_tensor[idxHorz * szHorz * 2 * incx];
            A_gmptr = d_A_tensor[idxHorz * szHorz * 2 + idxVert * szVert * lda * 2];

            WAIT_FLAG(V, MTE2, EVENT_ID0);
            matrix_gm2ubuf(tmpbuf, A_gmptr, szHorz_real * 2, szVert_real, lda * 2, szHorz_real_pad * 2);

            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            split_CVector(Abase_r, Abase_i, tmpbuf, szHorz_real_pad * szVert_real);

            if (conjugateA) {
                muls_v<ArchType::ASCEND_V220, float>(
                    Abase_i, Abase_i, -1.0f,
                    MAXALIGN_WITH(szHorz_real_pad * szVert_real, VEC_MIN_SIZE) / VEC_MIN_SIZE,
                    1, 1, 8, 8);
                PIPE_BARRIER(V);
            }

            cvec_gm2ubuf(Xbase_r, Xbase_i, Xtemp, x_gmptr, szHorz_real, incx);

            if (incx == 1)
                PIPE_BARRIER(V);
            else {
                SET_FLAG(S, V, EVENT_ID0);
                WAIT_FLAG(S, V, EVENT_ID0);
            }

            mulSComplex(Xbase_r, Xbase_i, tmpbuf_r, tmpbuf_i, MAXALIGN_WITH(szHorz_real, BLOCK_SIZE) / BLOCK_SIZE, alpha);
            PIPE_BARRIER(V);

            muls_v<ArchType::ASCEND_V220, float>(
                negXbase_i,
                Xbase_i,
                -1, MAXALIGN_WITH(szHorz_real, VEC_MIN_SIZE) / VEC_MIN_SIZE, 1, 1, 8, 8);
            PIPE_BARRIER(V);

            SET_FLAG(V, MTE2, EVENT_ID0);

            numSlide = MAXALIGN_WITH(szHorz_real, VEC_MIN_SIZE) / VEC_MIN_SIZE;
            for (idxSlide = 0; idxSlide < numSlide; idxSlide++) {
                slide_offset = idxSlide * VEC_MIN_SIZE;

                if (idxSlide == numSlide - 1) {
                    int64_t szSlideRemain = szHorz_real - idxSlide * VEC_MIN_SIZE;
                    uint64_t mask = -1;
                    mask >>= VEC_MIN_SIZE - szSlideRemain;
                    AscendC::SetMaskNorm();
                    AscendC::SetVectorMask<float>((uint64_t)0, mask);
                }

                if (idxSlide == 0) {
                    mul_v<ArchType::ASCEND_V220, float>(
                        tmpbuf_r,
                        Xbase_r,
                        Abase_r,
                        szVert_real,
                        1,
                        1,
                        1,
                        8,
                        0,
                        szHorz_real_pad / BLOCK_SIZE);

                    mul_v<ArchType::ASCEND_V220, float>(
                        tmpbuf_i,
                        Xbase_r,
                        Abase_i,
                        szVert_real,
                        1,
                        1,
                        1,
                        8,
                        0,
                        szHorz_real_pad / BLOCK_SIZE);
                } else {
                    vmla(
                    (__ubuf__ float*)tmpbuf_r.GetPhyAddr(),
                    (__ubuf__ float*)Xbase_r[slide_offset].GetPhyAddr(),
                    (__ubuf__ float*)Abase_r[slide_offset].GetPhyAddr(), szVert_real, 1, 1, 1, 8, 0,
                    szHorz_real_pad / BLOCK_SIZE);

                    vmla(
                    (__ubuf__ float*)tmpbuf_i.GetPhyAddr(),
                    (__ubuf__ float*)Xbase_r[slide_offset].GetPhyAddr(),
                    (__ubuf__ float*)Abase_i[slide_offset].GetPhyAddr(), szVert_real, 1, 1, 1, 8, 0,
                    szHorz_real_pad / BLOCK_SIZE);
                }
                PIPE_BARRIER(V);

                vmla(
                    (__ubuf__ float*)tmpbuf_r.GetPhyAddr(),
                    (__ubuf__ float*)negXbase_i[slide_offset].GetPhyAddr(),
                    (__ubuf__ float*)Abase_i[slide_offset].GetPhyAddr(), szVert_real, 1, 1, 1, 8, 0,
                    szHorz_real_pad / BLOCK_SIZE);

                vmla(
                    (__ubuf__ float*)tmpbuf_i.GetPhyAddr(),
                    (__ubuf__ float*)Xbase_i[slide_offset].GetPhyAddr(),
                    (__ubuf__ float*)Abase_r[slide_offset].GetPhyAddr(), szVert_real, 1, 1, 1, 8, 0,
                    szHorz_real_pad / BLOCK_SIZE);
                PIPE_BARRIER(V);
            }

            if (numSlide != 1) {
                AscendC::SetMaskNorm();
                AscendC::SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
            }

            cadd_v<ArchType::ASCEND_V220, float>(
                tmpbuf_r,
                tmpbuf_r,
                szVert_real,
                1,
                1,
                8);
            cadd_v<ArchType::ASCEND_V220, float>(
                tmpbuf_i,
                tmpbuf_i,
                szVert_real,
                1,
                1,
                8);
            PIPE_BARRIER(V);

            AscendC::SetMaskNorm();
            AscendC::SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

            AscendC::SetMaskCount();
            AscendC::SetVectorMask<float>((uint64_t)0, (uint64_t)szVert_real);

            add_v<ArchType::ASCEND_V220, float>(
                Ybase_r,
                Ybase_r,
                tmpbuf_r,
                1,
                1,
                1,
                1,
                8,
                8,
                8);

            add_v<ArchType::ASCEND_V220, float>(Ybase_i, Ybase_i, tmpbuf_i, 1, 1, 1, 1, 8, 8, 8);

            AscendC::SetMaskNorm();
            AscendC::SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
        }
        WAIT_FLAG(V, MTE2, EVENT_ID0);

        gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
            maskBuf,
            maskBuf_device_tensor,
            0, 1, MAXALIGN_WITH(szVert_real * 2 * sizeof(uint32_t), 256), 0, 0, 0, 0);
        if (incy == 1) {
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            PIPE_BARRIER(V);
        } else {
            SET_FLAG(MTE2, S, EVENT_ID0);
            WAIT_FLAG(MTE2, S, EVENT_ID0);
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
        }

        cvec_ubuf2gm(Ybase_r, Ybase_i, maskBuf, Ytemp, y_gmptr, szVert_real, incy, false);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
#endif
}

void cgemv_do_trans_kernel_do(GM_ADDR d_A, GM_ADDR d_x, GM_ADDR d_y_in, GM_ADDR maskBuf, GM_ADDR d_y,
                              GM_ADDR workSpace, GM_ADDR tilingGm,
                              uint32_t numBlocks, void *stream)
{
    cgemv_do_trans<<<numBlocks, nullptr, stream>>>(d_A, d_x, d_y_in, maskBuf, d_y, workSpace, tilingGm);
}