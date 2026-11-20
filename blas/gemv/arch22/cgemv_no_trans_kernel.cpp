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

extern "C" __global__ __aicore__ __vector__ void cgemv_no_trans(
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
    AscendC::GlobalTensor<float> d_y_in_tensor;
    AscendC::GlobalTensor<uint32_t> maskBuf_device_tensor;

    d_A_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_A));
    d_x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_x));
    d_y_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_y));
    d_y_in_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_y_in));
    maskBuf_device_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(maskBuf_device));

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_para_gm);

    int64_t _transA = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf));
    int64_t M = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 8));
    int64_t N = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 16));
    int64_t lda = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 24));
    int64_t incx = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 32));
    int64_t incy = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 40));
    int64_t sectionDim = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 48));
    if (sectionDim == 0) {
        sectionDim = 1;
    }
    float alpha_r = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 56));
    float alpha_i = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 60));
    float beta_r = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 64));
    float beta_i = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 68));

    PIPE_BARRIER(ALL);

    aclblasComplex alpha{alpha_r, alpha_i};
    aclblasComplex beta{beta_r, beta_i};

    constexpr int64_t SZ_UBUF = 192 * 1024;
    constexpr int64_t SZ_PINGPONG = SZ_UBUF / 2;

    constexpr int64_t SZ_A = 64 * 1024;
    constexpr int64_t SZ_X = 1024;
    constexpr int64_t SZ_Y = 1024;

    constexpr int64_t OFFSET_RIGHT = SZ_PINGPONG / 2;

    constexpr int64_t POS_PING_BUFFER = 0;
    constexpr int64_t POS_POGN_BUFFER = 0 + SZ_PINGPONG;

    constexpr int64_t OFFSET_A = 0;
    constexpr int64_t OFFSET_X = OFFSET_A + SZ_A;
    constexpr int64_t OFFSET_Y = OFFSET_X + SZ_X;

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

    AscendC::LocalTensor<float> tmpbuf_r = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 8 * 1024;
    AscendC::LocalTensor<float> tmpbuf_i = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 8 * 1024;

    AscendC::LocalTensor<uint32_t> maskbuf = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_NOW);
    UB_NOW += 8 * 1024;

    AscendC::LocalTensor<float> tmpbuf = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_NOW);
    UB_NOW += 64 * 1024;

    constexpr int64_t szVert = 512;
    constexpr int64_t szHorz = 16;

    int64_t aiv_id = AscendC::GetBlockIdx();
    int64_t aiv_num = AscendC::GetBlockNum();

    int64_t numVert = (M + szVert - 1) / szVert;
    if (numVert == 0) {
        numVert = 1;
    }
    int64_t numHorz = (N + szHorz - 1) / szHorz;
    int64_t numSection = (numHorz + sectionDim - 1) / sectionDim;
    int64_t numLoops = numVert * numSection;

    int64_t idxLoop, idxVert, idxSection, idxHorz_base, idxHorz;
    int64_t szHorz_real, szVert_real, szVert_real_pad;
    int64_t n_blocks, actual_repeats;
    int64_t szHorz_real_next;

    bool maskbuf_loaded = false;

    AscendC::GlobalTensor<float> y_gmptr_in = d_y_in_tensor;
    AscendC::GlobalTensor<float> y_gmptr = d_y_tensor;
    AscendC::GlobalTensor<float> A_gmptr = d_A_tensor;
    AscendC::GlobalTensor<float> x_gmptr = d_x_tensor;

    AscendC::GlobalTensor<float> A_gmptr_next = d_A_tensor;
    AscendC::GlobalTensor<float> x_gmptr_next = d_x_tensor;

    int64_t idxPixel = 0;
    int64_t offset_a = 0;
    AscendC::LocalTensor<float> A_r;
    AscendC::LocalTensor<float> A_i;
    float pixel_r = 0.0f;
    float pixel_i = 0.0f;

    SET_FLAG(MTE3, MTE2, EVENT_ID1);
    for (idxLoop = 0; idxLoop < numLoops; idxLoop++) {
        idxVert = idxLoop % numVert;
        idxSection = idxLoop / numVert;

        if (idxLoop % aiv_num != aiv_id)
            continue;

        y_gmptr_in = d_y_in_tensor[idxVert * szVert * 2 * incy];
        y_gmptr = d_y_tensor[idxVert * szVert * 2 * incy];
        szVert_real = (idxVert == numVert - 1) ? (M - idxVert * szVert) : szVert;
        szVert_real_pad = (szVert_real + 63) / 64 * 64;

        if (maskbuf_loaded == false) {
            gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
                maskbuf,
                maskBuf_device_tensor,
                0, 1, MAXALIGN_WITH((szVert * sizeof(uint32_t) * 2), 256), 0, 0, 0, 0);
            maskbuf_loaded = true;
        }

        idxHorz_base = idxSection * sectionDim;
        szHorz_real_next = (idxHorz_base == numHorz - 1) ? (N - idxHorz_base * szHorz) : szHorz;
        A_gmptr_next = d_A_tensor[idxHorz_base * szHorz * 2 * lda + idxVert * szVert * 2];
        x_gmptr_next = d_x_tensor[idxHorz_base * szHorz * 2 * incx];

        matrix_gm2ubuf(
            tmpbuf,
            A_gmptr_next,
            szVert_real * 2, szHorz_real_next, lda * 2, szVert_real_pad * 2);

        if (incx == 1) {
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                Xtemp,
                x_gmptr_next,
                0,
                1,
                szHorz_real_next * 2 * sizeof(float),
                0, 0, 0, 0);
        } else {
            int64_t temp_size = MIN(BLOCK_SIZE, incx * 2);
            matrix_gm2ubuf(
                Xtemp,
                x_gmptr_next,
                temp_size,
                szHorz_real_next,
                incx * 2,
                BLOCK_SIZE);
        }

        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        if (idxSection == 0) {
            cvec_gm2ubuf(Ybase_r, Ybase_i, Ytemp, y_gmptr_in, szVert_real, incy);

            if (incy == 1) {
                PIPE_BARRIER(V);
            } else {
                SET_FLAG(S, V, EVENT_ID1);
                WAIT_FLAG(S, V, EVENT_ID1);
            }

            n_blocks = (szVert + BLOCK_SIZE - 1) / BLOCK_SIZE;
            mulSComplex(Ybase_r, Ybase_i, tmpbuf_r, tmpbuf_i, n_blocks, aclblasComplex{beta.real - 1, beta.imag});
        }

        SET_FLAG(MTE2, V, EVENT_ID0);
        for (int64_t i = 0; i < sectionDim; i++) {
            idxHorz = idxHorz_base + i;
            if (!(idxHorz < numHorz))
                break;

            szHorz_real = szHorz_real_next;
            A_gmptr = A_gmptr_next;
            x_gmptr = x_gmptr_next;

            WAIT_FLAG(MTE2, V, EVENT_ID0);
            split_CVector(Abase_r, Abase_i, tmpbuf, szVert_real_pad * szHorz_real);

            if (incx == 1) {
                split_CVector(Xbase_r, Xbase_i, Xtemp, szHorz_real);
                PIPE_BARRIER(V);
            } else {
                for (int64_t i = 0; i < szHorz_real; i++) {
                    Xbase_r[i] = Xtemp[i * BLOCK_SIZE];
                    Xbase_i[i] = Xtemp[i * BLOCK_SIZE + 1];
                }

                SET_FLAG(S, V, EVENT_ID0);
                WAIT_FLAG(S, V, EVENT_ID0);
            }

            n_blocks = (szHorz + BLOCK_SIZE - 1) / BLOCK_SIZE;
            mulSComplex(Xbase_r, Xbase_i, tmpbuf_r, tmpbuf_i, n_blocks, alpha);

            SET_FLAG(V, MTE2, EVENT_ID0);
            WAIT_FLAG(V, MTE2, EVENT_ID0);

            if ((i + 1) < sectionDim && (idxHorz + 1) < numHorz) {
                szHorz_real_next = ((idxHorz + 1) == numHorz - 1) ? (N - (idxHorz + 1) * szHorz) : szHorz;
                A_gmptr_next = d_A_tensor[(idxHorz + 1) * szHorz * 2 * lda + idxVert * szVert * 2];
                x_gmptr_next = d_x_tensor[(idxHorz + 1) * szHorz * 2 * incx];

                matrix_gm2ubuf(
                    tmpbuf,
                    A_gmptr_next,
                    szVert_real * 2, szHorz_real_next, lda * 2, szVert_real_pad * 2);

                if (incx == 1) {
                    gm_to_ub_align<ArchType::ASCEND_V220, float>(
                        Xtemp,
                        x_gmptr_next,
                        0,
                        1,
                        szHorz_real_next * 2 * sizeof(float),
                        0, 0, 0, 0);
                } else {
                    int64_t temp_size = MIN(BLOCK_SIZE, incx * 2);
                    matrix_gm2ubuf(
                        Xtemp,
                        x_gmptr_next,
                        temp_size,
                        szHorz_real_next,
                        incx * 2,
                        BLOCK_SIZE);
                }
            }
            SET_FLAG(MTE2, V, EVENT_ID0);

            SET_FLAG(V, S, EVENT_ID0);
            actual_repeats = szVert_real_pad / 64;
            for (idxPixel = 0; idxPixel < szHorz_real; idxPixel++) {
                offset_a = szVert_real_pad * idxPixel;

                WAIT_FLAG(V, S, EVENT_ID0);
                A_r = Abase_r[offset_a];
                A_i = Abase_i[offset_a];
                pixel_r = Xbase_r.GetValue(idxPixel);
                pixel_i = Xbase_i.GetValue(idxPixel);

                SET_FLAG(S, V, EVENT_ID0);
                WAIT_FLAG(S, V, EVENT_ID0);

                if (idxPixel == 0 && idxHorz == idxHorz_base && idxHorz != 0) {
                    muls_v<ArchType::ASCEND_V220, float>(Ybase_r, A_r, pixel_r, actual_repeats, 1, 1, 8, 8);
                    muls_v<ArchType::ASCEND_V220, float>(Ybase_i, A_r, pixel_i, actual_repeats, 1, 1, 8, 8);
                    PIPE_BARRIER(V);

                    muls_v<ArchType::ASCEND_V220, float>(tmpbuf_i, A_i, pixel_r, actual_repeats, 1, 1, 8, 8);
                    muls_v<ArchType::ASCEND_V220, float>(tmpbuf_r, A_i, -pixel_i, actual_repeats, 1, 1, 8, 8);
                    PIPE_BARRIER(V);

                    add_v<ArchType::ASCEND_V220, float>(Ybase_r, Ybase_r, tmpbuf_r, actual_repeats, 1, 1, 1, 8, 8, 8);
                    add_v<ArchType::ASCEND_V220, float>(Ybase_i, Ybase_i, tmpbuf_i, actual_repeats, 1, 1, 1, 8, 8, 8);
                    PIPE_BARRIER(V);
                } else {
                    AscendC::Axpy(Ybase_r, A_r, pixel_r, actual_repeats * 64);
                    AscendC::Axpy(Ybase_i, A_r, pixel_i, actual_repeats * 64);
                    PIPE_BARRIER(V);

                    AscendC::Axpy(Ybase_i, A_i, pixel_r, actual_repeats * 64);
                    AscendC::Axpy(Ybase_r, A_i, -pixel_i, actual_repeats * 64);
                    PIPE_BARRIER(V);
                }
                SET_FLAG(V, S, EVENT_ID0);
            }
            WAIT_FLAG(V, S, EVENT_ID0);
        }
        WAIT_FLAG(MTE2, V, EVENT_ID0);

        if (incy == 1) {
            PIPE_BARRIER(V);
        } else {
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
        }

        cvec_ubuf2gm(Ybase_r, Ybase_i, maskbuf, Ytemp, y_gmptr, szVert_real, incy, true);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
#endif
}

void cgemv_no_trans_kernel_do(GM_ADDR d_A, GM_ADDR d_x, GM_ADDR d_y_in, GM_ADDR maskBuf, GM_ADDR d_y,
                              GM_ADDR workSpace, GM_ADDR tilingGm,
                              uint32_t numBlocks, void *stream)
{
    cgemv_no_trans<<<numBlocks, nullptr, stream>>>(d_A, d_x, d_y_in, maskBuf, d_y, workSpace, tilingGm);
}