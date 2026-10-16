/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// FP32 pipeline adapted from this repository's ssymm arch35 implementation.
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "chemm_tiling_data.h"
constexpr int64_t B32_C0_SIZE = 8;
constexpr uint16_t TWO_ALIGN = 2;
constexpr uint16_t ZERO_FLAG = 0;
constexpr uint16_t FIRST_FLAG = 1;
constexpr uint32_t DB_COUNT = 2;
constexpr int64_t L0A_SIZE = 64 * 1024;
constexpr int64_t L0B_SIZE = 64 * 1024;
constexpr int64_t L0C_SIZE = 256 * 1024;
constexpr int64_t L1_SIZE = 512 * 1024;
constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DB_COUNT / sizeof(float);
template <typename T>
__aicore__ inline void ChemmCubeCopyInA1(
    const AscendC::GlobalTensor<T>& aGlobal, const AscendC::LocalTensor<T>& al1Local, uint64_t curML1, uint64_t curKL1,
    uint64_t lda)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curML1;
    nd2nzParams.dValue = curKL1;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = lda;
    nd2nzParams.dstNzC0Stride = RoundUp<int64_t>(curML1, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
}

template <typename T>
__aicore__ inline void ChemmCubeCopyInB1(
    const AscendC::GlobalTensor<T>& bGlobal, const AscendC::LocalTensor<T>& bl1Local, uint64_t curNL1, uint64_t curKL1,
    uint64_t ldb)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curKL1;
    nd2nzParams.dValue = curNL1;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = ldb;
    nd2nzParams.dstNzC0Stride = RoundUp<int64_t>(curKL1, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
}

template <typename T>
__aicore__ inline void ChemmCubeCopyInA2(
    const AscendC::LocalTensor<T>& al0Local, const AscendC::LocalTensor<T>& al1Local, uint64_t curML1, uint64_t curKL1,
    uint64_t mL0, uint64_t kL0)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    loadDataParams.mStep = CeilDiv<int64_t>(mL0, AscendC::BLOCK_CUBE);
    loadDataParams.kStep = CeilDiv<int64_t>(kL0, B32_C0_SIZE);
    loadDataParams.srcStride = CeilDiv<int64_t>(curML1, AscendC::BLOCK_CUBE);
    loadDataParams.dstStride = loadDataParams.mStep;
    loadDataParams.ifTranspose = false;
    AscendC::LoadData<T>(al0Local, al1Local, loadDataParams);
}

template <typename T>
__aicore__ inline void ChemmCubeCopyInB2(
    const AscendC::LocalTensor<T>& bl0Local, const AscendC::LocalTensor<T>& bl1Local, uint64_t curKL1, uint64_t curKL0,
    uint64_t nL0)
{
    AscendC::LoadData2DParamsV2 b2LoadDataParams;
    b2LoadDataParams.mStartPosition = 0;
    b2LoadDataParams.kStartPosition = 0;
    b2LoadDataParams.mStep = CeilDiv<int64_t>(curKL0, AscendC::BLOCK_CUBE);
    b2LoadDataParams.kStep = CeilDiv<int64_t>(nL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
    b2LoadDataParams.dstStride = b2LoadDataParams.kStep >> 1;
    b2LoadDataParams.srcStride = CeilDiv<int64_t>(curKL1, AscendC::BLOCK_CUBE);
    b2LoadDataParams.ifTranspose = true;
    AscendC::LoadData<T>(bl0Local, bl1Local, b2LoadDataParams);
}

template <typename T>
__aicore__ inline void ChemmCubeCopyOut(
    const AscendC::GlobalTensor<T>& tempGlobal, const AscendC::LocalTensor<float>& c1Local, uint64_t mL0, uint64_t nL0,
    uint64_t tempRowStride)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.nSize = nL0;
    intriParams.mSize = mL0;
    intriParams.dstStride = tempRowStride;
    intriParams.srcStride = RoundUp<int64_t>(mL0, AscendC::BLOCK_CUBE);
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.reluPre = 0;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(tempGlobal, c1Local, intriParams);
}

template <typename T>
__aicore__ inline void ChemmCubeLoadKChunkDB(
    const AscendC::GlobalTensor<T>& aGlobal, const AscendC::GlobalTensor<T>& bGlobal,
    const AscendC::LocalTensor<T>& l1Local, const ChemmCubeGemmTilingData& tiling, uint32_t mOff, uint32_t nOff,
    uint32_t kOff, uint32_t curK, uint64_t mL0, uint64_t nL0, uint64_t l1BufStrideA, uint64_t l1BufStrideB,
    uint64_t l1BufId)
{
    const uint64_t curML1 = mL0;
    const uint64_t curKL1 = curK;
    const uint64_t offsetAL1 = l1BufStrideA * l1BufId;
    ChemmCubeCopyInA1<T>(aGlobal[mOff * tiling.lda + kOff], l1Local[offsetAL1], curML1, curKL1, tiling.lda);

    const uint64_t offsetBL1 = l1BufStrideA * DB_COUNT + l1BufStrideB * l1BufId;
    ChemmCubeCopyInB1<T>(bGlobal[kOff * tiling.ldb + nOff], l1Local[offsetBL1], nL0, curKL1, tiling.ldb);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
}

template <typename T>
__aicore__ inline void ProcessKChunkDB(
    const AscendC::GlobalTensor<T>& aGlobal, const AscendC::GlobalTensor<T>& bGlobal,
    const AscendC::LocalTensor<T>& l1Local, const AscendC::LocalTensor<T>& al0Local,
    const AscendC::LocalTensor<T>& bl0Local, const AscendC::LocalTensor<T>& l0cLocal,
    const ChemmCubeGemmTilingData& tiling, uint32_t mOff, uint32_t nOff, uint32_t kOff, uint32_t curK, uint64_t mL0,
    uint64_t nL0, uint64_t l1BufStrideA, uint64_t l1BufStrideB, uint64_t l1BufId, uint64_t& l0PingPong)
{
    ChemmCubeLoadKChunkDB<T>(
        aGlobal, bGlobal, l1Local, tiling, mOff, nOff, kOff, curK, mL0, nL0, l1BufStrideA, l1BufStrideB, l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

    const uint64_t curML1 = mL0;
    const uint64_t curKL1 = curK;
    const uint64_t offsetAL1 = l1BufStrideA * l1BufId;
    const uint64_t offsetBL1 = l1BufStrideA * DB_COUNT + l1BufStrideB * l1BufId;
    constexpr uint64_t kStep = 32;
    uint64_t kL0TileNum = CeilDiv<int64_t>(curKL1, kStep);
    uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * kStep;
    uint64_t offsetAL0 = offsetAL1;
    uint64_t offsetBL0 = offsetBL1;
    for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
        uint64_t curKL0 = (iter1 + 1 == kL0TileNum) ? tailKL0 : kStep;
        bool isFirstLoop = (kOff % (tiling.k / tiling.splitK) == 0 && iter1 == 0);
        uint64_t l0BufId = l0PingPong & 0x1;
        uint64_t l0Offset = HALF_L0_SIZE * l0BufId;

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

        ChemmCubeCopyInA2<T>(al0Local[l0Offset], l1Local[offsetAL0], curML1, curKL1, mL0, curKL0);
        ChemmCubeCopyInB2<T>(bl0Local[l0Offset], l1Local[offsetBL0], curKL1, curKL0, nL0);

        offsetAL0 += RoundUp<int64_t>(curML1, AscendC::BLOCK_CUBE) * kStep;
        offsetBL0 += kStep * B32_C0_SIZE;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

        AscendC::MmadParams mmadParams;
        mmadParams.m = mL0;
        mmadParams.n = nL0;
        mmadParams.k = curKL0;
        mmadParams.cmatrixSource = false;
        mmadParams.cmatrixInitVal = isFirstLoop;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(l0cLocal, al0Local[l0Offset], bl0Local[l0Offset], mmadParams);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <typename T>
__aicore__ inline void ProcessChemmCubeKSlice(
    const AscendC::GlobalTensor<T>& aGlobal, const AscendC::GlobalTensor<T>& bGlobal,
    const AscendC::LocalTensor<T>& l1Local, const AscendC::LocalTensor<T>& al0Local,
    const AscendC::LocalTensor<T>& bl0Local, const AscendC::LocalTensor<T>& l0cLocal,
    const ChemmCubeGemmTilingData& tiling, uint32_t K, uint32_t slice, uint32_t mOff, uint32_t nOff, uint64_t mL0,
    uint64_t nL0, uint64_t l1BufStrideA, uint64_t l1BufStrideB, uint64_t& l1PingPong, uint64_t& l0PingPong)
{
    const uint32_t sliceK = K / tiling.splitK;
    const uint32_t sliceEnd = (slice + 1) * sliceK;
    for (uint32_t kOff = slice * sliceK; kOff < sliceEnd; kOff += tiling.tileKChunk) {
        uint32_t curK = Min<uint32_t>(tiling.tileKChunk, sliceEnd - kOff);
        uint64_t l1BufId = l1PingPong & 0x1;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        ProcessKChunkDB<T>(
            aGlobal, bGlobal, l1Local, al0Local, bl0Local, l0cLocal, tiling, mOff, nOff, kOff, curK, mL0, nL0,
            l1BufStrideA, l1BufStrideB, l1BufId, l0PingPong);
        l1PingPong++;
    }
}

template <typename T>
__aicore__ inline void ProcessChemmCubeTile(
    const AscendC::GlobalTensor<T>& aGlobal, const AscendC::GlobalTensor<T>& bGlobal,
    const AscendC::GlobalTensor<T>& tempGlobal, const AscendC::LocalTensor<T>& l1Local,
    const AscendC::LocalTensor<T>& al0Local, const AscendC::LocalTensor<T>& bl0Local,
    const AscendC::LocalTensor<T>& l0cLocal, const ChemmCubeGemmTilingData& tiling, uint32_t K, uint32_t slice,
    uint32_t mOff, uint32_t nOff, uint32_t curTileM, uint32_t curTileN, uint64_t l1BufStrideA, uint64_t l1BufStrideB,
    uint64_t& l1PingPong, uint64_t& l0PingPong)
{
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
    ProcessChemmCubeKSlice<T>(
        aGlobal, bGlobal, l1Local, al0Local, bl0Local, l0cLocal, tiling, K, slice, mOff, nOff, curTileM, curTileN,
        l1BufStrideA, l1BufStrideB, l1PingPong, l0PingPong);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
    const uint64_t offsetD = mOff * tiling.tempRowStride + nOff;
    ChemmCubeCopyOut<T>(tempGlobal[offsetD], l0cLocal, curTileM, curTileN, tiling.tempRowStride);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
}

template <typename T>
__aicore__ inline void ProcessChemmCubeWork(
    __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmTemp, AscendC::GlobalTensor<T>& aGlobal,
    AscendC::GlobalTensor<T>& bGlobal, AscendC::GlobalTensor<T>& tempGlobal, const AscendC::LocalTensor<T>& l1Local,
    const AscendC::LocalTensor<T>& al0Local, const AscendC::LocalTensor<T>& bl0Local,
    const AscendC::LocalTensor<T>& l0cLocal, const ChemmCubeGemmTilingData& tiling, uint32_t K, uint32_t divN,
    uint32_t totalTiles, uint32_t tileIdx, uint64_t l1BufStrideA, uint64_t l1BufStrideB, uint64_t& l1PingPong,
    uint64_t& l0PingPong)
{
    const uint32_t work = tileIdx / totalTiles;
    const uint32_t plane = work % tiling.products;
    const uint32_t slice = work / tiling.products;
    const uint32_t localTile = tileIdx % totalTiles;
    const uint32_t ap = tiling.products == 3 ? plane : ((plane == 1 || plane == 3) ? 1 : 0);
    const uint32_t bp = tiling.products == 3 ? plane : ((plane == 1 || plane == 2) ? 1 : 0);
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmA) + uint64_t(ap) * tiling.m * tiling.k);
    bGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(const_cast<__gm__ uint8_t*>(gmB)) + uint64_t(bp) * tiling.k * tiling.n);
    tempGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmTemp) + uint64_t(work) * tiling.m * tiling.n);
    uint32_t coreIdxM = localTile / divN;
    uint32_t coreIdxN = localTile % divN;
    if (coreIdxM % 2 == 1)
        coreIdxN = divN - 1 - coreIdxN;
    const uint32_t mStart = coreIdxM * tiling.singleCoreM;
    const uint32_t nStart = coreIdxN * tiling.singleCoreN;
    const uint32_t mEnd = Min<uint32_t>(mStart + tiling.singleCoreM, tiling.m);
    const uint32_t nEnd = Min<uint32_t>(nStart + tiling.singleCoreN, tiling.n);
    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tiling.tileM) {
        const uint32_t curTileM = Min<uint32_t>(tiling.tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tiling.tileN) {
            ProcessChemmCubeTile<T>(
                aGlobal, bGlobal, tempGlobal, l1Local, al0Local, bl0Local, l0cLocal, tiling, K, slice, mOff, nOff,
                curTileM, Min<uint32_t>(tiling.tileN, nEnd - nOff), l1BufStrideA, l1BufStrideB, l1PingPong, l0PingPong);
        }
    }
}

__global__ __aicore__ void chemm_cube_kernel(
    __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmTemp, const ChemmCubeGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const uint32_t K = tiling.k;
    AscendC::GlobalTensor<float> aGlobal, bGlobal, tempGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmA));
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmB)));
    tempGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmTemp));
    AscendC::LocalTensor<float> al0Local{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<float> bl0Local{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> l0cLocal{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<float> l1Local{AscendC::TPosition::A1, 0, L1_SIZE};
    const uint32_t curBlockIdx = AscendC::GetBlockIdx();
    const uint32_t blockNum = AscendC::GetBlockNum();
    const uint32_t divM = CeilDiv<uint32_t>(tiling.m, tiling.singleCoreM);
    const uint32_t divN = CeilDiv<uint32_t>(tiling.n, tiling.singleCoreN);
    const uint32_t totalTiles = divM * divN;
    const uint64_t l1BufStrideA = uint64_t(tiling.tileM) * uint64_t(tiling.tileKChunk);
    const uint64_t l1BufStrideB =
        RoundUp<int64_t>(uint64_t(tiling.tileKChunk), AscendC::BLOCK_CUBE) * uint64_t(tiling.tileN);
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
    for (uint32_t tileIdx = curBlockIdx; tileIdx < totalTiles * tiling.products * tiling.splitK; tileIdx += blockNum) {
        ProcessChemmCubeWork<float>(
            gmA, gmB, gmTemp, aGlobal, bGlobal, tempGlobal, l1Local, al0Local, bl0Local, l0cLocal, tiling, K, divN,
            totalTiles, tileIdx, l1BufStrideA, l1BufStrideB, l1PingPong, l0PingPong);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
}
void chemm_cube_kernel_do(
    uint8_t* gmA, const uint8_t* gmB, uint8_t* gmTemp, const ChemmCubeGemmTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    chemm_cube_kernel<<<numBlocks, nullptr, stream>>>(gmA, gmB, gmTemp, tiling);
}
