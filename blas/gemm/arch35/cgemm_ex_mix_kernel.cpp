/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#define CGEMM_EX_CUBE3_SKIP_ENTRY
#include "cgemm_ex_cube3_kernel.cpp"

namespace cgemm_ex_kern {
constexpr int32_t TILE = 2048;
constexpr int32_t BUF_NUM = 2;

#include "cgemm_ex_combine_impl.h"
#include "cgemm_ex_combine_impl_tail.h"
using namespace AscendC;

} // namespace cgemm_ex_kern

using namespace AscendC;

constexpr uint16_t MIX_CUBE_DONE_FLAG = 8;

__aicore__ inline bool MixFlagPerTile(const GemmTilingData& tiling, int32_t m)
{
    (void)m;
    if (tiling.mBlocks > 1) {
        return false;
    }
    if (tiling.k <= 96) {
        return true;
    }
    return false;
}

__aicore__ inline void MixCombineSetup(
    __gm__ uint8_t* t1, __gm__ uint8_t* t2, __gm__ uint8_t* t3, __gm__ uint8_t* cOut, int32_t m, int32_t n,
    int32_t tempLdc, int32_t ldc, float ar, float ai, float br, float bi, int32_t cIsComplex,
    const GemmTilingData& tiling, AscendC::TPipe& pipe, cgemm_ex_kern::Combine3Kernel& op, uint32_t& used,
    uint32_t& divN, int32_t& bm, int32_t& bn, int32_t& cubeM, int32_t& cubeN)
{
    used = static_cast<uint32_t>(tiling.usedCoreNum);
    if (used < 1) {
        used = 1;
    }
    bm = tiling.baseM > 0 ? tiling.baseM : 256;
    bn = tiling.baseN > 0 ? tiling.baseN : 256;
    cubeM = tiling.m;
    cubeN = tiling.n;
    if (cubeM < 1) {
        cubeM = n;
    }
    if (cubeN < 1) {
        cubeN = m;
    }
    divN = static_cast<uint32_t>((cubeN + bn - 1) / bn);
    int32_t kParts = tiling.mBlocks;
    op.Init(t1, t2, t3, tempLdc, cOut, m, n, ldc, ar, ai, br, bi, cIsComplex, kParts, &pipe, 1, 0);
    op.SetupBuf();
}

__aicore__ inline void MixCombineRun(
    cgemm_ex_kern::Combine3Kernel& op, uint32_t aicBlk, uint32_t used, uint32_t divN, int32_t bm, int32_t bn,
    int32_t cubeM, int32_t cubeN, bool waitPerTile)
{
    uint32_t divM = static_cast<uint32_t>((cubeM + bm - 1) / bm);
    uint32_t total = divM * divN;
    uint64_t m64 = static_cast<uint64_t>(cubeM);
    uint64_t n64 = static_cast<uint64_t>(cubeN);
    uint64_t BM = static_cast<uint64_t>(bm);
    uint64_t BN = static_cast<uint64_t>(bn);
    for (uint32_t idx = aicBlk; idx < total; idx += used) {
        if (waitPerTile) {
            CrossCoreWaitFlag<2, PIPE_MTE2>(MIX_CUBE_DONE_FLAG);
        }
        uint64_t mOff = 0;
        uint64_t nOff = 0;
        uint64_t curM = 0;
        uint64_t curN = 0;
        cgemm_ex_blaze3::DecodeMnTile(idx, divN, BM, BN, m64, n64, mOff, nOff, curM, curN);
        op.ProcessRect(
            static_cast<int32_t>(nOff), static_cast<int32_t>(nOff + curN), static_cast<int32_t>(mOff),
            static_cast<int32_t>(mOff + curM));
    }
}

__aicore__ inline void MixOwnedCombine(
    __gm__ uint8_t* t1, __gm__ uint8_t* t2, __gm__ uint8_t* t3, __gm__ uint8_t* cOut, int32_t m, int32_t n,
    int32_t tempLdc, int32_t ldc, float ar, float ai, float br, float bi, int32_t cIsComplex,
    const GemmTilingData& tiling)
{
    uint32_t used = 1;
    uint32_t divN = 0;
    int32_t bm = 256;
    int32_t bn = 256;
    int32_t cubeM = n;
    int32_t cubeN = m;
    AscendC::TPipe pipe;
    cgemm_ex_kern::Combine3Kernel op;
    MixCombineSetup(
        t1, t2, t3, cOut, m, n, tempLdc, ldc, ar, ai, br, bi, cIsComplex, tiling, pipe, op, used, divN, bm, bn, cubeM,
        cubeN);
    uint32_t aicBlk = GetBlockIdx();
    if (aicBlk >= used || divN == 0) {
        return;
    }
    MixCombineRun(op, aicBlk, used, divN, bm, bn, cubeM, cubeN, MixFlagPerTile(tiling, m));
}

extern "C" __global__ __aicore__ void cgemm_ex_mix_kernel(
    __gm__ uint8_t* a1, __gm__ uint8_t* b1, __gm__ uint8_t* c1, __gm__ uint8_t* a2, __gm__ uint8_t* b2,
    __gm__ uint8_t* c2, __gm__ uint8_t* a3, __gm__ uint8_t* b3, __gm__ uint8_t* c3, __gm__ uint8_t* cOut, int32_t m,
    int32_t n, int32_t tempLdc, int32_t ldc, float ar, float ai, float br, float bi, int32_t cIsComplex,
    GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if ASCEND_IS_AIC {
        InitSocState();
        SetMMLayoutTransform(true);
        bool flagPerTile = MixFlagPerTile(tiling, m);
        cgemm_ex_blaze3::CubeThree(
            reinterpret_cast<__gm__ float*>(a1), reinterpret_cast<__gm__ float*>(b1),
            reinterpret_cast<__gm__ float*>(c1), reinterpret_cast<__gm__ float*>(a2),
            reinterpret_cast<__gm__ float*>(b2), reinterpret_cast<__gm__ float*>(c2),
            reinterpret_cast<__gm__ float*>(a3), reinterpret_cast<__gm__ float*>(b3),
            reinterpret_cast<__gm__ float*>(c3), tiling, flagPerTile);
        SetMMLayoutTransform(false);
        if (!flagPerTile) {
            CrossCoreSetFlag<2, PIPE_FIX>(MIX_CUBE_DONE_FLAG);
        }
    }
    if ASCEND_IS_AIV {
        bool flagPerTile = MixFlagPerTile(tiling, m);
        if (!flagPerTile) {
            CrossCoreWaitFlag<2, PIPE_MTE2>(MIX_CUBE_DONE_FLAG);
        }
        MixOwnedCombine(c1, c2, c3, cOut, m, n, tempLdc, ldc, ar, ai, br, bi, cIsComplex, tiling);
    }
}

void cgemm_ex_mix_do(
    uint32_t numBlocks, void* stream, GM_ADDR a1, GM_ADDR b1, GM_ADDR c1, GM_ADDR a2, GM_ADDR b2, GM_ADDR c2,
    GM_ADDR a3, GM_ADDR b3, GM_ADDR c3, GM_ADDR cOut, int32_t m, int32_t n, int32_t tempLdc, int32_t ldc, float ar,
    float ai, float br, float bi, int32_t cIsComplex, const GemmTilingData& tilingData)
{
    cgemm_ex_mix_kernel<<<numBlocks, nullptr, stream>>>(
        a1, b1, c1, a2, b2, c2, a3, b3, c3, cOut, m, n, tempLdc, ldc, ar, ai, br, bi, cIsComplex, tilingData);
}

#endif
