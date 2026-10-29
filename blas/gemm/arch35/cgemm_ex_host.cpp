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

#include <algorithm>
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "gemm_kernel.h"
#include "cgemm_ex_type.h"
#include "cgemm_ex_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/dtype_cast.h"
#include "common/helper/host_utils.h"

namespace {

static bool IsValidType(aclblasCgemmExType_t t)
{
    return t == ACLBLAS_R_32 || t == ACLBLAS_C_32 || t == ACLBLAS_H_R_32 || t == ACLBLAS_H_C_32;
}

static bool IsHalfType(aclblasCgemmExType_t t) { return t == ACLBLAS_H_R_32 || t == ACLBLAS_H_C_32; }

static size_t TypeElemBytes(aclblasCgemmExType_t t)
{
    if (t == ACLBLAS_C_32) {
        return 8;
    }
    if (t == ACLBLAS_H_R_32) {
        return 2;
    }
    if (t == ACLBLAS_H_C_32) {
        return 4;
    }
    return 4;
}

static bool IsZeroComplex(const aclblasComplex* z) { return z->real == 0.0f && z->imag == 0.0f; }

static int32_t CeilDivPositive(int32_t num, int32_t den)
{
    if (den <= 0) {
        return 1;
    }
    return (num + den - 1) / den;
}

static void CalcMultiCorePartition(GemmTilingData& tiling, uint32_t cubeCoreNum, int m, int n)
{
    int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    if (maxCores <= 0) {
        tiling.mBlocks = 1;
        tiling.nBlocks = 1;
        tiling.usedCoreNum = 1;
        return;
    }
    int32_t mTiles = CeilDivPositive(m, tiling.baseM);
    int32_t nTiles = CeilDivPositive(n, tiling.baseN);
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = 1;
    int32_t bestUtilization = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= maxCores; mb++) {
        int32_t nb = std::min(nTiles, maxCores / mb);
        if (nb < 1) {
            nb = 1;
        }
        int32_t utilization = mb * nb;
        if (utilization > bestUtilization && utilization <= maxCores) {
            bestUtilization = utilization;
            bestMBlocks = mb;
            bestNBlocks = nb;
        }
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    tiling.usedCoreNum = bestMBlocks * bestNBlocks;
}

static GemmTilingData MakeTiling(
    int m, int n, int k, int lda, int ldb, int ldc, aclblasOperation_t transa, aclblasOperation_t transb, float ar,
    float ai, float br, float bi)
{
    GemmTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.cLdc = ldc;
    tiling.baseM = GEMM_BASE_M;
    tiling.baseN = GEMM_BASE_N;
    tiling.baseK = GEMM_BASE_K;
    tiling.tileKChunk = GEMM_TILE_K_CHUNK;
    tiling.c0Size = GEMM_C0_SIZE;
    tiling.isTransA = (transa != ACLBLAS_OP_N) ? 1 : 0;
    tiling.isTransB = (transb != ACLBLAS_OP_N) ? 1 : 0;
    tiling.alphaReal = ar;
    tiling.alphaImag = ai;
    tiling.betaReal = br;
    tiling.betaImag = bi;
    tiling.hasBeta = (br != 0.0f || bi != 0.0f) ? 1 : 0;
    return tiling;
}

static void PrepareCubeTiling(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
    CalcMultiCorePartition(tiling, cubeCoreNum, tiling.m, tiling.n);
}

static bool IsValidOp(aclblasOperation_t op) { return op == ACLBLAS_OP_N || op == ACLBLAS_OP_T || op == ACLBLAS_OP_C; }

static aclblasStatus_t ValidateEnums(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, aclblasCgemmExType_t typeA,
    aclblasCgemmExType_t typeB, aclblasCgemmExType_t typeC)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCgemmEx", "handle is nullptr");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(IsValidOp(transa) && IsValidOp(transb), OP_LOGE("aclblasCgemmEx", "invalid transa/transb");
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(IsValidType(typeA) && IsValidType(typeB) && IsValidType(typeC),
              OP_LOGE("aclblasCgemmEx", "invalid typeA/typeB/typeC");
              return ACLBLAS_STATUS_INVALID_ENUM);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateDims(
    aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb, int ldc)
{
    CHECK_RET(m >= 0 && n >= 0 && k >= 0, OP_LOGE("aclblasCgemmEx", "negative dimension");
              return ACLBLAS_STATUS_INVALID_VALUE);
    int ldaMin = (transa != ACLBLAS_OP_N) ? std::max(1, k) : std::max(1, m);
    CHECK_RET(lda >= ldaMin,
              OP_LOGE("aclblasCgemmEx", "lda=%d < max(1, %s)=%d", lda, (transa != ACLBLAS_OP_N) ? "k" : "m", ldaMin);
              return ACLBLAS_STATUS_INVALID_VALUE);
    int ldbMin = (transb != ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    CHECK_RET(ldb >= ldbMin,
              OP_LOGE("aclblasCgemmEx", "ldb=%d < max(1, %s)=%d", ldb, (transb != ACLBLAS_OP_N) ? "n" : "k", ldbMin);
              return ACLBLAS_STATUS_INVALID_VALUE);
    int ldcMin = std::max(1, m);
    CHECK_RET(ldc >= ldcMin, OP_LOGE("aclblasCgemmEx", "ldc=%d < max(1, m)=%d", ldc, ldcMin);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidatePointers(
    int m, int n, int k, const aclblasComplex* alpha, const void* A, const void* B, const aclblasComplex* beta,
    const void* C)
{
    CHECK_RET(alpha != nullptr && beta != nullptr, OP_LOGE("aclblasCgemmEx", "alpha/beta is nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    bool needAB = (k > 0) && !IsZeroComplex(alpha);
    CHECK_RET(!needAB || (A != nullptr && B != nullptr),
              OP_LOGE("aclblasCgemmEx", "A/B is nullptr when m,n,k>0 and alpha!=0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(C != nullptr, OP_LOGE("aclblasCgemmEx", "C is nullptr when m>0 and n>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const void* A, aclblasCgemmExType_t typeA, int lda, const void* B, aclblasCgemmExType_t typeB,
    int ldb, const aclblasComplex* beta, const void* C, aclblasCgemmExType_t typeC, int ldc)
{
    aclblasStatus_t st = ValidateEnums(handle, transa, transb, typeA, typeB, typeC);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateDims(transa, transb, m, n, k, lda, ldb, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return ValidatePointers(m, n, k, alpha, A, B, beta, C);
}

static aclblasStatus_t ConvertHalfToDeviceC32(
    const void* src, aclblasCgemmExType_t type, int physRows, int physCols, int ld, void** dstDev, size_t* bytesOut);
static aclblasStatus_t ConvertC32DeviceToHalf(void* c32Dev, void* dst, aclblasCgemmExType_t type, int m, int n, int ldc);

static aclblasStatus_t ScaleHalfC(
    _aclblas_handle* h, int m, int n, int ldc, const aclblasComplex* beta, void* C, aclblasCgemmExType_t typeC,
    uint32_t blocks)
{
    void* c32 = nullptr;
    size_t dummy = 0;
    aclblasStatus_t st = ConvertHalfToDeviceC32(C, typeC, m, n, ldc, &c32, &dummy);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    gemm_scale_do(blocks, h->stream, reinterpret_cast<uint8_t*>(c32), m, n, ldc, beta->real, beta->imag, 1);
    if (aclrtSynchronizeStream(h->stream) != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtSynchronizeStream failed in ScaleHalfC");
        aclrtFree(c32);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    st = ConvertC32DeviceToHalf(c32, C, typeC, m, n, ldc);
    aclrtFree(c32);
    return st;
}

static aclblasStatus_t HandleScaleOnly(
    aclblasHandle_t handle, int m, int n, int ldc, const aclblasComplex* beta, void* C, aclblasCgemmExType_t typeC)
{
    if (beta->real == 1.0f && beta->imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t aiv = GetAivCoreCount();
    if (aiv == 0) {
        OP_LOGE("aclblasCgemmEx", "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t blocks = std::min(aiv, static_cast<uint32_t>(n));
    if (typeC == ACLBLAS_C_32) {
        gemm_scale_do(blocks, h->stream, reinterpret_cast<uint8_t*>(C), m, n, ldc, beta->real, beta->imag, 1);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (typeC == ACLBLAS_R_32) {
        gemm_scale_do(blocks, h->stream, reinterpret_cast<uint8_t*>(C), m, n, ldc, beta->real, 0.0f, 0);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!IsHalfType(typeC)) {
        OP_LOGE("aclblasCgemmEx", "unsupported typeC=%d on scale-only path", static_cast<int>(typeC));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ScaleHalfC(h, m, n, ldc, beta, C, typeC, blocks);
}

static bool UseSmallC32(int m, int n, int k) { return (m <= 16 && n <= 16 && k <= 16); }

static bool UseFusedCube3(int m, int n, int k) { return !(m >= 1000 && n >= 1000 && k >= 1000); }

static void SetTileBN(GemmTilingData& tiling, int bm, int bn, int bk, int kChunk)
{
    tiling.baseM = bm;
    tiling.baseN = bn;
    tiling.baseK = bk;
    tiling.tileKChunk = kChunk;
}

static bool TryTinyFastTiles(GemmTilingData& tiling, int t128, int mnMin, int k)
{
    if (!(t128 <= 2 && mnMin <= 128 && k < 256)) {
        return false;
    }
    SetTileBN(tiling, 32, 32, 32, (k >= 256) ? 256 : 64);
    return true;
}

static bool TrySkinnyK256Tiles(GemmTilingData& tiling, int cores, int t128, int t64, int mnMin, int k)
{
    if (!(k >= 256 && mnMin <= 256 && t128 < cores)) {
        return false;
    }
    if ((t64 >= cores && t128 * 4 <= cores * 3) || (t128 <= 2 && t64 > t128)) {
        SetTileBN(tiling, 64, 64, 32, 256);
        return true;
    }
    SetTileBN(tiling, 128, 128, 64, 64);
    return true;
}

static bool TrySkinnyMn160Tiles(GemmTilingData& tiling, int cores, int m, int n, int t64, int t32, int mnMin, int k)
{
    if (!(mnMin <= 160 && k >= 256)) {
        return false;
    }
    int tM64 = CeilDivPositive(m, 64);
    int tN64 = CeilDivPositive(n, 64);
    int tiles64 = tM64 * tN64;
    if (tiles64 >= cores && tiles64 <= cores * 8) {
        SetTileBN(tiling, 64, 64, 32, 256);
        return true;
    }
    if (std::min(tM64, tN64) <= 2 && tiles64 < cores) {
        SetTileBN(tiling, (t32 >= cores) ? 32 : 64, (t32 >= cores) ? 32 : 64, 32, 256);
        return true;
    }
    if (t32 >= cores) {
        SetTileBN(tiling, 32, 32, 32, 256);
        return true;
    }
    if (t64 < cores || t32 < (cores + 1) / 2) {
        SetTileBN(tiling, 64, 64, 32, 256);
        return true;
    }
    SetTileBN(tiling, 32, 32, 32, 256);
    return true;
}

static bool TryShortK96Tiles(GemmTilingData& tiling, int cores, int m, int n, int t128, int t64, int mnMin, int k)
{
    if (k > 96) {
        return false;
    }
    if (mnMin <= 400 && std::max(m, n) >= 2048 && t64 >= cores) {
        SetTileBN(tiling, 64, 64, 64, 64);
        return true;
    }
    if (t64 >= cores && t64 <= cores * 10) {
        SetTileBN(tiling, 64, 64, 64, 64);
        return true;
    }
    if (t128 < cores && t64 > t128) {
        SetTileBN(tiling, 64, 64, 64, 64);
        return true;
    }
    SetTileBN(tiling, 128, 128, 64, 64);
    return true;
}

static bool TryLongKMidMnTiles(GemmTilingData& tiling, int cores, int t64, int mnMin, int mnMax, int k)
{
    if (!(mnMin >= 350 && mnMin <= 600 && mnMax >= 800 && k >= 2000 && t64 >= cores && t64 <= cores * 5)) {
        return false;
    }
    SetTileBN(tiling, 64, 64, 32, 256);
    return true;
}

static void ApplyDefaultFastTiles(GemmTilingData& tiling, int cores, int t256, int t128, int t64, int k)
{
    if (t256 < cores * 4) {
        if (t128 < cores && t64 > t128) {
            SetTileBN(tiling, 64, 64, 32, (k >= 256) ? 256 : 64);
            return;
        }
        SetTileBN(tiling, 128, 128, 64, 64);
        return;
    }
    SetTileBN(tiling, 256, 256, 32, 128);
}

static void ApplyFastTiles(GemmTilingData& tiling, uint32_t cubeCores, int m, int n, int k)
{
    int cores = static_cast<int>(cubeCores);
    if (cores < 1) {
        cores = 1;
    }
    int t256 = CeilDivPositive(m, 256) * CeilDivPositive(n, 256);
    int t128 = CeilDivPositive(m, 128) * CeilDivPositive(n, 128);
    int t64 = CeilDivPositive(m, 64) * CeilDivPositive(n, 64);
    int t32 = CeilDivPositive(m, 32) * CeilDivPositive(n, 32);
    int mnMin = std::min(m, n);
    int mnMax = std::max(m, n);
    if (TryTinyFastTiles(tiling, t128, mnMin, k)) {
        return;
    }
    if (TrySkinnyK256Tiles(tiling, cores, t128, t64, mnMin, k)) {
        return;
    }
    if (TrySkinnyMn160Tiles(tiling, cores, m, n, t64, t32, mnMin, k)) {
        return;
    }
    if (TryShortK96Tiles(tiling, cores, m, n, t128, t64, mnMin, k)) {
        return;
    }
    if (TryLongKMidMnTiles(tiling, cores, t64, mnMin, mnMax, k)) {
        return;
    }
    ApplyDefaultFastTiles(tiling, cores, t256, t128, t64, k);
}

static uint32_t ComputeKParts(const GemmTilingData& tiling, uint32_t cubeCores)
{
    int bm = tiling.baseM > 0 ? tiling.baseM : 256;
    int bn = tiling.baseN > 0 ? tiling.baseN : 256;
    int tiles = std::max(1, CeilDivPositive(tiling.m, bm) * CeilDivPositive(tiling.n, bn));
    if (tiles >= static_cast<int>(cubeCores) || tiling.k < 128) {
        return 1;
    }
    uint32_t parts = cubeCores / static_cast<uint32_t>(tiles);
    uint32_t kMax = static_cast<uint32_t>(CeilDivPositive(tiling.k, 64));
    if (parts > kMax) {
        parts = kMax;
    }
    if (parts < 1) {
        parts = 1;
    }
    return parts;
}

static uint32_t FastCubeBlocks(const GemmTilingData& tiling, uint32_t cubeCores)
{
    int bm = tiling.baseM > 0 ? tiling.baseM : 256;
    int bn = tiling.baseN > 0 ? tiling.baseN : 256;
    int tM = CeilDivPositive(tiling.m, bm);
    int tN = CeilDivPositive(tiling.n, bn);
    int tiles = std::max(1, tM * tN);
    return std::min(cubeCores, static_cast<uint32_t>(tiles));
}

static void FillFastTiling(GemmTilingData& tiling, uint32_t cubeCores, int m, int n, int k)
{
    ApplyFastTiles(tiling, cubeCores, m, n, k);
    PrepareCubeTiling(tiling, cubeCores);
}

static void LaunchFastCube(
    aclrtStream stream, uint32_t cubeCores, uint8_t* a, uint8_t* b, uint8_t* c, const GemmTilingData& tiling)
{
    uint32_t blocks = FastCubeBlocks(tiling, cubeCores);
    if (tiling.baseM <= 64) {
        cgemm_ex_cube_small_do(blocks, stream, a, b, c, tiling);
    } else {
        cgemm_ex_cube_do(blocks, stream, a, b, c, tiling);
    }
}

static aclblasStatus_t LaunchCubeNN(
    aclrtStream stream, uint32_t cubeCores, uint8_t* a, uint8_t* b, uint8_t* c, int m, int n, int k, int lda, int ldb,
    int ldc, aclblasOperation_t transa, aclblasOperation_t transb)
{
    GemmTilingData tiling = MakeTiling(m, n, k, lda, ldb, ldc, transa, transb, 1.0f, 0.0f, 0.0f, 0.0f);
    tiling.ldc = ldc;
    FillFastTiling(tiling, cubeCores, tiling.m, tiling.n, tiling.k);
    LaunchFastCube(stream, cubeCores, a, b, c, tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchRealGemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const float* A,
    int lda, const float* B, int ldb, float* workspaceC, int ldcOut)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t cubeCores = GetAicCoreCount();
    if (cubeCores == 0) {
        OP_LOGE("aclblasCgemmEx", "GetAicCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    GemmTilingData tiling = MakeTiling(m, n, k, lda, ldb, ldcOut, transa, transb, 1.0f, 0.0f, 0.0f, 0.0f);
    tiling.ldc = ldcOut;
    FillFastTiling(tiling, cubeCores, tiling.m, tiling.n, tiling.k);
    uint8_t* aPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(B));
    uint8_t* bPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
    LaunchFastCube(h->stream, cubeCores, aPtr, bPtr, reinterpret_cast<uint8_t*>(workspaceC), tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

struct C32Workspace {
    uint8_t* reA = nullptr;
    uint8_t* imA = nullptr;
    uint8_t* sumA = nullptr;
    uint8_t* reB = nullptr;
    uint8_t* imB = nullptr;
    uint8_t* sumB = nullptr;
    uint8_t* nReB = nullptr;
    uint8_t* t1 = nullptr;
    uint8_t* t2 = nullptr;
    uint8_t* t3 = nullptr;
    size_t tempLdc = 0;
    int32_t kParts = 1;
};

static aclblasStatus_t AllocC32Workspace(_aclblas_handle* h, int m, int n, int k, int32_t kParts, C32Workspace& ws)
{
    if (kParts < 1) {
        kParts = 1;
    }
    size_t mk = static_cast<size_t>(m) * static_cast<size_t>(k);
    size_t kn = static_cast<size_t>(k) * static_cast<size_t>(n);
    ws.tempLdc = static_cast<size_t>(CeilAlign(m, GEMM_FRACTAL));
    ws.kParts = kParts;
    size_t mn = ws.tempLdc * static_cast<size_t>(n) * static_cast<size_t>(kParts);
    size_t fBytes = sizeof(float);
    size_t need = (mk * 3 + kn * 4 + mn * 3) * fBytes;
    aclblasStatus_t st = EnsureDefaultWorkspace(h, need);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    uint8_t* base = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    ws.reA = base;
    ws.imA = ws.reA + mk * fBytes;
    ws.sumA = ws.imA + mk * fBytes;
    ws.reB = ws.sumA + mk * fBytes;
    ws.imB = ws.reB + kn * fBytes;
    ws.sumB = ws.imB + kn * fBytes;
    ws.nReB = ws.sumB + kn * fBytes;
    ws.t1 = ws.nReB + kn * fBytes;
    ws.t2 = ws.t1 + mn * fBytes;
    ws.t3 = ws.t2 + mn * fBytes;
    return ACLBLAS_STATUS_SUCCESS;
}

static void DeinterleaveAB(
    aclrtStream stream, uint32_t aivCores, const void* A, const void* B, aclblasOperation_t transa,
    aclblasOperation_t transb, int physRowsA, int physColsA, int lda, int aLd, int physRowsB, int physColsB, int ldb,
    int bLd, const C32Workspace& ws, bool fused)
{
    uint8_t* aSrc = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(A));
    uint8_t* bSrc = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(B));
    int conjA = (transa == ACLBLAS_OP_C) ? 1 : 0;
    int conjB = (transb == ACLBLAS_OP_C) ? 1 : 0;
    if (fused) {
        cgemm_ex_deint2_do(
            aivCores, stream, aSrc, ws.reA, ws.imA, physRowsA, physColsA, lda, aLd, conjA, ws.sumA, bSrc, ws.reB,
            ws.imB, physRowsB, physColsB, ldb, bLd, conjB, ws.sumB);
        return;
    }
    cgemm_ex_deinterleave_do(aivCores, stream, aSrc, ws.reA, ws.imA, physRowsA, physColsA, lda, aLd, 0, conjA, ws.sumA);
    cgemm_ex_deinterleave_do(aivCores, stream, bSrc, ws.reB, ws.imB, physRowsB, physColsB, ldb, bLd, 0, conjB, ws.sumB);
}

static aclblasStatus_t LaunchSmallC32Path(
    _aclblas_handle* h, uint32_t aivCores, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const void* A, int lda, const void* B, int ldb, const aclblasComplex* beta, void* C,
    int ldc, int cIsComplex)
{
    uint32_t blocks = std::min(aivCores, static_cast<uint32_t>(std::max(n, 1)));
    int transA = (transa != ACLBLAS_OP_N) ? 1 : 0;
    int transB = (transb != ACLBLAS_OP_N) ? 1 : 0;
    int conjA = (transa == ACLBLAS_OP_C) ? 1 : 0;
    int conjB = (transb == ACLBLAS_OP_C) ? 1 : 0;
    cgemm_ex_small_c32_do(
        blocks, h->stream, const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(A)), lda, transA, conjA,
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(B)), ldb, transB, conjB, reinterpret_cast<uint8_t*>(C),
        ldc, cIsComplex, m, n, k, alpha->real, alpha->imag, beta->real, beta->imag);
    return ACLBLAS_STATUS_SUCCESS;
}

static void LaunchCubeOrCube3(
    _aclblas_handle* h, uint32_t cubeCores, uint32_t blocks, const C32Workspace& ws, int m, int n, int k, int physRowsA,
    int physRowsB, int outLd, aclblasOperation_t transa, aclblasOperation_t transb, GemmTilingData& tiling)
{
    if (UseFusedCube3(m, n, k)) {
        cgemm_ex_cube3_do(
            blocks, h->stream, ws.reB, ws.reA, ws.t1, ws.imB, ws.imA, ws.t2, ws.sumB, ws.sumA, ws.t3, tiling);
        return;
    }
    LaunchCubeNN(h->stream, cubeCores, ws.reB, ws.reA, ws.t1, m, n, k, physRowsA, physRowsB, outLd, transa, transb);
    LaunchCubeNN(h->stream, cubeCores, ws.imB, ws.imA, ws.t2, m, n, k, physRowsA, physRowsB, outLd, transa, transb);
    LaunchCubeNN(h->stream, cubeCores, ws.sumB, ws.sumA, ws.t3, m, n, k, physRowsA, physRowsB, outLd, transa, transb);
}

static aclblasStatus_t LaunchC32AfterDeint(
    _aclblas_handle* h, uint32_t cubeCores, uint32_t aivCores, uint32_t blocks, const C32Workspace& ws, int m, int n,
    int k, int physRowsA, int physRowsB, int outLd, aclblasOperation_t transa, aclblasOperation_t transb,
    const aclblasComplex* alpha, const aclblasComplex* beta, void* C, int ldc, int cIsComplex, int32_t kParts,
    GemmTilingData& tiling)
{
    bool mixOk = UseFusedCube3(m, n, k) && kParts <= 1;
    if (mixOk && !cIsComplex && (m % 8) != 0) {
        mixOk = false;
    }
    if (mixOk) {
        cgemm_ex_mix_do(
            blocks, h->stream, ws.reB, ws.reA, ws.t1, ws.imB, ws.imA, ws.t2, ws.sumB, ws.sumA, ws.t3,
            reinterpret_cast<uint8_t*>(C), m, n, outLd, ldc, alpha->real, alpha->imag, beta->real, beta->imag,
            cIsComplex, tiling);
        return ACLBLAS_STATUS_SUCCESS;
    }
    LaunchCubeOrCube3(h, cubeCores, blocks, ws, m, n, k, physRowsA, physRowsB, outLd, transa, transb, tiling);
    cgemm_ex_combine3_do(
        aivCores, h->stream, ws.t1, ws.t2, ws.t3, outLd, reinterpret_cast<uint8_t*>(C), m, n, ldc, alpha->real,
        alpha->imag, beta->real, beta->imag, cIsComplex, kParts);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchC32CubePath(
    _aclblas_handle* h, uint32_t cubeCores, uint32_t aivCores, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha, const void* A, int lda, const void* B, int ldb,
    const aclblasComplex* beta, void* C, int ldc, int cIsComplex)
{
    int physRowsA = (transa == ACLBLAS_OP_N) ? m : k;
    int physColsA = (transa == ACLBLAS_OP_N) ? k : m;
    int physRowsB = (transb == ACLBLAS_OP_N) ? k : n;
    int physColsB = (transb == ACLBLAS_OP_N) ? n : k;
    int outLd = CeilAlign(m, GEMM_FRACTAL);
    GemmTilingData tiling = MakeTiling(m, n, k, physRowsA, physRowsB, outLd, transa, transb, 1.0f, 0.0f, 0.0f, 0.0f);
    tiling.ldc = outLd;
    FillFastTiling(tiling, cubeCores, m, n, k);
    int32_t kParts = static_cast<int32_t>(ComputeKParts(tiling, cubeCores));
    tiling.mBlocks = kParts;
    int bm = tiling.baseM > 0 ? tiling.baseM : 256;
    int bn = tiling.baseN > 0 ? tiling.baseN : 256;
    int mnTiles = std::max(1, CeilDivPositive(tiling.m, bm) * CeilDivPositive(tiling.n, bn));
    uint32_t blocks = std::min(cubeCores, static_cast<uint32_t>(mnTiles) * static_cast<uint32_t>(kParts));
    if (blocks < 1) {
        blocks = 1;
    }
    tiling.usedCoreNum = static_cast<int32_t>(blocks);
    C32Workspace ws;
    aclblasStatus_t st = AllocC32Workspace(h, m, n, k, kParts, ws);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    DeinterleaveAB(
        h->stream, aivCores, A, B, transa, transb, physRowsA, physColsA, lda, physRowsA, physRowsB, physColsB, ldb,
        physRowsB, ws, true);
    return LaunchC32AfterDeint(
        h, cubeCores, aivCores, blocks, ws, m, n, k, physRowsA, physRowsB, outLd, transa, transb, alpha, beta, C, ldc,
        cIsComplex, kParts, tiling);
}

static aclblasStatus_t LaunchC32Path(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const void* A, int lda, const void* B, int ldb, const aclblasComplex* beta, void* C,
    int ldc, int cIsComplex)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t cubeCores = GetAicCoreCount();
    uint32_t aivCores = GetAivCoreCount();
    if (cubeCores == 0 || aivCores == 0) {
        OP_LOGE("aclblasCgemmEx", "GetAicCoreCount/GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (UseSmallC32(m, n, k)) {
        return LaunchSmallC32Path(
            h, aivCores, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, cIsComplex);
    }
    return LaunchC32CubePath(
        h, cubeCores, aivCores, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, cIsComplex);
}

static aclblasStatus_t ApplyR32RealScale(
    _aclblas_handle* h, float* wsC, void* C, int m, int n, int ldc, size_t tempLdc, const aclblasComplex* alpha,
    const aclblasComplex* beta, uint32_t comb)
{
    GemmTilingData ab{};
    ab.m = m;
    ab.n = n;
    ab.ldc = static_cast<int32_t>(tempLdc);
    ab.cLdc = ldc;
    ab.alphaReal = alpha->real;
    ab.betaReal = beta->real;
    ab.hasBeta = (beta->real != 0.0f) ? 1 : 0;
    gemm_alpha_beta_do(
        comb, h->stream, reinterpret_cast<uint8_t*>(wsC), reinterpret_cast<uint8_t*>(C), reinterpret_cast<uint8_t*>(C),
        ab);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ApplyR32ComplexScale(
    _aclblas_handle* h, uint8_t* ws, float* wsC, void* C, int m, int n, int ldc, size_t tempLdc, size_t mnBytes,
    const aclblasComplex* alpha, const aclblasComplex* beta, uint32_t comb)
{
    uint8_t* zero = ws + mnBytes;
    if (aclrtMemset(zero, mnBytes, 0, mnBytes) != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtMemset failed for R32 complex scale");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    cgemm_ex_combine3_do(
        comb, h->stream, reinterpret_cast<uint8_t*>(wsC), zero, reinterpret_cast<uint8_t*>(wsC),
        static_cast<int32_t>(tempLdc), reinterpret_cast<uint8_t*>(C), m, n, ldc, alpha->real, alpha->imag, beta->real,
        beta->imag, 1, 1);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchR32Workspace(
    aclblasHandle_t handle, _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n,
    int k, const aclblasComplex* alpha, const float* A, int lda, const float* B, int ldb, const aclblasComplex* beta,
    void* C, int ldc, size_t tempLdc, bool realC)
{
    size_t mnBytes = tempLdc * static_cast<size_t>(n) * sizeof(float);
    size_t need = realC ? mnBytes : (mnBytes * 2);
    aclblasStatus_t st = EnsureDefaultWorkspace(h, need);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    uint8_t* ws = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    float* wsC = reinterpret_cast<float*>(ws);
    st = LaunchRealGemm(handle, transa, transb, m, n, k, A, lda, B, ldb, wsC, static_cast<int>(tempLdc));
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    uint32_t aiv = GetAivCoreCount();
    uint32_t comb = std::min(aiv, static_cast<uint32_t>(n));
    if (realC) {
        return ApplyR32RealScale(h, wsC, C, m, n, ldc, tempLdc, alpha, beta, comb);
    }
    return ApplyR32ComplexScale(h, ws, wsC, C, m, n, ldc, tempLdc, mnBytes, alpha, beta, comb);
}

static aclblasStatus_t LaunchR32Path(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const float* A, int lda, const float* B, int ldb, const aclblasComplex* beta, void* C,
    int ldc, aclblasCgemmExType_t typeC)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    size_t tempLdc = static_cast<size_t>(CeilAlign(m, GEMM_FRACTAL));
    bool realC = (typeC == ACLBLAS_R_32);
    bool identity = (alpha->real == 1.0f && alpha->imag == 0.0f && beta->real == 0.0f && beta->imag == 0.0f);
    if (realC && identity) {
        return LaunchRealGemm(
            handle, transa, transb, m, n, k, A, lda, B, ldb, const_cast<float*>(reinterpret_cast<const float*>(C)),
            ldc);
    }
    return LaunchR32Workspace(handle, h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, tempLdc, realC);
}

static constexpr size_t kMaxHostConvertBytes = static_cast<size_t>(1) << 32;

static aclblasStatus_t MallocDevice(void** dev, size_t bytes)
{
    if (bytes == 0 || bytes > kMaxHostConvertBytes) {
        OP_LOGE("aclblasCgemmEx", "invalid device alloc size %zu", bytes);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (aclrtMalloc(dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtMalloc failed, bytes=%zu", bytes);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static bool HostCountOk(size_t count, size_t elemBytes)
{
    if (count == 0 || elemBytes == 0) {
        return false;
    }
    return count <= (kMaxHostConvertBytes / elemBytes);
}

static aclblasStatus_t CopyHostC32ToDev(void** dstDev, const void* host, size_t bytes)
{
    void* dev = nullptr;
    aclblasStatus_t st = MallocDevice(&dev, bytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (aclrtMemcpy(dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtMemcpy H2D failed, bytes=%zu", bytes);
        aclrtFree(dev);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    *dstDev = dev;
    return ACLBLAS_STATUS_SUCCESS;
}

static void FillHalfHost(const uint8_t* raw, size_t count, aclblasCgemmExType_t type, std::vector<aclblasComplex>& out)
{
    const uint16_t* p = reinterpret_cast<const uint16_t*>(raw);
    if (type == ACLBLAS_H_C_32) {
        for (size_t i = 0; i < count; i++) {
            out[i].real = blas_common::HalfToFloat(p[i * 2]);
            out[i].imag = blas_common::HalfToFloat(p[i * 2 + 1]);
        }
        return;
    }
    for (size_t i = 0; i < count; i++) {
        out[i].real = blas_common::HalfToFloat(p[i]);
        out[i].imag = 0.0f;
    }
}

static aclblasStatus_t PromoteR32ToC32(const void* src, int physRows, int physCols, int ld, void** dstDev)
{
    (void)physRows;
    if (ld <= 0 || physCols <= 0) {
        OP_LOGE("aclblasCgemmEx", "invalid R32 promote ld=%d physCols=%d", ld, physCols);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    size_t count = static_cast<size_t>(ld) * static_cast<size_t>(physCols);
    if (!HostCountOk(count, sizeof(aclblasComplex))) {
        OP_LOGE("aclblasCgemmEx", "R32 promote count overflow ld=%d physCols=%d", ld, physCols);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    std::vector<float> host(count);
    size_t inBytes = count * sizeof(float);
    aclError ret = aclrtMemcpy(host.data(), inBytes, src, inBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtMemcpy D2H failed in PromoteR32ToC32");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    std::vector<aclblasComplex> out(count);
    for (size_t i = 0; i < count; i++) {
        out[i].real = host[i];
        out[i].imag = 0.0f;
    }
    return CopyHostC32ToDev(dstDev, out.data(), count * sizeof(aclblasComplex));
}

static aclblasStatus_t ConvertHalfToDeviceC32(
    const void* src, aclblasCgemmExType_t type, int physRows, int physCols, int ld, void** dstDev, size_t* bytesOut)
{
    if (ld <= 0 || physCols <= 0) {
        OP_LOGE("aclblasCgemmEx", "invalid half convert ld=%d physCols=%d", ld, physCols);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    size_t count = static_cast<size_t>(ld) * static_cast<size_t>(physCols);
    size_t elemBytes = TypeElemBytes(type);
    if (!HostCountOk(count, sizeof(aclblasComplex)) || !HostCountOk(count, elemBytes)) {
        OP_LOGE("aclblasCgemmEx", "half convert count overflow ld=%d physCols=%d", ld, physCols);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    std::vector<uint8_t> raw(count * elemBytes);
    size_t rawBytes = raw.size();
    aclError ret = aclrtMemcpy(raw.data(), rawBytes, src, rawBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtMemcpy D2H failed in ConvertHalfToDeviceC32");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    std::vector<aclblasComplex> out(count);
    FillHalfHost(raw.data(), count, type, out);
    size_t outBytes = count * sizeof(aclblasComplex);
    aclblasStatus_t st = CopyHostC32ToDev(dstDev, out.data(), outBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    *bytesOut = outBytes;
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ConvertC32DeviceToHalf(void* c32Dev, void* dst, aclblasCgemmExType_t type, int m, int n, int ldc)
{
    if (ldc <= 0 || n <= 0) {
        OP_LOGE("aclblasCgemmEx", "invalid C32-to-half ldc=%d n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    size_t count = static_cast<size_t>(ldc) * static_cast<size_t>(n);
    if (!HostCountOk(count, sizeof(aclblasComplex))) {
        OP_LOGE("aclblasCgemmEx", "C32-to-half count overflow ldc=%d n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    std::vector<aclblasComplex> host(count);
    size_t inBytes = count * sizeof(aclblasComplex);
    aclError ret = aclrtMemcpy(host.data(), inBytes, c32Dev, inBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtMemcpy D2H failed in ConvertC32DeviceToHalf");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (type == ACLBLAS_H_C_32) {
        std::vector<uint16_t> out(count * 2);
        for (size_t i = 0; i < count; i++) {
            out[i * 2] = blas_common::FloatToHalf(host[i].real);
            out[i * 2 + 1] = blas_common::FloatToHalf(host[i].imag);
        }
        size_t outBytes = out.size() * sizeof(uint16_t);
        ret = aclrtMemcpy(dst, outBytes, out.data(), outBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        std::vector<uint16_t> out(count);
        for (size_t i = 0; i < count; i++) {
            out[i] = blas_common::FloatToHalf(host[i].real);
        }
        size_t outBytes = out.size() * sizeof(uint16_t);
        ret = aclrtMemcpy(dst, outBytes, out.data(), outBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtMemcpy H2D failed in ConvertC32DeviceToHalf");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static void FreeOwned(void* p)
{
    if (p != nullptr) {
        aclrtFree(p);
    }
}

static aclblasStatus_t MaybeConvertHalf(
    const void* src, aclblasCgemmExType_t type, int rows, int cols, int ld, void** owned, void** ptr)
{
    if (!IsHalfType(type)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    size_t dummy = 0;
    aclblasStatus_t st = ConvertHalfToDeviceC32(src, type, rows, cols, ld, owned, &dummy);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    *ptr = *owned;
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t SyncConvertHalfC(
    aclblasHandle_t handle, void* c32, void* C, aclblasCgemmExType_t typeC, int m, int n, int ldc)
{
    if (!IsHalfType(typeC)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* hh = reinterpret_cast<_aclblas_handle*>(handle);
    if (aclrtSynchronizeStream(hh->stream) != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmEx", "aclrtSynchronizeStream failed in SyncConvertHalfC");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ConvertC32DeviceToHalf(c32, C, typeC, m, n, ldc);
}

static aclblasStatus_t LaunchHalfPath(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const void* A, aclblasCgemmExType_t typeA, int lda, const void* B, aclblasCgemmExType_t typeB,
    int ldb, const aclblasComplex* beta, void* C, aclblasCgemmExType_t typeC, int ldc)
{
    void* a32 = const_cast<void*>(A);
    void* b32 = const_cast<void*>(B);
    void* c32 = C;
    void* aOwn = nullptr;
    void* bOwn = nullptr;
    void* cOwn = nullptr;
    int physColsA = (transa == ACLBLAS_OP_N) ? k : m;
    int physRowsA = (transa == ACLBLAS_OP_N) ? m : k;
    int physColsB = (transb == ACLBLAS_OP_N) ? n : k;
    int physRowsB = (transb == ACLBLAS_OP_N) ? k : n;
    aclblasStatus_t st = MaybeConvertHalf(A, typeA, physRowsA, physColsA, lda, &aOwn, &a32);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = MaybeConvertHalf(B, typeB, physRowsB, physColsB, ldb, &bOwn, &b32);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        FreeOwned(aOwn);
        return st;
    }
    st = MaybeConvertHalf(C, typeC, m, n, ldc, &cOwn, &c32);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        FreeOwned(aOwn);
        FreeOwned(bOwn);
        return st;
    }
    st = LaunchC32Path(handle, transa, transb, m, n, k, alpha, a32, lda, b32, ldb, beta, c32, ldc, 1);
    if (st == ACLBLAS_STATUS_SUCCESS) {
        st = SyncConvertHalfC(handle, c32, C, typeC, m, n, ldc);
    }
    FreeOwned(aOwn);
    FreeOwned(bOwn);
    FreeOwned(cOwn);
    return st;
}

static aclblasStatus_t LaunchMixedPromote(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const void* A, aclblasCgemmExType_t typeA, int lda, const void* B, aclblasCgemmExType_t typeB,
    int ldb, const aclblasComplex* beta, void* C, aclblasCgemmExType_t typeC, int ldc)
{
    void* aC32 = const_cast<void*>(A);
    void* bC32 = const_cast<void*>(B);
    void* aOwn = nullptr;
    void* bOwn = nullptr;
    int physColsA = (transa == ACLBLAS_OP_N) ? k : m;
    int physRowsA = (transa == ACLBLAS_OP_N) ? m : k;
    int physColsB = (transb == ACLBLAS_OP_N) ? n : k;
    int physRowsB = (transb == ACLBLAS_OP_N) ? k : n;
    aclblasStatus_t st = ACLBLAS_STATUS_SUCCESS;
    if (typeA != ACLBLAS_C_32) {
        st = PromoteR32ToC32(A, physRowsA, physColsA, lda, &aOwn);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
        aC32 = aOwn;
    }
    if (typeB != ACLBLAS_C_32) {
        st = PromoteR32ToC32(B, physRowsB, physColsB, ldb, &bOwn);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            FreeOwned(aOwn);
            return st;
        }
        bC32 = bOwn;
    }
    st = LaunchC32Path(
        handle, transa, transb, m, n, k, alpha, aC32, lda, bC32, ldb, beta, C, ldc, typeC == ACLBLAS_C_32 ? 1 : 0);
    FreeOwned(aOwn);
    FreeOwned(bOwn);
    return st;
}

} // namespace

extern "C" aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const void* A, int typeA, int lda, const void* B, int typeB, int ldb,
    const aclblasComplex* beta, void* C, int typeC, int ldc)
{
    auto ta = static_cast<aclblasCgemmExType_t>(typeA);
    auto tb = static_cast<aclblasCgemmExType_t>(typeB);
    auto tc = static_cast<aclblasCgemmExType_t>(typeC);
    OP_LOGI("aclblasCgemmEx", "entry m=%d n=%d k=%d type=%d/%d/%d", m, n, k, typeA, typeB, typeC);

    aclblasStatus_t st =
        ValidateCgemmEx(handle, transa, transb, m, n, k, alpha, A, ta, lda, B, tb, ldb, beta, C, tc, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (k == 0 || IsZeroComplex(alpha)) {
        return HandleScaleOnly(handle, m, n, ldc, beta, C, tc);
    }
    if (IsHalfType(ta) || IsHalfType(tb) || IsHalfType(tc)) {
        return LaunchHalfPath(handle, transa, transb, m, n, k, alpha, A, ta, lda, B, tb, ldb, beta, C, tc, ldc);
    }
    if (ta == ACLBLAS_C_32 && tb == ACLBLAS_C_32) {
        return LaunchC32Path(
            handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, tc == ACLBLAS_C_32 ? 1 : 0);
    }
    if (ta != ACLBLAS_C_32 && tb != ACLBLAS_C_32) {
        return LaunchR32Path(
            handle, transa, transb, m, n, k, alpha, static_cast<const float*>(A), lda, static_cast<const float*>(B),
            ldb, beta, C, ldc, tc);
    }
    return LaunchMixedPromote(handle, transa, transb, m, n, k, alpha, A, ta, lda, B, tb, ldb, beta, C, tc, ldc);
}

#endif
