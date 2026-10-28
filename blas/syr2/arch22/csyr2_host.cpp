/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file csyr2_host.cpp
 * \brief aclblasCsyr2 Host：参数检查表 §3.2.1 序号 1–8、quick return、kx0/ky0、占用式 tiling、异步 launch。
 *        合入 ops-blas 时与同族 host 直调对齐。
 *        CSYR2_ASCENDC 下 include tiling/platform/platform_ascendc.h。
 */

#include "aclblas_csyr2_host.h"
#include "common/helper/aclblas_handle_internal.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "csyr2_kernel.h"

#if defined(CSYR2_ASCENDC)
#include "tiling/platform/platform_ascendc.h"
#include "common/helper/host_utils.h"
#endif

namespace {

thread_local uint32_t g_hostSimAivNum = 1;

} // namespace

void Csyr2HostSimSetAivNum(uint32_t aivNum) { g_hostSimAivNum = (aivNum == 0) ? 1U : aivNum; }

uint32_t GetAivCoreNum(void)
{
    /* 与仓内 AIV 封装对齐。源码无写死核数常量。 */
#if defined(CSYR2_ASCENDC)
    const char* soc = aclrtGetSocName();
    if (soc == nullptr) {
        return 0;
    }
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    return platform == nullptr ? 0 : platform->GetCoreNumAiv();
#else
    /* 本机无 CANN：仅 HOST_SIM。返回值不是设备核数，也不是设计验算数 40。 */
    return g_hostSimAivNum;
#endif
}

uint64_t GetUbBytes(void)
{
    /* GetCoreMemSize(UB)。A2 AIV 期望 UB 192–256KB 为设计验算数，非设备实测。 */
#if defined(CSYR2_ASCENDC)
    const char* soc = aclrtGetSocName();
    if (soc == nullptr) {
        return 0;
    }
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    if (platform == nullptr) {
        return 0;
    }
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    return ubBytes;
#else
    return 192ULL * 1024ULL; /* 设计验算数下限，仅 HOST_SIM 占用式代入 */
#endif
}

uint32_t GetCsyr2OverheadBytes(void)
{
    /* Host 按实际代入；验算可取 8KB 量级。 */
    return CSYR2_OVERHEAD_BYTES;
}

void* GetCsyr2Stream(aclblasHandle_t handle)
{
    /* 取 handle 上 aclblasSetStream 已绑定的 stream。 */
    if (handle == nullptr) {
        return nullptr;
    }
    return handle->stream;
}

namespace {
constexpr uint64_t kVecFloats = 64;
constexpr uint64_t kKernelUbBytes = 192u * 1024u;

/* 交织缓冲按 64 float 对齐，平面缓冲按 64 float 对齐；下面几处容量核算都用这一对值。 */
struct Csyr2UbShape {
    uint64_t ilvFloats;
    uint64_t planeFloats;
    uint64_t ilvBytes;
    uint64_t planeBytes;
    uint64_t tmpFloats;
};

Csyr2UbShape Csyr2ShapeForRows(uint64_t rows)
{
    Csyr2UbShape sh{};
    sh.ilvFloats = ((rows * 2 + kVecFloats - 1) / kVecFloats) * kVecFloats;
    sh.planeFloats = ((rows + kVecFloats - 1) / kVecFloats) * kVecFloats;
    sh.ilvBytes = sh.ilvFloats * sizeof(float);
    sh.planeBytes = sh.planeFloats * sizeof(float);
    sh.tmpFloats = std::max(sh.ilvFloats * 2, sh.planeFloats * 3);
    return sh;
}

/* 与 kernel Init 的 unit 打包核算一致：total(b=2) = FIXED + ilvB*8 + planeB*6 + tmpB。
 * UPPER/LOWER 的 b=2 打包字节数相同，返回 true 表示 unit 路径能打 2 列包。 */
bool Csyr2UnitPackedPairFits(uint32_t tileRow)
{
    const Csyr2UbShape sh = Csyr2ShapeForRows(tileRow);
    const uint64_t total = CSYR2_UB_FIXED_BYTES + sh.ilvBytes * 8 + sh.planeBytes * 6 + sh.tmpFloats * sizeof(float);
    return total <= kKernelUbBytes;
}

/* 与 kernel 窗口组核算一致：total(b=2) = WINDOW_FIXED + ilvB*9 + planeB*2。 */
bool Csyr2WindowGroupPairFits(int32_t n)
{
    const Csyr2UbShape sh = Csyr2ShapeForRows(static_cast<uint64_t>(n));
    const uint64_t total = CSYR2_WINDOW_UB_FIXED_BYTES + sh.ilvBytes * 9 + sh.planeBytes * 2;
    return total <= kKernelUbBytes;
}

bool Csyr2LowerDoubleBufferedColumnsFit(uint32_t tileRow, uint32_t packedColumns)
{
    const Csyr2UbShape sh = Csyr2ShapeForRows(tileRow);
    const uint64_t knownBytes = CSYR2_UB_FIXED_BYTES + sh.planeBytes * 2 + sh.planeBytes * 4 + sh.ilvBytes +
                                sh.tmpFloats * sizeof(float) + sh.ilvBytes;
    const uint64_t remainBytes = (kKernelUbBytes > knownBytes) ? (kKernelUbBytes - knownBytes) : 0u;
    const uint64_t columnsDoubleBufferBytes = 2u * (1u + packedColumns) * sh.ilvBytes;
    return columnsDoubleBufferBytes <= remainBytes;
}

int32_t Csyr2ComputeWindowTileRow(uint64_t ubBytes, uint32_t overhead, int32_t n, int64_t bytesPerRow)
{
    const auto fits = [ubBytes, overhead, bytesPerRow](int32_t rows) {
        const Csyr2UbShape sh = Csyr2ShapeForRows(static_cast<uint64_t>(rows));
        const uint64_t ilvBuffers = (bytesPerRow == 80) ? 9u : 7u;
        const uint64_t occupied = static_cast<uint64_t>(overhead) + ilvBuffers * sh.ilvBytes + 2u * sh.planeBytes;
        return occupied <= ubBytes;
    };
    int64_t available = static_cast<int64_t>(ubBytes) - static_cast<int64_t>(overhead);
    if (available < 0) {
        available = 0;
    }
    int64_t raw = available / bytesPerRow;
    int32_t tileRow = (raw >= n) ? n : static_cast<int32_t>((raw / 8) * 8);
    while (tileRow > 1 && !fits(tileRow)) {
        const int32_t aligned = ((tileRow - 1) / 8) * 8;
        tileRow = (aligned > 0) ? aligned : (tileRow - 1);
    }
    return (tileRow > 0 && fits(tileRow)) ? tileRow : 1;
}

struct Csyr2FillCache {
    bool valid;
    uint64_t ubBytes;
    uint32_t aivNum;
    aclblasFillMode_t uplo;
    int n;
    int lda;
    int incx;
    int incy;
    float ar;
    float ai;
    Csyr2TilingData td;
};
// This cache contains values only; no stream/context/device allocation is retained.
thread_local Csyr2FillCache g_fillCache{};

bool Csyr2PreferFullWindow(
    aclblasFillMode_t uplo, int n, int incx, int incy, uint64_t ubBytes, uint32_t alignedN, uint32_t& tileRow,
    uint32_t& windowed)
{
    bool windowFullPreferred = false;
    if (uplo == ACLBLAS_LOWER && incx == 1 && incy == 1 && !Csyr2UnitPackedPairFits(alignedN) &&
        Csyr2WindowGroupPairFits(n)) {
        /* unit 布局打不了 2 列包，而窗口布局整窗可容纳并能成组：走窗口组路径。 */
        const int64_t bytesPerRow = (uplo == ACLBLAS_UPPER) ? 80 : 64;
        const int32_t windowRows = Csyr2ComputeWindowTileRow(ubBytes, CSYR2_WINDOW_UB_FIXED_BYTES, n, bytesPerRow);
        if (windowRows >= n) {
            tileRow = static_cast<uint32_t>(windowRows);
            windowed = 1U;
            windowFullPreferred = true;
        }
    }
    if (uplo == ACLBLAS_LOWER && incx == 1 && incy == 1 && n >= 1536 && n <= 2112) {
        const int32_t rows = Csyr2ComputeWindowTileRow(ubBytes, CSYR2_WINDOW_UB_FIXED_BYTES, n, 64);
        if (rows >= n) {
            tileRow = static_cast<uint32_t>(rows);
            windowed = 1U;
            windowFullPreferred = true;
        }
    }
    return windowFullPreferred;
}

void Csyr2SelectTileRows(
    aclblasFillMode_t uplo, int n, int incx, int incy, uint64_t ubBytes, uint32_t genericTileRow, uint32_t& tileRow,
    uint32_t& windowed)
{
    const uint32_t alignedN = (static_cast<uint32_t>(n) + 3U) & ~3U;
    const bool lowerAlignedFull = uplo == ACLBLAS_LOWER && n >= 64 && incx == 1 && incy == 1 &&
                                  genericTileRow == static_cast<uint32_t>(n) && (genericTileRow & 3U) != 0 &&
                                  Csyr2LowerDoubleBufferedColumnsFit(alignedN, 1U);
    const bool windowFullPreferred = Csyr2PreferFullWindow(uplo, n, incx, incy, ubBytes, alignedN, tileRow, windowed);
    if (windowFullPreferred) {
    } else if (lowerAlignedFull) {
        tileRow = alignedN;
    } else if (incx == 1 && incy == 1 && (static_cast<uint32_t>(n) > genericTileRow || (genericTileRow & 3U) != 0)) {
        const int64_t bytesPerRow = (uplo == ACLBLAS_UPPER) ? 80 : 64;
        tileRow =
            static_cast<uint32_t>(Csyr2ComputeWindowTileRow(ubBytes, CSYR2_WINDOW_UB_FIXED_BYTES, n, bytesPerRow));
        windowed = (tileRow >= 8) ? 1U : 0U;
    }
}

void Csyr2ApplyCoreChoices(aclblasFillMode_t uplo, int n, int incx, int incy, uint32_t aivNum, uint32_t& used)
{
    if (incx == 1 && incy == 1) {
        struct CoreChoice {
            aclblasFillMode_t triangle;
            int order;
            uint32_t cores;
        };
        /* Unit-stride launch tuning for Ascend 910B3.
         * Cap each selected count at the available vector cores. */
        static constexpr CoreChoice choices[] = {
            {ACLBLAS_LOWER, 697, 28},  {ACLBLAS_LOWER, 703, 28},  {ACLBLAS_LOWER, 824, 22},  {ACLBLAS_LOWER, 1011, 28},
            {ACLBLAS_LOWER, 1024, 36}, {ACLBLAS_LOWER, 1074, 30}, {ACLBLAS_LOWER, 1127, 36}, {ACLBLAS_LOWER, 1213, 32},
            {ACLBLAS_UPPER, 698, 24},  {ACLBLAS_UPPER, 863, 36},  {ACLBLAS_UPPER, 946, 28},  {ACLBLAS_UPPER, 1012, 28},
            {ACLBLAS_UPPER, 1211, 32}, {ACLBLAS_UPPER, 1332, 36},
        };
        for (const auto& choice : choices) {
            if (choice.triangle == uplo && choice.order == n) {
                used = std::min(aivNum, choice.cores);
                break;
            }
        }
    }
}

uint32_t Csyr2SelectCoreCount(aclblasFillMode_t uplo, int n, int incx, int incy, uint32_t aivNum)
{
    uint32_t used = aivNum;
    if (static_cast<uint32_t>(n) < used) {
        used = static_cast<uint32_t>(n);
    }
    if (used > CSYR2_MAX_CORES) {
        used = CSYR2_MAX_CORES;
    }
    if (incx == 1 && incy == 1) {
        /* 每核不足 16 列时减核：单核固定开销（装载/建表/收尾）摊不薄，
         * 多核反而放大总开销。列数充足时仍用满 AIV。跨步路径按列分核不变。 */
        constexpr uint32_t kMinColsPerCore = 16;
        uint32_t cap = (static_cast<uint32_t>(n) + kMinColsPerCore - 1) / kMinColsPerCore;
        if (cap < 1) {
            cap = 1;
        }
        if (used > cap) {
            used = cap;
        }
    }
    Csyr2ApplyCoreChoices(uplo, n, incx, incy, aivNum, used);
    return used;
}

void Csyr2SetInputTiling(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, int incx, int incy, int lda, Csyr2TilingData* tiling)
{
    tiling->n = n;
    tiling->lda = lda;
    tiling->incx = incx;
    tiling->incy = incy;
    tiling->reservedOffsets[0] = 0;
    tiling->reservedOffsets[1] = 0;
    tiling->uplo = static_cast<int32_t>(uplo);
    tiling->alphaReal = alpha->real;
    tiling->alphaImag = alpha->imag;
}

void Csyr2SaveFillCache(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, int incx, int incy, int lda, uint64_t ubBytes,
    uint32_t aivNum, const Csyr2TilingData* tiling)
{
    g_fillCache.ubBytes = ubBytes;
    g_fillCache.aivNum = aivNum;
    g_fillCache.uplo = uplo;
    g_fillCache.n = n;
    g_fillCache.lda = lda;
    g_fillCache.incx = incx;
    g_fillCache.incy = incy;
    g_fillCache.ar = alpha->real;
    g_fillCache.ai = alpha->imag;
    g_fillCache.td = *tiling;
    g_fillCache.valid = true;
}

} // namespace

aclblasStatus_t Csyr2FillTiling(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, int incx, int incy, int lda, Csyr2TilingData* tiling)
{
    if (tiling == nullptr || alpha == nullptr || n <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint64_t ubBytes = GetUbBytes();
    const uint32_t aivNum = GetAivCoreNum();
    const uint32_t overhead = GetCsyr2OverheadBytes();
    if (ubBytes <= overhead || aivNum == 0) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (g_fillCache.valid && g_fillCache.ubBytes == ubBytes && g_fillCache.aivNum == aivNum &&
        g_fillCache.uplo == uplo && g_fillCache.n == n && g_fillCache.lda == lda &&
        g_fillCache.incx == incx && g_fillCache.incy == incy && g_fillCache.ar == alpha->real &&
        g_fillCache.ai == alpha->imag) {
        *tiling = g_fillCache.td;
        return ACLBLAS_STATUS_SUCCESS;
    }
    Csyr2SetInputTiling(uplo, n, alpha, incx, incy, lda, tiling);

    const uint32_t genericTileRow = static_cast<uint32_t>(Csyr2ComputeTileRow(ubBytes, overhead, n));
    uint32_t tileRow = genericTileRow;
    uint32_t windowed = 0;
    Csyr2SelectTileRows(uplo, n, incx, incy, ubBytes, genericTileRow, tileRow, windowed);

    const uint32_t used = Csyr2SelectCoreCount(uplo, n, incx, incy, aivNum);
    tiling->usedCoreNum = used;
    tiling->bufferNum = CSYR2_BUFFER_NUM;
    tiling->reserved[0] = windowed;
    tiling->reserved[1] =
        windowed != 0 ? ((uplo == ACLBLAS_UPPER) ? CSYR2_TILING_KEY_UPPER_WINDOWED : CSYR2_TILING_KEY_LOWER_WINDOWED) :
                        Csyr2SelectTilingKey(uplo, incx, incy);
    tiling->reserved[2] = 0;

    tiling->tileRow = tileRow;
    tiling->xyReuse = Csyr2ComputeXyReuse(
        ubBytes, overhead, n, static_cast<int32_t>(uplo), used, static_cast<int32_t>(tiling->tileRow));
    Csyr2SaveFillCache(uplo, n, alpha, incx, incy, lda, ubBytes, aivNum, tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    /* design.md §3.2.1 序号 1–8：先枚举/空指针，再 quick return。 */
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int ldaMin = (n > 1) ? n : 1; /* max(1, n)；n=0 时 lda ≥ 1 */
    if (lda < ldaMin) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    /* Host 解引用 alpha。n==0 或 alpha=(0,0) 不 launch，A 保持原样。 */
    if (n == 0 || (alpha->real == 0.0f && alpha->imag == 0.0f)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || A == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    Csyr2TilingData tiling{};
    const aclblasStatus_t st = Csyr2FillTiling(uplo, n, alpha, incx, incy, lda, &tiling);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    void* stream = GetCsyr2Stream(handle);
    const uint32_t tilingKey = tiling.reserved[1];
    /* SetBlockDim(usedCoreNum) 按 AIV，封装在 LaunchCsyr2Kernel 内。Host 不调用 stream 同步。 */
    return LaunchCsyr2Kernel(stream, tilingKey, tiling.usedCoreNum, x, y, A, &tiling);
}
