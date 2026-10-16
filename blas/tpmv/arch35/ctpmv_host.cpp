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
 * \file ctpmv_host.cpp
 * \brief Single-precision complex tpmv host-side implementation.
 *
 * 结构对齐同目录 stpmv_host.cpp（arch35 母版），差异点：
 *  1) complex64：AP/x 为 aclblasComplex（8 字节实虚交错），workspace 按 n*sizeof(aclblasComplex)；
 *  2) 非法 uplo/trans/diag 返回 ACLBLAS_STATUS_INVALID_ENUM（任务书 §2.4，
 *     区别于 stpmv 的 INVALID_VALUE）；
 *  3) 校验顺序：handle → 枚举 → n<0 → incx==0 → n==0 快速返回（先于空指针，
 *     与 Netlib "参数检查先于 quick return、n=0 不引用 AP/x" 语义一致）→ 空指针。
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "ctpmv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

void ctpmv_arch35_kernel_do(
    uint8_t* aP, uint8_t* x, uint8_t* y, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream);

void ctpmv_arch35_scatter_do(
    uint8_t* dst, uint8_t* src, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream);

void ctpmv_arch35_reduce_do(uint8_t* ws, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream);

void ctpmv_arch35_vec_tc_do(
    uint8_t* aPacked, uint8_t* x, uint8_t* snapshot, uint8_t* scrA, uint8_t* scrB, uint8_t* zeros,
    const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream);

void ctpmv_arch35_reduce2_do(
    uint8_t* ws, uint8_t* mid, uint8_t* ySlot, const CtpmvTilingData& tiling, uint32_t parts, void* stream);

void ctpmv_arch35_preswap_do(
    uint8_t* snap, uint8_t* scrA, uint8_t* scrB, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream);

void ctpmv_arch35_tiny_do(uint8_t* aP, uint8_t* x, uint8_t* y, const CtpmvTilingData& tiling, void* stream);

void ctpmv_arch35_nsmall_do(uint8_t* aP, uint8_t* x, uint8_t* y, const CtpmvTilingData& tiling, void* stream);

void ctpmv_arch35_nsmall2_do(uint8_t* aP, uint8_t* x, uint8_t* y, const CtpmvTilingData& tiling, void* stream);

void ctpmv_arch35_reduce4_do(uint8_t* src, uint8_t* dst, const CtpmvTilingData& tiling, uint32_t rows, void* stream);

void ctpmv_arch35_naxpy_do(
    uint8_t* aP, uint8_t* x, uint8_t* y, uint8_t* zeros, uint8_t* gap, const CtpmvTilingData& tiling, void* stream);

void ctpmv_arch35_ngap_do(uint8_t* aP, uint8_t* x, uint8_t* gap, const CtpmvTilingData& tiling, void* stream);

static aclblasStatus_t ValidateCtpmvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int incx)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasCtpmv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasCtpmv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
              OP_LOGE("aclblasCtpmv", "invalid diag=%d", static_cast<int>(diag));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(n >= 0, OP_LOGE("aclblasCtpmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasCtpmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static CtpmvTilingData CalCtpmvTilingData(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t nU32, int incx,
    uint32_t numThreads)
{
    CtpmvTilingData tiling{};
    tiling.n = nU32;
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = incx;
    tiling.numThreads = numThreads;
    return tiling;
}

static aclblasStatus_t GetCtpmvWorkspace(
    aclblasHandle_t handle, int n, size_t rows, size_t extraBytes, uint8_t** workspaceOut)
{
    *workspaceOut = nullptr;
    const size_t nSize = static_cast<size_t>(n);
    CHECK_RET(n > 0 && nSize <= std::numeric_limits<size_t>::max() / sizeof(aclblasComplex),
              OP_LOGE("aclblasCtpmv", "workspace row size overflow, n=%d", n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    const size_t slotBytes = nSize * sizeof(aclblasComplex);
    CHECK_RET(rows <= (std::numeric_limits<size_t>::max() - extraBytes) / slotBytes,
              OP_LOGE("aclblasCtpmv", "workspace size overflow, n=%d rows=%zu", n, rows);
              return ACLBLAS_STATUS_INVALID_VALUE);
    const size_t requiredBytes = rows * slotBytes + extraBytes;
    CHECK_RET(CheckEffectiveWorkspaceSize(handle, requiredBytes), return ACLBLAS_STATUS_EXECUTION_FAILED);
    *workspaceOut = static_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    CHECK_RET(*workspaceOut != nullptr, OP_LOGE("aclblasCtpmv", "handle workspace is nullptr");
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    return ACLBLAS_STATUS_SUCCESS;
}

static void ExecuteKernel(
    const aclblasComplex* AP, aclblasComplex* x, uint8_t* workspace, const CtpmvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    ctpmv_arch35_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(AP)), reinterpret_cast<uint8_t*>(x), workspace, tiling,
        numBlocks, stream);
    // v5.0: 两遍分组归约 (mid 8行 + y 行位于 parts 行之后); v5.1: 单 block 时行 0 即终值, 跳过
    if (numBlocks > 1u) {
        const size_t slotBytes = static_cast<size_t>(tiling.n) * sizeof(aclblasComplex);
        uint8_t* mid = workspace + static_cast<size_t>(numBlocks) * slotBytes;
        uint8_t* ySlot = mid + 8u * slotBytes;
        ctpmv_arch35_reduce2_do(workspace, mid, ySlot, tiling, numBlocks, stream);
    }
}

static aclblasStatus_t WriteBackResult(
    aclblasComplex* x, uint8_t* workspace, int n, int incx, const CtpmvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    // v5.1: 单 block 时 y 在行 0; 多 block 时在 parts+8 行(mid 之后)
    const size_t yRow = (numBlocks > 1u) ? (static_cast<size_t>(numBlocks) + 8u) : 0u;
    uint8_t* ySrc = workspace + yRow * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t workspaceSize = static_cast<size_t>(n) * sizeof(aclblasComplex);
    if (incx == 1) {
        aclError aclRet = aclrtMemcpyAsync(x, workspaceSize, ySrc, workspaceSize, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  OP_LOGE("aclblasCtpmv", "aclrtMemcpyAsync D2D (workspace->x) failed, ret=%d", aclRet);
                  return ACLBLAS_STATUS_INTERNAL_ERROR);
    } else {
        ctpmv_arch35_scatter_do(reinterpret_cast<uint8_t*>(x), ySrc, tiling, numBlocks, stream);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// v5.1: 小 n 单 block 直出——SIMT kernel 行 0 即完整 y, 省去两遍 reduce kernel(小 n 是启动开销主导)
// v5.3: 阈值 256->8——950 实测单核无延迟隐藏, n=64 恶化到 141us/n=128 331us; 仅 n<=8 值得省 reduce
constexpr uint32_t CTPMV_DIRECT_MAX_N = 8;
constexpr uint32_t CTPMV_NS2_CHUNKS = 32u;

// 线程/块配置: N 路径按行块总数配线程(latency-bound, 加线程=加装载并发), 小 n 单 block 直出。
static void ComputeCtpmvLaunch(bool isOpN, uint32_t nU32, uint32_t aivCoreNum, uint32_t* blocks, uint32_t* threads)
{
    uint32_t useNumBlocks = std::min(nU32, aivCoreNum);
    useNumBlocks = std::max<uint32_t>(useNumBlocks, 1);
    if (nU32 <= CTPMV_DIRECT_MAX_N) {
        useNumBlocks = 1;
    }
    uint32_t numThreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(nU32, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    const uint32_t tcThreads =
        std::min(CeilAlign<uint32_t>(nU32 / 8u > 0u ? nU32 / 8u : 1u, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    numThreads = std::max(numThreads, tcThreads);
    if (isOpN) {
        const uint32_t rbThreads = CeilAlign<uint32_t>(CeilDiv<uint32_t>(nU32, 8u), SIMT_MIN_THREAD_NUM);
        numThreads = std::min(std::max(numThreads, rbThreads), SIMT_MAX_THREAD_NUM);
    }
    *blocks = useNumBlocks;
    *threads = numThreads;
}

// T/C + incx==1: preswap + vec 双 kernel(所有 uplo/diag 组合); 实测远快于 SIMT 路径
static aclblasStatus_t RunCtpmvVecPath(
    aclblasHandle_t handle, const aclblasComplex* AP, aclblasComplex* x, int n, uint32_t nU32, uint32_t useNumBlocks,
    const CtpmvTilingData& tiling, void* stream)
{
    uint8_t* ws = nullptr;
    const aclblasStatus_t st = GetCtpmvWorkspace(handle, n, 3U, 32768U, &ws);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const size_t slotBytes = static_cast<size_t>(n) * sizeof(aclblasComplex);
    uint8_t* snapshot = ws;
    uint8_t* scrA = ws + slotBytes;
    uint8_t* scrB = ws + 2U * slotBytes;
    uint8_t* zeros = ws + 3U * slotBytes;
    aclError zRet = aclrtMemsetAsync(zeros, 32768U, 0, 32768U, stream);
    CHECK_RET(zRet == ACL_SUCCESS, OP_LOGE("aclblasCtpmv", "zero memset async failed, ret=%d", zRet);
              return ACLBLAS_STATUS_INTERNAL_ERROR);
    aclError cpRet = aclrtMemcpyAsync(snapshot, slotBytes, x, slotBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    CHECK_RET(cpRet == ACL_SUCCESS, OP_LOGE("aclblasCtpmv", "snapshot memcpy async failed, ret=%d", cpRet);
              return ACLBLAS_STATUS_INTERNAL_ERROR);
    const uint32_t vecBlocks = std::min(useNumBlocks, static_cast<uint32_t>((nU32 + 15u) / 16u));
    ctpmv_arch35_preswap_do(snapshot, scrA, scrB, tiling, useNumBlocks, stream);
    ctpmv_arch35_vec_tc_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(AP)), reinterpret_cast<uint8_t*>(x), snapshot, scrA,
        scrB, zeros, tiling, vecBlocks, stream);
    return ACLBLAS_STATUS_SUCCESS;
}

// N 路径 n<=8: 主 SIMT kernel 12 模板实例代码体积拖固定成本(~8us), 专用朴素 kernel ~3us
static aclblasStatus_t RunCtpmvTinyPath(
    aclblasHandle_t handle, const aclblasComplex* AP, aclblasComplex* x, int n, int incx, const CtpmvTilingData& tiling,
    void* stream)
{
    uint8_t* ws = nullptr;
    const aclblasStatus_t st = GetCtpmvWorkspace(handle, n, 2U, 0, &ws);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    ctpmv_arch35_tiny_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(AP)), reinterpret_cast<uint8_t*>(x), ws, tiling, stream);
    return WriteBackResult(x, ws, n, incx, tiling, 1u, stream);
}

// N 路径 8<n<=32: 行独占直出(nsmall)
static aclblasStatus_t RunCtpmvNsmallPath(
    aclblasHandle_t handle, const aclblasComplex* AP, aclblasComplex* x, int n, int incx, const CtpmvTilingData& tiling,
    void* stream)
{
    uint8_t* ws = nullptr;
    const aclblasStatus_t st = GetCtpmvWorkspace(handle, n, 2U, 0, &ws);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    ctpmv_arch35_nsmall_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(AP)), reinterpret_cast<uint8_t*>(x), ws, tiling, stream);
    return WriteBackResult(x, ws, n, incx, tiling, 1u, stream);
}

// N 路径 32<n<=128: 分行分块(8 块各认领一行段) + reduce_final 归约
static aclblasStatus_t RunCtpmvNs2Path(
    aclblasHandle_t handle, const aclblasComplex* AP, aclblasComplex* x, int n, int incx, const CtpmvTilingData& tiling,
    void* stream)
{
    uint8_t* ws = nullptr;
    const aclblasStatus_t st = GetCtpmvWorkspace(handle, n, CTPMV_NS2_CHUNKS + 1u, 0, &ws);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const size_t slotBytes = static_cast<size_t>(n) * sizeof(aclblasComplex);
    uint8_t* ySlot = ws + static_cast<size_t>(CTPMV_NS2_CHUNKS) * slotBytes;
    ctpmv_arch35_nsmall2_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(AP)), reinterpret_cast<uint8_t*>(x), ws, tiling, stream);
    ctpmv_arch35_reduce4_do(ws, ySlot, tiling, CTPMV_NS2_CHUNKS, stream);
    return WriteBackResult(x, ySlot, n, incx, tiling, 1u, stream);
}

// 通用路径: SIMT kernel + 两遍分组归约 + 写回
static aclblasStatus_t RunCtpmvGeneralPath(
    aclblasHandle_t handle, const aclblasComplex* AP, aclblasComplex* x, int n, int incx, uint32_t useNumBlocks,
    const CtpmvTilingData& tiling, void* stream)
{
    uint8_t* workspaceDevice = nullptr;
    const aclblasStatus_t st =
        GetCtpmvWorkspace(handle, n, static_cast<size_t>(useNumBlocks) + 9u, 0, &workspaceDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    ExecuteKernel(AP, x, workspaceDevice, tiling, useNumBlocks, stream);
    return WriteBackResult(x, workspaceDevice, n, incx, tiling, useNumBlocks, stream);
}

// 专用通路选择: 命中则执行并把结果写入 *st, 返回 true
static bool TryCtpmvSpecialPath(
    aclblasHandle_t handle, const aclblasComplex* AP, aclblasComplex* x, bool isOpN, int n, int incx,
    uint32_t useNumBlocks, const CtpmvTilingData& tiling, aclblasStatus_t* st, void* stream)
{
    if (!isOpN && incx == 1 && n >= 8) {
        *st = RunCtpmvVecPath(handle, AP, x, n, tiling.n, useNumBlocks, tiling, stream);
        return true;
    }
    if (isOpN && n <= static_cast<int>(CTPMV_DIRECT_MAX_N)) {
        *st = RunCtpmvTinyPath(handle, AP, x, n, incx, tiling, stream);
        return true;
    }
    if (isOpN && n > 8 && n <= 32) {
        *st = RunCtpmvNsmallPath(handle, AP, x, n, incx, tiling, stream);
        return true;
    }
    if (isOpN && n > 32 && n <= 128) {
        *st = RunCtpmvNs2Path(handle, AP, x, n, incx, tiling, stream);
        return true;
    }
    return false;
}

aclblasStatus_t aclblasCtpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCtpmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    // 校验顺序对齐 Netlib：枚举 -> n<0 -> incx==0 -> n=0 快速返回（不引用 AP/x）-> 空指针
    aclblasStatus_t st = ValidateCtpmvParams(uplo, trans, diag, n, incx);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(AP != nullptr, OP_LOGE("aclblasCtpmv", "AP must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasCtpmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0, OP_LOGE("aclblasCtpmv", "GetAivCoreCount failed");
              return ACLBLAS_STATUS_EXECUTION_FAILED);

    const uint32_t nU32 = static_cast<uint32_t>(n);
    const bool isOpN = (trans == ACLBLAS_OP_N);
    uint32_t useNumBlocks = 0;
    uint32_t numThreads = 0;
    ComputeCtpmvLaunch(isOpN, nU32, aivCoreNum, &useNumBlocks, &numThreads);

    CtpmvTilingData tiling = CalCtpmvTilingData(uplo, trans, diag, nU32, incx, numThreads);
    OP_LOGD(
        "aclblasCtpmv", "tiling: n=%u uplo=%u trans=%u diag=%u incx=%lld numThreads=%u numBlocks=%u", tiling.n,
        tiling.uplo, tiling.trans, tiling.diag, tiling.incx, tiling.numThreads, useNumBlocks);
    OP_LOGI("aclblasCtpmv", "launching kernel");

    st = ACLBLAS_STATUS_SUCCESS;
    if (TryCtpmvSpecialPath(handle, AP, x, isOpN, n, incx, useNumBlocks, tiling, &st, handle->stream)) {
        return st;
    }
    return RunCtpmvGeneralPath(handle, AP, x, n, incx, useNumBlocks, tiling, handle->stream);
}
