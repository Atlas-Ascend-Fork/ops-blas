/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <exception>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"

void strsv_kernel_do(
    uint8_t* A, uint8_t* x, uint8_t* tilingGm, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int lda, uint32_t numBlocks, void* stream);

constexpr uint32_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;

// issue #1110: 三角方程求解行间存在顺序依赖（后行消费前行结果）且 kernel 无跨核
// 同步，多核按行拆分不可行；恒单核下发。删除恒不可达的多核 tiling 脚手架
// （MAX_CORE_NUM 数组与按核分行循环），tiling 结构同步简化为 {n, lda}。
struct StrsvTilingData {
    uint32_t n;
    uint32_t lda;
};

static StrsvTilingData CalStrsvTilingData(int n, int lda)
{
    StrsvTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    return tiling;
}

static void PackStrsvVector(
    const std::vector<float>& xHost, int n, int incx, size_t absIncx, std::vector<float>& xPacked)
{
    for (int i = 0; i < n; i++) {
        const size_t offset = incx > 0 ? static_cast<size_t>(i) * absIncx : static_cast<size_t>(n - 1 - i) * absIncx;
        xPacked[i] = xHost[offset];
    }
}

static void ScatterStrsvVector(
    const std::vector<float>& xPacked, int n, int incx, size_t absIncx, std::vector<float>& xHost)
{
    for (int i = 0; i < n; i++) {
        const size_t offset = incx > 0 ? static_cast<size_t>(i) * absIncx : static_cast<size_t>(n - 1 - i) * absIncx;
        xHost[offset] = xPacked[i];
    }
}

static bool IsValidStrsvEnums(aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    return (uplo == ACLBLAS_LOWER || uplo == ACLBLAS_UPPER) &&
           (trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C) &&
           (diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT);
}

static bool HasStrsvSpanOverflow(int n, int lda, int incx)
{
    const uint64_t stride = incx < 0 ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx);
    const uint64_t extent = static_cast<uint64_t>(n - 1) * stride + 1;
    return static_cast<uint64_t>(n) * static_cast<uint64_t>(lda) > UINT32_MAX || extent > UINT32_MAX;
}

// issue #1098: kernel 侧按 BUFFER_NUM × n × sizeof(float) 字节申请 UB
//（strsv_kernel.cpp UB_SIZE=192KB），n 超过 24572 时 InitBuffer 触发设备侧失败。
// host 侧提前拒绝，返回参数错误而非执行期错误。
static bool HasStrsvUbOverflow(int n)
{
    constexpr uint64_t kStrsvUbSize = 192 * 1024;
    constexpr uint64_t kStrsvBufferNum = 2;
    constexpr uint64_t kStrsvElementBytes = 4;
    constexpr uint64_t kStrsvUbAlignBytes = 32;
    // 按 UB 32B 块对齐逐 buffer 计算：kernel 侧 aQueue/xQueue 均以 BUFFER_NUM 个
    // InitBuffer 申请，每块在 UB 上按 32B 对齐（strsv_kernel.cpp）。原实现仅从
    // 192KB 减一个 32B 块，边界 n=24569~24572 仍被放行却在设备侧 InitBuffer 失败。
    auto alignUp32 = [](uint64_t bytes) -> uint64_t {
        return (bytes + kStrsvUbAlignBytes - 1) / kStrsvUbAlignBytes * kStrsvUbAlignBytes;
    };
    const uint64_t aQueueBytes = kStrsvBufferNum * alignUp32(static_cast<uint64_t>(n) * kStrsvElementBytes);
    const uint64_t xQueueBytes = kStrsvBufferNum * alignUp32(kStrsvElementBytes);
    return aQueueBytes + xQueueBytes > kStrsvUbSize;
}

static aclblasStatus_t ValidateStrsvParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, const float* x, int incx)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasStrsv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasStrsv", "invalid n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!IsValidStrsvEnums(uplo, trans, diag)) {
        OP_LOGE(
            "aclblasStrsv", "invalid uplo=%d trans=%d diag=%d", static_cast<int>(uplo), static_cast<int>(trans),
            static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasStrsv", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasStrsv", "invalid lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr || x == nullptr) {
        OP_LOGE("aclblasStrsv", "A or x is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (HasStrsvSpanOverflow(n, lda, incx)) {
        OP_LOGE("aclblasStrsv", "matrix or vector extent exceeds uint32 range: n=%d lda=%d incx=%d", n, lda, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (HasStrsvUbOverflow(n)) {
        OP_LOGE("aclblasStrsv", "n exceeds UB capacity of the kernel: n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// issue #1028/#949：异常防护所需的资源与暂存集合。句柄集中在结构体内，
// 便于失败/异常路径统一释放（ReleaseStrsvStage）。
struct StrsvStage {
    uint8_t* xDevice = nullptr;
    uint8_t* tilingDevice = nullptr;
    size_t xSize = 0;
    size_t xSpanSize = 0;
    size_t absIncx = 0;
    StrsvTilingData tiling{};
    std::vector<float> xHost;
    std::vector<float> xPacked;
};

static void ReleaseStrsvStage(StrsvStage& st)
{
    if (st.xDevice != nullptr) {
        aclrtFree(st.xDevice);
        st.xDevice = nullptr;
    }
    if (st.tilingDevice != nullptr) {
        aclrtFree(st.tilingDevice);
        st.tilingDevice = nullptr;
    }
}

// 装载段：分配设备向量缓冲，把跨步 x 段拷回 host、按逻辑向量打包后 H2D
// （原 aclblasStrsv 内联逻辑逐行搬运；失败路径释放已取得资源后返回精确状态码）。
static aclblasStatus_t StageStrsvInput(int n, int incx, const float* x, StrsvStage& st)
{
    st.absIncx = static_cast<size_t>(incx < 0 ? -static_cast<int64_t>(incx) : incx);
    const size_t xElements = (static_cast<size_t>(n) - 1) * st.absIncx + 1;
    st.xSize = static_cast<size_t>(n) * sizeof(float);
    st.xSpanSize = xElements * sizeof(float);

    aclError aclRet = aclrtMalloc(reinterpret_cast<void**>(&st.xDevice), st.xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    // A and x are device pointers. Stage the strided x span on the host,
    // then pack the logical vector for the contiguous Arch22 kernel.
    st.xHost.resize(xElements);
    st.xPacked.resize(n);
    aclRet = aclrtMemcpy(st.xHost.data(), st.xSpanSize, x, st.xSpanSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        ReleaseStrsvStage(st);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    PackStrsvVector(st.xHost, n, incx, st.absIncx, st.xPacked);

    aclRet = aclrtMemcpy(st.xDevice, st.xSize, st.xPacked.data(), st.xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        ReleaseStrsvStage(st);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// tiling 上传段：分配 tiling 设备缓冲并 H2D（原 aclblasStrsv 内联逻辑逐行搬运）。
static aclblasStatus_t UploadStrsvTiling(int n, int lda, StrsvStage& st)
{
    st.tiling = CalStrsvTilingData(n, lda);

    aclError aclRet =
        aclrtMalloc(reinterpret_cast<void**>(&st.tilingDevice), sizeof(StrsvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        ReleaseStrsvStage(st);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMemcpy(
        st.tilingDevice, sizeof(StrsvTilingData), &st.tiling, sizeof(StrsvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        ReleaseStrsvStage(st);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// 结果回写段：kernel 结果 D2H、跨步散回原 x 存储、H2D 写回
// （原 aclblasStrsv 内联逻辑逐行搬运；失败路径释放全部资源后返回精确状态码）。
static aclblasStatus_t ScatterStrsvResult(int n, int incx, float* x, StrsvStage& st)
{
    aclError aclRet = aclrtMemcpy(st.xPacked.data(), st.xSize, st.xDevice, st.xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        ReleaseStrsvStage(st);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    ScatterStrsvVector(st.xPacked, n, incx, st.absIncx, st.xHost);
    aclRet = aclrtMemcpy(x, st.xSpanSize, st.xHost.data(), st.xSpanSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        ReleaseStrsvStage(st);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasStrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx)
{
    aclblasStatus_t status = ValidateStrsvParams(handle, uplo, trans, diag, n, A, lda, x, incx);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclrtContext currentCtx = nullptr;
    aclError aclRet = aclrtGetCurrentContext(&currentCtx);
    if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    // handle 已在入口判空，直接取 stream
    auto* h = handle;
    aclrtStream useStream = h->stream;

    // Resource allocation, staging buffers and kernel dispatch allocate via
    // std::vector/aclrt; a std::exception here must not cross the extern "C"
    // boundary (sgemm_grouped_batched contract). Parameter validation stays
    // outside the guard. Staging/upload/scatter segments are extracted into
    // helpers to keep the entry function within the single-function size gate;
    // failure paths inside the helpers release what has already been acquired.
    StrsvStage st;
    try {
        status = StageStrsvInput(n, incx, x, st);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        status = UploadStrsvTiling(n, lda, st);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }

        strsv_kernel_do(
            reinterpret_cast<uint8_t*>(const_cast<float*>(A)), st.xDevice, st.tilingDevice, uplo, trans, diag, n, lda,
            1, useStream);

        aclRet = aclrtSynchronizeStream(useStream);
        if (aclRet != ACL_SUCCESS) {
            ReleaseStrsvStage(st);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }

        status = ScatterStrsvResult(n, incx, x, st);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }

        ReleaseStrsvStage(st);
    } catch (const std::exception&) {
        // Release whatever was already acquired; handles not yet allocated are
        // still nullptr here and ReleaseStrsvStage skips them.
        ReleaseStrsvStage(st);
        OP_LOGE("aclblasStrsv", "host exception in kernel dispatch path");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
