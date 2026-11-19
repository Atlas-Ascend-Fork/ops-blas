/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <climits>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "cgeru_tiling_data.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void cgeru_kernel_do(const CgeruTilingData& tiling, uint32_t numBlocks, void* stream);

static uint32_t CalCgeruBlockNum(int m, int n)
{
    // GetAivCoreCount() 每次都走 platform_ascendc 的平台查询；核数在同一进程/设备内不变，
    // 缓存一次即可，省掉每次调用几微秒的 host 侧固定开销（小尺寸用例上占比可观）。
    static uint32_t aivCoreNum = 0;
    if (aivCoreNum == 0) {
        aivCoreNum = GetAivCoreCount();
        if (aivCoreNum == 0) {
            aivCoreNum = 1;
        }
    }

    // 形状自适应核数：列数少时 40 路并发把 GM 端口挤满、每核流水又太短，实测少用核更快
    // （191 例官方形状 × 核数 16/20/24/28/32/36/40 逐例扫描，见 work/analysis1104/rule.py：
    //  列数 <= 1200 时 24 核有 1.0~1.8µs 收益且不产生新的不及格，其余形状仍是满核最优）。
    uint32_t cap = aivCoreNum;
    const long long workItems = static_cast<long long>(m) * static_cast<long long>(n);
    if (n > 0 && n <= 1200 && workItems <= 2000000LL && cap > 24U) {
        cap = 24U;
    }

    return std::min(static_cast<uint32_t>(n), cap);
}

static CgeruTilingData CalCgeruTilingData(
    int m, int n, int lda, int incx, int incy, uint32_t coreNum, const aclblasComplex& alpha)
{
    CgeruTilingData tiling = {};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.alphaReal = alpha.real;
    tiling.alphaImag = alpha.imag;
    tiling.rowTile = CGERU_ROW_TILE_MAX;

    if (coreNum == 0) {
        coreNum = 1;
    }

    tiling.colsPerCore = static_cast<uint32_t>(n + static_cast<int>(coreNum) - 1) / coreNum;
    tiling.useCoreNum = coreNum;

    return tiling;
}

// 步长的跨度是否在可寻址范围内：负步长时逻辑首元素在物理向量末尾，
// 物理跨度为 1 + (len - 1) * |inc|。INT_MIN 的 |inc| 是 2^31，
// 只有 len <= 1 时跨度（1 个元素）合法，len > 1 必然越过地址空间，提前拒绝。
static bool CgeruStrideSpanValid(int length, int inc)
{
    if (inc == 0) {
        return false;
    }
    if (inc != INT_MIN) {
        return true;
    }
    return length <= 1;
}

static aclblasStatus_t ValidateCgeruScalars(int m, int n, const aclblasComplex* alpha, int incx, int incy, int lda)
{
    if (m < 0 || n < 0) {
        OP_LOGE("aclblasCgeru", "matrix dimensions m and n must be non-negative, got m=%d, n=%d", m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // 参数校验顺序与 arch35（ValidateCgeruShape）保持一致：先校验 alpha/incx/incy/lda，
    // 再处理 m==0/n==0 与 alpha==0 的 quick return。零维输入同样要经过标量校验，
    // 否则同一个非法步长在 arch22 与 arch35 上会得到不同状态码。
    if (alpha == nullptr) {
        OP_LOGE("aclblasCgeru", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!CgeruStrideSpanValid(m, incx)) {
        OP_LOGE("aclblasCgeru", "incx must be non-zero and its span must be addressable, got incx=%d, m=%d", incx, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!CgeruStrideSpanValid(n, incy)) {
        OP_LOGE("aclblasCgeru", "incy must be non-zero and its span must be addressable, got incy=%d, n=%d", incy, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) {
        OP_LOGE("aclblasCgeru", "lda must be at least max(1, m), got lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateCgeruPointers(const aclblasComplex* x, const aclblasComplex* y,
                                             const aclblasComplex* A)
{
    if (x == nullptr) {
        OP_LOGE("aclblasCgeru", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasCgeru", "y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasCgeru", "A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCgeruKernel(aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha,
                                         const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
                                         aclblasComplex* A, int lda)
{
    auto* h = handle;
    aclrtStream useStream = h->stream;
    if (useStream == nullptr) {
        OP_LOGE("aclblasCgeru", "stream not initialized");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    const uint32_t numBlocks = CalCgeruBlockNum(m, n);
    CgeruTilingData tiling = CalCgeruTilingData(m, n, lda, incx, incy, numBlocks, *alpha);
    tiling.A = reinterpret_cast<uint64_t>(A);
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.y = reinterpret_cast<uint64_t>(y);

    cgeru_kernel_do(tiling, tiling.useCoreNum, useStream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCgeru(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgeru", "handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t status = ValidateCgeruScalars(m, n, alpha, incx, incy, lda);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // alpha == (0, 0) 时 A 保持不变，直接快速返回（Netlib cgeru quick return 语义）。
    // 任务书 §2.4：x/y/A 仅在 alpha 非零时才要求判空，故该快返回须早于判空。
    if (alpha->real == 0.0f && alpha->imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    status = ValidateCgeruPointers(x, y, A);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    return LaunchCgeruKernel(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
