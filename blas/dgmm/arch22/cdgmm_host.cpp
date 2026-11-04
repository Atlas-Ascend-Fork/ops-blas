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
 * \file cdgmm_host.cpp
 * \brief Host-side implementation of aclblasCdgmm (arch22).
 *        Complex diagonal matrix-matrix multiplication: C = diag(x) * A
 *        (mode=LEFT).  Row-major storage.  RIGHT mode returns
 *        ACLBLAS_STATUS_NOT_SUPPORTED.
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cdgmm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

constexpr uint32_t CDGMM_DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t CDGMM_COMPLEX_NUM = 2;
constexpr uint32_t CDGMM_FP32_BYTE_SIZE = 4;
constexpr uint32_t CDGMM_MAX_DATA_COUNT = 32 * 1024 / sizeof(float);
constexpr size_t CDGMM_WORKSPACE_SIZE = 1024;

// ==========================================================================
// RAII wrapper for device buffers — guarantees aclrtFree on all paths.
// ==========================================================================
class AclDeviceBuffer {
public:
    AclDeviceBuffer() = default;
    ~AclDeviceBuffer()
    {
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }

    aclError Allocate(size_t size)
    {
        data_ = nullptr;
        return aclrtMalloc(reinterpret_cast<void**>(&data_), size, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    uint8_t* Get() const { return data_; }

    AclDeviceBuffer(const AclDeviceBuffer&) = delete;
    AclDeviceBuffer& operator=(const AclDeviceBuffer&) = delete;

private:
    uint8_t* data_ = nullptr;
};

// ==========================================================================
// Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateCdgmmParams(
    aclblasSideMode_t mode, int m, int n, const aclblasComplex* A, int lda, const aclblasComplex* x, int incx,
    aclblasComplex* C, int ldc)
{
    if (mode != ACLBLAS_SIDE_LEFT && mode != ACLBLAS_SIDE_RIGHT) {
        OP_LOGE("aclblasCdgmm", "mode must be SIDE_LEFT(141) or SIDE_RIGHT(142), got %d", static_cast<int>(mode));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (mode == ACLBLAS_SIDE_RIGHT) {
        OP_LOGE("aclblasCdgmm", "ACLBLAS_SIDE_RIGHT is not supported in the current row-major implementation");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (m < 0) {
        OP_LOGE("aclblasCdgmm", "m must be >= 0, got %d", m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE("aclblasCdgmm", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasCdgmm", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // issue #780: INT32_MIN 步长无合法使用场景（x 寻址必然越出任何实际分配），
    // 对齐 scnrm2（arch35）显式拒绝口径
    if (incx == INT32_MIN) {
        OP_LOGE("aclblasCdgmm", "incx must not be INT32_MIN");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasCdgmm", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, n)) {
        OP_LOGE("aclblasCdgmm", "ldc must be >= max(1, n), got ldc=%d, n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && (A == nullptr || x == nullptr || C == nullptr)) {
        OP_LOGE("aclblasCdgmm", "A/x/C must not be nullptr when m>0 and n>0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == C && lda != ldc) {
        OP_LOGE("aclblasCdgmm", "in-place execution (A==C) requires lda==ldc, got lda=%d, ldc=%d", lda, ldc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Unified numBlocks calculation — shared by tiling and host launch.
// ==========================================================================
static uint32_t CalcCdgmmNumBlocks(uint32_t m, uint32_t aivCoreNum)
{
    uint32_t clamped = std::max(1U, std::min(aivCoreNum, CDGMM_DEFAULT_VECTOR_NUM));
    return std::min(clamped, m);
}

// ==========================================================================
// Tiling: row-split decomposition (Task 1 + Task 2)
// ==========================================================================
static CdgmmTilingData CalCdgmmTilingData(
    uint32_t mode, uint32_t m, uint32_t n, int32_t incx, uint32_t lda, uint32_t ldc, uint32_t aivCoreNum)
{
    CdgmmTilingData tiling{};

    tiling.mode = mode;
    tiling.m = m;
    tiling.n = n;
    tiling.incx = incx;
    tiling.lda = lda;
    tiling.ldc = ldc;

    // Guard: empty matrix — return zero-initialized tiling without division.
    if (m == 0) {
        return tiling;
    }

    const uint32_t numBlocks = CalcCdgmmNumBlocks(m, aivCoreNum);
    if (numBlocks == 0) {
        return tiling;
    }
    const uint32_t rowsPerCore = m / numBlocks;
    const uint32_t remainder = m % numBlocks;

    uint32_t currRow = 0;
    for (uint32_t i = 0; i < numBlocks; i++) {
        uint32_t currCount = (i < remainder) ? rowsPerCore + 1 : rowsPerCore;
        tiling.startRow[i] = currRow;
        tiling.rowCount[i] = currCount;
        currRow += currCount;
    }

    return tiling;
}

// ==========================================================================
// Aug offset table builder
// ==========================================================================
static std::vector<uint32_t> CreateAugCdgmm()
{
    uint32_t complexCount = CDGMM_MAX_DATA_COUNT / CDGMM_COMPLEX_NUM;
    std::vector<uint32_t> aug(CDGMM_MAX_DATA_COUNT);

    for (uint32_t i = 0; i < complexCount; i++) {
        aug[CDGMM_COMPLEX_NUM * i] = CDGMM_FP32_BYTE_SIZE * i;
        aug[CDGMM_COMPLEX_NUM * i + 1] = CDGMM_FP32_BYTE_SIZE * (i + complexCount);
    }
    return aug;
}

// ==========================================================================
// Build launch configuration (tiling, numBlocks, aug)
// ==========================================================================
struct CdgmmLaunchConfig {
    CdgmmTilingData tiling{};
    uint32_t numBlocks = 0;
    std::vector<uint32_t> aug;
};

static aclblasStatus_t BuildCdgmmLaunchConfig(
    aclblasSideMode_t mode, int m, int n, int lda, int incx, int ldc, uint32_t aivCoreNum, CdgmmLaunchConfig& config)
{
    uint32_t modeNorm = (mode == ACLBLAS_SIDE_LEFT) ? CDGMM_MODE_LEFT : CDGMM_MODE_RIGHT;

    config.tiling = CalCdgmmTilingData(
        modeNorm, static_cast<uint32_t>(m), static_cast<uint32_t>(n), incx, static_cast<uint32_t>(lda),
        static_cast<uint32_t>(ldc), aivCoreNum);

    config.numBlocks = CalcCdgmmNumBlocks(static_cast<uint32_t>(m), aivCoreNum);
    config.aug = CreateAugCdgmm();

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Launch: allocate device buffers, copy, kernel, sync (Task 3)
// ==========================================================================
static aclblasStatus_t LaunchCdgmm(
    aclrtStream stream, const aclblasComplex* A, const aclblasComplex* x, aclblasComplex* C,
    const CdgmmLaunchConfig& config)
{
    AclDeviceBuffer augDevice;
    AclDeviceBuffer workspaceDevice;
    AclDeviceBuffer tilingDevice;

    size_t augByteSize = config.aug.size() * sizeof(uint32_t);

    aclError aclRet = augDevice.Allocate(augByteSize);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCdgmm", "aclrtMalloc aug failed. ERROR: %d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = workspaceDevice.Allocate(CDGMM_WORKSPACE_SIZE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCdgmm", "aclrtMalloc workspace failed. ERROR: %d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = tilingDevice.Allocate(sizeof(CdgmmTilingData));
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCdgmm", "aclrtMalloc tiling failed. ERROR: %d", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMemcpy(augDevice.Get(), augByteSize, config.aug.data(), augByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCdgmm", "aclrtMemcpy aug failed. ERROR: %d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(
        tilingDevice.Get(), sizeof(CdgmmTilingData), &config.tiling, sizeof(CdgmmTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCdgmm", "aclrtMemcpy tiling failed. ERROR: %d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    cdgmm_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(x)), reinterpret_cast<GM_ADDR>(C), augDevice.Get(),
        workspaceDevice.Get(), tilingDevice.Get(), config.numBlocks, stream);

    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCdgmm", "aclrtSynchronizeStream failed. ERROR: %d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Public API — orchestrates validate → build config → launch
// ==========================================================================
extern "C" aclblasStatus_t aclblasCdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCdgmm", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t st = ValidateCdgmmParams(mode, m, n, A, lda, x, incx, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCdgmm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    CdgmmLaunchConfig config;
    BuildCdgmmLaunchConfig(mode, m, n, lda, incx, ldc, aivCoreNum, config);

    return LaunchCdgmm(handle->stream, A, x, C, config);
}
