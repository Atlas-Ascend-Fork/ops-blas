/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rotex_host.cpp
 * \brief Givens rotation (RotEx) host-side implementation.
 *        Arch35 (ascend950) host-side implementation.
 *        Dual-path: SIMD for continuous stride (incx==1 && incy==1 or incx==-1 && incy==-1),
 *        SIMT for discrete stride.
 *        S group only: FP32/FP16/BF16 x/y types with FP32 executionType.
 *        D/C/Z groups not supported on arch35 (return ACLBLAS_STATUS_NOT_SUPPORTED).
 */

#include <cstdint>
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "rotex_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/dtype_cast.h"

// ==========================================================================
//  ReadScalarValues — read c/s from host/device pointer into tiling
//  Supports S group: FP32, FP16, BF16 scalar types.
// ==========================================================================
static void ReadScalarValues(const void* c, const void* s, aclDataType csType,
                             RotExTilingData& tiling)
{
    switch (csType) {
        case ACL_FLOAT:
            tiling.cReal = *static_cast<const float*>(c);
            tiling.sReal = *static_cast<const float*>(s);
            break;
        case ACL_FLOAT16: {
            uint16_t cRaw = *static_cast<const uint16_t*>(c);
            uint16_t sRaw = *static_cast<const uint16_t*>(s);
            tiling.cReal = blas_common::HalfToFloat(cRaw);
            tiling.sReal = blas_common::HalfToFloat(sRaw);
            break;
        }
        case ACL_BF16: {
            uint16_t cRaw = *static_cast<const uint16_t*>(c);
            uint16_t sRaw = *static_cast<const uint16_t*>(s);
            tiling.cReal = blas_common::Bf16ToFloat(cRaw);
            tiling.sReal = blas_common::Bf16ToFloat(sRaw);
            break;
        }
        default:
            tiling.cReal = 0.0f;
            tiling.sReal = 0.0f;
            break;
    }
}

// ==========================================================================
//  CheckAndCopyScalar — check if scalar pointer is on device, copy to host
//  Returns ACLBLAS_STATUS_SUCCESS on success,
//          ACLBLAS_STATUS_INTERNAL_ERROR on aclrtMemcpy failure.
//  issue #1082: device 标量经 handle->stream 上的异步拷贝 + 流同步读回，
//  与同 stream 上先前提交的算子（如 aclblasRotg 的输出）建立流序依赖，
//  避免读到旧值（对齐 strmm ResolveStrmmScalars 口径）。
// ==========================================================================
static aclblasStatus_t CheckAndCopyScalar(const void* ptr, size_t csBytes,
    uint8_t* hostBuf, bool& isDevice, const char* name, aclrtStream stream)
{
    aclrtPtrAttributes attr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &attr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasRotEx",
                "aclrtPointerGetAttributes failed for %s, ret=%d", name, aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    isDevice = (attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    if (!isDevice) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMemcpyAsync(hostBuf, csBytes, ptr, csBytes,
                                    ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasRotEx",
                "aclrtMemcpyAsync %s from device to host failed, ret=%d", name, ret);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasRotEx",
                "aclrtSynchronizeStream for %s D2H failed, ret=%d", name, ret);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  DetermineTilingKey — compute tilingKey from stride pattern + executionType
//  Return values:
//    0 — SIMD S group (ACL_FLOAT, contiguous stride)
//    2 — SIMT path (discrete stride)
// ==========================================================================
static uint32_t DetermineTilingKey(int incx, int incy, aclDataType xType, aclDataType yType,
                                   aclDataType executionType)
{
    // RotEx 逐元素独立 (每个 x[k] 只和对应 y[k] 配对), 不依赖遍历顺序.
    // 因此双向负步长 (incx==-1 && incy==-1) 等价于正向遍历, 可走 SIMD 连续路径.
    bool isContiguous = (incx == 1 && incy == 1) || (incx == -1 && incy == -1);
    // SIMD S 组: 仅当连续步长 + xType==yType + ACL_FLOAT executionType
    if (isContiguous && xType == yType && executionType == ACL_FLOAT) {
        return 0;
    }
    // 所有其他情况 (离散步长) 走 SIMT
    return 2;
}

// ==========================================================================
//  SetupRotExTilingData — fill RotExTilingData fields from host parameters
//  Computes tilingKey-dependent fields (tileSize/perCoreN for SIMD,
//  nthreads for SIMT) and static fields (kx, ky, types, etc.).
// ==========================================================================
static void SetupRotExTilingData(RotExTilingData& tiling, int n, int incx, int incy,
    uint32_t tilingKey, aclDataType executionType,
    aclDataType xType, aclDataType yType, aclDataType csType,
    uint32_t numBlocks)
{
    tiling.n = n;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.kx = (incx >= 0) ? 0 : (1LL - n) * incx;
    tiling.ky = (incy >= 0) ? 0 : (1LL - n) * incy;
    tiling.tilingKey = tilingKey;
    tiling.executionType = static_cast<uint32_t>(executionType);
    tiling.xType = static_cast<uint32_t>(xType);
    tiling.yType = static_cast<uint32_t>(yType);
    tiling.csType = static_cast<uint32_t>(csType);

    if (tilingKey == 0) {
        // SIMD S 组: UB 切分配置
        if (xType == ACL_FLOAT) {
            // 纯 FP32: 248*1024 / (4 * 1) = 63488 bytes / 4 = 15872 elements
            uint32_t ubPerQueue = UB_SIZE / 4;
            tiling.tileSize = ubPerQueue / sizeof(float);  // 15872
        } else {
            // 混合精度: 24B/element, 对齐到 16 (32/2)
            uint32_t rawTileSize = UB_SIZE / 24;
            tiling.tileSize = (rawTileSize / 16) * 16;
        }
        tiling.perCoreN = static_cast<uint32_t>(n) / numBlocks;
        tiling.remainder = static_cast<uint32_t>(n) % numBlocks;
    }
    if (tilingKey == 2) {
        // SIMT 路径: 计算 nthreads
        uint32_t avgElements = CeilDiv<uint32_t>(static_cast<uint32_t>(n), numBlocks);
        tiling.nthreads = std::min(
            CeilAlign<uint32_t>(avgElements, SIMT_MIN_THREAD_NUM),
            SIMT_MAX_THREAD_NUM);
    }
}

// ==========================================================================
//  LogRotExLaunch — log tiling data and kernel launch info via dlog
// ==========================================================================
static void LogRotExLaunch(const RotExTilingData& tiling, uint32_t tilingKey,
    uint32_t numBlocks, uint32_t aivCoreNum)
{
    OP_LOGD("aclblasRotEx",
            "tiling: n=%d, key=%u, blocks=%u, incx=%d, incy=%d, kx=%ld, ky=%ld",
            tiling.n, tilingKey, numBlocks, tiling.incx, tiling.incy,
            tiling.kx, tiling.ky);
    OP_LOGD("aclblasRotEx",
            "tiling: cReal=%f, sReal=%f, "
            "execType=%u, xType=%u, csType=%u",
            tiling.cReal, tiling.sReal,
            tiling.executionType, tiling.xType, tiling.csType);
    if (tilingKey == 0) {
        OP_LOGD("aclblasRotEx", "tiling: tileSize=%u, perCoreN=%u, remainder=%u",
                tiling.tileSize, tiling.perCoreN, tiling.remainder);
    }
    if (tilingKey == 2) {
        OP_LOGD("aclblasRotEx", "tiling: nthreads=%u", tiling.nthreads);
    }
    OP_LOGI("aclblasRotEx", "launching kernel: blocks=%u, cores=%u, key=%u",
            numBlocks, aivCoreNum, tilingKey);
}

// ==========================================================================
//  CheckRotExTypeSupport — type validation helper
//  Returns ACLBLAS_STATUS_NOT_SUPPORTED for unsupported type combinations.
// ==========================================================================
static aclblasStatus_t CheckRotExTypeSupport(aclDataType executionType,
    aclDataType xType, aclDataType yType, aclDataType csType)
{
    // 仅支持 S 组 (executionType=ACL_FLOAT); D/C/Z 组在 arch35 上不支持
    if (executionType != ACL_FLOAT) {
        OP_LOGE("aclblasRotEx",
                "executionType=%d is not supported on arch35, "
                "only ACL_FLOAT is supported",
                static_cast<int>(executionType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    // xType ∈ {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}
    if (xType != ACL_FLOAT && xType != ACL_FLOAT16 && xType != ACL_BF16) {
        OP_LOGE("aclblasRotEx",
                "executionType=ACL_FLOAT, xType must be ACL_FLOAT/ACL_FLOAT16/"
                "ACL_BF16, got %d",
                static_cast<int>(xType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    // yType ∈ {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}
    if (yType != ACL_FLOAT && yType != ACL_FLOAT16 && yType != ACL_BF16) {
        OP_LOGE("aclblasRotEx",
                "executionType=ACL_FLOAT, yType must be ACL_FLOAT/ACL_FLOAT16/"
                "ACL_BF16, got %d",
                static_cast<int>(yType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    // csType ∈ {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}
    if (csType != ACL_FLOAT && csType != ACL_FLOAT16 && csType != ACL_BF16) {
        OP_LOGE("aclblasRotEx",
                "executionType=ACL_FLOAT, csType must be ACL_FLOAT/ACL_FLOAT16/"
                "ACL_BF16, got %d",
                static_cast<int>(csType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    // 仅支持对称类型: xType == yType == csType
    if (xType != yType || xType != csType) {
        OP_LOGE("aclblasRotEx",
                "only symmetric types are supported (xType==yType==csType), "
                "got xType=%d, yType=%d, csType=%d",
                static_cast<int>(xType), static_cast<int>(yType),
                static_cast<int>(csType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  ValidateRotExParams — parameter validation
//  Only S group (ACL_FLOAT executionType) is supported on arch35.
//  D/C/Z groups return ACLBLAS_STATUS_NOT_SUPPORTED.
// ==========================================================================
static aclblasStatus_t ValidateRotExParams(
    aclblasHandle_t handle, int n,
    void* x, aclDataType xType, int incx,
    void* y, aclDataType yType, int incy,
    const void* c, const void* s, aclDataType csType,
    aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasRotEx", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasRotEx", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasRotEx", "incx must not be zero, got %d", incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasRotEx", "incy must not be zero, got %d", incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasRotEx", "x must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasRotEx", "y must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (c == nullptr) {
        OP_LOGE("aclblasRotEx", "c must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (s == nullptr) {
        OP_LOGE("aclblasRotEx", "s must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    return CheckRotExTypeSupport(executionType, xType, yType, csType);
}

// ==========================================================================
//  LaunchRotExKernel — tiling computation + kernel launch
// ==========================================================================
static aclblasStatus_t LaunchRotExKernel(
    aclblasHandle_t handle, int n,
    void* x, aclDataType xType, int incx,
    void* y, aclDataType yType, int incy,
    const void* c, const void* s, aclDataType csType,
    aclDataType executionType)
{
    auto* h = static_cast<_aclblas_handle*>(handle);

    // n == 0: 无需启动 kernel，直接返回成功
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 1. 获取核心数
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasRotEx", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // 2. 确定 TilingKey 和 block 数
    uint32_t tilingKey = DetermineTilingKey(incx, incy, xType, yType, executionType);
    uint32_t numBlocks = (tilingKey == 0)
        ? std::min(static_cast<uint32_t>(n), aivCoreNum)
        : std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM),
                   aivCoreNum);

    // 3. 检测 c/s 指针类型, 若为 device 指针则复制到 host 临时缓冲区
    size_t csBytes = (csType == ACL_FLOAT) ? 4 : 2;
    uint8_t cHostBuf[4] = {0};
    uint8_t sHostBuf[4] = {0};
    bool cIsDevice = false;
    bool sIsDevice = false;
    aclblasStatus_t ret = CheckAndCopyScalar(c, csBytes, cHostBuf, cIsDevice, "c", h->stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = CheckAndCopyScalar(s, csBytes, sHostBuf, sIsDevice, "s", h->stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    const void* cRead = cIsDevice ? cHostBuf : c;
    const void* sRead = sIsDevice ? sHostBuf : s;

    // 4. 填充 TilingData
    RotExTilingData tiling{};
    SetupRotExTilingData(tiling, n, incx, incy, tilingKey, executionType,
                         xType, yType, csType, numBlocks);
    ReadScalarValues(cRead, sRead, csType, tiling);

    // 5. 日志 + 异步 launch kernel (无需同步, 上层调用方负责)
    LogRotExLaunch(tiling, tilingKey, numBlocks, aivCoreNum);
    rotex_kernel_do(static_cast<uint8_t*>(x), static_cast<uint8_t*>(y),
                    tiling, numBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  aclblasRotEx — public API entry
// ==========================================================================
aclblasStatus_t aclblasRotEx(
    aclblasHandle_t handle,
    int n,
    void* x,
    aclDataType xType,
    int incx,
    void* y,
    aclDataType yType,
    int incy,
    const void* c,
    const void* s,
    aclDataType csType,
    aclDataType executionType)
{
    // 参数校验 (包含 handle 空指针 / n<0 / n==0 快速返回)
    aclblasStatus_t st = ValidateRotExParams(
        handle, n, x, xType, incx, y, yType, incy, c, s, csType, executionType);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Tiling 计算 + Kernel launch
    return LaunchRotExKernel(
        handle, n, x, xType, incx, y, yType, incy, c, s, csType, executionType);
}
