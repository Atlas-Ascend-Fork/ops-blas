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
 * \file host_utils.h
 * \brief Host-side utilities.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// ==========================================================================
//  CeilDiv — integer ceiling division:  ceil(a / b)
// ==========================================================================

template <typename R = uint32_t, typename T1, typename T2>
static inline R CeilDiv(T1 a, T2 b)
{
    static_assert(
        std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value, "CeilDiv arguments must be arithmetic types");
    static_assert(std::is_arithmetic<R>::value, "CeilDiv return type must be arithmetic");
    auto ra = static_cast<R>(a);
    auto rb = static_cast<R>(b);
    if (rb == 0) {
        return std::numeric_limits<R>::max();
    }
    // Overflow guard: if a + b - 1 wraps around, return max
    if (ra + rb - 1 < ra) {
        return std::numeric_limits<R>::max();
    }
    return (ra + rb - 1) / rb;
}

// ==========================================================================
//  CeilAlign — round val up to a multiple of align:  CeilDiv(val,align)*align
// ==========================================================================

template <typename R = uint32_t, typename T1, typename T2>
static inline R CeilAlign(T1 val, T2 align)
{
    return CeilDiv<R>(val, align) * static_cast<R>(align);
}

// ==========================================================================
//  GetAivCoreCount — get the number of AIV (vector) cores on current device
// ==========================================================================

static inline uint32_t GetAivCoreCount()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("[ERROR] GetAivCoreCount Failed", "PlatformAscendCManager is null, return 0");
        return 0;
    }
    return platform->GetCoreNumAiv();
}

// ==========================================================================
//  GetAicCoreCount — get the number of AIC (Cube) cores on current device
// ==========================================================================

static inline uint32_t GetAicCoreCount()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("[ERROR] GetAicCoreCount Failed", "PlatformAscendCManager is null, return 0");
        return 0;
    }
    return platform->GetCoreNumAic();
}

// ==========================================================================
//  CheckPtrLocation — determine whether a pointer lives on host or device.
//  Uses aclrtPointerGetAttributes. On query failure returns ACLBLAS_STATUS_INVALID_VALUE.
//  Shared by all operators that support host-or-device scalar pointers.
// ==========================================================================
static inline aclblasStatus_t CheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("[ERROR] CheckPtrLocation", "aclrtPointerGetAttributes failed, aclRet=%d", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  ReadHostScalarFromDevice — copy one float scalar from a device pointer to
//  host through the handle stream (D2H async + sync). Used by operators whose
//  tiling carries host scalar values while accepting host-or-device alpha
//  pointers (e.g. ssyr/sspr, issue #510).
// ==========================================================================
static inline aclblasStatus_t ReadHostScalarFromDevice(
    const float* devicePtr, float& hostVal, aclrtStream stream, const char* opTag)
{
    hostVal = 0.0f;
    aclError aclRet = aclrtMemcpyAsync(
        &hostVal, sizeof(float), devicePtr, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opTag, "aclrtMemcpyAsync scalar D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opTag, "aclrtSynchronizeStream for scalar D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
