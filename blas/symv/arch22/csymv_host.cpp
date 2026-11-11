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
 * \file csymv_host.cpp
 * \brief CSYMV Host implementation for Atlas A2/A3 (DAV_2201)
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "csymv_tiling_data.h"

namespace {

struct ComplexScalarSource {
    aclblasComplex value{0.0f, 0.0f};
    bool isDevice = false;
};

struct CsymvArguments {
    aclblasHandle_t handle;
    aclblasFillMode_t uplo;
    int n;
    const aclblasComplex* alpha;
    const aclblasComplex* a;
    int lda;
    const aclblasComplex* x;
    int incx;
    const aclblasComplex* beta;
    aclblasComplex* y;
    int incy;
};

struct CsymvLaunchPlan {
    CsymvTilingData tiling{};
    aclrtStream stream = nullptr;
    uint32_t numBlocks = 0U;
    uint8_t* workspace = nullptr;
};

inline bool IsZero(const aclblasComplex& value) { return value.real == 0.0f && value.imag == 0.0f; }

inline bool IsOne(const aclblasComplex& value) { return value.real == 1.0f && value.imag == 0.0f; }

uint32_t ChooseKernelRoute(int n, int lda, int incx)
{
    if (incx != 1 || n < 16 || n > 8192 || static_cast<uint64_t>(lda) * sizeof(aclblasComplex) > UINT32_MAX) {
        return CSYMV_GENERIC;
    }
    return CSYMV_OWNER;
}

aclblasStatus_t PrepareScalarSource(const aclblasComplex* ptr, const char* name, ComplexScalarSource& source)
{
    aclrtPtrAttributes attributes{};
    aclError ret = aclrtPointerGetAttributes(ptr, &attributes);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCsymv", "aclrtPointerGetAttributes failed for %s, ret=%d", name, ret);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    source.isDevice = attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE;
    if (!source.isDevice) {
        source.value = *ptr;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ResolveDeviceScalars(
    const aclblasComplex* alpha, const aclblasComplex* beta, aclrtStream stream, ComplexScalarSource& alphaSource,
    ComplexScalarSource& betaSource)
{
    bool copied = false;
    if (alphaSource.isDevice) {
        aclError ret = aclrtMemcpyAsync(
            &alphaSource.value, sizeof(alphaSource.value), alpha, sizeof(*alpha), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCsymv", "copying alpha from device failed, ret=%d", ret);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        copied = true;
    }
    if (betaSource.isDevice) {
        aclError ret = aclrtMemcpyAsync(
            &betaSource.value, sizeof(betaSource.value), beta, sizeof(*beta), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCsymv", "copying beta from device failed, ret=%d", ret);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        copied = true;
    }
    if (copied) {
        aclError ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCsymv", "synchronizing scalar copies failed, ret=%d", ret);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateFixedParams(
    aclblasFillMode_t uplo, int n, int lda, int incx, int incy, const aclblasComplex* alpha, const aclblasComplex* beta)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCsymv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasCsymv", "invalid lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        OP_LOGE("aclblasCsymv", "incx and incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclblasCsymv", "alpha and beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

CsymvTilingData MakeTilingData(
    uint32_t numBlocks, int n, int lda, aclblasFillMode_t uplo, const ComplexScalarSource& alpha,
    const ComplexScalarSource& beta, int incx, int incy)
{
    CsymvTilingData tiling{};
    tiling.tiled = ChooseKernelRoute(n, lda, incx);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.alphaReal = alpha.value.real;
    tiling.alphaImag = alpha.value.imag;
    tiling.betaReal = beta.value.real;
    tiling.betaImag = beta.value.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.alphaIsDevice = alpha.isDevice ? 1U : 0U;
    tiling.betaIsDevice = beta.isDevice ? 1U : 0U;
    tiling.useCoreNum = numBlocks;
    return tiling;
}

aclblasStatus_t ValidateCall(const CsymvArguments& args, bool& complete)
{
    complete = false;
    if (args.handle == nullptr) {
        OP_LOGE("aclblasCsymv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (args.n < 0) {
        OP_LOGE("aclblasCsymv", "invalid n=%d", args.n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (args.n == 0) {
        complete = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ValidateFixedParams(args.uplo, args.n, args.lda, args.incx, args.incy, args.alpha, args.beta);
}

aclblasStatus_t PrepareExecution(
    const CsymvArguments& args, ComplexScalarSource& alphaSource, ComplexScalarSource& betaSource, bool& complete)
{
    aclblasStatus_t status = PrepareScalarSource(args.alpha, "alpha", alphaSource);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = PrepareScalarSource(args.beta, "beta", betaSource);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    const bool hasMissingDataPointer = args.a == nullptr || args.x == nullptr || args.y == nullptr;
    if (hasMissingDataPointer) {
        status = ResolveDeviceScalars(args.alpha, args.beta, args.handle->stream, alphaSource, betaSource);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    const bool scalarValuesKnown = (!alphaSource.isDevice && !betaSource.isDevice) || hasMissingDataPointer;
    if (!scalarValuesKnown) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const bool alphaIsZero = IsZero(alphaSource.value);
    complete = alphaIsZero && IsOne(betaSource.value);
    if (!alphaIsZero && (args.a == nullptr || args.x == nullptr)) {
        OP_LOGE("aclblasCsymv", "A and x must not be nullptr when alpha is nonzero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!complete && args.y == nullptr) {
        OP_LOGE("aclblasCsymv", "y must not be nullptr when the result is written");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ResolveAivCoreCount(uint32_t& count)
{
    // Cache values only, never runtime handles or device allocations. Device reset
    // cannot leave a dangling resource here; switching devices refreshes the count.
    static thread_local int32_t cachedDevice = -1;
    static thread_local uint32_t cachedCount = 0U;
    int32_t device = -1;
    const aclError result = aclrtGetDevice(&device);
    if (result != ACL_SUCCESS) {
        OP_LOGE("aclblasCsymv", "aclrtGetDevice failed, ret=%d", result);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (cachedDevice != device || cachedCount == 0U) {
        // SyncAll requires every block to fit the current device allocation,
        // which can expose fewer AIVs than the full SoC on a virtual NPU.
        int64_t queriedCount = 0;
        const aclError queryResult =
            aclrtGetDeviceInfo(static_cast<uint32_t>(device), ACL_DEV_ATTR_VECTOR_CORE_NUM, &queriedCount);
        if (queryResult != ACL_SUCCESS || queriedCount <= 0) {
            OP_LOGE("aclblasCsymv", "aclrtGetDeviceInfo vector core count failed, ret=%d", queryResult);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        cachedDevice = device;
        cachedCount = static_cast<uint32_t>(queriedCount);
    }
    count = cachedCount;
    return ACLBLAS_STATUS_SUCCESS;
}

void ConfigureTrianglePlan(const CsymvArguments& args, uint32_t aivCoreNum, CsymvLaunchPlan& plan)
{
    // Small macro grids leave AIVs idle with 128x128 tiles. Use 64x64 tiles
    // for intermediate sizes and short tails; larger grids use 128x128 tiles
    // to reduce coefficient preparation and workspace traffic.
    const bool useSmallMacro = (args.n >= 640 && args.n <= 768) || (args.n > 1024 && args.n <= 1280);
    const bool useLargeMacro = args.n >= 769 && args.n <= 4096 && !useSmallMacro;
    if (plan.tiling.tiled != CSYMV_OWNER || (!useSmallMacro && !useLargeMacro)) {
        return;
    }
    auto workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(args.handle));
    const uint32_t tile = useSmallMacro ? 64U : 128U;
    const uint32_t sourceBlocks = (static_cast<uint32_t>(args.n) + tile - 1U) / tile;
    const uint64_t requiredBytes = 2ULL * tile * sourceBlocks * sourceBlocks * sizeof(float);
    if (workspace == nullptr || GetEffectiveWorkspaceSize(args.handle) < requiredBytes) {
        return;
    }
    // Each triangular task publishes both directions; finalizers reuse producer UB after SyncAll.
    plan.workspace = workspace;
    plan.tiling.tiled = tile == 64U ? CSYMV_TRIANGLE_SMALL : CSYMV_OWNER;
    plan.tiling.triangle = 1U;
    plan.tiling.sourceBlocks = sourceBlocks;
    plan.numBlocks = std::min(sourceBlocks * (sourceBlocks + 1U) / 2U, aivCoreNum);
    plan.tiling.useCoreNum = plan.numBlocks;
    plan.tiling.finalizerBlocks = std::min((static_cast<uint32_t>(args.n) + 31U) / 32U, plan.numBlocks);
}

void ConfigurePhaseWindowPlan(
    const CsymvArguments& args, const ComplexScalarSource& alpha, const ComplexScalarSource& beta, uint32_t aivCoreNum,
    CsymvLaunchPlan& plan)
{
    constexpr uint32_t tile = 64U;
    constexpr uint32_t maxBlocks = 8U;
    if (args.n < 4 * static_cast<int>(tile) + 2 || args.n > static_cast<int>(maxBlocks * tile) - 2 || args.n % 4 != 2 ||
        args.lda % 4 != 2 || args.lda > 131070 || args.uplo != ACLBLAS_LOWER || args.incx != 1 || args.incy != 1 ||
        alpha.isDevice || beta.isDevice || !IsOne(alpha.value) || !IsZero(beta.value)) {
        return;
    }
    const uint32_t blocks = (static_cast<uint32_t>(args.n) + tile - 1U) / tile;
    const uint32_t tail = static_cast<uint32_t>(args.n) % tile;
    // Sparse last tiles keep the fixed producer cost but reduce useful work.
    const uint32_t minimumTail = blocks == 5U ? 54U : 30U;
    const uint32_t tasks = blocks * (blocks + 1U) / 2U;
    auto workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(args.handle));
    uintptr_t pointerBits = reinterpret_cast<uintptr_t>(args.a);
    pointerBits |= reinterpret_cast<uintptr_t>(args.x);
    pointerBits |= reinterpret_cast<uintptr_t>(args.y);
    pointerBits |= reinterpret_cast<uintptr_t>(workspace);
    if (tail < minimumTail || aivCoreNum < tasks || workspace == nullptr ||
        GetEffectiveWorkspaceSize(args.handle) < 1536U * tasks || (pointerBits & 31U) != 0U) {
        return;
    }
    plan.workspace = workspace;
    plan.tiling.triangle = 2U;
    plan.tiling.sourceBlocks = blocks;
    plan.tiling.finalizerBlocks = 2U * blocks;
    plan.tiling.useCoreNum = tasks;
    plan.numBlocks = tasks;
}

aclblasStatus_t BuildLaunchPlan(
    const CsymvArguments& args, const ComplexScalarSource& alphaSource, const ComplexScalarSource& betaSource,
    CsymvLaunchPlan& plan)
{
    uint32_t aivCoreNum = 0U;
    const aclblasStatus_t status = ResolveAivCoreCount(aivCoreNum);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const bool tiled = ChooseKernelRoute(args.n, args.lda, args.incx) != CSYMV_GENERIC;
    const uint32_t outputGroups = tiled ? (static_cast<uint32_t>(args.n) + 15U) / 16U : static_cast<uint32_t>(args.n);
    plan.numBlocks = std::min(outputGroups, aivCoreNum);
    plan.tiling =
        MakeTilingData(plan.numBlocks, args.n, args.lda, args.uplo, alphaSource, betaSource, args.incx, args.incy);
    ConfigureTrianglePlan(args, aivCoreNum, plan);
    ConfigurePhaseWindowPlan(args, alphaSource, betaSource, aivCoreNum, plan);
    plan.stream = args.handle->stream;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchCsymv(
    const CsymvArguments& args, const ComplexScalarSource& alphaSource, const ComplexScalarSource& betaSource,
    const CsymvLaunchPlan& plan)
{
    auto alpha = alphaSource.isDevice ? reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(args.alpha)) : nullptr;
    auto beta = betaSource.isDevice ? reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(args.beta)) : nullptr;
    auto a = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(args.a));
    auto x = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(args.x));
    auto y = reinterpret_cast<uint8_t*>(args.y);
    const aclError result =
        csymv_kernel_do(alpha, a, x, beta, y, plan.workspace, plan.numBlocks, plan.tiling, plan.stream);
    if (result != ACL_SUCCESS) {
        OP_LOGE("aclblasCsymv", "kernel launch failed, ret=%d", result);
    }
    return result == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
}

aclblasStatus_t ExecuteCsymv(const CsymvArguments& args)
{
    bool complete = false;
    aclblasStatus_t status = ValidateCall(args, complete);
    if (status != ACLBLAS_STATUS_SUCCESS || complete) {
        return status;
    }
    ComplexScalarSource alphaSource;
    ComplexScalarSource betaSource;
    status = PrepareExecution(args, alphaSource, betaSource, complete);
    if (status != ACLBLAS_STATUS_SUCCESS || complete) {
        return status;
    }
    CsymvLaunchPlan plan;
    status = BuildLaunchPlan(args, alphaSource, betaSource, plan);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return LaunchCsymv(args, alphaSource, betaSource, plan);
}

} // namespace

aclblasStatus_t aclblasCsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    const CsymvArguments args{handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy};
    return ExecuteCsymv(args);
}
