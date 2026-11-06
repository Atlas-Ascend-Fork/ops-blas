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
 * \file caxpy_host.cpp
 * \brief
 */

#include <cstdint>
#include <cstddef>
#include <limits>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <new>
#include "acl/acl.h"
#include "acl/acl_rt_api.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "caxpy_dispatch_policy.h"
#include "caxpy_kernel.h"
#include "caxpy_tiling_data.h"

constexpr uint32_t FLOATS_PER_COMPLEX = 2;
constexpr uint32_t K_FACTOR_4 = 4;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t MAX_DATA_COUNT = 38 * 1024 / sizeof(float);
constexpr uint32_t STRIDED_TILE_COMPLEX_COUNT = 512;
constexpr uint32_t BLOCKED_FLOATS_PER_COMPLEX = 8;
constexpr uint32_t GATHER_PADDING_COUNT = 64;
constexpr uint32_t X_NORMALIZE_OFFSET_COUNT =
    STRIDED_TILE_COMPLEX_COUNT * BLOCKED_FLOATS_PER_COMPLEX + GATHER_PADDING_COUNT;
constexpr uint32_t STORE_OFFSET_COUNT = STRIDED_TILE_COMPLEX_COUNT * FLOATS_PER_COMPLEX + GATHER_PADDING_COUNT;
constexpr uint32_t X_NORMALIZE_OFFSET_START = MAX_DATA_COUNT;
constexpr uint32_t STORE_OFFSET_START = X_NORMALIZE_OFFSET_START + X_NORMALIZE_OFFSET_COUNT;
constexpr uint32_t Y_GATHER_OFFSET_COUNT = STRIDED_TILE_COMPLEX_COUNT * FLOATS_PER_COMPLEX + GATHER_PADDING_COUNT;
constexpr uint32_t Y_GATHER_OFFSET_START = STORE_OFFSET_START + STORE_OFFSET_COUNT;
constexpr uint32_t DENSE_SWAP_OFFSET_START = Y_GATHER_OFFSET_START + Y_GATHER_OFFSET_COUNT;
constexpr uint32_t MASK_DATA_COUNT = DENSE_SWAP_OFFSET_START + MAX_DATA_COUNT;
static_assert(STORE_OFFSET_START + STORE_OFFSET_COUNT <= MASK_DATA_COUNT);
static_assert(Y_GATHER_OFFSET_START + Y_GATHER_OFFSET_COUNT <= MASK_DATA_COUNT);
static_assert(DENSE_SWAP_OFFSET_START + MAX_DATA_COUNT <= MASK_DATA_COUNT);

// CaxpyTilingData is defined once in the shared header and included by both the
// host and the device kernel; see caxpy_tiling_data.h.

CaxpyTilingData CalTilingData(
    uint32_t n, uint32_t vecCoreNum, float alphaReal, float alphaImag, int64_t incx, int64_t incy,
    uint32_t logicalBase = 0, uint32_t totalN = 0)
{
    CaxpyTilingData tilingData{};
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;
    tilingData.totalN = totalN == 0 ? n : totalN;
    tilingData.activeBlocks = vecCoreNum;
    tilingData.alphaReal = alphaReal;
    tilingData.alphaImag = alphaImag;
    tilingData.incx = incx;
    tilingData.incy = incy;
    (void)logicalBase;
    return tilingData;
}

// Magnitude of a BLAS increment. Computed as -(inc+1)+1 rather than -inc so
// that INT64_MIN would not overflow -- though callers reject INT64_MIN upstream
// (ValidateCaxpyArguments), this keeps the helper correct in isolation.
uint64_t AbsIncrement(int64_t increment)
{
    return increment > 0 ? static_cast<uint64_t>(increment) : static_cast<uint64_t>(-(increment + 1)) + 1;
}

bool CheckedVectorSpan(int64_t n, int64_t increment, uint64_t& span)
{
    const uint64_t absIncrement = AbsIncrement(increment);
    if (n <= 1) {
        span = 1;
        return true;
    }
    const uint64_t logicalDistance = static_cast<uint64_t>(n - 1);
    if (logicalDistance > (std::numeric_limits<uint64_t>::max() - 1) / absIncrement) {
        return false;
    }
    span = 1 + logicalDistance * absIncrement;
    return span <= std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
}

// Maps a logical element index (0..n-1) to its physical index in memory. For a
// negative increment BLAS walks the vector backwards, so logical 0 sits at the
// far end.
uint64_t PhysicalIndex(uint64_t logicalIndex, uint64_t n, int64_t increment)
{
    const uint64_t absIncrement = AbsIncrement(increment);
    return increment > 0 ? logicalIndex * absIncrement : (n - 1 - logicalIndex) * absIncrement;
}

// The mask buffer holds precomputed gather/scatter offset tables so the device
// kernel never has to derive strided addresses on its hot path. Each Gen*Offsets
// routine fills one section (see the *_OFFSET_START layout constants above).

// Interleaved -> planar: real parts land in the first half, imaginary in the
// second, so downstream vector ops see two contiguous real-valued lanes.
void GenComputeOffsets(uint32_t* offsets)
{
    const uint32_t complexCount = MAX_DATA_COUNT / FLOATS_PER_COMPLEX;
    for (uint32_t i = 0; i < complexCount; i++) {
        offsets[FLOATS_PER_COMPLEX * i] = K_FACTOR_4 * i;
        offsets[FLOATS_PER_COMPLEX * i + 1] = K_FACTOR_4 * (i + complexCount);
    }
}

// Swaps real/imaginary byte positions within each complex element in place.
void GenDenseSwapOffsets(uint32_t* offsets)
{
    const uint32_t complexCount = MAX_DATA_COUNT / FLOATS_PER_COMPLEX;
    for (uint32_t i = 0; i < complexCount; ++i) {
        offsets[FLOATS_PER_COMPLEX * i] = (FLOATS_PER_COMPLEX * i + 1) * sizeof(float);
        offsets[FLOATS_PER_COMPLEX * i + 1] = FLOATS_PER_COMPLEX * i * sizeof(float);
    }
}

// Gathers strided x into the blocked (8-float-per-complex) working layout. For
// increments where the shared pipeline's span read still fits its lowScratch
// UB buffer (UseCaxpySharedXSpanPipeline), the span read brings the data in
// contiguously and only the blocked layout stride remains; larger increments
// fall back to the disjoint pipeline, which reads each element individually
// and does not use this span-read addressing at all.
void GenXNormalizeOffsets(uint32_t* offsets, uint32_t xIncrement)
{
    for (uint32_t i = 0; i < STRIDED_TILE_COMPLEX_COUNT; ++i) {
        const uint32_t sourceByteOffset = UseCaxpySharedXSpanPipeline(static_cast<int64_t>(xIncrement)) ?
                                              i * xIncrement * FLOATS_PER_COMPLEX * sizeof(float) :
                                              i * BLOCKED_FLOATS_PER_COMPLEX * sizeof(float);
        for (uint32_t lane = 0; lane < BLOCKED_FLOATS_PER_COMPLEX; ++lane) {
            offsets[i * BLOCKED_FLOATS_PER_COMPLEX + lane] =
                lane == 1 ? sourceByteOffset + sizeof(float) : sourceByteOffset;
        }
    }
}

// Scatters results from the blocked layout back to packed interleaved complex.
void GenStoreOffsets(uint32_t* offsets)
{
    for (uint32_t i = 0; i < STRIDED_TILE_COMPLEX_COUNT; ++i) {
        const uint32_t sourceByteOffset = i * BLOCKED_FLOATS_PER_COMPLEX * sizeof(float);
        offsets[i * FLOATS_PER_COMPLEX] = sourceByteOffset;
        offsets[i * FLOATS_PER_COMPLEX + 1] = sourceByteOffset + sizeof(float);
    }
}

// Gathers strided y (real, imag) pairs for the accumulation step.
void GenYGatherOffsets(uint32_t* offsets, uint32_t yIncrement)
{
    for (uint32_t i = 0; i < STRIDED_TILE_COMPLEX_COUNT; ++i) {
        const uint32_t sourceByteOffset = i * yIncrement * FLOATS_PER_COMPLEX * sizeof(float);
        offsets[i * FLOATS_PER_COMPLEX] = sourceByteOffset;
        offsets[i * FLOATS_PER_COMPLEX + 1] = sourceByteOffset + sizeof(float);
    }
}

void GenMaskData(uint32_t* maskData, uint32_t xIncrement, uint32_t yIncrement)
{
    std::fill(maskData, maskData + MASK_DATA_COUNT, 0);
    GenComputeOffsets(maskData);
    GenXNormalizeOffsets(maskData + X_NORMALIZE_OFFSET_START, xIncrement);
    GenStoreOffsets(maskData + STORE_OFFSET_START);
    GenYGatherOffsets(maskData + Y_GATHER_OFFSET_START, yIncrement);
    GenDenseSwapOffsets(maskData + DENSE_SWAP_OFFSET_START);
}

// Reuse pinned staging only after its copy event completes. Querying an event
// never drains the caller's stream; outstanding uploads get distinct slots.
aclblasStatus_t AcquireCaxpyUpload(aclblasHandle_t handle, size_t bytes, CaxpyUploadBuffer*& upload)
{
    for (auto* slot = handle->caxpy_uploads; slot != nullptr; slot = slot->next) {
        if (slot->size < bytes || (slot->pending && !slot->recorded)) {
            continue;
        }
        if (slot->pending) {
            aclrtEventRecordedStatus state = ACL_EVENT_RECORDED_STATUS_NOT_READY;
            const aclError result = aclrtQueryEventStatus(slot->event, &state);
            CHECK_RET(result == ACL_SUCCESS, LOG_PRINT("aclrtQueryEventStatus failed. ERROR: %d\n", result);
                      return ACLBLAS_STATUS_INTERNAL_ERROR);
            if (state != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
                continue;
            }
        }
        slot->pending = false;
        slot->recorded = false;
        upload = slot;
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* slot = new (std::nothrow) CaxpyUploadBuffer;
    CHECK_RET(slot != nullptr, LOG_PRINT("Caxpy upload allocation failed.\n"); return ACLBLAS_STATUS_ALLOC_FAILED);
    aclError result = aclrtMallocHost(&slot->host, bytes);
    CHECK_RET(result == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", result); delete slot;
              return ACLBLAS_STATUS_ALLOC_FAILED);
    result = aclrtCreateEventWithFlag(&slot->event, ACL_EVENT_SYNC);
    CHECK_RET(result == ACL_SUCCESS, LOG_PRINT("aclrtCreateEventWithFlag failed. ERROR: %d\n", result);
              aclrtFreeHost(slot->host); delete slot; return ACLBLAS_STATUS_INTERNAL_ERROR);
    slot->size = bytes;
    slot->next = handle->caxpy_uploads;
    handle->caxpy_uploads = slot;
    upload = slot;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t SubmitCaxpyUpload(aclblasHandle_t handle, void* destination, size_t bytes, CaxpyUploadBuffer& upload)
{
    const aclError copyResult =
        aclrtMemcpyAsync(destination, bytes, upload.host, bytes, ACL_MEMCPY_HOST_TO_DEVICE, handle->stream);
    CHECK_RET(copyResult == ACL_SUCCESS, LOG_PRINT("aclrtMemcpyAsync failed. ERROR: %d\n", copyResult);
              return ACLBLAS_STATUS_INTERNAL_ERROR);
    upload.pending = true;
    const aclError eventResult = aclrtRecordEvent(upload.event, handle->stream);
    // On record failure the pending source remains owned, but cannot be reused.
    CHECK_RET(eventResult == ACL_SUCCESS, LOG_PRINT("aclrtRecordEvent failed. ERROR: %d\n", eventResult);
              return ACLBLAS_STATUS_INTERNAL_ERROR);
    upload.recorded = true;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t GetCaxpyMaskCache(aclblasHandle_t handle, uint64_t absIncx, uint64_t absIncy, uint8_t** maskDevice)
{
    if (handle->caxpy_mask_cache == nullptr) {
        void* buffer = nullptr;
        const aclError aclRet = aclrtMalloc(&buffer, MASK_DATA_COUNT * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
                  return ACLBLAS_STATUS_ALLOC_FAILED);
        handle->caxpy_mask_cache = buffer;
    }
    if (handle->caxpy_mask_cache_incx != absIncx || handle->caxpy_mask_cache_incy != absIncy) {
        CaxpyUploadBuffer* upload = nullptr;
        aclblasStatus_t status = AcquireCaxpyUpload(handle, MASK_DATA_COUNT * sizeof(uint32_t), upload);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
        GenMaskData(
            static_cast<uint32_t*>(upload->host), static_cast<uint32_t>(absIncx), static_cast<uint32_t>(absIncy));
        // This overwrite follows earlier kernels in the same stream. Invalidate
        // the key first so a failed event record cannot leave a stale cache hit.
        handle->caxpy_mask_cache_incx = 0;
        handle->caxpy_mask_cache_incy = 0;
        status = SubmitCaxpyUpload(handle, handle->caxpy_mask_cache, MASK_DATA_COUNT * sizeof(uint32_t), *upload);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
        handle->caxpy_mask_cache_incx = absIncx;
        handle->caxpy_mask_cache_incy = absIncy;
    }
    *maskDevice = reinterpret_cast<uint8_t*>(handle->caxpy_mask_cache);
    return ACLBLAS_STATUS_SUCCESS;
}

struct CaxpyExecution {
    aclblasHandle_t handle;
    int64_t n;
    int64_t incx;
    int64_t incy;
    uint32_t numBlocks;
    bool useStridedKernel;
    bool packX;
    bool packY;
    CaxpyKernelVariant kernelVariant;
    aclrtStream stream;
    uint8_t* tilingDevice;
    uint8_t* maskDevice;
    CaxpyTilingData tiling;
};

struct CaxpyPackedBuffers {
    uint8_t* x = nullptr;
    uint8_t* y = nullptr;
};

aclblasStatus_t ValidateCaxpyArguments(
    aclblasHandle_t handle, int64_t n, const aclblasComplex* alpha, const aclblasComplex* x, int64_t incx,
    const aclblasComplex* y, int64_t incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCaxpy", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasCaxpy", "n must be >= 0, got %lld", static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasCaxpy", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        OP_LOGE("aclblasCaxpy", "incx and incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == std::numeric_limits<int64_t>::min() || incy == std::numeric_limits<int64_t>::min()) {
        OP_LOGE("aclblasCaxpy", "incx and incy must not be INT64_MIN");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && (x == nullptr || y == nullptr)) {
        OP_LOGE("aclblasCaxpy", "x and y must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > std::numeric_limits<uint32_t>::max() / FLOATS_PER_COMPLEX) {
        OP_LOGE("aclblasCaxpy", "n is too large: %lld", static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    uint64_t span = 0;
    if (n > 0 && (!CheckedVectorSpan(n, incx, span) || !CheckedVectorSpan(n, incy, span))) {
        OP_LOGE("aclblasCaxpy", "vector span overflows addressable memory");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t BuildCaxpyExecution(
    aclblasHandle_t handle, int64_t n, const aclblasComplex& alpha, int64_t incx, int64_t incy,
    CaxpyExecution& execution)
{
    uint32_t numBlocks = GetAivCoreCount();
    CHECK_RET(numBlocks != 0, LOG_PRINT("GetAivCoreCount failed.\n"); return ACLBLAS_STATUS_EXECUTION_FAILED);
    numBlocks = std::min(numBlocks, DEFAULT_VECTOR_NUM);
    numBlocks = std::min(numBlocks, static_cast<uint32_t>(n));
    const uint64_t absIncx = AbsIncrement(incx);
    const uint64_t absIncy = AbsIncrement(incy);
    const uint64_t maxIncrement = std::max(absIncx, absIncy);
    const bool incrementFits = maxIncrement <= std::numeric_limits<uint32_t>::max() / sizeof(aclblasComplex);
    const bool addressFits =
        n <= 1 || (incrementFits && static_cast<uint64_t>(n - 1) <=
                                        std::numeric_limits<uint32_t>::max() / (FLOATS_PER_COMPLEX * maxIncrement));
    const bool contiguous = incx == 1 && incy == 1;
    const bool useStridedKernel = !contiguous && incrementFits && addressFits;
    const CaxpyKernelVariant kernelVariant =
        SelectCaxpyKernelVariant(static_cast<uint64_t>(n), incx, incy, useStridedKernel, numBlocks);
    CHECK_RET(
        sizeof(CaxpyTilingData) <= GetEffectiveWorkspaceSize(handle),
        LOG_PRINT("workspace need %zu, available %zu\n", sizeof(CaxpyTilingData), GetEffectiveWorkspaceSize(handle));
        return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint8_t* maskDevice = nullptr;
    if (kernelVariant != CaxpyKernelVariant::STRIDED_SCALAR) {
        const aclblasStatus_t maskStatus = GetCaxpyMaskCache(handle, absIncx, absIncy, &maskDevice);
        CHECK_RET(maskStatus == ACLBLAS_STATUS_SUCCESS, return maskStatus);
    }
    // When a vector is strided but we are not taking the strided kernel path,
    // fall back to packing it into a contiguous scratch buffer and running the
    // dense kernel. Unit-stride vectors need no packing.
    const bool packX = !contiguous && !useStridedKernel && incx != 1;
    const bool packY = !contiguous && !useStridedKernel && incy != 1;
    execution = {
        handle,
        n,
        incx,
        incy,
        numBlocks,
        useStridedKernel,
        packX,
        packY,
        kernelVariant,
        handle->stream,
        reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle)),
        maskDevice,
        CalTilingData(
            static_cast<uint32_t>(n), numBlocks, alpha.real, alpha.imag, useStridedKernel ? incx : 1,
            useStridedKernel ? incy : 1)};
    return ACLBLAS_STATUS_SUCCESS;
}

// Growth may wait only on the packing fallback. The old allocation remains
// owned if synchronization or allocation fails; queued work never loses it.
aclblasStatus_t EnsureCaxpyPackedBuffer(aclblasHandle_t handle, size_t bytes, void*& buffer, size_t& capacity)
{
    if (capacity >= bytes) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (buffer != nullptr) {
        const aclblasStatus_t status = SynchronizeHandleStream(handle);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
    }
    void* replacement = nullptr;
    const aclError result = aclrtMalloc(&replacement, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(result == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", result);
              return ACLBLAS_STATUS_ALLOC_FAILED);
    if (buffer != nullptr) {
        aclrtFree(buffer);
    }
    buffer = replacement;
    capacity = bytes;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t AllocatePackedBuffers(const CaxpyExecution& execution, CaxpyPackedBuffers& buffers)
{
    const size_t bytes = static_cast<size_t>(execution.n) * sizeof(aclblasComplex);
    auto* handle = execution.handle;
    if (execution.packX) {
        const aclblasStatus_t status =
            EnsureCaxpyPackedBuffer(handle, bytes, handle->caxpy_packed_x, handle->caxpy_packed_x_size);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
        buffers.x = static_cast<uint8_t*>(handle->caxpy_packed_x);
    }
    if (execution.packY) {
        const aclblasStatus_t status =
            EnsureCaxpyPackedBuffer(handle, bytes, handle->caxpy_packed_y, handle->caxpy_packed_y_size);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
        buffers.y = static_cast<uint8_t*>(handle->caxpy_packed_y);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t CopyStridedVector(
    uint8_t* packed, uint8_t* strided, uint64_t n, int64_t increment, aclrtStream stream, bool unpack)
{
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const size_t elementBytes = sizeof(aclblasComplex);
    if (n == 1) {
        // A one-element vector has no physical stride, including INT_MAX and
        // negative increments. Avoid 2D pitch limits and batch-copy restrictions.
        uint8_t* destination = unpack ? strided : packed;
        const uint8_t* source = unpack ? packed : strided;
        const aclError result =
            aclrtMemcpyAsync(destination, elementBytes, source, elementBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        CHECK_RET(result == ACL_SUCCESS, LOG_PRINT("aclrtMemcpyAsync failed. ERROR: %d\n", result);
                  return ACLBLAS_STATUS_INTERNAL_ERROR);
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint64_t absIncrement = AbsIncrement(increment);
    if (increment > 0) {
        uint8_t* destination = unpack ? strided : packed;
        const uint8_t* source = unpack ? packed : strided;
        const size_t destinationPitch = unpack ? absIncrement * elementBytes : elementBytes;
        const size_t sourcePitch = unpack ? elementBytes : absIncrement * elementBytes;
        const aclError result = aclrtMemcpy2dAsync(
            destination, destinationPitch, source, sourcePitch, elementBytes, n, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        CHECK_RET(result == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy2dAsync failed. ERROR: %d\n", result);
                  return ACLBLAS_STATUS_INTERNAL_ERROR);
        return ACLBLAS_STATUS_SUCCESS;
    }

    std::vector<uint8_t*> destinations(n);
    std::vector<uint8_t*> sources(n);
    std::vector<size_t> destinationSizes(n, elementBytes);
    std::vector<size_t> copySizes(n, elementBytes);
    for (uint64_t i = 0; i < n; ++i) {
        const size_t packedOffset = static_cast<size_t>(i) * elementBytes;
        const size_t stridedOffset = static_cast<size_t>((n - 1 - i) * absIncrement) * elementBytes;
        destinations[i] = unpack ? strided + stridedOffset : packed + packedOffset;
        sources[i] = unpack ? packed + packedOffset : strided + stridedOffset;
    }
    aclrtMemcpyBatchAttr attribute{};
    attribute.dstLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    attribute.srcLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    const aclError result = aclrtMemcpyBatchAsync(
        destinations.data(), destinationSizes.data(), sources.data(), copySizes.data(), n, attribute, stream);
    CHECK_RET(result == ACL_SUCCESS, LOG_PRINT("aclrtMemcpyBatchAsync failed. ERROR: %d\n", result);
              return ACLBLAS_STATUS_INTERNAL_ERROR);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t PreparePackedInputs(
    const CaxpyExecution& execution, const aclblasComplex* x, aclblasComplex* y, CaxpyPackedBuffers& buffers)
{
    aclblasStatus_t status = AllocatePackedBuffers(execution, buffers);
    CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
    if (execution.packX) {
        status = CopyStridedVector(
            buffers.x, const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(x)), execution.n, execution.incx,
            execution.stream, false);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
    }
    if (execution.packY) {
        status = CopyStridedVector(
            buffers.y, reinterpret_cast<uint8_t*>(y), execution.n, execution.incy, execution.stream, false);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t UploadTiling(const CaxpyExecution& execution, const CaxpyTilingData& tiling)
{
    CaxpyUploadBuffer* upload = nullptr;
    const aclblasStatus_t status = AcquireCaxpyUpload(execution.handle, sizeof(tiling), upload);
    CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
    *static_cast<CaxpyTilingData*>(upload->host) = tiling;
    return SubmitCaxpyUpload(execution.handle, execution.tilingDevice, sizeof(tiling), *upload);
}

CaxpyTilingData BuildStridedTiling(
    const CaxpyExecution& execution, const aclblasComplex& alpha, uint32_t count, uint32_t blocks)
{
    CaxpyTilingData tiling =
        CalTilingData(count, blocks, alpha.real, alpha.imag, execution.incx, execution.incy, 0, count);

    // Only the shared pipeline's run_shared_pipeline/shared_tile_bounds read
    // this field; the disjoint and scalar kernel entry points ignore it and
    // keep using the fixed CAXPY_STRIDED_TILE_COUNT capacity, so it is safe
    // to always populate it here for every strided variant.
    tiling.sharedTileCapacity = static_cast<uint32_t>(CaxpySharedTileCapacity(AbsIncrement(execution.incx)));
    return tiling;
}

uint64_t CaxpySegmentStorageOffset(uint64_t totalN, uint64_t segmentStart, uint64_t segmentN, int64_t increment)
{
    const uint64_t absIncrement = AbsIncrement(increment);
    return increment > 0 ? segmentStart * absIncrement : (totalN - segmentStart - segmentN) * absIncrement;
}

void LaunchCaxpyStridedPipelined(const CaxpyExecution& execution, uint8_t* x, uint8_t* y, uint32_t blocks)
{
    if (execution.kernelVariant == CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC ||
        execution.kernelVariant == CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_SCALAR_TAIL) {
        caxpy_strided_shared_kernel_do(
            x, execution.maskDevice, y, nullptr, execution.tilingDevice, blocks, execution.stream);
        return;
    }
    caxpy_strided_disjoint_kernel_do(
        x, execution.maskDevice, y, nullptr, execution.tilingDevice, blocks, execution.stream);
}

aclblasStatus_t LaunchCaxpyStridedSegment(
    const CaxpyExecution& execution, const aclblasComplex& alpha, const aclblasComplex* x, aclblasComplex* y,
    uint64_t segmentStart, uint64_t segmentN, bool scalar)
{
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(execution.numBlocks, segmentN));
    CaxpyTilingData tiling = BuildStridedTiling(execution, alpha, static_cast<uint32_t>(segmentN), blocks);
    const aclblasStatus_t status = UploadTiling(execution, tiling);
    CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);

    const uint64_t totalN = static_cast<uint64_t>(execution.n);
    const uint64_t xOffset = CaxpySegmentStorageOffset(totalN, segmentStart, segmentN, execution.incx);
    const uint64_t yOffset = CaxpySegmentStorageOffset(totalN, segmentStart, segmentN, execution.incy);
    auto* segmentX = const_cast<aclblasComplex*>(x + xOffset);
    auto* segmentY = y + yOffset;
    if (scalar) {
        caxpy_strided_scalar_kernel_do(
            reinterpret_cast<uint8_t*>(segmentX), reinterpret_cast<uint8_t*>(segmentY), execution.tilingDevice, blocks,
            execution.stream);
    } else {
        LaunchCaxpyStridedPipelined(
            execution, reinterpret_cast<uint8_t*>(segmentX), reinterpret_cast<uint8_t*>(segmentY), blocks);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchCaxpy(
    const CaxpyExecution& execution, const aclblasComplex& alpha, const aclblasComplex* x, aclblasComplex* y,
    const CaxpyPackedBuffers& buffers)
{
    if (execution.useStridedKernel) {
        if (execution.kernelVariant == CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_SCALAR_TAIL ||
            execution.kernelVariant == CaxpyKernelVariant::STRIDED_DISJOINT_PIPELINED_SCALAR_TAIL) {
            const uint64_t tailN = CaxpyFinalWaveCount(static_cast<uint64_t>(execution.n), execution.numBlocks);
            const uint64_t prefixN = static_cast<uint64_t>(execution.n) - tailN;
            aclblasStatus_t status = LaunchCaxpyStridedSegment(execution, alpha, x, y, 0, prefixN, false);
            CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
            status = LaunchCaxpyStridedSegment(execution, alpha, x, y, prefixN, tailN, true);
            CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
            return ACLBLAS_STATUS_SUCCESS;
        }
        const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(execution.numBlocks, execution.n));
        CaxpyTilingData tiling = BuildStridedTiling(execution, alpha, static_cast<uint32_t>(execution.n), blocks);
        const aclblasStatus_t status = UploadTiling(execution, tiling);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
        if (execution.kernelVariant == CaxpyKernelVariant::STRIDED_SCALAR) {
            caxpy_strided_scalar_kernel_do(
                const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(x)), reinterpret_cast<uint8_t*>(y),
                execution.tilingDevice, blocks, execution.stream);
        } else {
            LaunchCaxpyStridedPipelined(
                execution, const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(x)), reinterpret_cast<uint8_t*>(y),
                blocks);
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t status = UploadTiling(execution, execution.tiling);
    CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
    caxpy_kernel_do(
        execution.packX ? buffers.x : const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(x)), execution.maskDevice,
        execution.packY ? buffers.y : reinterpret_cast<uint8_t*>(y), nullptr, execution.tilingDevice,
        execution.numBlocks, execution.stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t FinishCaxpy(const CaxpyExecution& execution, aclblasComplex* y, CaxpyPackedBuffers& buffers)
{
    if (execution.packY) {
        const aclblasStatus_t status = CopyStridedVector(
            buffers.y, reinterpret_cast<uint8_t*>(y), execution.n, execution.incy, execution.stream, true);
        CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
    }
    // Scratch belongs to the handle and is reused in stream order, so neither
    // the packX-only path nor unpacking y requires a host-side wait or free.
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCaxpy(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, aclblasComplex* y,
    int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCaxpy", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t argumentStatus = ValidateCaxpyArguments(handle, n, alpha, x, incx, y, incy);
    CHECK_RET(argumentStatus == ACLBLAS_STATUS_SUCCESS, return argumentStatus);
    CaxpyExecution execution{};
    const aclblasStatus_t buildStatus = BuildCaxpyExecution(handle, n, *alpha, incx, incy, execution);
    CHECK_RET(buildStatus == ACLBLAS_STATUS_SUCCESS, return buildStatus);
    CaxpyPackedBuffers buffers;
    const aclblasStatus_t prepareStatus = PreparePackedInputs(execution, x, y, buffers);
    CHECK_RET(prepareStatus == ACLBLAS_STATUS_SUCCESS, return prepareStatus);
    const aclblasStatus_t launchStatus = LaunchCaxpy(execution, *alpha, x, y, buffers);
    CHECK_RET(launchStatus == ACLBLAS_STATUS_SUCCESS, return launchStatus);
    return FinishCaxpy(execution, y, buffers);
}
