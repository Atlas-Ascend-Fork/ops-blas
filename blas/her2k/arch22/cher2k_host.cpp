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
#include <cstdint>
#include <limits>
#include <vector>

#include "securec.h"
#include "cann_ops_blas.h"
#include "cher2k_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"

namespace {

constexpr uint32_t CHER2K_N_ALIGN = 256U;
// The medium-K packed SG3 producer owns 128x128 tiles.  Aligning this graph to
// 256 inflated irregular tails (for example n=2143) without satisfying any
// stronger Cube contract.
constexpr uint32_t CHER2K_DIRECT_N_ALIGN = 128U;
constexpr uint32_t CHER2K_DIRECT_OUTPUT_N_ALIGN = 128U;
constexpr uint32_t CHER2K_SMALL_CUBE_N_ALIGN = 64U;
constexpr uint32_t CHER2K_SMALL_CUBE_PATH = 2U;
// The active arch22 kernel schedules independent output tiles: one AIC owns a
// whole 256-row band, so no paired-C (128-row) schedule exists.
constexpr uint32_t CHER2K_CUBE_M = 256U;
constexpr uint32_t CHER2K_CUBE_N = 128U;
constexpr uint32_t CHER2K_K_ALIGN = 64U;
constexpr uint32_t CHER2K_DIRECT_K_ALIGN = 64U;
// Above this reduction length, use four products to limit cancellation.
constexpr uint32_t CHER2K_THREE_M_K_LIMIT = 2048U;

void BuildCher2kInterleaveTable(std::vector<uint32_t>& offsets, uint32_t tableBase, uint32_t rows, uint32_t cols)
{
    const uint32_t planeElements = rows * cols;
    for (uint32_t col = 0U; col < cols; ++col) {
        for (uint32_t row = 0U; row < rows; ++row) {
            const uint32_t output = tableBase + (col * rows + row) * 2U;
            const uint32_t planar = col * rows + row;
            offsets[output] = planar * sizeof(float);
            offsets[output + 1U] = (planeElements + planar) * sizeof(float);
        }
    }
}

void BuildCher2kTransposeTable(std::vector<uint32_t>& offsets)
{
    for (uint32_t row = 0U; row < CHER2K_TRANSPOSE_OFFSET_ROWS; ++row) {
        for (uint32_t col = 0U; col < CHER2K_TRANSPOSE_OFFSET_COLS; ++col) {
            offsets[CHER2K_TRANSPOSE_OFFSET + row * CHER2K_TRANSPOSE_OFFSET_COLS + col] =
                (col * CHER2K_TRANSPOSE_OFFSET_ROWS + row) * sizeof(float);
        }
    }
}

void BuildCher2kPostTransposeTable(std::vector<uint32_t>& offsets)
{
    for (uint32_t row = 0U; row < CHER2K_POST_TRANSPOSE_ROWS; ++row) {
        for (uint32_t col = 0U; col < CHER2K_POST_TRANSPOSE_COLS; ++col) {
            offsets[CHER2K_POST_TRANSPOSE_OFFSET + row * CHER2K_POST_TRANSPOSE_COLS + col] =
                (col * CHER2K_POST_TRANSPOSE_ROWS + row) * sizeof(float);
        }
    }
}

struct Cher2kPathConfig {
    bool performanceScalar;
    bool smallOutputAligned;
    bool smallCubeEligible;
    bool directHermitianCandidate;
    bool directOutputCandidate;
    bool sg3MmadEligible;
    uint32_t nAlign;
    uint32_t kAlign;
};

bool IsCher2kPerformanceScalar(const aclblasComplex& alpha, float beta, bool deviceScalarFast)
{
    return deviceScalarFast || (alpha.real == 1.0f && alpha.imag == 0.0f && beta == 0.0f);
}

bool IsCher2kSmallOutputAligned(aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int ldc)
{
    return trans != ACLBLAS_OP_N || (uplo == ACLBLAS_UPPER && (n <= 2 || (ldc % 4) == 0));
}

bool IsCher2kSmallCubeEligible(bool smallOutputAligned, bool performanceScalar, aclblasOperation_t trans, int n, int k)
{
    return smallOutputAligned && performanceScalar && trans == ACLBLAS_OP_N && k <= n && n > 32 && n <= 128;
}

bool IsCher2kDirectHermitianCandidate(bool performanceScalar, aclblasOperation_t trans, int n, int k)
{
    // OP_C uses complete padded workspace tiles; raw n need not be aligned.
    // The producer zero-pads its input tails and the consumer bounds C stores
    // by n/uplo. Keep the same reduction contract as useThreeM below.
    // OP_N chooses its single-panel, scalar-guarded graph in the launcher.
    const uint64_t alignedK =
        k > 0 ? (static_cast<uint64_t>(k) + CHER2K_K_ALIGN - 1U) / CHER2K_K_ALIGN * CHER2K_K_ALIGN : 0U;
    return performanceScalar && trans == ACLBLAS_OP_C && n >= CHER2K_CUBE_M && alignedK > 0U &&
           alignedK <= static_cast<uint64_t>(n) && alignedK < CHER2K_THREE_M_K_LIMIT;
}

bool IsCher2kDirectOutputCandidate(bool directHermitianCandidate, bool deviceScalarFast, aclblasFillMode_t uplo, int k)
{
    // The fused consumer uses one K panel. Device scalars additionally
    // require its upper-triangle scalar handling; other inputs use postprocess.
    return directHermitianCandidate && (!deviceScalarFast || uplo == ACLBLAS_UPPER) &&
           k <= static_cast<int>(CHER2K_DIRECT_K_ALIGN);
}

uint32_t SelectCher2kNAlign(bool directOutputCandidate, bool directHermitianCandidate, bool smallCubeEligible)
{
    return directOutputCandidate ?
               CHER2K_DIRECT_OUTPUT_N_ALIGN :
               (directHermitianCandidate ? CHER2K_DIRECT_N_ALIGN :
                                           (smallCubeEligible ? CHER2K_SMALL_CUBE_N_ALIGN : CHER2K_N_ALIGN));
}

bool ValidateCher2kGeometry(uint64_t nAligned, uint64_t kAligned)
{
    if (nAligned > UINT32_MAX || kAligned > UINT32_MAX) {
        return false;
    }
    const uint64_t nk = nAligned * kAligned;
    const uint64_t nn = nAligned * nAligned;
    const uint64_t maxFloats = static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(float);
    return nk <= maxFloats / 6ULL && nn <= maxFloats / 4ULL && 6ULL * nk <= maxFloats - 4ULL * nn;
}

uint64_t Cher2kScheduledAicTasks(uint64_t nAligned, uint32_t smallPath, uint32_t useThreeM, uint32_t sg3MmadPath)
{
    const uint32_t schedulerM = smallPath == CHER2K_SMALL_CUBE_PATH ? 64U : CHER2K_CUBE_M;
    const uint32_t schedulerN = smallPath == CHER2K_SMALL_CUBE_PATH ? 64U : CHER2K_CUBE_N;
    const uint64_t tileCount = (nAligned / schedulerM) * (nAligned / schedulerN);
    const uint64_t productCount = useThreeM != 0U ? 3ULL : 4ULL;
    const uint64_t packedTileAxis = nAligned / CHER2K_DIRECT_OUTPUT_N_ALIGN;
    return sg3MmadPath != 0U ? packedTileAxis * packedTileAxis : tileCount * productCount;
}

void LimitCher2kDirectAicCores(Cher2kTilingData& tiling)
{
    if (tiling.directHermitianPath == 0U) {
        return;
    }
    // directHermitianPath 仅在 trans == OP_C 时由 host 置位（IsCher2kDirectHermitianCandidate
    // 要求 OP_C），进入此分支时 trans 恒为 OP_C，无需再判。
    if (tiling.directOutputPath != 0U) {
        const uint64_t directTileAxis =
            (tiling.nAligned + CHER2K_DIRECT_OUTPUT_N_ALIGN - 1ULL) / CHER2K_DIRECT_OUTPUT_N_ALIGN;
        const uint64_t directTasks = directTileAxis * (directTileAxis + 1ULL) / 2ULL;
        tiling.aicCoreNum = static_cast<uint32_t>(std::min<uint64_t>(tiling.aicCoreNum, directTasks));
    }
}

bool IsCher2kDirectCorrectnessFallback(
    bool deviceScalarFast, bool computeProduct, bool performanceScalar, aclblasOperation_t trans, int n, int k)
{
    if (deviceScalarFast || !computeProduct || performanceScalar) {
        return false;
    }
    return (trans == ACLBLAS_OP_C && n <= 1024 && k <= 256) || (trans == ACLBLAS_OP_N && k <= 32 && n <= 2048);
}

uint32_t SelectCher2kSmallPath(
    bool computeProduct, bool smallOutputAligned, bool performanceScalar, bool smallCubeEligible,
    bool directCorrectnessFallback, int n)
{
    if (!computeProduct || !smallOutputAligned) {
        return 0U;
    }
    if ((performanceScalar && n <= 32) || directCorrectnessFallback) {
        return 1U;
    }
    return smallCubeEligible ? CHER2K_SMALL_CUBE_PATH : 0U;
}

size_t CalculateCher2kWorkspace(const Cher2kTilingData& tiling, size_t sgemmBytes, size_t regularBytes)
{
    if (tiling.computeProduct == 0U || tiling.smallPath == 1U) {
        return 0U;
    }
    return tiling.sg3MmadPath != 0U ? sgemmBytes : regularBytes;
}

Cher2kPathConfig BuildCher2kPathConfig(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex& alpha, float beta, int ldc,
    bool deviceScalarFast)
{
    Cher2kPathConfig config{};
    config.performanceScalar = IsCher2kPerformanceScalar(alpha, beta, deviceScalarFast);
    config.smallOutputAligned = IsCher2kSmallOutputAligned(uplo, trans, n, ldc);
    config.smallCubeEligible =
        IsCher2kSmallCubeEligible(config.smallOutputAligned, config.performanceScalar, trans, n, k);
    config.directHermitianCandidate = IsCher2kDirectHermitianCandidate(config.performanceScalar, trans, n, k);
    config.directOutputCandidate =
        IsCher2kDirectOutputCandidate(config.directHermitianCandidate, deviceScalarFast, uplo, k);
    config.nAlign =
        SelectCher2kNAlign(config.directOutputCandidate, config.directHermitianCandidate, config.smallCubeEligible);
    config.kAlign = config.directHermitianCandidate ? CHER2K_DIRECT_K_ALIGN : CHER2K_K_ALIGN;
    config.sg3MmadEligible = false;
    return config;
}

aclblasStatus_t GetCher2kInterleaveOffsets(_aclblas_handle* h, GM_ADDR& offsets)
{
    if (h->cher2k_interleave_offsets != nullptr) {
        offsets = reinterpret_cast<GM_ADDR>(h->cher2k_interleave_offsets);
        return ACLBLAS_STATUS_SUCCESS;
    }

    std::vector<uint32_t> hostOffsets(CHER2K_INTERLEAVE_OFFSET_COUNT);
    BuildCher2kInterleaveTable(hostOffsets, 0U, CHER2K_DIRECT_INTERLEAVE_ROWS, CHER2K_DIRECT_INTERLEAVE_COLS);
    BuildCher2kInterleaveTable(
        hostOffsets, CHER2K_SMALL_INTERLEAVE_OFFSET, CHER2K_SMALL_INTERLEAVE_ROWS, CHER2K_SMALL_INTERLEAVE_COLS);
    BuildCher2kTransposeTable(hostOffsets);
    BuildCher2kInterleaveTable(
        hostOffsets, CHER2K_TINY_INTERLEAVE_OFFSET, CHER2K_TINY_INTERLEAVE_ROWS, CHER2K_TINY_INTERLEAVE_COLS);
    BuildCher2kPostTransposeTable(hostOffsets);

    void* deviceOffsets = nullptr;
    const size_t bytes = hostOffsets.size() * sizeof(uint32_t);
    aclError ret = aclrtMalloc(&deviceOffsets, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(
            "aclblasCher2k", "GetCher2kInterleaveOffsets: aclrtMalloc failed, bytes=%zu ret=%d", bytes,
            static_cast<int>(ret));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(deviceOffsets, bytes, hostOffsets.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(deviceOffsets);
        OP_LOGE(
            "aclblasCher2k", "GetCher2kInterleaveOffsets: aclrtMemcpy failed, bytes=%zu ret=%d", bytes,
            static_cast<int>(ret));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    h->cher2k_interleave_offsets = deviceOffsets;
    offsets = reinterpret_cast<GM_ADDR>(deviceOffsets);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateCher2kEnums(aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCher2k", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (trans == ACLBLAS_OP_T) {
        OP_LOGE("aclblasCher2k", "OP_T is not supported");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_C) {
        OP_LOGE("aclblasCher2k", "invalid trans value %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateCher2kParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const float* beta, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    aclblasStatus_t status = ValidateCher2kEnums(uplo, trans);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n < 0 || k < 0) {
        OP_LOGE("aclblasCher2k", "n and k must be non-negative, got n=%d k=%d", n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclblasCher2k", "alpha and beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int minLd = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < minLd || ldb < minLd || ldc < std::max(1, n)) {
        OP_LOGE("aclblasCher2k", "invalid leading dimensions lda=%d ldb=%d ldc=%d", lda, ldb, ldc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // C is an output buffer. Its pointer is checked after reading alpha/beta,
    // because the beta==0/no-product quick return does not access C at all.
    if (n > 0 && k > 0 && (a == nullptr || b == nullptr)) {
        OP_LOGE("aclblasCher2k", "A and B must not be nullptr when n and k are positive");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

void FillCher2kTilingBase(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex& alpha, int lda, int ldb,
    float beta, int ldc, uint64_t nAligned, uint64_t kAligned, bool deviceScalars, Cher2kTilingData& tiling)
{
    tiling = {};
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.nAligned = static_cast<uint32_t>(nAligned);
    tiling.kAligned = static_cast<uint32_t>(kAligned);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.aivCoreNum = std::max(1U, GetAivCoreCount());
    if (deviceScalars) {
        // alpha/beta live in device memory, so the host has no value to record.
        // Every kernel re-reads them from the GM pointers (Cher2kResolveScalars)
        // and classifies them on device; keep the tiling copy neutral instead of
        // fabricating a unit scalar that would silently drive path selection.
        tiling.deviceScalars = 1U;
        tiling.computeProduct = k > 0 ? 1U : 0U;
        return;
    }
    tiling.alphaReal = alpha.real;
    tiling.alphaImag = alpha.imag;
    tiling.beta = beta;
    tiling.computeProduct = (k > 0 && (alpha.real != 0.0f || alpha.imag != 0.0f)) ? 1U : 0U;
}

void FillCher2kTilingPaths(
    const Cher2kPathConfig& path, aclblasOperation_t trans, int n, int k, bool deviceScalarFast, uint64_t nAligned,
    uint64_t kAligned, Cher2kTilingData& tiling)
{
    // Three-product complex multiplication saves one MMAD, but its reconstructed
    // imaginary term accumulates substantially more cancellation at very large K.
    // Keep the optimized acceptance paths below K=2048 and use 4M from K=2048.
    tiling.useThreeM = tiling.computeProduct != 0U && path.performanceScalar &&
                               (n >= CHER2K_CUBE_M || path.smallCubeEligible) && kAligned <= nAligned &&
                               kAligned < CHER2K_THREE_M_K_LIMIT ?
                           1U :
                           0U;
    const bool directFallback = IsCher2kDirectCorrectnessFallback(
        deviceScalarFast, tiling.computeProduct != 0U, path.performanceScalar, trans, n, k);
    tiling.smallPath = SelectCher2kSmallPath(
        tiling.computeProduct != 0U, path.smallOutputAligned, path.performanceScalar, path.smallCubeEligible,
        directFallback, n);
    tiling.outputTransposePath = 0U;
    tiling.sg3MmadPath = path.sg3MmadEligible ? 1U : 0U;
    tiling.directHermitianPath = path.directHermitianCandidate && !path.sg3MmadEligible ? 1U : 0U;
    tiling.directOutputPath = path.directOutputCandidate && !path.sg3MmadEligible ? 1U : 0U;
    const uint64_t scheduledAicTasks =
        Cher2kScheduledAicTasks(nAligned, tiling.smallPath, tiling.useThreeM, tiling.sg3MmadPath);
    tiling.aicCoreNum =
        static_cast<uint32_t>(std::max<uint64_t>(1ULL, std::min<uint64_t>(GetAicCoreCount(), scheduledAicTasks)));
    LimitCher2kDirectAicCores(tiling);
}

bool TryBuildCher2kTiling(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex& alpha, int lda, int ldb,
    float beta, int ldc, Cher2kTilingData& tiling, size_t& workspaceBytes, bool deviceScalarFast = false)
{
    const Cher2kPathConfig path = BuildCher2kPathConfig(uplo, trans, n, k, alpha, beta, ldc, deviceScalarFast);
    const uint32_t nAlign = path.nAlign;
    const uint32_t kAlign = path.kAlign;
    const uint64_t nAligned = CeilAlign<uint64_t>(static_cast<uint64_t>(n), nAlign);
    const uint64_t kAligned = CeilAlign<uint64_t>(static_cast<uint64_t>(k), kAlign);
    if (!ValidateCher2kGeometry(nAligned, kAligned)) {
        return false;
    }

    // Path selection keeps each producer paired with its matching workspace
    // layout and uses direct 4M fallbacks where 3M cancellation is unsafe.
    FillCher2kTilingBase(uplo, trans, n, k, alpha, lda, ldb, beta, ldc, nAligned, kAligned, deviceScalarFast, tiling);
    FillCher2kTilingPaths(path, trans, n, k, deviceScalarFast, nAligned, kAligned, tiling);
    workspaceBytes = CalculateCher2kWorkspace(
        tiling, CalcCher2kSgemm3BatchWorkspaceBytes(tiling.nAligned, tiling.kAligned),
        CalcCher2kWorkspaceBytes(tiling.nAligned, tiling.kAligned, tiling.useThreeM != 0U));
    return true;
}

aclblasStatus_t ReadCher2kScalar(const void* pointer, void* hostValue, size_t bytes)
{
    bool isDevice = false;
    aclblasStatus_t status = CheckPtrLocation(pointer, &isDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCher2k", "ReadCher2kScalar: failed to resolve pointer location, status=%d",
            static_cast<int>(status));
        return status;
    }
    if (!isDevice) {
        if (memcpy_s(hostValue, bytes, pointer, bytes) != EOK) {
            OP_LOGE("aclblasCher2k", "ReadCher2kScalar: host copy failed, bytes=%zu", bytes);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclError ret = aclrtMemcpy(hostValue, bytes, pointer, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCher2k", "ReadCher2kScalar: device copy failed, ret=%d", static_cast<int>(ret));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ReadCher2kScalars(
    const aclblasComplex* alpha, const float* beta, aclblasComplex& alphaValue, float& betaValue)
{
    aclblasStatus_t status = ReadCher2kScalar(alpha, &alphaValue, sizeof(alphaValue));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k", "ReadCher2kScalars: failed to read alpha, status=%d", static_cast<int>(status));
        return status;
    }
    status = ReadCher2kScalar(beta, &betaValue, sizeof(betaValue));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k", "ReadCher2kScalars: failed to read beta, status=%d", static_cast<int>(status));
    }
    return status;
}

aclblasStatus_t ResolveCher2kScalarLocations(
    _aclblas_handle* h, const aclblasComplex* alpha, const float* beta, bool& alphaDevice, bool& betaDevice)
{
    if (h->cher2k_scalar_location_valid && h->cher2k_scalar_alpha_ptr == alpha && h->cher2k_scalar_beta_ptr == beta) {
        alphaDevice = h->cher2k_scalar_alpha_device;
        betaDevice = h->cher2k_scalar_beta_device;
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t status = CheckPtrLocation(alpha, &alphaDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCher2k", "ResolveCher2kScalarLocations: failed to locate alpha, status=%d",
            static_cast<int>(status));
        return status;
    }
    status = CheckPtrLocation(beta, &betaDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCher2k", "ResolveCher2kScalarLocations: failed to locate beta, status=%d",
            static_cast<int>(status));
        return status;
    }
    h->cher2k_scalar_location_valid = true;
    h->cher2k_scalar_alpha_ptr = alpha;
    h->cher2k_scalar_beta_ptr = beta;
    h->cher2k_scalar_alpha_device = alphaDevice;
    h->cher2k_scalar_beta_device = betaDevice;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateCher2kNullOutput(const aclblasComplex* alpha, int k, const float* beta, bool& handled)
{
    handled = false;
    aclblasComplex alphaValue{1.0f, 0.0f};
    float betaValue = 0.0f;
    aclblasStatus_t status = ReadCher2kScalars(alpha, beta, alphaValue, betaValue);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const bool noProduct = k == 0 || (alphaValue.real == 0.0f && alphaValue.imag == 0.0f);
    if (betaValue == 0.0f && noProduct) {
        handled = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    OP_LOGE("aclblasCher2k", "ValidateCher2kNullOutput: C must not be nullptr when output is required");
    return ACLBLAS_STATUS_INVALID_VALUE;
}

bool TryLaunchCher2kSmallIngress(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const float* beta, aclblasComplex* c, int ldc,
    bool alphaDevice, bool betaDevice)
{
    const bool smallIngress = n <= 32 || (n <= 128 && k <= 4);
    const bool smallOutputAligned = IsCher2kSmallOutputAligned(uplo, trans, n, ldc);
    if (!smallIngress || !smallOutputAligned || alphaDevice != betaDevice) {
        return false;
    }
    Cher2kTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.aivCoreNum = std::max(1U, GetAivCoreCount());
    tiling.smallPath = 1U;
    if (!alphaDevice) {
        tiling.alphaReal = alpha->real;
        tiling.alphaImag = alpha->imag;
        tiling.beta = *beta;
    } else {
        tiling.deviceScalars = 1U;
    }
    GM_ADDR interleaveOffsets = nullptr;
    if (GetCher2kInterleaveOffsets(h, interleaveOffsets) != ACLBLAS_STATUS_SUCCESS) {
        // The shared path below retries the allocation and reports the failure.
        return false;
    }
    cher2k_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(a)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(b)), reinterpret_cast<GM_ADDR>(c), nullptr, tiling,
        h->stream, false, alphaDevice ? reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(alpha)) : nullptr,
        betaDevice ? reinterpret_cast<GM_ADDR>(const_cast<float*>(beta)) : nullptr, interleaveOffsets);
    return true;
}

aclblasStatus_t PrepareCher2kLaunchBuffers(
    _aclblas_handle* h, const Cher2kTilingData& tiling, size_t workspaceBytes, GM_ADDR& interleaveOffsets)
{
    if (workspaceBytes > 0U) {
        const aclblasStatus_t status = EnsureDefaultWorkspace(h, workspaceBytes);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            OP_LOGE("aclblasCher2k", "workspace allocation failed, required=%zu", workspaceBytes);
            return status;
        }
    }
    const bool needsInterleaveOffsets = (tiling.smallPath == 1U && tiling.trans == ACLBLAS_OP_N &&
                                         tiling.n <= CHER2K_SMALL_INTERLEAVE_ROWS && tiling.k <= 4U) ||
                                        (tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C);
    if (!needsInterleaveOffsets) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t status = GetCher2kInterleaveOffsets(h, interleaveOffsets);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k", "interleave metadata unavailable, status=%d", static_cast<int>(status));
    }
    return status;
}

aclblasStatus_t BuildAndLaunchCher2k(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const float* beta, aclblasComplex* c, int ldc,
    bool deviceScalarFast)
{
    // Device scalars are never read on the host: the kernels classify them from
    // GM, so no host value (and no fabricated unit scalar) takes part in the
    // tiling or path selection.
    aclblasComplex alphaValue{0.0f, 0.0f};
    float betaValue = 0.0f;
    if (!deviceScalarFast) {
        const aclblasStatus_t readStatus = ReadCher2kScalars(alpha, beta, alphaValue, betaValue);
        if (readStatus != ACLBLAS_STATUS_SUCCESS) {
            return readStatus;
        }
    }
    Cher2kTilingData tiling{};
    size_t workspaceBytes = 0U;
    if (!TryBuildCher2kTiling(
            uplo, trans, n, k, alphaValue, lda, ldb, betaValue, ldc, tiling, workspaceBytes, deviceScalarFast)) {
        OP_LOGE("aclblasCher2k", "workspace size calculation overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    GM_ADDR interleaveOffsets = nullptr;
    const aclblasStatus_t status = PrepareCher2kLaunchBuffers(h, tiling, workspaceBytes, interleaveOffsets);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    cher2k_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(a)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(b)), reinterpret_cast<GM_ADDR>(c),
        reinterpret_cast<GM_ADDR>(GetEffectiveWorkspace(h)), tiling, h->stream, false,
        deviceScalarFast ? reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(alpha)) : nullptr,
        deviceScalarFast ? reinterpret_cast<GM_ADDR>(const_cast<float*>(beta)) : nullptr, interleaveOffsets);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const float* beta, aclblasComplex* c, int ldc)
{
    aclblasStatus_t status = ValidateCher2kParams(handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    bool alphaDevice = false;
    bool betaDevice = false;
    status = ResolveCher2kScalarLocations(h, alpha, beta, alphaDevice, betaDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCher2k", "aclblasCher2k: failed to resolve alpha/beta location, status=%d",
            static_cast<int>(status));
        return status;
    }

    // A null C pointer is valid only when the operation has no observable
    // output.  Check this before the value-independent small ingress, which
    // otherwise would pass the null destination to a device kernel.
    if (c == nullptr) {
        bool handled = false;
        status = ValidateCher2kNullOutput(alpha, k, beta, handled);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        if (handled) {
            return ACLBLAS_STATUS_SUCCESS;
        }
        OP_LOGE("aclblasCher2k", "C must not be nullptr when output is required");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Small exact work is cheaper than synchronously copying two device
    // scalars to the host.  Keep this path value-independent: the fused AIV
    // kernel reads alpha/beta from GM on every invocation, so in-place scalar
    // updates retain normal BLAS semantics.
    if (TryLaunchCher2kSmallIngress(
            h, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc, alphaDevice, betaDevice)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Device scalars stay in GM; host scalars are read once to select the graph.
    return BuildAndLaunchCher2k(h, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc, alphaDevice && betaDevice);
}
