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
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "ctrmm_tiling_data.h"
#include "ctrmm_constants.h"

namespace {
using namespace ctrmm;
bool MatrixBytes(int rows, int cols, int ld, uint64_t& bytes)
{
    const uint64_t count = static_cast<uint64_t>(cols - 1) * ld + rows;
    if (count > std::numeric_limits<size_t>::max() / sizeof(aclblasComplex)) {
        return false;
    }
    bytes = count * sizeof(aclblasComplex);
    return true;
}
bool Overlaps(const void* a, uint64_t aBytes, const void* b, uint64_t bBytes)
{
    const uintptr_t x = reinterpret_cast<uintptr_t>(a);
    const uintptr_t y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < aBytes : x - y < bBytes;
}
bool ValidModes(aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    return (side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT) &&
           (uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER) &&
           (trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C) &&
           (diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT);
}

CtrmmTiling MakeTiling(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* alpha, int lda, int ldb, int ldc)
{
    CtrmmTiling t{};
    t.m = m;
    t.n = n;
    t.lda = lda;
    t.ldb = ldb;
    t.ldc = ldc;
    t.q = side == ACLBLAS_SIDE_LEFT ? m : n;
    t.r = side == ACLBLAS_SIDE_LEFT ? n : m;
    t.right = side == ACLBLAS_SIDE_RIGHT;
    t.transpose = (trans != ACLBLAS_OP_N) != static_cast<bool>(t.right);
    t.conjugate = trans == ACLBLAS_OP_C;
    t.upper = (uplo == ACLBLAS_UPPER) != static_cast<bool>(t.transpose);
    t.unit = diag == ACLBLAS_UNIT;
    t.originalTrans = static_cast<uint32_t>(trans);
    t.alphaReal = alpha->real;
    t.alphaImag = alpha->imag;
    t.vectorCores = GetAivCoreCount();
    return t;
}

bool ValidInputStorage(
    const CtrmmTiling& t, const aclblasComplex* a, const aclblasComplex* b, const aclblasComplex* c, uint64_t cBytes)
{
    if (a == nullptr || b == nullptr) {
        return false;
    }
    uint64_t aBytes;
    uint64_t bBytes;
    if (!MatrixBytes(t.q, t.q, t.lda, aBytes) || !MatrixBytes(t.m, t.n, t.ldb, bBytes)) {
        return false;
    }
    return !Overlaps(a, aBytes, c, cBytes) && (b == c || !Overlaps(b, bBytes, c, cBytes));
}

void ConfigureWorkspace(CtrmmTiling& t)
{
    t.qp = (t.q + kVectorRoundUp) / kVectorFloats * kVectorFloats;
    // Bound the domain used by the fast kernels. This is not an input
    // restriction or a universal precision certificate: Pack also checks
    // coefficient conditioning, and all other inputs use Reference.
    const double alphaNorm = std::abs(static_cast<double>(t.alphaReal)) + std::abs(static_cast<double>(t.alphaImag));
    t.fastInputLimit = std::isfinite(alphaNorm) && alphaNorm <= kFastAlphaNormLimit ?
                           (t.q > kCubeTile && t.q <= kCompactMaxOrder ? kCompactInputLimit : kFastInputLimit) :
                           0.0f;
    const uint32_t tileN = kCubeTile;
    t.rp = (t.r + tileN - 1U) / tileN * tileN;
    t.aPlane = static_cast<uint64_t>(t.qp) * t.qp;
    t.xPlane = static_cast<uint64_t>(t.qp) * t.rp;
    // Cube couples the real/imaginary products before long accumulation.
    t.useCube = t.q >= kCubeMinOrder && t.r >= kCubeMinRhs;
    t.cubeCores = GetAicCoreCount();
    if (t.cubeCores == 0) {
        t.useCube = 0;
    }
}

aclblasStatus_t PrepareWorkspace(_aclblas_handle* h, CtrmmTiling& t)
{
    ConfigureWorkspace(t);
    constexpr uint64_t maxFloats = ACLBLAS_MAX_WORKSPACE_SIZE / sizeof(float);
    const uint64_t xCopies = t.useCube ? kCubeXPlanes : kVectorXPlanes;
    if (t.aPlane > maxFloats / kOriginalAPlanes || t.xPlane > maxFloats / xCopies) {
        OP_LOGE(
            "aclblasCtrmm", "Workspace planes exceed limit: aPlane=%llu xPlane=%llu xCopies=%llu maxFloats=%llu",
            static_cast<unsigned long long>(t.aPlane), static_cast<unsigned long long>(t.xPlane),
            static_cast<unsigned long long>(xCopies), static_cast<unsigned long long>(maxFloats));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    t.flagsOffset = t.useCube ? kCubeAPlanes * t.aPlane + kCubeXPlanes * t.xPlane + kDiagonalRowFloats * t.qp :
                                kOriginalAPlanes * t.aPlane + xCopies * t.xPlane;
    const uint64_t totalFloats = t.flagsOffset + static_cast<uint64_t>(t.vectorCores) * kFlagSlotFloats;
    if (totalFloats > maxFloats) {
        OP_LOGE(
            "aclblasCtrmm", "Workspace exceeds limit: required=%llu bytes, limit=%zu bytes",
            static_cast<unsigned long long>(totalFloats * sizeof(float)), ACLBLAS_MAX_WORKSPACE_SIZE);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    return EnsureDefaultWorkspace(h, static_cast<size_t>(totalFloats * sizeof(float)));
}
} // namespace

aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc)
{
    CHECK_RET(
        handle != nullptr, OP_LOGE("aclblasCtrmm", "handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    if (!ValidModes(side, uplo, trans, diag) || m < 0 || n < 0 || alpha == nullptr) {
        OP_LOGE(
            "aclblasCtrmm", "Invalid arguments: side=%d uplo=%d trans=%d diag=%d m=%d n=%d alphaNull=%d", side, uplo,
            trans, diag, m, n, alpha == nullptr);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Task CSV defines zero-dimensional calls as no-ops even with ld == 0.
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const int q = side == ACLBLAS_SIDE_LEFT ? m : n;
    uint64_t cBytes;
    if (lda < std::max(1, q) || ldb < std::max(1, m) || ldc < std::max(1, m) || C == nullptr ||
        !MatrixBytes(m, n, ldc, cBytes)) {
        OP_LOGE(
            "aclblasCtrmm", "Invalid C/strides: m=%d n=%d q=%d lda=%d ldb=%d ldc=%d cNull=%d", m, n, q, lda, ldb, ldc,
            C == nullptr);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto t = MakeTiling(side, uplo, trans, diag, m, n, alpha, lda, ldb, ldc);
    CHECK_RET(t.vectorCores != 0, OP_LOGE("aclblasCtrmm", "No available AIV cores on the current device");
              return ACLBLAS_STATUS_INTERNAL_ERROR);
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    if (alpha->real == 0.0f && alpha->imag == 0.0f) {
        ctrmm_kernel_do(nullptr, nullptr, reinterpret_cast<GM_ADDR>(C), nullptr, t, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!ValidInputStorage(t, A, B, C, cBytes)) {
        OP_LOGE("aclblasCtrmm", "Invalid A/B storage: null pointer, size overflow or unsupported overlap with C");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const auto status = PrepareWorkspace(h, t);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCtrmm", "Workspace preparation failed, status=%d", status);
        return status;
    }
    ctrmm_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)), reinterpret_cast<GM_ADDR>(C),
        reinterpret_cast<GM_ADDR>(GetEffectiveWorkspace(h)), t, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
