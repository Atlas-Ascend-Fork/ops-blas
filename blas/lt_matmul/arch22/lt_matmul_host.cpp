/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lt_matmul_host.h"

#include "aclblaslt_algo_heuristic.h"
#include "aclblaslt_handle_impl.h"
#include "aclblaslt_layout_impl.h"
#include "host_utils.h"
#include "lt_matmul_kernel.h"
#include "matmul_engine.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using Layout = aclblasLtMatrixLayoutImpl;
using Desc = aclblasLtMatmulDescImpl;

bool CheckedProduct(size_t a, size_t b, size_t& result)
{
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    result = a * b;
    return true;
}

constexpr uint32_t H0_CUBE_TILE = LT_MATMUL22_TILE_M;
constexpr uint32_t H1_CUBE_TILE = LT_MATMUL22_H1_TILE_M;

struct H1Storage {
    uint32_t productPitch = 0;
    size_t bytes = 0;
};

// H1 reserves a padded product for nontrivial alpha/beta. Native ND output
// otherwise writes D directly. Heuristic and execution share this fixed bound.
aclblasStatus_t BuildH1Storage(uint64_t m, uint64_t n, H1Storage& storage)
{
    storage = H1Storage{};
    if (n < 16U) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (n > std::numeric_limits<uint64_t>::max() - 15U) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint64_t pitch = ((n + 15U) / 16U) * 16U;
    if (pitch > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (m > std::numeric_limits<size_t>::max()) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    size_t elements = 0;
    if (!CheckedProduct(static_cast<size_t>(m), static_cast<size_t>(pitch), elements) ||
        !CheckedProduct(elements, sizeof(float), storage.bytes)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (storage.bytes > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    storage.productPitch = static_cast<uint32_t>(pitch);
    return ACLBLAS_STATUS_SUCCESS;
}

// H3 has four independent partial products, followed by a fixed-order FP32 sum.
// The 16-byte reserve admits an internal 32-byte-aligned plane from a legal
// public 16-byte-aligned pointer. H0/H1/H2 capacities and capsules are unchanged.
aclblasStatus_t BuildH3Storage(const LtMatmul22Tiling& t, H1Storage& storage)
{
    if (t.m < 16U || t.m > 64U || t.n < 32U || t.n > 128U || t.k < 512U || t.k > 2048U) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const aclblasStatus_t status = BuildH1Storage(t.m, t.n, storage);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    storage.bytes = storage.bytes * 4U + 16U;
    return ACLBLAS_STATUS_SUCCESS;
}

struct H2Storage {
    H1Storage product;
    size_t aBytes = 0;
    size_t bBytes = 0;
    size_t bytes = 0;
};

aclblasStatus_t BuildH2Storage(const LtMatmul22Tiling& t, H2Storage& storage)
{
    storage = H2Storage{};
    const aclblasStatus_t status = BuildH1Storage(t.m, t.n, storage.product);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    // NZ physical dimensions must remain representable by the Matmul SDK.
    if (t.m == 0U || t.m > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) - 15U || t.k == 0U ||
        t.k > 4096U) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const auto packedBytes = [](uint32_t rows, uint32_t cols, size_t& bytes) {
        const size_t height = (static_cast<size_t>(rows) + 15U) / 16U * 16U;
        const size_t width = (static_cast<size_t>(cols) + 7U) / 8U * 8U;
        size_t elements = 0;
        return CheckedProduct(height, width, elements) && CheckedProduct(elements, sizeof(float), bytes);
    };
    if (!packedBytes(t.transA ? t.k : t.m, t.transA ? t.m : t.k, storage.aBytes) ||
        !packedBytes(t.transB ? t.n : t.k, t.transB ? t.k : t.n, storage.bBytes)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Reserve 16 bytes so a public 16B-aligned workspace can contain 32B-aligned
    // NZ planes. Their sizes are multiples of 512 bytes, so only one gap is needed.
    size_t total = storage.product.bytes;
    const size_t limit = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    for (size_t bytes : {storage.aBytes, storage.bBytes, size_t(16U)}) {
        if (bytes > limit - total) {
            return ACLBLAS_STATUS_NOT_SUPPORTED;
        }
        total += bytes;
    }
    storage.bytes = total;
    return ACLBLAS_STATUS_SUCCESS;
}

struct H4Storage {
    H1Storage product;
    size_t aBytes = 0;
    size_t bytes = 0;
};

aclblasStatus_t BuildH4Storage(const LtMatmul22Tiling& t, H4Storage& storage)
{
    storage = H4Storage{};
    if (!t.transA || t.m < 16U || t.m > 64U || t.n < 512U || t.k < 128U || t.k > 512U) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const aclblasStatus_t status = BuildH1Storage(t.m, t.n, storage.product);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    storage.aBytes = size_t((t.k + 15U) / 16U * 16U) * ((t.m + 7U) / 8U * 8U) * 4U;
    const size_t limit = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    if (storage.product.bytes > limit - storage.aBytes - 16U)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    storage.bytes = storage.product.bytes + storage.aBytes + 16U;
    return ACLBLAS_STATUS_SUCCESS;
}

// H10 owns packed FP32 NZ A/B and a padded NZ product. Public workspace
// alignment remains 16B; 496B admits the next 512B boundary in every case.
aclblasStatus_t BuildFusedShortStorage(const LtMatmul22Tiling& t, H2Storage& storage)
{
    storage = H2Storage{};
    if (!t.transA || !t.transB || t.m < 16U || t.m > 2048U || t.n < 16U || t.n > 1440U || t.k == 0U || t.k > 32U ||
        t.lda > 65535U || t.ldb > 65535U || t.ldc > 65535U || t.ldd > 65535U) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const size_t mp = (size_t(t.m) + 15U) / 16U * 16U;
    const size_t np = (size_t(t.n) + 15U) / 16U * 16U;
    const size_t kp = (size_t(t.k) + 15U) / 16U * 16U;
    storage.aBytes = mp * kp * sizeof(float);
    storage.bBytes = np * kp * sizeof(float);
    storage.product.productPitch = static_cast<uint32_t>(np);
    storage.product.bytes = mp * np * sizeof(float);
    storage.bytes = storage.aBytes + storage.bBytes + storage.product.bytes + 496U;
    return ACLBLAS_STATUS_SUCCESS;
}

// New short-reduction plans own distinct capsules; H1's 128x128 maximum stays fixed.
bool IsTTRectPlan(uint32_t id, const LtMatmul22Tiling& t)
{
    return id == LT_MATMUL22_ALGO_H11 && t.transA && t.transB && t.m > 1024U && t.m <= 2048U && t.n >= 512U &&
           t.n <= 1024U && t.k > 64U && t.k <= 96U;
}

bool IsNNRectPlan(uint32_t id, const LtMatmul22Tiling& t)
{
    return id == LT_MATMUL22_ALGO_H12 && !t.transA && !t.transB && t.m > 512U && t.m <= 1024U && t.n > 1024U &&
           t.n <= 1536U && t.k > 96U && t.k <= 160U;
}

aclblasStatus_t BuildRectStorage(uint32_t id, const LtMatmul22Tiling& t, H1Storage& storage)
{
    storage = H1Storage{};
    const bool tt = IsTTRectPlan(id, t);
    const bool nn = IsNNRectPlan(id, t);
    // The installed ND2NZ load encodes physical source pitches in 16 bits.
    if ((!tt && !nn) || t.lda > 65535U || t.ldb > 65535U)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    return BuildH1Storage(t.m, t.n, storage);
}

bool IsTranspose(aclblasOperation_t op) { return op == ACLBLAS_OP_N || op == ACLBLAS_OP_T || op == ACLBLAS_OP_C; }

bool IsKnownEpilogue(aclblasLtEpilogue_t epilogue)
{
    constexpr aclblasLtEpilogue_t known[] = {
        ACLBLASLT_EPILOGUE_DEFAULT,
        ACLBLASLT_EPILOGUE_RELU,
        ACLBLASLT_EPILOGUE_BIAS,
        ACLBLASLT_EPILOGUE_RELU_BIAS,
        ACLBLASLT_EPILOGUE_GELU,
        ACLBLASLT_EPILOGUE_GELU_BIAS,
        ACLBLASLT_EPILOGUE_RELU_AUX,
        ACLBLASLT_EPILOGUE_RELU_AUX_BIAS,
        ACLBLASLT_EPILOGUE_DRELU,
        ACLBLASLT_EPILOGUE_DRELU_BGRAD,
        ACLBLASLT_EPILOGUE_GELU_AUX,
        ACLBLASLT_EPILOGUE_GELU_AUX_BIAS,
        ACLBLASLT_EPILOGUE_DGELU,
        ACLBLASLT_EPILOGUE_DGELU_BGRAD,
        ACLBLASLT_EPILOGUE_BGRADA,
        ACLBLASLT_EPILOGUE_BGRADB,
        ACLBLASLT_EPILOGUE_SIGMOID,
        ACLBLASLT_EPILOGUE_SWISH_EXT,
        ACLBLASLT_EPILOGUE_SWISH_BIAS_EXT,
        ACLBLASLT_EPILOGUE_CLAMP_EXT,
        ACLBLASLT_EPILOGUE_CLAMP_BIAS_EXT,
        ACLBLASLT_EPILOGUE_CLAMP_AUX_EXT,
        ACLBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT};
    for (aclblasLtEpilogue_t value : known) {
        if (epilogue == value) {
            return true;
        }
    }
    return false;
}

bool MatrixBytes(uint64_t rows, uint64_t cols, uint64_t pitch, size_t& bytes)
{
    bytes = 0;
    if (rows == 0 || cols == 0) {
        return true;
    }
    size_t startLastRow = 0;
    if (!CheckedProduct(static_cast<size_t>(rows - 1U), static_cast<size_t>(pitch), startLastRow)) {
        return false;
    }
    constexpr size_t maxElements = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(float);
    if (startLastRow > maxElements || cols > maxElements - startLastRow) {
        return false;
    }
    return CheckedProduct(startLastRow + static_cast<size_t>(cols), sizeof(float), bytes);
}

bool NormalizeLayout(const Layout& layout, uint64_t rows, uint64_t cols, uint32_t& pitch)
{
    constexpr uint64_t maxDim = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    if (layout.rows != rows || layout.cols != cols || rows > maxDim || cols > maxDim || layout.ld < 0) {
        return false;
    }
    const uint64_t rowWidth = std::max<uint64_t>(1U, cols);
    const uint64_t ld = layout.ld == 0 ? rowWidth : static_cast<uint64_t>(layout.ld);
    if (ld < rowWidth || ld > maxDim) {
        return false;
    }
    size_t bytes = 0;
    if (!MatrixBytes(rows, cols, ld, bytes)) {
        return false;
    }
    pitch = static_cast<uint32_t>(ld);
    return true;
}

bool PreferNarrowNNNZ(const LtMatmul22Tiling& t)
{
    return !t.transA && !t.transB && t.m >= 512U && t.n >= 16U && t.n <= 64U && t.k > 512U && t.k <= 1024U;
}

bool PreferSmallNNNZ(const LtMatmul22Tiling& t)
{
    return !t.transA && !t.transB && t.m >= 16U && t.m <= 32U && t.n >= 512U && t.n <= 1024U && t.k > 1024U &&
           t.k <= 2048U;
}

bool H2NTLayout(const LtMatmul22Tiling& t)
{
    return t.transB && t.m >= 64U && t.m <= 128U && t.n >= 128U && t.n <= 256U && t.k >= 512U && t.k <= 2048U;
}

bool H2TransposeFamily(const LtMatmul22Tiling& t)
{
    const bool nt = H2NTLayout(t);
    const bool narrowM = t.m >= 64U && t.m <= 128U && t.n >= 128U && t.k >= 512U;
    const bool narrowN = t.m >= 512U && t.n >= 16U && t.n <= 64U && t.k >= 128U && t.k <= 512U;
    const bool tallM = t.m >= 512U && t.n >= 128U && t.n <= 256U && t.k >= 512U;
    return (t.transA || nt) && (narrowM || narrowN || tallM);
}

bool PreferH2(const LtMatmul22Tiling& t, const H2Storage& storage)
{
    // These two families have measured ND-conversion bottlenecks. H0 and H1
    // retain their original workspace contracts and remain available fallbacks.
    const bool family = PreferSmallNNNZ(t) || PreferNarrowNNNZ(t) || H2TransposeFamily(t);
    if (!family) {
        return false;
    }
    size_t a = 0, b = 0, d = 0;
    if (!MatrixBytes(t.transA ? t.k : t.m, t.transA ? t.m : t.k, t.lda, a) ||
        !MatrixBytes(t.transB ? t.n : t.k, t.transB ? t.k : t.n, t.ldb, b) || !MatrixBytes(t.m, t.n, t.ldd, d)) {
        return false;
    }
    // A+B+D is a conservative descriptor bound for the independent PF inputs;
    // C may be absent or alias D. Compare exactly without floating point.
    return static_cast<__uint128_t>(storage.bytes) * 5U <= (static_cast<__uint128_t>(a) + b + d) * 6U;
}

bool SupportsH5(const LtMatmul22Tiling& t)
{
    return t.m > 0U && t.m <= 1920U && t.n >= 16U && t.n <= 160U && t.k <= 128U;
}

bool PreferH5(const LtMatmul22Tiling& t)
{
    return SupportsH5(t) && !t.transA && !t.transB && t.m >= 512U && t.n > 128U && t.k > 64U;
}

aclblasStatus_t ValidateLayoutKinds(const Layout& a, const Layout& b, const Layout& c, const Layout& d)
{
    const Layout* layouts[] = {&a, &b, &c, &d};
    for (const Layout* layout : layouts) {
        if (layout->magic != ACLBLASLT_LAYOUT_MAGIC) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (layout->order < ACLBLASLT_ORDER_COL || layout->order > ACLBLASLT_ORDER_COL32_2R_4R4) {
            return ACLBLAS_STATUS_INVALID_ENUM;
        }
        if (layout->order != ACLBLASLT_ORDER_ROW) {
            return ACLBLAS_STATUS_NOT_SUPPORTED;
        }
        if (layout->batchCount != 1 || layout->stridedBatchOffset != 0) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool SupportedLayoutTypes(const Desc& desc, const Layout& a, const Layout& b, const Layout& c, const Layout& d)
{
    return !(
        desc.computeType != ACLBLAS_COMPUTE_32F || desc.scaleType != ACL_FLOAT ||
        desc.epilogue != ACLBLASLT_EPILOGUE_DEFAULT || a.type != ACL_FLOAT || b.type != ACL_FLOAT ||
        c.type != ACL_FLOAT || d.type != ACL_FLOAT);
}

aclblasStatus_t ValidateLayouts(
    const Desc& desc, const Layout& a, const Layout& b, const Layout& c, const Layout& d, LtMatmul22Tiling& t)
{
    const aclblasStatus_t layoutStatus = ValidateLayoutKinds(a, b, c, d);
    if (layoutStatus != ACLBLAS_STATUS_SUCCESS)
        return layoutStatus;
    if (!IsTranspose(desc.transA) || !IsTranspose(desc.transB) || !IsKnownEpilogue(desc.epilogue)) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (!SupportedLayoutTypes(desc, a, b, c, d))
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    const uint64_t m = d.rows;
    const uint64_t n = d.cols;
    const uint64_t k = desc.transA == ACLBLAS_OP_N ? a.cols : a.rows;
    const bool ta = desc.transA != ACLBLAS_OP_N;
    const bool tb = desc.transB != ACLBLAS_OP_N;
    if (!NormalizeLayout(a, ta ? k : m, ta ? m : k, t.lda) || !NormalizeLayout(b, tb ? n : k, tb ? k : n, t.ldb) ||
        !NormalizeLayout(c, m, n, t.ldc) || !NormalizeLayout(d, m, n, t.ldd)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    t.m = static_cast<uint32_t>(m);
    t.n = static_cast<uint32_t>(n);
    t.k = static_cast<uint32_t>(k);
    t.nCube = t.n / 16U * 16U;
    t.transA = ta;
    t.transB = tb;
    return ACLBLAS_STATUS_SUCCESS;
}

PackedAlgo PlanCapsule(uint32_t id)
{
    PackedAlgo packed{};
    packed.magic = ACLBLASLT_ALGO_MAGIC;
    packed.algoId = id;
    const uint32_t tile = id == LT_MATMUL22_ALGO_H0 ? H0_CUBE_TILE : H1_CUBE_TILE;
    packed.l1mDiv16 = id == LT_MATMUL22_ALGO_H9 ? 80U / 16U : tile / 16U;
    packed.l1nDiv16 = id == LT_MATMUL22_ALGO_H9 ? 288U / 16U : tile / 16U;
    if (id == LT_MATMUL22_ALGO_H4) {
        packed.l1mDiv16 = 4U;
        packed.l1nDiv16 = 8U;
    }
    if (id == LT_MATMUL22_ALGO_H11) {
        packed.l1mDiv16 = 10U;
        packed.l1nDiv16 = 8U;
    }
    if (id == LT_MATMUL22_ALGO_H12) {
        packed.l1mDiv16 = 8U;
        packed.l1nDiv16 = 12U;
    }
    if (id == LT_MATMUL22_ALGO_H10) {
        packed.l1mDiv16 = 2U;
        packed.l1nDiv16 = 30U;
    }
    if (id == LT_MATMUL22_ALGO_H5) {
        packed.l1mDiv16 = 6U;
        packed.l1nDiv16 = 10U;
    }
    packed.policy = DISPATCH_POLICY_MMAD_SYNC;
    packed.numBuffers = id == LT_MATMUL22_ALGO_H10 ? 2U : 1U;
    packed.splitK = id == LT_MATMUL22_ALGO_H3 ? 4U : 1U;
    if (id == LT_MATMUL22_ALGO_H3) {
        packed.l1mDiv16 = 2U;
        packed.l1nDiv16 = 4U;
    }
    return packed;
}

aclblasStatus_t DecodePlan(const aclblasLtMatmulAlgo_t* algo, uint32_t& id)
{
    id = LT_MATMUL22_ALGO_H0;
    if (algo == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    PackedAlgo packed{};
    if (!DecodeAlgo(*algo, &packed)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (packed.algoId != LT_MATMUL22_ALGO_H0 && packed.algoId != LT_MATMUL22_ALGO_H1 &&
        packed.algoId != LT_MATMUL22_ALGO_H2 && packed.algoId != LT_MATMUL22_ALGO_H3 &&
        packed.algoId != LT_MATMUL22_ALGO_H4 && packed.algoId != LT_MATMUL22_ALGO_H11 &&
        packed.algoId != LT_MATMUL22_ALGO_H12 && packed.algoId != LT_MATMUL22_ALGO_H10 &&
        packed.algoId != LT_MATMUL22_ALGO_H9 && packed.algoId != LT_MATMUL22_ALGO_H5) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const PackedAlgo expected = PlanCapsule(packed.algoId);
    if (packed.l1mDiv16 != expected.l1mDiv16 || packed.l1nDiv16 != expected.l1nDiv16 ||
        packed.policy != expected.policy || packed.numBuffers != expected.numBuffers ||
        packed.splitK != expected.splitK || packed.flags != expected.flags) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    id = packed.algoId;
    return ACLBLAS_STATUS_SUCCESS;
}

bool UsesH1Storage(uint32_t id)
{
    return id == LT_MATMUL22_ALGO_H1 || id == LT_MATMUL22_ALGO_H9 || id == LT_MATMUL22_ALGO_H5;
}

aclblasStatus_t RequiredPlanWorkspace(
    uint32_t id, const LtMatmul22Tiling& t, const H1Storage& h1, aclblasStatus_t h1Status, size_t& required)
{
    if (id == LT_MATMUL22_ALGO_H5 && !SupportsH5(t)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (UsesH1Storage(id) && h1Status != ACLBLAS_STATUS_SUCCESS) {
        return h1Status;
    }
    required = UsesH1Storage(id) ? h1.bytes : 0;
    if (id == LT_MATMUL22_ALGO_H2) {
        H2Storage h2;
        const aclblasStatus_t status = BuildH2Storage(t, h2);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        required = h2.bytes;
    }
    if (id == LT_MATMUL22_ALGO_H3) {
        H1Storage h3;
        const aclblasStatus_t status = BuildH3Storage(t, h3);
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
        required = h3.bytes;
    }
    if (id == LT_MATMUL22_ALGO_H4) {
        H4Storage h4;
        const aclblasStatus_t status = BuildH4Storage(t, h4);
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
        required = h4.bytes;
    }
    if (id == LT_MATMUL22_ALGO_H11 || id == LT_MATMUL22_ALGO_H12) {
        H1Storage rect;
        const aclblasStatus_t status = BuildRectStorage(id, t, rect);
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
        required = rect.bytes;
    }
    if (id == LT_MATMUL22_ALGO_H10) {
        H2Storage fused;
        const aclblasStatus_t status = BuildFusedShortStorage(t, fused);
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
        required = fused.bytes;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateWorkspace(
    const aclblasLtMatmulAlgo_t* algo, const LtMatmul22Tiling& t, void* workspace, size_t bytes, uint32_t& id)
{
    if (workspace != nullptr && (reinterpret_cast<uintptr_t>(workspace) & 15U) != 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const aclblasStatus_t decodeStatus = DecodePlan(algo, id);
    if (decodeStatus != ACLBLAS_STATUS_SUCCESS) {
        return decodeStatus;
    }
    H1Storage h1;
    const aclblasStatus_t h1Status = BuildH1Storage(t.m, t.n, h1);
    if (algo == nullptr && workspace != nullptr && h1Status == ACLBLAS_STATUS_SUCCESS && h1.bytes != 0 &&
        bytes >= h1.bytes) {
        id = LT_MATMUL22_ALGO_H1;
    }
    // issue #837: 显式 H10 算法 + 非平凡标量（alpha!=1 或 beta!=0）时实际路由
    // ordered 路径（见 RequiresOrdered 的 fusedShortOrdered 判定），该路径不使用
    // caller workspace——此处按实际路由校验：不构建 fused 存储、不受其容量与
    // 序列化 algo 的 max_workspace_bytes 约束、允许空 workspace，避免本可执行的
    // 调用被误报 INVALID_VALUE。
    if (id == LT_MATMUL22_ALGO_H10 && (t.alpha != 1.0f || t.beta != 0.0f)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    size_t required = 0;
    const aclblasStatus_t requiredStatus = RequiredPlanWorkspace(id, t, h1, h1Status, required);
    if (requiredStatus != ACLBLAS_STATUS_SUCCESS)
        return requiredStatus;
    const size_t advertised = algo == nullptr ? 0U : algo->max_workspace_bytes;
    if (bytes < std::max(required, advertised) || (required != 0 && workspace == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateOperandPointers(
    const LtMatmul22Tiling& t, const void* a, const void* b, const void* c, const void* d)
{
    const bool product = t.alpha != 0.0f && t.k != 0;
    if (d == nullptr || (product && (a == nullptr || b == nullptr)) || (t.beta != 0.0f && c == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return c == d && t.ldc != t.ldd ? ACLBLAS_STATUS_INVALID_VALUE : ACLBLAS_STATUS_SUCCESS;
}

struct AddressRange {
    uintptr_t begin = 0;
    uintptr_t end = 0;
};

bool MakeRange(const void* pointer, size_t bytes, AddressRange& range)
{
    range.begin = reinterpret_cast<uintptr_t>(pointer);
    if (bytes > std::numeric_limits<uintptr_t>::max() - range.begin) {
        return false;
    }
    range.end = range.begin + bytes;
    return true;
}

bool OperandRange(const void* pointer, const Layout& layout, uint32_t pitch, bool active, AddressRange& range)
{
    if (!active) {
        return true;
    }
    size_t bytes = 0;
    return MatrixBytes(layout.rows, layout.cols, pitch, bytes) && MakeRange(pointer, bytes, range);
}

bool Overlaps(const AddressRange& a, const AddressRange& b)
{
    return a.begin < a.end && b.begin < b.end && a.begin < b.end && b.begin < a.end;
}

aclblasStatus_t ValidateOverlaps(
    const void* a, const void* b, const void* c, const void* d, const Layout& aLayout, const Layout& bLayout,
    const Layout& cLayout, const Layout& dLayout, const LtMatmul22Tiling& t, const void* workspace,
    size_t workspaceBytes)
{
    const bool product = t.alpha != 0.0f && t.k != 0;
    AddressRange ar, br, cr, dr, wr;
    if (!OperandRange(a, aLayout, t.lda, product, ar) || !OperandRange(b, bLayout, t.ldb, product, br) ||
        !OperandRange(c, cLayout, t.ldc, t.beta != 0.0f, cr) || !OperandRange(d, dLayout, t.ldd, true, dr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Overlaps(dr, ar) || Overlaps(dr, br) || (c != d && Overlaps(dr, cr))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (workspace != nullptr && workspaceBytes != 0) {
        if (!MakeRange(workspace, workspaceBytes, wr) || Overlaps(wr, ar) || Overlaps(wr, br) || Overlaps(wr, cr) ||
            Overlaps(wr, dr)) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline uint8_t* Bytes(const void* pointer) { return static_cast<uint8_t*>(const_cast<void*>(pointer)); }

struct Hardware {
    uint32_t aic = 0;
    uint32_t aiv = 0;
    uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0;
    uint64_t ubBytes = 0;
};

bool SpecialPlanHardware(uint32_t id, const Hardware& hardware)
{
    // Frozen H10 static-tiling helper: base32x480x16, double-buffered A/B/C.
    if (id == LT_MATMUL22_ALGO_H10) {
        return hardware.aic != 0U && hardware.aiv >= 2U && hardware.ubBytes >= 192512U + 256U &&
               hardware.l1 >= 212224U + 4096U && hardware.l0a >= 4096U && hardware.l0b >= 61440U &&
               hardware.l0c >= 122880U;
    }
    // CONFIG_NORM real bases 160x128/K96 and 128x192/K160, baseK32.
    // Values are verified from GetMatmulApiTiling, including double-buffered L0A/B.
    if (id == LT_MATMUL22_ALGO_H11 || id == LT_MATMUL22_ALGO_H12) {
        const bool tt = id == LT_MATMUL22_ALGO_H11;
        return hardware.aic != 0 && hardware.aiv >= 2U && hardware.l1 >= (tt ? 111616U : 206336U) + 4096U &&
               hardware.l0a >= (tt ? 40960U : 32768U) && hardware.l0b >= (tt ? 32768U : 49152U) &&
               hardware.l0c >= (tt ? 81920U : 98304U);
    }
    return true;
}

bool QueryHardware(bool needCube, uint32_t cubeTile, Hardware& hardware, uint32_t id)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return false;
    }
    hardware.aic = platform->GetCoreNumAic();
    hardware.aiv = platform->GetCoreNumAiv();
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    hardware.ubBytes = ub;
    // Budget includes UB padding/scratch of the narrow-tail and epilogue paths.
    if (hardware.aiv == 0 || ub < 39424U + 4096U) {
        return false;
    }
    if (!needCube) {
        return true;
    }
    auto& l1 = hardware.l1;
    auto& l0a = hardware.l0a;
    auto& l0b = hardware.l0b;
    auto& l0c = hardware.l0c;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    if (id == LT_MATMUL22_ALGO_H10 || id == LT_MATMUL22_ALGO_H11 || id == LT_MATMUL22_ALGO_H12) {
        return SpecialPlanHardware(id, hardware);
    }
    // CANN 9.1.0 static-tiling dump for all four transpose configurations:
    // H0 baseK=32, H1 baseK=64; L0A/B double buffered, L0C single buffered.
    // H1 includes the 96x128 short-K tile. L1 checks use the maximum across
    // transpose modes and supported tiles plus 4096 bytes of reserve.
    const bool largeTile = cubeTile == H1_CUBE_TILE;
    const uint64_t l0abBytes = largeTile ? 64U * 1024U : 16U * 1024U;
    const uint64_t l0cBytes = largeTile ? 64U * 1024U : 16U * 1024U;
    const uint64_t l1Bytes = (largeTile ? 459776U : 279040U) + 4096U;
    return hardware.aic != 0 && hardware.aiv >= 2U && l1 >= l1Bytes && l0a >= l0abBytes && l0b >= l0abBytes &&
           l0c >= l0cBytes;
}

// The manual FP32 loader uses uint16_t ND pitches and K padded to32.
// The H9 capsule is independent of the legacy H0/H1/H2 capacities.
bool H9ShapeSupported(const LtMatmul22Tiling& t)
{
    return !t.transA && !t.transB && t.m != 0U && t.m <= 1280U && t.n >= 16U && t.n <= 576U && t.k != 0U &&
           t.k <= 32U && t.lda <= 65535U && t.ldb <= 65535U;
}

bool H9ResourcesSupported(const Hardware& h)
{
    return h.aic != 0U && h.l1 >= 47104U + 4096U && h.l0a >= 10240U && h.l0b >= 36864U && h.l0c >= 92160U;
}

bool PreferH9(const LtMatmul22Tiling& t)
{
    Hardware h;
    return H9ShapeSupported(t) && t.m >= 256U && t.n >= 256U &&
           QueryHardware(true, H1_CUBE_TILE, h, LT_MATMUL22_ALGO_H9) && H9ResourcesSupported(h) &&
           CeilDivU64(t.m, 80U) * CeilDivU64(t.n, 288U) <= h.aic;
}

aclblasStatus_t LastLaunchStatus()
{
    // This only observes errors already reported by the runtime. Device errors
    // that occur later still belong to the caller's stream synchronization.
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS :
                                                                      ACLBLAS_STATUS_EXECUTION_FAILED;
}

void SelectTallNT(LtMatmul22Tiling& t, uint32_t cores)
{
    // A tall, narrow NT output can leave part of the Cube grid processing a
    // second 64-row tile. Reuse the existing H1 128x128 configuration when it
    // fits the entire row grid into one wave, keeping each full K reduction
    // on its owner and the original H1 storage/resource contract.
    if (!t.shortK && !t.transA && t.transB && t.cubeTile == 64U && t.cubeTileN == 128U && t.n > 64U && t.n <= 128U &&
        t.k > 512U && CeilDivU64(t.m, 64U) > cores && CeilDivU64(t.m, 128U) <= cores) {
        t.cubeTile = 128U;
    }
}

void SelectLongTile(LtMatmul22Tiling& t, uint32_t cores)
{
    // Long TN reductions with two output-job waves benefit from narrower
    // columns. Keep the full K reduction and the existing H1 resource envelope.
    // Avoid an extra wave and retain the original route outside this family.
    if (!t.shortK && t.cubeTile == 128U && t.cubeTileN == 128U && t.transA && !t.transB && t.k > 512U && t.n > 256U &&
        t.n <= 384U) {
        const uint64_t rows = CeilDivU64(t.m, 128U);
        const uint64_t originalJobs = rows * CeilDivU64(t.n, 128U);
        const uint64_t narrowerJobs = rows * CeilDivU64(t.n, 112U);
        if (originalJobs > cores && narrowerJobs <= 2ULL * cores &&
            CeilDivU64(originalJobs, cores) == CeilDivU64(narrowerJobs, cores)) {
            t.cubeTileN = 112U;
        }
    }
    SelectTallNT(t, cores);
}

void SelectMediumTile(LtMatmul22Tiling& t, uint32_t cores)
{
    // Medium NN outputs above three 96-wide tiles with unaligned input pitches benefit from fitting
    // the output into fewer waves. Keep the measured MDL route for smaller,
    // aligned and transposed inputs, where this estimate caused regressions.
    if (!t.transA && !t.transB && t.m > 288U && t.n > 288U && (t.lda % 128U != 0U || t.ldb % 128U != 0U)) {
        constexpr uint32_t shapes[][2] = {{32U, 64U}, {48U, 96U}, {64U, 64U}, {80U, 96U}, {96U, 128U}, {128U, 128U}};
        uint64_t bestCost = std::numeric_limits<uint64_t>::max();
        for (const auto& shape : shapes) {
            const uint64_t jobs = CeilDivU64(t.m, shape[0]) * CeilDivU64(t.n, shape[1]);
            const uint64_t waves = CeilDivU64(jobs, cores);
            const uint64_t cost = waves * (static_cast<uint64_t>(shape[0]) * shape[1] + 1024U);
            if (cost < bestCost) {
                bestCost = cost;
                t.cubeTile = shape[0];
                t.cubeTileN = shape[1];
            }
        }
    }
}

bool IsShortTransposeRectangle(const LtMatmul22Tiling& t)
{
    return t.transA && !t.transB && t.k > 64U && t.k <= 128U && t.m > 512U && t.n > 512U;
}

bool IsMediumTransposeRectangle(const LtMatmul22Tiling& t)
{
    return t.transA && !t.transB && t.m > 128U && t.m <= 256U && t.n > 512U && t.k > 128U && t.k <= 512U &&
           static_cast<uint64_t>(t.m) * t.n > 512U * 512U;
}

void SelectSmallTile(LtMatmul22Tiling& t, uint32_t cores)
{
    // Small aligned outputs benefit from more Cube groups and the lower
    // NORM setup cost. Larger outputs retain the measured 64x128/128x128
    // configuration; area-only wave estimates do not model ND2NZ traffic.
    if (t.k <= 512U && static_cast<uint64_t>(t.m) * t.n <= 128U * 256U) {
        t.shortK = 1U;
        t.cubeTile = 32U;
        t.cubeTileN = 64U;
    } else if (IsShortTransposeRectangle(t)) {
        // Measured transposed short reductions benefit from the existing full-K
        // 128x128 MDL static configuration. Capacity and splitK remain H1.
        t.shortK = 1U;
        t.cubeTile = 128U;
        t.cubeTileN = 128U;
    } else if (IsMediumTransposeRectangle(t)) {
        // Keep the existing large-output tile, but select its measured full-K
        // 512 bound for medium transposed rectangles such as 188x1686x360.
        t.shortK = 1U;
        t.cubeTile = 128U;
        t.cubeTileN = 128U;
    } else if (t.k <= 64U) {
        // A short reduction is dominated by output-tile setup and movement.
        // Reuse B across four base-M tiles inside one 256x128 logical tile.
        // The SDK keeps only 64x128 FP32 accumulators live in L0C.
        t.shortK = 1U;
        t.cubeTile = 256U;
        t.cubeTileN = 128U;
    } else if (t.k <= 512U && t.m <= 512U && t.n <= 512U) {
        t.shortK = 1U;
        t.cubeTile = 64U;
        t.cubeTileN = 128U;
        SelectMediumTile(t, cores);
    } else if (t.n >= 64U && t.n <= 128U && t.m >= 128U && t.m <= 512U) {
        // Narrow N otherwise leaves only four Cube groups in cases such as
        // 254x74x1274. Split output rows/columns, keeping each full K reduction
        // on its owner and retaining the original FP32 operation contract.
        t.cubeTile = 32U;
        t.cubeTileN = 64U;
    }
    SelectLongTile(t, cores);
}

// Populate more Cube owners only when the long-TN output remains in one wave.
// Runtime pitches, logical tails, full K reduction and the H1 capsule are unchanged.
bool PreferMoreLongTNOwners(const LtMatmul22Tiling& t, uint32_t cores)
{
    if (t.shortK || !t.transA || t.transB || t.k <= 512U || t.k > 4096U || t.cubeTile != 128U || t.cubeTileN != 128U) {
        return false;
    }
    const uint64_t oldJobs = CeilDivU64(t.m, 128U) * CeilDivU64(t.nCube, 128U);
    const uint64_t newJobs = CeilDivU64(t.m, 64U) * CeilDivU64(t.nCube, 192U);
    return oldJobs < newJobs && newJobs <= cores;
}

aclblasStatus_t FillHeuristic(uint32_t id, size_t bytes, aclblasLtMatmulHeuristicResult_t& result)
{
    result = aclblasLtMatmulHeuristicResult_t{};
    const aclblasStatus_t status = LtMatmul22AlgoInit(static_cast<int>(id), result.algo);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    result.workspaceSize = id == LT_MATMUL22_ALGO_H0 ? 0 : bytes;
    result.algo.max_workspace_bytes = result.workspaceSize;
    result.state = ACLBLAS_STATUS_SUCCESS;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchFullColumns(
    const MatmulProblem& problem, LtMatmul22Tiling& t, uint32_t id, const Hardware& hardware, uint32_t vectors)
{
    const uint64_t tileCount = CeilDivU64(t.m, t.cubeTile) * CeilDivU64(t.nCube, t.cubeTileN);
    const bool mixed = id == LT_MATMUL22_ALGO_H0 && problem.C == problem.D && t.beta != 0.0f;
    const uint32_t coreLimit = mixed ? std::min(hardware.aic, hardware.aiv / 2U) : hardware.aic;
    const uint32_t groups = static_cast<uint32_t>(std::min<uint64_t>(tileCount, coreLimit));
    const bool trivialEpilogue = t.alpha == 1.0f && t.beta == 0.0f;
    if (mixed) {
        LtMatmul22Mix(
            groups, problem.stream, Bytes(problem.A), Bytes(problem.B), Bytes(problem.C), Bytes(problem.D), t);
        return LastLaunchStatus();
    }
    // Native ND copy-out writes the logical N exactly, including its tail.
    // The default epilogue can therefore target D with any legal row pitch.
    const bool staging = id == LT_MATMUL22_ALGO_H1 && !trivialEpilogue;
    void* productAddress = staging ? problem.workspace : problem.D;
    if (!staging) {
        t.productLd = t.ldd;
    }
    if (id == LT_MATMUL22_ALGO_H1 && PreferMoreLongTNOwners(t, hardware.aic)) {
        t.cubeTile = 64U;
        t.cubeTileN = 192U;
        const uint32_t owners = static_cast<uint32_t>(CeilDivU64(t.m, 64U) * CeilDivU64(t.nCube, 192U));
        LtMatmul22CubeTN64x192(owners, problem.stream, Bytes(problem.A), Bytes(problem.B), Bytes(productAddress), t);
    } else {
        LtMatmul22Cube(groups, problem.stream, Bytes(problem.A), Bytes(problem.B), Bytes(productAddress), t);
    }
    if (LastLaunchStatus() != ACLBLAS_STATUS_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (!trivialEpilogue || staging) {
        LtMatmul22Epilogue(vectors, problem.stream, Bytes(productAddress), Bytes(problem.C), Bytes(problem.D), t);
    }
    return LastLaunchStatus();
}

aclblasStatus_t LaunchFusedShort(const MatmulProblem& problem, LtMatmul22Tiling& t, const Hardware& hardware)
{
    H2Storage storage;
    const aclblasStatus_t status = BuildFusedShortStorage(t, storage);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    const uint32_t chunks = (t.m + 15U) / 16U;
    const uint32_t groups = std::min(std::min(hardware.aic, hardware.aiv / 2U), chunks);
    if (groups == 0U || (chunks + groups - 1U) / groups * 16U > 128U) {
        return ACLBLAS_STATUS_ARCH_MISMATCH;
    }
    void* syncAddress = nullptr;
    if (aclrtGetHardwareSyncAddr(&syncAddress) != ACL_SUCCESS || syncAddress == nullptr) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    t.cubeTile = 64U;
    t.cubeTileN = 480U;
    t.shortK = 1U;
    t.nCube = t.n;
    t.productLd = t.ldd;
    LtMatmul22FusedShort(
        groups, hardware.aiv, problem.stream, Bytes(problem.A), Bytes(problem.B), Bytes(problem.D),
        Bytes(problem.workspace), t, reinterpret_cast<uint64_t>(syncAddress));
    return LastLaunchStatus();
}

aclblasStatus_t LaunchRect(
    const MatmulProblem& problem, LtMatmul22Tiling& t, uint32_t id, const Hardware& hardware, uint32_t vectors)
{
    const bool tt = id == LT_MATMUL22_ALGO_H11;
    t.cubeTile = tt ? 160U : 128U;
    t.cubeTileN = tt ? 128U : 192U;
    const bool staging = t.alpha != 1.0f || t.beta != 0.0f;
    H1Storage storage;
    const aclblasStatus_t status = BuildRectStorage(id, t, storage);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    t.productLd = staging ? storage.productPitch : t.ldd;
    const uint64_t jobs = CeilDivU64(t.m, t.cubeTile) * CeilDivU64(t.n, t.cubeTileN);
    LtMatmul22RectShort(
        static_cast<uint32_t>(std::min<uint64_t>(hardware.aic, jobs)), problem.stream, Bytes(problem.A),
        Bytes(problem.B), Bytes(staging ? problem.workspace : problem.D), t);
    if (LastLaunchStatus() != ACLBLAS_STATUS_SUCCESS)
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    if (staging)
        LtMatmul22Epilogue(vectors, problem.stream, Bytes(problem.workspace), Bytes(problem.C), Bytes(problem.D), t);
    return LastLaunchStatus();
}

aclblasStatus_t LaunchSplit(const MatmulProblem& problem, LtMatmul22Tiling& t, const Hardware& hardware)
{
    t.productLd = (t.n + 15U) / 16U * 16U;
    t.cubeTile = 32U;
    t.cubeTileN = 64U;
    auto* partials =
        reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(problem.workspace) + 31U) & ~uintptr_t(31U));
    const uint64_t jobs = CeilDivU64(t.m, 32U) * CeilDivU64(t.n, 64U) * 4U;
    LtMatmul22SplitCube(
        static_cast<uint32_t>(std::min<uint64_t>(hardware.aic, jobs)), problem.stream, Bytes(problem.A),
        Bytes(problem.B), partials, t, 4U);
    if (LastLaunchStatus() != ACLBLAS_STATUS_SUCCESS)
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    LtMatmul22SplitReduce(
        std::min(hardware.aiv, static_cast<uint32_t>(CeilDivU64(t.m, 8U))), problem.stream, partials, Bytes(problem.C),
        Bytes(problem.D), t, 4U);
    return LastLaunchStatus();
}

aclblasStatus_t LaunchANZ(const MatmulProblem& problem, LtMatmul22Tiling& t, const Hardware& hardware, uint32_t vectors)
{
    H4Storage storage;
    const aclblasStatus_t status = BuildH4Storage(t, storage);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    auto* aNz = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(problem.workspace) + storage.product.bytes + 31U) & ~uintptr_t(31U));
    LtMatmul22PackOneNZ(
        std::min(hardware.aiv, static_cast<uint32_t>(CeilDivU64(t.k, 16U))), problem.stream, Bytes(problem.A), aNz, t,
        0U);
    if (LastLaunchStatus() != ACLBLAS_STATUS_SUCCESS)
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    t.cubeTile = 64U;
    t.cubeTileN = 128U;
    const bool staging = t.alpha != 1.0f || t.beta != 0.0f;
    t.productLd = staging ? storage.product.productPitch : t.ldd;
    const uint64_t jobs = CeilDivU64(t.m, 64U) * CeilDivU64(t.n, 128U);
    LtMatmul22MixedNZCube(
        static_cast<uint32_t>(std::min<uint64_t>(hardware.aic, jobs)), problem.stream, aNz, Bytes(problem.B),
        Bytes(staging ? problem.workspace : problem.D), t, 0U);
    if (LastLaunchStatus() != ACLBLAS_STATUS_SUCCESS)
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    if (staging)
        LtMatmul22Epilogue(vectors, problem.stream, Bytes(problem.workspace), Bytes(problem.C), Bytes(problem.D), t);
    return LastLaunchStatus();
}

void LaunchNZPacking(
    const MatmulProblem& problem, const LtMatmul22Tiling& t, const Hardware& hardware, uint8_t* aNz, uint8_t* bNz)
{
    const bool pack80 = PreferSmallNNNZ(t) && hardware.ubBytes >= 166400U + 4096U;
    const bool pack64 =
        (PreferNarrowNNNZ(t) || (((!t.transA && t.transB && t.m >= 64U && t.m <= 128U) || (t.transA && t.m >= 512U)) &&
                                 t.n >= 128U && t.n <= 256U && t.k >= 512U)) &&
        hardware.ubBytes >= 133120U + 4096U;
    const uint32_t packRows = pack80 ? 80U : pack64 ? 64U : 16U;
    const uint64_t aTiles = CeilDivU64(t.transA ? t.k : t.m, packRows) * CeilDivU64(t.transA ? t.m : t.k, 256U);
    const uint64_t bTiles = CeilDivU64(t.transB ? t.n : t.k, packRows) * CeilDivU64(t.transB ? t.k : t.n, 256U);
    const uint32_t packBlocks = static_cast<uint32_t>(std::min<uint64_t>(hardware.aiv, aTiles + bTiles));
    if (pack80) {
        LtMatmul22PackNZ80(packBlocks, problem.stream, Bytes(problem.A), Bytes(problem.B), aNz, bNz, t);
    } else if (pack64) {
        LtMatmul22PackNZ64(packBlocks, problem.stream, Bytes(problem.A), Bytes(problem.B), aNz, bNz, t);
    } else {
        LtMatmul22PackNZ(packBlocks, problem.stream, Bytes(problem.A), Bytes(problem.B), aNz, bNz, t);
    }
}

void SelectNZTile(LtMatmul22Tiling& t, bool alignedDirect)
{
    t.cubeTile = t.n <= 64U ? 32U : static_cast<uint64_t>(t.m) * t.n <= 512U * 512U ? 64U : 128U;
    t.cubeTileN = t.n <= 64U ? 64U : 128U;
    // Keep four full-K owners in the small direct TN family while reducing
    // the padded N work. Packing, public alignment and H2 capacity are unchanged.
    if (alignedDirect && t.n <= 192U && t.k <= 2048U) {
        t.cubeTileN = 96U;
    }
    if (!t.transA && t.transB && t.m >= 64U && t.m <= 128U && t.n >= 128U && t.n <= 256U && t.k > 512U &&
        t.k <= 2048U && t.alpha == 1.0f && t.beta == 0.0f) {
        t.cubeTile = 32U;
        t.cubeTileN = 128U;
    }
    if (PreferNarrowNNNZ(t)) {
        t.cubeTile = 64U;
        t.cubeTileN = 64U;
    }
    if (PreferSmallNNNZ(t)) {
        t.cubeTile = 32U;
        t.cubeTileN = 64U;
    }
}

bool NZAlignedDirect(const LtMatmul22Tiling& t, bool staging, size_t productBytes)
{
    return !staging && t.transA && !t.transB && t.m >= 65U && t.m <= 128U && t.n >= 129U && t.n <= 256U && t.k > 512U &&
           t.k <= 4096U && productBytes >= 512U;
}

bool NZAlignedInputs(const LtMatmul22Tiling& t, bool staging, size_t productBytes, bool alignedDirect)
{
    const bool nt = !t.transA && t.transB && t.m >= 64U && t.m <= 128U && t.n >= 128U && t.n <= 256U && t.k >= 512U;
    return alignedDirect || ((PreferSmallNNNZ(t) || PreferNarrowNNNZ(t) || nt) && !staging && productBytes >= 512U);
}

aclblasStatus_t LaunchNZ(const MatmulProblem& problem, LtMatmul22Tiling& t, const Hardware& hardware, uint32_t vectors)
{
    H2Storage storage;
    const aclblasStatus_t status = BuildH2Storage(t, storage);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const bool staging = t.alpha != 1.0f || t.beta != 0.0f;
    // A direct output does not consume the reserved product plane. In this
    // measured narrow-M family, reuse that space to align both packed inputs
    // to 512B. Each physical NZ matrix is an integral number of 512B tiles.
    // A legal 16B workspace needs at most 496B of prefix; the unchanged product
    // reserve is at least 512B, so both inputs remain inside the H2 capacity.
    const bool alignedDirect = NZAlignedDirect(t, staging, storage.product.bytes);
    const bool alignedInputs = NZAlignedInputs(t, staging, storage.product.bytes, alignedDirect);
    const uintptr_t base = reinterpret_cast<uintptr_t>(problem.workspace);
    const uintptr_t plane =
        alignedInputs ? (base + 511U) & ~uintptr_t(511U) : (base + storage.product.bytes + 31U) & ~uintptr_t(31U);
    auto* aNz = reinterpret_cast<uint8_t*>(plane);
    auto* bNz = aNz + storage.aBytes;
    LaunchNZPacking(problem, t, hardware, aNz, bNz);
    if (LastLaunchStatus() != ACLBLAS_STATUS_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    SelectNZTile(t, alignedDirect);
    const uint64_t jobs = CeilDivU64(t.m, t.cubeTile) * CeilDivU64(t.n, t.cubeTileN);
    void* product = staging ? problem.workspace : problem.D;
    t.productLd = staging ? storage.product.productPitch : t.ldd;
    LtMatmul22CubeNZ(
        static_cast<uint32_t>(std::min<uint64_t>(hardware.aic, jobs)), problem.stream, aNz, bNz, Bytes(product), t);
    if (LastLaunchStatus() != ACLBLAS_STATUS_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (staging) {
        LtMatmul22Epilogue(vectors, problem.stream, Bytes(product), Bytes(problem.C), Bytes(problem.D), t);
    }
    return LastLaunchStatus();
}

struct HeuristicPlans {
    H1Storage h1, h3, h11, h12;
    H2Storage h2, h10;
    H4Storage h4;
    bool hasH1 = false, hasH2 = false, hasH3 = false, hasH4 = false;
    bool hasH10 = false, hasH11 = false, hasH12 = false;

    aclblasStatus_t Build(const LtMatmul22Tiling& t, size_t budget)
    {
        const aclblasStatus_t h1Status = BuildH1Storage(t.m, t.n, h1);
        if (h1Status != ACLBLAS_STATUS_SUCCESS && h1Status != ACLBLAS_STATUS_NOT_SUPPORTED) {
            return h1Status;
        }
        // A zero preference is a real zero-byte budget. Unsupported aligned pitch
        // or narrow N produces only H0 rather than truncating an H1 pitch.
        hasH1 = h1Status == ACLBLAS_STATUS_SUCCESS && h1.bytes != 0 && h1.bytes <= budget;
        hasH2 = BuildH2Storage(t, h2) == ACLBLAS_STATUS_SUCCESS && h2.bytes <= budget && PreferH2(t, h2);
        hasH3 = BuildH3Storage(t, h3) == ACLBLAS_STATUS_SUCCESS && h3.bytes <= budget;
        hasH4 = BuildH4Storage(t, h4) == ACLBLAS_STATUS_SUCCESS && h4.bytes <= budget;
        hasH11 = BuildRectStorage(LT_MATMUL22_ALGO_H11, t, h11) == ACLBLAS_STATUS_SUCCESS && h11.bytes <= budget;
        hasH12 = BuildRectStorage(LT_MATMUL22_ALGO_H12, t, h12) == ACLBLAS_STATUS_SUCCESS && h12.bytes <= budget;
        hasH10 = BuildFusedShortStorage(t, h10) == ACLBLAS_STATUS_SUCCESS && h10.bytes <= budget;
        return ACLBLAS_STATUS_SUCCESS;
    }

    int Append(const LtMatmul22Tiling& t, uint32_t* ids, size_t* capacities) const
    {
        int available = 0;
        if (hasH1 && PreferH9(t)) {
            ids[available] = LT_MATMUL22_ALGO_H9;
            capacities[available++] = h1.bytes;
        }
        if (hasH10) {
            ids[available] = LT_MATMUL22_ALGO_H10;
            capacities[available++] = h10.bytes;
        }
        if (hasH11) {
            ids[available] = LT_MATMUL22_ALGO_H11;
            capacities[available++] = h11.bytes;
        }
        if (hasH12) {
            ids[available] = LT_MATMUL22_ALGO_H12;
            capacities[available++] = h12.bytes;
        }
        if (hasH3) {
            ids[available] = LT_MATMUL22_ALGO_H3;
            capacities[available++] = h3.bytes;
        }
        if (hasH1 && PreferH5(t)) {
            ids[available] = LT_MATMUL22_ALGO_H5;
            capacities[available++] = h1.bytes;
        }
        if (hasH2) {
            ids[available] = LT_MATMUL22_ALGO_H2;
            capacities[available++] = h2.bytes;
        }
        if (hasH4) {
            ids[available] = LT_MATMUL22_ALGO_H4;
            capacities[available++] = h4.bytes;
        }
        if (hasH1) {
            ids[available] = LT_MATMUL22_ALGO_H1;
            capacities[available++] = h1.bytes;
        }
        ids[available++] = LT_MATMUL22_ALGO_H0;
        return available;
    }
};

aclblasStatus_t InitializeLaunchPlan(LtMatmul22Tiling& t, uint32_t id)
{
    t.cubeTile = H0_CUBE_TILE;
    t.cubeTileN = H0_CUBE_TILE;
    if (id == LT_MATMUL22_ALGO_H1 || id == LT_MATMUL22_ALGO_H2 || id == LT_MATMUL22_ALGO_H4 ||
        id == LT_MATMUL22_ALGO_H11 || id == LT_MATMUL22_ALGO_H12 || id == LT_MATMUL22_ALGO_H10 ||
        id == LT_MATMUL22_ALGO_H9 || id == LT_MATMUL22_ALGO_H5) {
        H1Storage h1;
        const aclblasStatus_t h1Status = BuildH1Storage(t.m, t.n, h1);
        if (h1Status != ACLBLAS_STATUS_SUCCESS) {
            return h1Status;
        }
        // nCube is covered logical columns, not the padded output pitch.
        t.nCube = t.n;
        t.productLd = h1.productPitch;
        t.cubeTile = H1_CUBE_TILE;
        t.cubeTileN = H1_CUBE_TILE;
        // H1's capsule retains the baseline configuration and workspace contract.
        // Internal tiles must fit its L1/L0 resource limits; split small outputs
        // to populate more Cube groups without changing storage.
        const bool longRectangle = t.k > 512U && std::min(t.m, t.n) > 256U && std::max(t.m, t.n) > 512U;
        if (static_cast<uint64_t>(t.m) * t.n <= 512U * 512U && !longRectangle) {
            t.cubeTile = H0_CUBE_TILE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool LongNZBeta(const LtMatmul22Tiling& t, uint32_t id)
{
    return id == LT_MATMUL22_ALGO_H2 && t.transA && !t.transB && t.m >= 1024U && t.m <= 2048U && t.n > 128U &&
           t.n <= 256U && t.k >= 512U && t.k <= 2048U && t.beta != 0.0f;
}

bool RequiresOrdered(const LtMatmul22Tiling& t, uint32_t id, bool product)
{
    // Tiny matrices do not benefit from Cube tiling. Separate FP32 multiply
    // and reduction preserve product-overflow behavior: Cube's wider internal
    // accumulation can cancel overflow to a finite value where Netlib has
    // already produced Inf/NaN.
    const bool tinyVector = product && t.n <= 16U && static_cast<uint64_t>(t.m) * t.n <= 8192U / t.k;
    // Amplified long reductions can exceed the Netlib maximum-error bound
    // despite a good matched ratio. Preserve its FP32 operation order for this
    // numerical regime and for nonfinite scalars; no caller workspace is used.
    // In this long-K H2 family, adding a large beta*C after a Cube product
    // can exceed Netlib's maximum-error bound. Preserve its beta initialization
    // and FP32 accumulation order whenever beta is active. PF alpha1/beta0
    // continues to use the measured NZ pipeline.
    const bool longNzBeta = LongNZBeta(t, id);
    // H10's fused path computes alpha1/beta0. Other scalars retain the
    // existing Netlib-ordered FP32 fallback without using the packed planes.
    const bool fusedShortOrdered = id == LT_MATMUL22_ALGO_H10 && (t.alpha != 1.0f || t.beta != 0.0f);
    return product && (fusedShortOrdered || tinyVector || t.k > 4096U || !std::isfinite(t.alpha) ||
                       !std::isfinite(t.beta) || (std::abs(t.alpha) > 16.0f && t.k >= 256U) || longNzBeta);
}

aclblasStatus_t LaunchRows160(
    const MatmulProblem& problem, LtMatmul22Tiling& t, const Hardware& hardware, uint32_t vectors)
{
    const bool staging = t.alpha != 1.0f || t.beta != 0.0f;
    if (!staging)
        t.productLd = t.ldd;
    void* productAddress = staging ? problem.workspace : problem.D;
    const uint32_t groups = static_cast<uint32_t>(std::min<uint64_t>(hardware.aic, CeilDivU64(t.m, 8U)));
    const uint64_t chunks = CeilDivU64(t.m, 8U);
    const uint64_t ownedRows = CeilDivU64(chunks, groups) * 8U;
    if (ownedRows <= 96U) {
        LtMatmul22Rows160(groups, problem.stream, Bytes(problem.A), Bytes(problem.B), Bytes(productAddress), t);
    } else {
        // Keep low-core-count hardware inside H5's maximum 96x160 capsule.
        // The existing 64x128 native path also uses one complete K reduction.
        t.cubeTile = 64U;
        t.cubeTileN = 128U;
        t.shortK = 1U;
        const uint64_t jobs = CeilDivU64(t.m, 64U) * CeilDivU64(t.n, 128U);
        LtMatmul22Cube(
            static_cast<uint32_t>(std::min<uint64_t>(hardware.aic, jobs)), problem.stream, Bytes(problem.A),
            Bytes(problem.B), Bytes(productAddress), t);
    }
    if (LastLaunchStatus() != ACLBLAS_STATUS_SUCCESS)
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    if (staging)
        LtMatmul22Epilogue(vectors, problem.stream, Bytes(productAddress), Bytes(problem.C), Bytes(problem.D), t);
    return LastLaunchStatus();
}

aclblasStatus_t LaunchNative(
    const MatmulProblem& problem, LtMatmul22Tiling& t, uint32_t id, const Hardware& hardware, uint32_t vectors)
{
    if (t.nCube != 0) {
        const aclblasStatus_t prefixStatus = LaunchFullColumns(problem, t, id, hardware, vectors);
        if (prefixStatus != ACLBLAS_STATUS_SUCCESS) {
            return prefixStatus;
        }
    }
    // A separate stream task avoids prefix/tail stores sharing a cache line concurrently.
    if (t.nCube != t.n) {
        LtMatmul22Tail(
            vectors, problem.stream, Bytes(problem.A), Bytes(problem.B), Bytes(problem.C), Bytes(problem.D), t);
        return LastLaunchStatus();
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchProduct(
    const MatmulProblem& problem, LtMatmul22Tiling& t, uint32_t id, const Hardware& hardware, uint32_t vectors)
{
    if (id == LT_MATMUL22_ALGO_H9) {
        if (H9ShapeSupported(t) && H9ResourcesSupported(hardware) && t.alpha == 1.0f && t.beta == 0.0f) {
            const uint64_t jobs = CeilDivU64(t.m, 80U) * CeilDivU64(t.n, 288U);
            LtMatmul22ShortKColumns(
                static_cast<uint32_t>(std::min<uint64_t>(hardware.aic, jobs)), problem.stream, Bytes(problem.A),
                Bytes(problem.B), Bytes(problem.D), t);
            return LastLaunchStatus();
        }
        // H9 reserves the full H1 product, so unsupported layouts, scalar
        // epilogues, pitches and hardware use the established H1 implementation.
        id = LT_MATMUL22_ALGO_H1;
    }
    if (id == LT_MATMUL22_ALGO_H10)
        return LaunchFusedShort(problem, t, hardware);
    if (id == LT_MATMUL22_ALGO_H11 || id == LT_MATMUL22_ALGO_H12) {
        return LaunchRect(problem, t, id, hardware, vectors);
    }
    if (id == LT_MATMUL22_ALGO_H3) {
        return LaunchSplit(problem, t, hardware);
    }
    if (id == LT_MATMUL22_ALGO_H4) {
        return LaunchANZ(problem, t, hardware, vectors);
    }
    if (id == LT_MATMUL22_ALGO_H5)
        return LaunchRows160(problem, t, hardware, vectors);
    if (id == LT_MATMUL22_ALGO_H2) {
        return LaunchNZ(problem, t, hardware, vectors);
    }
    return LaunchNative(problem, t, id, hardware, vectors);
}

aclblasStatus_t LaunchScale(const MatmulProblem& problem, const LtMatmul22Tiling& t, uint32_t id, uint32_t vectors)
{
    if (id == LT_MATMUL22_ALGO_H10 && t.beta != 0.0f && t.beta != 1.0f) {
        LtMatmul22FusedScale(vectors, problem.stream, Bytes(problem.C), Bytes(problem.D), t);
    } else {
        LtMatmul22Scale(vectors, problem.stream, Bytes(problem.C), Bytes(problem.D), t);
    }
    return LastLaunchStatus();
}

LtMatmul22Tiling ProblemTiling(const MatmulProblem& problem)
{
    return LtMatmul22Tiling{
        static_cast<uint32_t>(problem.m),
        static_cast<uint32_t>(problem.n),
        static_cast<uint32_t>(problem.k),
        problem.lda,
        problem.ldb,
        problem.ldc,
        problem.ldd,
        static_cast<uint32_t>(problem.n / 16U * 16U),
        static_cast<uint32_t>(problem.transA),
        static_cast<uint32_t>(problem.transB),
        problem.ldd,
        problem.alpha,
        problem.beta};
}

aclblasStatus_t LaunchValidated(
    const Desc& desc, const Layout& aLayout, const Layout& bLayout, const Layout& cLayout, const Layout& dLayout,
    const LtMatmul22Tiling& t, const void* alpha, const void* beta, const void* a, const void* b, const void* c,
    void* d, const aclblasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceBytes, aclrtStream stream)
{
    MatmulProblem problem;
    BuildMatmulProblem(
        &desc, &aLayout, &bLayout, &cLayout, &dLayout, t.m, t.n, t.k, alpha, beta, a, b, c, d, algo, workspace,
        workspaceBytes, 0, stream, problem);
    problem.lda = t.lda;
    problem.ldb = t.ldb;
    problem.ldc = t.ldc;
    problem.ldd = t.ldd;
    return MatmulLaunch(problem);
}

} // namespace

aclblasStatus_t LtMatmul22AlgoInit(int id, aclblasLtMatmulAlgo_t& algo)
{
    const uint32_t planId = id == 0 ? LT_MATMUL22_ALGO_H0 : static_cast<uint32_t>(id);
    if (planId != LT_MATMUL22_ALGO_H0 && planId != LT_MATMUL22_ALGO_H1 && planId != LT_MATMUL22_ALGO_H2 &&
        planId != LT_MATMUL22_ALGO_H3 && planId != LT_MATMUL22_ALGO_H4 && planId != LT_MATMUL22_ALGO_H11 &&
        planId != LT_MATMUL22_ALGO_H12 && planId != LT_MATMUL22_ALGO_H10 && planId != LT_MATMUL22_ALGO_H9 &&
        planId != LT_MATMUL22_ALGO_H5) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const PackedAlgo packed = PlanCapsule(planId);
    if (!MemcpySSucceeds(algo.data, sizeof(algo.data), &packed, sizeof(packed))) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    // AlgoInit has no shape; launch derives the actual H1 capacity from layouts.
    algo.max_workspace_bytes = 0;
    return ACLBLAS_STATUS_SUCCESS;
}

// issue #808: AlgoCheck 的本模块分流入口。与执行链（LtMatmul22Api）同序复用宿主侧校验：
// 描述符 magic/pointerMode → 空矩阵短路 → ValidateLayouts（布局/类型/维度）→ DecodePlan
// （plan 域与胶囊字段一致性）→ RequiredPlanWorkspace（形状门控 + workspace 内在需求）。
// 运行期资源项（硬件查询 / 操作数指针重叠 / workspace 容量与对齐）依执行链入参不可见，
// 与 legacy AlgoCheck「无设备查询、无内核启动、无分配」的边界一致。
// result 回填沿用 legacy FillAlgoCheckResult 口径：原样回显 algo（不重写
// max_workspace_bytes，用户可安全复用/再序列化）、state=SUCCESS、wavesCount=1.0f、
// reserved 清零；workspaceSize = max(plan 内在需求, algo->max_workspace_bytes)。
aclblasStatus_t LtMatmul22AlgoCheck(
    aclblasLtMatmulDesc_t computeDesc, aclblasLtMatrixLayout_t aDesc, aclblasLtMatrixLayout_t bDesc,
    aclblasLtMatrixLayout_t cDesc, aclblasLtMatrixLayout_t dDesc, const aclblasLtMatmulAlgo_t* algo,
    aclblasLtMatmulHeuristicResult_t* result)
{
    const auto& desc = *reinterpret_cast<const Desc*>(computeDesc);
    if (desc.magic != ACLBLASLT_DESC_MAGIC || desc.pointerMode != ACLBLASLT_POINTER_MODE_HOST) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const auto& aLayout = *reinterpret_cast<const Layout*>(aDesc);
    const auto& bLayout = *reinterpret_cast<const Layout*>(bDesc);
    const auto& cLayout = *reinterpret_cast<const Layout*>(cDesc);
    const auto& dLayout = *reinterpret_cast<const Layout*>(dDesc);
    // 执行链契约：空矩阵先于布局/类型校验短路成功；AlgoCheck 需回填 result，按 legacy
    // no-op 口径回填 workspaceSize=0。
    // issue #835: 该校验顺序（legacy 链路为标量校验先于 no-op）与 OP_C 放行、非 FP32
    // scaleType 错误码同属已声明的 per-arch 差异，见 blasLt/README.md"架构支持域与
    // 描述符约定"对照表（含空矩阵行为行）与 blas/lt_matmul/README.md 校验顺序声明。
    if (dLayout.rows == 0U || dLayout.cols == 0U) {
        *result = aclblasLtMatmulHeuristicResult_t{};
        result->algo = *algo;
        result->workspaceSize = 0;
        result->state = ACLBLAS_STATUS_SUCCESS;
        result->wavesCount = 1.0f;
        return ACLBLAS_STATUS_SUCCESS;
    }
    LtMatmul22Tiling t{};
    const aclblasStatus_t layoutStatus = ValidateLayouts(desc, aLayout, bLayout, cLayout, dLayout, t);
    if (layoutStatus != ACLBLAS_STATUS_SUCCESS) {
        return layoutStatus;
    }
    uint32_t id = LT_MATMUL22_ALGO_H0;
    const aclblasStatus_t planStatus = DecodePlan(algo, id);
    if (planStatus != ACLBLAS_STATUS_SUCCESS) {
        return planStatus;
    }
    H1Storage h1;
    const aclblasStatus_t h1Status = BuildH1Storage(t.m, t.n, h1);
    size_t required = 0;
    const aclblasStatus_t requiredStatus = RequiredPlanWorkspace(id, t, h1, h1Status, required);
    if (requiredStatus != ACLBLAS_STATUS_SUCCESS) {
        return requiredStatus;
    }
    *result = aclblasLtMatmulHeuristicResult_t{};
    result->algo = *algo;
    const size_t advertised = algo->max_workspace_bytes;
    result->workspaceSize = required > advertised ? required : advertised;
    result->state = ACLBLAS_STATUS_SUCCESS;
    result->wavesCount = 1.0f;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LtMatmul22Api(
    aclblasLtHandle_t handle, aclblasLtMatmulDesc_t computeDesc, const void* alpha, const void* a,
    aclblasLtMatrixLayout_t aDesc, const void* b, aclblasLtMatrixLayout_t bDesc, const void* beta, const void* c,
    aclblasLtMatrixLayout_t cDesc, void* d, aclblasLtMatrixLayout_t dDesc, const aclblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceBytes, aclrtStream stream)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (computeDesc == nullptr || aDesc == nullptr || bDesc == nullptr || cDesc == nullptr || dDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const auto& desc = *reinterpret_cast<const Desc*>(computeDesc);
    if (desc.magic != ACLBLASLT_DESC_MAGIC || desc.pointerMode != ACLBLASLT_POINTER_MODE_HOST) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const auto& aLayout = *reinterpret_cast<const Layout*>(aDesc);
    const auto& bLayout = *reinterpret_cast<const Layout*>(bDesc);
    const auto& cLayout = *reinterpret_cast<const Layout*>(cDesc);
    const auto& dLayout = *reinterpret_cast<const Layout*>(dDesc);
    // The contract intentionally places this before layout/type/enum validation.
    // issue #835: 空矩阵 no-op 先于标量/枚举校验的顺序差异已显式声明，见
    // blasLt/README.md 架构支持域对照表的"空矩阵"行与 arch22 算子说明。
    if (dLayout.rows == 0U || dLayout.cols == 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    LtMatmul22Tiling t{};
    aclblasStatus_t status = ValidateLayouts(desc, aLayout, bLayout, cLayout, dLayout, t);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    t.alpha = *static_cast<const float*>(alpha);
    t.beta = *static_cast<const float*>(beta);
    status = ValidateOperandPointers(t, a, b, c, d);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateOverlaps(a, b, c, d, aLayout, bLayout, cLayout, dLayout, t, workspace, workspaceBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    uint32_t plan = 0;
    status = ValidateWorkspace(algo, t, workspace, workspaceBytes, plan);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return LaunchValidated(
        desc, aLayout, bLayout, cLayout, dLayout, t, alpha, beta, a, b, c, d, algo, workspace, workspaceBytes, stream);
}

aclblasStatus_t LtMatmul22Heuristic(
    aclblasLtHandle_t handle, aclblasLtMatmulDesc_t computeDesc, aclblasLtMatrixLayout_t aDesc,
    aclblasLtMatrixLayout_t bDesc, aclblasLtMatrixLayout_t cDesc, aclblasLtMatrixLayout_t dDesc,
    aclblasLtMatmulPreference_t preference, int requested, aclblasLtMatmulHeuristicResult_t* results, int* count)
{
    if (count == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *count = 0;
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (computeDesc == nullptr || aDesc == nullptr || bDesc == nullptr || cDesc == nullptr || dDesc == nullptr ||
        results == nullptr || requested <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const auto& desc = *reinterpret_cast<const Desc*>(computeDesc);
    if (desc.magic != ACLBLASLT_DESC_MAGIC || desc.pointerMode != ACLBLASLT_POINTER_MODE_HOST) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    LtMatmul22Tiling t{};
    const aclblasStatus_t status = ValidateLayouts(
        desc, *reinterpret_cast<const Layout*>(aDesc), *reinterpret_cast<const Layout*>(bDesc),
        *reinterpret_cast<const Layout*>(cDesc), *reinterpret_cast<const Layout*>(dDesc), t);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    size_t budget = DEFAULT_WORKSPACE_SIZE;
    if (preference != nullptr) {
        budget = reinterpret_cast<const aclblasLtMatmulPreferenceImpl*>(preference)->maxWorkspaceBytes;
    }
    HeuristicPlans plans;
    const aclblasStatus_t planStatus = plans.Build(t, budget);
    if (planStatus != ACLBLAS_STATUS_SUCCESS)
        return planStatus;
    uint32_t ids[10]{};
    size_t capacities[10]{};
    int available = plans.Append(t, ids, capacities);
    available = std::min(requested, available);
    for (int i = 0; i < available; ++i) {
        const aclblasStatus_t fillStatus = FillHeuristic(ids[i], capacities[i], results[i]);
        if (fillStatus != ACLBLAS_STATUS_SUCCESS) {
            return fillStatus;
        }
    }
    *count = available;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LtMatmul22Launch(const MatmulProblem& problem)
{
    LtMatmul22Tiling t = ProblemTiling(problem);
    if (t.m == 0 || t.n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    uint32_t id = 0;
    const aclblasStatus_t workspaceStatus =
        ValidateWorkspace(problem.algo, t, problem.workspace, problem.workspaceSize, id);
    if (workspaceStatus != ACLBLAS_STATUS_SUCCESS) {
        return workspaceStatus;
    }
    const aclblasStatus_t planStatus = InitializeLaunchPlan(t, id);
    if (planStatus != ACLBLAS_STATUS_SUCCESS)
        return planStatus;
    const bool product = t.alpha != 0.0f && t.k != 0;
    if (!product && t.beta == 1.0f && problem.C == problem.D) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // CANN 9.1 validates the stream header with an eight-byte atomic load.
    // Reject malformed representations before that load, as the newer runtime
    // handle guard does. This does not validate arbitrary aligned/stale handles.
    if (problem.stream != nullptr && (reinterpret_cast<uintptr_t>(problem.stream) & 7U) != 0U) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const bool ordered = RequiresOrdered(t, id, product);
    Hardware hardware;
    if (!QueryHardware(product && t.nCube != 0 && !ordered, t.cubeTileN, hardware, id)) {
        return ACLBLAS_STATUS_ARCH_MISMATCH;
    }
    if ((id == LT_MATMUL22_ALGO_H1 || id == LT_MATMUL22_ALGO_H9) && product && !ordered) {
        SelectSmallTile(t, hardware.aic);
    }
    const uint32_t vectors = std::min(hardware.aiv, t.m);
    if (!product)
        return LaunchScale(problem, t, id, vectors);
    if (ordered) {
        const uint64_t jobs = static_cast<uint64_t>(t.m) * CeilDivU64(t.n, 64U);
        const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(hardware.aiv, jobs));
        LtMatmul22Ordered(
            blocks, problem.stream, Bytes(problem.A), Bytes(problem.B), Bytes(problem.C), Bytes(problem.D), t);
        return LastLaunchStatus();
    }
    return LaunchProduct(problem, t, id, hardware, vectors);
}
