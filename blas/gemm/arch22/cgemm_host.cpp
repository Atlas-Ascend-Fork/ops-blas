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
 * \file cgemm_host.cpp
 * \brief arch22 (Atlas A2/A3) host implementation of aclblasCgemm.
 *
 *        C = alpha * op(A) * op(B) + beta * C   (column-major complex64)
 *
 *        Three-stage pipeline (see cgemm_tiling_data.h for the layout contract):
 *          Phase 0 (AIV) split   : A -> (Ar, Ai), B -> (Br, Bi), each a packed
 *                                  real column-major plane
 *          Phase 1 (AIC) gemm    : four real GEMMs issued through the arch22 real
 *                                  aclblasSgemm entry on a sub-handle whose
 *                                  workspace is a scratch slice
 *          Phase 2 (AIV) combine : C = alpha * (re + i*im) + beta * C_old
 *
 *        With A = Ar + i*Ai, B = Br + i*Bi and op(X) = conj(X) for trans = 'C',
 *        writing s_a / s_b = -1 for the conjugated operand and +1 otherwise:
 *          Re(op(A)*op(B)) = Ar*Br - (s_a*s_b)*Ai*Bi
 *          Im(op(A)*op(B)) = s_b*Ar*Bi + s_a*Ai*Br
 *        so the conjugation is carried entirely by the two scalar signs of the
 *          Phase 1 launches, and Phase 0 never has to negate the imaginary plane.
 *          Doing it this way rather than folding the sign into the split keeps one
 *          sign pattern for all four trans combinations.
 *
 *        Reusing aclblasSgemm instead of adding a second Cube kernel is what keeps
 *        this file small: the real GEMM already carries the arch22 tiling, K
 *        splitting and dispatch, and the sub-handle trick below confines it to a
 *        scratch slice so it cannot reclaim the packed planes.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "common/helper/host_utils.h"
#include "cgemm_kernel.h"
#include "cgemm_tiling_data.h"

namespace {
constexpr const char* OP_TAG = "aclblasCgemm";

// Packed planes are row-aligned to 64B so every row start is a whole burst.
constexpr uint32_t CGEMM_PLANE_ALIGN = 16U;
// Extra headroom handed to the real GEMM on top of the modelled requirement, in
// floats. aclblasSgemm picks its own tiling at runtime, so the exact figure is
// not knowable here; the margin keeps a slightly larger internal choice from
// turning into a spurious ALLOC_FAILED.
constexpr uint64_t CGEMM_SCRATCH_MARGIN = 4ULL * 1024ULL * 1024ULL;
// Upper bound on the reduction length handed to one delegated real GEMM. The
// arch22 real entry mis-computes K > 1024 when M and N are both below the
// whole-grid threshold (ops-blas#1077, measured on 910B3: 512x512x1536 returns
// 262123/262144 wrong elements, maxAbsErr 273). Chunking the reduction keeps every
// delegated call inside the verified-correct interval; the partial products are
// accumulated into the same real/imaginary plane, which is exact fp32 addition and
// so no less accurate than a single long reduction. Remove this once #1077 lands.
constexpr uint32_t CGEMM_K_CHUNK = 1024U;

inline uint64_t AlignPlane(uint64_t elements)
{
    return (elements + CGEMM_PLANE_ALIGN - 1U) / CGEMM_PLANE_ALIGN * CGEMM_PLANE_ALIGN;
}

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0F && v.imag == 0.0F; }

inline bool IsComplexOne(const aclblasComplex& v) { return v.real == 1.0F && v.imag == 0.0F; }

// Physical (stored) shape of an operand plus the row stride of its packed plane.
// op(X) is logicalRows x logicalCols; a transposed flag stores it as the
// transpose, so rows and cols swap.
struct CgemmPlaneShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t packedLd;
};

inline CgemmPlaneShape ResolvePlane(bool transposed, uint32_t logicalRows, uint32_t logicalCols)
{
    CgemmPlaneShape p{};
    p.rows = transposed ? logicalCols : logicalRows;
    p.cols = transposed ? logicalRows : logicalCols;
    p.packedLd = static_cast<uint32_t>(AlignPlane(p.rows));
    return p;
}

inline uint64_t PlaneFloats(const CgemmPlaneShape& p) { return static_cast<uint64_t>(p.packedLd) * p.cols; }

// The conjugation signs of the four real products. With s_a / s_b = -1 for a
// conjugated operand and +1 otherwise,
//   Re(op(A)*op(B)) = Ar*Br - (s_a*s_b)*Ai*Bi
//   Im(op(A)*op(B)) = s_b*Ar*Bi + s_a*Ai*Br
// so the caller hands these three scalars straight to the launch table.
struct CgemmSigns {
    float aiBi; // multiplies Ai*Bi inside the real product
    float arBi; // multiplies Ar*Bi inside the imaginary product
    float aiBr; // multiplies Ai*Br inside the imaginary product
};

CgemmSigns ResolveCgemmSigns(aclblasOperation_t transa, aclblasOperation_t transb)
{
    const float signA = (transa == ACLBLAS_OP_C) ? -1.0F : 1.0F;
    const float signB = (transb == ACLBLAS_OP_C) ? -1.0F : 1.0F;
    return CgemmSigns{-signA * signB, signB, signA};
}

// ============================================================================
//  Parameter validation (same checks and order as the arch35 implementation)
// ============================================================================
// Trans-dependent leading dimensions. op(A) is m x k and op(B) is k x n, so the
// stored row count of each operand follows its own transpose flag.
static aclblasStatus_t ValidateCgemmLeadingDims(
    aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb, int ldc)
{
    const int ldaMin = (transa != ACLBLAS_OP_N) ? std::max(1, k) : std::max(1, m);
    if (lda < ldaMin) {
        OP_LOGE(OP_TAG, "lda=%d must be >= %d", lda, ldaMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int ldbMin = (transb != ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (ldb < ldbMin) {
        OP_LOGE(OP_TAG, "ldb=%d must be >= %d", ldb, ldbMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE(OP_TAG, "ldc=%d must be >= %d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateCgemmParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, int lda, int ldb, const aclblasComplex* beta, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) {
        OP_LOGE(OP_TAG, "invalid transa=%d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C) {
        OP_LOGE(OP_TAG, "invalid transb=%d", static_cast<int>(transb));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0) {
        OP_LOGE(OP_TAG, "m, n, k must be non-negative: m=%d n=%d k=%d", m, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const aclblasStatus_t ldRet = ValidateCgemmLeadingDims(transa, transb, m, n, k, lda, ldb, ldc);
    if (ldRet != ACLBLAS_STATUS_SUCCESS) {
        return ldRet;
    }
    if (alpha == nullptr) {
        OP_LOGE(OP_TAG, "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE(OP_TAG, "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateCgemmPointers(
    int k, const aclblasComplex* A, const aclblasComplex* B, const aclblasComplex* C)
{
    if (C == nullptr) {
        OP_LOGE(OP_TAG, "C must not be nullptr for a non-empty output");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k > 0 && (A == nullptr || B == nullptr)) {
        OP_LOGE(OP_TAG, "A and B must not be nullptr when k > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Workspace planning
// ============================================================================
// Layout, all in floats: Ar, Ai, Br, Bi, re, im, then the scratch slice handed to
// the real GEMM. Every size is accumulated through the checked helpers so an
// extreme dimension is reported as an unsatisfiable allocation instead of
// wrapping to a small request.
struct CgemmWorkspace {
    uint8_t* ar = nullptr;
    uint8_t* ai = nullptr;
    uint8_t* br = nullptr;
    uint8_t* bi = nullptr;
    uint8_t* re = nullptr;
    uint8_t* im = nullptr;
    uint8_t* prodRe = nullptr;
    uint8_t* prodIm = nullptr;
    uint8_t* scratch = nullptr;
    uint64_t planeRFloats = 0U;
    uint64_t scratchFloats = 0U;
    uint32_t ldAr = 0U;
    uint32_t ldBr = 0U;
    uint32_t ldRe = 0U;
};

// Byte requirement of the whole layout, in floats, with every step checked.
// `scratch` receives the slice handed to the real GEMM. Returns 0 on overflow so
// the caller can report an unsatisfiable allocation instead of a request that
// wrapped to something small.
uint64_t CgemmTotalFloats(uint64_t planeA, uint64_t planeB, uint64_t planeR, uint64_t& scratch)
{
    uint64_t twoA = 0U;
    uint64_t twoB = 0U;
    uint64_t fourR = 0U;
    uint64_t persistent = 0U;
    if (!CBlas3TryMulU64(planeA, 2U, &twoA) || !CBlas3TryMulU64(planeB, 2U, &twoB) ||
        !CBlas3TryMulU64(planeR, 4U, &fourR) || !CBlas3TryAddU64(twoA, twoB, &persistent) ||
        !CBlas3TryAddU64(persistent, fourR, &persistent)) {
        return 0U;
    }
    uint64_t inner = 0U;
    uint64_t total = 0U;
    scratch = 0U;
    if (!CBlas3TryAddU64(planeA, planeB, &inner) || !CBlas3TryAddU64(inner, fourR, &inner) ||
        !CBlas3TryMulU64(inner, 2U, &scratch) || !CBlas3TryAddU64(scratch, CGEMM_SCRATCH_MARGIN, &scratch) ||
        !CBlas3TryAddU64(persistent, scratch, &total)) {
        return 0U;
    }
    return total;
}

aclblasStatus_t PlanCgemmWorkspace(
    _aclblas_handle* h, const CgemmPlaneShape& pa, const CgemmPlaneShape& pb, uint32_t m, uint32_t n,
    CgemmWorkspace& ws)
{
    ws.ldAr = pa.packedLd;
    ws.ldBr = pb.packedLd;
    ws.ldRe = static_cast<uint32_t>(AlignPlane(m));

    const uint64_t planeA = PlaneFloats(pa);
    const uint64_t planeB = PlaneFloats(pb);
    const uint64_t planeR = static_cast<uint64_t>(ws.ldRe) * n;
    uint64_t scratchFloats = 0U;
    const uint64_t totalFloats = CgemmTotalFloats(planeA, planeB, planeR, scratchFloats);
    if (totalFloats == 0U || totalFloats > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        OP_LOGE(OP_TAG, "workspace size overflows 64 bits: m=%u n=%u", m, n);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t ret = EnsureDefaultWorkspace(h, static_cast<size_t>(totalFloats) * sizeof(float));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "workspace ensure failed, required=%llu floats", static_cast<unsigned long long>(totalFloats));
        return ret;
    }

    uint8_t* base = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (base == nullptr) {
        OP_LOGE(OP_TAG, "GetEffectiveWorkspace returned nullptr");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    auto advance = [&base](uint64_t floats) {
        uint8_t* cur = base;
        base += floats * sizeof(float);
        return cur;
    };
    ws.ar = advance(planeA);
    ws.ai = advance(planeA);
    ws.br = advance(planeB);
    ws.bi = advance(planeB);
    ws.re = advance(planeR);
    ws.im = advance(planeR);
    ws.prodRe = advance(planeR);
    ws.prodIm = advance(planeR);
    ws.planeRFloats = planeR;
    ws.scratch = base;
    ws.scratchFloats = scratchFloats;
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Phase 0: complex -> packed real planes
// ============================================================================
static void RunCgemmSplits(
    void* stream, const CgemmPlaneShape& pa, const CgemmPlaneShape& pb, uint32_t lda, uint32_t ldb,
    const aclblasComplex* A, const aclblasComplex* B, const CgemmWorkspace& ws, uint32_t aivCoreNum)
{
    CBlas3SplitTilingData split{};
    split.rows = pa.rows;
    split.cols = pa.cols;
    split.lda = lda;
    split.packedLd = pa.packedLd;
    const uint32_t blocks = std::min(aivCoreNum, std::max(1U, pa.cols));
    cgemm_split_kernel_do(
        blocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), ws.ar, ws.ai, split);

    split.rows = pb.rows;
    split.cols = pb.cols;
    split.lda = ldb;
    split.packedLd = pb.packedLd;
    const uint32_t blocksB = std::min(aivCoreNum, std::max(1U, pb.cols));
    cgemm_split_kernel_do(
        blocksB, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(B)), ws.br, ws.bi, split);
}

// ============================================================================
//  Phase 1: four real GEMMs
// ============================================================================
// The real GEMM is driven through a sub-handle whose workspace is the scratch
// slice: aclblasSgemm plans and grows its own workspace through
// EnsureDefaultWorkspace, and confining it to a user-owned slice stops it from
// reclaiming the packed planes that are still live.
static void AccumulateRealPlane(
    void* stream, uint8_t* dst, uint8_t* src, uint64_t totalFloats, float scale, bool overwrite, uint32_t aivCoreNum)
{
    constexpr uint64_t CGEMM_ACCUM_CHUNK = 1024ULL;
    const uint64_t chunks = (totalFloats + CGEMM_ACCUM_CHUNK - 1ULL) / CGEMM_ACCUM_CHUNK;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum, std::max<uint64_t>(1ULL, chunks)));
    CgemmAccumTilingData tiling{totalFloats, scale, overwrite ? 1U : 0U};
    cgemm_accum_kernel_do(blocks, stream, dst, src, tiling);
}

struct CgemmRealProduct {
    float alpha;
    const uint8_t* a;
    const uint8_t* b;
    bool firstWriter;
    uint8_t* c;
    uint8_t* prod;
};

static aclblasStatus_t RunOneCgemmProduct(
    _aclblas_handle* sub, aclblasOperation_t ta, aclblasOperation_t tb, uint32_t m, uint32_t n, uint32_t kBase,
    uint32_t kCount, uint64_t aOffset, uint64_t bOffset, const CgemmWorkspace& ws, const CgemmRealProduct& product,
    uint32_t aivCoreNum)
{
    const float one = 1.0F;
    const float zero = 0.0F;
    const bool overwrite = (kBase == 0U && product.firstWriter);
    const bool direct = overwrite && (product.alpha == 1.0F);
    uint8_t* gemmOut = direct ? product.c : product.prod;
    const float* a = reinterpret_cast<const float*>(product.a) + aOffset;
    const float* b = reinterpret_cast<const float*>(product.b) + bOffset;
    const aclblasStatus_t ret = aclblasSgemmEx(
        reinterpret_cast<aclblasHandle_t>(sub), ta, tb, static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(kCount), &one, a, ACL_FLOAT, static_cast<int>(ws.ldAr), b, ACL_FLOAT,
        static_cast<int>(ws.ldBr), &zero, reinterpret_cast<float*>(gemmOut), ACL_FLOAT,
        static_cast<int>(ws.ldRe));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "real GEMM product failed at k=%u, ret=%d", kBase, static_cast<int>(ret));
        return ret;
    }
    if (!direct) {
        AccumulateRealPlane(
            sub->stream, product.c, product.prod, ws.planeRFloats, product.alpha, overwrite, aivCoreNum);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t RunCgemmRealProducts(
    _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, uint32_t m, uint32_t n, uint32_t k,
    const CgemmWorkspace& ws, const CgemmSigns& signs, uint32_t aivCoreNum)
{
    _aclblas_handle sub{};
    sub.stream = h->stream;
    sub.workspace = ws.scratch;
    sub.workspace_size = static_cast<size_t>(ws.scratchFloats) * sizeof(float);
    sub.workspace_owner = AclblasWorkspaceOwner::User;

    const aclblasOperation_t ta = (transa == ACLBLAS_OP_C) ? ACLBLAS_OP_T : transa;
    const aclblasOperation_t tb = (transb == ACLBLAS_OP_C) ? ACLBLAS_OP_T : transb;
    const CgemmRealProduct products[4] = {
        {1.0F, ws.ar, ws.br, true, ws.re, ws.prodRe},        // re = Ar*Br
        {signs.aiBi, ws.ai, ws.bi, false, ws.re, ws.prodRe}, // re += -(s_a*s_b) * Ai*Bi
        {signs.arBi, ws.ar, ws.bi, true, ws.im, ws.prodIm},  // im = s_b * Ar*Bi
        {signs.aiBr, ws.ai, ws.br, false, ws.im, ws.prodIm}, // im += s_a * Ai*Br
    };

    // SplitBody writes each packed plane column-major (col * packedLd + row).
    // For A: ta==N stores (m x k) so K advances along columns (stride ldAr);
    // ta==T stores (k x m) so K advances along rows (stride 1).
    // For B: tb==N stores (k x n) so K advances along rows (stride 1);
    // tb==T stores (n x k) so K advances along columns (stride ldBr).
    const uint64_t aKStride = (ta == ACLBLAS_OP_N) ? static_cast<uint64_t>(ws.ldAr) : 1ULL;
    const uint64_t bKStride = (tb == ACLBLAS_OP_N) ? 1ULL : static_cast<uint64_t>(ws.ldBr);
    // Avoid aclblasSgemm's RowsLeftTransMatmul path (m%16==0 && n%16==0 && k%8==0 && transB==T),
    // where BindRowsTransposedLeft passes unaligned (bGm + rowBase) offsets across AIC cores.
    const bool avoidLeftTransDirect = (tb != ACLBLAS_OP_N) && (m % 16U == 0U) && (n % 16U == 0U);

    for (uint32_t kBase = 0U; kBase < k;) {
        uint32_t kCount = std::min(CGEMM_K_CHUNK, k - kBase);
        if (avoidLeftTransDirect && (kCount % 8U == 0U)) {
            kCount = (kCount > 4U) ? (kCount - 4U) : 1U;
        }
        const uint64_t aOffset = static_cast<uint64_t>(kBase) * aKStride;
        const uint64_t bOffset = static_cast<uint64_t>(kBase) * bKStride;
        for (const CgemmRealProduct& product : products) {
            const aclblasStatus_t ret =
                RunOneCgemmProduct(&sub, ta, tb, m, n, kBase, kCount, aOffset, bOffset, ws, product, aivCoreNum);
            if (ret != ACLBLAS_STATUS_SUCCESS) {
                return ret;
            }
        }
        kBase += kCount;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Phase 2: complex epilogue
// ============================================================================
static void RunCgemmCombine(
    void* stream, uint32_t m, uint32_t n, uint32_t ldc, aclblasComplex* C, const CgemmWorkspace& ws, bool hasTemp,
    const aclblasComplex& alphaVal, const aclblasComplex& betaVal, uint32_t aivCoreNum)
{
    CgemmCombineTilingData combine{};
    combine.m = m;
    combine.n = n;
    combine.ldc = ldc;
    combine.ldRe = hasTemp ? ws.ldRe : ldc;
    combine.ldIm = hasTemp ? ws.ldRe : ldc;
    combine.hasTemp = hasTemp ? 1U : 0U;
    combine.alphaRe = alphaVal.real;
    combine.alphaIm = alphaVal.imag;
    combine.betaRe = betaVal.real;
    combine.betaIm = betaVal.imag;
    const uint32_t blocks = std::min(aivCoreNum, std::max(1U, n));
    OP_LOGI(OP_TAG, "phase2 combine: m=%u n=%u aivBlocks=%u hasTemp=%d", m, n, blocks, static_cast<int>(hasTemp));
    uint8_t* const reGm = hasTemp ? ws.re : reinterpret_cast<uint8_t*>(C);
    uint8_t* const imGm = hasTemp ? ws.im : reinterpret_cast<uint8_t*>(C);
    cgemm_combine_kernel_do(blocks, stream, reGm, imGm, reinterpret_cast<uint8_t*>(C), combine);
}

// Phase 0 + Phase 1: de-interleave both operands, then issue the four real
// products through aclblasSgemm on a scratch sub-handle.
static aclblasStatus_t RunCgemmProducts(
    _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, const aclblasComplex* A, const aclblasComplex* B, const CgemmWorkspace& ws,
    uint32_t aivCoreNum)
{
    const CgemmPlaneShape pa = ResolvePlane(transa != ACLBLAS_OP_N, m, k);
    const CgemmPlaneShape pb = ResolvePlane(transb != ACLBLAS_OP_N, k, n);
    RunCgemmSplits(h->stream, pa, pb, lda, ldb, A, B, ws, aivCoreNum);
    return RunCgemmRealProducts(h, transa, transb, m, n, k, ws, ResolveCgemmSigns(transa, transb), aivCoreNum);
}

static aclblasStatus_t LaunchCgemmKernel(
    _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    void* stream = h->stream;
    aclblasComplex alphaVal{0.0F, 0.0F};
    aclblasComplex betaVal{0.0F, 0.0F};
    const aclblasStatus_t readRet = CBlas3ReadScalarsFromDevice(alpha, beta, alphaVal, betaVal, stream, OP_TAG);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }
    OP_LOGD(
        OP_TAG, "alpha=(%f,%f) beta=(%f,%f)", static_cast<double>(alphaVal.real), static_cast<double>(alphaVal.imag),
        static_cast<double>(betaVal.real), static_cast<double>(betaVal.imag));

    const bool hasTemp = !IsComplexZero(alphaVal) && (k != 0);
    // Nothing contributes from A and B and C keeps its value: no kernel at all.
    if (!hasTemp && IsComplexOne(betaVal)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint32_t mU = static_cast<uint32_t>(m);
    const uint32_t nU = static_cast<uint32_t>(n);
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    CgemmWorkspace ws{};
    if (hasTemp) {
        const CgemmPlaneShape pa = ResolvePlane(transa != ACLBLAS_OP_N, mU, static_cast<uint32_t>(k));
        const CgemmPlaneShape pb = ResolvePlane(transb != ACLBLAS_OP_N, static_cast<uint32_t>(k), nU);
        const aclblasStatus_t wsRet = PlanCgemmWorkspace(h, pa, pb, mU, nU, ws);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            return wsRet;
        }

        const aclblasStatus_t gemmRet = RunCgemmProducts(
            h, transa, transb, mU, nU, static_cast<uint32_t>(k), static_cast<uint32_t>(lda), static_cast<uint32_t>(ldb),
            A, B, ws, aivCoreNum);
        if (gemmRet != ACLBLAS_STATUS_SUCCESS) {
            return gemmRet;
        }
    }

    RunCgemmCombine(stream, mU, nU, static_cast<uint32_t>(ldc), C, ws, hasTemp, alphaVal, betaVal, aivCoreNum);
    const aclError err = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    if (err != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "kernel launch failed, err=%d", static_cast<int>(err));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

// ============================================================================
//  Public API entry
// ============================================================================
extern "C" aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    aclblasStatus_t st = ValidateCgemmParams(handle, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    st = ValidateCgemmPointers(k, A, B, C);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return LaunchCgemmKernel(
        static_cast<_aclblas_handle*>(handle), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
