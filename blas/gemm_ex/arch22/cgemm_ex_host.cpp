/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log/log.h"
#include "gemm_ex_host_helpers.h"
#include "../gemm_ex_validation.h"
#include "cgemm_ex_kernel.h"
#include "gemm_ex_tiling.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include <limits>

namespace {
struct PlanarSizes {
    uint64_t na = 0;
    uint64_t nb = 0;
    uint64_t nc = 0;
    uint64_t scratch = 0;
    uint64_t total = 0;
};

aclblasStatus_t GetPlanarSizes(
    aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, int lda, int ldb, PlanarSizes& sizes)
{
    const auto align = [](uint64_t v) { return (v + 15) / 16 * 16; };
    const uint64_t na = align(static_cast<uint64_t>(lda) * (ta == ACLBLAS_OP_N ? k : m));
    const uint64_t nb = align(static_cast<uint64_t>(ldb) * (tb == ACLBLAS_OP_N ? n : k));
    const uint64_t nc = align(static_cast<uint64_t>(m) * n);
    // Bound each operand before combining sizes; int-sized dimensions can
    // otherwise overflow the multiplied sum even though each product fits.
    constexpr uint64_t maxFloats = ACLBLAS_MAX_WORKSPACE_SIZE / sizeof(float);
    if (na > maxFloats || nb > maxFloats || nc > maxFloats) {
        OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: planar operands exceed workspace limit");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    // Reserve backend scratch separately from the persistent planar operands/products.
    const uint64_t scratch = 2 * (na + nb + 2 * nc) + 4ULL * 1024 * 1024;
    const uint64_t total = 2 * (na + nb + nc) + scratch;
    if (total > ACLBLAS_MAX_WORKSPACE_SIZE / sizeof(float)) {
        OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: planar workspace exceeds the allocation limit");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    sizes = PlanarSizes{na, nb, nc, scratch, total};
    return ACLBLAS_STATUS_SUCCESS;
}

struct PlanarProduct {
    float alpha;
    const float* a;
    const float* b;
    float beta;
    float* output;
};

aclblasStatus_t ComputePlanarProducts(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, int lda, int ldb,
    const PlanarProduct (&products)[4])
{
    for (const auto& product : products) {
        auto status = GemmExPlanarBackend(
            handle, ta, tb, m, n, k, &product.alpha, product.a, lda, product.b, ldb, &product.beta, product.output, m);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t PlanarGemm(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k,
    const aclblasComplex* alpha, const void* a, int lda, const void* b, int ldb, const aclblasComplex* beta, void* c,
    int ldc, uint32_t cores)
{
    CgemmExLayout t{m, n, ldc, alpha->real, alpha->imag, beta->real, beta->imag, 0, 0};
    PlanarSizes sizes;
    auto status = GetPlanarSizes(ta, tb, m, n, k, lda, ldb, sizes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const uint64_t na = sizes.na;
    const uint64_t nb = sizes.nb;
    const uint64_t nc = sizes.nc;
    const uint64_t scratch = sizes.scratch;
    const uint64_t total = sizes.total;
    status = EnsureDefaultWorkspace(handle, total * sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    auto* ar = static_cast<float*>(GetEffectiveWorkspace(handle));
    auto* ai = ar + na;
    auto* br = ai + na;
    auto* bi = br + nb;
    auto* re = bi + nb;
    auto* im = re + nc;
    cgemm_ex_split_do(
        cores, handle->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(a)), reinterpret_cast<uint8_t*>(ar),
        reinterpret_cast<uint8_t*>(ai), static_cast<uint64_t>(lda) * (ta == ACLBLAS_OP_N ? k : m),
        ta == ACLBLAS_OP_C ? -1.0f : 1.0f);
    cgemm_ex_split_do(
        cores, handle->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(b)), reinterpret_cast<uint8_t*>(br),
        reinterpret_cast<uint8_t*>(bi), static_cast<uint64_t>(ldb) * (tb == ACLBLAS_OP_N ? n : k),
        tb == ACLBLAS_OP_C ? -1.0f : 1.0f);
    _aclblas_handle sub;
    sub.stream = handle->stream;
    sub.workspace = im + nc;
    sub.workspace_size = scratch * sizeof(float);
    sub.workspace_owner = AclblasWorkspaceOwner::User;
    const PlanarProduct products[4] = {
        {1.0f, ar, br, 0.0f, re}, {-1.0f, ai, bi, 1.0f, re}, {1.0f, ar, bi, 0.0f, im}, {1.0f, ai, br, 1.0f, im}};
    status = ComputePlanarProducts(&sub, ta, tb, m, n, k, lda, ldb, products);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    cgemm_ex_combine_do(
        cores, handle->stream, reinterpret_cast<uint8_t*>(re), reinterpret_cast<uint8_t*>(im),
        reinterpret_cast<uint8_t*>(c), t);
    return ACLBLAS_STATUS_SUCCESS;
}
aclblasStatus_t RawTranspose114(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k,
    const aclblasComplex* alpha, const void* a, int lda, const void* b, int ldb, const aclblasComplex* beta, void* c,
    int ldc, uint32_t vectors)
{
    const int mp = (m + 15) / 16 * 16;
    const int np = (n + 15) / 16 * 16;
    const int kp = (k + 63) / 64 * 64;
    const uint64_t nb = 4ULL * np * kp;
    const uint64_t nc = 2ULL * mp * np;
    auto status = EnsureDefaultWorkspace(handle, (nb + nc) * sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    auto* bx = static_cast<float*>(GetEffectiveWorkspace(handle));
    auto* product = bx + nb;
    CgemmExpandedLayout t{m, n, k, mp, np, kp, lda, ldb, ta == ACLBLAS_OP_T ? 1 : 2, tb == ACLBLAS_OP_T ? 1 : 2, 0};
    cgemm_ex_rawb114_do(
        vectors, handle->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(b)), reinterpret_cast<uint8_t*>(bx), t);
    GemmExTiling native{};
    native.m = m;
    native.n = 2 * np;
    native.k = 2 * k;
    native.lda = 2 * lda;
    native.ldb = 2 * np;
    native.ldc = mp;
    native.transA = 1;
    native.transB = 1;
    native.alpha = 1.0f;
    native.beta = 0.0f;
    const uint32_t cubes = GetAicCoreCount();
    if (!cubes) {
        OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    cgemm_ex_raw114_do(
        cubes, handle->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(a)), reinterpret_cast<uint8_t*>(bx),
        reinterpret_cast<uint8_t*>(product), native);
    CgemmExLayout finish{m, n, ldc, alpha->real, alpha->imag, beta->real, beta->imag, 3U, mp};
    cgemm_ex_combine_do(
        vectors, handle->stream, reinterpret_cast<uint8_t*>(product), nullptr, reinterpret_cast<uint8_t*>(c), finish);
    return ACLBLAS_STATUS_SUCCESS;
}
aclblasStatus_t CrossNn115(
    aclblasHandle_t handle, int m, int n, int k, const void* a, int lda, const void* b, int ldb, void* c, int ldc,
    uint32_t vectors)
{
    const int mp = (m + 15) / 16 * 16;
    const int np = (n + 15) / 16 * 16;
    const int kp = (k + 63) / 64 * 64;
    const uint64_t nb = 2ULL * np * kp;
    const uint64_t nc = 4ULL * mp * np;
    auto status = EnsureDefaultWorkspace(handle, (nb + nc) * sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    auto* bx = static_cast<float*>(GetEffectiveWorkspace(handle));
    auto* product = bx + nb;
    const uint32_t cubes = GetAicCoreCount();
    if (!cubes) {
        OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    CgemmExpandedLayout t{m, n, k, mp, np, kp, lda, ldb, 0, 0, 0};
    CgemmExLayout finish{m, n, ldc, 1.0f, 0.0f, 0.0f, 0.0f, 0, mp};
    cgemm_ex_cross115_do(
        cubes, vectors, handle->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(a)),
        reinterpret_cast<uint8_t*>(const_cast<void*>(b)), reinterpret_cast<uint8_t*>(bx),
        reinterpret_cast<uint8_t*>(product), reinterpret_cast<uint8_t*>(c), t, finish);
    return ACLBLAS_STATUS_SUCCESS;
}
struct ComplexGemmParams {
    aclblasHandle_t handle;
    aclblasOperation_t ta;
    aclblasOperation_t tb;
    int m;
    int n;
    int k;
    const aclblasComplex* alpha;
    const void* a;
    int lda;
    const void* b;
    int ldb;
    const aclblasComplex* beta;
    void* c;
    int ldc;
    uint32_t cores;
};

class ExpandedGemmRun : private ComplexGemmParams {
public:
    explicit ExpandedGemmRun(const ComplexGemmParams& params) : ComplexGemmParams(params) {}
    aclblasStatus_t Run()
    {
        constexpr int maxExpandedDim = (std::numeric_limits<int>::max() - 63) / 2;
        if (m > maxExpandedDim || n > maxExpandedDim || k > maxExpandedDim) {
            return RunPlanar();
        }
        if (UseCrossNn()) {
            return CrossNn115(handle, m, n, k, a, lda, b, ldb, c, ldc, cores);
        }
        if (UseRawTranspose()) {
            return RawTranspose114(handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cores);
        }
        ConfigureLayout();
        if (!PrepareSizes()) {
            return RunPlanar();
        }
        auto status = EnsureDefaultWorkspace(handle, total * sizeof(float));
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        AttachWorkspace();
        ExpandOperands();
        ConfigureNative();
        status = Dispatch();
        return Finish(status);
    }

private:
    int mp = 0;
    int np = 0;
    int kp = 0;
    int rows = 0;
    int cols = 0;
    bool canonicalB = false;
    bool rightTrans = false;
    bool compactDeep = false;
    bool directOutput = false;
    bool finished = false;
    uint64_t na = 0;
    uint64_t nb = 0;
    uint64_t nc = 0;
    uint64_t originalNa = 0;
    uint64_t originalNb = 0;
    uint64_t scratch = 0;
    uint64_t total = 0;
    float* ax = nullptr;
    float* bx = nullptr;
    float* out = nullptr;
    float* product = nullptr;
    _aclblas_handle sub;
    CgemmExpandedLayout layout{};
    GemmExTiling native{};

    bool UnitOutput() const
    {
        return alpha->real == 1.0f && alpha->imag == 0.0f && beta->real == 0.0f && beta->imag == 0.0f;
    }
    bool UseCrossNn() const
    {
        return ta == ACLBLAS_OP_N && tb == ACLBLAS_OP_N && m >= 512 && m <= 2048 && n <= 128 && k >= 256 && k <= 2048 &&
               UnitOutput();
    }
    bool UseRawTranspose() const
    {
        return ta != ACLBLAS_OP_N && tb != ACLBLAS_OP_N && n <= 32 && m >= 256 && m <= 4096 && k >= 128 && k <= 1024 &&
               lda <= std::numeric_limits<int>::max() / 2;
    }
    bool UseCompactDeep() const
    {
        return ta == ACLBLAS_OP_N && tb == ACLBLAS_OP_N && m == mp && n == np && kp >= 4096 && kp <= 8192 &&
               (2 * kp) % 128 == 0 && 2 * mp >= 512 && np >= 512 && 2 * mp <= 8192 && np <= 8192 && ldc == m &&
               UnitOutput();
    }
    bool UseShortNn() const
    {
        return ta == ACLBLAS_OP_N && tb == ACLBLAS_OP_N && k > 16 && k <= 32 && m >= 512 && m <= 1536 && n >= 128 &&
               n <= 1024;
    }
    bool UseNativeNarrow() const
    {
        return (cols <= 32 && rows >= 256 && rows <= 4096 && 2 * kp >= 256 && 2 * kp <= 4096) ||
               (rows <= 256 && cols >= 512 && cols <= 4096 && 2 * kp <= 1024) ||
               (rows > 512 && rows <= 640 && cols > 128 && cols <= 512 && 2 * kp <= 1024);
    }
    bool UseNativeWide() const
    {
        return (rows <= 128 && cols >= 512 && cols <= 4096 && 2 * kp <= 1024 && native.transA != 0) ||
               (rows >= 2048 && rows <= 4096 && cols <= 256 && 2 * kp <= 256) ||
               (rows >= 1024 && rows <= 4096 && cols >= 1024 && cols <= 4096 && 2 * kp <= 512);
    }
    aclblasStatus_t RunPlanar()
    {
        return PlanarGemm(handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cores);
    }

    bool UseOriginalCanonical() const
    {
        return tb != ACLBLAS_OP_N && k <= 128 && static_cast<uint64_t>(m) * n >= 1048576 &&
               (ta == ACLBLAS_OP_N ? mp : kp) <= 2048;
    }
    bool UseAddedCanonical() const
    {
        return tb != ACLBLAS_OP_N && k >= 48 && kp <= 512 && m <= n && static_cast<uint64_t>(m) * n >= 32768 &&
               (ta == ACLBLAS_OP_N ? mp : kp) <= 2048;
    }

    void ConfigureLayout()
    {
        mp = (m + 15) / 16 * 16;
        np = (n + 15) / 16 * 16;
        kp = (k + 15) / 16 * 16;
        // Match the real backend reduction alignment during expansion, avoiding
        // a second full operand repack for transposed left operands.
        if (ta != ACLBLAS_OP_N && 2 * kp > 96) {
            kp = (k + 63) / 64 * 64;
        }
        const bool originalCanonical = UseOriginalCanonical();
        const bool addedCanonical = UseAddedCanonical();
        canonicalB = originalCanonical || addedCanonical;
        if (addedCanonical) {
            kp = (kp + 63) / 64 * 64;
        }
        rightTrans = tb != ACLBLAS_OP_N && !canonicalB;
        // Interleaved real output has leading dimension 2*mp. Align short-K
        // output rows to 512B without adding any K terms or changing public C.
        if (canonicalB && k <= 128 && m >= 256 && m <= 2048 && n <= 2048) {
            mp = (m + 63) / 64 * 64;
        }
    }

    bool PrepareSizes()
    {
        na = (rightTrans ? 2ULL : 4ULL) * mp * kp;
        nb = (rightTrans ? 4ULL : 2ULL) * kp * np;
        nc = 2ULL * mp * np;
        constexpr uint64_t maxFloats = ACLBLAS_MAX_WORKSPACE_SIZE / sizeof(float);
        if (na > maxFloats || nb > maxFloats || nc > maxFloats) {
            return false;
        }
        compactDeep = UseCompactDeep();
        originalNa = na;
        originalNb = nb;
        if (compactDeep) {
            na *= 2;
            nb *= 2;
        }
        scratch = compactDeep ? 1 * nc : 2 * (na + nb + 2 * nc) + 4ULL * 1024 * 1024;
        total = na + nb + (compactDeep ? originalNa + originalNb : nc) + scratch;
        if (total > ACLBLAS_MAX_WORKSPACE_SIZE / sizeof(float)) {
            return false;
        }
        return true;
    }

    void AttachWorkspace()
    {
        ax = static_cast<float*>(GetEffectiveWorkspace(handle));
        bx = ax + na;
        out = bx + nb;
        layout = CgemmExpandedLayout{
            m,
            n,
            k,
            mp,
            np,
            kp,
            lda,
            ldb,
            ta == ACLBLAS_OP_N ? 0 : (ta == ACLBLAS_OP_T ? 1 : 2),
            tb == ACLBLAS_OP_N ? 0 : (tb == ACLBLAS_OP_T ? 1 : 2),
            canonicalB ? 1 : 0};
    }

    void ExpandOperands()
    {
        if (compactDeep) {
            cgemm_ex_expand_do(
                cores, handle->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(a)),
                reinterpret_cast<uint8_t*>(const_cast<void*>(b)), reinterpret_cast<uint8_t*>(out + scratch),
                reinterpret_cast<uint8_t*>(out + scratch + originalNa), layout);

            cgemm_pad305_do(
                cores, handle->stream, reinterpret_cast<uint8_t*>(out + scratch),
                reinterpret_cast<uint8_t*>(out + scratch + originalNa), reinterpret_cast<uint8_t*>(ax),
                reinterpret_cast<uint8_t*>(bx), 2 * mp, kp, np);
        } else {
            cgemm_ex_expand_do(
                cores, handle->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(a)),
                reinterpret_cast<uint8_t*>(const_cast<void*>(b)), reinterpret_cast<uint8_t*>(ax),
                reinterpret_cast<uint8_t*>(bx), layout);
        }
    }

    void ConfigureNative()
    {
        sub.stream = handle->stream;
        sub.workspace = out + (compactDeep ? 0 : nc);
        sub.workspace_size = scratch * sizeof(float);
        sub.workspace_owner = AclblasWorkspaceOwner::User;
        rows = rightTrans ? mp : 2 * mp;
        cols = rightTrans ? 2 * np : np;
        // With no padded output coordinates the real expansion is exactly the
        // public complex storage. A unit-alpha/zero-beta epilogue is a no-op.
        directOutput = !rightTrans && m == mp && n == np && ldc == m && UnitOutput();
        product = directOutput ? static_cast<float*>(c) : out;
        native.m = rows;
        native.n = cols;
        native.k = compactDeep ? 4 * kp : 2 * kp;
        native.lda = ta == ACLBLAS_OP_N ? rows : 2 * kp;
        native.ldb = rightTrans ? 2 * np : (compactDeep ? 4 * kp : 2 * kp);
        native.ldc = rows;
        native.transA = ta == ACLBLAS_OP_N ? 0 : 1;
        native.transB = rightTrans ? 1 : 0;
        native.alpha = 1.0f;
        native.beta = 0.0f;
        native.cubeBlocks = GetAicCoreCount();
    }

    aclblasStatus_t RunShortNn()
    {
        if (!native.cubeBlocks) {
            OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        cgemm_short347_do(
            native.cubeBlocks, handle->stream, reinterpret_cast<uint8_t*>(ax), reinterpret_cast<uint8_t*>(bx),
            reinterpret_cast<uint8_t*>(product), native);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunCompactDeep()
    {
        native.pm = rows;
        native.pn = cols;
        native.computeK = native.k;
        native.splitParts = 1;
        native.productPlane = static_cast<uint64_t>(rows) * cols;
        native.productOffset = 0;
        const uint64_t area = static_cast<uint64_t>(rows) * cols;
        const uint32_t tile = area < 65536 ? 64 : (area < 4194304 ? 128 : 256);
        const uint32_t rowTile = static_cast<uint64_t>(rows) * cols < 262144 ? 64 : tile;
        const uint32_t blocks = 1 * ((rows + tile - 1) / tile) * ((cols + rowTile - 1) / rowTile);
        cgemm_long301_do(
            blocks, handle->stream, reinterpret_cast<uint8_t*>(ax), reinterpret_cast<uint8_t*>(bx),
            reinterpret_cast<uint8_t*>(sub.workspace), native);
        gemmex_fallback_wholegrid_epilogue_do(
            GetAivCoreCount(), handle->stream, reinterpret_cast<uint8_t*>(sub.workspace),
            reinterpret_cast<uint8_t*>(product), native);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunSmallRows()
    {
        const uint64_t partialBytes = 8ULL * rows * cols * sizeof(float);
        if (partialBytes > sub.workspace_size || !native.cubeBlocks) {
            OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: %s failed with ACLBLAS_STATUS_ALLOC_FAILED", __func__);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (UnitOutput()) {
            CgemmExLayout finish{m, n, ldc, 1.0f, 0.0f, 0.0f, 0.0f, rightTrans ? 3U : 2U, rows};
            cgemm_ex_split107_finish_do(
                native.cubeBlocks, handle->stream, reinterpret_cast<uint8_t*>(ax), reinterpret_cast<uint8_t*>(bx),
                reinterpret_cast<uint8_t*>(sub.workspace), reinterpret_cast<uint8_t*>(c), native, finish);
            finished = true;
            return ACLBLAS_STATUS_SUCCESS;
        }
        cgemm_ex_split104_do(
            native.cubeBlocks, handle->stream, reinterpret_cast<uint8_t*>(ax), reinterpret_cast<uint8_t*>(bx),
            reinterpret_cast<uint8_t*>(sub.workspace), reinterpret_cast<uint8_t*>(product), native);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunNativeNarrow()
    {
        if (!native.cubeBlocks) {
            OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        // Preserve16-coordinate rounding for DMA efficiency, but omit a full
        // redundant16-coordinate tile when complex padding doubled it.
        if (!rightTrans) {
            native.m = (2 * m + 15) / 16 * 16;
        }
        // Only remove complete32-coordinate padding groups; keep safe K
        // alignment and both operands' physical512-coordinate strides.
        if (canonicalB && m <= 128) {
            native.k = (2 * k + 31) / 32 * 32;
        }
        cgemm_ex_native110_do(
            native.cubeBlocks, handle->stream, reinterpret_cast<uint8_t*>(ax), reinterpret_cast<uint8_t*>(bx),
            reinterpret_cast<uint8_t*>(product), native);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunNativeWide()
    {
        if (!native.cubeBlocks) {
            OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        gemmex_fallback_catlass_do(
            native.cubeBlocks, handle->stream, reinterpret_cast<uint8_t*>(ax), reinterpret_cast<uint8_t*>(bx),
            reinterpret_cast<uint8_t*>(product), native);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunFallback()
    {
        const float one = 1.0f;
        const float zero = 0.0f;
        return GemmExPlanarBackend(
            &sub, ta, canonicalB ? ACLBLAS_OP_N : tb, rows, cols, 2 * kp, &one, ax, ta == ACLBLAS_OP_N ? rows : 2 * kp,
            bx, rightTrans ? 2 * np : 2 * kp, &zero, product, rows);
    }

    aclblasStatus_t Finish(aclblasStatus_t status)
    {
        if (directOutput || finished) {
            return status;
        }
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        CgemmExLayout finish{m, n, ldc, alpha->real, alpha->imag, beta->real, beta->imag, rightTrans ? 3U : 2U, rows};
        cgemm_ex_combine_do(
            cores, handle->stream, reinterpret_cast<uint8_t*>(out), nullptr, reinterpret_cast<uint8_t*>(c), finish);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t Dispatch()
    {
        if (UseShortNn()) {
            return RunShortNn();
        }
        if (compactDeep && directOutput && static_cast<uint64_t>(rows) * cols * sizeof(float) <= sub.workspace_size) {
            return RunCompactDeep();
        }
        if (native.transA == 0 && rows <= 128 && cols >= 128 && cols <= 2048 && 2 * kp >= 1024 && 2 * kp <= 4096) {
            return RunSmallRows();
        }
        if (UseNativeNarrow()) {
            return RunNativeNarrow();
        }
        if (UseNativeWide()) {
            return RunNativeWide();
        }
        return RunFallback();
    }
};

aclblasStatus_t ExpandedGemm(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k,
    const aclblasComplex* alpha, const void* a, int lda, const void* b, int ldb, const aclblasComplex* beta, void* c,
    int ldc, uint32_t cores)
{
    const ComplexGemmParams params{handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cores};
    return ExpandedGemmRun(params).Run();
}

} // namespace

extern "C" aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k,
    const aclblasComplex* alpha, const void* a, aclDataType at, int lda, const void* b, aclDataType bt, int ldb,
    const aclblasComplex* beta, void* c, aclDataType ct, int ldc)
{
    auto status = gemm_ex::Validate(
        handle, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc, ACL_COMPLEX64, "aclblasCgemmEx");
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0 || !c) {
        return status;
    }
    const bool scale = k == 0 || gemm_ex::IsZero(*alpha);
    if (scale && beta->real == 1.0f && beta->imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        OP_LOGE("aclblasCgemmEx", "aclblasCgemmEx: no available AIV cores");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    CgemmExLayout t{m, n, ldc, alpha->real, alpha->imag, beta->real, beta->imag, scale ? 1U : 0U};
    auto* out = reinterpret_cast<uint8_t*>(c);
    if (scale) {
        cgemm_ex_combine_do(cores, handle->stream, nullptr, nullptr, out, t);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (gemm_ex::VolumeAtMost(m, n, k, 32768)) {
        CgemmExDirectLayout direct{
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            ta == ACLBLAS_OP_N ? 0 : (ta == ACLBLAS_OP_C ? 2 : 1),
            tb == ACLBLAS_OP_N ? 0 : (tb == ACLBLAS_OP_C ? 2 : 1),
            alpha->real,
            alpha->imag,
            beta->real,
            beta->imag};
        cgemm_ex_direct_do(
            cores, handle->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<void*>(b)), out, direct);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (static_cast<uint64_t>(m) * n <= 256) {
        return PlanarGemm(handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cores);
    }
    return ExpandedGemm(handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cores);
}
