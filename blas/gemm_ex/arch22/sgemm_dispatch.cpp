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
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
namespace {
aclblasStatus_t RunSmallTt(const gemm_ex::RealDispatchParams& p)
{
    const uint32_t cubes = GetAicCoreCount();
    const uint32_t vectors = GetAivCoreCount();
    if (!cubes || !vectors) {
        OP_LOGE("aclblasSgemmEx", "aclblasSgemmEx: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    GemmExTiling t = p.MakeTiling(cubes);
    GemmExTiling padded = t;
    padded.lda = (p.k + 127) / 128 * 128;
    padded.ldb = (p.n + 127) / 128 * 128;
    padded.ldc = (p.m + 127) / 128 * 128;
    const uint64_t ae = static_cast<uint64_t>(p.m) * padded.lda;
    const uint64_t be = static_cast<uint64_t>(p.k) * padded.ldb;
    const uint64_t ce = static_cast<uint64_t>(p.n) * padded.ldc;
    const auto status = EnsureDefaultWorkspace(p.handle, static_cast<size_t>(ae + be + ce) * sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    auto* pa = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(p.handle));
    auto* pb = pa + ae * sizeof(float);
    auto* product = pb + be * sizeof(float);
    sgemm_ex_arch22_catlass_pack_do(
        vectors, p.handle->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(p.a)),
        reinterpret_cast<uint8_t*>(const_cast<float*>(p.b)), pa, pb, t, padded);
    padded.ldc = p.ldc;
    SgemmExSmallTt259Do(cubes, p.handle->stream, pa, pb, reinterpret_cast<uint8_t*>(p.c), padded);
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace
extern "C" aclblasStatus_t SgemmExDispatchBackend(
    aclblasHandle_t h, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const float* alpha,
    const float* a, int lda, const float* b, int ldb, const float* beta, float* c, int ldc)
{
    const gemm_ex::RealDispatchParams p{h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc};
    if (p.IsShortNn()) {
        const uint32_t cores = GetAicCoreCount();
        const uint32_t vectors = GetAivCoreCount();
        const uint32_t grid = p.ShortNnGrid();
        if (cores && vectors && grid <= cores && (static_cast<uint32_t>(n) + 15) / 16 <= vectors) {
            return gemm_ex::RunShortNn(p, cores, vectors, grid, sgemmex_nn333_do, sgemmex_nn333_output_do);
        }
    }
    if (p.IsSmallTt()) {
        return RunSmallTt(p);
    }
    return SgemmExArch22Backend(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

extern "C" aclblasStatus_t SgemmExArch22Backend(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* a, int lda, const float* b, int ldb, const float* beta, float* c, int ldc)
{
    static const RealBackendKernels kernels{
        sgemm_ex_arch22_catlass_copy_do,  sgemm_ex_arch22_catlass_do,         sgemm_ex_arch22_catlass_pack_do,
        sgemm_ex_arch22_column_owner_do,  sgemm_ex_arch22_dma_pack_do,        sgemm_ex_arch22_moderate_nz_do,
        gemmex_fallback_singleload174_do, sgemm_ex_arch22_nz_fourrow_copy_do, sgemm_ex_arch22_nz_loop_copy_do,
        sgemm_ex_arch22_short_native_do,  gemmex_fallback_split232_do,        sgemm_ex_arch22_split_nt_do,
        sgemm_ex_arch22_split_sum_do,     gemmex_fallback_tt_stagger159_do};
    return RunGemmExRealBackend(kernels, handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
