/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gemm_ex_host_helpers.h"
#include "../gemm_ex_validation.h"
#include "sgemm_ex_direct_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
namespace {
bool UseDirectReal(int m, int n, int k, float alpha)
{
    // Amplifying scalars retain the compensated path: scaling a rounded Cube
    // product can exceed the FP32 golden error bound. Keep this numerical
    // fallback until an accelerated equivalent is verified.
    constexpr uint64_t COMPENSATED_WORK_LIMIT = 256ULL * 256 * 256;
    constexpr int DIRECT_REDUCTION_LIMIT = 4096;
    return alpha > 1.0f || alpha < -1.0f || gemm_ex::VolumeAtMost(m, n, k, 4096) ||
           (k <= DIRECT_REDUCTION_LIMIT && static_cast<uint64_t>(m) * n <= 256) ||
           (alpha != 1.0f && gemm_ex::VolumeAtMost(m, n, k, COMPENSATED_WORK_LIMIT));
}

aclblasStatus_t Compute(
    bool sgemm, aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k,
    const float* alpha, const float* a, int lda, const float* b, int ldb, const float* beta, float* c, int ldc)
{
    if (UseDirectReal(m, n, k, *alpha)) {
        const uint32_t cores = GetAivCoreCount();
        if (!cores) {
            const char* api = sgemm ? "aclblasSgemmEx" : "aclblasGemmEx";
            OP_LOGE(api, "%s: no available AIV cores", api);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        SgemmExDirectLayout t{m,      n,    k, lda, ldb, ldc, ta == ACLBLAS_OP_N ? 0 : 1, tb == ACLBLAS_OP_N ? 0 : 1,
                              *alpha, *beta};
        sgemm_ex_direct_do(
            cores, handle->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), reinterpret_cast<uint8_t*>(c), t);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (sgemm) {
        return SgemmExDispatchBackend(handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    return GemmExArch22Backend(handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
} // namespace

extern "C" aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const float* alpha,
    const void* a, aclDataType at, int lda, const void* b, aclDataType bt, int ldb, const float* beta, void* c,
    aclDataType ct, int ldc)
{
    const auto status = gemm_ex::Validate(
        handle, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc, ACL_FLOAT, "aclblasSgemmEx");
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0 || !c) {
        return status;
    }
    return Compute(
        true, handle, ta, tb, m, n, k, alpha, static_cast<const float*>(a), lda, static_cast<const float*>(b), ldb,
        beta, static_cast<float*>(c), ldc);
}

extern "C" aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const void* alpha,
    const void* a, aclDataType at, int lda, const void* b, aclDataType bt, int ldb, const void* beta, void* c,
    aclDataType ct, int ldc, aclblasComputeType_t compute, aclblasGemmAlgo_t algo)
{
    const auto* av = static_cast<const float*>(alpha);
    const auto* bv = static_cast<const float*>(beta);
    const auto status = gemm_ex::Validate(
        handle, ta, tb, m, n, k, av, a, at, lda, b, bt, ldb, bv, c, ct, ldc, ACL_FLOAT, "aclblasGemmEx");
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (compute != ACLBLAS_COMPUTE_32F || algo != ACLBLAS_GEMM_DEFAULT) {
        OP_LOGE(
            "aclblasGemmEx", "aclblasGemmEx: unsupported computeType=%d or algo=%d", static_cast<int>(compute),
            static_cast<int>(algo));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (m == 0 || n == 0 || !c) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return Compute(
        false, handle, ta, tb, m, n, k, av, static_cast<const float*>(a), lda, static_cast<const float*>(b), ldb, bv,
        static_cast<float*>(c), ldc);
}

extern "C" aclblasStatus_t GemmExArch22Backend(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const float* alpha,
    const float* a, int lda, const float* b, int ldb, const float* beta, float* c, int ldc)
{
    const gemm_ex::RealDispatchParams p{handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc};
    if (p.IsShortNn()) {
        const uint32_t cores = GetAicCoreCount();
        const uint32_t vectors = GetAivCoreCount();
        const uint32_t grid = p.ShortNnGrid();
        if (cores && vectors && grid <= cores && (static_cast<uint32_t>(n) + 15) / 16 <= vectors) {
            return gemm_ex::RunShortNn(p, cores, vectors, grid, gemmex_nn263_do, gemmex_nn263_output_do);
        }
    }
    return SgemmExDispatchBackend(handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
