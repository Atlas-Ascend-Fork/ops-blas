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
 * \file matmul_engine.cpp
 * \brief Matmul engine: MMAD dtype routing (FP32 / MXFP8 / MXFP4) followed by the alpha/beta epilogue.
 */

#include "matmul_engine.h"

#include "aclblaslt_algo_heuristic.h"
#include "host_utils.h"

#include "epilogue_alpha_beta_host.h"
#include "matmul_fp32_host.h"
#include "matmul_fp32_kernel.h"
#include "matmul_fp32_tiling_data.h"
#include "matmul_mxfp4_host.h"
#include "matmul_mxfp8_host.h"
#include "matmul_mxfp8_kernel.h"
#include "quant_matmul_tiling_data.h"

#include <cstdint>

namespace {

inline uint8_t* AsBytes(const void* p)
{
    return static_cast<uint8_t*>(const_cast<void*>(p));
}

// Step 1: route to the MMAD kernel matching the input dtype pair (FP32 / MXFP8 / MXFP4) and write the
// raw product into dRawAddr. decodedAlgo only steers the FP32 auto-tiling when hasDecodedAlgo is set.
aclblasStatus_t LaunchMmadKernel(
    const MatmulProblem& problem, void* dRawAddr, bool hasDecodedAlgo, const PackedAlgo& decodedAlgo)
{
    const uint64_t m = problem.m;
    const uint64_t n = problem.n;
    const uint64_t k = problem.k;
    const uint32_t numBlocks = problem.numBlocks;
    const aclrtStream stream = problem.stream;
    const aclDataType dtypeA = problem.dtypeA;
    const aclDataType dtypeB = problem.dtypeB;
    const aclDataType dtypeD = problem.dtypeD;
    const bool transA = problem.transA;
    const bool transB = problem.transB;

    if (dtypeA == ACL_FLOAT && dtypeB == ACL_FLOAT) {
        MatmulFp32TilingData fp32Tiling;
        matmul_fp32_get_tiling(m, n, k, transA, transB, problem.lda, problem.ldb, numBlocks, fp32Tiling);
        if (hasDecodedAlgo) {
            ApplyAlgoTilingOverrideFp32(decodedAlgo, m, n, k, fp32Tiling);
        }
        matmul_fp32_do(AsBytes(problem.A), AsBytes(problem.B), static_cast<uint8_t*>(dRawAddr), fp32Tiling, numBlocks,
            stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (IsMxfp8Type(dtypeA) && IsMxfp8Type(dtypeB)) {
        if (problem.scaleA == nullptr || problem.scaleB == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        QuantMatmulTilingData mxfp8Tiling;
        matmul_mxfp8_get_tiling(m, n, k, transA, transB, numBlocks, mxfp8Tiling);
        matmul_mxfp8_do(AsBytes(problem.A), AsBytes(problem.B), static_cast<uint8_t*>(problem.scaleA),
            static_cast<uint8_t*>(problem.scaleB), static_cast<uint8_t*>(dRawAddr), mxfp8Tiling, transA, transB,
            stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (IsMxfp4Type(dtypeA) && IsMxfp4Type(dtypeB)) {
        if (problem.scaleA == nullptr || problem.scaleB == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        QuantMatmulTilingData mxfp4Tiling;
        matmul_mxfp4_get_tiling(m, n, k, transA, transB, numBlocks, mxfp4Tiling);
        matmul_mxfp4_do(AsBytes(problem.A), AsBytes(problem.B), static_cast<uint8_t*>(problem.scaleA),
            static_cast<uint8_t*>(problem.scaleB), static_cast<uint8_t*>(dRawAddr), mxfp4Tiling, dtypeA, dtypeB,
            dtypeD, transA, transB, stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ACLBLAS_STATUS_NOT_SUPPORTED;
}

// Step 2: apply the alpha*D_raw + beta*C epilogue, writing the final result into problem.D. When the
// C/D buffers overlap, D_raw was staged in workspace so the workspace budget is validated here.
aclblasStatus_t LaunchEpilogue(const MatmulProblem& problem, void* dRawAddr, bool cOverlap)
{
    const uint64_t m = problem.m;
    const uint64_t n = problem.n;
    const aclrtStream stream = problem.stream;
    const aclDataType dtypeA = problem.dtypeA;
    const aclDataType dtypeB = problem.dtypeB;
    const aclDataType dtypeD = problem.dtypeD;
    const float alphaValue = problem.alpha;
    const float betaValue = problem.beta;

    if (!problem.devicePointerMode && betaValue != 0.0f && problem.C == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const aclDataType dtypeC = problem.dtypeC;
    const aclDataType dtypeDRaw = (dtypeA == ACL_FLOAT && dtypeB == ACL_FLOAT) ? ACL_FLOAT : dtypeD;
    const uint32_t ldc = problem.ldc;
    const uint32_t ldd = problem.ldd;
    const uint32_t lddRaw = (dRawAddr == problem.D) ? ldd : static_cast<uint32_t>(n);

    if (cOverlap) {
        const size_t dRawElemSize = (dtypeDRaw == ACL_BF16) ? sizeof(uint16_t) : sizeof(float);
        const size_t requiredWorkspace = static_cast<size_t>(m) * static_cast<size_t>(n) * dRawElemSize;
        if (problem.workspace == nullptr || problem.workspaceSize < requiredWorkspace) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }

    epilogue_alpha_beta_do(static_cast<uint8_t*>(dRawAddr),
        (problem.devicePointerMode || betaValue != 0.0f) ? AsBytes(problem.C) : nullptr,
        static_cast<uint8_t*>(problem.D), static_cast<uint32_t>(m), static_cast<uint32_t>(n), ldc, ldd, lddRaw,
        alphaValue, betaValue, dtypeC, dtypeDRaw, dtypeD, stream,
        problem.devicePointerMode ? static_cast<const float*>(problem.alphaDevice) : nullptr,
        problem.devicePointerMode ? static_cast<const float*>(problem.betaDevice) : nullptr);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t MatmulLaunch(const MatmulProblem& problem)
{
    if (problem.orderA != ACLBLASLT_ORDER_ROW || problem.orderB != ACLBLASLT_ORDER_ROW ||
        problem.orderC != ACLBLASLT_ORDER_ROW || problem.orderD != ACLBLASLT_ORDER_ROW) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    const uint64_t k = problem.k;
    const aclDataType dtypeA = problem.dtypeA;
    const float alphaValue = problem.alpha;
    const float betaValue = problem.beta;
    const bool needEpilogue = problem.devicePointerMode || (alphaValue != 1.0f || betaValue != 0.0f);
    const bool cOverlap = (problem.C == problem.D) && needEpilogue;
    void* dRawAddr = (needEpilogue && cOverlap) ? problem.workspace : problem.D;

    // Decode the caller-supplied algorithm blob so the configured tiling can drive the kernel. An
    // absent or corrupt algo (magic mismatch) leaves the auto-tiling path untouched.
    PackedAlgo decodedAlgo{};
    const bool hasDecodedAlgo = (problem.algo != nullptr) && DecodeAlgo(*problem.algo, &decodedAlgo);

    if ((IsMxfp8Type(dtypeA) || IsMxfp4Type(dtypeA)) && (k % 32 != 0)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const aclblasStatus_t mmadStatus = LaunchMmadKernel(problem, dRawAddr, hasDecodedAlgo, decodedAlgo);
    if (mmadStatus != ACLBLAS_STATUS_SUCCESS) {
        return mmadStatus;
    }

    if (needEpilogue) {
        const aclblasStatus_t epilogueStatus = LaunchEpilogue(problem, dRawAddr, cOverlap);
        if (epilogueStatus != ACLBLAS_STATUS_SUCCESS) {
            return epilogueStatus;
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}
