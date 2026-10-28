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
// The MXFP8/MXFP4 subtrees are compiled into libops_blasLt.so only when the CMake capability
// gate ENABLE_BLASLT_MXFP8 is on (blasLt/CMakeLists.txt excludes the whole subtrees otherwise);
// the top-level CMakeLists mirrors that gate into a same-named compile definition. Keep the
// includes and the call sites below behind the same macro so this translation unit never
// references definitions that were excluded from the library (undefined symbols at link time).
#if defined(ENABLE_BLASLT_MXFP8) && (ENABLE_BLASLT_MXFP8)
#include "matmul_mxfp4_host.h"
#include "matmul_mxfp8_host.h"
#include "matmul_mxfp8_kernel.h"
#include "quant_matmul_tiling_data.h"
#endif // ENABLE_BLASLT_MXFP8

#include <cstdint>

namespace {

inline uint8_t* AsBytes(const void* p) { return static_cast<uint8_t*>(const_cast<void*>(p)); }

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
#if defined(ENABLE_BLASLT_MXFP8) && (ENABLE_BLASLT_MXFP8)
    const aclDataType dtypeD = problem.dtypeD; // consumed by the MXFP4 launch entry only
#endif // ENABLE_BLASLT_MXFP8
    const bool transA = problem.transA;
    const bool transB = problem.transB;

    if (dtypeA == ACL_FLOAT && dtypeB == ACL_FLOAT) {
        MatmulFp32TilingData fp32Tiling;
        matmul_fp32_get_tiling(m, n, k, transA, transB, problem.lda, problem.ldb, numBlocks, fp32Tiling);
        if (hasDecodedAlgo) {
            ApplyAlgoTilingOverrideFp32(decodedAlgo, m, n, k, fp32Tiling);
        }
        matmul_fp32_do(
            AsBytes(problem.A), AsBytes(problem.B), static_cast<uint8_t*>(dRawAddr), fp32Tiling, numBlocks, stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    // MXFP8/MXFP4 call sites are compiled only together with their definitions (see the include
    // guard note above). With the gate off, well-formed MX requests fall through to the
    // NOT_SUPPORTED return below, matching the unsupported-dtype fallback semantics.
#if defined(ENABLE_BLASLT_MXFP8) && (ENABLE_BLASLT_MXFP8)
    if (IsMxfp8Type(dtypeA) && IsMxfp8Type(dtypeB)) {
        if (problem.scaleA == nullptr || problem.scaleB == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        QuantMatmulTilingData mxfp8Tiling;
        matmul_mxfp8_get_tiling(m, n, k, transA, transB, numBlocks, mxfp8Tiling);
        matmul_mxfp8_do(
            AsBytes(problem.A), AsBytes(problem.B), static_cast<uint8_t*>(problem.scaleA),
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
        matmul_mxfp4_do(
            AsBytes(problem.A), AsBytes(problem.B), static_cast<uint8_t*>(problem.scaleA),
            static_cast<uint8_t*>(problem.scaleB), static_cast<uint8_t*>(dRawAddr), mxfp4Tiling, dtypeA, dtypeB, dtypeD,
            transA, transB, stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
#endif // ENABLE_BLASLT_MXFP8
    return ACLBLAS_STATUS_NOT_SUPPORTED;
}

// Raw MMAD output element width: FP32 kernels write float; quant kernels write bfloat16 when the
// declared D dtype is BF16 (MXFP8 always writes bfloat16).
inline aclDataType RawOutputDtype(const MatmulProblem& problem)
{
    return (problem.dtypeA == ACL_FLOAT && problem.dtypeB == ACL_FLOAT) ? ACL_FLOAT : problem.dtypeD;
}

// Workspace bytes needed to stage the raw product when the raw buffer is not D itself and the
// epilogue runs (C/D overlap, or the raw element width differs from the declared D dtype).
inline size_t RequiredRawWorkspace(const MatmulProblem& problem)
{
    const size_t elemSize = (RawOutputDtype(problem) == ACL_BF16) ? sizeof(uint16_t) : sizeof(float);
    return static_cast<size_t>(problem.m) * static_cast<size_t>(problem.n) * elemSize;
}

// Step 2: apply the alpha*D_raw + beta*C epilogue, writing the final result into problem.D. When the
// raw product was staged in workspace (C/D overlap, or raw width differs from D dtype) the workspace
// budget is validated here.
aclblasStatus_t LaunchEpilogue(const MatmulProblem& problem, void* dRawAddr, bool rawStaged)
{
    const uint64_t m = problem.m;
    const uint64_t n = problem.n;
    const aclrtStream stream = problem.stream;
    const float alphaValue = problem.alpha;
    const float betaValue = problem.beta;

    if (!problem.devicePointerMode && betaValue != 0.0f && problem.C == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const aclDataType dtypeC = problem.dtypeC;
    const aclDataType dtypeDRaw = RawOutputDtype(problem);
    const uint32_t ldc = problem.ldc;
    const uint32_t ldd = problem.ldd;
    const uint32_t lddRaw = (dRawAddr == problem.D) ? ldd : static_cast<uint32_t>(n);

    if (rawStaged) {
        // 兜底校验：MatmulLaunch 入口已前置校验，此处保留以防绕过入口的内部调用。
        if (problem.workspace == nullptr || problem.workspaceSize < RequiredRawWorkspace(problem)) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }

    epilogue_alpha_beta_do(
        static_cast<uint8_t*>(dRawAddr),
        (problem.devicePointerMode || betaValue != 0.0f) ? AsBytes(problem.C) : nullptr,
        static_cast<uint8_t*>(problem.D), static_cast<uint32_t>(m), static_cast<uint32_t>(n), ldc, ldd, lddRaw,
        alphaValue, betaValue, dtypeC, dtypeDRaw, problem.dtypeD, stream,
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

    // !691 检视意见修复：原始乘积的元素宽度与 D dtype 不一致时（FP32 内核按 float 写出、
    // D 声明为 BF16），不得把 D 直接当 raw 缓冲——MMAD 会按 4 字节/元素越界写 2 字节缓冲。
    // 凡需要 epilogue 且 raw 宽度与 D dtype 不一致（或 C==D overlap）的场景，一律先 staging
    // 到 workspace，由 epilogue 按目标 dtype 转换写出。
    const bool rawStaged = needEpilogue && (cOverlap || RawOutputDtype(problem) != problem.dtypeD);

    // issue #583: 原始乘积写入 workspace 的场景，workspace 判空/容量校验必须前移到
    // MMAD 启动之前，否则 workspace 为空时内核已对设备地址 0 发起写。
    if (rawStaged && (problem.workspace == nullptr || problem.workspaceSize < RequiredRawWorkspace(problem))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    void* dRawAddr = rawStaged ? problem.workspace : problem.D;

    // Decode the caller-supplied algorithm blob so the configured tiling can drive the kernel. An
    // absent or corrupt algo (magic mismatch) leaves the auto-tiling path untouched.
    PackedAlgo decodedAlgo{};
    const bool hasDecodedAlgo = (problem.algo != nullptr) && DecodeAlgo(*problem.algo, &decodedAlgo);

    // issue #584: k=0 会通过 k%32 检查，但量化 tiling 的 CalScaleFactors 随后执行 0/0 整数除零
    // （主机端 SIGFPE），MXFP 路径对 k=0 显式拒绝。
    if ((IsMxfp8Type(dtypeA) || IsMxfp4Type(dtypeA)) && (k == 0 || k % 32 != 0)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const aclblasStatus_t mmadStatus = LaunchMmadKernel(problem, dRawAddr, hasDecodedAlgo, decodedAlgo);
    if (mmadStatus != ACLBLAS_STATUS_SUCCESS) {
        return mmadStatus;
    }

    if (needEpilogue) {
        const aclblasStatus_t epilogueStatus = LaunchEpilogue(problem, dRawAddr, rawStaged);
        if (epilogueStatus != ACLBLAS_STATUS_SUCCESS) {
            return epilogueStatus;
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}
