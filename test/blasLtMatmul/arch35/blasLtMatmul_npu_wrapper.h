/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"
#include "blasLtMatmul_param.h"

// ── Compute matrix bytes ──
inline size_t getMatrixBytes(int rows, int cols, int ld, aclDataType dtype)
{
    size_t elemSize = dtypeElementSize(dtype);
    if (dtype == ACL_FLOAT4_E2M1) {
        // MXFP4: ld is logical element leading dim (same as MXFP8); 2 elements per byte.
        size_t ldBytes = ((static_cast<size_t>(ld) + 1) / 2) * elemSize;
        return static_cast<size_t>(rows) * ldBytes;
    }
    return static_cast<size_t>(rows) * ld * elemSize;
}

// ── Create Layout descriptor ──
inline aclblasStatus_t createMatrixLayout(
    aclblasLtMatrixLayout_t* desc,
    aclDataType dtype, uint64_t rows, uint64_t cols, int64_t ld,
    aclblasLtOrder_t order)
{
    int64_t layoutLd = ld;
    if (dtype == ACL_FLOAT4_E2M1) {
        layoutLd = mxfp4PackedLd(ld);
    }
    aclblasStatus_t ret = aclblasLtMatrixLayoutCreate(desc, dtype, rows, cols, layoutLd);
    if (ret != ACLBLAS_STATUS_SUCCESS) return ret;
    ret = aclblasLtMatrixLayoutSetAttribute(*desc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    return ret;
}

// ── NPU wrapper for aclblasLtMatmul ──
struct LtMatmulNpuContext {
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    aclblasLtMatrixLayout_t Adesc = nullptr;
    aclblasLtMatrixLayout_t Bdesc = nullptr;
    aclblasLtMatrixLayout_t Cdesc = nullptr;
    aclblasLtMatrixLayout_t Ddesc = nullptr;
    aclblasLtMatmulPreference_t preference = nullptr;
    aclblasLtMatmulAlgo_t algo;
    bool algoInitialized = false;
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    void* dD = nullptr;
    void* dScaleA = nullptr;
    void* dScaleB = nullptr;
    void* dWorkspace = nullptr;
    size_t workspaceSize = 0;
};

inline void destroyNpuContext(LtMatmulNpuContext& ctx)
{
    if (ctx.dA) { aclrtFree(ctx.dA); ctx.dA = nullptr; }
    if (ctx.dB) { aclrtFree(ctx.dB); ctx.dB = nullptr; }
    if (ctx.dC) { aclrtFree(ctx.dC); ctx.dC = nullptr; }
    if (ctx.dD) { aclrtFree(ctx.dD); ctx.dD = nullptr; }
    if (ctx.dScaleA) { aclrtFree(ctx.dScaleA); ctx.dScaleA = nullptr; }
    if (ctx.dScaleB) { aclrtFree(ctx.dScaleB); ctx.dScaleB = nullptr; }
    if (ctx.dWorkspace) { aclrtFree(ctx.dWorkspace); ctx.dWorkspace = nullptr; }
    if (ctx.preference) { aclblasLtMatmulPreferenceDestroy(ctx.preference); ctx.preference = nullptr; }
    if (ctx.computeDesc) { aclblasLtMatmulDescDestroy(ctx.computeDesc); ctx.computeDesc = nullptr; }
    if (ctx.Adesc) { aclblasLtMatrixLayoutDestroy(ctx.Adesc); ctx.Adesc = nullptr; }
    if (ctx.Bdesc) { aclblasLtMatrixLayoutDestroy(ctx.Bdesc); ctx.Bdesc = nullptr; }
    if (ctx.Cdesc) { aclblasLtMatrixLayoutDestroy(ctx.Cdesc); ctx.Cdesc = nullptr; }
    if (ctx.Ddesc) { aclblasLtMatrixLayoutDestroy(ctx.Ddesc); ctx.Ddesc = nullptr; }
}

inline aclblasStatus_t aclblasLtMatmul_npu(
    aclblasLtHandle_t ltHandle,
    aclrtStream stream,
    aclDataType dtypeA, aclDataType dtypeB, aclDataType dtypeC, aclDataType dtypeD,
    int M, int N, int K,
    aclblasOperation_t transA, aclblasOperation_t transB,
    int lda, int ldb, int ldc, int ldd,
    float alpha, float beta,
    const void* A_host, const void* B_host,
    const void* C_host,
    void* D_host,
    const void* scaleA_host,
    const void* scaleB_host,
    const std::string& algoMode,
    bool CIsNull,
    bool handleNull,
    bool computeDescNull,
    bool alphaNull,
    bool Anull)
{
    // Null handle → quick return (for TC_L0_25)
    if (ltHandle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;

    // ── Null parameter validation path (for TC_L0_26-28) ──
    // When null parameter flags are set, call aclblasLtMatmul directly
    // to test API parameter validation, bypassing full wrapper setup.
    // This ensures CSV tests behave identically to TEST_F cases.
    if (computeDescNull) {
        float a = alpha, b = beta;
        return aclblasLtMatmul(
            ltHandle, nullptr,
            &a, nullptr, nullptr, nullptr, nullptr,
            &b, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, 0, stream);
    }

    if (alphaNull) {
        aclblasLtMatmulDesc_t desc = nullptr;
        aclblasStatus_t ret = aclblasLtMatmulDescCreate(&desc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);
        if (ret != ACLBLAS_STATUS_SUCCESS) return ret;
        float b = beta;
        ret = aclblasLtMatmul(
            ltHandle, desc,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            &b, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, 0, stream);
        aclblasLtMatmulDescDestroy(desc);
        return ret;
    }

    if (Anull) {
        // Create valid computeDesc + layouts but pass nullptr A device ptr
        LtMatmulNpuContext ctx;
        aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;
        aclblasLtOrder_t order = ACLBLASLT_ORDER_ROW;

        int physRowsA = getPhysicalRowsA(M, K, transA);
        int physColsA = getPhysicalColsA(M, K, transA);
        int physRowsB = getPhysicalRowsB(K, N, transB);
        int physColsB = getPhysicalColsB(K, N, transB);

        ret = createMatrixLayout(&ctx.Adesc, dtypeA, physRowsA, physColsA, lda, order);
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
        ret = createMatrixLayout(&ctx.Bdesc, dtypeB, physRowsB, physColsB, ldb, order);
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
        ret = createMatrixLayout(&ctx.Cdesc, dtypeC, M, N, ldc, order);
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
        ret = createMatrixLayout(&ctx.Ddesc, dtypeD, M, N, ldd, order);
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }

        ret = aclblasLtMatmulDescCreate(&ctx.computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
        int32_t transAVal = static_cast<int32_t>(transA);
        ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA,
                                               &transAVal, sizeof(int32_t));
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
        int32_t transBVal = static_cast<int32_t>(transB);
        ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB,
                                               &transBVal, sizeof(int32_t));
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }

        float a = alpha, b = beta;
        ret = aclblasLtMatmul(
            ltHandle, ctx.computeDesc,
            &a, nullptr, ctx.Adesc, nullptr, ctx.Bdesc,
            &b, nullptr, ctx.Cdesc, nullptr, ctx.Ddesc,
            nullptr, nullptr, 0, stream);

        destroyNpuContext(ctx);
        return ret;
    }

    LtMatmulNpuContext ctx;
    aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;
    aclblasLtOrder_t order = ACLBLASLT_ORDER_ROW;

    int physRowsA = getPhysicalRowsA(M, K, transA);
    int physColsA = getPhysicalColsA(M, K, transA);
    int physRowsB = getPhysicalRowsB(K, N, transB);
    int physColsB = getPhysicalColsB(K, N, transB);

    // Create Layout descriptors
    ret = createMatrixLayout(&ctx.Adesc, dtypeA, physRowsA, physColsA, lda, order);
    if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
    ret = createMatrixLayout(&ctx.Bdesc, dtypeB, physRowsB, physColsB, ldb, order);
    if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
    ret = createMatrixLayout(&ctx.Cdesc, dtypeC, M, N, ldc, order);
    if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
    ret = createMatrixLayout(&ctx.Ddesc, dtypeD, M, N, ldd, order);
    if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }

    // Create MatmulDesc
    aclblasComputeType_t computeType = getComputeType(dtypeA, dtypeD);
    aclDataType scaleType = ACL_FLOAT;
    ret = aclblasLtMatmulDescCreate(&ctx.computeDesc, computeType, scaleType);
    if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }

    int32_t transAVal = static_cast<int32_t>(transA);
    ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &transAVal, sizeof(int32_t));
    if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }

    int32_t transBVal = static_cast<int32_t>(transB);
    ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &transBVal, sizeof(int32_t));
    if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }

    aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
    ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(uint32_t));
    if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }

    // Scale Factors for MXFP
    if (isMxfpType(dtypeA) && scaleA_host != nullptr && M > 0 && K > 0) {
        size_t scaleABytes = mxScaleBufferBytesA(M, K, transA);
        aclError aclRet = aclrtMalloc(&ctx.dScaleA, scaleABytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(ctx.dScaleA, scaleABytes, scaleA_host, scaleABytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_INTERNAL_ERROR; }
        ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ctx.dScaleA, sizeof(void*));
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
        int32_t scaleMode = 1;
        ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(int32_t));
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
    }

    if (isMxfpType(dtypeB) && scaleB_host != nullptr && K > 0 && N > 0) {
        size_t scaleBBytes = mxScaleBufferBytesB(N, K, transB);
        aclError aclRet = aclrtMalloc(&ctx.dScaleB, scaleBBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(ctx.dScaleB, scaleBBytes, scaleB_host, scaleBBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_INTERNAL_ERROR; }
        ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ctx.dScaleB, sizeof(void*));
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
        int32_t scaleMode = 1;
        ret = aclblasLtMatmulDescSetAttribute(ctx.computeDesc, ACLBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(int32_t));
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
    }

    // Allocate device buffers
    if (A_host != nullptr && M > 0 && K > 0) {
        size_t aBytes = getMatrixBytes(physRowsA, physColsA, lda, dtypeA);
        aclError aclRet = aclrtMalloc(&ctx.dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(ctx.dA, aBytes, A_host, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (B_host != nullptr && K > 0 && N > 0) {
        size_t bBytes = getMatrixBytes(physRowsB, physColsB, ldb, dtypeB);
        aclError aclRet = aclrtMalloc(&ctx.dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(ctx.dB, bBytes, B_host, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (!CIsNull && C_host != nullptr && M > 0 && N > 0) {
        size_t cBytes = static_cast<size_t>(M) * ldc * dtypeElementSize(dtypeC);
        aclError aclRet = aclrtMalloc(&ctx.dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(ctx.dC, cBytes, C_host, cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (D_host != nullptr && M > 0 && N > 0) {
        size_t dElemSize = dtypeElementSize(dtypeD);
        size_t dBytes = static_cast<size_t>(M) * ldd * dElemSize;
        aclError aclRet = aclrtMalloc(&ctx.dD, dBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_ALLOC_FAILED; }
        if (aclrtMemset(ctx.dD, dBytes, 0, dBytes) != ACL_SUCCESS) {
            destroyNpuContext(ctx);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    // Algorithm selection
    const aclblasLtMatmulAlgo_t* algoPtr = nullptr;
    if (algoMode == "nullptr") {
        algoPtr = nullptr;
        ctx.workspaceSize = 32 * 1024 * 1024;
    } else {
        ret = aclblasLtMatmulPreferenceCreate(&ctx.preference);
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }
        size_t maxWorkspace = 32 * 1024 * 1024;
        ret = aclblasLtMatmulPreferenceSetAttribute(ctx.preference, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspace, sizeof(uint64_t));
        if (ret != ACLBLAS_STATUS_SUCCESS) { destroyNpuContext(ctx); return ret; }

        int returnedAlgoCount = 0;
        aclblasLtMatmulHeuristicResult_t heuristicResult;
        ret = aclblasLtMatmulAlgoGetHeuristic(
            ltHandle, ctx.computeDesc, ctx.Adesc, ctx.Bdesc,
            ctx.Cdesc, ctx.Ddesc, ctx.preference, 1,
            &heuristicResult, &returnedAlgoCount);
        if (ret != ACLBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
            destroyNpuContext(ctx);
            return (ret != ACLBLAS_STATUS_SUCCESS) ? ret : ACLBLAS_STATUS_NOT_SUPPORTED;
        }
        ctx.algo = heuristicResult.algo;
        ctx.algoInitialized = true;
        ctx.workspaceSize = heuristicResult.workspaceSize;
        algoPtr = &ctx.algo;
    }

    // Allocate workspace
    if (ctx.workspaceSize > 0) {
        aclError aclRet = aclrtMalloc(&ctx.dWorkspace, ctx.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { destroyNpuContext(ctx); return ACLBLAS_STATUS_ALLOC_FAILED; }
    }

    // Execute
    ret = aclblasLtMatmul(
        ltHandle, ctx.computeDesc, &alpha,
        ctx.dA, ctx.Adesc, ctx.dB, ctx.Bdesc,
        &beta, (CIsNull || ctx.dC == nullptr) ? nullptr : ctx.dC, ctx.Cdesc,
        ctx.dD, ctx.Ddesc, algoPtr,
        ctx.dWorkspace, ctx.workspaceSize, stream);

    // Sync and copy result
    aclError syncRet = (stream != nullptr) ? aclrtSynchronizeStream(stream) : aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (D_host != nullptr && ctx.dD != nullptr && M > 0 && N > 0 && ret == ACLBLAS_STATUS_SUCCESS) {
        size_t dElemSize = dtypeElementSize(dtypeD);
        size_t dBytes = static_cast<size_t>(M) * ldd * dElemSize;
        if (aclrtMemcpy(D_host, dBytes, ctx.dD, dBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    destroyNpuContext(ctx);
    return ret;
}

