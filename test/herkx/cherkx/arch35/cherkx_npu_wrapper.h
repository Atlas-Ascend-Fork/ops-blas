/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHERKX_NPU_WRAPPER_H
#define CHERKX_NPU_WRAPPER_H

#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Device buffer RAII holder for the dual-matrix cherkx wrapper.
struct CherkxDeviceBuffers {
    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dB = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;

    ~CherkxDeviceBuffers()
    {
        if (dAlpha) aclrtFree(dAlpha);
        if (dA) aclrtFree(dA);
        if (dB) aclrtFree(dB);
        if (dBeta) aclrtFree(dBeta);
        if (dC) aclrtFree(dC);
    }
};

static aclblasStatus_t CherkxAllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    aclError aclRet = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Allocate+copy only when hostPtr is non-null and bytes > 0; leaves devPtr null
// otherwise so the caller can pass nullptr straight through to the API (this is
// how the nullAlpha / nullBeta / nullA / nullB / nullC cases reach the operator).
static aclblasStatus_t CherkxTryAllocAndCopy(
    const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return CherkxAllocAndCopy(devPtr, hostPtr, bytes);
}

// Matrix upload stage (A → B → C). Any failure returns immediately; the RAII
// destructor of CherkxDeviceBuffers releases the already-uploaded prefix.
static aclblasStatus_t CherkxUploadMatrices(
    CherkxDeviceBuffers& bufs, const aclblasComplex* A, size_t aBytes,
    const aclblasComplex* B, size_t bBytes, const aclblasComplex* C, size_t cBytes)
{
    aclblasStatus_t st = CherkxTryAllocAndCopy(A, aBytes, bufs.dA);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CherkxTryAllocAndCopy(B, bBytes, bufs.dB);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return CherkxTryAllocAndCopy(C, cBytes, bufs.dC);
}

// Device scalar upload helper (对齐 ssymm 终态形态 AllocSsymmDeviceScalars，
// 测试方案 §2.1 步骤 6)：alpha 为 8B 复数、beta 为 4B 实数，**仅当对应 onDevice
// 标志为 true 时**分配 + H2D；标志为 false（缺省）时保持 host 指针直通。
// nullAlpha/nullBeta 的 effPtr 为 nullptr → TryAllocAndCopy 不分配，nullptr 直通给 API。
static aclblasStatus_t AllocCherkxDeviceScalars(
    void*& dAlpha, void*& dBeta,
    const aclblasComplex* alpha, const float* beta,
    bool alphaOnDevice, bool betaOnDevice)
{
    constexpr size_t alphaBytes = sizeof(aclblasComplex); // complex<float> = 8B
    constexpr size_t betaBytes = sizeof(float);           // float = 4B
    if (alphaOnDevice) {
        aclblasStatus_t st = CherkxTryAllocAndCopy(alpha, alphaBytes, dAlpha);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    if (betaOnDevice) {
        aclblasStatus_t st = CherkxTryAllocAndCopy(beta, betaBytes, dBeta);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// "Prefer the device buffer, otherwise pass the host pointer through" — in the
// nullAlpha/nullBeta/host-mode cases there is no device buffer, so the caller's
// pointer reaches the operator as-is.
template <typename T>
static T* CherkxPickPtr(void* dev, T* host)
{
    return dev != nullptr ? static_cast<T*>(dev) : host;
}

// Const flavour for the input scalars (alpha / beta).
template <typename T>
static const T* CherkxPickConstPtr(void* dev, const T* host)
{
    return dev != nullptr ? static_cast<const T*>(dev) : host;
}

// Tail after a successful call: synchronize, then D2H-copy the whole ldc×n C buffer
// (padding rows included, so the ldc>n padding-protection cases are observable).
// C==nullptr with beta==0 returns SUCCESS without any write → nothing to copy back.
static aclblasStatus_t CherkxSyncAndCopyBack(aclblasStatus_t ret, void* dC, aclblasComplex* C, size_t cBytes)
{
    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (C != nullptr && dC != nullptr) {
        aclRet = aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ret;
}

// ── 以下三个纯表达式 helper 系 codecheck「超大圈复杂度」治理（CP5 非阻塞项）：
//    从 aclblasCherkx_npu 原样搬出三组三目表达式（操作数/求值顺序/类型逐位不变，
//    行为零变化），主函数不再含三目（对任何圈复杂度计数口径免疫）。──

// Effective host scalar pointer: nullptr when the null flag is set, so the API sees null
// (nullAlpha/nullBeta 语义；原内联形态 `nullFlag ? nullptr : ptr`）。
template <typename T>
static const T* CherkxEffScalarPtr(bool isNull, const T* ptr)
{
    return isNull ? nullptr : ptr;
}

// Device-side byte footprint of the three matrices. Guard A/B allocation with k>0
// (mirrors aclblasSsyrkx_npu): when k<=0 the operator short-circuits (k==0) or
// rejects (k<0) without accessing A/B, and abCols could be negative for k<0 which
// would overflow size_t. C is always allocated for n>0.
struct CherkxFootprint {
    size_t aBytes;
    size_t bBytes;
    size_t cBytes;
};

static CherkxFootprint CherkxComputeFootprint(
    aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    const int abCols = (k > 0) ? ((trans == ACLBLAS_OP_N) ? k : n) : 0;
    CherkxFootprint fp{};
    fp.aBytes = (k > 0) ? static_cast<size_t>(lda) * static_cast<size_t>(abCols) * sizeof(aclblasComplex) : 0;
    fp.bBytes = (k > 0) ? static_cast<size_t>(ldb) * static_cast<size_t>(abCols) * sizeof(aclblasComplex) : 0;
    fp.cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    return fp;
}

// Matrix argument for the API call: when bytes==0 (k<=0) pass nullptr so the
// operator's own k<=0 path is exercised (it must not dereference A/B); otherwise
// prefer the uploaded device buffer (原内联形态 `(bytes > 0) ? dev : nullptr`）。
static const aclblasComplex* CherkxMatrixArg(const void* dev, size_t bytes)
{
    return (bytes > 0) ? static_cast<const aclblasComplex*>(dev) : nullptr;
}

// NPU wrapper for aclblasCherkx (cuBLAS 对齐改造后签名：alpha 为 const aclblasComplex*).
// Mirrors aclblasCher2k_npu / aclblasSsyrkx_npu with complex element sizes
// (aclblasComplex = 8 bytes) and the dual A/B matrices.
//
// When handle is null or n<=0 it forwards directly (no device allocation) so the
// operator's own validation / quick-return paths are exercised with the caller's
// pointers unchanged (handle-null → HANDLE_IS_NULLPTR; n==0 → SUCCESS 早退，TC_AL_15).
//
// 标量指针模式（差距 4，测试方案 §3.1 指针模式因子）：
//   alphaOnDevice/betaOnDevice 缺省 false = **Host 模式**（对齐 ssymm 终态形态；
//   改造前 wrapper 强制 Device 标量）。两者须同置：(D,H)/(H,D) 混合模式由算子在
//   解引用前拒绝（INVALID_VALUE，TC_AL_18/19）——wrapper 只负责如实构造该组合。
//
// nullAlpha / nullBeta / nullA / nullB / nullC: when set, the wrapper passes nullptr
// for the corresponding pointer (no device alloc/copy), exercising the API's
// conditional-validation paths (α 复数零 / k==0 允许 A/B 空；β==0 允许 C 空)。
inline aclblasStatus_t aclblasCherkx_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc,
    bool nullAlpha = false, bool nullBeta = false,
    bool alphaOnDevice = false, bool betaOnDevice = false)
{
    // Effective host pointers: nullptr when the null flag is set, so the API sees null.
    const aclblasComplex* effAlpha = CherkxEffScalarPtr(nullAlpha, alpha);
    const float* effBeta = CherkxEffScalarPtr(nullBeta, beta);

    // Fast-path passthrough for validation-only invocations (handle null or n<=0):
    // skip device allocation so nullptr inputs reach the operator unchanged.
    if (handle == nullptr || n <= 0) {
        return aclblasCherkx(handle, uplo, trans, n, k, effAlpha, A, lda, B, ldb, effBeta, C, ldc);
    }

    const CherkxFootprint fp = CherkxComputeFootprint(trans, n, k, lda, ldb, ldc);

    CherkxDeviceBuffers bufs;
    aclblasStatus_t st = CherkxUploadMatrices(bufs, A, fp.aBytes, B, fp.bBytes, C, fp.cBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = AllocCherkxDeviceScalars(bufs.dAlpha, bufs.dBeta, effAlpha, effBeta, alphaOnDevice, betaOnDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // fp.aBytes/fp.bBytes == 0 (k<=0) → CherkxMatrixArg passes nullptr for A/B so the
    // operator's own k<=0 path is exercised (it must not dereference A/B).
    aclblasStatus_t ret = aclblasCherkx(
        handle, uplo, trans, n, k,
        CherkxPickConstPtr(bufs.dAlpha, effAlpha),
        CherkxMatrixArg(bufs.dA, fp.aBytes), lda,
        CherkxMatrixArg(bufs.dB, fp.bBytes), ldb,
        CherkxPickConstPtr(bufs.dBeta, effBeta),
        CherkxPickPtr(bufs.dC, C), ldc);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    return CherkxSyncAndCopyBack(ret, bufs.dC, C, fp.cBytes);
}

#endif // CHERKX_NPU_WRAPPER_H
