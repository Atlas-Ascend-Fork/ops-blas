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
#include <cstdint>
#include <cstddef>
#include <acl/acl.h>
#include "cann_ops_blas_common.h"

using aclblasHandle = struct _aclblas_handle*;
using aclblasLogCallback = void (*)(const char*);

typedef struct _aclblas_handle* aclblasHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates an ops-blas handle.
 *
 * Allocates a handle on the heap and returns it via @p handle. A 32 MiB default workspace
 * (device memory, managed by the library) is pre-allocated at creation time. The internal
 * structure is opaque to callers.
 *
 * @param handle Output parameter for the created handle (void**). *handle must be nullptr
 *               before the call; a non-null value is treated as an existing handle to
 *               prevent memory leaks.
 * @return ACLBLAS_STATUS_SUCCESS on success.
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR if handle is null.
 *         ACLBLAS_STATUS_INVALID_VALUE if *handle is non-null.
 *         ACLBLAS_STATUS_ALLOC_FAILED if memory allocation fails.
 */
aclblasStatus_t aclblasCreate(aclblasHandle_t* handle);

/**
 * @brief Destroys an ops-blas handle.
 *
 * Synchronizes the associated stream, frees the library default workspace and handle
 * resources, and clears user workspace references without freeing user memory.
 *
 * @param handle Handle to destroy (void*).
 * @return ACLBLAS_STATUS_SUCCESS on success.
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR if handle is null.
 *         ACLBLAS_STATUS_EXECUTION_FAILED if stream synchronization fails.
 */
aclblasStatus_t aclblasDestroy(aclblasHandle_t handle);

/**
 * @brief Sets the stream bound to a handle.
 *
 * Switching streams automatically resets to the default workspace.
 *
 * @param handle aclblas handle (void*).
 * @param stream Stream to bind; nullptr selects the default stream.
 * @return ACLBLAS_STATUS_SUCCESS on success.
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR if handle is null.
 */
aclblasStatus_t aclblasSetStream(aclblasHandle_t handle, aclrtStream stream);

/**
 * @brief Gets the stream bound to a handle.
 * @param handle aclblas handle (void*).
 * @param stream Output parameter for the current stream.
 * @return ACLBLAS_STATUS_SUCCESS on success.
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR if handle is null.
 *         ACLBLAS_STATUS_INVALID_VALUE if stream is null.
 */
aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream* stream);

/**
 * @brief Sets the workspace used by a handle.
 *
 * Borrows user device memory for temporary operator storage; the library does not take
 * ownership. Supports grow-only updates: the setting is updated only when the new size is
 * larger than the current user workspace size. Both @p workspace and @p workspaceSize must
 * be valid; use aclblasSetStream() to restore the library default workspace (cuBLAS-compatible).
 *
 * @param handle aclblas handle (void*).
 * @param workspace User-allocated device memory; must not be nullptr.
 * @param workspaceSize Workspace size in bytes; must be greater than 0.
 * @return ACLBLAS_STATUS_SUCCESS on success.
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR if handle is null.
 *         ACLBLAS_STATUS_INVALID_VALUE if workspace is null or workspaceSize is 0.
 */
aclblasStatus_t aclblasSetWorkspace(aclblasHandle_t handle, void* workspace, size_t workspaceSize);

/**
 * @brief Gets the ops-blas library version.
 *
 * The version is encoded as MAJOR * 10000 + MINOR * 100 + PATCH, e.g. 10000 for 1.0.0.
 *
 * @param handle Handle pointer; may be null.
 * @param version Output parameter for the version number.
 * @return ACLBLAS_STATUS_SUCCESS on success.
 *         ACLBLAS_STATUS_INVALID_VALUE if an argument is invalid.
 */
aclblasStatus_t aclblasGetVersion(aclblasHandle_t handle, int* version);

/**
 * @brief Configures logging for the ops-blas library.
 * @param logIsOn Whether to turn logging on/off completely (non-zero = on, 0 = off).
 * @param logToStdOut Whether to write logs to standard output (non-zero = yes, 0 = no).
 * @param logToStdErr Whether to write logs to standard error (non-zero = yes, 0 = no).
 * @param logFile Log file path; no file output when null. Caller must ensure the
 *                pointer remains valid during logging output (not copied).
 * @return ACLBLAS_STATUS_SUCCESS on success.
 */
aclblasStatus_t aclblasLoggerConfigure(
    int logIsOn, int logToStdOut, int logToStdErr, const char* logFile);

/**
 * @brief Sets the log callback function.
 * @param userCallback User-defined log callback function. Pass nullptr to clear
 *                     the previously installed callback.
 * @return ACLBLAS_STATUS_SUCCESS on success.
 */
aclblasStatus_t aclblasSetLoggerCallback(aclblasLogCallback userCallback);

/**
 * @brief Gets the log callback function.
 * @param userCallback Output parameter for the user-defined log callback function.
 * @return ACLBLAS_STATUS_SUCCESS on success.
 *         ACLBLAS_STATUS_INVALID_VALUE if userCallback is null.
 */
aclblasStatus_t aclblasGetLoggerCallback(aclblasLogCallback* userCallback);

aclblasStatus_t aclblasScopy_legacy(
    aclblasHandle_t handle, uint8_t* x, uint8_t* y, const int64_t n, const int64_t incx, const int64_t incy);

aclblasStatus_t aclblasScopy(aclblasHandle_t handle, int n, const float* x, int incx, float* y, int incy);

aclblasStatus_t aclblasSspmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x,
    int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasSger(
    aclblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy,
    float* A, int lda);

aclblasStatus_t aclblasCgeru(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda);

aclblasStatus_t aclblasStrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx);

aclblasStatus_t aclblasCtrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx);

aclblasStatus_t aclblasSsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x,
    int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasCsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy);

aclblasStatus_t aclblasSsbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasStpsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx);

aclblasStatus_t aclblasCtpsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx);

aclblasStatus_t aclblasStpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx);

aclblasStatus_t aclblasCtpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx);

// Early arch22 implementation. Prefer aclblasStpmv for new code; this interface may evolve or be removed in future
// releases.
aclblasStatus_t aclblasStpmv_legacy(
    aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, int64_t n,
    const float* aPacked, const float* x, float* y, int64_t incx);

// Early implementation with a non-standard parameter layout. Prefer aclblasStbmv for new code; this interface may
// evolve or be removed in future releases.
aclblasStatus_t aclblasStbmv_legacy(
    aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, const float* a,
    const int64_t lda, const float* x, float* y, const int64_t n, const int64_t k, const int64_t incx);

aclblasStatus_t aclblasStbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const float* A, int lda, float* x, int incx);

aclblasStatus_t aclblasCdotu(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y,
    int incy, aclblasComplex* result);

aclblasStatus_t aclblasCdotc(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y,
    int incy, aclblasComplex* result);

aclblasStatus_t aclblasSasum(aclblasHandle_t handle, int n, const float* x, int incx, float* result);

aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result);

aclblasStatus_t aclblasIsamax(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
aclblasStatus_t aclblasIcamax(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);

aclblasStatus_t aclblasCsrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y,
    int incy, const float* c, const float* s);

aclblasStatus_t aclblasComplexMatDot(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* matx, const aclblasComplex* maty,
    aclblasComplex* result);

aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy, int batchCount);

aclblasStatus_t aclblasCgerc(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda);

aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float *const Aarray[], int lda, const float *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t *const Aarray[], int lda, const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t *const Aarray[], int lda, const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasTSTgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t *const Aarray[], int lda, const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasTSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t *const Aarray[], int lda, const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float* A, int lda, int64_t strideA, const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasCcopy(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx,
    aclblasComplex* y, int incy);

aclblasStatus_t aclblasSdot(
    aclblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);

aclblasStatus_t aclblasDotEx(
    aclblasHandle_t handle, int n,
    const void* x, aclDataType xType, int incx,
    const void* y, aclDataType yType, int incy,
    void* result, aclDataType resultType, aclDataType executionType);

aclblasStatus_t aclblasSnrm2(aclblasHandle_t handle, int n, const float* x, int incx, float* result);

aclblasStatus_t aclblasScnrm2(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result);

aclblasStatus_t aclblasSnrm2Ex(
    aclblasHandle_t handle, aclDataType xtype, const void* x, const int64_t n, const int64_t incx, void* result);

aclblasStatus_t aclblasSscal(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx);

aclblasStatus_t aclblasScalex(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, void* x, aclDataType xType, int incx,
    aclDataType executionType);


aclblasStatus_t aclblasRotEx(
    aclblasHandle_t handle, int n,
    void *x, aclDataType xType, int incx,
    void *y, aclDataType yType, int incy,
    const void *c, const void *s,
    aclDataType csType, aclDataType executionType);
aclblasStatus_t aclblasCsscal(
    aclblasHandle_t handle, int n, const float* alpha, aclblasComplex* x, int incx);

aclblasStatus_t aclblasCscal(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, aclblasComplex* x, int incx);

aclblasStatus_t aclblasSswap(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy);

aclblasStatus_t aclblasCswap(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y,
    int incy);

aclblasStatus_t aclblasCaxpy(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* y, int incy);

aclblasStatus_t aclblasCgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);

aclblasStatus_t aclblasStrmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx);

aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda,
    const float* B, int ldb, float* C, int ldc);

aclblasStatus_t aclblasStrsm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb);

aclblasStatus_t aclblasCtrsm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    aclblasComplex* B, int ldb);

aclblasStatus_t aclblasSsymm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, const float* alpha,
    const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);

aclblasStatus_t aclblasSsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const float* alpha, const float* x, const int incx,
    float* A, const int lda);

aclblasStatus_t aclblasSsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const float* alpha, const float* x, const int incx,
    const float* y, const int incy, float* A, const int lda);

aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha,
    const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* A, int lda);

aclblasStatus_t aclblasSspr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* ap);

aclblasStatus_t aclblasSspr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx,
    const float* y, int incy, float* ap);

aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* ap);

aclblasStatus_t aclblasCtrmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx);

aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A,
    int lda, const float* x, int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy);

aclblasStatus_t aclblasStpttr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* AP, float* A, int lda);

aclblasStatus_t aclblasStrttp(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* A, int lda, float* AP);

aclblasStatus_t aclblasSrotm(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);

aclblasStatus_t aclblasSrot(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s);


aclblasStatus_t aclblasCrotg(
    aclblasHandle_t handle,
    aclblasComplex* a,
    aclblasComplex* b,
    float* c,
    aclblasComplex* s);

aclblasStatus_t aclblasSrotg(aclblasHandle_t handle, float* a, float* b, float* c, float* s);

aclblasStatus_t aclblasSrotm(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);

aclblasStatus_t aclblasSrotmg(
    aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);

aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy);

aclblasStatus_t aclblasAxpyEx(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, const void* x, aclDataType xType, int incx,
    void* y, aclDataType yType, int incy, aclDataType executionType);

aclblasStatus_t aclblasSgeqrfBatched(
    aclblasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info,
    int batchSize);

aclblasStatus_t aclblasSgetrfBatched(
    aclblasHandle_t handle, int n, float* const Aarray[], int lda, int* PivotArray, int* infoArray, int batchSize);
aclblasStatus_t aclblasCgetrfBatched(
    aclblasHandle_t handle, int n, aclblasComplex* const Aarray[], int lda, int* PivotArray, int* infoArray,
    int batchSize);
aclblasStatus_t aclblasSgelsBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda,
    float* const Carray[], int ldc, int* devInfo, int batchSize);

aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);

aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasChemm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc);

aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* const Aarray[], aclDataType Atype, int lda,
    const void* const Barray[], aclDataType Btype, int ldb,
    const void* beta, void* const Carray[], aclDataType Ctype, int ldc,
    int batchCount, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasGemmStridedBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* A, aclDataType Atype, int lda, int64_t strideA,
    const void* B, aclDataType Btype, int ldb, int64_t strideB,
    const void* beta, void* C, aclDataType Ctype, int ldc, int64_t strideC,
    int batchCount, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta, float* const Carray[], int ldc,
    int batchCount);

aclblasStatus_t aclblasCgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* const Aarray[], int lda,
    const aclblasComplex* const Barray[], int ldb,
    const aclblasComplex* beta, aclblasComplex* const Carray[], int ldc,
    int batchCount);

aclblasStatus_t aclblasSgetriBatched(
    aclblasHandle_t handle, int n, const float* const Aarray[], int lda, const int* PivotArray, float* const Carray[],
    int ldc, int* infoArray, int batchSize);

aclblasStatus_t aclblasCgetriBatched(
    aclblasHandle_t handle, int n, const aclblasComplex* const Aarray[], int lda, const int* PivotArray,
    aclblasComplex* const Carray[], int ldc, int* infoArray, int batchSize);

aclblasStatus_t aclblasSmatinvBatched(
    aclblasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info,
    int batchSize);

aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const void* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb,
    const void* beta, void* C, aclDataType Ctype, int ldc, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* const* Aarray, const int* ldaArray,
    const float* const* Barray, const int* ldbArray,
    const float* betaArray, float* const* Carray, const int* ldcArray,
    const int* groupSizeArray);

aclblasStatus_t aclblasSgetrsBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda,
    const int* devIpiv, float* const Barray[], int ldb, int* info, int batchCount);

aclblasStatus_t aclblasGemmGroupedBatchedEx(
    aclblasHandle_t handle, const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void* alphaArray,
    const void* const Aarray[], aclDataType Atype, const int ldaArray[],
    const void* const Barray[], aclDataType Btype, const int ldbArray[],
    const void* betaArray, void* const Carray[], aclDataType Ctype, const int ldcArray[],
    int groupCount, const int groupSize[], aclblasComputeType_t computeType);

aclblasStatus_t aclblasStbsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const float* A, int lda, float* x, int incx);

aclblasStatus_t aclblasCtbsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const aclblasComplex* A, int lda, aclblasComplex* x, int incx);

aclblasStatus_t aclblasSdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n,
    const float* A, int lda, const float* x, int incx,
    float* C, int ldc);

aclblasStatus_t aclblasCdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx,
    aclblasComplex* C, int ldc);

aclblasStatus_t aclblasSgeam(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc);

aclblasStatus_t aclblasCgeam(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* beta, const aclblasComplex* B, int ldb,
    aclblasComplex* C, int ldc);

aclblasStatus_t aclblasSgemm3m(
    aclblasHandle handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc);

aclblasStatus_t aclblasSsyrk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const float* A, int lda, const float* beta, float* C, int ldc);

aclblasStatus_t aclblasSsyr2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc);

aclblasStatus_t aclblasCsyr2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* const A[], int lda,
    float* const B[], int ldb,
    int batchCount);

aclblasStatus_t aclblasSgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, int64_t strideA, const float* B, int ldb, int64_t strideB,
    const float* beta, float* C, int ldc, int64_t strideC, int batchCount);

aclblasStatus_t aclblasSsyrkx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc);

aclblasStatus_t aclblasCherk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha,
    const aclblasComplex* A, int lda,
    const float* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCsyrk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCsymm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc);

#ifdef __cplusplus
}
#endif
