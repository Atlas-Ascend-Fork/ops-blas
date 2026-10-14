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
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "cann_ops_blas.h"

extern "C" void cherk_(
    const char* uplo, const char* trans, const int* n, const int* k, const float* alpha, const aclblasComplex* a,
    const int* lda, const float* beta, aclblasComplex* c, const int* ldc);

extern "C" void cgemm_(
    const char* transA, const char* transB, const int* m, const int* n, const int* k, const aclblasComplex* alpha,
    const aclblasComplex* a, const int* lda, const aclblasComplex* b, const int* ldb, const aclblasComplex* beta,
    aclblasComplex* c, const int* ldc);

// Sentinel: all validations passed, proceed to compute.
static const int CHERK_CPU_VALID_OK = 0x7FFFFFFF;

static bool CherkIsValidUplo(aclblasFillMode_t uplo) { return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER; }

static bool CherkIsValidTrans(aclblasOperation_t trans) { return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_C; }

struct CherkNetlibBlock {
    int row;
    int col;
    int rows;
    int cols;
};

static void RunCherkNetlibBlock(
    const CherkNetlibBlock& block, char uplo, char trans, int k, float alpha, const aclblasComplex* a, int lda,
    float beta, aclblasComplex* c, int ldc)
{
    aclblasComplex* cBlock = c + block.row + static_cast<size_t>(block.col) * ldc;
    const aclblasComplex* aRow = trans == 'N' ? a + block.row : a + static_cast<size_t>(block.row) * lda;

    if (block.row == block.col) {
        cherk_(&uplo, &trans, &block.rows, &k, &alpha, aRow, &lda, &beta, cBlock, &ldc);
        return;
    }

    const aclblasComplex* aCol = trans == 'N' ? a + block.col : a + static_cast<size_t>(block.col) * lda;
    const char transA = trans == 'N' ? 'N' : 'C';
    const char transB = trans == 'N' ? 'C' : 'N';
    const aclblasComplex complexAlpha{alpha, 0.0f};
    const aclblasComplex complexBeta{beta, 0.0f};
    cgemm_(
        &transA, &transB, &block.rows, &block.cols, &k, &complexAlpha, aRow, &lda, aCol, &lda, &complexBeta, cBlock,
        &ldc);
}

static void RunCherkNetlibParallel(
    char uplo, char trans, int n, int k, float alpha, const aclblasComplex* a, int lda, float beta, aclblasComplex* c,
    int ldc)
{
    constexpr int blockSize = 256;
    std::vector<CherkNetlibBlock> blocks;
    for (int col = 0; col < n; col += blockSize) {
        const int cols = std::min(blockSize, n - col);
        const int rowBegin = uplo == 'U' ? 0 : col;
        const int rowEnd = uplo == 'U' ? col + 1 : n;
        for (int row = rowBegin; row < rowEnd; row += blockSize) {
            blocks.push_back({row, col, std::min(blockSize, n - row), cols});
        }
    }

    const unsigned int hardwareThreads = std::max(1U, std::thread::hardware_concurrency());
    const size_t threadCount = std::min<size_t>(blocks.size(), std::min(8U, hardwareThreads));
    std::atomic<size_t> nextBlock{0U};
    const auto worker = [&]() {
        for (size_t index = nextBlock.fetch_add(1U, std::memory_order_relaxed); index < blocks.size();
             index = nextBlock.fetch_add(1U, std::memory_order_relaxed)) {
            RunCherkNetlibBlock(blocks[index], uplo, trans, k, alpha, a, lda, beta, c, ldc);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (size_t i = 0; i < threadCount; ++i) {
        workers.emplace_back(worker);
    }
    for (auto& thread : workers) {
        thread.join();
    }
}

// Parameter validation aligned with ssyrk_golden.h; complex version differs
// only in A/C pointer types (aclblasComplex*). See 测试方案 §2.4.
static int ValidateCherkCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldc,
    const float* alpha, const float* beta, const aclblasComplex* A, const aclblasComplex* C)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (!CherkIsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (trans == ACLBLAS_OP_T) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (!CherkIsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    bool isTrans = (trans != ACLBLAS_OP_N);
    int minLda = isTrans ? std::max(1, k) : std::max(1, n);
    if (lda < minLda) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (A == nullptr && n > 0 && k > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (C == nullptr && n > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CHERK_CPU_VALID_OK;
}

// BLAS libraries differ in nonfinite complex intermediate arithmetic. Use
// component arithmetic with real alpha/beta for nonfinite input matrices.
static bool CherkHasNonfiniteInput(aclblasOperation_t trans, int n, int k, const aclblasComplex* a, int lda)
{
    const int rows = trans == ACLBLAS_OP_N ? n : k;
    const int cols = trans == ACLBLAS_OP_N ? k : n;
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            const auto value = a[row + static_cast<size_t>(col) * lda];
            if (!std::isfinite(value.real) || !std::isfinite(value.imag)) {
                return true;
            }
        }
    }
    return false;
}

static aclblasComplex CherkReferenceProduct(
    aclblasOperation_t trans, int row, int col, int k, const aclblasComplex* a, int lda)
{
    float real = 0.0f;
    float imag = 0.0f;
    for (int l = 0; l < k; ++l) {
        const auto x =
            trans == ACLBLAS_OP_N ? a[row + static_cast<size_t>(l) * lda] : a[l + static_cast<size_t>(row) * lda];
        const auto y =
            trans == ACLBLAS_OP_N ? a[col + static_cast<size_t>(l) * lda] : a[l + static_cast<size_t>(col) * lda];
        real += x.real * y.real + x.imag * y.imag;
        imag += trans == ACLBLAS_OP_N ? x.imag * y.real - x.real * y.imag : x.real * y.imag - x.imag * y.real;
    }
    return {real, imag};
}

static void RunCherkNonfiniteReference(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, float alpha, const aclblasComplex* a, int lda,
    float beta, aclblasComplex* c, int ldc)
{
    for (int col = 0; col < n; ++col) {
        const int begin = uplo == ACLBLAS_UPPER ? 0 : col;
        const int end = uplo == ACLBLAS_UPPER ? col + 1 : n;
        for (int row = begin; row < end; ++row) {
            const auto product = CherkReferenceProduct(trans, row, col, k, a, lda);
            auto& value = c[row + static_cast<size_t>(col) * ldc];
            value.real = alpha * product.real + (beta == 0.0f ? 0.0f : beta * value.real);
            value.imag = row == col ? 0.0f : alpha * product.imag + (beta == 0.0f ? 0.0f : beta * value.imag);
        }
    }
}

// CPU golden for aclblasCherk. Signature mirrors the NPU API exactly so the
// test harness can swap NPU/CPU calls one-for-one. Returns aclblasStatus_t and
// applies the same validation as the NPU operator (CBLAS would crash on null
// pointers, so validation must precede the Netlib CHERK call).
//
// BLAS CHERK accepts only no-transpose and conjugate-transpose. OP_T is a
// valid enum, but it is unsupported by this API and rejected above.
inline aclblasStatus_t aclblasCherk_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const aclblasComplex* A, int lda, const float* beta, aclblasComplex* C, int ldc)
{
    int st = ValidateCherkCpuParams(handle, uplo, trans, n, k, lda, ldc, alpha, beta, A, C);
    if (st != CHERK_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    float alphaVal = *alpha;
    float betaVal = *beta;

    if (n > 0 && k > 0 && alphaVal != 0.0f && CherkHasNonfiniteInput(trans, n, k, A, lda)) {
        RunCherkNonfiniteReference(uplo, trans, n, k, alphaVal, A, lda, betaVal, C, ldc);
        return ACLBLAS_STATUS_SUCCESS;
    }

    const char netlibUplo = uplo == ACLBLAS_UPPER ? 'U' : 'L';
    const char netlibTrans = trans == ACLBLAS_OP_N ? 'N' : 'C';
    if (n < 512) {
        cherk_(&netlibUplo, &netlibTrans, &n, &k, &alphaVal, A, &lda, &betaVal, C, &ldc);
    } else {
        RunCherkNetlibParallel(netlibUplo, netlibTrans, n, k, alphaVal, A, lda, betaVal, C, ldc);
    }

    return ACLBLAS_STATUS_SUCCESS;
}
