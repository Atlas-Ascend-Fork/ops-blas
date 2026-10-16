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
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cblas.h>
#include <dlfcn.h>

#include "cann_ops_blas.h"

namespace cgemm_strided_batched_test {

// Bound CPU reference concurrency without changing the reference computation.
constexpr int GOLDEN_DEFAULT_THREADS = 8;
constexpr int GOLDEN_MAX_THREADS = 32;
constexpr uint64_t GOLDEN_PARALLEL_WORK_THRESHOLD = 1048576;

using CgemmFn = decltype(&cblas_cgemm);

inline CgemmFn GetNetlibCgemm()
{
    static CgemmFn function = [] {
        const char* configuredPath = std::getenv("NETLIB_BLAS_LIBRARY");
        const char* path = configuredPath == nullptr ? "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3" : configuredPath;
        void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (library == nullptr) {
            throw std::runtime_error(dlerror());
        }
        auto result = reinterpret_cast<CgemmFn>(dlsym(library, "cblas_cgemm"));
        if (result == nullptr) {
            dlclose(library);
            throw std::runtime_error("Netlib cblas_cgemm not found");
        }
        std::cout << "NETLIB_GOLDEN=" << path << std::endl;
        return result;
    }();
    return function;
}

inline bool ValidCpuTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

inline bool InvalidCpuShape(int m, int n, int k, int batchCount) { return m < 0 || n < 0 || k < 0 || batchCount < 0; }

inline bool CpuNoOp(int m, int n, int batchCount) { return m == 0 || n == 0 || batchCount == 0; }

inline bool InvalidCpuStorage(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C,
    int ldc)
{
    const int minLda = std::max(1, transA == ACLBLAS_OP_N ? m : k);
    const int minLdb = std::max(1, transB == ACLBLAS_OP_N ? k : n);
    return alpha == nullptr || A == nullptr || B == nullptr || beta == nullptr || C == nullptr || lda < minLda ||
           ldb < minLdb || ldc < std::max(1, m);
}

inline aclblasStatus_t ValidateCpuArguments(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc, int batchCount)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!ValidCpuTrans(transA) || !ValidCpuTrans(transB)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (InvalidCpuShape(m, n, k, batchCount)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (CpuNoOp(m, n, batchCount)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (InvalidCpuStorage(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCgemmStridedBatched_cpu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, int64_t strideA, const aclblasComplex* B, int ldb,
    int64_t strideB, const aclblasComplex* beta, aclblasComplex* C, int ldc, int64_t strideC, int batchCount)
{
    auto status =
        ValidateCpuArguments(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (CpuNoOp(m, n, batchCount)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    CgemmFn netlib = GetNetlibCgemm();
    auto calculateBatch = [&](int index) {
        netlib(
            CblasColMajor, static_cast<CBLAS_TRANSPOSE>(transA), static_cast<CBLAS_TRANSPOSE>(transB), m, n, k, alpha,
            A + index * strideA, lda, B + index * strideB, ldb, beta, C + index * strideC, ldc);
    };

    int goldenThreads = GOLDEN_DEFAULT_THREADS;
    if (const char* configuredThreads = std::getenv("CGEMM_SB_GOLDEN_THREADS")) {
        goldenThreads = std::max(1, std::min(GOLDEN_MAX_THREADS, std::atoi(configuredThreads)));
    }
    const int workers = uint64_t(m) * n * k > GOLDEN_PARALLEL_WORK_THRESHOLD ? std::min(batchCount, goldenThreads) : 1;
    if (workers == 1) {
        for (int index = 0; index < batchCount; ++index) {
            calculateBatch(index);
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    std::vector<std::future<void>> pending;
    for (int worker = 0; worker < workers; ++worker) {
        pending.emplace_back(std::async(std::launch::async, [&, worker] {
            for (int index = worker; index < batchCount; index += workers) {
                calculateBatch(index);
            }
        }));
    }
    for (auto& done : pending) {
        done.get();
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace cgemm_strided_batched_test
