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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cann_ops_blas.h"
#include "cgemm_strided_batched_param.h"
#include "device.h"
#include "fill.h"

namespace cgemm_strided_batched_test {

// Padding sentinels detect accidental writes; separate seeds keep components independent.
constexpr size_t MATRIX_TAIL_GUARD_ELEMENTS = 16;
constexpr int MATRIX_PADDING_SENTINEL = -999;
constexpr uint32_t IMAGINARY_SEED_OFFSET = 1000;

// Preserve the package's [-5, 5] value/mean range and sampled normal sigma.
constexpr int RANDOM_VALUE_MIN = -5;
constexpr int RANDOM_VALUE_MAX = 5;
constexpr float NORMAL_SIGMA_MIN = 0.1f;
constexpr int NORMAL_SIGMA_MAX = 2;
constexpr int BATCH_SEED_STEP = 3;

inline aclblasStatus_t aclblasCgemmStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, int64_t strideA, const aclblasComplex* B, int ldb,
    int64_t strideB, const aclblasComplex* beta, aclblasComplex* C, int ldc, int64_t strideC, int batchCount)
{
    return aclblasCgemmStridedBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

struct Matrix {
    std::vector<Complex> host;
    std::unique_ptr<DeviceBuffer> device;
    int64_t base = 0;

    Matrix(
        int rows, int cols, int ld, int64_t stride, int batch, const std::string& fill, uint32_t seed, int offset,
        bool normal, bool preservePackage)
    {
        const size_t matrixSize = size_t(std::max(1, ld)) * std::max(1, cols);
        const size_t gap = size_t(std::max(0, batch - 1)) * std::abs(stride);
        base = offset + (stride < 0 ? gap : 0);
        host.assign(
            matrixSize + gap + offset + MATRIX_TAIL_GUARD_ELEMENTS,
            Complex{MATRIX_PADDING_SENTINEL, MATRIX_PADDING_SENTINEL});
        std::mt19937 realRng(seed);
        std::mt19937 imagRng(seed + IMAGINARY_SEED_OFFSET);
        std::uniform_real_distribution<float> uniform(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX);
        std::uniform_real_distribution<float> sigma(NORMAL_SIGMA_MIN, NORMAL_SIGMA_MAX);
        const float muR = uniform(realRng);
        const float sigmaR = sigma(realRng);
        const float muI = uniform(imagRng);
        const float sigmaI = sigma(imagRng);
        std::normal_distribution<float> normalR(muR, sigmaR);
        std::normal_distribution<float> normalI(muI, sigmaI);
        for (int batchIndex = 0; batchIndex < std::max(1, batch); ++batchIndex) {
            if (stride == 0 && batchIndex > 0) {
                break;
            }
            auto matrix = makeBlasComplexMatrix(
                std::max(1, rows), std::max(1, cols), std::max(1, ld), BlasFillMode(fill),
                seed + batchIndex * BATCH_SEED_STEP);
            for (int64_t index = 0; rows > 0 && index < int64_t(rows) * cols; ++index) {
                const int64_t column = index / rows;
                const int64_t row = index % rows;
                Complex value = matrix[size_t(column) * ld + row];
                if (fill == "RANDOM_NORM_5_5" && !preservePackage) {
                    value = normal ? Complex{normalR(realRng), normalI(imagRng)} :
                                     Complex{uniform(realRng), uniform(imagRng)};
                }
                host[base + batchIndex * stride + column * ld + row] = value;
            }
        }
        device = std::make_unique<DeviceBuffer>(host.size() * sizeof(Complex));
        device->copyFromHost(host.data(), host.size() * sizeof(Complex));
    }

    Complex* Ptr() { return reinterpret_cast<Complex*>(device->ptr()) + base; }
};

struct Invocation {
    const Case& p;
    aclblasHandle_t handle;
    int m;
    int n;
    int k;
    int batch;
    int ta;
    int tb;
    int lda;
    int ldb;
    int ldc;
    int64_t sa;
    int64_t sb;
    int64_t sc;
    Complex alpha;
    Complex beta;
    bool alphaNull;
    bool betaNull;
    bool aNull;
    bool bNull;
    bool cNull;

    Invocation(const Case& param, aclblasHandle_t supplied) : p(param), handle(supplied)
    {
        m = p.I("m");
        n = p.I("n");
        k = p.I("k");
        batch = p.I("batchCount");
        ta = ParseTrans(p.S("transA"));
        tb = ParseTrans(p.S("transB"));
        lda = p.S("lda").empty() ? std::max(1, ta == ACLBLAS_OP_N ? m : k) : p.I("lda");
        ldb = p.S("ldb").empty() ? std::max(1, tb == ACLBLAS_OP_N ? k : n) : p.I("ldb");
        ldc = p.S("ldc").empty() ? std::max(1, m) : p.I("ldc");
        sa = p.S("strideA").empty() ? int64_t(lda) * std::max(1, ta == ACLBLAS_OP_N ? k : m) : p.L("strideA");
        sb = p.S("strideB").empty() ? int64_t(ldb) * std::max(1, tb == ACLBLAS_OP_N ? n : k) : p.L("strideB");
        sc = p.S("strideC").empty() ? int64_t(ldc) * std::max(1, n) : p.L("strideC");
        alpha = {p.F("alpha_real"), p.F("alpha_imag")};
        beta = {p.F("beta_real"), p.F("beta_imag")};
        alphaNull = p.Flag("alpha_null");
        betaNull = p.Flag("beta_null");
        aNull = p.Flag("a_null");
        bNull = p.Flag("b_null");
        cNull = p.Flag("c_null");
        if (p.I("handle_null") || p.S("description").find("handle_null") != std::string::npos) {
            handle = nullptr;
        }
    }

    aclblasStatus_t Run(Complex* a, Complex* b, Complex* c) const
    {
        return aclblasCgemmStridedBatched_npu(
            handle, static_cast<aclblasOperation_t>(ta), static_cast<aclblasOperation_t>(tb), m, n, k,
            alphaNull ? nullptr : &alpha, aNull ? nullptr : a, lda, sa, bNull ? nullptr : b, ldb, sb,
            betaNull ? nullptr : &beta, cNull ? nullptr : c, ldc, sc, batch);
    }
};

} // namespace cgemm_strided_batched_test
