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

#include <cmath>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>

#include "cher2k_golden.h"
#include "cher2k_npu_wrapper.h"

// Reuse the same device addresses across unit/non-unit scalar transitions.
// Guard regions and leading-dimension padding must remain bit-for-bit intact.
class Cher2kScalarRegression {
public:
    Cher2kScalarRegression(
        aclblasHandle_t handle, aclrtStream stream, int n, int k, aclblasFillMode_t uplo, aclblasOperation_t trans)
        : handle_(handle),
          stream_(stream),
          n_(n),
          k_(k),
          uplo_(uplo),
          trans_(trans),
          ld_((trans == ACLBLAS_OP_N ? n : k) + 4),
          ldc_(n + 4),
          a_(static_cast<size_t>(ld_) * (trans == ACLBLAS_OP_N ? k : n)),
          b_(a_.size()),
          initial_(static_cast<size_t>(ldc_) * n + 2U * GUARD, {0.25f, -0.5f})
    {
        for (size_t i = 0; i < a_.size(); ++i) {
            a_[i] = {
                static_cast<float>(static_cast<int>(i % 7U) - 3) / 16.0f,
                static_cast<float>(static_cast<int>(i % 5U) - 2) / 16.0f};
            b_[i] = {
                static_cast<float>(static_cast<int>(i % 11U) - 5) / 16.0f,
                static_cast<float>(static_cast<int>(i % 3U) - 1) / 16.0f};
        }
    }

    void Run(bool hostAlpha, bool hostBeta)
    {
        const aclblasComplex initialAlpha{1.0f, 0.0f};
        const float initialBeta = 0.0f;
        ASSERT_EQ(
            Cher2kPrepareDeviceBuffers(
                buffers_, &initialAlpha, a_.data(), Bytes(a_), b_.data(), Bytes(b_), &initialBeta, initial_.data(),
                Bytes(initial_)),
            ACLBLAS_STATUS_SUCCESS);
        const aclblasComplex alphas[] = {{1.0f, 0.0f}, {0.5f, 0.0f},   {0.0f, 0.0f}, {-0.5f, 0.25f},
                                         {0.0f, 0.0f}, {-0.0f, -0.0f}, {1.0f, 0.0f}};
        const float betas[] = {0.0f, 0.5f, 0.0f, -0.25f, 1.0f, 1.0f, 0.0f};
        for (size_t i = 0; i < sizeof(betas) / sizeof(betas[0]); ++i) {
            SCOPED_TRACE(
                ::testing::Message() << "n=" << n_ << " k=" << k_ << " scalar step=" << i << " host alpha=" << hostAlpha
                                     << " host beta=" << hostBeta);
            ASSERT_NO_FATAL_FAILURE(RunSetting(alphas[i], betas[i], hostAlpha, hostBeta));
        }
    }

private:
    static constexpr size_t GUARD = 32U;

    static size_t Bytes(const std::vector<aclblasComplex>& values) { return values.size() * sizeof(aclblasComplex); }

    void RunSetting(const aclblasComplex& alpha, float beta, bool hostAlpha, bool hostBeta)
    {
        ASSERT_EQ(
            aclrtMemcpy(buffers_.c, Bytes(initial_), initial_.data(), Bytes(initial_), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(buffers_.alpha, sizeof(alpha), &alpha, sizeof(alpha), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(buffers_.beta, sizeof(beta), &beta, sizeof(beta), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        const auto* alphaPtr = hostAlpha ? &alpha : static_cast<const aclblasComplex*>(buffers_.alpha);
        const auto* betaPtr = hostBeta ? &beta : static_cast<const float*>(buffers_.beta);
        ASSERT_EQ(
            aclblasCher2k(
                handle_, uplo_, trans_, n_, k_, alphaPtr, static_cast<const aclblasComplex*>(buffers_.a), ld_,
                static_cast<const aclblasComplex*>(buffers_.b), ld_, betaPtr,
                static_cast<aclblasComplex*>(buffers_.c) + GUARD, ldc_),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        std::vector<aclblasComplex> actual(initial_.size());
        ASSERT_EQ(
            aclrtMemcpy(actual.data(), Bytes(actual), buffers_.c, Bytes(actual), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        Cher2kGoldenDbl expected;
        ASSERT_EQ(
            Cher2kRunGoldenDbl(
                handle_, uplo_, trans_, n_, k_, &alpha, a_.data(), ld_, b_.data(), ld_, &beta, initial_.data() + GUARD,
                ldc_, expected),
            ACLBLAS_STATUS_SUCCESS);
        const bool noOp = beta == 1.0f && (k_ == 0 || (alpha.real == 0.0f && alpha.imag == 0.0f));
        ASSERT_NO_FATAL_FAILURE(CheckOutput(actual, expected.c, noOp));
        ASSERT_NO_FATAL_FAILURE(CheckInput(buffers_.a, a_));
        ASSERT_NO_FATAL_FAILURE(CheckInput(buffers_.b, b_));
    }

    void CheckOutput(
        const std::vector<aclblasComplex>& actual, const std::vector<aclblasDoubleComplex>& expected, bool noOp)
    {
        for (size_t i = 0; i < actual.size(); ++i) {
            const bool matrix = i >= GUARD && i < actual.size() - GUARD;
            const size_t row = matrix ? (i - GUARD) % ldc_ : 0U;
            const size_t col = matrix ? (i - GUARD) / ldc_ : 0U;
            const bool selected =
                matrix && row < static_cast<size_t>(n_) && (uplo_ == ACLBLAS_UPPER ? row <= col : row >= col);
            if (selected && !noOp) {
                ASSERT_NEAR(actual[i].real, expected[i - GUARD].real, 1.0e-3) << "index=" << i;
                ASSERT_NEAR(actual[i].imag, expected[i - GUARD].imag, 1.0e-3) << "index=" << i;
                if (row == col) {
                    ASSERT_EQ(actual[i].imag, 0.0f) << "diagonal=" << row;
                }
            } else {
                ASSERT_EQ(0, std::memcmp(&actual[i], &initial_[i], sizeof(aclblasComplex)))
                    << (noOp && matrix ? "no-op matrix index=" : "guard/padding index=") << i;
            }
        }
    }

    static void CheckInput(const void* device, const std::vector<aclblasComplex>& expected)
    {
        std::vector<aclblasComplex> actual(expected.size());
        ASSERT_EQ(
            aclrtMemcpy(actual.data(), Bytes(actual), device, Bytes(actual), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        ASSERT_EQ(0, std::memcmp(actual.data(), expected.data(), Bytes(expected)));
    }

    aclblasHandle_t handle_;
    aclrtStream stream_;
    int n_;
    int k_;
    aclblasFillMode_t uplo_;
    aclblasOperation_t trans_;
    int ld_;
    int ldc_;
    std::vector<aclblasComplex> a_;
    std::vector<aclblasComplex> b_;
    std::vector<aclblasComplex> initial_;
    Cher2kDeviceBuffers buffers_;
};
