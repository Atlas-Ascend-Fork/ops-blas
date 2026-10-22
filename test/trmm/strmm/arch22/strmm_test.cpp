/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "strmm_param.h"
#include "strmm_golden.h"
#include "strmm_npu_wrapper.h"

// Column-major matrix buffer: m×n with leading dimension ld.
// makeBlasMatrix handles m<=0 / n<=0 / ld<=0 by returning an empty vector.
static inline std::vector<float> MakeMatrix(int rows, int cols, int ld,
    const BlasFillMode& fill, uint32_t seed)
{
    return makeBlasMatrix(rows, cols, ld, fill, seed);
}

// Triangular matrix: dimA×dimA column-major, only the uplo triangle is filled.
// The full matrix is generated with the original fill (preserving the CSV-specified
// val1/val2 value range), then the non-triangular part is manually zeroed by uplo.
static inline std::vector<float> MakeTriMatrix(int dimA, int lda,
    aclblasFillMode_t uplo, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<float> data = makeBlasMatrix(dimA, dimA, lda, fill, seed);
    if (data.empty()) return data;
    for (int col = 0; col < dimA; ++col) {
        for (int row = 0; row < dimA; ++row) {
            bool inTriangle = (uplo == ACLBLAS_UPPER) ? (row <= col) : (row >= col);
            if (!inTriangle) {
                data[static_cast<size_t>(col) * lda + row] = 0.0f;
            }
        }
    }
    return data;
}

static inline void ApplySpecialValues(const std::string& desc,
    std::vector<float>& aHost, std::vector<float>& bHost)
{
    if (desc.find("nanA") != std::string::npos || desc.find("nan_A") != std::string::npos) {
        std::fill(aHost.begin(), aHost.end(), std::numeric_limits<float>::quiet_NaN());
    }
    if (desc.find("infA") != std::string::npos || desc.find("inf_A") != std::string::npos) {
        std::fill(aHost.begin(), aHost.end(), std::numeric_limits<float>::infinity());
    }
    if (desc.find("nanB") != std::string::npos || desc.find("nan_B") != std::string::npos) {
        std::fill(bHost.begin(), bHost.end(), std::numeric_limits<float>::quiet_NaN());
    }
}

class StrmmArch22Test : public BlasTest<StrmmParam> {};

TEST_F(StrmmArch22Test, NullHandle)
{
    float alpha = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasStrmm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, 4,
        &alpha, a.data(), 4, b.data(), 4, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

static bool MallocDev(void** dPtr, size_t bytes)
{
    aclError ret = aclrtMalloc(dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(static_cast<int>(ret), ACL_SUCCESS) << "aclrtMalloc failed for dC";
    return ret == ACL_SUCCESS;
}

static std::vector<float> MakeUpperTriangular(int lda, int m)
{
    std::vector<float> a(static_cast<size_t>(lda) * m, 0.0f);
    for (int col = 0; col < m; ++col) {
        for (int row = 0; row <= col; ++row) {
            a[static_cast<size_t>(col) * lda + row] = 1.0f;
        }
    }
    return a;
}

static void VerifyAlphaZeroPadding(const std::vector<float>& cHost, int m, int n, int ldc, float sentinel)
{
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            EXPECT_FLOAT_EQ(cHost[static_cast<size_t>(j) * ldc + i], 0.0f)
                << "active row " << i << " col " << j << " should be zeroed";
        }
        for (int i = m; i < ldc; ++i) {
            EXPECT_FLOAT_EQ(cHost[static_cast<size_t>(j) * ldc + i], sentinel)
                << "padding row " << i << " col " << j << " should be preserved";
        }
    }
}

// alpha==0 fast path with ldc > m: HandleStrmmAlphaZero zeros the m active rows
// of each column via per-column memset, while padding rows (m..ldc-1) are
// preserved. Device C is pre-filled with a non-zero sentinel to verify that
// only the active rows are cleared and padding rows remain untouched.
TEST_F(StrmmArch22Test, AlphaZeroLdcPadding)
{
    const int m = 4;
    const int n = 4;
    const int lda = 4;
    const int ldb = 4;
    const int ldc = 8;  // ldc > m — padding rows exist (rows 4..7)
    const float alpha = 0.0f;
    const float sentinel = 42.0f;

    // Valid A (4x4 upper triangular) and B (4x4). When alpha==0 the host
    // fast path skips A/B entirely, but valid pointers exercise the full
    // parameter validation path before the alpha==0 early return.
    std::vector<float> aHost = MakeUpperTriangular(lda, m);
    std::vector<float> bHost(static_cast<size_t>(ldb) * n, 1.0f);

    // Pre-fill device C with a non-zero sentinel.
    const size_t cCount = static_cast<size_t>(ldc) * static_cast<size_t>(n);
    const size_t cBytes = cCount * sizeof(float);
    std::vector<float> cHost(cCount, sentinel);

    void* dC = nullptr;
    if (!MallocDev(&dC, cBytes)) return;
    aclError aclRet = aclrtMemcpy(dC, cBytes, cHost.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        EXPECT_EQ(static_cast<int>(aclRet), ACL_SUCCESS) << "aclrtMemcpy H2D failed for dC";
        FreeDev(dC);
        return;
    }

    aclblasStatus_t ret = aclblasStrmm(
        StrmmArch22Test::handle_,
        ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        m, n, &alpha, aHost.data(), lda, bHost.data(), ldb,
        static_cast<float*>(dC), ldc);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = StrmmSyncAndCopyD2H(StrmmArch22Test::handle_, dC, cHost.data(), cBytes);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    }
    FreeDev(dC);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        VerifyAlphaZeroPadding(cHost, m, n, ldc, sentinel);
    }
}

TEST_F(StrmmArch22Test, InPlaceBEqualsC)
{
    constexpr int n = 4;
    const float alpha = 1.25F;
    for (int m : {4, 16, 32, 64, 129, 257}) {
        const auto uplo = (m == 16 || m == 32) ? ACLBLAS_LOWER : ACLBLAS_UPPER;
        std::vector<float> a(static_cast<size_t>(m) * m, 0.0F);
        std::vector<float> b(static_cast<size_t>(m) * n);
        for (int col = 0; col < m; ++col) {
            for (int row = 0; row < m; ++row) {
                if (uplo == ACLBLAS_UPPER ? row <= col : row >= col) {
                    a[static_cast<size_t>(col) * m + row] =
                        row == col ? 1.0F + 0.01F * row : 0.1F * (col - row + 1);
                }
            }
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.125F;
        }
        std::vector<float> expected = b;
        ASSERT_EQ(aclblasStrmm_cpu(StrmmArch22Test::handle_, ACLBLAS_SIDE_LEFT,
                                   uplo, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                                   m, n, &alpha, a.data(), m, b.data(), m, expected.data(), m),
                  ACLBLAS_STATUS_SUCCESS);

        void* dA = nullptr;
        void* dBC = nullptr;
        const size_t aBytes = a.size() * sizeof(float);
        const size_t bBytes = b.size() * sizeof(float);
        ASSERT_EQ(AllocCopyH2D(dA, a.data(), aBytes), ACL_SUCCESS);
        ASSERT_EQ(AllocCopyH2D(dBC, b.data(), bBytes), ACL_SUCCESS);
        auto* deviceA = static_cast<const float*>(dA);
        auto* deviceBC = static_cast<float*>(dBC);
        aclblasStatus_t ret = aclblasStrmm(StrmmArch22Test::handle_, ACLBLAS_SIDE_LEFT,
                                           uplo, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                                           m, n, &alpha, deviceA, m, deviceBC, m, deviceBC, m);
        EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
        if (ret == ACLBLAS_STATUS_SUCCESS) {
            std::vector<float> actual(b.size());
            ret = StrmmSyncAndCopyD2H(StrmmArch22Test::handle_, dBC, actual.data(), bBytes);
            EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
            if (ret == ACLBLAS_STATUS_SUCCESS) {
                for (size_t i = 0; i < actual.size(); ++i) {
                    EXPECT_NEAR(actual[i], expected[i], 1e-3F) << "m=" << m << " element=" << i;
                }
            }
        }
        FreeDev(dBC);
        FreeDev(dA);
    }
}

TEST_F(StrmmArch22Test, DeviceAlpha)
{
    constexpr int m = 8;
    constexpr int n = 4;
    const float alpha = 0.75F;
    const std::vector<float> a = MakeUpperTriangular(m, m);
    std::vector<float> b(static_cast<size_t>(m) * n);
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
    }
    std::vector<float> actual(b.size(), 0.0F);
    std::vector<float> expected(b.size(), 0.0F);
    ASSERT_EQ(aclblasStrmm_cpu(StrmmArch22Test::handle_, ACLBLAS_SIDE_LEFT,
                               ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                               m, n, &alpha, a.data(), m, b.data(), m, expected.data(), m),
              ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasStrmm_npu(StrmmArch22Test::handle_, ACLBLAS_SIDE_LEFT,
                               ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                               m, n, &alpha, a.data(), m, b.data(), m, actual.data(), m,
                               true),
              ACLBLAS_STATUS_SUCCESS);
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-4F) << "element=" << i;
    }
}

TEST_F(StrmmArch22Test, IgnoresUnstoredTriangleAndUnitDiagonal)
{
    constexpr int n = 2;
    const float alpha = 1.0F;
    for (int m : {8, 16, 32, 64}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T}) {
                std::vector<float> a(static_cast<size_t>(m) * m, std::numeric_limits<float>::quiet_NaN());
                for (int col = 0; col < m; ++col) {
                    for (int row = 0; row < m; ++row) {
                        const bool selected = (uplo == ACLBLAS_UPPER) ? row < col : row > col;
                        if (selected) {
                            a[static_cast<size_t>(col) * m + row] =
                                0.01F * static_cast<float>(row - col);
                        }
                    }
                }
                std::vector<float> b(static_cast<size_t>(m) * n, 0.5F);
                std::vector<float> actual(b.size(), 0.0F);
                std::vector<float> expected(b.size(), 0.0F);
                ASSERT_EQ(aclblasStrmm_cpu(StrmmArch22Test::handle_, ACLBLAS_SIDE_LEFT,
                                           uplo, trans, ACLBLAS_UNIT,
                                           m, n, &alpha, a.data(), m, b.data(), m, expected.data(), m),
                          ACLBLAS_STATUS_SUCCESS);
                ASSERT_EQ(aclblasStrmm_npu(StrmmArch22Test::handle_, ACLBLAS_SIDE_LEFT,
                                           uplo, trans, ACLBLAS_UNIT,
                                           m, n, &alpha, a.data(), m, b.data(), m, actual.data(), m),
                          ACLBLAS_STATUS_SUCCESS);
                for (size_t i = 0; i < actual.size(); ++i) {
                    EXPECT_NEAR(actual[i], expected[i], 1e-3F)
                        << "m=" << m << " uplo=" << static_cast<int>(uplo)
                        << " trans=" << static_cast<int>(trans) << " element=" << i;
                }
            }
        }
    }
}

static void MakeMixedSpecialInputs(int m, int n, aclblasFillMode_t uplo,
    aclblasDiagType_t diag, int variant, int lda, int ldb,
    std::vector<float>& a, std::vector<float>& b)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    a.assign(static_cast<size_t>(lda) * m, nan);
    for (int col = 0; col < m; ++col) {
        for (int row = 0; row < m; ++row) {
            if ((uplo == ACLBLAS_UPPER ? row <= col : row >= col) &&
                (diag != ACLBLAS_UNIT || row != col)) {
                a[static_cast<size_t>(col) * lda + row] = variant < 4
                    ? (row == col ? 1.25F : 0.125F)
                    : ((variant == 4 || col % 8 < 4) ? 1.0e19F : -1.0e19F);
            }
        }
    }
    b.assign(static_cast<size_t>(ldb) * n, variant < 4 ? 0.5F : 1.0e19F);
    if (variant < 4) {
        const int k = m / 2;
        const int row = uplo == ACLBLAS_UPPER ? k - 1 : k + 1;
        a[static_cast<size_t>(k) * lda + row] = variant < 2 ? inf : nan;
        for (int col = 0; col < n; ++col) {
            b[static_cast<size_t>(col) * ldb + k] = 0.0F;
        }
        b[static_cast<size_t>(ldb) + 1] = variant % 2 == 0 ? inf : nan;
        b[static_cast<size_t>(2 * ldb) + m - 2] = -inf;
    }
}

static void CheckMixedSpecialCase(aclblasHandle_t handle, int m, aclblasFillMode_t uplo,
    aclblasDiagType_t diag, int variant)
{
    constexpr int n = 3;
    const float alpha = 0.75F;
    SCOPED_TRACE(::testing::Message() << "m=" << m << " uplo=" << uplo
                 << " diag=" << diag << " variant=" << variant);
    const int lda = m + (variant % 2) * 3;
    const int ldb = m + (variant % 2) * 5;
    std::vector<float> a;
    std::vector<float> b;
    MakeMixedSpecialInputs(m, n, uplo, diag, variant, lda, ldb, a, b);
    std::vector<float> actual(static_cast<size_t>(m) * n);
    std::vector<float> expected(actual.size());
    ASSERT_EQ(aclblasStrmm_cpu(handle, ACLBLAS_SIDE_LEFT,
                               uplo, ACLBLAS_OP_N, diag, m, n, &alpha,
                               a.data(), lda, b.data(), ldb, expected.data(), m),
              ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasStrmm_npu(handle, ACLBLAS_SIDE_LEFT,
                               uplo, ACLBLAS_OP_N, diag, m, n, &alpha,
                               a.data(), lda, b.data(), ldb, actual.data(), m),
              ACLBLAS_STATUS_SUCCESS);
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(actual[i])) << "element=" << i;
        } else if (std::isinf(expected[i])) {
            EXPECT_EQ(actual[i], expected[i]) << "element=" << i;
        } else {
            EXPECT_NEAR(actual[i], expected[i], 1e-4F) << "element=" << i;
        }
    }
}

// Mixed special values must not leak across the unstored triangle. A zero B
// skips even a stored Inf/NaN A value, as in Netlib's LEFT/N algorithm.
TEST_F(StrmmArch22Test, DirectMixedSpecialValues)
{
    for (int m : {16, 32}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                for (int variant = 0; variant < 6; ++variant) {
                    if (variant >= 4 && uplo != ACLBLAS_LOWER) continue;
                    ASSERT_NO_FATAL_FAILURE(CheckMixedSpecialCase(handle_, m, uplo, diag, variant));
                }
            }
        }
    }
}

TEST_F(StrmmArch22Test, SmallLowerTileTailsAndPadding)
{
    for (int m : {16, 32}) {
        for (int n : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32}) {
            for (auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                for (int padding : {0, 3}) {
                    const int lda = m + padding;
                    const int ldb = m + padding;
                    std::vector<float> a(static_cast<size_t>(lda) * m,
                                         std::numeric_limits<float>::quiet_NaN());
                    for (int col = 0; col < m; ++col) {
                        for (int row = col + (diag == ACLBLAS_UNIT); row < m; ++row) {
                            a[static_cast<size_t>(col) * lda + row] =
                                static_cast<float>((row * 3 + col) % 11 - 5) * 0.125F;
                        }
                    }
                    std::vector<float> b(static_cast<size_t>(ldb) * n, 0.0F);
                    for (int col = 0; col < n; ++col) {
                        for (int row = 0; row < m; ++row) {
                            b[static_cast<size_t>(col) * ldb + row] =
                                static_cast<float>((row + col * 5) % 13 - 6) * 0.0625F;
                        }
                    }
                    for (float alpha : {0.75F, -1.25F}) {
                        SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n
                                     << " diag=" << diag << " pad=" << padding << " alpha=" << alpha);
                        std::vector<float> actual(static_cast<size_t>(m) * n);
                        std::vector<float> expected(actual.size());
                        ASSERT_EQ(aclblasStrmm_cpu(StrmmArch22Test::handle_, ACLBLAS_SIDE_LEFT,
                                                   ACLBLAS_LOWER, ACLBLAS_OP_N, diag, m, n, &alpha,
                                                   a.data(), lda, b.data(), ldb, expected.data(), m),
                                  ACLBLAS_STATUS_SUCCESS);
                        ASSERT_EQ(aclblasStrmm_npu(StrmmArch22Test::handle_, ACLBLAS_SIDE_LEFT,
                                                   ACLBLAS_LOWER, ACLBLAS_OP_N, diag, m, n, &alpha,
                                                   a.data(), lda, b.data(), ldb, actual.data(), m),
                                  ACLBLAS_STATUS_SUCCESS);
                        for (size_t i = 0; i < actual.size(); ++i) {
                            EXPECT_NEAR(actual[i], expected[i], 1e-4F) << "element=" << i;
                        }
                    }
                }
            }
        }
    }
}

// Keep the defaults in sync with report_perf.py. Explicit overrides also allow
// reproducing the historical ten-sample reports.
static constexpr int PERF_WARMUP = 5;
static int PerfInteger(const char* name, int defaultValue, int minimum, int maximum)
{
    const char* value = std::getenv(name);
    if (value == nullptr) return defaultValue;
    char* end = nullptr;
    const long number = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && number >= minimum && number <= maximum
               ? static_cast<int>(number) : -1;
}

static int PerfSamples() { return PerfInteger("STRMM_PERF_SAMPLES", 20, 1, 10000); }

struct PerfCase {
    int m;
    int n;
    aclblasSideMode_t side;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
};

static void FillPerfProfileInputs(int dim, aclblasFillMode_t uplo,
    std::vector<float>& a, std::vector<float>& b)
{
    for (int col = 0; col < dim; ++col) {
        for (int row = 0; row < dim; ++row) {
            if ((uplo == ACLBLAS_UPPER && row <= col) ||
                (uplo == ACLBLAS_LOWER && row >= col)) {
                a[static_cast<size_t>(col) * dim + row] =
                    row == col ? 1.0F : 0.001F * (1 + ((row + col) % 7));
            }
        }
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.05F;
    }
}

// Run explicitly under msprof: five warmups then twenty effective samples.
// No CPU golden is repeated inside the measured loop.
TEST_F(StrmmArch22Test, DISABLED_PerfProfile)
{
    static constexpr PerfCase cases[] = {
        {256, 256, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT},
        {512, 512, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT},
        {1024, 1024, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT},
        {2048, 2048, ACLBLAS_SIDE_RIGHT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT},
        {4096, 4096, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT},
    };
    const char* selected = std::getenv("STRMM_PERF_CASE");
    ASSERT_NE(selected, nullptr) << "Set STRMM_PERF_CASE to 0..4";
    const int index = std::atoi(selected);
    ASSERT_GE(index, 0);
    ASSERT_LT(index, 5);
    const auto& p = cases[index];
    const int samples = PerfSamples();
    ASSERT_GT(samples, 0) << "STRMM_PERF_SAMPLES must be an integer in 1..10000";
    const int dim = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    std::vector<float> a(static_cast<size_t>(dim) * dim, 0.0F);
    std::vector<float> b(static_cast<size_t>(p.m) * p.n);
    FillPerfProfileInputs(dim, p.uplo, a, b);
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    const size_t aBytes = a.size() * sizeof(float);
    const size_t bBytes = b.size() * sizeof(float);
    ASSERT_EQ(AllocCopyH2D(dA, a.data(), aBytes), ACL_SUCCESS);
    ASSERT_EQ(AllocCopyH2D(dB, b.data(), bBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dC, bBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    const float alpha = 1.0F;
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    for (int repeat = 0; repeat < PERF_WARMUP + samples; ++repeat) {
        status = aclblasStrmm(StrmmArch22Test::handle_, p.side, p.uplo, p.trans, p.diag,
                              p.m, p.n, &alpha, static_cast<const float*>(dA), dim,
                              static_cast<const float*>(dB), p.m, static_cast<float*>(dC), p.m);
        if (status != ACLBLAS_STATUS_SUCCESS) break;
    }
    aclError sync = aclrtSynchronizeStream(StrmmArch22Test::stream_);
    FreeDev(dC);
    FreeDev(dB);
    FreeDev(dA);
    EXPECT_EQ(status, ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(sync, ACL_SUCCESS);
}

static void RunPerfCase(aclblasHandle_t handle, aclrtStream stream, const StrmmParam& p, int samples)
{
    const int dim = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    auto a = MakeTriMatrix(dim, p.lda, p.uplo, p.aFill, p.randomSeed);
    auto b = MakeMatrix(p.m, p.n, p.ldb, p.bFill, p.randomSeed + 1);
    ASSERT_FALSE(a.empty()) << p.caseName;
    ASSERT_FALSE(b.empty()) << p.caseName;
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    const size_t aBytes = a.size() * sizeof(float);
    const size_t bBytes = b.size() * sizeof(float);
    const size_t cBytes = static_cast<size_t>(p.ldc) * p.n * sizeof(float);
    ASSERT_EQ(AllocCopyH2D(dA, a.data(), aBytes), ACL_SUCCESS) << p.caseName;
    ASSERT_EQ(AllocCopyH2D(dB, b.data(), bBytes), ACL_SUCCESS) << p.caseName;
    ASSERT_EQ(aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS) << p.caseName;
    std::cout << "PERF_CASE_START " << p.caseName << std::endl;
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    for (int repeat = 0; repeat < PERF_WARMUP + samples; ++repeat) {
        status = aclblasStrmm(handle, p.side, p.uplo, p.trans, p.diag,
                              p.m, p.n, &p.alpha, static_cast<const float*>(dA), p.lda,
                              static_cast<const float*>(dB), p.ldb, static_cast<float*>(dC), p.ldc);
        if (status != ACLBLAS_STATUS_SUCCESS) break;
    }
    aclError sync = aclrtSynchronizeStream(stream);
    FreeDev(dC);
    FreeDev(dB);
    FreeDev(dA);
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS) << p.caseName;
    ASSERT_EQ(sync, ACL_SUCCESS) << p.caseName;
    std::cout << "PERF_CASE_DONE " << p.caseName << std::endl;
}

// Collect all 200 task performance cases in one profiler session. There is no
// CPU golden in this path; the CSV-driven tests above verify numerical results.
TEST_F(StrmmArch22Test, DISABLED_PerfAll)
{
    const auto cases = GetCasesFromCsv<StrmmParam>(ReplaceFileExtension2Csv(__FILE__));
    const char* only = std::getenv("STRMM_PERF_ONLY");
    const int samples = PerfSamples();
    ASSERT_GT(samples, 0) << "STRMM_PERF_SAMPLES must be an integer in 1..10000";
    const int total = static_cast<int>(std::count_if(cases.begin(), cases.end(),
        [](const StrmmParam& p) { return p.caseName.rfind("TC_PF_", 0) == 0; }));
    const int first = PerfInteger("STRMM_PERF_FIRST", 0, 0, total - 1);
    ASSERT_GE(first, 0) << "STRMM_PERF_FIRST is outside the performance case list";
    const int count = PerfInteger("STRMM_PERF_COUNT", total - first, 1, total - first);
    ASSERT_GT(count, 0) << "STRMM_PERF_COUNT is outside the remaining performance case list";
    ASSERT_TRUE(only == nullptr || (std::getenv("STRMM_PERF_FIRST") == nullptr &&
                                   std::getenv("STRMM_PERF_COUNT") == nullptr))
        << "Use either STRMM_PERF_ONLY or FIRST/COUNT, not both";
    int measured = 0;
    int perfIndex = 0;
    for (const auto& p : cases) {
        if (p.caseName.rfind("TC_PF_", 0) != 0) continue;
        const int index = perfIndex++;
        if (only != nullptr ? p.caseName != only : (index < first || index >= first + count)) continue;
        ASSERT_NO_FATAL_FAILURE(RunPerfCase(handle_, stream_, p, samples));
        ++measured;
    }
    EXPECT_EQ(measured, only == nullptr ? count : 1);
}

INSTANTIATE_TEST_SUITE_P(
    Strmm, StrmmArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<StrmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StrmmParam>);

// Builds the triangular A and rectangular B host buffers, applies special
// NaN/Inf values from the case description, and resolves the alpha/A/B pointers
// (nullptr when the corresponding null flag is set). Sets alphaOnDevice when the
// case description contains "dev_alpha" (device-pointer alpha path).
static void PrepareTestData(const StrmmParam& p, int aDim,
    std::vector<float>& aHost, std::vector<float>& bHost,
    const float*& alphaPtr, const float*& aPtr, const float*& bPtr,
    bool& alphaOnDevice)
{
    aHost = MakeTriMatrix(aDim, p.lda, p.uplo, p.aFill, p.randomSeed);
    bHost = MakeMatrix(p.m, p.n, p.ldb, p.bFill, p.randomSeed + 1);
    ApplySpecialValues(p.description, aHost, bHost);

    alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    aPtr = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    bPtr = p.nullB ? nullptr : (bHost.empty() ? nullptr : bHost.data());

    alphaOnDevice = (!p.nullAlpha && p.description.find("dev_alpha") != std::string::npos);
}

// alpha==0 with nullA/nullB: the host alpha==0 fast path memsets C without
// reading A/B. A device dC is used so the host's aclrtMemset fast path works on
// device memory, then D2H back to the host C buffer. The result is
// bit-reproducible (C is all zeros), verified with EXACT equality.
static void HandleAlphaZeroNullAB(const StrmmParam& p, aclblasHandle_t handle,
    const float* alphaPtr, const float* aPtr, const float* bPtr, float* cPtr,
    bool alphaOnDevice)
{
    const size_t cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);
    const size_t cBytes = cCount * sizeof(float);
    void* dC = nullptr;
    if (!MallocDev(&dC, cBytes)) return;
    void* dAlpha = nullptr;
    const float* alphaArg = alphaPtr;
    if (alphaOnDevice && alphaPtr != nullptr) {
        aclError alphaRet = aclrtMalloc(&dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (alphaRet != ACL_SUCCESS) {
            EXPECT_EQ(static_cast<int>(alphaRet), ACL_SUCCESS) << "aclrtMalloc failed for dAlpha";
            FreeDev(dC);
            return;
        }
        alphaRet = aclrtMemcpy(dAlpha, sizeof(float), alphaPtr, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (alphaRet != ACL_SUCCESS) {
            EXPECT_EQ(static_cast<int>(alphaRet), ACL_SUCCESS) << "aclrtMemcpy failed for dAlpha";
            FreeDev(dAlpha);
            FreeDev(dC);
            return;
        }
        alphaArg = static_cast<const float*>(dAlpha);
    }
    aclblasStatus_t ret = aclblasStrmm(
        handle,
        p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        alphaArg, aPtr, p.lda, bPtr, p.ldb,
        static_cast<float*>(dC), p.ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS && cPtr != nullptr) {
        ret = StrmmSyncAndCopyD2H(handle, dC, cPtr, cBytes);
    }
    FreeDev(dAlpha);
    FreeDev(dC);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS && cPtr != nullptr) {
        for (size_t i = 0; i < cCount; i++) {
            EXPECT_EQ(cPtr[i], 0.0f) << "C should be all zeros when alpha==0";
        }
    }
}

// Computes the golden C via aclblasStrmm_cpu and verifies the NPU result with
// mixed tolerance. alpha==0 uses EXACT equality (C = 0, bit-reproducible).
static void RunGoldenAndVerify(const StrmmParam& p, aclblasHandle_t handle,
    const float* alphaPtr, const float* aPtr, const float* bPtr, float* cPtr)
{
    std::vector<float> cGolden = std::vector<float>(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n), 0.0f);
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    aclblasStatus_t goldenRet = aclblasStrmm_cpu(
        handle,
        p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, goldPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    const size_t cCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.n);
    if (cCount == 0) return;
    std::vector<float> actual(cCount);
    std::vector<float> expected(cCount);
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            const size_t logical = static_cast<size_t>(col) * p.m + row;
            actual[logical] = cPtr[static_cast<size_t>(col) * p.ldc + row];
            expected[logical] = goldPtr[static_cast<size_t>(col) * p.ldc + row];
        }
    }

    VerifyConfig cfg;
    if (p.alpha == 0.0f) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, expected.data(), cCount);
        // This A2/A3 task uses the FLOAT32 mixed tolerance from its task sheet.
        cfg.mixedAtol = 1.0 / 8192.0;
        cfg.mixedRtol = 1.0 / 8192.0;
    }
    EXPECT_TRUE(Verifier::verifyVector(actual.data(), expected.data(), cCount, 1, cfg, p.caseName));
}

static void MakeTiledReferenceInputs(const StrmmParam& p, int dim,
    std::vector<float>& a, std::vector<float>& b)
{
    a.assign(static_cast<size_t>(p.lda) * dim, std::numeric_limits<float>::quiet_NaN());
    for (int col = 0; col < dim; ++col) {
        for (int row = 0; row < dim; ++row) {
            const bool stored = p.uplo == ACLBLAS_UPPER ? row <= col : row >= col;
            if (stored && !(row == col && p.diag == ACLBLAS_UNIT)) {
                a[static_cast<size_t>(col) * p.lda + row] = row == col ? 1.0F :
                    0.002F * static_cast<float>((row + 3 * col) % 11 - 5);
            }
        }
    }
    b.assign(static_cast<size_t>(p.ldb) * p.n, -999.0F);
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            b[static_cast<size_t>(col) * p.ldb + row] =
                0.03F * static_cast<float>((3 * row + col) % 19 - 9);
        }
    }
}

static void VerifyTiledOutputPadding(const StrmmParam& p, const std::vector<float>& c, bool inPlace)
{
    for (int col = 0; col < p.n; ++col) {
        for (int row = p.m; row < p.ldc; ++row) {
            EXPECT_EQ(c[static_cast<size_t>(col) * p.ldc + row], inPlace ? -999.0F : 0.0F)
                << "output padding modified";
        }
    }
}

static void RunTiledReferenceInPlace(aclblasHandle_t handle, const StrmmParam& p,
    const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c)
{
    void* aDevice = nullptr;
    void* bDevice = nullptr;
    ASSERT_EQ(AllocCopyH2D(aDevice, a.data(), a.size() * sizeof(float)), ACL_SUCCESS);
    const aclError bStatus = AllocCopyH2D(bDevice, b.data(), b.size() * sizeof(float));
    if (bStatus != ACL_SUCCESS) {
        FreeDev(aDevice);
        ASSERT_EQ(bStatus, ACL_SUCCESS);
    }
    aclblasStatus_t status = aclblasStrmm(handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha,
        static_cast<const float*>(aDevice), p.lda, static_cast<const float*>(bDevice), p.ldb,
        static_cast<float*>(bDevice), p.ldc);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = StrmmSyncAndCopyD2H(handle, bDevice, c.data(), c.size() * sizeof(float));
    }
    FreeDev(aDevice);
    FreeDev(bDevice);
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS);
}

static void CheckTiledReference(aclblasHandle_t handle, int m, int n, aclblasSideMode_t side,
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int padding,
    bool inPlace = false, float alpha = 1.0F, bool compactOutput = false)
{
    SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n << " side=" << side
        << " uplo=" << uplo << " trans=" << trans << " diag=" << diag << " padding=" << padding);
    StrmmParam p(csv_map{});
    p.caseName = "tiling_boundary";
    p.m = m;
    p.n = n;
    p.side = side;
    p.uplo = uplo;
    p.trans = trans;
    p.diag = diag;
    p.alpha = alpha;
    const int dim = side == ACLBLAS_SIDE_LEFT ? m : n;
    p.lda = dim + padding;
    p.ldb = m + padding;
    p.ldc = m + (padding == 0 || compactOutput ? 0 : padding + 4);
    if (inPlace) p.ldc = p.ldb;
    std::vector<float> a;
    std::vector<float> b;
    MakeTiledReferenceInputs(p, dim, a, b);
    std::vector<float> c(static_cast<size_t>(p.ldc) * n, 0.0F);
    if (inPlace) {
        ASSERT_NO_FATAL_FAILURE(RunTiledReferenceInPlace(handle, p, a, b, c));
    } else {
        ASSERT_EQ(aclblasStrmm_npu(handle, side, uplo, trans, diag, m, n, &p.alpha,
                                  a.data(), p.lda, b.data(), p.ldb, c.data(), p.ldc),
                  ACLBLAS_STATUS_SUCCESS);
    }
    RunGoldenAndVerify(p, handle, &p.alpha, a.data(), b.data(), c.data());
    VerifyTiledOutputPadding(p, c, inPlace);
}

TEST_F(StrmmArch22Test, TinyTransposeBoundaries)
{
    for (int m : {3, 4, 5, 7, 8, 9}) {
        for (int n : {1, 2, 7, 8, 9, 31, 32, 33}) {
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                    for (int layout = 0; layout < 3; ++layout) {
                        CheckTiledReference(handle_, m, n, ACLBLAS_SIDE_LEFT, uplo, ACLBLAS_OP_T,
                            diag, layout == 0 ? 0 : 3, false, (m + n) % 2 == 0 ? 0.75F : -1.25F,
                            layout == 2);
                    }
                    if ((m == 4 || m == 8) && n == 8) {
                        CheckTiledReference(handle_, m, n, ACLBLAS_SIDE_LEFT, uplo, ACLBLAS_OP_T,
                                            diag, 0, true, -0.75F);
                        CheckTiledReference(handle_, m, n, ACLBLAS_SIDE_LEFT, uplo, ACLBLAS_OP_T,
                                            diag, 3, true, -0.75F);
                    }
                }
            }
        }
    }
}

TEST_F(StrmmArch22Test, NarrowTransposeTilingBoundaries)
{
    for (int m : {64, 65, 96, 127, 128, 129}) {
        for (int n : {2559, 2560, 2561, 3259, 3840, 3841}) {
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                    CheckTiledReference(handle_, m, n, ACLBLAS_SIDE_LEFT, uplo, ACLBLAS_OP_T,
                                        diag, (m + n) % 2 == 0 ? 0 : 3);
                }
            }
        }
    }
}

TEST_F(StrmmArch22Test, PackedTriangleBoundaries)
{
    for (int dim : {33, 63, 65, 127, 128, 129, 4095, 4096, 4097}) {
        for (auto side : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T}) {
                    for (auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                        CheckTiledReference(handle_, side == ACLBLAS_SIDE_LEFT ? dim : 7,
                                            side == ACLBLAS_SIDE_LEFT ? 7 : dim,
                                            side, uplo, trans, diag, 3);
                    }
                }
            }
        }
    }
}

TEST_F(StrmmArch22Test, RightAlignedInputBoundaries)
{
    for (int m : {511, 512, 513, 1025, 2251, 4095, 4096, 4097}) {
        for (int n : {64, 65, 100, 128, 129}) {
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T}) {
                    for (auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                        CheckTiledReference(handle_, m, n, ACLBLAS_SIDE_RIGHT, uplo, trans, diag,
                            (m + n) % 2 == 0 ? 0 : 3, false, (m + n) % 3 == 0 ? -0.75F : 1.0F);
                    }
                }
            }
        }
    }
    for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T}) {
        CheckTiledReference(handle_, 2251, 100, ACLBLAS_SIDE_RIGHT, ACLBLAS_LOWER, trans,
                            ACLBLAS_NON_UNIT, 3, true, -0.75F);
    }
}

TEST_P(StrmmArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    const int aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;

    std::vector<float> aHost, bHost;
    const float* alphaPtr = nullptr;
    const float* aPtr = nullptr;
    const float* bPtr = nullptr;
    bool alphaOnDevice = false;
    PrepareTestData(p, aDim, aHost, bHost, alphaPtr, aPtr, bPtr, alphaOnDevice);

    if (p.description == "null_handle") {
        float dummy = 0.0f;
        aclblasStatus_t ret = aclblasStrmm(nullptr, p.side, p.uplo, p.trans, p.diag,
                                           p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb,
                                           &dummy, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    if (p.m <= 0 || p.n <= 0 || p.lda <= 0 || p.ldb <= 0 || p.ldc <= 0) {
        float dummy = 0.0f;
        aclblasStatus_t ret = aclblasStrmm_npu(
            StrmmArch22Test::handle_,
            p.side, p.uplo, p.trans, p.diag, p.m, p.n,
            alphaPtr, aPtr, p.lda, bPtr, p.ldb, &dummy, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    std::vector<float> cHost(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n), 0.0f);
    float* cPtr = p.nullC ? nullptr : cHost.data();

    if (p.alpha == 0.0f && !p.nullAlpha && (p.nullA || p.nullB)) {
        HandleAlphaZeroNullAB(p, StrmmArch22Test::handle_, alphaPtr, aPtr, bPtr, cPtr, alphaOnDevice);
        return;
    }

    aclblasStatus_t ret = aclblasStrmm_npu(
        StrmmArch22Test::handle_,
        p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, cPtr, p.ldc,
        alphaOnDevice);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    RunGoldenAndVerify(p, StrmmArch22Test::handle_, alphaPtr, aPtr, bPtr, cPtr);
}
