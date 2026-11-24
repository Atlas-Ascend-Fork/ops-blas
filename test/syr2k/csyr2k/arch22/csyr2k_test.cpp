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
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"
#include "csyr2k_param.h"
#include "csyr2k_golden.h"
#include "csyr2k_npu_wrapper.h"

// arch22 (Atlas A2/A3) unit tests for aclblasCsyr2k.
// Beyond the CSV grid the cases below target the three places where the arch22
// three-stage pipeline can silently go wrong:
//   1. the K-chunk loop: exceeding the kernel's compile-time singleK is not
//      diagnosed at runtime, it silently mis-computes, so a shape past that
//      boundary must be covered;
//   2. the packed-plane addressing: the packed row stride is the stored row
//      count, so a padded leading dimension must not shift any logical read;
//   3. the Phase 2 combine: alpha = 0 / k = 0 skips the temp planes entirely and
//      beta must then be the only contributor.
class Csyr2kTest : public BlasTest<Csyr2kParam> {};

namespace {
constexpr int kSingleK = 8192; // cblas3::SINGLE_K, mirrored by csyr2k_gemm_single_k()

aclblasComplex MakeComplex(float re, float im) { return aclblasComplex{re, im}; }

// aclblasComplex is a plain struct without operator==, so gtest cannot compare
// vectors of it directly; compare the two components instead.
void ExpectComplexVectorEq(
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& expected, const std::string& what)
{
    ASSERT_EQ(actual.size(), expected.size()) << what;
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_EQ(actual[i].real, expected[i].real) << what << " index=" << i;
        ASSERT_EQ(actual[i].imag, expected[i].imag) << what << " index=" << i;
    }
}

int Csyr2kMatrixRows(const Csyr2kParam& p) { return (p.trans == ACLBLAS_OP_N) ? p.n : p.k; }

int Csyr2kMatrixCols(const Csyr2kParam& p) { return (p.trans == ACLBLAS_OP_N) ? p.k : p.n; }

// Splits the output into the four per-component uplo / non-uplo vectors so the
// real and imaginary parts can be checked with their own tolerances, and so the
// unselected triangle can be checked for exact preservation.
void CollectCsyr2kComponents(
    const Csyr2kParam& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& reference,
    std::vector<float>& uploRe, std::vector<float>& uploIm, std::vector<float>& refRe, std::vector<float>& refIm,
    std::vector<float>& otherRe, std::vector<float>& otherIm, std::vector<float>& otherRefRe,
    std::vector<float>& otherRefIm)
{
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.n; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldc + row;
            const bool selected = p.uplo == ACLBLAS_UPPER ? row <= col : row >= col;
            if (selected) {
                uploRe.push_back(actual[offset].real);
                uploIm.push_back(actual[offset].imag);
                refRe.push_back(reference[offset].real);
                refIm.push_back(reference[offset].imag);
            } else {
                otherRe.push_back(actual[offset].real);
                otherIm.push_back(actual[offset].imag);
                otherRefRe.push_back(reference[offset].real);
                otherRefIm.push_back(reference[offset].imag);
            }
        }
    }
}

void VerifyCsyr2kAgainstGolden(
    const Csyr2kParam& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden,
    const std::vector<aclblasComplex>& input)
{
    std::vector<float> uploRe;
    std::vector<float> uploIm;
    std::vector<float> goldRe;
    std::vector<float> goldIm;
    std::vector<float> otherRe;
    std::vector<float> otherIm;
    std::vector<float> otherGoldRe;
    std::vector<float> otherGoldIm;
    CollectCsyr2kComponents(
        p, actual, golden, uploRe, uploIm, goldRe, goldIm, otherRe, otherIm, otherGoldRe, otherGoldIm);

    // The real part is T1 - T2, where each product is O(k) in magnitude while their
    // difference is only O(sqrt(k)); combined with the host-side K chunking at the
    // kernel's compile-time singleK, the fp32 accumulation error grows with k. The
    // fixed 1e-2 absolute limit holds up to roughly k = 5000; beyond that it is
    // scaled the same way the arch35 cherk test scales its own limit, with a
    // coefficient fitted to the measured error at k = 8321 (1.05e-2).
    VerifyConfig cfgRe;
    applyMixedTolerance(cfgRe, ACL_FLOAT, goldRe.data(), goldRe.size());
    cfgRe.mixedMaxAbsErrorLimit = std::max(cfgRe.mixedMaxAbsErrorLimit, 2.0e-6 * static_cast<double>(p.k));
    EXPECT_TRUE(Verifier::verifyVector(uploRe.data(), goldRe.data(), uploRe.size(), 1, cfgRe, p.caseName + "_real"));
    VerifyConfig cfgIm;
    applyMixedTolerance(cfgIm, ACL_FLOAT, goldIm.data(), goldIm.size());
    cfgIm.mixedMaxAbsErrorLimit = std::max(cfgIm.mixedMaxAbsErrorLimit, 2.0e-6 * static_cast<double>(p.k));
    EXPECT_TRUE(Verifier::verifyVector(uploIm.data(), goldIm.data(), uploIm.size(), 1, cfgIm, p.caseName + "_imag"));

    // The unselected triangle carries the "value unchanged" contract: NaN and
    // Inf must stay exactly where they were and must not spread.
    if (!otherRe.empty()) {
        VerifyConfig cfgOther;
        cfgOther.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(
            otherRe.data(), otherGoldRe.data(), otherRe.size(), 1, cfgOther, p.caseName + "_other_real"));
        EXPECT_TRUE(Verifier::verifyVector(
            otherIm.data(), otherGoldIm.data(), otherIm.size(), 1, cfgOther, p.caseName + "_other_imag"));
    }
    // Leading-dimension padding must not be touched either.
    for (int col = 0; col < p.n; ++col) {
        for (int row = p.n; row < p.ldc; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldc + row;
            ASSERT_EQ(actual[offset].real, input[offset].real) << p.caseName << " padding row=" << row;
            ASSERT_EQ(actual[offset].imag, input[offset].imag) << p.caseName << " padding row=" << row;
        }
    }
}

// Runs one shape end to end: fills A/B/C, computes the CPU golden into a copy,
// calls the NPU entry, and compares. `extra` mutates the parameter before the
// fill so a case can point at a different alpha/beta/ld triple.
bool RunCsyr2kShape(aclblasHandle_t handle, const Csyr2kParam& base, std::function<void(Csyr2kParam&)> extra = nullptr)
{
    Csyr2kParam p = base;
    if (extra != nullptr) {
        extra(p);
    }
    const int rows = Csyr2kMatrixRows(p);
    const int cols = Csyr2kMatrixCols(p);
    if (rows <= 0 || cols <= 0 || p.n <= 0) {
        return false;
    }
    std::vector<aclblasComplex> a = makeBlasComplexMatrix(rows, cols, p.lda, p.fillA, p.randomSeed);
    std::vector<aclblasComplex> b = makeBlasComplexMatrix(rows, cols, p.ldb, p.fillB, p.randomSeed + 1U);
    std::vector<aclblasComplex> c = makeBlasComplexMatrix(p.n, p.n, p.ldc, p.fillC, p.randomSeed + 2U);
    std::vector<aclblasComplex> input = c;
    std::vector<aclblasComplex> golden = c;

    const aclblasStatus_t npuRet = aclblasCsyr2k_npu(
        handle, p.uplo, p.trans, p.n, p.k, &p.alpha, a.data(), p.lda, b.data(), p.ldb, &p.beta, c.data(), p.ldc);
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << p.caseName;
    if (npuRet != ACLBLAS_STATUS_SUCCESS) {
        return false;
    }
    const aclblasStatus_t goldenRet = aclblasCsyr2k_cpu(
        handle, p.uplo, p.trans, p.n, p.k, &p.alpha, a.data(), p.lda, b.data(), p.ldb, &p.beta, golden.data(), p.ldc);
    EXPECT_EQ(static_cast<int>(goldenRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << p.caseName;
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        return false;
    }
    VerifyCsyr2kAgainstGolden(p, c, golden, input);
    return true;
}

Csyr2kParam MakeCsyr2kCase(
    const std::string& name, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, uint32_t seed)
{
    Csyr2kParam p(csv_map{});
    p.caseName = name;
    p.uplo = uplo;
    p.trans = trans;
    p.n = n;
    p.k = k;
    p.alpha = MakeComplex(1.0F, 0.0F);
    p.beta = MakeComplex(0.0F, 0.0F);
    p.lda = std::max(1, (trans == ACLBLAS_OP_N) ? n : k);
    p.ldb = p.lda;
    p.ldc = std::max(1, n);
    p.randomSeed = seed;
    return p;
}
} // namespace

TEST_F(Csyr2kTest, NullHandle)
{
    aclblasComplex alpha = MakeComplex(1.0F, 0.0F);
    aclblasComplex beta = MakeComplex(0.0F, 0.0F);
    std::vector<aclblasComplex> data(16, MakeComplex(0.0F, 0.0F));
    EXPECT_EQ(
        aclblasCsyr2k(
            nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, data.data(), 4, data.data(), 4, &beta, data.data(), 4),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// Validation must run before the n == 0 quick return, so an illegal enum or a
// null scalar is an error even for an empty problem.
TEST_F(Csyr2kTest, ValidationPrecedesZeroDimQuickReturn)
{
    aclblasComplex scalar = MakeComplex(1.0F, 0.0F);
    std::vector<aclblasComplex> data(4, MakeComplex(0.0F, 0.0F));
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, static_cast<aclblasFillMode_t>(999), ACLBLAS_OP_N, 0, 4, &scalar, data.data(), 4, data.data(), 4,
            &scalar, data.data(), 4),
        ACLBLAS_STATUS_INVALID_ENUM);
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, static_cast<aclblasOperation_t>(999), 0, 4, &scalar, data.data(), 4, data.data(), 4,
            &scalar, data.data(), 4),
        ACLBLAS_STATUS_INVALID_ENUM);
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 4, nullptr, data.data(), 4, data.data(), 4, &scalar, data.data(),
            4),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 4, &scalar, data.data(), 4, data.data(), 4, nullptr, data.data(),
            4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(Csyr2kTest, InvalidDimsAndLeadingDimensions)
{
    aclblasComplex alpha = MakeComplex(1.0F, 0.0F);
    aclblasComplex beta = MakeComplex(0.0F, 0.0F);
    std::vector<aclblasComplex> data(64, MakeComplex(0.0F, 0.0F));
    // Negative dimensions.
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, -1, 4, &alpha, data.data(), 4, data.data(), 4, &beta, data.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, -1, &alpha, data.data(), 4, data.data(), 4, &beta, data.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    // lda / ldb below the trans-dependent minimum.
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 8, 4, &alpha, data.data(), 7, data.data(), 8, &beta, data.data(), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_T, 8, 4, &alpha, data.data(), 4, data.data(), 4, &beta, data.data(), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
    // ldc below max(1, n).
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 8, 4, &alpha, data.data(), 8, data.data(), 8, &beta, data.data(), 7),
        ACLBLAS_STATUS_INVALID_VALUE);
    // Null matrix pointers while k > 0.
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 8, 4, &alpha, nullptr, 8, data.data(), 8, &beta, data.data(), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 8, 4, &alpha, data.data(), 8, nullptr, 8, &beta, data.data(), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 8, 4, &alpha, data.data(), 8, data.data(), 8, &beta, nullptr, 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// n == 0 is a legal no-op: nothing is read or written, and null pointers are
// acceptable there.
TEST_F(Csyr2kTest, ZeroDimIsNoOp)
{
    aclblasComplex alpha = MakeComplex(1.0F, 0.0F);
    aclblasComplex beta = MakeComplex(0.0F, 0.0F);
    std::vector<aclblasComplex> data(4, MakeComplex(3.0F, -3.0F));
    const std::vector<aclblasComplex> before = data;
    EXPECT_EQ(
        aclblasCsyr2k(
            handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 4, &alpha, nullptr, 4, nullptr, 4, &beta, data.data(), 4),
        ACLBLAS_STATUS_SUCCESS);
    ExpectComplexVectorEq(data, before, "zero_dim_no_op");
}

// alpha = (0,0) or k = 0 skips the temp planes entirely, so beta is the only
// contributor; C is symmetric, so the diagonal imaginary part survives.
TEST_F(Csyr2kTest, AlphaZeroAndZeroKUseBetaOnly)
{
    ASSERT_TRUE(RunCsyr2kShape(
        handle_, MakeCsyr2kCase("alpha_zero", ACLBLAS_UPPER, ACLBLAS_OP_N, 8, 4, 20261001U),
        [](Csyr2kParam& p) { p.alpha = MakeComplex(0.0F, 0.0F); }));
    ASSERT_TRUE(RunCsyr2kShape(
        handle_, MakeCsyr2kCase("alpha_zero_complex_beta", ACLBLAS_LOWER, ACLBLAS_OP_T, 10, 6, 20261002U),
        [](Csyr2kParam& p) {
            p.alpha = MakeComplex(0.0F, 0.0F);
            p.beta = MakeComplex(-0.25F, 0.5F);
        }));
    ASSERT_TRUE(RunCsyr2kShape(
        handle_, MakeCsyr2kCase("k_zero", ACLBLAS_UPPER, ACLBLAS_OP_N, 8, 0, 20261003U),
        [](Csyr2kParam& p) { p.beta = MakeComplex(1.5F, -0.5F); }));
}

// With alpha = 0 and beta = 1 nothing contributes at all, so no kernel is
// launched and C must come back bit-identical.
TEST_F(Csyr2kTest, AlphaZeroBetaOneIsBitExactNoOp)
{
    Csyr2kParam p = MakeCsyr2kCase("alpha_zero_beta_one", ACLBLAS_UPPER, ACLBLAS_OP_N, 8, 4, 20261004U);
    p.alpha = MakeComplex(0.0F, 0.0F);
    p.beta = MakeComplex(1.0F, 0.0F);
    std::vector<aclblasComplex> a = makeBlasComplexMatrix(8, 4, 8, p.fillA, p.randomSeed);
    std::vector<aclblasComplex> b = makeBlasComplexMatrix(8, 4, 8, p.fillB, p.randomSeed + 1U);
    std::vector<aclblasComplex> c = makeBlasComplexMatrix(8, 8, 8, p.fillC, p.randomSeed + 2U);
    const std::vector<aclblasComplex> before = c;
    ASSERT_EQ(
        aclblasCsyr2k_npu(
            handle_, p.uplo, p.trans, p.n, p.k, &p.alpha, a.data(), p.lda, b.data(), p.ldb, &p.beta, c.data(), p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    ExpectComplexVectorEq(c, before, "alpha_zero_beta_one");
}

// Host-side alpha/beta must be dereferenced rather than run through a failed
// device-to-host copy.
namespace {
// Uploads A/B/C, invokes the public API with host-side alpha/beta addresses, and
// downloads C back. Returns false on any device API failure.
bool RunCsyr2kHostScalarOnDevice(
    aclblasHandle_t handle, aclrtStream stream, int n, int k, const aclblasComplex& alpha, const aclblasComplex& beta,
    const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b, std::vector<aclblasComplex>& c)
{
    const int ld = n;
    const size_t bytes = c.size() * sizeof(aclblasComplex);
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    if (aclrtMalloc(&dA, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dB, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dC, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dB);
        aclrtFree(dC);
        return false;
    }
    bool ok = aclrtMemcpy(dA, bytes, a.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    ok = ok && aclrtMemcpy(dB, bytes, b.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    ok = ok && aclrtMemcpy(dC, bytes, c.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    if (ok) {
        const aclblasStatus_t callRet = aclblasCsyr2k(
            handle, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, static_cast<const aclblasComplex*>(dA), ld,
            static_cast<const aclblasComplex*>(dB), ld, &beta, static_cast<aclblasComplex*>(dC), ld);
        ok = callRet == ACLBLAS_STATUS_SUCCESS;
    }
    if (ok) {
        ok = aclrtSynchronizeDevice() == ACL_SUCCESS;
    }
    if (ok) {
        ok = aclrtMemcpy(c.data(), bytes, dC, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    }
    aclrtFree(dA);
    aclrtFree(dB);
    aclrtFree(dC);
    return ok;
}

// Runs one host-scalar shape end to end and checks it against the CPU golden.
bool RunCsyr2kHostScalarShape(
    aclblasHandle_t handle, aclrtStream stream, int n, int k, const aclblasComplex& alpha, const aclblasComplex& beta,
    uint32_t seed)
{
    const int ld = n;
    std::vector<aclblasComplex> a = makeBlasComplexMatrix(n, k, ld, BlasFillMode("RANDOM_NORM_5_5"), seed);
    std::vector<aclblasComplex> b = makeBlasComplexMatrix(n, k, ld, BlasFillMode("RANDOM_NORM_5_5"), seed + 1U);
    std::vector<aclblasComplex> c = makeBlasComplexMatrix(n, n, ld, BlasFillMode("RANDOM_NORM_5_5"), seed + 2U);
    const std::vector<aclblasComplex> input = c;
    std::vector<aclblasComplex> golden = c;

    if (!RunCsyr2kHostScalarOnDevice(handle, stream, n, k, alpha, beta, a, b, c)) {
        return false;
    }
    const aclblasStatus_t goldenRet = aclblasCsyr2k_cpu(
        handle, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, a.data(), ld, b.data(), ld, &beta, golden.data(), ld);
    EXPECT_EQ(static_cast<int>(goldenRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        return false;
    }
    Csyr2kParam verify = MakeCsyr2kCase("host_scalar", ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, seed);
    VerifyCsyr2kAgainstGolden(verify, c, golden, input);
    return true;
}

TEST_F(Csyr2kTest, HostScalarAlphaBeta)
{
    const aclblasComplex alpha = MakeComplex(2.0F, -1.0F);
    const aclblasComplex beta = MakeComplex(0.5F, 0.25F);
    ASSERT_TRUE(RunCsyr2kHostScalarShape(handle_, stream_, 6, 5, alpha, beta, 20261005U));
    ASSERT_TRUE(RunCsyr2kHostScalarShape(handle_, stream_, 9, 7, alpha, beta, 20261006U));
}

// A reduction longer than the kernel's compile-time singleK must be split by the
// host; exceeding it is not diagnosed at runtime, it silently mis-computes.
TEST_F(Csyr2kTest, ReductionPastSingleKBoundary)
{
    const int n = 64;
    ASSERT_TRUE(
        RunCsyr2kShape(handle_, MakeCsyr2kCase("k_at_single_k", ACLBLAS_UPPER, ACLBLAS_OP_N, n, kSingleK, 20261008U)));
    ASSERT_TRUE(RunCsyr2kShape(
        handle_, MakeCsyr2kCase("k_past_single_k", ACLBLAS_LOWER, ACLBLAS_OP_N, n, kSingleK + 37, 20261009U)));
    ASSERT_TRUE(RunCsyr2kShape(
        handle_, MakeCsyr2kCase("k_past_single_k_trans", ACLBLAS_UPPER, ACLBLAS_OP_T, n, kSingleK + 64, 20261010U),
        [](Csyr2kParam& p) { p.alpha = MakeComplex(0.5F, -0.5F); }));
}

// Padded leading dimensions: the packed row stride is the stored row count, so a
// larger lda/ldb/ldc must not shift any logical read or write, and the padding
// itself must stay untouched.
TEST_F(Csyr2kTest, PaddedLeadingDimensions)
{
    ASSERT_TRUE(RunCsyr2kShape(
        handle_, MakeCsyr2kCase("padded_n", ACLBLAS_UPPER, ACLBLAS_OP_N, 40, 24, 20261011U), [](Csyr2kParam& p) {
            p.lda = 48;
            p.ldb = 53;
            p.ldc = 47;
        }));
    ASSERT_TRUE(RunCsyr2kShape(
        handle_, MakeCsyr2kCase("padded_t", ACLBLAS_LOWER, ACLBLAS_OP_T, 33, 41, 20261012U), [](Csyr2kParam& p) {
            p.lda = 45;
            p.ldb = 39;
            p.ldc = 36;
        }));
}

// The non-uplo triangle keeps its input value, including non-finite entries:
// the arch22 pipeline must never read or write outside the requested triangle.
TEST_F(Csyr2kTest, NonUploTriangleIsPreserved)
{
    Csyr2kParam p = MakeCsyr2kCase("non_uplo_preserved", ACLBLAS_UPPER, ACLBLAS_OP_N, 12, 6, 20261013U);
    const float quietNaN = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    std::vector<aclblasComplex> a = makeBlasComplexMatrix(12, 6, 12, p.fillA, p.randomSeed);
    std::vector<aclblasComplex> b = makeBlasComplexMatrix(12, 6, 12, p.fillB, p.randomSeed + 1U);
    std::vector<aclblasComplex> c = makeBlasComplexMatrix(12, 12, 12, p.fillC, p.randomSeed + 2U);
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.n; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldc + row;
            if (row > col) {
                c[offset] = MakeComplex(quietNaN, infinity);
            }
        }
    }
    const std::vector<aclblasComplex> input = c;
    std::vector<aclblasComplex> golden = c;

    ASSERT_EQ(
        aclblasCsyr2k_npu(
            handle_, p.uplo, p.trans, p.n, p.k, &p.alpha, a.data(), p.lda, b.data(), p.ldb, &p.beta, c.data(), p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCsyr2k_cpu(
            handle_, p.uplo, p.trans, p.n, p.k, &p.alpha, a.data(), p.lda, b.data(), p.ldb, &p.beta, golden.data(),
            p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    for (int col = 0; col < p.n; ++col) {
        for (int row = col + 1; row < p.n; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldc + row;
            ASSERT_TRUE(std::isnan(c[offset].real)) << "row=" << row << " col=" << col;
            ASSERT_TRUE(std::isinf(c[offset].imag)) << "row=" << row << " col=" << col;
            ASSERT_EQ(c[offset].real, input[offset].real);
            ASSERT_EQ(c[offset].imag, input[offset].imag);
        }
    }
    VerifyCsyr2kAgainstGolden(p, c, golden, input);
}

} // namespace
INSTANTIATE_TEST_SUITE_P(
    Csyr2kArch22, Csyr2kTest, ::testing::ValuesIn(GetCasesFromCsv<Csyr2kParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Csyr2kParam>);

TEST_P(Csyr2kTest, CsvDriven)
{
    const auto& p = GetParam();
    const int rows = Csyr2kMatrixRows(p);
    const int cols = Csyr2kMatrixCols(p);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || rows <= 0 || cols <= 0 || p.n <= 0) {
        // Invalid-parameter rows only assert the status code.
        // An empty dimension is not an error for an invalid-parameter row: the
        // operator only has to return the expected status, so hand it a null pointer.
        auto makeMatrix = [](int rows, int cols, int ld, const BlasFillMode& fill, uint32_t seed) {
            if (rows <= 0 || cols <= 0 || ld <= 0) {
                return std::vector<aclblasComplex>{};
            }
            return makeBlasComplexMatrix(rows, cols, ld, fill, seed);
        };
        std::vector<aclblasComplex> a = makeMatrix(rows, cols, p.lda, p.fillA, p.randomSeed);
        std::vector<aclblasComplex> b = makeMatrix(rows, cols, p.ldb, p.fillB, p.randomSeed + 1U);
        std::vector<aclblasComplex> c = makeMatrix(p.n, p.n, p.ldc, p.fillC, p.randomSeed + 2U);
        // The CSV marks a null argument through its own column (alpha_real == "null"
        // for the scalars, nullA/nullB/nullC for the matrices), so honour it here:
        // otherwise an "expect INVALID_VALUE" row would hand the operator a valid
        // pointer and get SUCCESS back.
        const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
        const aclblasComplex* betaPtr = p.nullBeta ? nullptr : &p.beta;
        const aclblasComplex* aPtr = p.nullA ? nullptr : a.data();
        const aclblasComplex* bPtr = p.nullB ? nullptr : b.data();
        aclblasComplex* cPtr = p.nullC ? nullptr : c.data();
        const aclblasStatus_t ret = aclblasCsyr2k_npu(
            handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    ASSERT_TRUE(RunCsyr2kShape(handle_, p)) << p.caseName;
}
