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
#include <cstdint>
#include <limits>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <future>
#include <cctype>
#include <cblas.h>
#include <dlfcn.h>
#include "acl/acl.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cblas_compat.h"

#include "cgemm_strided_batched_golden.h"
#include "cgemm_strided_batched_param.h"
#include "cgemm_strided_batched_npu_wrapper.h"

class CgemmSbArch22Test : public BlasTest<CgemmSbParam> {};

INSTANTIATE_TEST_SUITE_P(
    CgemmStridedBatched, CgemmSbArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<CgemmSbParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemmSbParam>);

namespace {
void FillInputs(
    const CgemmSbParam& p, size_t aCount, size_t bCount, size_t cCount, bool negative, bool quick, bool checkZeroK,
    std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
{
    auto makeA = [&] {
        return (!negative && !quick && !p.aNull) ? MakeData(aCount, p.aFill, p.randomSeed) : std::vector<float>{};
    };
    auto makeB = [&] {
        return (!negative && !quick && !p.bNull) ? MakeData(bCount, p.bFill, p.randomSeed + 1) : std::vector<float>{};
    };
    auto makeC = [&] {
        return (!negative && (!quick || checkZeroK) && !p.cNull) ? MakeData(cCount, p.cFill, p.randomSeed + 2) :
                                                                   std::vector<float>{};
    };
    if (!negative && aCount + bCount + cCount >= 1048576U) {
        auto fa = std::async(std::launch::async, makeA);
        auto fb = std::async(std::launch::async, makeB);
        auto fc = std::async(std::launch::async, makeC);
        a = fa.get();
        b = fb.get();
        c = fc.get();
    } else {
        a = makeA();
        b = makeB();
        c = makeC();
    }
}

void CheckOutputGuards(const CgemmSbParam& p, void* dc, size_t cSize, aclrtStream stream)
{
    if (dc && p.cOffset != 0U) {
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        std::vector<float> guarded(cSize + 2U * p.cOffset + 16U);
        ASSERT_EQ(
            aclrtMemcpy(
                guarded.data(), guarded.size() * sizeof(float), dc, guarded.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        for (size_t i = 0; i < 2U * p.cOffset; ++i) {
            EXPECT_EQ(guarded[i], 123.25f);
        }
        for (size_t i = 2U * p.cOffset + cSize; i < guarded.size(); ++i) {
            EXPECT_EQ(guarded[i], 123.25f);
        }
    }
}
aclblasStatus_t InvokeCase(const CgemmSbParam& p, const CgemmDeviceBuffers& buffers, aclblasHandle_t handle_)
{
    void* da = buffers.a;
    void* db = buffers.b;
    void* dc = buffers.c;
    aclblasHandle_t handle = p.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR ? nullptr : handle_;
    const aclblasComplex* pa = p.alphaNull ? nullptr : &p.alpha;
    const aclblasComplex* pb = p.betaNull ? nullptr : &p.beta;
    const auto* A = p.aNull ? nullptr : (da ? static_cast<const aclblasComplex*>(da) + p.aOffset : nullptr);
    const auto* B = p.bNull ? nullptr : (db ? static_cast<const aclblasComplex*>(db) + p.bOffset : nullptr);
    auto* C = p.cNull ? nullptr : (dc ? static_cast<aclblasComplex*>(dc) + p.cOffset : nullptr);
    aclblasStatus_t status = aclblasCgemmStridedBatched(
        handle, p.transA, p.transB, p.m, p.n, p.k, pa, A, p.lda, p.strideA, B, p.ldb, p.strideB, pb, C, p.ldc,
        p.strideC, p.batchCount);
    return status;
}

} // namespace

TEST_P(CgemmSbArch22Test, CsvDriven)
{
    (void)GetNetlibReference();
    const auto& p = GetParam();
    const int aCols = p.transA == ACLBLAS_OP_N ? p.k : p.m;
    const int bCols = p.transB == ACLBLAS_OP_N ? p.n : p.k;
    const size_t aCount = Span(p.strideA, static_cast<int64_t>(p.lda) * aCols, p.batchCount);
    const size_t bCount = Span(p.strideB, static_cast<int64_t>(p.ldb) * bCols, p.batchCount);
    const size_t cCount = Span(p.strideC, static_cast<int64_t>(p.ldc) * p.n, p.batchCount);
    // Negative cases are validated before any device address is used.
    const bool negative = p.expectResult != ACLBLAS_STATUS_SUCCESS;
    const bool quick = p.m <= 0 || p.n <= 0 || p.k <= 0 || p.batchCount <= 0;
    const bool checkZeroK = !negative && p.m > 0 && p.n > 0 && p.k == 0 && p.batchCount > 0 && !p.cNull;
    std::vector<float> a, b, c;
    FillInputs(p, aCount, bCount, cCount, negative, quick, checkZeroK, a, b, c);
    std::vector<float> golden = c;
    std::cout << std::setprecision(10) << "INPUT_SCALARS " << p.caseName << " alpha=" << p.alpha.real << ","
              << p.alpha.imag << " beta=" << p.beta.real << "," << p.beta.imag << '\n';
    CgemmDeviceBuffers buffers;
    void*& da = buffers.a;
    void*& db = buffers.b;
    void*& dc = buffers.c;
    if (!negative && (!quick || checkZeroK)) {
        ASSERT_EQ(CopyToDevice(&da, a, p.aOffset), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(CopyToDevice(&db, b, p.bOffset), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(CopyToDevice(&dc, c, p.cOffset), ACLBLAS_STATUS_SUCCESS);
    }
    auto* C = p.cNull ? nullptr : (dc ? static_cast<aclblasComplex*>(dc) + p.cOffset : nullptr);
    const aclblasStatus_t status = InvokeCase(p, buffers, handle_);
    std::cout << "STATUS_RESULT " << p.caseName << " actual=" << static_cast<int>(status)
              << " expected=" << static_cast<int>(p.expectResult) << '\n';
    EXPECT_EQ(status, p.expectResult) << p.caseName;
    if (status == ACLBLAS_STATUS_SUCCESS && checkZeroK) {
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(c.data(), c.size() * sizeof(float), C, c.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        EXPECT_EQ(c, golden) << "k=0 must leave C unchanged, including beta=0";
    }
    if (status == ACLBLAS_STATUS_SUCCESS && !quick) {
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(c.data(), c.size() * sizeof(float), C, c.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        GenerateGolden(p, a, b, golden);
        CheckOutput(p, c, golden, cCount);
    }
    ASSERT_NO_FATAL_FAILURE(CheckOutputGuards(p, dc, c.size(), stream_));
}

// Exercise DMA byte strides beyond UINT32_MAX without allocating a huge Host mirror.
class CgemmSbWideStrideTest : public BlasTest<int> {};
INSTANTIATE_TEST_SUITE_P(CgemmStridedBatched, CgemmSbWideStrideTest, ::testing::Values(0, 1, 2));

namespace {
std::vector<aclblasComplex> WideStrideGolden(int k)
{
    // Pack only the logical columns on Host; use the same Netlib reference as the task CSV.
    std::vector<aclblasComplex> a(2U * k), b(2U * k), c(4U);
    for (int i = 0; i < k; ++i) {
        a[i] = {1.0f, 0.0f};
        a[k + i] = {2.0f, 0.0f};
        b[i] = {1.0f, 0.0f};
        b[k + i] = {3.0f, 0.0f};
    }
    const aclblasComplex alpha{1.0f, 0.0f}, beta{0.0f, 0.0f};
    GetNetlibReference().gemm(
        CblasColMajor, CblasTrans, CblasNoTrans, 2, 2, k, &alpha, a.data(), k, b.data(), k, &beta, c.data(), 2);
    return c;
}

void AllocateTwoColumns(void** dst, int rows, int ld, float first, float second)
{
    const size_t bytes = (static_cast<size_t>(ld) + rows) * sizeof(aclblasComplex);
    ASSERT_EQ(aclrtMalloc(dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    // Make incorrectly wrapped DMA reads deterministic, without touching the whole gap.
    ASSERT_EQ(aclrtMemset(*dst, bytes, 0, std::min<size_t>(bytes, 4096U)), ACL_SUCCESS);
    const size_t columnBytes = static_cast<size_t>(rows) * sizeof(aclblasComplex);
    for (int col = 0; col < 2; ++col) {
        std::vector<aclblasComplex> values(rows, aclblasComplex{col == 0 ? first : second, 0.0f});
        auto* column = static_cast<aclblasComplex*>(*dst) + static_cast<size_t>(col) * ld;
        ASSERT_EQ(aclrtMemcpy(column, columnBytes, values.data(), columnBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    }
}
} // namespace

TEST_P(CgemmSbWideStrideTest, LeadingDimensionBeyondDmaRange)
{
    // A is physically K x M and transposed. Each case needs just over 4 GiB of HBM.
    constexpr int m = 2, n = 2, k = 128;
    constexpr int wideLd = 536870976; // (wideLd - 64) * sizeof(COMPLEX64) == 2^32.
    const auto golden = WideStrideGolden(k);
    const int lda = GetParam() == 0 ? wideLd : k;
    const int ldb = GetParam() == 1 ? wideLd : k;
    const int ldc = GetParam() == 2 ? wideLd : m;
    CgemmDeviceBuffers buffers;
    ASSERT_NO_FATAL_FAILURE(AllocateTwoColumns(&buffers.a, k, lda, 1.0f, 2.0f));
    ASSERT_NO_FATAL_FAILURE(AllocateTwoColumns(&buffers.b, k, ldb, 1.0f, 3.0f));
    ASSERT_NO_FATAL_FAILURE(AllocateTwoColumns(&buffers.c, m, ldc, -17.0f, -19.0f));
    const aclblasComplex alpha{1.0f, 0.0f}, beta{0.0f, 0.0f};
    ASSERT_EQ(
        aclblasCgemmStridedBatched(
            handle_, ACLBLAS_OP_T, ACLBLAS_OP_N, m, n, k, &alpha, static_cast<const aclblasComplex*>(buffers.a), lda, 0,
            static_cast<const aclblasComplex*>(buffers.b), ldb, 0, &beta, static_cast<aclblasComplex*>(buffers.c), ldc,
            0, 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    for (int col = 0; col < n; ++col) {
        aclblasComplex actual[m]{};
        auto* source = static_cast<aclblasComplex*>(buffers.c) + static_cast<size_t>(col) * ldc;
        ASSERT_EQ(aclrtMemcpy(actual, sizeof(actual), source, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        for (int row = 0; row < m; ++row) {
            EXPECT_FLOAT_EQ(actual[row].real, golden[col * m + row].real);
            EXPECT_FLOAT_EQ(actual[row].imag, golden[col * m + row].imag);
        }
    }
}
