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
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "gemm_golden.h"
#include "gemm_param.h"
#include "../arch35/gemm_npu_wrapper.h"
#include "../../frame/npu_event_timer.h"

namespace {

struct SgemmCase {
    aclblasOperation_t transa;
    aclblasOperation_t transb;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    float alpha;
    float beta;
};

float Pattern(int i)
{
    return static_cast<float>((i * 17 + 5) % 23 - 11) * 0.125f;
}

void RunSgemmCase(aclblasHandle_t handle, const SgemmCase& tc)
{
    const int aCols = tc.transa == ACLBLAS_OP_N ? tc.k : tc.m;
    const int bCols = tc.transb == ACLBLAS_OP_N ? tc.n : tc.k;
    std::vector<float> a(static_cast<size_t>(tc.lda) * std::max(1, aCols));
    std::vector<float> b(static_cast<size_t>(tc.ldb) * std::max(1, bCols));
    std::vector<float> c(static_cast<size_t>(tc.ldc) * std::max(1, tc.n));
    for (size_t i = 0; i < a.size(); ++i) a[i] = Pattern(static_cast<int>(i));
    for (size_t i = 0; i < b.size(); ++i) b[i] = Pattern(static_cast<int>(i) + 101);
    for (size_t i = 0; i < c.size(); ++i) c[i] = Pattern(static_cast<int>(i) + 211);
    const std::vector<float> before = c;
    std::vector<float> golden = c;

    ASSERT_EQ(aclblasSgemm_cpu(handle, tc.transa, tc.transb, tc.m, tc.n, tc.k,
                               tc.alpha, a.data(), tc.lda, b.data(), tc.ldb,
                               tc.beta, golden.data(), tc.ldc), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSgemm_npu(handle, tc.transa, tc.transb, tc.m, tc.n, tc.k,
                               &tc.alpha, a.data(), tc.lda, b.data(), tc.ldb,
                               &tc.beta, c.data(), tc.ldc), ACLBLAS_STATUS_SUCCESS);

    for (int col = 0; col < tc.n; ++col) {
        for (int row = 0; row < tc.m; ++row) {
            const size_t pos = static_cast<size_t>(row) + static_cast<size_t>(col) * tc.ldc;
            EXPECT_NEAR(c[pos], golden[pos], 1.0e-4f) << "row=" << row << " col=" << col;
        }
        for (int row = tc.m; row < tc.ldc; ++row) {
            const size_t pos = static_cast<size_t>(row) + static_cast<size_t>(col) * tc.ldc;
            EXPECT_EQ(c[pos], before[pos]) << "padding overwritten at col=" << col;
        }
    }
}

void MeasureSgemmCase(aclblasHandle_t handle, aclrtStream stream, const SgemmCase& tc,
                      int warmup, int measurements, const std::string& caseName)
{
    const int aCols = tc.transa == ACLBLAS_OP_N ? tc.k : tc.m;
    const int bCols = tc.transb == ACLBLAS_OP_N ? tc.n : tc.k;
    const size_t aBytes = static_cast<size_t>(tc.lda) * std::max(1, aCols) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(tc.ldb) * std::max(1, bCols) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(tc.ldc) * std::max(1, tc.n) * sizeof(float);
    std::vector<float> a(aBytes / sizeof(float));
    std::vector<float> b(bBytes / sizeof(float));
    std::vector<float> c(cBytes / sizeof(float));
    for (size_t i = 0; i < a.size(); ++i) a[i] = Pattern(static_cast<int>(i));
    for (size_t i = 0; i < b.size(); ++i) b[i] = Pattern(static_cast<int>(i) + 101);
    for (size_t i = 0; i < c.size(); ++i) c[i] = Pattern(static_cast<int>(i) + 211);

    GemmDeviceBuf dev;
    void* dInitialC = nullptr;
    ASSERT_EQ(GemmAllocDevice(dev, tc.k, a.data(), b.data(), c.data(), aBytes, bBytes, cBytes),
              ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(GemmAllocAndCopy(&dInitialC, c.data(), cBytes), ACLBLAS_STATUS_SUCCESS);
    const float alpha = tc.alpha;
    const float beta = tc.beta;
    auto resetC = [&]() {
        return aclrtMemcpy(dev.dC, cBytes, dInitialC, cBytes, ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_SUCCESS;
    };
    auto launch = [&]() {
        return aclblasSgemm(handle, tc.transa, tc.transb, tc.m, tc.n, tc.k,
                            &alpha, static_cast<const float*>(dev.dA), tc.lda,
                            static_cast<const float*>(dev.dB), tc.ldb,
                            &beta, static_cast<float*>(dev.dC), tc.ldc) == ACLBLAS_STATUS_SUCCESS;
    };
    std::vector<double> timings;
    ASSERT_EQ(blas_test::MeasureNpuEvent(stream, warmup, measurements, resetC, launch, timings), 0);
    GemmFreeBuffers(dev.dA, dev.dB, dev.dC);
    aclrtFree(dInitialC);
    std::cout << std::fixed << std::setprecision(3)
              << "PERFORMANCE_SUMMARY_JSON={\"case\":\"" << caseName
              << "\",\"n\":" << tc.m
              << ",\"p50_us\":" << blas_test::MedianUs(timings)
              << ",\"avg_us\":" << blas_test::MeanUs(timings)
              << ",\"warmups\":" << warmup
              << ",\"measurements\":" << measurements
              << ",\"timing\":\"npu_event\"}\n";
}

void FillNoSplitPatterns(std::vector<float>& a, std::vector<float>& b, int m, int n, int k, int qM, int qN,
                         int qK)
{
    constexpr int panel = 1024;
    for (int p = 0; p < k; ++p) {
        const int kp = p / panel;
        for (int mp = 0; mp < qM; ++mp) {
            const float value = static_cast<float>(1 + mp + 2 * kp);
            std::fill_n(a.begin() + static_cast<size_t>(p) * m + mp * panel,
                        panel, value);
        }
    }
    for (int j = 0; j < n; ++j) {
        const int np = j / panel;
        for (int kp = 0; kp < qK; ++kp) {
            const float value = static_cast<float>(1 + np + 3 * kp);
            std::fill_n(b.begin() + static_cast<size_t>(j) * k + kp * panel,
                        panel, value);
        }
    }
}

void VerifyNoSplitPatternOutput(const std::vector<float>& c, int m, int qM, int qN, int qK)
{
    constexpr int panel = 1024;
    for (int np = 0; np < qN; ++np) {
        for (int mp = 0; mp < qM; ++mp) {
            float expected = 0.0f;
            for (int kp = 0; kp < qK; ++kp) {
                expected += static_cast<float>(panel * (1 + mp + 2 * kp) *
                                               (1 + np + 3 * kp));
            }
            for (int rowBlock = 0; rowBlock < 16; ++rowBlock) {
                for (const int localCol : {rowBlock * 64, rowBlock * 64 + 63}) {
                    for (const int localRow : {0, 1023}) {
                        const int row = mp * panel + localRow;
                        const int col = np * panel + localCol;
                        EXPECT_EQ(c[static_cast<size_t>(row) + static_cast<size_t>(col) * m],
                                  expected)
                            << "mp=" << mp << " np=" << np
                            << " localRow=" << localRow << " localCol=" << localCol;
                    }
                }
            }
        }
    }
}

void RunWholeGridNoSplitPatternCase(aclblasHandle_t handle, int m, int n, int k)
{
    constexpr int panel = 1024;
    ASSERT_EQ(m % panel, 0);
    ASSERT_EQ(n % panel, 0);
    ASSERT_EQ(k % panel, 0);
    const int qM = m / panel;
    const int qN = n / panel;
    const int qK = k / panel;
    std::vector<float> a(static_cast<size_t>(m) * k);
    std::vector<float> b(static_cast<size_t>(k) * n);
    std::vector<float> c(static_cast<size_t>(m) * n,
                         std::numeric_limits<float>::quiet_NaN());

    FillNoSplitPatterns(a, b, m, n, k, qM, qN, qK);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    ASSERT_EQ(aclblasSgemm_npu(handle, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
                               &alpha, a.data(), m, b.data(), k, &beta, c.data(), m),
              ACLBLAS_STATUS_SUCCESS);
    VerifyNoSplitPatternOutput(c, m, qM, qN, qK);
}

class Arch22SgemmTest : public BlasTest<int> {};

class Arch22SgemmCsvTest : public BlasTest<GemmParam> {};

TEST_P(Arch22SgemmCsvTest, CsvDriven)
{
    const auto& p = GetParam();
    ASSERT_EQ(p.func, "sgemm");
    RunSgemmCase(handle_, {p.transA, p.transB, p.m, p.n, p.k,
                           p.lda, p.ldb, p.ldc, p.alphaReal, p.betaReal});
    if (p.measure && blas_test::MeasureRequested()) {
        MeasureSgemmCase(handle_, stream_, {p.transA, p.transB, p.m, p.n, p.k,
                           p.lda, p.ldb, p.ldc, p.alphaReal, p.betaReal},
                         p.warmup, p.measurements, p.caseName);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Arch22, Arch22SgemmCsvTest,
    ::testing::ValuesIn(GetCasesFromCsv<GemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemmParam>);

TEST_F(Arch22SgemmTest, DenseNnProduct)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 128, 128, 128, 128, 128, 128, 1.0f, 0.0f});
}

TEST_F(Arch22SgemmTest, DirectOfficialTransposeMatmulAllPublicTransposeModes)
{
    // Compact, aligned inputs select the direct Matmul transpose path rather
    // than the vector pack fallback.
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_T, 128, 128, 128, 128, 128, 128, 0.5f, -0.75f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, 128, 128, 128, 128, 128, 128, -0.625f, 0.375f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, 128, 128, 128, 128, 128, 128, 0.875f, -0.125f});
}

TEST_F(Arch22SgemmTest, DirectOfficialTransposeMatmulFull512Tile)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 512, 512, 512, 512, 512, 512, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, 512, 512, 512, 512, 512, 512, 0.75f, -0.25f});
}

TEST_F(Arch22SgemmTest, DirectMatmulWritesCompactCAllTransposeModes)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_T, 128, 128, 128, 128, 128, 128, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, 128, 128, 128, 128, 128, 128, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, 128, 128, 128, 128, 128, 128, 1.0f, 0.0f});
}

TEST_F(Arch22SgemmTest, DirectMultiCubeRowOwnershipBoundaries)
{
    for (const int n : {48, 64, 80, 128, 144}) {
        RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 128, n, 128,
                               128, 128, 128, 1.0f, 0.0f});
    }
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, 128, 144, 128,
                           128, 128, 128, 1.0f, 0.0f});
    // transB=T slices columns from physical row-major [K][N]. The full N
    // passed through SetOrgShape must remain the stride between K rows.
    for (const int n : {80, 144}) {
        RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_T, 128, n, 128,
                               128, n, 128, 1.0f, 0.0f});
        RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, 128, n, 128,
                               128, n, 128, 1.0f, 0.0f});
    }
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 128, 144, 128,
                           128, 128, 128, 0.625f, -0.375f});
}

TEST_F(Arch22SgemmTest, DirectMultiCubeFullPanelSliceBoundaries)
{
    // The Cube row partition pads each panel to whole waves of 20 blocks.
    // pn = 976/1008/1024 gives rowsPerBlock = 49/51/52 with a short final
    // slice; cross every slice boundary in all four transpose modes.
    const aclblasOperation_t ops[] = {ACLBLAS_OP_N, ACLBLAS_OP_T};
    for (const int n : {976, 1008, 1024}) {
        for (const auto transa : ops) {
            for (const auto transb : ops) {
                const int ldb = transb == ACLBLAS_OP_N ? 256 : n;
                RunSgemmCase(handle_, {transa, transb, 256, n, 256,
                                       256, ldb, 256, 0.75f, -0.25f});
            }
        }
    }
    // Same boundaries through the packed path with padded leading dimensions.
    for (const int n : {976, 1008, 1024}) {
        RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_T, 256, n, 256,
                               272, n + 16, 272, 1.0f, 0.0f});
        RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, 256, n, 256,
                               272, 272, 272, 1.0f, 0.0f});
    }
}

TEST_F(Arch22SgemmTest, DirectTiled2dNnRectangularLegality)
{
    // The retained 2D path is selected when row-only ownership would expose
    // at most two 64-row Cube blocks. Exercise both selected boundaries and
    // adjacent shapes that must stay on the direct row path.
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 1024, 64, 1024,
                           1024, 1024, 1024, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 1024, 128, 512,
                           1024, 512, 1024, 0.75f, -0.25f});
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 512, 128, 1024,
                           512, 1024, 512, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 1024, 129, 1024,
                           1024, 1024, 1024, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 511, 128, 1024,
                           511, 1024, 511, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 1024, 128, 511,
                           1024, 511, 1024, 1.0f, 0.0f});
}

TEST_F(Arch22SgemmTest, DirectOutputBetaZeroIgnoresPoisonedC)
{
    constexpr int m = 128, n = 128, k = 128;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::vector<float> a(static_cast<size_t>(m) * k);
    std::vector<float> b(static_cast<size_t>(k) * n);
    std::vector<float> c(static_cast<size_t>(m) * n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> golden(static_cast<size_t>(m) * n, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = Pattern(static_cast<int>(i) + 601);
    for (size_t i = 0; i < b.size(); ++i) b[i] = Pattern(static_cast<int>(i) + 701);
    ASSERT_EQ(aclblasSgemm_cpu(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
                               alpha, a.data(), m, b.data(), k, beta, golden.data(), m),
              ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSgemm_npu(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
                               &alpha, a.data(), m, b.data(), k, &beta, c.data(), m),
              ACLBLAS_STATUS_SUCCESS);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_TRUE(std::isfinite(c[i]));
        EXPECT_NEAR(c[i], golden[i], 1.0e-4f) << "index=" << i;
    }
}

TEST_F(Arch22SgemmTest, DirectGuardDoesNotApplyToPublicOuterTiles)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 16, 1280, 8, 16, 8, 16, 1.0f, 0.0f});
}

TEST_F(Arch22SgemmTest, DenseOddNnProduct)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 67, 53, 71, 67, 71, 67, 1.0f, 0.0f});
}

TEST_F(Arch22SgemmTest, PackedTransposeAlphaBetaAndOutputPadding)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, 65, 49, 73, 80, 64, 72, 0.5f, -0.75f});
}

TEST_F(Arch22SgemmTest, PaddedAlignedTtUsesRawStridePacking)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, 128, 144, 128,
                           136, 152, 136, 0.75f, -0.25f});
}

TEST_F(Arch22SgemmTest, PaddedAlignedNtUsesRawStridePacking)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_T, 128, 144, 128,
                           136, 152, 136, 0.75f, -0.25f});
}

TEST_F(Arch22SgemmTest, PaddedAlignedTnUsesRawStridePacking)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, 128, 144, 128,
                           136, 136, 136, -0.625f, 0.375f});
}

TEST_F(Arch22SgemmTest, PackedNoTransposeInputPadding)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 67, 53, 71, 80, 88, 80, -1.25f, 0.25f});
}

TEST_F(Arch22SgemmTest, PackedNtInputAndOutputPadding)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_T, 17, 31, 19, 24, 40, 24, 0.75f, -0.5f});
}

TEST_F(Arch22SgemmTest, PackedTnInputAndOutputPadding)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, 31, 17, 19, 24, 32, 40, -0.625f, 0.375f});
}

TEST_F(Arch22SgemmTest, CorrectnessShapeStrideTransposeMatrix)
{
    struct Shape { int m, n, k; };
    const Shape shapes[] = {
        {1, 1, 1}, {7, 9, 5}, {15, 17, 7}, {16, 16, 8}, {17, 15, 9},
        {31, 33, 63}, {32, 48, 64}, {64, 80, 72}, {65, 79, 73}, {128, 144, 128},
    };
    const aclblasOperation_t ops[] = {ACLBLAS_OP_N, ACLBLAS_OP_T};
    int caseIndex = 0;
    for (const auto& shape : shapes) {
        for (const auto transa : ops) {
            for (const auto transb : ops) {
                for (const bool padded : {false, true}) {
                    const int ldaMin = transa == ACLBLAS_OP_N ? shape.m : shape.k;
                    const int ldbMin = transb == ACLBLAS_OP_N ? shape.k : shape.n;
                    const int lda = ldaMin + (padded ? 3 + caseIndex % 5 : 0);
                    const int ldb = ldbMin + (padded ? 5 + caseIndex % 7 : 0);
                    const int ldc = shape.m + (padded ? 7 + caseIndex % 3 : 0);
                    const float alpha = caseIndex % 3 == 0 ? 1.0f :
                                        (caseIndex % 3 == 1 ? 0.75f : -0.625f);
                    const float beta = caseIndex % 4 == 0 ? 0.0f :
                                       (caseIndex % 4 == 1 ? -0.25f : 0.375f);
                    SCOPED_TRACE(::testing::Message()
                        << "case=" << caseIndex << " transa=" << transa << " transb=" << transb
                        << " m=" << shape.m << " n=" << shape.n << " k=" << shape.k
                        << " lda=" << lda << " ldb=" << ldb << " ldc=" << ldc);
                    RunSgemmCase(handle_, {transa, transb, shape.m, shape.n, shape.k,
                                           lda, ldb, ldc, alpha, beta});
                    ++caseIndex;
                }
            }
        }
    }
    EXPECT_EQ(caseIndex, 80);
}

TEST_F(Arch22SgemmTest, CorrectnessDeterministicRandomizedDifferential)
{
    // Fixed LCG rather than std::random_device: every CI run executes exactly
    // the same 256 public BLAS contracts and failures are reproducible by case.
    uint32_t state = 0x6d2b79f5u;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    const int boundary[] = {1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 127, 128, 129};
    const aclblasOperation_t ops[] = {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C};
    const float alphas[] = {1.0f, 0.75f, -0.625f, 1.25f};
    const float betas[] = {0.0f, -0.25f, 0.375f, 1.0f};

    for (int caseIndex = 0; caseIndex < 256; ++caseIndex) {
        int m = 1 + static_cast<int>(next() % 129);
        int n = 1 + static_cast<int>(next() % 129);
        int k = 1 + static_cast<int>(next() % 129);
        // Force one dimension onto an alignment transition every fourth case.
        const int forced = boundary[(caseIndex / 4) % (sizeof(boundary) / sizeof(boundary[0]))];
        if (caseIndex % 4 == 0) m = forced;
        if (caseIndex % 4 == 1) n = forced;
        if (caseIndex % 4 == 2) k = forced;

        const auto transa = ops[next() % 3];
        const auto transb = ops[next() % 3];
        const int ldaMin = transa == ACLBLAS_OP_N ? m : k;
        const int ldbMin = transb == ACLBLAS_OP_N ? k : n;
        const int lda = ldaMin + static_cast<int>(next() % 17);
        const int ldb = ldbMin + static_cast<int>(next() % 17);
        const int ldc = m + static_cast<int>(next() % 17);
        const float alpha = alphas[next() % 4];
        const float beta = betas[next() % 4];

        SCOPED_TRACE(::testing::Message()
            << "random_case=" << caseIndex << " seed=0x6d2b79f5"
            << " transa=" << transa << " transb=" << transb
            << " m=" << m << " n=" << n << " k=" << k
            << " lda=" << lda << " ldb=" << ldb << " ldc=" << ldc
            << " alpha=" << alpha << " beta=" << beta);
        RunSgemmCase(handle_, {transa, transb, m, n, k,
                               lda, ldb, ldc, alpha, beta});
    }
}

TEST_F(Arch22SgemmTest, CorrectnessLargeDirectRowsBoundaries)
{
    struct Shape { int m, n, k; };
    const Shape shapes[] = {{2048, 64, 64}, {64, 2048, 64}, {64, 64, 2048}};
    const aclblasOperation_t ops[] = {ACLBLAS_OP_N, ACLBLAS_OP_T};
    int caseIndex = 0;
    for (const auto& shape : shapes) {
        for (const auto transa : ops) {
            for (const auto transb : ops) {
                const int lda = transa == ACLBLAS_OP_N ? shape.m : shape.k;
                const int ldb = transb == ACLBLAS_OP_N ? shape.k : shape.n;
                SCOPED_TRACE(::testing::Message()
                    << "large_case=" << caseIndex << " transa=" << transa << " transb=" << transb
                    << " m=" << shape.m << " n=" << shape.n << " k=" << shape.k);
                RunSgemmCase(handle_, {transa, transb, shape.m, shape.n, shape.k,
                                       lda, ldb, shape.m, 1.0f, 0.0f});
                ++caseIndex;
            }
        }
    }
    // Large direct Matmul followed by the generic alpha/beta epilogue.
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 2048, 64, 64,
                           2048, 64, 2064, 0.75f, -0.25f});
    EXPECT_EQ(caseIndex, 12);
}

TEST_F(Arch22SgemmTest, CorrectnessOuterPanelsInteractAcrossMnk1024)
{
    // Every public axis crosses the retained 1024 panel boundary. This checks
    // panel base addressing, two K contributions, accumulator ownership, and
    // the final public-C scatter independently of the performance benchmark.
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 1040, 1040, 1040,
                           1040, 1040, 1040, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, 1040, 1040, 1040,
                           1040, 1040, 1056, 0.75f, -0.25f});
}

TEST_F(Arch22SgemmTest, WholeGridNoSplitNonPanelRandomCorrectness)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 2064, 2128, 2056,
                           2064, 2056, 2064, 1.0f, 0.0f});
}

TEST_F(Arch22SgemmTest, WholeGridSquareAllPublicLayoutsAndEpilogues)
{
    constexpr int m = 2064;
    constexpr int n = 2128;
    constexpr int k = 2056;
    // Direct compact inputs plus general alpha/beta/padded-C epilogue.
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
                           m, k, m + 16, 0.75f, -0.25f});
    // Packed input normalization with direct compact-C output.
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_T, m, n, k,
                           m + 16, n + 16, m, 1.0f, 0.0f});
    // Packed input normalization plus general output epilogue.
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, m, n, k,
                           k + 8, k + 8, m + 16, -0.625f, 0.375f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, m, n, k,
                           k + 8, n + 16, m + 16, 0.875f, -0.125f});
}

TEST_F(Arch22SgemmTest, WholeGridSquareRightTransposeBaseKAlignment)
{
    constexpr int m = 2048;
    constexpr int n = 2048;
    // Aligned K keeps the no-padding TN/TT paths. Padded public inputs with
    // compact direct output also verify that input normalization is independent
    // of the output epilogue.
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, m, n, 2048,
                           2056, 2056, m, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, m, n, 2048,
                           2056, n + 16, m, 1.0f, 0.0f});
    // baseK tails 8 and 16. Both operands use padded public LDs so the
    // whole-grid adapter must preserve logical K while normalizing compute K.
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, m, n, 2056,
                           2064, 2064, m + 16, -0.625f, 0.375f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, m, n, 2064,
                           2072, n + 16, m + 16, 0.875f, -0.125f});
}

TEST_F(Arch22SgemmTest, WholeGridNoSplitRectangularCorrectness)
{
    RunWholeGridNoSplitPatternCase(handle_, 2048, 3072, 2048);
}

TEST_F(Arch22SgemmTest, WholeGridNoSplit4096Correctness)
{
    RunWholeGridNoSplitPatternCase(handle_, 4096, 4096, 4096);
}

TEST_F(Arch22SgemmTest, WholeGridNoSplit8192Correctness)
{
    RunWholeGridNoSplitPatternCase(handle_, 8192, 8192, 8192);
}

TEST_F(Arch22SgemmTest, ConjugateTransposeEqualsTransposeForFp32)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_C, ACLBLAS_OP_C, 17, 19, 23, 32, 24, 24, 0.875f, -0.125f});
}

TEST_F(Arch22SgemmTest, TileBoundaryShapes)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 15, 17, 16, 15, 16, 15, 1.0f, 0.0f});
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, 17, 15, 16, 16, 16, 17, 1.0f, 0.0f});
}

TEST_F(Arch22SgemmTest, PublicDimensionsCanExceedSingleMatmulShape)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 8201, 2, 3, 8201, 3, 8201, 0.75f, -0.25f});
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 2, 8201, 3, 2, 3, 2, -0.5f, 0.375f});
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 2, 3, 8201, 2, 8201, 2, 1.25f, -0.125f});
}

TEST_F(Arch22SgemmTest, OuterTilesInteractAcrossMnk)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_T, 273, 269, 265,
                           272, 280, 280, 0.625f, -0.375f});
}

TEST_F(Arch22SgemmTest, PackedKContributionBoundary)
{
    // K <= 256 has one host contribution and can feed the epilogue directly;
    // K == 257 still requires accumulation across two contributions.
    for (const int k : {255, 256, 257}) {
        RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 33, 35, k,
                               40, k + 8, 40, 0.625f, -0.375f});
    }
}

TEST_F(Arch22SgemmTest, BetaZeroIgnoresPoisonedC)
{
    constexpr int m = 17, n = 13, k = 19, lda = 24, ldb = 24, ldc = 24;
    const float alpha = 0.875f;
    const float beta = 0.0f;
    std::vector<float> a(static_cast<size_t>(lda) * k);
    std::vector<float> b(static_cast<size_t>(ldb) * n);
    std::vector<float> c(static_cast<size_t>(ldc) * n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> golden(static_cast<size_t>(ldc) * n, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = Pattern(static_cast<int>(i) + 401);
    for (size_t i = 0; i < b.size(); ++i) b[i] = Pattern(static_cast<int>(i) + 503);
    ASSERT_EQ(aclblasSgemm_cpu(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
                               alpha, a.data(), lda, b.data(), ldb, beta, golden.data(), ldc),
              ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSgemm_npu(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
                               &alpha, a.data(), lda, b.data(), ldb, &beta, c.data(), ldc),
              ACLBLAS_STATUS_SUCCESS);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const size_t pos = static_cast<size_t>(row) + static_cast<size_t>(col) * ldc;
            EXPECT_TRUE(std::isfinite(c[pos]));
            EXPECT_NEAR(c[pos], golden[pos], 1.0e-4f);
        }
        for (int row = m; row < ldc; ++row) {
            EXPECT_TRUE(std::isnan(c[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc]));
        }
    }
}

TEST_F(Arch22SgemmTest, DegenerateBetaZeroIgnoresPoisonedC)
{
    constexpr int m = 7, n = 5, lda = 8, ldb = 16, ldc = 11;
    for (const bool kZero : {false, true}) {
        const int k = kZero ? 0 : 9;
        const float alpha = kZero ? 3.0f : 0.0f;
        const float beta = 0.0f;
        std::vector<float> a(static_cast<size_t>(lda) * std::max(1, k), 1.0f);
        std::vector<float> b(static_cast<size_t>(ldb) * n, 1.0f);
        std::vector<float> c(static_cast<size_t>(ldc) * n, std::numeric_limits<float>::quiet_NaN());
        ASSERT_EQ(aclblasSgemm_npu(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
                                   &alpha, a.data(), lda, b.data(), ldb, &beta, c.data(), ldc),
                  ACLBLAS_STATUS_SUCCESS);
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                EXPECT_EQ(c[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc], 0.0f);
            }
            for (int row = m; row < ldc; ++row) {
                EXPECT_TRUE(std::isnan(c[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc]));
            }
        }
    }
}

TEST_F(Arch22SgemmTest, AlphaZeroScalesOnlyLogicalOutput)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_T, ACLBLAS_OP_N, 37, 29, 41, 48, 48, 48, 0.0f, -0.5f});
}

TEST_F(Arch22SgemmTest, KZeroScalesOnlyLogicalOutput)
{
    RunSgemmCase(handle_, {ACLBLAS_OP_N, ACLBLAS_OP_N, 37, 29, 0, 48, 48, 48, 9.0f, -0.5f});
}

TEST_F(Arch22SgemmTest, ValidatesLeadingDimensionsAndNullScalars)
{
    float scalar = 1.0f;
    float zero = 0.0f;
    auto* fake = reinterpret_cast<float*>(static_cast<uintptr_t>(0x1000));
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4,
                            &scalar, nullptr, 3, nullptr, 4, &scalar, nullptr, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4,
                            nullptr, nullptr, 4, nullptr, 4, &scalar, nullptr, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4,
                            &scalar, fake, 4, fake, 4, nullptr, fake, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, -1, 4, 4,
                            &scalar, fake, 4, fake, 4, &scalar, fake, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_T, ACLBLAS_OP_N, 4, 4, 4,
                            &scalar, fake, 3, fake, 4, &scalar, fake, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_T, 4, 4, 4,
                            &scalar, fake, 4, fake, 3, &scalar, fake, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4,
                            &scalar, nullptr, 4, fake, 4, &scalar, fake, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4,
                            &scalar, fake, 4, nullptr, 4, &scalar, fake, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4,
                            &scalar, fake, 4, fake, 4, &scalar, nullptr, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, static_cast<aclblasOperation_t>(99), ACLBLAS_OP_N, 4, 4, 4,
                            &scalar, nullptr, 4, nullptr, 4, &scalar, nullptr, 4),
              ACLBLAS_STATUS_INVALID_ENUM);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, static_cast<aclblasOperation_t>(99), 4, 4, 4,
                            &scalar, fake, 4, fake, 4, &scalar, fake, 4),
              ACLBLAS_STATUS_INVALID_ENUM);
    EXPECT_EQ(aclblasSgemm(nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4,
                            &scalar, nullptr, 4, nullptr, 4, &scalar, nullptr, 4),
              ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 0,
                            &scalar, nullptr, 4, nullptr, 1, &zero, nullptr, 4),
              ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 0, 4, 4,
                            &scalar, nullptr, 0, nullptr, 0, &scalar, nullptr, 0),
              ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasSgemm(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 0, 4,
                            &scalar, nullptr, 0, nullptr, 0, &scalar, nullptr, 0),
              ACLBLAS_STATUS_SUCCESS);
}

TEST_F(Arch22SgemmTest, AlphaZeroDoesNotRequireAB)
{
    constexpr int m = 7, n = 5, k = 9, ldc = 11;
    const float alpha = 0.0f;
    const float beta = -0.25f;
    std::vector<float> c(static_cast<size_t>(ldc) * n);
    for (size_t i = 0; i < c.size(); ++i) c[i] = Pattern(static_cast<int>(i) + 313);
    const std::vector<float> before = c;
    ASSERT_EQ(aclblasSgemm_npu(handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
                               &alpha, nullptr, m, nullptr, k, &beta, c.data(), ldc),
              ACLBLAS_STATUS_SUCCESS);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const size_t pos = static_cast<size_t>(row) + static_cast<size_t>(col) * ldc;
            EXPECT_EQ(c[pos], beta * before[pos]);
        }
        for (int row = m; row < ldc; ++row) {
            const size_t pos = static_cast<size_t>(row) + static_cast<size_t>(col) * ldc;
            EXPECT_EQ(c[pos], before[pos]);
        }
    }
}

} // namespace
