/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ctrmm_test_inputs.h"
#include "ctrmm_arch22_test.h"

INSTANTIATE_TEST_SUITE_P(
    Ctrmm, CtrmmArch22Test, ::testing::ValuesIn(GetCasesFromCsv<CtrmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtrmmParam>);
TEST_P(CtrmmArch22Test, CsvDriven) { Run(GetParam()); }

TEST_F(CtrmmArch22Test, EmulatedFmaProducts)
{
    CtrmmParam p(
        csv_map{
            {"case_name", "emulated_fma_products"},
            {"m", "65"},
            {"n", "257"},
            {"trans", "T"},
            {"random_seed", "139291"}});
    Run(
        p, false, 1,
        [=](auto& a, auto&) {
            for (int j = 0; j < p.m; ++j) {
                for (int i = 0; i < p.m; ++i) {
                    if (i != j) {
                        a[static_cast<size_t>(j) * p.lda + i] = {0, 0};
                    }
                }
            }
        },
        true);
    p.caseName += "_exponents";
    p.n = 1027;

    Run(
        p, false, 1,
        [=](auto& a, auto& b) {
            std::fill(a.begin(), a.end(), aclblasComplex{0, 0});
            for (int i = 0; i < p.m; ++i) {
                a[static_cast<size_t>(i) * p.lda + i] = {ExponentInputValue(i + 1U), ExponentInputValue(i + 71U)};
            }
            for (size_t i = 0; i < b.size(); ++i) {
                b[i] = {ExponentInputValue(i + 7U), ExponentInputValue(i + 19U)};
            }
        },
        true);
}

// Independent distributions: the CSV's RANDOM_NORM_5_5 is uniform [-5,5].
// Exercise both sides of the old numerical routing thresholds, including equality.
TEST_F(CtrmmArch22Test, NumericalRoutingBoundaries)
{
    for (int q : {8, 65}) {
        for (const auto& [side, trans] : SideTransModes()) {
            for (int edge : {-1, 0, 1}) {
                const int m = side == ACLBLAS_SIDE_LEFT ? q : 17;
                const int n = side == ACLBLAS_SIDE_LEFT ? 17 : q;
                CtrmmParam p(
                    csv_map{
                        {"case_name", "numerical_boundary_q" + std::to_string(q) + "_s" + std::to_string(side) + "_t" +
                                          std::to_string(trans) + "_e" + std::to_string(edge)},
                        {"m", std::to_string(m)},
                        {"n", std::to_string(n)},
                        {"lda", std::to_string(q + 3)},
                        {"ldb", std::to_string(m + 5)},
                        {"ldc", std::to_string(m + 7)},
                        {"random_seed", "38657"}});
                p.side = side;
                p.trans = trans;
                p.uplo = edge == 0 ? ACLBLAS_LOWER : ACLBLAS_UPPER;
                p.diag = edge == 1 ? ACLBLAS_UNIT : ACLBLAS_NON_UNIT;
                const float magnitude = edge < 0 ? std::nextafter(32.0f, 0.0f) :
                                        edge > 0 ? std::nextafter(32.0f, 64.0f) :
                                                   32.0f;
                SCOPED_TRACE(p.caseName);
                Run(p, false, 1, [=](auto& a, auto& b) { FillRoutingBoundaryInputs(a, b, magnitude); });
                p.caseName += "_alpha";
                p.SetAlpha(
                    {edge < 0 ? std::nextafter(8.0f, 0.0f) :
                     edge > 0 ? std::nextafter(8.0f, 16.0f) :
                                8.0f,
                     0.0f});
                Run(p, false, 1);
                p.caseName += "_imag";
                p.SetAlpha({0.0f, p.alphaReal});
                Run(p, false, 1);
            }
        }
    }
}

TEST_F(CtrmmArch22Test, CancellationAndReductionLength)
{
    for (int q : {16, 64, 128, 257, 1024, 4096}) {
        CtrmmParam p(
            csv_map{
                {"case_name", "cancellation_q" + std::to_string(q)},
                {"m", std::to_string(q)},
                {"n", "16"},
                {"random_seed", "51671"}});
        SCOPED_TRACE(p.caseName);
        Run(p, false, 1, [](auto& a, auto& b) {
            for (auto& z : a) {
                z = {32.0f, 32.0f};
            }
            for (auto& z : b) {
                z = {32.0f, 31.999755859375f};
            }
        });
        p.caseName = "long_random_q" + std::to_string(q);
        p.trans = ACLBLAS_OP_C;
        Run(p, false, 1);
    }
}

TEST_F(CtrmmArch22Test, CertifiedAndOrderedBranches)
{
    for (const auto& [side, uplo, trans, diag] : AllBlasModes()) {
        const int m = side == ACLBLAS_SIDE_LEFT ? 65 : 131;
        const int n = side == ACLBLAS_SIDE_LEFT ? 131 : 65;
        CtrmmParam p(
            csv_map{
                {"case_name", "ordered_s" + std::to_string(side) + "_u" + std::to_string(uplo) + "_t" +
                                  std::to_string(trans) + "_d" + std::to_string(diag)},
                {"m", std::to_string(m)},
                {"n", std::to_string(n)},
                {"lda", "69"},
                {"ldb", std::to_string(m + 3)},
                {"ldc", std::to_string(m + 7)},
                {"random_seed", "197703"}});
        p.side = side;
        p.uplo = uplo;
        p.trans = trans;
        p.diag = diag;
        SCOPED_TRACE(p.caseName);
        Run(p, true, 1, [](auto& a, auto& b) { ZeroStridedInputs(a, b); }, true);
        const std::string originalName = p.caseName;
        for (auto alpha : {aclblasComplex{-1, 0}, {0, 1}, {0, -1}}) {
            p.SetAlpha(alpha);
            p.caseName = originalName + "_rotation_" + std::to_string(alpha.real) + "_" + std::to_string(alpha.imag);
            Run(p, false, 1, {}, true);
        }
        p.SetAlpha({1, 0});
        p.caseName = originalName;
        p.caseName += "_small_norm";
        Run(p, false, 1, [](auto& a, auto& b) { FillSmallNormInputs(a, b); }, diag == ACLBLAS_UNIT);
        p.caseName += "_tiny_component";
        Run(
            p, false, 1,
            [](auto& a, auto& b) {
                a[0] = {0x1p-31f, -0x1p-31f};
                b[0] = {0x1p-31f, 0x1p-30f};
            },
            true);
    }
}

TEST_F(CtrmmArch22Test, InPlaceOffsetAndPadding)
{
    for (const auto& [side, uplo, trans, diag] : AllBlasModes()) {
        CtrmmParam p(
            csv_map{
                {"case_name", "inplace"},
                {"m", "67"},
                {"n", "71"},
                {"lda", "79"},
                {"ldb", "73"},
                {"ldc", "73"},
                {"alpha_real", "0.5"},
                {"alpha_imag", "-0.25"},
                {"random_seed", "1991"}});
        p.side = side;
        p.uplo = uplo;
        p.trans = trans;
        p.diag = diag;
        SCOPED_TRACE(
            std::to_string(side) + "/" + std::to_string(uplo) + "/" + std::to_string(trans) + "/" +
            std::to_string(diag));
        Run(p, true, 1);
        p.ldc = 79;
        Run(p, true, 1);
    }
}

TEST_F(CtrmmArch22Test, AlphaZeroPreservesPadding)
{
    CtrmmParam p(
        csv_map{
            {"case_name", "alpha0_padding"},
            {"m", "7"},
            {"n", "9"},
            {"lda", "7"},
            {"ldb", "11"},
            {"ldc", "13"},
            {"alpha_real", "0"},
            {"nullA", "1"},
            {"nullB", "1"}});
    Run(p, false, 3);
}

TEST_F(CtrmmArch22Test, RejectPartialOverlap)
{
    std::vector<aclblasComplex> input(32, {1.0f, -0.25f});
    CtrmmDeviceBuffer a(input);
    CtrmmDeviceBuffer b(input);
    const aclblasComplex alpha{1, 0};
    for (aclblasComplex* output : {a.Data(), a.Data() + 1, b.Data() + 1}) {
        EXPECT_EQ(
            aclblasCtrmm(
                handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, 4, &alpha, a.Data(), 4,
                b.Data(), 4, output, 4),
            ACLBLAS_STATUS_INVALID_VALUE);
    }
}

TEST_F(CtrmmArch22Test, SmallTileOffsetAndPadding)
{
    for (const auto shape : {std::pair<int, int>{3, 7}, {7, 8}, {15, 13}, {31, 29}, {31, 33}, {63, 61}}) {
        for (const auto& [uplo, trans, diag] : UploTransDiagModes()) {
            CtrmmParam p(
                csv_map{
                    {"case_name", "small_offset"},
                    {"m", std::to_string(shape.first)},
                    {"n", std::to_string(shape.second)},
                    {"lda", std::to_string(shape.first + 5)},
                    {"ldb", std::to_string(shape.first + 3)},
                    {"ldc", std::to_string(shape.first + 7)},
                    {"alpha_real", "0.5"},
                    {"alpha_imag", "-0.25"},
                    {"random_seed", "81291"}});
            p.side = ACLBLAS_SIDE_LEFT;
            p.uplo = uplo;
            p.trans = trans;
            p.diag = diag;
            SCOPED_TRACE(
                std::to_string(shape.first) + "/" + std::to_string(shape.second) + "/" + std::to_string(uplo) + "/" +
                std::to_string(trans) + "/" + std::to_string(diag));
            Run(p, false, 1);
            p.SetAlpha({1, 0});
            Run(p, false, 1);
        }
    }
}

TEST_F(CtrmmArch22Test, RectangularOffsetAndPadding)
{
    for (int q : {64, 65, 100, 128}) {
        for (int r : {513, 641}) {
            for (const auto& [side, uplo, trans, diag] : AllBlasModes()) {
                const int m = side == ACLBLAS_SIDE_LEFT ? q : r;
                const int n = side == ACLBLAS_SIDE_LEFT ? r : q;
                CtrmmParam p(
                    csv_map{
                        {"case_name", "rectangle_offset"},
                        {"m", std::to_string(m)},
                        {"n", std::to_string(n)},
                        {"lda", std::to_string(q + 5)},
                        {"ldb", std::to_string(m + 3)},
                        {"ldc", std::to_string(m + 7)},
                        {"alpha_real", "0.5"},
                        {"alpha_imag", "-0.25"},
                        {"random_seed", "13921"}});
                p.side = side;
                p.uplo = uplo;
                p.trans = trans;
                p.diag = diag;
                SCOPED_TRACE(
                    std::to_string(q) + "/" + std::to_string(r) + "/" + std::to_string(side) + "/" +
                    std::to_string(uplo) + "/" + std::to_string(trans) + "/" + std::to_string(diag));
                Run(p, false, 1);
                if (q == 65) {
                    Run(p, true, 1);
                }
            }
        }
    }
}

TEST_F(CtrmmArch22Test, QueuedCallsShareWorkspace)
{
    CtrmmParam p(
        csv_map{
            {"case_name", "queued"},
            {"m", "67"},
            {"n", "71"},
            {"lda", "71"},
            {"ldb", "70"},
            {"ldc", "70"},
            {"random_seed", "671"}});
    auto a = makeBlasComplexMatrix(p.m, p.m, p.lda, p.aFill, p.randomSeed);
    auto b = makeBlasComplexMatrix(p.m, p.n, p.ldb, p.bFill, p.randomSeed + 1);
    std::vector<aclblasComplex> initial(p.ldc * p.n, {1, -1});
    CtrmmDeviceBuffer da(a);
    CtrmmDeviceBuffer db(b);
    CtrmmDeviceBuffer first(initial);
    CtrmmDeviceBuffer second(initial);
    auto g1 = initial;
    auto g2 = initial;
    p.SetAlpha({100, -0.25f});
    const aclblasComplex firstAlpha = p.GetAlpha();
    CtrmmGolden(p, a, b, g1);
    ASSERT_EQ(
        aclblasCtrmm(
            handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &firstAlpha, da.Data(), p.lda, db.Data(), p.ldb,
            first.Data(), p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    p.SetAlpha({1, 0});
    const aclblasComplex secondAlpha = p.GetAlpha();
    CtrmmGolden(p, a, b, g2);
    ASSERT_EQ(
        aclblasCtrmm(
            handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &secondAlpha, da.Data(), p.lda, db.Data(), p.ldb,
            second.Data(), p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    CtrmmDeviceBuffer::Check(aclrtSynchronizeStream(stream_));
    auto v1 = first.Read();
    auto v2 = second.Read();
    for (size_t i = 0; i < initial.size(); ++i) {
        EXPECT_NEAR(v1[i].real, g1[i].real, 1e-2);
        EXPECT_NEAR(v1[i].imag, g1[i].imag, 1e-2);
        EXPECT_NEAR(v2[i].real, g2[i].real, 1e-2);
        EXPECT_NEAR(v2[i].imag, g2[i].imag, 1e-2);
    }
}

TEST_F(CtrmmArch22Test, IndependentMagnitudeAndLength)
{
    for (int q : {65, 128, 257, 1024, 4096}) {
        for (int magnitude : {1, 5, 16, 32}) {
            for (unsigned seed : {10937U, 77077U}) {
                CtrmmParam p(
                    csv_map{
                        {"case_name", "independent_q" + std::to_string(q) + "_mag" + std::to_string(magnitude) +
                                          "_seed" + std::to_string(seed)},
                        {"m", std::to_string(q)},
                        {"n", "17"},
                        {"trans", "T"},
                        {"random_seed", std::to_string(seed)}});
                SCOPED_TRACE(p.caseName);
                Run(p, false, 1, [=](auto& a, auto& b) {
                    const float scale = magnitude / 5.0f;
                    for (auto& x : a) {
                        x.real *= scale;
                        x.imag *= scale;
                    }
                    for (auto& x : b) {
                        x.real *= scale;
                        x.imag *= scale;
                    }
                });
            }
        }
    }
}

TEST_F(CtrmmArch22Test, RepeatedRoundingAmplification)
{
    for (int q : {257, 1024, 4096}) {
        CtrmmParam p(
            csv_map{
                {"case_name", "repeated_rounding_q" + std::to_string(q)},
                {"m", std::to_string(q)},
                {"n", "17"},
                {"trans", "T"}});
        Run(p, false, 1, [=](auto& a, auto& b) {
            for (auto& v : a) {
                v = {0x1.8bf97p+1f, 0x1.bd2cep-2f};
            }
            for (auto& v : b) {
                v = {0x1.3a7b1p+2f, -0x1.c65ab8p+1f};
            }
        });
    }
}

TEST_F(CtrmmArch22Test, MediumMatrixOffsetsAndCancellation)
{
    for (int q : {65, 100, 127, 128}) {
        for (const auto& [uplo, trans, diag] : UploTransDiagModes()) {
            CtrmmParam p(
                csv_map{
                    {"case_name", "medium_q" + std::to_string(q) + "_u" + std::to_string(uplo) + "_t" +
                                      std::to_string(trans) + "_d" + std::to_string(diag)},
                    {"m", std::to_string(q)},
                    {"n", "97"},
                    {"lda", std::to_string(q + 3)},
                    {"ldb", std::to_string(q + 5)},
                    {"ldc", std::to_string(q + 7)},
                    {"random_seed", "53819"}});
            p.uplo = uplo;
            p.trans = trans;
            p.diag = diag;
            SCOPED_TRACE(p.caseName);
            Run(p, false, 1);
            p.caseName += "_cancel";
            Run(p, false, 1, [](auto& a, auto& b) {
                for (auto& z : a) {
                    z = {32.0f, 32.0f};
                }
                for (auto& z : b) {
                    z = {32.0f, 31.999755859375f};
                }
            });
            p.caseName += "_alpha";
            p.SetAlpha({0.5f, -0.25f});
            Run(p, false, 1);
        }
    }
}

TEST_F(CtrmmArch22Test, CubeShortReductionBranches)
{
    // r=131 bypasses the small LEFT kernel so both sides exercise Cube.
    for (int q : {64, 100, 128, 166, 192, 256}) {
        for (const auto& [side, uplo, trans, diag] : AllBlasModes()) {
            const int m = side == ACLBLAS_SIDE_LEFT ? q : 131;
            const int n = side == ACLBLAS_SIDE_LEFT ? 131 : q;
            const std::string prefix = "cube_short_q" + std::to_string(q) + "_s" + std::to_string(side) + "_u" +
                                       std::to_string(uplo) + "_t" + std::to_string(trans) + "_d" +
                                       std::to_string(diag);
            CtrmmParam p(
                csv_map{
                    {"case_name", prefix},
                    {"m", std::to_string(m)},
                    {"n", std::to_string(n)},
                    {"lda", std::to_string(q + 3)},
                    {"ldb", std::to_string(m + 5)},
                    {"ldc", std::to_string(m + 7)},
                    {"random_seed", "38737"}});
            p.side = side;
            p.uplo = uplo;
            p.trans = trans;
            p.diag = diag;
            SCOPED_TRACE(prefix);
            for (int magnitude : {5, 16, 32}) {
                p.caseName = prefix + "_mag" + std::to_string(magnitude);
                Run(p, false, 1, [=](auto& a, auto& b) {
                    const float scale = magnitude / 5.0f;
                    for (auto& z : a) {
                        z.real *= scale;
                        z.imag *= scale;
                    }
                    for (auto& z : b) {
                        z.real *= scale;
                        z.imag *= scale;
                    }
                });
            }
            p.caseName = prefix + "_cancel";
            Run(p, false, 1, [](auto& a, auto& b) {
                for (auto& z : a) {
                    z = {32.0f, 32.0f};
                }
                for (auto& z : b) {
                    z = {32.0f, 31.999755859375f};
                }
            });
        }
    }
}

TEST_F(CtrmmArch22Test, HeterogeneousCancellation)
{
    // Correlated nonconstant components exercise cancellation without relying
    // on one repeated, exactly representable product or the CSV distribution.
    for (int q : {64, 100, 128, 256, 1024}) {
        for (const auto& [side, uplo, trans] : SideUploTransModes()) {
            for (int magnitude : {1, 5, 32}) {
                const int r = q <= 128 ? 131 : 17;
                const int m = side == ACLBLAS_SIDE_LEFT ? q : r;
                const int n = side == ACLBLAS_SIDE_LEFT ? r : q;
                CtrmmParam p(
                    csv_map{
                        {"case_name", "heterogeneous_q" + std::to_string(q) + "_s" + std::to_string(side) + "_u" +
                                          std::to_string(uplo) + "_t" + std::to_string(trans) + "_mag" +
                                          std::to_string(magnitude)},
                        {"m", std::to_string(m)},
                        {"n", std::to_string(n)},
                        {"lda", std::to_string(q + 3)},
                        {"ldb", std::to_string(m + 5)},
                        {"ldc", std::to_string(m + 7)},
                        {"random_seed", "72923"}});
                p.side = side;
                p.uplo = uplo;
                p.trans = trans;
                SCOPED_TRACE(p.caseName);
                Run(p, false, 1, [=](auto& a, auto& b) {
                    for (auto& z : a) {
                        const float v = magnitude * (0.5f + 0.1f * std::abs(z.real));
                        z = {v, v};
                    }
                    for (auto& z : b) {
                        const float v = magnitude * (0.5f + 0.1f * std::abs(z.real));
                        float w = v;
                        const int steps = static_cast<int>(std::abs(z.imag) * 3.0f) + 1;
                        for (int i = 0; i < steps; ++i) {
                            w = std::nextafter(w, 0.0f);
                        }
                        z = {v, w};
                    }
                });
            }
        }
    }
}

TEST_F(CtrmmArch22Test, SmallHeterogeneousCancellation)
{
    for (int q : {8, 16, 32, 64, 100, 128}) {
        for (const auto& [uplo, trans, diag] : UploTransDiagModes()) {
            CtrmmParam p(
                csv_map{
                    {"case_name", "small_heterogeneous_q" + std::to_string(q) + "_u" + std::to_string(uplo) + "_t" +
                                      std::to_string(trans) + "_d" + std::to_string(diag)},
                    {"m", std::to_string(q)},
                    {"n", std::to_string(std::min(q, 17))},
                    {"lda", std::to_string(q + 3)},
                    {"ldb", std::to_string(q + 5)},
                    {"ldc", std::to_string(q + 7)},
                    {"random_seed", "91733"}});
            p.uplo = uplo;
            p.trans = trans;
            p.diag = diag;
            SCOPED_TRACE(p.caseName);
            Run(p, false, 1, [](auto& a, auto& b) {
                for (auto& z : a) {
                    const float v = 32.0f * (0.5f + 0.1f * std::abs(z.real));
                    z = {v, 0.5f * v};
                }
                for (auto& z : b) {
                    const float v = 32.0f * (0.5f + 0.1f * std::abs(z.real));
                    float w = v;
                    const int steps = static_cast<int>(std::abs(z.imag) * 3.0f) + 1;
                    for (int i = 0; i < steps; ++i) {
                        w = std::nextafter(w, 0.0f);
                    }
                    z = {0.5f * v, w};
                }
            });
        }
    }
}

TEST_F(CtrmmArch22Test, MediumRoutingBoundaries)
{
    // Exercise both sides of the magnitude guard used by grouped MMAD, as
    // well as the neighbouring reduction lengths. Force an effective B
    // entry to the boundary; a random draw alone need not reach it.
    for (int q : {128, 129, 192, 256, 257}) {
        for (const auto& [side, uplo, trans] : SideUploTransModes()) {
            for (int delta : {-1, 0, 1}) {
                const int r = 19;
                const int m = side == ACLBLAS_SIDE_LEFT ? q : r;
                const int n = side == ACLBLAS_SIDE_LEFT ? r : q;
                CtrmmParam p(
                    csv_map{
                        {"case_name", "medium_boundary_q" + std::to_string(q) + "_s" + std::to_string(side) + "_u" +
                                          std::to_string(uplo) + "_t" + std::to_string(trans) + "_delta" +
                                          std::to_string(delta)},
                        {"m", std::to_string(m)},
                        {"n", std::to_string(n)},
                        {"lda", std::to_string(q + 3)},
                        {"ldb", std::to_string(m + 5)},
                        {"ldc", std::to_string(m + 7)},
                        {"random_seed", "67129"}});
                p.side = side;
                p.uplo = uplo;
                p.trans = trans;
                SCOPED_TRACE(p.caseName);
                const float bound = delta == 0 ? 16.0f : std::nextafter(16.0f, delta < 0 ? 0.0f : 32.0f);
                Run(p, false, 1, [=](auto& a, auto& b) {
                    for (auto& z : a) {
                        z.real *= bound / 5.0f;
                        z.imag *= bound / 5.0f;
                    }
                    for (auto& z : b) {
                        z.real *= bound / 5.0f;
                        z.imag *= bound / 5.0f;
                    }
                    b[0].real = bound;
                });
            }
        }
    }
}

TEST_F(CtrmmArch22Test, ShortPackedCancellation)
{
    // r >= 512 selects four-row A packing and the short wide writeback.
    for (int q : {64, 100, 128}) {
        for (const auto& [side, uplo, trans, diag] : AllBlasModes()) {
            const int m = side == ACLBLAS_SIDE_LEFT ? q : 513;
            const int n = side == ACLBLAS_SIDE_LEFT ? 513 : q;
            CtrmmParam p(
                csv_map{
                    {"case_name", "short_packed_cancel_q" + std::to_string(q) + "_s" + std::to_string(side) + "_u" +
                                      std::to_string(uplo) + "_t" + std::to_string(trans) + "_d" +
                                      std::to_string(diag)},
                    {"m", std::to_string(m)},
                    {"n", std::to_string(n)},
                    {"lda", std::to_string(q + 3)},
                    {"ldb", std::to_string(m + 5)},
                    {"ldc", std::to_string(m + 7)},
                    {"random_seed", "45179"}});
            p.side = side;
            p.uplo = uplo;
            p.trans = trans;
            p.diag = diag;
            SCOPED_TRACE(p.caseName);
            Run(p, false, 1, [](auto& a, auto& b) {
                for (auto& z : a) {
                    const float v = 32.0f * (0.5f + 0.1f * std::abs(z.real));
                    z = {v, v};
                }
                for (auto& z : b) {
                    const float v = 32.0f * (0.5f + 0.1f * std::abs(z.real));
                    z = {v, std::nextafter(v, 0.0f)};
                }
            });
        }
    }
}

// The upstream RANDOM_NORM names use a uniform generator. Exercise a true
// Gaussian independently, paired 1:1 with uniform inputs of equal variance.
TEST_F(CtrmmArch22Test, PairedUniformGaussianDistributions)
{
    auto paired = [&](int q, int r, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
                      aclblasDiagType_t diag) {
        for (bool gaussian : {false, true}) {
            CtrmmParam p(
                csv_map{
                    {"case_name", std::string(gaussian ? "gaussian_" : "uniform_") + "q" + std::to_string(q) + "_r" +
                                      std::to_string(r) + "_s" + std::to_string(side) + "_u" + std::to_string(uplo) +
                                      "_t" + std::to_string(trans) + "_d" + std::to_string(diag)},
                    {"m", std::to_string(side == ACLBLAS_SIDE_LEFT ? q : r)},
                    {"n", std::to_string(side == ACLBLAS_SIDE_LEFT ? r : q)}});
            p.side = side;
            p.uplo = uplo;
            p.trans = trans;
            p.diag = diag;
            p.lda = q + 3;
            p.ldb = p.m + 5;
            p.ldc = p.m + 7;
            p.SetAlpha(trans == ACLBLAS_OP_N ? aclblasComplex{1.0f, 0.0f} : aclblasComplex{0.75f, -0.25f});
            SCOPED_TRACE(p.caseName);
            Run(p, false, 1, [=](auto& a, auto& b) {
                std::mt19937 rng(910307U + 13U * q + 17U * r + 19U * side + 23U * uplo + 29U * trans + 31U * diag);
                std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
                std::normal_distribution<float> normal(0.0f, 5.0f / std::sqrt(3.0f));
                auto sample = [&]() { return gaussian ? normal(rng) : uniform(rng); };
                for (auto& v : a) {
                    v = {sample(), sample()};
                }
                for (auto& v : b) {
                    v = {sample(), sample()};
                }
            });
        }
    };
    // All 24 modes across Small, compact Cube and single-complex-term Cube.
    for (int q : {8, 64, 100, 128, 129, 192, 256, 257}) {
        for (const auto& [side, uplo, trans, diag] : AllBlasModes()) {
            paired(q, q <= 64 ? 17 : 131, side, uplo, trans, diag);
        }
    }
    for (int q : {1024, 4096}) {
        paired(q, 17, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT);
        paired(q, 17, ACLBLAS_SIDE_RIGHT, ACLBLAS_LOWER, ACLBLAS_OP_C, ACLBLAS_UNIT);
    }
}

TEST_F(CtrmmArch22Test, ShortCancellationAlphaScale)
{
    for (int q : {1, 2, 3, 7, 15}) {
        for (int r : {129, 257}) {
            for (const char* side : {"LEFT", "RIGHT"}) {
                for (const char* trans : {"N", "T", "C"}) {
                    const auto p = MakeShortCancellationParam(
                        "short_cancel_q" + std::to_string(q) + "_r" + std::to_string(r) + side + trans, q, r, side,
                        trans);
                    Run(p, false, 0, FillShortCancellationInputs);
                }
            }
        }
    }
}

// Application-inspired synthetic inputs. The distributions below are test
// coverage choices, not BLAS input restrictions or samples of a model trace.
TEST_F(CtrmmArch22Test, ApplicationLlmTriangularProducts)
{
    for (int q : {16, 32, 64}) {
        for (int r : {64, 128, 256}) {
            for (const char* trans : {"N", "T"}) {
                for (float betaScale : {1.0f, 2.0f}) {
                    CtrmmParam p(
                        csv_map{
                            {"case_name", "app_gdn_q" + std::to_string(q) + "_r" + std::to_string(r) + trans + "_beta" +
                                              std::to_string(betaScale)},
                            {"m", std::to_string(q)},
                            {"n", std::to_string(r)},
                            {"side", "LEFT"},
                            {"uplo", "LOWER"},
                            {"trans", trans},
                            {"diag", "UNIT"},
                            {"alpha_real", "1"},
                            {"alpha_imag", "0"}});
                    Run(p, false, 0, [=](auto& a, auto& b) { FillGdnInputs(p, q, r, betaScale, a, b); });
                }
            }
        }
    }
    // SSD intra-chunk factor: T_ij=(C_i dot B_j)*exp(sum log(decay)), i>=j.
    for (int q : {32, 64, 128}) {
        for (int r : {64, 128}) {
            CtrmmParam p(
                csv_map{
                    {"case_name", "app_ssd_q" + std::to_string(q) + "_r" + std::to_string(r)},
                    {"m", std::to_string(q)},
                    {"n", std::to_string(r)},
                    {"side", "LEFT"},
                    {"uplo", "LOWER"},
                    {"trans", "N"},
                    {"diag", "NON_UNIT"}});
            Run(p, false, 0, [=](auto& a, auto& b) { FillSsdInputs(p, q, r, a, b); });
        }
    }
}

TEST_F(CtrmmArch22Test, ApplicationComplexFactorProducts)
{
    for (int q : {32, 128, 256}) {
        for (int r : {129, 512}) {
            for (const char* side : {"LEFT", "RIGHT"}) {
                for (float scale : {0.01f, 1.0f, 100.0f}) {
                    const bool right = std::string(side) == "RIGHT";
                    CtrmmParam p(
                        csv_map{
                            {"case_name", "app_factor_q" + std::to_string(q) + "_r" + std::to_string(r) + side + "_s" +
                                              std::to_string(scale)},
                            {"m", std::to_string(right ? r : q)},
                            {"n", std::to_string(right ? q : r)},
                            {"side", side},
                            {"uplo", "UPPER"},
                            {"trans", "C"},
                            {"diag", "NON_UNIT"}});
                    Run(p, false, 0, [=](auto& a, auto& b) {
                        std::mt19937 rng(85167 + q + r);
                        std::normal_distribution<float> normal(0, 1);
                        const float sigma = 1.0f / std::sqrt(static_cast<float>(q));
                        for (auto& z : a) {
                            z = {normal(rng) * sigma * scale, normal(rng) * sigma * scale};
                        }
                        for (int i = 0; i < q; ++i) {
                            a[i * p.lda + i] = {scale * (1.0f + std::abs(normal(rng))), 0};
                        }
                        for (auto& z : b) {
                            z = {normal(rng) * scale, normal(rng) * scale};
                        }
                    });
                }
            }
        }
    }
}

TEST_F(CtrmmArch22Test, ShortCancellationSmallShapes)
{
    for (int q : {1, 2, 3, 7, 15, 16}) {
        for (int r : {1, 2, 17, 64, 128, 129}) {
            for (const char* side : {"LEFT", "RIGHT"}) {
                for (const char* trans : {"N", "C"}) {
                    const auto p = MakeShortCancellationParam(
                        "short_all_q" + std::to_string(q) + "_r" + std::to_string(r) + side + trans, q, r, side, trans);
                    Run(p, false, 0, FillShortCancellationInputs);
                }
            }
        }
    }
}

TEST_F(CtrmmArch22Test, BatchedFmaDomainBoundaries)
{
    for (int q : {17, 65}) {
        for (const char* side : {"LEFT", "RIGHT"}) {
            for (const char* trans : {"N", "T", "C"}) {
                for (float alpha : {0.00390625f, 8.0f, 256.0f}) {
                    for (int edge : {-1, 0, 1, 2, 3, 4}) {
                        const bool right = std::string(side) == "RIGHT";
                        const int r = 67;
                        CtrmmParam p(
                            csv_map{
                                {"case_name", "fma_boundary_q" + std::to_string(q) + side + trans + "_alpha" +
                                                  std::to_string(alpha) + "_edge" + std::to_string(edge)},
                                {"m", std::to_string(right ? r : q)},
                                {"n", std::to_string(right ? q : r)},
                                {"side", side},
                                {"uplo", "LOWER"},
                                {"trans", trans},
                                {"diag", "NON_UNIT"},
                                {"alpha_real", std::to_string(alpha)},
                                {"alpha_imag", "0"}});
                        p.SetAlpha({alpha, 0});
                        Run(p, false, 3, [=](auto& a, auto& b) { FillBatchedFmaBoundary(a, b, edge); });
                    }
                }
            }
        }
    }
}
