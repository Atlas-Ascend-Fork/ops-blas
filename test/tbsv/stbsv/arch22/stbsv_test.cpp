/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <initializer_list>
#include <optional>
#include <random>
#include <vector>
#include <securec.h>

#include "verify.h"
#include "stbsv_arch22_test.h"
#include "csv_loader.h"
#include "device.h"
#include "fill.h"
#include "stbsv_golden.h"

void StbsvArch22Test::SetUpTestSuite()
{
    BlasTest<StbsvParam>::SetUpTestSuite();
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    // Taking &aclblasStbsv may return this executable's PLT trampoline.
    // Resolve the implementation in the process's default symbol scope.
    void* implementation = dlsym(RTLD_DEFAULT, "aclblasStbsv");
    ASSERT_NE(implementation, nullptr);
    Dl_info info{};
    ASSERT_NE(dladdr(implementation, &info), 0);
    ASSERT_NE(info.dli_fname, nullptr);
    // glibc allocates the canonical path; RAII also frees it on assertion failure.
    std::unique_ptr<char, decltype(&std::free)> actualPath(canonicalize_file_name(info.dli_fname), &std::free);
    ASSERT_NE(actualPath.get(), nullptr);
    const std::string actual(actualPath.get());
    std::cout << "[STBSV-LOADED-LIB] " << actual << std::endl;
    if (const char* expected = std::getenv("STBSV_EXPECTED_LIBRARY")) {
        std::unique_ptr<char, decltype(&std::free)> expectedPath(canonicalize_file_name(expected), &std::free);
        ASSERT_NE(expectedPath.get(), nullptr) << "Expected library does not exist: " << expected;
        const std::string canonicalExpected(expectedPath.get());
        ASSERT_EQ(actual, canonicalExpected) << "Refusing a mixed-version validation run";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Stbsv, StbsvArch22Test, ::testing::ValuesIn(GetCasesFromCsv<StbsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StbsvParam>);

// Keep the supplied CSV unchanged. This separate suite supplies the required
// 50/50 uniform/normal raw data, without diagonal strengthening or conditioning.
static std::vector<StbsvParam> SupplementalCases()
{
    std::vector<StbsvParam> cases;
    const int shapes[][2] = {{1, 0},     {7, 1},     {31, 7},     {65, 16}, {255, 33},  {512, 32},
                             {1024, 16}, {2048, 64}, {4096, 128}, {9, 16},  {32769, 0}, {32769, 1}};
    for (const auto& shape : shapes) {
        for (const auto* uplo : {"UPPER", "LOWER"}) {
            for (const auto* trans : {"N", "T", "C"}) {
                for (const auto* diag : {"NON_UNIT", "UNIT"}) {
                    for (int incx : {1, -3}) {
                        for (int distribution : {0, 1}) {
                            std::string name =
                                "RAW_" + std::to_string(distribution) + "_" + std::to_string(cases.size());
                            csv_map values{
                                {"case_name", name},
                                {"description", "unconditioned_distribution"},
                                {"uplo", uplo},
                                {"trans", trans},
                                {"diag", diag},
                                {"n", std::to_string(shape[0])},
                                {"k", std::to_string(shape[1])},
                                {"lda", std::to_string(shape[1] + 4)},
                                {"incx", std::to_string(incx)},
                                {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
                                {"random_seed", std::to_string(20261001 + cases.size())}};
                            cases.emplace_back(values);
                        }
                    }
                }
            }
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Supplemental, StbsvArch22Test, ::testing::ValuesIn(SupplementalCases()), PrintCaseInfoString<StbsvParam>);

struct StbsvFixture {
    const StbsvParam& p;
    int effK = 0;
    int allocLda = 1;
    int allocN = 1;
    int absIncx = 0;
    size_t xSize = 0;
    bool isUpper = false;
    bool needDevA = false;
    bool needDevX = false;
    std::vector<float> aHost;
    std::vector<float> xHost;
    std::vector<float> golden;
    std::optional<DeviceBuffer> aDev;
    std::optional<DeviceBuffer> xDev;

    explicit StbsvFixture(const StbsvParam& param) : p(param) {}

    void InitDims()
    {
        effK = (p.n > 0 && p.k >= 0) ? std::min(p.k, p.n - 1) : 0;
        allocLda = std::max(1, std::max(p.lda, effK + 1));
        allocN = std::max(1, p.n);
        isUpper = (p.uplo == ACLBLAS_UPPER);
        absIncx = (p.incx == INT_MIN) ? 0 : std::abs(p.incx);
        xSize = (p.n > 0 && absIncx > 0) ? static_cast<size_t>((p.n - 1) * absIncx + 1) : 0;
        // A real buffer keeps the invalid-stride check independent of nullptr validation.
        if (p.n > 0 && p.incx == INT_MIN) {
            xSize = 1;
        }
    }

    void PrepareHostData()
    {
        if (p.caseName.rfind("RAW_", 0) == 0) {
            PrepareRawData();
            return;
        }
        const int kl = isUpper ? 0 : effK;
        const int ku = isUpper ? effK : 0;
        aHost = makeBlasBanded(p.n, p.n, kl, ku, allocLda, p.aFill, p.randomSeed);
        StrengthenDiagonal();
        if (p.incx != INT_MIN) {
            xHost = makeBlasStrided(p.n, p.incx, p.xFill, p.randomSeed + 1);
        } else {
            xHost.assign(xSize, kBlasSentinel);
        }
        golden = xHost;
        needDevA = (p.n > 0 && !aHost.empty());
        needDevX = (xSize > 0 && !xHost.empty());
    }

    void PrepareRawData()
    {
        effK = p.k; // The diagonal row is k even when k >= n.
        std::mt19937 random(p.randomSeed);
        std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
        // Scan the full allowed mean and standard-deviation ranges.
        float mean = -5.0f + static_cast<float>(p.randomSeed % 11);
        float sigma = 0.1f + 0.1f * static_cast<float>((p.randomSeed / 2) % 20);
        std::normal_distribution<float> normal(mean, sigma);
        bool useNormal = p.caseName.rfind("RAW_1_", 0) == 0;
        auto sample = [&]() { return useNormal ? normal(random) : uniform(random); };
        aHost.assign(static_cast<size_t>(allocLda) * allocN, kBlasSentinel);
        for (int col = 0; col < p.n; ++col) {
            int begin = isUpper ? std::max(0, col - p.k) : col;
            int end = isUpper ? col : std::min(p.n - 1, col + p.k);
            for (int row = begin; row <= end; ++row) {
                int bandRow = isUpper ? p.k + row - col : row - col;
                aHost[static_cast<size_t>(col) * allocLda + bandRow] = sample();
            }
        }
        xHost.assign(xSize, kBlasSentinel);
        for (int i = 0; i < p.n; ++i) {
            size_t offset = static_cast<size_t>(p.incx > 0 ? i : p.n - 1 - i) * absIncx;
            xHost[offset] = sample();
        }
        golden = xHost;
        needDevA = !aHost.empty();
        needDevX = !xHost.empty();
    }

    void StrengthenDiagonal()
    {
        if (aHost.empty() || p.n <= 0) {
            return;
        }
        const int diagRow = isUpper ? effK : 0;
        for (int j = 0; j < p.n; j++) {
            float& diag = aHost[diagRow + static_cast<size_t>(j) * allocLda];
            diag += (diag >= 0.0f) ? 5.0f : -5.0f;
        }
    }

    void PrepareDevice()
    {
        if (needDevA) {
            size_t aBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(allocN) * sizeof(float);
            aDev.emplace(aBytes);
            aDev->copyFromHost(aHost.data(), aBytes);
        }
        if (needDevX) {
            size_t xBytes = xSize * sizeof(float);
            xDev.emplace(xBytes);
            xDev->copyFromHost(xHost.data(), xBytes);
        }
    }

    aclblasStatus_t CallNpu(aclblasHandle_t handle)
    {
        const float* aPtr = aDev.has_value() ? static_cast<const float*>(aDev->ptr()) : nullptr;
        float* xPtr = xDev.has_value() ? static_cast<float*>(xDev->ptr()) : nullptr;
        return aclblasStbsv(handle, p.uplo, p.trans, p.diag, p.n, p.k, aPtr, p.lda, xPtr, p.incx);
    }

    void Verify(aclblasHandle_t handle, aclblasStatus_t ret)
    {
        if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
            EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
            return;
        }
        ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS))
            << "Unexpected NPU error code: " << static_cast<int>(ret);
        if (p.n <= 0) {
            return;
        }
        if (xDev.has_value()) {
            aclrtSynchronizeDevice();
            xDev->copyToHost(xHost.data(), xSize * sizeof(float));
        }
        aclblasStbsv_cpu(handle, p.uplo, p.trans, p.diag, p.n, effK, aHost.data(), allocLda, golden.data(), p.incx);
        const float* outPtr = (p.incx < 0) ? xHost.data() + (p.n - 1) * absIncx : xHost.data();
        const float* goldPtr = (p.incx < 0) ? golden.data() + (p.n - 1) * absIncx : golden.data();
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(p.n));
        constexpr double taskTolerance = 0.0001220703125; // 2^-13, required by the task document.
        cfg.mixedRtol = taskTolerance;
        cfg.mixedAtol = taskTolerance;
        cfg.mixedRequiredMatchedRatio = 0.99;
        cfg.mixedMaxAbsErrorLimit = 1.0e-2;
        EXPECT_TRUE(Verifier::verifyVector(
            outPtr, goldPtr, static_cast<size_t>(p.n), (p.incx < 0) ? -absIncx : absIncx, cfg, p.caseName));
    }

    void MeasureDevelopmentEvents(aclblasHandle_t handle, aclrtStream stream, const std::vector<float>& originalInput)
    {
        // Optional diagnostic, not the mandated msprof op acceptance metric.
        // Restore b for every launch; never benchmark repeated solves of x.
        struct ScopedEvent {
            aclrtEvent value = nullptr;
            ~ScopedEvent()
            {
                if (value != nullptr)
                    aclrtDestroyEvent(value);
            }
        };
        ScopedEvent start;
        ScopedEvent end;
        ASSERT_EQ(aclrtCreateEventExWithFlag(&start.value, ACL_EVENT_TIME_LINE), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateEventExWithFlag(&end.value, ACL_EVENT_TIME_LINE), ACL_SUCCESS);
        constexpr int warmup = 5;
        constexpr int samples = 15;
        double totalUs = 0.0;
        std::cout << "[DEV-EVENT " << p.caseName << "] samples_us=";
        for (int iteration = -warmup; iteration < samples; ++iteration) {
            xDev->copyFromHost(originalInput.data(), xSize * sizeof(float));
            ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
            ASSERT_EQ(aclrtRecordEvent(start.value, stream), ACL_SUCCESS);
            ASSERT_EQ(CallNpu(handle), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtRecordEvent(end.value, stream), ACL_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
            float elapsedMs = 0.0f;
            ASSERT_EQ(aclrtEventElapsedTime(&elapsedMs, start.value, end.value), ACL_SUCCESS);
            ASSERT_TRUE(std::isfinite(elapsedMs) && elapsedMs > 0.0f);
            if (iteration >= 0) {
                double us = static_cast<double>(elapsedMs) * 1000.0;
                totalUs += us;
                std::cout << (iteration == 0 ? "" : ",") << std::fixed << std::setprecision(3) << us;
            }
        }
        std::cout << " avg_us=" << totalUs / samples << " NOT_FORMAL_ACCEPTANCE\n";
        // Recheck the final timed output against the already computed golden.
        xDev->copyToHost(xHost.data(), xSize * sizeof(float));
        const float* out = p.incx < 0 ? xHost.data() + (p.n - 1) * absIncx : xHost.data();
        const float* gold = p.incx < 0 ? golden.data() + (p.n - 1) * absIncx : golden.data();
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, gold, static_cast<size_t>(p.n));
        cfg.mixedRtol = cfg.mixedAtol = 0.0001220703125;
        cfg.mixedRequiredMatchedRatio = 0.99;
        cfg.mixedMaxAbsErrorLimit = 1.0e-2;
        EXPECT_TRUE(Verifier::verifyVector(
            out, gold, static_cast<size_t>(p.n), p.incx < 0 ? -absIncx : absIncx, cfg, p.caseName + "_timed"));
    }
};

TEST_F(StbsvArch22Test, NullHandle)
{
    aclblasStatus_t ret =
        aclblasStbsv(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, 0, nullptr, 1, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(StbsvArch22Test, RejectIntMinStrideWithNonNullBuffers)
{
    EXPECT_EQ(
        aclblasStbsv(
            StbsvArch22Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 0, 0, nullptr, 1, nullptr,
            INT_MIN),
        ACLBLAS_STATUS_INVALID_VALUE);
    constexpr int maxN = 16;
    constexpr int lda = 5;
    std::vector<float> a(lda * maxN, 2.0f);
    std::vector<float> x(maxN, 4.0f);
    DeviceBuffer aDev(a.size() * sizeof(float));
    DeviceBuffer xDev(x.size() * sizeof(float));
    aDev.copyFromHost(a.data(), a.size() * sizeof(float));
    xDev.copyFromHost(x.data(), x.size() * sizeof(float));
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            for (auto diag : {ACLBLAS_NON_UNIT, ACLBLAS_UNIT}) {
                for (int n : {0, 1, 2, maxN}) {
                    for (int k : {0, 1, 4}) {
                        ASSERT_EQ(
                            aclblasStbsv(
                                StbsvArch22Test::handle_, uplo, trans, diag, n, k, aDev.floatPtr(), lda,
                                xDev.floatPtr(), INT_MIN),
                            ACLBLAS_STATUS_INVALID_VALUE)
                            << "n=" << n << " k=" << k;
                    }
                }
            }
        }
    }
    ASSERT_EQ(aclrtSynchronizeStream(StbsvArch22Test::stream_), ACL_SUCCESS);
    std::vector<float> actual(maxN);
    xDev.copyToHost(actual.data(), actual.size() * sizeof(float));
    EXPECT_EQ(actual, x);
}

TEST_F(StbsvArch22Test, UnitDiagonalSpecialValuesAreIgnored)
{
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            for (float diagonal : {INFINITY, NAN}) {
                std::vector<float> a(6, kBlasSentinel);
                a[uplo == ACLBLAS_UPPER ? 1 : 0] = diagonal;
                a[uplo == ACLBLAS_UPPER ? 4 : 3] = diagonal;
                a[uplo == ACLBLAS_UPPER ? 3 : 1] = 1.0f;
                std::vector<float> b{2.0f, 5.0f};
                DeviceBuffer aDev(a.size() * sizeof(float));
                DeviceBuffer xDev(b.size() * sizeof(float));
                aDev.copyFromHost(a.data(), a.size() * sizeof(float));
                xDev.copyFromHost(b.data(), b.size() * sizeof(float));
                ASSERT_EQ(
                    aclblasStbsv(
                        StbsvArch22Test::handle_, uplo, trans, ACLBLAS_UNIT, 2, 1, aDev.floatPtr(), 3, xDev.floatPtr(),
                        1),
                    ACLBLAS_STATUS_SUCCESS);
                ASSERT_EQ(aclrtSynchronizeStream(StbsvArch22Test::stream_), ACL_SUCCESS);
                xDev.copyToHost(b.data(), b.size() * sizeof(float));
                bool upperSolve = (uplo == ACLBLAS_UPPER) == (trans == ACLBLAS_OP_N);
                EXPECT_EQ(b[0], upperSolve ? -3.0f : 2.0f);
                EXPECT_EQ(b[1], upperSolve ? 5.0f : 3.0f);
            }
        }
    }
}

TEST_F(StbsvArch22Test, TinyDiagonalAndRhs)
{
    for (float tiny : {std::numeric_limits<float>::min(), std::numeric_limits<float>::denorm_min()}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                std::vector<float> a(6, kBlasSentinel);
                a[uplo == ACLBLAS_UPPER ? 1 : 0] = tiny;
                a[uplo == ACLBLAS_UPPER ? 4 : 3] = tiny;
                a[uplo == ACLBLAS_UPPER ? 3 : 1] = tiny;
                std::vector<float> b{tiny, tiny};
                auto golden = b;
                DeviceBuffer aDev(a.size() * sizeof(float));
                DeviceBuffer xDev(b.size() * sizeof(float));
                aDev.copyFromHost(a.data(), a.size() * sizeof(float));
                xDev.copyFromHost(b.data(), b.size() * sizeof(float));
                ASSERT_EQ(
                    aclblasStbsv(
                        StbsvArch22Test::handle_, uplo, trans, ACLBLAS_NON_UNIT, 2, 1, aDev.floatPtr(), 3,
                        xDev.floatPtr(), 1),
                    ACLBLAS_STATUS_SUCCESS);
                ASSERT_EQ(aclrtSynchronizeStream(StbsvArch22Test::stream_), ACL_SUCCESS);
                xDev.copyToHost(b.data(), b.size() * sizeof(float));
                aclblasStbsv_cpu(
                    StbsvArch22Test::handle_, uplo, trans, ACLBLAS_NON_UNIT, 2, 1, a.data(), 3, golden.data(), 1);
                EXPECT_EQ(b[0], golden[0]);
                EXPECT_EQ(b[1], golden[1]);
            }
        }
    }
}

TEST_F(StbsvArch22Test, VectorExponentBoundaries)
{
    constexpr int n = 33;
    constexpr int k = 16;
    constexpr int lda = 19;
    for (int exponent : {-120, 0, 100, 123}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                std::vector<float> a(lda * n, kBlasSentinel);
                for (int col = 0; col < n; ++col) {
                    for (int row = 0; row < n; ++row) {
                        int delta = row - col;
                        if ((uplo == ACLBLAS_UPPER && delta <= 0 && delta >= -k) ||
                            (uplo == ACLBLAS_LOWER && delta >= 0 && delta <= k)) {
                            int bandRow = uplo == ACLBLAS_UPPER ? k + delta : delta;
                            a[col * lda + bandRow] = row == col ? NAN : 0.03125f * float((row * 7 + col * 3) % 9 - 4);
                        }
                    }
                }
                std::vector<float> b(n);
                for (int i = 0; i < n; ++i) {
                    b[i] = std::ldexp(float((i * 5) % 13 - 6) * 0.25f, exponent);
                }
                auto golden = b;
                DeviceBuffer aDev(a.size() * sizeof(float));
                DeviceBuffer xDev(b.size() * sizeof(float));
                aDev.copyFromHost(a.data(), a.size() * sizeof(float));
                xDev.copyFromHost(b.data(), b.size() * sizeof(float));
                ASSERT_EQ(
                    aclblasStbsv(
                        StbsvArch22Test::handle_, uplo, trans, ACLBLAS_UNIT, n, k, aDev.floatPtr(), lda,
                        xDev.floatPtr(), 1),
                    ACLBLAS_STATUS_SUCCESS);
                ASSERT_EQ(aclrtSynchronizeStream(StbsvArch22Test::stream_), ACL_SUCCESS);
                xDev.copyToHost(b.data(), b.size() * sizeof(float));
                aclblasStbsv_cpu(
                    StbsvArch22Test::handle_, uplo, trans, ACLBLAS_UNIT, n, k, a.data(), lda, golden.data(), 1);
                for (int i = 0; i < n; ++i) {
                    EXPECT_EQ(b[i], golden[i]) << "exponent=" << exponent << " index=" << i;
                }
            }
        }
    }
}

static StbsvParam MakeSpecialValueParam(
    int n, int k, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int incx)
{
    StbsvParam p(csv_map{});
    p.n = n;
    p.k = k;
    p.lda = k + 3;
    p.uplo = uplo;
    p.trans = trans;
    p.diag = diag;
    p.incx = incx;
    return p;
}

static void RunSpecialValueDevice(
    const StbsvParam& p, const std::vector<float>& a, std::vector<float>& b, aclblasHandle_t handle, aclrtStream stream)
{
    DeviceBuffer aDev(a.size() * sizeof(float));
    DeviceBuffer xDev(b.size() * sizeof(float));
    aDev.copyFromHost(a.data(), a.size() * sizeof(float));
    xDev.copyFromHost(b.data(), b.size() * sizeof(float));
    ASSERT_EQ(
        aclblasStbsv(handle, p.uplo, p.trans, p.diag, p.n, p.k, aDev.floatPtr(), p.lda, xDev.floatPtr(), p.incx),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    xDev.copyToHost(b.data(), b.size() * sizeof(float));
}

static void ExpectSpecialValueOutput(const std::vector<float>& actual, const std::vector<float>& golden, int k)
{
    ASSERT_EQ(actual.size(), golden.size());
    // Check every active element and padding slot, without a ratio cutoff.
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(golden[i])) {
            EXPECT_TRUE(std::isnan(actual[i])) << "k=" << k << " index=" << i;
        } else {
            EXPECT_EQ(actual[i], golden[i]) << "k=" << k << " index=" << i;
        }
    }
}

TEST_F(StbsvArch22Test, NanPropagationThroughZeroBand)
{
    constexpr int n = 17;
    for (int k : {0, 1, 8}) {
        int lda = k + 3;
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                for (auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                    for (int incx : {1, -3}) {
                        for (int nanRow : {0, n / 2, n - 1}) {
                            std::vector<float> a(lda * n, 0.0f);
                            int diagonal = uplo == ACLBLAS_UPPER ? k : 0;
                            for (int col = 0; col < n; ++col) {
                                a[col * lda + diagonal] = diag == ACLBLAS_UNIT ? NAN : 1.0f;
                            }
                            int stride = std::abs(incx);
                            std::vector<float> b(1 + (n - 1) * stride, 12345.0f);
                            for (int row = 0; row < n; ++row) {
                                int offset = (incx > 0 ? row : n - 1 - row) * stride;
                                b[offset] = row == nanRow ? NAN : static_cast<float>(row % 3 - 1);
                            }
                            auto golden = b;
                            const auto p = MakeSpecialValueParam(n, k, uplo, trans, diag, incx);
                            ASSERT_NO_FATAL_FAILURE(RunSpecialValueDevice(p, a, b, handle_, stream_));
                            aclblasStbsv_cpu(
                                StbsvArch22Test::handle_, uplo, trans, diag, n, k, a.data(), lda, golden.data(), incx);
                            ASSERT_NO_FATAL_FAILURE(ExpectSpecialValueOutput(b, golden, k));
                        }
                    }
                }
            }
        }
    }
}

static void VerifyZeroPivotCase(
    const StbsvParam& p, float zero, float special, aclblasHandle_t handle, aclrtStream stream)
{
    SCOPED_TRACE(
        ::testing::Message() << "n=" << p.n << " k=" << p.k << " uplo=" << p.uplo << " trans=" << p.trans
                             << " diag=" << p.diag << " incx=" << p.incx << " negative_zero=" << std::signbit(zero)
                             << " nan_diagonal=" << std::isnan(special));
    std::vector<float> a(p.lda * p.n, 0.0f);
    const int diagonal = p.uplo == ACLBLAS_UPPER ? p.k : 0;
    for (int col = 0; col < p.n; ++col) {
        a[col * p.lda + diagonal] = special;
    }
    const int stride = std::abs(p.incx);
    std::vector<float> b(1 + (p.n - 1) * stride, 12345.0f);
    for (int row = 0; row < p.n; ++row) {
        b[row * stride] = zero;
    }
    auto golden = b;
    ASSERT_EQ(
        aclblasStbsv_cpu(handle, p.uplo, p.trans, p.diag, p.n, p.k, a.data(), p.lda, golden.data(), p.incx),
        ACLBLAS_STATUS_SUCCESS);
    const bool expectNaN = p.diag == ACLBLAS_NON_UNIT && p.trans != ACLBLAS_OP_N;
    for (int row = 0; row < p.n; ++row) {
        const float expected = golden[row * stride];
        if (expectNaN) {
            ASSERT_TRUE(std::isnan(expected)) << "golden row=" << row;
        } else {
            ASSERT_EQ(expected, 0.0f) << "golden row=" << row;
        }
    }
    ASSERT_NO_FATAL_FAILURE(RunSpecialValueDevice(p, a, b, handle, stream));
    ASSERT_NO_FATAL_FAILURE(ExpectSpecialValueOutput(b, golden, p.k));
}

static void RunZeroPivotModes(int n, int k, aclblasFillMode_t uplo, aclblasHandle_t handle, aclrtStream stream)
{
    for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
        for (auto diag : {ACLBLAS_NON_UNIT, ACLBLAS_UNIT}) {
            for (int incx : {1, -3}) {
                const auto p = MakeSpecialValueParam(n, k, uplo, trans, diag, incx);
                for (float zero : {0.0f, -0.0f}) {
                    for (float special : {0.0f, std::numeric_limits<float>::quiet_NaN()}) {
                        ASSERT_NO_FATAL_FAILURE(VerifyZeroPivotCase(p, zero, special, handle, stream));
                    }
                }
            }
        }
    }
}

TEST_F(StbsvArch22Test, ZeroPivotSpecialDiagonalMatchesNetlib)
{
    // Tiny/global, cached, owned, narrow-window and over-budget/global shapes.
    // Netlib skips the NO_TRANS divide for a zero pivot, but not the T/C divide.
    const int shapes[][2] = {{1, 0}, {17, 1}, {1024, 256}, {3001, 8}, {32769, 1}};
    for (const auto& shape : shapes) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            ASSERT_NO_FATAL_FAILURE(RunZeroPivotModes(shape[0], shape[1], uplo, handle_, stream_));
        }
    }
}

TEST_F(StbsvArch22Test, ScalarDiagonalCostDiagnostic)
{
    // k=0 isolates the sequential pivot load/store and diagonal division.
    // Controlled diagonal is diagnostic-only; never replaces task CSV inputs.
    for (int n : {1, 512, 2048}) {
        for (const char* diag : {"UNIT", "NON_UNIT"}) {
            std::string name = "DEV_SCALAR_DIAGONAL_" + std::to_string(n) + "_" + diag;
            csv_map values{
                {"case_name", name},
                {"description", "scalar_diagonal_cost_diagnostic"},
                {"uplo", "LOWER"},
                {"trans", "N"},
                {"diag", diag},
                {"n", std::to_string(n)},
                {"k", "0"},
                {"lda", "1"},
                {"incx", "1"},
                {"random_seed", "20261200"},
                {"expect_result", "ACLBLAS_STATUS_SUCCESS"}};
            StbsvParam parameter(values);
            StbsvFixture fixture(parameter);
            fixture.InitDims();
            fixture.PrepareHostData();
            std::fill(fixture.aHost.begin(), fixture.aHost.end(), 2.0f);
            auto original = fixture.xHost;
            fixture.PrepareDevice();
            fixture.Verify(handle_, fixture.CallNpu(handle_));
            ASSERT_FALSE(HasFatalFailure());
            if (std::getenv("STBSV_EVENT_PERF") != nullptr)
                fixture.MeasureDevelopmentEvents(handle_, stream_, original);
        }
    }
}

struct FmaTriple {
    float a, alpha, c;
};

static constexpr FmaTriple midpointCases[] = {
    {0x1.8p-7f, -0x1.64c49ep+38f, 0x1.0b9376p+31f},
    {0x1.f8a5c8p+2f, -0x1.fp+44f, 0x1.e8e09ap+46f},
    {0x1.2c883p-6f, 0x1.dp+36f, -0x1.105b6cp+30f},
    {-0x1.f8p+7f, -0x1.055eep+54f, -0x1.50517ap-7f},
    {0x1.c4p-17f, -0x1.99f64p+54f, 0x1.69eb64p+37f},
    {-0x1.c44p-14f, 0x1.b624p-7f, 0x1.83028cp-21f},
    {0x1.4fbe06p+12f, 0x1.4p+77f, 0x1.38f17p+3f},
    {-0x1.85d34p+19f, -0x1.d4p+86f, 0x1.53f88p+55f},
    {-0x1.85d34p+19f, -0x1.d4p+86f, -0x1.645318p+105f},
    {0x1.14833p+6f, -0x1.fp+72f, 0x1.0bdf16p+78f},
    {0x1.c9adp-2f, 0x1.3a8p-37f, -0x1.192184p-39f},
    {-0x1.8718p-13f, 0x1.c34p+55f, 0x1.29ea6ap-41f},
    // Tiny addends on either side of a true product midpoint must still
    // select exact FMA, even if hardware flushes the subnormal operand.
    {0x1.000002p+0f, 0x1.8p+0f, 0x1p-149f},
    {0x1.000002p+0f, 0x1.8p+0f, -0x1p-149f},
    {-0x1.000002p+0f, 0x1.8p+0f, 0x1p-149f},
    {0x1p-20f, 0x1p-40f, -0x1p-149f},
    {0x1.000002p+0f, 0x1.8p+100f, 0x1p-149f},
    {0x1.000002p+0f, 0x1.8p+100f, -0x1p-149f},
    {-0x1.000002p+0f, 0x1.8p+100f, 0x1p-149f},
    {0x1p-20f, 0x1p+91f, -0x1p-149f},
};
static constexpr int midpointCaseCount = sizeof(midpointCases) / sizeof(midpointCases[0]);

static uint32_t FloatBits(float value)
{
    uint32_t out = 0;
    EXPECT_EQ(memcpy_s(&out, sizeof(out), &value, sizeof(value)), EOK);
    return out;
}

static float SampleExponentFloat(std::mt19937& rng, int minimum, int maximum)
{
    uint32_t raw =
        (rng() & 0x807fffff) | (static_cast<uint32_t>(127 + minimum + rng() % (maximum - minimum + 1)) << 23);
    float out = 0.0f;
    EXPECT_EQ(memcpy_s(&out, sizeof(out), &raw, sizeof(raw)), EOK);
    return out;
}

static void VerifyStbsvCase(const csv_map& values, aclblasHandle_t handle)
{
    StbsvParam parameter(values);
    StbsvFixture fixture(parameter);
    fixture.InitDims();
    fixture.PrepareHostData();
    fixture.PrepareDevice();
    fixture.Verify(handle, fixture.CallNpu(handle));
}

// Isolate a vector-length update: later zero coefficients cannot hide rounding errors.
static void VerifyFmaMidpoint(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, aclblasOperation_t trans, int group,
    std::mt19937& rng)
{
    constexpr int n = 33;
    constexpr int k = 32;
    constexpr int lda = 37;
    bool forward = (uplo == ACLBLAS_LOWER) == (trans == ACLBLAS_OP_N);
    int pivot = forward ? 0 : n - 1;
    float alpha = group < midpointCaseCount ? midpointCases[group].alpha :
                                              std::ldexp(1.0f, (-40 + (group - midpointCaseCount) % 8 * 23));
    std::vector<float> a(lda * n, 0.0f);
    std::vector<float> x(n, 0.0f);
    for (int col = 0; col < n; ++col)
        a[col * lda + (uplo == ACLBLAS_UPPER ? k : 0)] = NAN;
    x[pivot] = -alpha;
    for (int lane = 0; lane < n - 1; ++lane) {
        int target = forward ? lane + 1 : lane;
        int aRow = trans == ACLBLAS_OP_N ? target : pivot;
        int aCol = trans == ACLBLAS_OP_N ? pivot : target;
        int bandRow = uplo == ACLBLAS_UPPER ? k + aRow - aCol : aRow - aCol;
        a[aCol * lda + bandRow] = group < midpointCaseCount ?
                                      midpointCases[group].a :
                                      SampleExponentFloat(rng, -20, std::fabs(alpha) > 0x1p110f ? 1 : 19);
        x[target] = group < midpointCaseCount ? midpointCases[group].c : SampleExponentFloat(rng, -60, 109);
    }
    auto golden = x;
    DeviceBuffer aDev(a.size() * sizeof(float));
    DeviceBuffer xDev(x.size() * sizeof(float));
    aDev.copyFromHost(a.data(), a.size() * sizeof(float));
    xDev.copyFromHost(x.data(), x.size() * sizeof(float));
    ASSERT_EQ(
        aclblasStbsv(handle, uplo, trans, ACLBLAS_UNIT, n, k, aDev.floatPtr(), lda, xDev.floatPtr(), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    xDev.copyToHost(x.data(), x.size() * sizeof(float));
    aclblasStbsv_cpu(handle, uplo, trans, ACLBLAS_UNIT, n, k, a.data(), lda, golden.data(), 1);
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(FloatBits(x[i]), FloatBits(golden[i]))
            << "group=" << group << " index=" << i << " uplo=" << uplo << " trans=" << trans;
    }
}

static void RunMidpointStress(aclblasHandle_t handle, aclrtStream stream)
{
    std::mt19937 rng(20261005);
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            for (int group = 0; group < midpointCaseCount + 64; ++group) {
                VerifyFmaMidpoint(handle, stream, uplo, trans, group, rng);
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
            }
        }
    }
}

static void RunNormalizationTransitions(aclblasHandle_t handle, aclrtStream stream)
{
    // Alternate scaled/unscaled pivots while update lengths grow/shrink. A
    // persistent mask reusing normalization scratch must restore every lane,
    // including lanes outside the first short update after a scaled pivot.
    constexpr int transitionN = 513;
    constexpr int transitionK = 129;
    constexpr int transitionLda = 137;
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            std::vector<float> a(transitionLda * transitionN, 0.0f);
            std::vector<float> x(transitionN);
            for (int row = 0; row < transitionN; ++row) {
                a[row * transitionLda + (uplo == ACLBLAS_UPPER ? transitionK : 0)] = 1.0f;
                x[row] = row % 3 == 0 ? 0x1.8p100f : row % 3 == 1 ? -0x1.8p0f : 0x1.4p-30f;
            }
            auto golden = x;
            DeviceBuffer aDev(a.size() * sizeof(float));
            DeviceBuffer xDev(x.size() * sizeof(float));
            aDev.copyFromHost(a.data(), a.size() * sizeof(float));
            xDev.copyFromHost(x.data(), x.size() * sizeof(float));
            ASSERT_EQ(
                aclblasStbsv(
                    handle, uplo, trans, ACLBLAS_NON_UNIT, transitionN, transitionK, aDev.floatPtr(), transitionLda,
                    xDev.floatPtr(), 1),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
            xDev.copyToHost(x.data(), x.size() * sizeof(float));
            aclblasStbsv_cpu(
                handle, uplo, trans, ACLBLAS_NON_UNIT, transitionN, transitionK, a.data(), transitionLda, golden.data(),
                1);
            for (int row = 0; row < transitionN; ++row) {
                ASSERT_EQ(FloatBits(x[row]), FloatBits(golden[row]))
                    << "transition row=" << row << " uplo=" << uplo << " trans=" << trans;
            }
        }
    }
}

static void RunRawRecurrenceStress(aclblasHandle_t handle)
{
    // Extra RAW recurrence seeds, without conditioning input or changing limits.
    for (uint32_t seed = 20261100; seed < 20261164; ++seed) {
        std::string name = "RAW_" + std::to_string(seed % 2) + "_MIDPOINT_STRESS_" + std::to_string(seed);
        csv_map values{
            {"case_name", name},
            {"description", "raw_recurrence_stress"},
            {"uplo", seed % 4 < 2 ? "UPPER" : "LOWER"},
            {"trans", seed % 3 == 0 ? "N" : "T"},
            {"diag", seed % 5 == 0 ? "UNIT" : "NON_UNIT"},
            {"n", seed % 2 ? "513" : "1025"},
            {"k", seed % 4 < 2 ? "32" : "64"},
            {"lda", "69"},
            {"incx", seed % 3 ? "1" : "-3"},
            {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
            {"random_seed", std::to_string(seed)}};
        VerifyStbsvCase(values, handle);
        ASSERT_FALSE(::testing::Test::HasFatalFailure()) << name;
    }
}

TEST_F(StbsvArch22Test, CompensatedFmaMidpointAndRawStress)
{
    RunMidpointStress(handle_, stream_);
    ASSERT_FALSE(HasFatalFailure());
    RunNormalizationTransitions(handle_, stream_);
    ASSERT_FALSE(HasFatalFailure());
    RunRawRecurrenceStress(handle_);
    ASSERT_FALSE(HasFatalFailure());
}

struct BoundaryShape {
    int n, k, lda;
    const char* tag;
    const char* description;
};

static void VerifyBoundaryModes(
    const BoundaryShape& shape, std::initializer_list<const char*> transposes, uint32_t& seed, aclblasHandle_t handle)
{
    csv_map values{{"description", shape.description}, {"diag", "NON_UNIT"},
                   {"n", std::to_string(shape.n)},     {"k", std::to_string(shape.k)},
                   {"lda", std::to_string(shape.lda)}, {"expect_result", "ACLBLAS_STATUS_SUCCESS"}};
    for (const auto* uplo : {"UPPER", "LOWER"}) {
        for (const auto* trans : transposes) {
            for (int incx : {1, -3}) {
                for (int distribution : {0, 1}) {
                    std::string name = "RAW_" + std::to_string(distribution) + shape.tag + std::to_string(seed);
                    values["case_name"] = name;
                    values["uplo"] = uplo;
                    values["trans"] = trans;
                    values["incx"] = std::to_string(incx);
                    values["random_seed"] = std::to_string(seed++);
                    VerifyStbsvCase(values, handle);
                    ASSERT_FALSE(::testing::Test::HasFatalFailure()) << name;
                }
            }
        }
    }
}

TEST_F(StbsvArch22Test, TransposeCacheWindowBoundaries)
{
    // Odd n and both compact/padded lda exercise repeated forward/backward
    // cache-window rebuilds, last-block DMA padding and signed x strides.
    // These are additional RAW tests; the task CSV and distributions stay intact.
    uint32_t seed = 20261005;
    for (int lda : {65, 69}) {
        VerifyBoundaryModes({2053, 64, lda, "_CACHE_", "transpose_cache_window_boundaries"}, {"T", "C"}, seed, handle_);
        ASSERT_FALSE(HasFatalFailure());
    }
}

TEST_F(StbsvArch22Test, TransposeEightRowTileBoundaries)
{
    // Force both full-matrix-cache and row-tile paths, including a partial
    // final 8/16/32-row group. RAW data and Netlib tolerances remain unchanged.
    uint32_t seed = 20261700;
    const int shapes[][2] = {{17, 16},    {24, 16},    {25, 16},   {31, 16},   {33, 32},   {65, 64},   {257, 128},
                             {513, 64},   {745, 248},  {383, 193}, {749, 479}, {1025, 32}, {65, 33},   {97, 33},
                             {1493, 455}, {1509, 500}, {513, 128}, {641, 200}, {769, 257}, {513, 129}, {519, 129},
                             {1543, 500}, {186, 177},  {3001, 8},  {4096, 8},  {4097, 15}, {15873, 8}};
    for (const auto& shape : shapes) {
        for (int lda : {shape[1] + 1, shape[1] + 4, shape[1] + 1 + (40000 + shape[0] - 1) / shape[0]}) {
            VerifyBoundaryModes(
                {shape[0], shape[1], lda, "_TILE_", "eight_row_tile_boundaries"}, {"N", "T", "C"}, seed, handle_);
            ASSERT_FALSE(HasFatalFailure());
        }
    }
}

TEST_P(StbsvArch22Test, CsvDriven)
{
    StbsvFixture f(GetParam());
    f.InitDims();
    f.PrepareHostData();
    f.PrepareDevice();
    bool measureEvents = std::getenv("STBSV_EVENT_PERF") != nullptr && f.p.caseName.find("TC_PF_") == 0 &&
                         f.p.expectResult == ACLBLAS_STATUS_SUCCESS && f.p.n > 0 && f.xDev.has_value();
    auto originalInput = measureEvents ? f.xHost : std::vector<float>{};
    aclblasStatus_t ret = f.CallNpu(StbsvArch22Test::handle_);
    f.Verify(StbsvArch22Test::handle_, ret);
    if (measureEvents && ret == ACLBLAS_STATUS_SUCCESS && !HasFatalFailure()) {
        f.MeasureDevelopmentEvents(StbsvArch22Test::handle_, StbsvArch22Test::stream_, originalInput);
    }
}
