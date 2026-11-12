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
#include <cblas.h>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "blas_test.h"
#include "chemv_check.h"
#include "chemv_npu_wrapper.h"
#include "chemv_param.h"

namespace {

inline bool IsZero(const aclblasComplex& z) { return z.real == 0.0f && z.imag == 0.0f; }

inline bool IsOne(const aclblasComplex& z) { return z.real == 1.0f && z.imag == 0.0f; }

size_t VectorSpan(int n, int inc)
{
    if (n <= 0) {
        return 0U;
    }
    const int64_t absInc = inc >= 0 ? static_cast<int64_t>(inc) : -static_cast<int64_t>(inc);
    return 1U + static_cast<size_t>(n - 1) * static_cast<size_t>(absInc);
}

size_t LogicalOffset(int i, int n, int inc)
{
    const int64_t absInc = inc >= 0 ? static_cast<int64_t>(inc) : -static_cast<int64_t>(inc);
    return inc > 0 ? static_cast<size_t>(i) * static_cast<size_t>(absInc) :
                     static_cast<size_t>(n - 1 - i) * static_cast<size_t>(absInc);
}

std::vector<aclblasComplex> GenerateComplexBuffer(size_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (count == 0U || fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    if (count > 64U * 1024U * 1024U) {
        throw std::runtime_error("Chemv test allocation exceeds 512 MiB budget");
    }
    return makeBlasComplexMatrix(static_cast<int>(count), 1, static_cast<int>(count), fill, seed);
}

void PoisonUnreferencedMatrixEntries(std::vector<aclblasComplex>& a, aclblasFillMode_t uplo, int n, int lda)
{
    if (n <= 0 || lda < n || a.size() < static_cast<size_t>(lda) * static_cast<size_t>(n)) {
        return;
    }
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    const float infVal = std::numeric_limits<float>::infinity();
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < lda; ++row) {
            const size_t idx = static_cast<size_t>(col) * static_cast<size_t>(lda) + static_cast<size_t>(row);
            if (row >= n || (uplo == ACLBLAS_UPPER ? row > col : row < col)) {
                a[idx] = {nanVal, nanVal};
            } else if (row == col) {
                a[idx].imag = (col % 2 == 0) ? nanVal : infVal;
            }
        }
    }
}

std::vector<aclblasComplex> GeneratePerfComplexBuffer(size_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (count == 0U || fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    constexpr size_t kPeriod = 8191U;
    if (count <= kPeriod) {
        return GenerateComplexBuffer(count, fill, seed);
    }
    auto pattern = GenerateComplexBuffer(kPeriod, fill, seed);
    std::vector<aclblasComplex> out(count);
    size_t pos = 0U;
    while (pos < count) {
        const size_t chunk = std::min(kPeriod, count - pos);
        std::copy_n(pattern.begin(), chunk, out.begin() + static_cast<std::ptrdiff_t>(pos));
        pos += chunk;
    }
    return out;
}

std::string SiblingFilePath(const std::string& baseCsvPath, const std::string& fileName)
{
    const size_t pos = baseCsvPath.find_last_of("/\\");
    return (pos == std::string::npos ? std::string() : baseCsvPath.substr(0, pos + 1U)) + fileName;
}

std::vector<ChemvParam> LoadMainCases()
{
    const std::string csvPath = ReplaceFileExtension2Csv(__FILE__);
    auto cases = GetCasesFromCsv<ChemvParam>(csvPath);
    struct BaselineRow {
        csv_map fields;
        explicit BaselineRow(const csv_map& m) : fields(m) {}
    };
    const auto baselines = GetCasesFromCsv<BaselineRow>(SiblingFilePath(csvPath, "gpu_baseline.csv"));
    size_t perfIdx = 0U;
    std::set<std::string> names;
    for (auto& p : cases) {
        if (!names.insert(p.caseName).second) {
            throw std::runtime_error("Duplicate case_name in chemv_test.csv: " + p.caseName);
        }
        if (!p.IsPerformanceCase()) {
            continue;
        }
        if (perfIdx >= baselines.size()) {
            throw std::runtime_error("Missing GPU baseline row for " + p.caseName);
        }
        const auto& b = baselines[perfIdx++].fields;
        if (parseFillMode(ReadMap(b, "uplo")) != p.uplo || ParseIntColumn(b, "n") != p.n ||
            ParseIntColumn(b, "lda") != p.lda || ParseIntColumn(b, "incx") != p.incx ||
            ParseIntColumn(b, "incy") != p.incy) {
            throw std::runtime_error("GPU baseline shape mismatch for " + p.caseName);
        }
        p.gpuBaselineMs = std::stod(ReadMap(b, "gpu_ms"));
    }
    const std::string gaussPath = SiblingFilePath(csvPath, "chemv_gauss_supplement.csv");
    std::ifstream gaussCheck(gaussPath);
    if (gaussCheck.good()) {
        gaussCheck.close();
        auto gaussCases = GetCasesFromCsv<ChemvParam>(gaussPath);
        for (auto& gp : gaussCases) {
            if (!names.insert(gp.caseName).second) {
                throw std::runtime_error("Duplicate case_name in chemv_gauss_supplement.csv: " + gp.caseName);
            }
            cases.push_back(std::move(gp));
        }
    }
    return cases;
}

} // namespace

class ChemvArch22TestBase : public ::testing::Test {
protected:
    static aclblasHandle_t handle_;
    static aclrtStream stream_;

    static void SetUpTestSuite()
    {
        const aclError ret = aclInit(nullptr);
        ASSERT_TRUE(ret == ACL_SUCCESS || ret == ACL_ERROR_REPEAT_INITIALIZE);
        ASSERT_EQ(aclrtSetDevice(TEST_DEVICE_ID), ACL_SUCCESS);
        ASSERT_EQ(aclblasCreate(&handle_), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&stream_), ACL_SUCCESS);
        ASSERT_EQ(aclblasSetStream(handle_, stream_), ACLBLAS_STATUS_SUCCESS);
    }

    static void TearDownTestSuite()
    {
        if (stream_ != nullptr) {
            (void)aclrtSynchronizeStream(stream_);
        }
        if (handle_ != nullptr) {
            (void)aclblasDestroy(handle_);
            handle_ = nullptr;
        }
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        (void)aclrtResetDevice(TEST_DEVICE_ID);
        (void)aclFinalize();
    }

    static void CompareAndReportAccuracy(
        const ChemvParam& p, const std::vector<aclblasComplex>& hA, const std::vector<aclblasComplex>& hX,
        const std::vector<aclblasComplex>& hY, const ChemvGuardedBuffer& dA, const ChemvGuardedBuffer& dX,
        const ChemvGuardedBuffer& dY)
    {
        auto goldenY = hY;
        cblas_chemv(
            CblasColMajor, p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, p.n, &p.alpha,
            hA.empty() ? nullptr : hA.data(), p.lda, hX.empty() ? nullptr : hX.data(), p.incx, &p.beta, goldenY.data(),
            p.incy);

        const auto actualRaw = dY.ReadToHost();
        ChemvComponentStats realStats, imagStats;
        std::vector<bool> writable(actualRaw.size(), false);
        for (int i = 0; i < p.n; ++i) {
            const size_t pos = LogicalOffset(i, p.n, p.incy);
            writable.at(CHEMV_GUARD_ELEMENTS + pos) = true;
            realStats.Observe(actualRaw.at(CHEMV_GUARD_ELEMENTS + pos).real, goldenY.at(pos).real);
            imagStats.Observe(actualRaw.at(CHEMV_GUARD_ELEMENTS + pos).imag, goldenY.at(pos).imag);
        }
        for (size_t idx = 0; idx < actualRaw.size(); ++idx) {
            if (!writable[idx]) {
                ASSERT_TRUE(dY.ElementUnchanged(actualRaw, idx)) << "Modified non-target y index " << idx;
            }
        }
        EXPECT_TRUE(dA.AllUnchanged());
        EXPECT_TRUE(dX.AllUnchanged());

        for (bool isImag : {false, true}) {
            const auto& st = isImag ? imagStats : realStats;
            std::cout << std::setprecision(10) << "[CHEMV_ACC] case=" << p.caseName
                      << " comp=" << (isImag ? "imag" : "real") << " finite=" << st.finite
                      << " matched=" << st.matched << " ratio=" << st.MatchedRatio() << " max_abs=" << st.maxAbs
                      << " nonfinite=" << st.nonfinite << " nonfinite_ok=" << st.nonfiniteMatched << std::endl;
            EXPECT_TRUE(st.Passed());
        }
    }

    static void VerifyAccuracyExecution(const ChemvParam& p, bool useDeviceScalars)
    {
        SCOPED_TRACE(p.caseName);
        const bool dummyAlloc = p.IsQuickOrNegativeCase();
        const size_t aElems = dummyAlloc ? 1U : static_cast<size_t>(p.lda) * static_cast<size_t>(p.n);
        const size_t xElems = dummyAlloc ? 1U : VectorSpan(p.n, p.incx);
        const size_t yElems = dummyAlloc ? 1U : VectorSpan(p.n, p.incy);

        auto hA = GenerateComplexBuffer(aElems, p.aFill, p.randomSeed);
        auto hX = GenerateComplexBuffer(xElems, p.xFill, p.randomSeed + 17U);
        auto hY = GenerateComplexBuffer(yElems, p.yFill, p.randomSeed + 37U);
        if (!dummyAlloc && !hA.empty()) {
            PoisonUnreferencedMatrixEntries(hA, p.uplo, p.n, p.lda);
        }

        ChemvGuardedBuffer dA(hA), dX(hX), dY(hY);
        std::unique_ptr<ChemvScalarDeviceBuffer> dAlpha, dBeta;
        const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
        const aclblasComplex* betaPtr = p.nullBeta ? nullptr : &p.beta;
        if (useDeviceScalars && !dummyAlloc) {
            if (!p.nullAlpha) {
                dAlpha = std::make_unique<ChemvScalarDeviceBuffer>(p.alpha, stream_);
                alphaPtr = dAlpha->Data();
            }
            if (!p.nullBeta) {
                dBeta = std::make_unique<ChemvScalarDeviceBuffer>(p.beta, stream_);
                betaPtr = dBeta->Data();
            }
        }

        const aclblasStatus_t status = aclblasChemv(
            handle_, p.uplo, p.n, alphaPtr, dA.Data(), p.lda, dX.Data(), p.incx, betaPtr, dY.Data(), p.incy);
        ASSERT_EQ(status, p.expectResult);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

        if (status != ACLBLAS_STATUS_SUCCESS || p.n <= 0 || (IsZero(p.alpha) && IsOne(p.beta))) {
            EXPECT_TRUE(dA.AllUnchanged());
            EXPECT_TRUE(dX.AllUnchanged());
            EXPECT_TRUE(dY.AllUnchanged());
            return;
        }
        CompareAndReportAccuracy(p, hA, hX, hY, dA, dX, dY);
    }

    static void RunPerfLoop(
        const ChemvParam& p, const ChemvGuardedBuffer& dA, const ChemvGuardedBuffer& dX, ChemvGuardedBuffer& dY,
        int iterations)
    {
        for (int i = 0; i < iterations; ++i) {
            ASSERT_EQ(
                aclblasChemv(
                    handle_, p.uplo, p.n, &p.alpha, dA.Data(), p.lda, dX.Data(), p.incx, &p.beta, dY.Data(), p.incy),
                ACLBLAS_STATUS_SUCCESS);
        }
    }

    static void VerifyPerformanceExecution(const ChemvParam& p)
    {
        SCOPED_TRACE(p.caseName);
        const size_t aElems = static_cast<size_t>(p.lda) * static_cast<size_t>(p.n);
        const size_t vecElems = static_cast<size_t>(p.n);
        auto hA = GeneratePerfComplexBuffer(aElems, p.aFill, p.randomSeed);
        auto hX = GeneratePerfComplexBuffer(vecElems, p.xFill, p.randomSeed + 17U);
        auto hY = GeneratePerfComplexBuffer(vecElems, p.yFill, p.randomSeed + 37U);
        ChemvGuardedBuffer dA(hA), dX(hX), dY(hY);

        constexpr int kWarmup = 5;
        constexpr int kSamples = 10;
        RunPerfLoop(p, dA, dX, dY, kWarmup);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

        aclrtEvent evStart = nullptr;
        aclrtEvent evEnd = nullptr;
        ASSERT_EQ(aclrtCreateEventWithFlag(&evStart, ACL_EVENT_TIME_LINE), ACL_SUCCESS);
        std::unique_ptr<void, decltype(&aclrtDestroyEvent)> startGuard(evStart, &aclrtDestroyEvent);
        ASSERT_EQ(aclrtCreateEventWithFlag(&evEnd, ACL_EVENT_TIME_LINE), ACL_SUCCESS);
        std::unique_ptr<void, decltype(&aclrtDestroyEvent)> endGuard(evEnd, &aclrtDestroyEvent);

        ASSERT_EQ(aclrtRecordEvent(evStart, stream_), ACL_SUCCESS);
        RunPerfLoop(p, dA, dX, dY, kSamples);
        ASSERT_EQ(aclrtRecordEvent(evEnd, stream_), ACL_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeEvent(evEnd), ACL_SUCCESS);

        float totalMs = 0.0f;
        ASSERT_EQ(aclrtEventElapsedTime(&totalMs, evStart, evEnd), ACL_SUCCESS);
        const double avgUs = (static_cast<double>(totalMs) * 1000.0) / static_cast<double>(kSamples);
        const double baselineUs = p.gpuBaselineMs * 1000.0;
        constexpr double kRequiredRatio = 0.8;
        const double thresholdUs = baselineUs / kRequiredRatio;
        const double ratio = avgUs > 0.0 ? (baselineUs / avgUs) : 0.0;
        const bool passed = (avgUs > 0.0 && avgUs <= thresholdUs);

        std::cout << std::setprecision(8) << "[PERF] case=" << p.caseName
                  << " uplo=" << (p.uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER") << " n=" << p.n << " avg_us=" << avgUs
                  << " baseline_us=" << baselineUs << " ratio=" << ratio << " threshold_us=" << thresholdUs
                  << " verdict=" << (passed ? "PASS" : "FAIL") << std::endl;
        EXPECT_TRUE(passed);
    }
};

aclblasHandle_t ChemvArch22TestBase::handle_ = nullptr;
aclrtStream ChemvArch22TestBase::stream_ = nullptr;

class ChemvArch22CsvTest : public ChemvArch22TestBase, public ::testing::WithParamInterface<ChemvParam> {};

TEST_P(ChemvArch22CsvTest, CsvCases)
{
    const ChemvParam& p = GetParam();
    if (p.IsPerformanceCase()) {
        VerifyPerformanceExecution(p);
    } else {
        VerifyAccuracyExecution(p, (p.randomSeed & 2U) != 0U);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TaskCsv, ChemvArch22CsvTest, ::testing::ValuesIn(LoadMainCases()), PrintCaseInfoString<ChemvParam>);

TEST_F(ChemvArch22TestBase, NullHandlePrecedesAllChecks)
{
    EXPECT_EQ(
        aclblasChemv(
            nullptr, static_cast<aclblasFillMode_t>(999), -1, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChemv(nullptr, ACLBLAS_UPPER, 0, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(ChemvArch22TestBase, DeviceScalarQuickReturnAndScale)
{
    ChemvScalarDeviceBuffer devZero({0.0f, 0.0f}, stream_);
    ChemvScalarDeviceBuffer devOne({1.0f, 0.0f}, stream_);
    ChemvScalarDeviceBuffer devBeta({0.0f, 2.0f}, stream_);

    EXPECT_EQ(
        aclblasChemv(handle_, ACLBLAS_UPPER, 16, devZero.Data(), nullptr, 16, nullptr, 1, devOne.Data(), nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);

    std::vector<aclblasComplex> hY(4, aclblasComplex{3.0f, -1.0f});
    ChemvGuardedBuffer dY(hY);
    ASSERT_EQ(
        aclblasChemv(
            handle_, ACLBLAS_LOWER, 4, devZero.Data(), nullptr, 4, nullptr, 1, devBeta.Data(), dY.Data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    const auto out = dY.ReadToHost();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[CHEMV_GUARD_ELEMENTS + i].real, 2.0f);
        EXPECT_FLOAT_EQ(out[CHEMV_GUARD_ELEMENTS + i].imag, 6.0f);
    }
}

TEST_F(ChemvArch22TestBase, BetaOnePreservesInfiniteY)
{
    const float inf = std::numeric_limits<float>::infinity();
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{1.0f, 0.0f};
    std::vector<aclblasComplex> hA{{2.0f, std::numeric_limits<float>::quiet_NaN()}};
    std::vector<aclblasComplex> hX{{1.5f, 0.0f}};
    std::vector<aclblasComplex> hY{{inf, 4.0f}};
    ChemvGuardedBuffer dA(hA), dX(hX), dY(hY);

    ASSERT_EQ(
        aclblasChemv(handle_, ACLBLAS_UPPER, 1, &alpha, dA.Data(), 1, dX.Data(), 1, &beta, dY.Data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    const auto out = dY.ReadToHost();
    EXPECT_EQ(out[CHEMV_GUARD_ELEMENTS].real, inf);
    EXPECT_FLOAT_EQ(out[CHEMV_GUARD_ELEMENTS].imag, 4.0f);
}

TEST(ChemvComparatorUnitTest, SpecialValuesAndTolerance)
{
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    const float infVal = std::numeric_limits<float>::infinity();
    ChemvComponentStats s;
    s.Observe(nanVal, nanVal);
    s.Observe(infVal, nanVal);
    s.Observe(nanVal, -infVal);
    s.Observe(-infVal, -infVal);
    EXPECT_TRUE(s.Passed());
    s.Observe(infVal, -infVal);
    EXPECT_FALSE(s.Passed());

    ChemvComponentStats mismatch;
    mismatch.Observe(1.0f, nanVal);
    EXPECT_FALSE(mismatch.Passed());
}
