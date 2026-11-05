/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cher_param.h"
#include "cher_golden.h"
#include "cher_npu_wrapper.h"

static constexpr double kCherPerfRatio = 0.4; // NPU <= GPU / 0.4  <=>  GPU/NPU >= 0.4
static constexpr int kCherWarmup = 5;
static constexpr int kCherSamples = 10; // task book §3.3; override via CHER_BENCH_SAMPLES

static bool IsUnsafeIncx(int inc)
{
    return inc == 0 || inc == INT_MIN;
}

static std::vector<aclblasComplex> MakeComplexStrided(
    int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count < 0 || IsUnsafeIncx(inc)) {
        return {};
    }
    std::vector<float> re = makeBlasStrided(count, inc, fill, seed);
    std::vector<float> im = makeBlasStrided(count, inc, fill, seed + 1000U);
    std::vector<aclblasComplex> out(re.size());
    for (size_t i = 0; i < re.size(); ++i) {
        out[i] = aclblasComplex{re[i], im[i]};
    }
    return out;
}

static std::vector<aclblasComplex> MakeComplexMatrixHermitianPrep(
    int n, int lda, bool upper, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || n <= 0 || lda <= 0) {
        return {};
    }
    std::vector<aclblasComplex> a = makeBlasComplexMatrix(n, n, lda, fill, seed);
    // Walk the actual buffer (lda * n). Using i,j in [0,n) with idx=j*lda+i OOBs when lda < n
    // (invalid-lda negative cases still call this helper before expectResult fails).
    (void)upper;
    for (size_t idx = 0; idx < a.size(); ++idx) {
        const int i = static_cast<int>(idx % static_cast<size_t>(lda));
        const int j = static_cast<int>(idx / static_cast<size_t>(lda));
        if (i < n && j < n && i == j) {
            a[idx].imag = 0.0f;
        }
    }
    return a;
}

static std::string CherNormUplo(const std::string& uplo)
{
    if (uplo.rfind("ACLBLAS_", 0) == 0) {
        return uplo.substr(8);
    }
    return uplo;
}

static double CherGpuBaselineUs(const CherParam& p)
{
    static std::map<std::tuple<std::string, int, float, int>, double> byKey;
    static std::vector<double> byIndex;
    static bool loaded = false;
    if (!loaded) {
        const std::string src = __FILE__;
        const std::string path = src.substr(0, src.find_last_of("/\\") + 1) + "gpu_baseline.csv";
        const auto rows = GetCasesFromCsv<csv_map>(path);
        byIndex.reserve(rows.size());
        for (const auto& row : rows) {
            const std::string uplo = CherNormUplo(row.at("uplo"));
            const int n = std::stoi(row.at("n"));
            const float alpha = std::stof(row.at("alpha"));
            const int incx = std::stoi(row.at("incx"));
            const double gpuUs = std::stod(row.at("gpu_ms")) * 1000.0; // ms -> us
            byKey.emplace(std::make_tuple(uplo, n, alpha, incx), gpuUs);
            byIndex.push_back(gpuUs);
        }
        loaded = true;
    }

    const std::string uplo = (p.uplo == ACLBLAS_UPPER) ? "UPPER" : "LOWER";
    auto it = byKey.find(std::make_tuple(uplo, p.n, p.alpha, p.incx));
    if (it != byKey.end()) {
        return it->second;
    }
    if (p.caseName.rfind("TC_PF_", 0) == 0 && p.caseName.size() >= 10) {
        const int idx = std::stoi(p.caseName.substr(6)) - 1001;
        if (idx >= 0 && idx < static_cast<int>(byIndex.size())) {
            return byIndex[static_cast<size_t>(idx)];
        }
    }
    return -1.0;
}

static int CherBenchSampleCount()
{
    if (const char* env = std::getenv("CHER_BENCH_SAMPLES")) {
        return std::max(1, std::atoi(env));
    }
    return kCherSamples;
}

static double MeasureCherNpuUs(
    const CherParam& p, aclblasHandle_t handle, aclrtStream stream, void* dX, void* dA, size_t aBytes,
    const std::vector<aclblasComplex>& aHost, int samples)
{
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    EXPECT_EQ(aclrtCreateEvent(&start), ACL_SUCCESS);
    EXPECT_EQ(aclrtCreateEvent(&stop), ACL_SUCCESS);
    double totalUs = 0.0;
    for (int sample = -kCherWarmup; sample < samples; ++sample) {
        EXPECT_EQ(aclrtMemcpy(dA, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        EXPECT_EQ(aclrtRecordEvent(start, stream), ACL_SUCCESS);
        const aclblasStatus_t st = aclblasCher(
            handle, p.uplo, p.n, &p.alpha, static_cast<const aclblasComplex*>(dX), p.incx,
            static_cast<aclblasComplex*>(dA), p.lda);
        EXPECT_EQ(static_cast<int>(st), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        EXPECT_EQ(aclrtRecordEvent(stop, stream), ACL_SUCCESS);
        EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        float ms = 0.0f;
        EXPECT_EQ(aclrtEventElapsedTime(&ms, start, stop), ACL_SUCCESS);
        if (sample >= 0) {
            totalUs += static_cast<double>(ms) * 1000.0;
        }
    }
    EXPECT_EQ(aclrtDestroyEvent(start), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyEvent(stop), ACL_SUCCESS);
    return totalUs / static_cast<double>(samples);
}

static void BenchmarkCherPerf(const CherParam& p, aclblasHandle_t handle, aclrtStream stream)
{
    ASSERT_GT(p.n, 0);
    ASSERT_NE(p.incx, 0);
    ASSERT_FALSE(p.alphaNull);

    const bool upper = (p.uplo == ACLBLAS_UPPER);
    std::vector<aclblasComplex> xHost = MakeComplexStrided(p.n, p.incx, p.xFill, p.randomSeed);
    std::vector<aclblasComplex> aHost =
        MakeComplexMatrixHermitianPrep(p.n, std::max(1, p.lda), upper, p.aFill, p.randomSeed + 1U);
    ASSERT_FALSE(xHost.empty());
    ASSERT_FALSE(aHost.empty());

    const int absIncx = std::abs(p.incx);
    const size_t xBytes = static_cast<size_t>((p.n - 1) * absIncx + 1) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(std::max(1, p.lda)) * static_cast<size_t>(p.n) * sizeof(aclblasComplex);
    void* dX = nullptr;
    void* dA = nullptr;
    ASSERT_EQ(CherAllocCopy(dX, xHost.data(), xBytes), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(CherAllocCopy(dA, aHost.data(), aBytes), ACLBLAS_STATUS_SUCCESS);

    const int samples = CherBenchSampleCount();
    const double avgUs = MeasureCherNpuUs(p, handle, stream, dX, dA, aBytes, aHost, samples);
    ASSERT_EQ(aclrtFree(dX), ACL_SUCCESS);
    ASSERT_EQ(aclrtFree(dA), ACL_SUCCESS);

    const double gpuUs = CherGpuBaselineUs(p);
    ASSERT_GT(gpuUs, 0.0) << "[" << p.caseName << "] missing gpu_baseline";
    EXPECT_TRUE(std::isfinite(avgUs) && avgUs > 0.0) << "[" << p.caseName << "] bad NPU timing";
    EXPECT_GT(gpuUs, kCherPerfRatio * avgUs)
        << "[" << p.caseName << "] perf fail npu_us=" << avgUs << " gpu_us=" << gpuUs
        << " ratio=" << (avgUs > 0.0 ? (gpuUs / avgUs) : 0.0);
    std::cout << "CHER_PERF," << p.caseName << ",n=" << p.n << ",npu_us=" << avgUs << ",gpu_us=" << gpuUs
              << ",ratio=" << (gpuUs / avgUs) << ",samples=" << samples << std::endl;
}

class CherArch35Test : public BlasTest<CherParam> {};

TEST_F(CherArch35Test, NullHandle)
{
    float alpha = 1.0f;
    aclblasComplex dummy{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCher(nullptr, ACLBLAS_UPPER, 4, &alpha, &dummy, 1, &dummy, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(CherArch35Test, ZeroDimensionQuickReturn)
{
    EXPECT_EQ(
        static_cast<int>(aclblasCher(CherArch35Test::handle_, static_cast<aclblasFillMode_t>(999), 0, nullptr, nullptr,
                                     0, nullptr, 0)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}

TEST_F(CherArch35Test, NegativeDimensionRejected)
{
    const float alpha = 1.0f;
    aclblasComplex value{0.0f, 0.0f};
    EXPECT_EQ(
        static_cast<int>(aclblasCher(CherArch35Test::handle_, ACLBLAS_UPPER, -1, &alpha, &value, 1, &value, 1)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(CherArch35Test, HostZeroAllowsNullData)
{
    const float alpha = 0.0f;
    EXPECT_EQ(
        static_cast<int>(aclblasCher(CherArch35Test::handle_, ACLBLAS_UPPER, 4, &alpha, nullptr, 1, nullptr, 4)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}

TEST_F(CherArch35Test, DeviceZeroAllowsNullData)
{
    const float alpha = 0.0f;
    EXPECT_EQ(
        static_cast<int>(
            aclblasCher_npu(CherArch35Test::handle_, ACLBLAS_LOWER, 4, &alpha, nullptr, 1, nullptr, 4, {true})),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}

TEST_F(CherArch35Test, NonzeroRejectsNullData)
{
    const float alpha = 1.0f;
    aclblasComplex x[4] = {};
    aclblasComplex matrix[16] = {};
    EXPECT_EQ(
        static_cast<int>(aclblasCher_npu(CherArch35Test::handle_, ACLBLAS_UPPER, 4, &alpha, nullptr, 1, matrix, 4)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    EXPECT_EQ(
        static_cast<int>(aclblasCher_npu(CherArch35Test::handle_, ACLBLAS_LOWER, 4, &alpha, x, 1, nullptr, 4)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    EXPECT_EQ(
        static_cast<int>(aclblasCher_npu(CherArch35Test::handle_, ACLBLAS_UPPER, 4, nullptr, x, 1, matrix, 4)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(CherArch35Test, InvalidStructureRejectedBeforeAlphaNoop)
{
    const float zero = 0.0f;
    aclblasComplex x[4] = {};
    aclblasComplex matrix[16] = {};
    EXPECT_EQ(
        static_cast<int>(aclblasCher(
            CherArch35Test::handle_, static_cast<aclblasFillMode_t>(999), 4, &zero, x, 1, matrix, 4)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM));
    EXPECT_EQ(
        static_cast<int>(aclblasCher(CherArch35Test::handle_, ACLBLAS_UPPER, 4, &zero, x, 1, matrix, 2)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    EXPECT_EQ(
        static_cast<int>(aclblasCher(CherArch35Test::handle_, ACLBLAS_UPPER, 4, &zero, x, 0, matrix, 4)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    EXPECT_EQ(
        static_cast<int>(aclblasCher(CherArch35Test::handle_, ACLBLAS_UPPER, 4, &zero, x, INT_MIN, matrix, 4)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(CherArch35Test, DeviceAlphaComputesHermitianUpdate)
{
    // x = {1+i, 2-i}; alpha = 2 (real); A = 0 (lower).
    // A(0,0) = 2*|1+i|^2 = 4; A(1,1) = 2*|2-i|^2 = 10;
    // A(1,0) = alpha*x(1)*conj(x(0)) = 2*(2-i)*(1-i) = 2-6i.
    const float alpha = 2.0f;
    aclblasComplex x[2] = {{1.0f, 1.0f}, {2.0f, -1.0f}};
    aclblasComplex matrix[4] = {};
    ASSERT_EQ(
        static_cast<int>(
            aclblasCher_npu(CherArch35Test::handle_, ACLBLAS_LOWER, 2, &alpha, x, 1, matrix, 2, {true})),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    EXPECT_FLOAT_EQ(matrix[0].real, 4.0f);
    EXPECT_FLOAT_EQ(matrix[0].imag, 0.0f);
    EXPECT_FLOAT_EQ(matrix[1].real, 2.0f);
    EXPECT_FLOAT_EQ(matrix[1].imag, -6.0f);
    EXPECT_FLOAT_EQ(matrix[3].real, 10.0f);
    EXPECT_FLOAT_EQ(matrix[3].imag, 0.0f);
    EXPECT_FLOAT_EQ(matrix[2].real, 0.0f);
    EXPECT_FLOAT_EQ(matrix[2].imag, 0.0f);
}

TEST_F(CherArch35Test, CsvCaseCount)
{
    // Guards against silent CSV shrinkage (dropped/duplicate rows would otherwise
    // leave the suite green with less coverage). Matches MaskHunter review bar.
    const auto cases = GetCasesFromCsv<CherParam>(ReplaceFileExtension2Csv(__FILE__));
    EXPECT_EQ(cases.size(), 1200u);
    size_t uniqueNames = 0;
    size_t perfCases = 0;
    {
        std::map<std::string, int> seen;
        for (const auto& c : cases) {
            seen[c.caseName]++;
            if (c.caseName.rfind("TC_PF_", 0) == 0) {
                ++perfCases;
            }
        }
        uniqueNames = seen.size();
    }
    EXPECT_EQ(uniqueNames, 1200u);
    EXPECT_EQ(perfCases, 200u);
}

INSTANTIATE_TEST_SUITE_P(
    Cher, CherArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CherParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CherParam>);

static bool CherSameFloat(float a, float b)
{
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    if (std::isinf(a) && std::isinf(b)) {
        return (a > 0) == (b > 0);
    }
    return a == b;
}

static void CollectUploVectors(
    bool upper, int n, int lda, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden,
    std::vector<float>& outRe, std::vector<float>& outIm, std::vector<float>& goldRe, std::vector<float>& goldIm)
{
    auto isNonFinite = [](float v) { return std::isnan(v) || std::isinf(v); };
    for (int j = 0; j < n; ++j) {
        const int iStart = upper ? 0 : j;
        const int iEnd = upper ? j : (n - 1);
        for (int i = iStart; i <= iEnd; ++i) {
            const size_t idx = static_cast<size_t>(j) * static_cast<size_t>(lda) + static_cast<size_t>(i);
            float orr = actual[idx].real;
            float ori = actual[idx].imag;
            float gr = golden[idx].real;
            float gi = golden[idx].imag;
            // Inf/NaN pairs are equivalent (TC_FL_108).
            if (isNonFinite(orr) || isNonFinite(ori) || isNonFinite(gr) || isNonFinite(gi)) {
                orr = ori = gr = gi = 0.0f;
            }
            outRe.push_back(orr);
            outIm.push_back(ori);
            goldRe.push_back(gr);
            goldIm.push_back(gi);
        }
    }
}

static void VerifyCherAccuracy(
    const CherParam& p, bool upper, const std::vector<aclblasComplex>& aHost,
    const std::vector<aclblasComplex>& aOrig, const std::vector<aclblasComplex>& golden)
{
    std::vector<float> outRe, outIm, goldRe, goldIm;
    CollectUploVectors(upper, p.n, p.lda, aHost, golden, outRe, outIm, goldRe, goldIm);

    VerifyConfig cfgRe;
    applyMixedTolerance(cfgRe, ACL_FLOAT, goldRe.data(), goldRe.size());
    EXPECT_TRUE(Verifier::verifyVector(outRe.data(), goldRe.data(), outRe.size(), 1, cfgRe, p.caseName + "_re"));
    VerifyConfig cfgIm;
    applyMixedTolerance(cfgIm, ACL_FLOAT, goldIm.data(), goldIm.size());
    EXPECT_TRUE(Verifier::verifyVector(outIm.data(), goldIm.data(), outIm.size(), 1, cfgIm, p.caseName + "_im"));

    double maxDiagImag = 0.0;
    for (int i = 0; i < p.n; ++i) {
        const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(p.lda) + static_cast<size_t>(i);
        maxDiagImag = std::max(maxDiagImag, static_cast<double>(std::fabs(aHost[idx].imag)));
    }
    EXPECT_LE(maxDiagImag, 1.52587890625e-5) << "[" << p.caseName << "] diag imag";

    uint32_t unchErr = 0;
    for (int j = 0; j < p.n; ++j) {
        for (int i = 0; i < p.lda; ++i) {
            const size_t idx = static_cast<size_t>(j) * static_cast<size_t>(p.lda) + static_cast<size_t>(i);
            const bool inUplo = (i < p.n) && (upper ? (i <= j) : (i >= j));
            if (!inUplo && (!CherSameFloat(aHost[idx].real, aOrig[idx].real) ||
                            !CherSameFloat(aHost[idx].imag, aOrig[idx].imag))) {
                unchErr++;
            }
        }
    }
    EXPECT_EQ(unchErr, 0u) << "[" << p.caseName << "] unchanged region errors=" << unchErr;
}

static void RunCherAccuracyCase(const CherParam& p, aclblasHandle_t handle)
{
    const bool upper = (p.uplo == ACLBLAS_UPPER);
    std::vector<aclblasComplex> xHost = MakeComplexStrided(p.n, p.incx, p.xFill, p.randomSeed);
    std::vector<aclblasComplex> aHost =
        MakeComplexMatrixHermitianPrep(p.n, std::max(1, p.lda), upper, p.aFill, p.randomSeed + 1U);
    std::vector<aclblasComplex> aOrig = aHost;

    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();

    const aclblasStatus_t ret = aclblasCher_npu(handle, p.uplo, p.n, alphaPtr, xPtr, p.incx, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (p.n == 0 || aPtr == nullptr || (alphaPtr != nullptr && *alphaPtr == 0.0f)) {
        if (aPtr != nullptr && !aOrig.empty()) {
            for (size_t i = 0; i < aHost.size(); ++i) {
                EXPECT_EQ(aHost[i].real, aOrig[i].real) << p.caseName << " noop real@" << i;
                EXPECT_EQ(aHost[i].imag, aOrig[i].imag) << p.caseName << " noop imag@" << i;
            }
        }
        return;
    }

    std::vector<aclblasComplex> golden = aOrig;
    const aclblasStatus_t gst =
        aclblasCher_cpu(handle, p.uplo, p.n, alphaPtr, xHost.data(), p.incx, golden.data(), p.lda);
    ASSERT_EQ(static_cast<int>(gst), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    for (int i = 0; i < p.n; ++i) {
        const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(p.lda) + static_cast<size_t>(i);
        golden[idx].imag = 0.0f;
    }
    VerifyCherAccuracy(p, upper, aHost, aOrig, golden);
}

TEST_P(CherArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    if (p.caseName.rfind("TC_PF_", 0) == 0) {
        ASSERT_NO_FATAL_FAILURE(BenchmarkCherPerf(p, CherArch35Test::handle_, CherArch35Test::stream_));
        return;
    }
    ASSERT_NO_FATAL_FAILURE(RunCherAccuracyCase(p, CherArch35Test::handle_));
}
