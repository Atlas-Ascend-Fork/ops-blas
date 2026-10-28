/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file csyr2_test.cpp
 * \brief 板测 CSV ST：INSTANTIATE_TEST_SUITE_P 读 csyr2_test.csv。
 *        不 include aclblas_handle_internal.h / log.h，避免 TEST 宏被吃掉。
 *        Host/Kernel 源不改。
 */

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "csyr2_arch22_test.h"
#include "csyr2_golden.h"
#include "csyr2_npu_wrapper.h"

namespace {

std::string Csyr2Dirname(const std::string& p)
{
    const size_t slash = p.find_last_of("/\\");
    if (slash == std::string::npos) {
        return std::string();
    }
    return p.substr(0, slash);
}

std::string Csyr2ExeDir()
{
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return std::string();
    }
    buf[n] = '\0';
    return Csyr2Dirname(std::string(buf, static_cast<size_t>(n)));
}

bool Csyr2FileExists(const std::string& p)
{
    if (p.empty()) {
        return false;
    }
    std::ifstream ifs(p);
    return ifs.good();
}

std::string Csyr2Join(const std::string& dir, const char* name)
{
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/' || dir.back() == '\\') {
        return dir + name;
    }
    return dir + "/" + name;
}

/* CMake copies the case CSV beside the test executable. */
std::string Csyr2CsvPath()
{
    std::vector<std::string> dirs;
    dirs.push_back(Csyr2Dirname(__FILE__));
    dirs.push_back(std::string("."));
    dirs.push_back(Csyr2ExeDir());
    const char* names[] = {"csyr2_test.csv"};
    for (const auto& d : dirs) {
        for (const char* n : names) {
            const std::string cand = Csyr2Join(d, n);
            if (Csyr2FileExists(cand)) {
                std::printf("csyr2 CSV: %s\n", cand.c_str());
                return cand;
            }
        }
    }
    std::printf("csyr2 CSV: NOT FOUND (__FILE__=%s exe=%s)\n", __FILE__, Csyr2ExeDir().c_str());
    return std::string();
}

std::vector<Csyr2Arch22Param> Csyr2CsvHardFail(const char* why)
{
    std::printf("csyr2 CSV: HARD FAIL %s\n", why);
    csv_map row;
    row["case_name"] = "TC_CSV_MISSING";
    row["n"] = "0";
    row["expect_result"] = "ACLBLAS_STATUS_INTERNAL_ERROR";
    return {Csyr2Arch22Param(row)};
}

std::vector<Csyr2Arch22Param> LoadCsyr2Cases()
{
    const std::string path = Csyr2CsvPath();
    if (path.empty()) {
        return Csyr2CsvHardFail("not found next to source/cwd/binary (csyr2_test.csv)");
    }
    try {
        auto cases = GetCasesFromCsv<Csyr2Arch22Param>(path);
        if (cases.empty()) {
            return Csyr2CsvHardFail("opened but 0 cases");
        }
        std::printf("csyr2 CSV: loaded %zu cases\n", cases.size());
        return cases;
    } catch (const std::exception& e) {
        std::printf("csyr2 CSV: load failed: %s\n", e.what());
        return Csyr2CsvHardFail(e.what());
    }
}

std::vector<aclblasComplex> MakeComplexStrided(int n, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    const std::vector<float> re = makeBlasStrided(n, inc, fill, seed);
    const std::vector<float> im = makeBlasStrided(n, inc, fill, seed + 1000U);
    std::vector<aclblasComplex> out(re.size());
    for (size_t i = 0; i < re.size(); ++i) {
        out[i].real = re[i];
        out[i].imag = im[i];
    }
    return out;
}

bool IsPerfCase(const std::string& name) { return name.size() >= 6 && name.compare(0, 6, "TC_PF_") == 0; }

bool SameFloat(float a, float b)
{
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    return a == b;
}

bool SameComplex(const aclblasComplex& a, const aclblasComplex& b)
{
    return SameFloat(a.real, b.real) && SameFloat(a.imag, b.imag);
}

bool InUploTriangle(aclblasFillMode_t uplo, int n, int i, int j)
{
    if (i < 0 || j < 0 || i >= n || j >= n) {
        return false;
    }
    return (uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
}

} // namespace

void Csyr2Arch22Test::RunPerformance(
    const Csyr2Arch22Param& p, const aclblasComplex* xPtr, const aclblasComplex* yPtr, aclblasComplex* aPtr)
{
    Csyr2DeviceBuf dev;
    ASSERT_EQ(Csyr2Upload(dev, p.n, xPtr, p.incx, yPtr, p.incy, aPtr, p.lda), ACLBLAS_STATUS_SUCCESS);
    constexpr int kWarmup = 10;
    /* 任务书 §3.3 要求有效采样 >50 次，默认 51。
       CSYR2_PERF_ITERS 只用于 A/B 对比时降低方差；连发次数越多，
       每次调用的下发开销摊得越薄，与默认口径测的不是同一个量，
       不可用于对照任务书的标杆耗时。 */
    const int kIters = []() {
        const char* e = std::getenv("CSYR2_PERF_ITERS");
        if (e == nullptr) {
            return 51;
        }
        const int v = std::atoi(e);
        return (v > 50) ? v : 51;
    }();
    for (int i = 0; i < kWarmup; ++i) {
        ASSERT_EQ(Csyr2Launch(Csyr2Arch22Test::handle_, p.uplo, p.n, &p.alpha, dev, p.lda), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(Csyr2Arch22Test::stream_), ACL_SUCCESS);
    }
    const auto b0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        ASSERT_EQ(Csyr2Launch(Csyr2Arch22Test::handle_, p.uplo, p.n, &p.alpha, dev, p.lda), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(Csyr2Arch22Test::stream_), ACL_SUCCESS);
    const auto b1 = std::chrono::steady_clock::now();
    const double avgBatched = std::chrono::duration<double, std::micro>(b1 - b0).count() / static_cast<double>(kIters);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        ASSERT_EQ(Csyr2Launch(Csyr2Arch22Test::handle_, p.uplo, p.n, &p.alpha, dev, p.lda), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(Csyr2Arch22Test::stream_), ACL_SUCCESS);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double avgSync = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(kIters);
    std::printf("avg_us=%.4f avg_us_persync=%.4f\n", avgBatched, avgSync);
}

void Csyr2Arch22Test::CheckResult(
    const Csyr2Arch22Param& p, const std::vector<aclblasComplex>& aHost, const std::vector<aclblasComplex>& aOrig,
    const std::vector<aclblasComplex>& xHost, const std::vector<aclblasComplex>& yHost)
{
    std::vector<aclblasComplex> golden = aOrig;
    csyr2_arch22_test::Csyr2Golden(
        p.uplo, p.n, p.alpha, xHost.data(), p.incx, yHost.data(), p.incy, golden.data(), p.lda);

    std::vector<float> triOut;
    std::vector<float> triGold;
    for (int j = 0; j < p.n; ++j) {
        const int iStart = (p.uplo == ACLBLAS_UPPER) ? 0 : j;
        const int iEnd = (p.uplo == ACLBLAS_UPPER) ? j : (p.n - 1);
        for (int i = iStart; i <= iEnd; ++i) {
            const int64_t idx = static_cast<int64_t>(j) * p.lda + i;
            triOut.push_back(aHost[static_cast<size_t>(idx)].real);
            triOut.push_back(aHost[static_cast<size_t>(idx)].imag);
            triGold.push_back(golden[static_cast<size_t>(idx)].real);
            triGold.push_back(golden[static_cast<size_t>(idx)].imag);
        }
    }
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, triGold.data(), triGold.size());
    EXPECT_TRUE(Verifier::verifyVector(triOut.data(), triGold.data(), triOut.size(), 1, cfg, p.caseName));

    uint32_t unchErr = 0;
    for (int j = 0; j < p.n; ++j) {
        for (int i = 0; i < p.lda; ++i) {
            if (InUploTriangle(p.uplo, p.n, i, j)) {
                continue;
            }
            const int64_t idx = static_cast<int64_t>(j) * p.lda + i;
            if (!SameComplex(aHost[static_cast<size_t>(idx)], aOrig[static_cast<size_t>(idx)])) {
                ++unchErr;
            }
        }
    }
    EXPECT_EQ(unchErr, 0u) << "[" << p.caseName << "] Unchanged region: " << unchErr << " errors";
}

TEST_F(Csyr2Arch22Test, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex dummy{0.0f, 0.0f};
    const aclblasStatus_t ret = aclblasCsyr2(nullptr, ACLBLAS_UPPER, 4, &alpha, &dummy, 1, &dummy, 1, &dummy, 4);
    EXPECT_NE(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}

INSTANTIATE_TEST_SUITE_P(
    Csyr2, Csyr2Arch22Test, ::testing::ValuesIn(LoadCsyr2Cases()), PrintCaseInfoString<Csyr2Arch22Param>);

TEST_P(Csyr2Arch22Test, CsvDriven)
{
    const auto& p = GetParam();
    if (p.caseName == "TC_CSV_MISSING") {
        FAIL() << "csyr2 CSV not found; will not swallow as 0 tests. "
                  "Looked for csyr2_test.csv next to source, cwd, and binary.";
    }

    const size_t aSize = (p.n > 0 && p.lda > 0) ? static_cast<size_t>(p.lda) * static_cast<size_t>(p.n) : 0;
    std::vector<aclblasComplex> xHost = MakeComplexStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<aclblasComplex> yHost = MakeComplexStrided(p.n, p.incy, p.y, p.randomSeed + 1U);
    std::vector<aclblasComplex> aHost = (p.a.method == BlasFillMode::M_NULLPTR || aSize == 0) ?
                                            std::vector<aclblasComplex>{} :
                                            makeBlasComplexMatrix(p.n, p.n, p.lda, p.a, p.randomSeed);
    if (!aHost.empty() && aHost.size() < aSize) {
        aHost.resize(aSize);
    }
    const std::vector<aclblasComplex> aOrig = aHost;

    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    const aclblasComplex* yPtr = yHost.empty() ? nullptr : yHost.data();
    aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();

    if (IsPerfCase(p.caseName) && p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0 && aPtr != nullptr) {
        RunPerformance(p, xPtr, yPtr, aPtr);
        return;
    }

    const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const aclblasStatus_t expect = p.ExpectedStatus();
    const aclblasStatus_t ret =
        aclblasCsyr2_npu(Csyr2Arch22Test::handle_, p.uplo, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(expect));
    if (expect != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (p.n == 0 || aPtr == nullptr) {
        return;
    }

    CheckResult(p, aHost, aOrig, xHost, yHost);
}

#include "csyr2_review_test.h"
