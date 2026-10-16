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
#include <iostream>
#include <limits>
#include <vector>
#include <random>
#include <complex>
#include <cblas.h>
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "chemm_param.h"

class ChemmTest : public BlasTest<ChemmParam> {};
struct DeviceArray {
    void* ptr = nullptr;
    explicit DeviceArray(const std::vector<float>& src)
    {
        if (!src.empty()) {
            if (aclrtMalloc(&ptr, src.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
                throw std::runtime_error("aclrtMalloc failed");
            if (aclrtMemcpy(ptr, src.size() * 4, src.data(), src.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE) !=
                ACL_SUCCESS) {
                aclrtFree(ptr);
                ptr = nullptr;
                throw std::runtime_error("H2D failed");
            }
        }
    }
    ~DeviceArray()
    {
        if (ptr)
            aclrtFree(ptr);
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};
static std::vector<float> MakeInput(int64_t ld, int64_t cols, const BlasFillMode& fill, int seed)
{
    if (ld <= 0 || cols <= 0 || fill.method == BlasFillMode::M_NULLPTR)
        return {};
    return makeBlasArray(size_t(ld) * cols * 2, fill, seed);
}
static bool CheckComponent(
    const std::vector<float>& out, const std::vector<float>& golden, int64_t m, int64_t n, int64_t ld, int component,
    const std::string& name)
{
    size_t matched = 0;
    size_t count = 0;
    double maxerr = 0;
    bool maxPass = true;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) {
            size_t q = 2 * (j * ld + i) + component;
            float a = out[q];
            float g = golden[q];
            ++count;
            if (!std::isfinite(g) || !std::isfinite(a)) {
                bool same = (std::isnan(a) && std::isnan(g)) || (std::isinf(a) && a == g);
                if (!same)
                    std::cout << "NONFINITE_MISMATCH," << name << "," << q << "," << a << "," << g << "\n";
                matched += same;
                maxPass &= same;
                continue;
            }
            double error = std::abs(double(a) - double(g));
            float neighbor = std::nextafter(g, std::numeric_limits<float>::infinity());
            if (!std::isfinite(neighbor))
                neighbor = std::nextafter(g, 0.0f);
            double ulp = std::abs(double(neighbor) - g);
            maxerr = std::max(maxerr, error);
            matched += error <= std::ldexp(1., -16) + std::ldexp(1., -10) * std::abs(double(g));
            if (error > std::max(0.01, 32 * ulp) && std::getenv("CHEMM_DIAG"))
                std::cout << "ABS_MISMATCH," << name << "," << q << ",actual=" << a << ",golden=" << g
                          << ",error=" << error << ",ulp=" << ulp << "\n";
            maxPass &= error <= std::max(0.01, 32 * ulp);
        }
    double ratio = count ? double(matched) / count : 1;
    std::cout << "PRECISION," << name << ',' << component << ',' << ratio << ',' << maxerr << '\n';
    return ratio >= 0.99 && maxPass;
}
INSTANTIATE_TEST_SUITE_P(
    Chemm, ChemmTest, ::testing::ValuesIn(GetCasesFromCsv<ChemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ChemmParam>);
struct ChemmDiagnosticInputs {
    const ChemmParam& p;
    const std::vector<float>& a;
    const std::vector<float>& b;
    const std::vector<float>& golden;
    const std::vector<float>& result;
    const std::vector<float>& oldC;
    aclblasComplex alpha;
    aclblasComplex beta;
    int64_t k;
};

static bool OutsideAbsTolerance(double actual, double golden)
{
    double ulp = std::abs(double(std::nextafter(float(golden), INFINITY)) - golden);
    return std::abs(actual - golden) > std::max(.01, 32 * ulp);
}

static void ApplyPairedDistribution(
    const ChemmParam& p, int64_t k, std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
{
    std::vector<float>* arrays[] = {&a, &b, &c};
    for (size_t v = 0; v < 3; ++v) {
        std::mt19937 rng(p.randomSeed + v);
        std::uniform_real_distribution<float> uniform(-5, 5);
        std::normal_distribution<float> real(p.meanReal[v], p.sigmaReal[v]);
        std::normal_distribution<float> imag(p.meanImag[v], p.sigmaImag[v]);
        for (size_t q = 0; q < arrays[v]->size(); q += 2) {
            (*arrays[v])[q] = p.inputDistribution == "uniform" ? uniform(rng) : real(rng);
            (*arrays[v])[q + 1] = p.inputDistribution == "uniform" ? uniform(rng) : imag(rng);
        }
    }
    for (int64_t col = 0; col < k; ++col) {
        a[2 * (col * p.lda + col) + 1] = 0;
        for (int64_t row = 0; row < col; ++row) {
            const int64_t upper = 2 * (col * p.lda + row);
            const int64_t lower = 2 * (row * p.lda + col);
            const int64_t src = p.uplo == ACLBLAS_UPPER ? upper : lower;
            const int64_t dst = p.uplo == ACLBLAS_UPPER ? lower : upper;
            a[dst] = a[src];
            a[dst + 1] = -a[src + 1];
        }
    }
    std::cout << "DISTRIBUTION," << p.caseName << ',' << p.inputDistribution << ',' << p.randomSeed << ','
              << p.alphaReal << ',' << p.alphaImag << ',' << p.betaReal << ',' << p.betaImag << '\n';
}

static void ApplyExtraInputPolicy(
    const ChemmParam& p, int64_t k, std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
{
    if (p.caseName.rfind("TC_EXTRA_NORMAL", 0) == 0) {
        std::mt19937 rng(p.randomSeed);
        std::uniform_real_distribution<float> mean(-5, 5);
        std::uniform_real_distribution<float> sigma(.1, 2);
        std::normal_distribution<float> dist(mean(rng), sigma(rng));
        for (auto* v : {&a, &b, &c})
            for (float& x : *v)
                x = dist(rng);
    }
    if (p.caseName.rfind("TC_EXTRA_NORMAL", 0) != 0 && p.caseName.rfind("TC_EXTRA_UNIFORM", 0) != 0 &&
        p.caseName != "TC_EXTRA_FINITE_RANGE")
        return;

    if (p.caseName == "TC_EXTRA_FINITE_RANGE") {
        std::fill(a.begin(), a.end(), 2.0e38f);
        std::fill(b.begin(), b.end(), 1.0e-38f);
        return;
    }
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i < k; ++i) {
            const bool unused = p.uplo == ACLBLAS_UPPER ? i > j : i < j;
            if (unused)
                a[2 * (j * p.lda + i)] = a[2 * (j * p.lda + i) + 1] = std::numeric_limits<float>::quiet_NaN();
            if (i == j)
                a[2 * (j * p.lda + i) + 1] = std::numeric_limits<float>::quiet_NaN();
        }
}

static void PrintOrderCandidate(const ChemmDiagnosticInputs& in, int64_t row, int64_t col)
{
    const int64_t pos = 2 * (col * in.p.ldc + row);
    const bool upper = in.p.uplo == ACLBLAS_UPPER;
    float tr = 0;
    float ti = 0;
    for (int64_t q = upper ? 0 : row + 1; q < (upper ? row : in.p.m); ++q) {
        const float xr = in.a[2 * (row * in.p.lda + q)];
        const float xi = -in.a[2 * (row * in.p.lda + q) + 1];
        const float yr = in.b[2 * (col * in.p.ldb + q)];
        const float yi = in.b[2 * (col * in.p.ldb + q) + 1];
        tr += yr * xr - yi * xi;
        ti += yr * xi + yi * xr;
    }
    const float yr = in.b[2 * (col * in.p.ldb + row)];
    const float yi = in.b[2 * (col * in.p.ldb + row) + 1];
    const float xr = in.alpha.real * yr - in.alpha.imag * yi;
    const float xi = in.alpha.real * yi + in.alpha.imag * yr;
    float rr = xr * in.a[2 * (row * in.p.lda + row)];
    float ii = xi * in.a[2 * (row * in.p.lda + row)];
    if (in.beta.real != 0 || in.beta.imag != 0) {
        float cr = in.oldC[pos];
        float ci = in.oldC[pos + 1];
        rr = (in.beta.real * cr - in.beta.imag * ci) + rr;
        ii = (in.beta.real * ci + in.beta.imag * cr) + ii;
    }
    rr += in.alpha.real * tr - in.alpha.imag * ti;
    ii += in.alpha.real * ti + in.alpha.imag * tr;
    for (int64_t q = row + (upper ? 1 : -1); q != (upper ? in.p.m : -1); q += (upper ? 1 : -1)) {
        float br = in.b[2 * (col * in.p.ldb + q)];
        float bi = in.b[2 * (col * in.p.ldb + q) + 1];
        float pr = in.alpha.real * br - in.alpha.imag * bi;
        float pi = in.alpha.real * bi + in.alpha.imag * br;
        float ar = in.a[2 * (q * in.p.lda + row)];
        float ai = in.a[2 * (q * in.p.lda + row) + 1];
        rr += pr * ar - pi * ai;
        ii += pr * ai + pi * ar;
    }
    std::cout << "ORDER_DIAG," << in.p.caseName << ',' << row << ',' << col << ",npu=" << in.result[pos] << '/'
              << in.result[pos + 1] << ",clone=" << rr << '/' << ii << ",netlib=" << in.golden[pos] << '/'
              << in.golden[pos + 1] << ",npu_clone=" << in.result[pos] - rr << '/' << in.result[pos + 1] - ii
              << ",clone_netlib=" << rr - in.golden[pos] << '/' << ii - in.golden[pos + 1] << '\n';
}

static void PrintOrderDiagnostic(const ChemmDiagnosticInputs& in)
{
    int printed = 0;
    for (int64_t col = 0; col < in.p.n && printed < 8; ++col)
        for (int64_t row = 0; row < in.p.m && printed < 8; ++row) {
            const int64_t pos = 2 * (col * in.p.ldc + row);
            bool bad = false;
            for (int comp = 0; comp < 2; ++comp)
                bad |= OutsideAbsTolerance(in.result[pos + comp], in.golden[pos + comp]);
            if (!bad)
                continue;
            PrintOrderCandidate(in, row, col);
            ++printed;
        }
}

static void PrintRiskCandidate(const ChemmDiagnosticInputs& in, int64_t row, int64_t col)
{
    const int64_t idx = 2 * (col * in.p.ldc + row);
    double rr[4]{};
    double ii[4]{};
    double ss[4]{};
    const int64_t chunk = ((in.k + 127) / 128 * 128) / 4;
    for (int64_t z = 0; z < in.k; ++z) {
        int64_t x = in.p.side == ACLBLAS_SIDE_LEFT ? row : z;
        int64_t y = in.p.side == ACLBLAS_SIDE_LEFT ? z : col;
        bool d = in.p.uplo == ACLBLAS_UPPER ? x <= y : x >= y;
        int64_t off = d ? y * in.p.lda + x : x * in.p.lda + y;
        int64_t bo = in.p.side == ACLBLAS_SIDE_LEFT ? col * in.p.ldb + z : z * in.p.ldb + row;
        float ar = in.a[2 * off];
        float ai = x == y ? 0 : (d ? in.a[2 * off + 1] : -in.a[2 * off + 1]);
        float br = in.b[2 * bo];
        float bi = in.b[2 * bo + 1];
        rr[z / chunk] += double(ar) * br;
        ii[z / chunk] += double(ai) * bi;
        ss[z / chunk] += double(float(ar + ai)) * float(br + bi);
    }
    double re = 0;
    double im = 0;
    for (int t = 0; t < 4; ++t) {
        re += rr[t] - ii[t];
        im += ss[t] - rr[t] - ii[t];
    }
    double pr = 0;
    double pi = 0;
    double riskR = 0;
    double riskI = 0;
    double mrr = INFINITY;
    double mii = INFINITY;
    double mss = INFINITY;
    for (int t = 0; t < 4; ++t) {
        pr += rr[t] - ii[t];
        pi += ss[t] - rr[t] - ii[t];
        riskR += std::pow(std::max(std::abs(pr), std::abs(re - pr)), 2);
        riskI += std::pow(std::max(std::abs(pi), std::abs(im - pi)), 2);
        mrr = std::min(mrr, std::abs(rr[t]));
        mii = std::min(mii, std::abs(ii[t]));
        mss = std::min(mss, .5 * std::abs(ss[t]));
    }
    const double u = std::ldexp(1., -24);
    const double fac = u * std::sqrt(double(in.k) / 4) * std::abs(in.alpha.real);
    std::cout << "RISK_PROXY," << in.p.caseName << ',' << row << ',' << col << ",rms=" << fac * std::sqrt(riskR) << '/'
              << fac * std::sqrt(riskI)
              << ",drift=" << .5 * u * (double(in.k) / 4) * std::abs(in.alpha.real) * std::max({mrr, mii, mss})
              << ",limit=" << std::max(.01, 32 * u * std::abs(in.alpha.real * re)) << '/'
              << std::max(.01, 32 * u * std::abs(in.alpha.real * im)) << '\n';
}

static void PrintRiskDiagnostic(const ChemmDiagnosticInputs& in)
{
    int printed = 0;
    for (int64_t col = 0; col < in.p.n && printed < 8; ++col)
        for (int64_t row = 0; row < in.p.m && printed < 8; ++row) {
            const int64_t idx = 2 * (col * in.p.ldc + row);
            bool bad = false;
            for (int comp = 0; comp < 2; ++comp)
                bad |= OutsideAbsTolerance(in.result[idx + comp], in.golden[idx + comp]);
            if (!bad)
                continue;
            PrintRiskCandidate(in, row, col);
            ++printed;
        }
}

static std::complex<double> ExactChemmElement(const ChemmDiagnosticInputs& in, int64_t i, int64_t j)
{
    std::complex<double> sum{};
    for (int64_t q = 0; q < in.k; ++q) {
        int64_t x = in.p.side == ACLBLAS_SIDE_LEFT ? i : q;
        int64_t y = in.p.side == ACLBLAS_SIDE_LEFT ? q : j;
        bool d = in.p.uplo == ACLBLAS_UPPER ? x <= y : x >= y;
        int64_t off = d ? y * in.p.lda + x : x * in.p.lda + y;
        std::complex<double> av(in.a[2 * off], x == y ? 0 : (d ? in.a[2 * off + 1] : -in.a[2 * off + 1]));
        int64_t bo = in.p.side == ACLBLAS_SIDE_LEFT ? j * in.p.ldb + q : q * in.p.ldb + i;
        sum += av * std::complex<double>(in.b[2 * bo], in.b[2 * bo + 1]);
    }
    const int64_t pos = 2 * (j * in.p.ldc + i);
    return std::complex<double>(in.alpha.real, in.alpha.imag) * sum +
           std::complex<double>(in.beta.real, in.beta.imag) * std::complex<double>(in.oldC[pos], in.oldC[pos + 1]);
}

static void PrintFp64Diagnostic(const ChemmDiagnosticInputs& in)
{
    double worstA = 0;
    double worstG = 0;
    size_t checked = 0;
    for (int64_t j = 0; j < in.p.n; ++j)
        for (int64_t i = 0; i < in.p.m; ++i) {
            const int64_t idx = 2 * (j * in.p.ldc + i);
            if (std::abs(in.result[idx] - in.golden[idx]) < 0.009 &&
                std::abs(in.result[idx + 1] - in.golden[idx + 1]) < 0.009)
                continue;
            ++checked;
            const auto exact = ExactChemmElement(in, i, j);
            for (int component = 0; component < 2; ++component) {
                const double av = in.result[idx + component];
                const double gv = in.golden[idx + component];
                const double ev = component == 0 ? exact.real() : exact.imag();
                const double ulp = std::abs(double(std::nextafter(float(gv), INFINITY)) - gv);
                if (std::abs(av - gv) > std::max(.01, 32 * ulp))
                    std::cout << "FP64_FAILED_ELEMENT," << in.p.caseName << ',' << i << ',' << j << ',' << component
                              << ",npu_error=" << av - ev << ",netlib_error=" << gv - ev
                              << ",allowed=" << std::max(.01, 32 * ulp) << '\n';
            }
            worstA = std::max(
                worstA, std::max(std::abs(in.result[idx] - exact.real()), std::abs(in.result[idx + 1] - exact.imag())));
            worstG = std::max(
                worstG, std::max(std::abs(in.golden[idx] - exact.real()), std::abs(in.golden[idx + 1] - exact.imag())));
        }
    std::cout << "FP64_DIAG_CANDIDATES," << in.p.caseName << ",checked=" << checked << ",actual=" << worstA
              << ",netlib=" << worstG << '\n';
}

static void CheckResult(const ChemmParam& p, const std::vector<float>& result, const std::vector<float>& golden)
{
    ASSERT_TRUE(CheckComponent(result, golden, p.m, p.n, p.ldc, 0, p.caseName));
    ASSERT_TRUE(CheckComponent(result, golden, p.m, p.n, p.ldc, 1, p.caseName));
    for (int64_t j = 0; j < p.n; ++j)
        for (int64_t i = p.m; i < p.ldc; ++i)
            ASSERT_EQ(std::memcmp(result.data() + 2 * (j * p.ldc + i), golden.data() + 2 * (j * p.ldc + i), 8), 0)
                << "padding overwritten";
}

template <typename Call>
static void RunChemmPerformance(const ChemmParam& p, Call&& call, aclrtStream stream)
{
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(call(), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    ASSERT_EQ(aclrtCreateEvent(&start), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&stop), ACL_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(start, stream), ACL_SUCCESS);
    for (int i = 0; i < 60; ++i)
        ASSERT_EQ(call(), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(stop, stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeEvent(stop), ACL_SUCCESS);
    float ms = 0;
    ASSERT_EQ(aclrtEventElapsedTime(&ms, start, stop), ACL_SUCCESS);
    ASSERT_EQ(aclrtDestroyEvent(start), ACL_SUCCESS);
    ASSERT_EQ(aclrtDestroyEvent(stop), ACL_SUCCESS);
    double us = ms * 1000 / 60;
    std::cout << "PERFORMANCE," << p.caseName << ',' << p.m << ',' << p.n << ",60," << us << '\n';
    if (p.caseName == "TC_PF_1001" || p.caseName == "TC_PF_1002" || p.caseName == "TC_PF_1003" ||
        p.caseName == "TC_PF_1004") {
        double limit = p.side == ACLBLAS_SIDE_LEFT ? (p.m == 1024 ? 564.14 : 3824.68) : (p.m == 1024 ? 547.6 : 3683.6);
        EXPECT_LE(us, limit);
    }
}

static void ReportChemmMemory(const ChemmParam& p, bool enabled, size_t freeBefore, size_t totalBefore)
{
    if (!enabled)
        return;
    size_t freeAfter = 0;
    size_t totalAfter = 0;
    if (aclrtGetMemInfo(ACL_HBM_MEM, &freeAfter, &totalAfter) == ACL_SUCCESS)
        std::cout << "MEMORY," << p.caseName << "," << freeBefore << "," << freeAfter << "," << totalAfter << "\n";
}

static void PrintChemmDiagnostics(const ChemmDiagnosticInputs& diagnostic)
{
    if (std::getenv("CHEMM_ORDER_DIAG") && diagnostic.p.side == ACLBLAS_SIDE_LEFT)
        PrintOrderDiagnostic(diagnostic);
    if (std::getenv("CHEMM_RISK_DIAG"))
        PrintRiskDiagnostic(diagnostic);
    if (std::getenv("CHEMM_DIAG"))
        PrintFp64Diagnostic(diagnostic);
}

TEST_P(ChemmTest, CsvDriven)
{
    const auto& p = GetParam();
    const int64_t k = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    auto a = MakeInput(p.lda, k, p.aFill, p.randomSeed);
    auto b = MakeInput(p.ldb, p.n, p.bFill, p.randomSeed + 1);
    auto c = MakeInput(p.ldc, p.n, p.cFill, p.randomSeed + 2);
    if (!p.inputDistribution.empty())
        ApplyPairedDistribution(p, k, a, b, c);
    ApplyExtraInputPolicy(p, k, a, b, c);

    auto golden = c;
    size_t freeBefore = 0;
    size_t totalBefore = 0;
    const bool measureMemory =
        p.caseName.rfind("TC_PF", 0) == 0 && aclrtGetMemInfo(ACL_HBM_MEM, &freeBefore, &totalBefore) == ACL_SUCCESS;
    DeviceArray da(a);
    DeviceArray db(b);
    DeviceArray dc(c);
    aclblasComplex alpha{p.alphaReal, p.alphaImag};
    aclblasComplex beta{p.betaReal, p.betaImag};
    auto call = [&]() {
        return aclblasChemm(
            handle_, p.side, p.uplo, p.m, p.n, p.nullAlpha ? nullptr : &alpha, (aclblasComplex*)da.ptr, p.lda,
            (aclblasComplex*)db.ptr, p.ldb, p.nullBeta ? nullptr : &beta, (aclblasComplex*)dc.ptr, p.ldc);
    };
    ASSERT_EQ(call(), p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    if (p.m == 0 || p.n == 0)
        return;
    ReportChemmMemory(p, measureMemory, freeBefore, totalBefore);

    cblas_chemm(
        CblasColMajor, p.side == ACLBLAS_SIDE_LEFT ? CblasLeft : CblasRight,
        p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, p.m, p.n, &alpha, a.data(), p.lda, b.data(), p.ldb, &beta,
        golden.data(), p.ldc);
    const auto diagnosticOldC =
        (std::getenv("CHEMM_DIAG") || std::getenv("CHEMM_ORDER_DIAG")) ? c : std::vector<float>{};
    auto& result = c;
    ASSERT_EQ(
        aclrtMemcpy(result.data(), result.size() * 4, dc.ptr, result.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    ChemmDiagnosticInputs diagnostic{p, a, b, golden, result, diagnosticOldC, alpha, beta, k};
    PrintChemmDiagnostics(diagnostic);
    CheckResult(p, result, golden);
    if (p.caseName.rfind("TC_PF", 0) == 0)
        RunChemmPerformance(p, call, stream_);
}
TEST_F(ChemmTest, NullHandle)
{
    EXPECT_EQ(
        aclblasChemm(
            nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, 0, 0, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

static void RunIndependentChemmCalls(
    aclblasHandle_t primaryHandle, aclblasHandle_t independentHandle, int n, DeviceArray& da, DeviceArray& db,
    DeviceArray& dc0, DeviceArray& dc1)
{
    aclblasComplex one{1, 0};
    aclblasComplex two{2, 0};
    aclblasComplex zero{0, 0};
    // Same-sized concurrent calls must not share data caches or internal streams.
    for (int j = 0; j < 4; ++j) {
        ASSERT_EQ(
            aclblasChemm(
                primaryHandle, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, n, n, &one, (aclblasComplex*)da.ptr, n,
                (aclblasComplex*)db.ptr, n, &zero, (aclblasComplex*)dc0.ptr, n),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasChemm(
                independentHandle, ACLBLAS_SIDE_RIGHT, ACLBLAS_LOWER, n, n, &two, (aclblasComplex*)da.ptr, n,
                (aclblasComplex*)db.ptr, n, &zero, (aclblasComplex*)dc1.ptr, n),
            ACLBLAS_STATUS_SUCCESS);
    }
}

TEST_F(ChemmTest, IndependentHandlesStreamsWorkspace)
{
    constexpr int n = 128;
    std::vector<float> a(2 * n * n, 0);
    std::vector<float> b(2 * n * n, 0);
    std::vector<float> c(2 * n * n, 0);
    for (int i = 0; i < n; ++i)
        a[2 * (i * n + i)] = 1;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = float(i % 17) * 0.125f;
    DeviceArray da(a);
    DeviceArray db(b);
    DeviceArray dc0(c);
    DeviceArray dc1(c);
    aclblasHandle_t h = nullptr;
    aclrtStream other = nullptr;
    ASSERT_EQ(aclblasCreate(&h), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtCreateStream(&other), ACL_SUCCESS);
    ASSERT_EQ(aclblasSetStream(h, other), ACLBLAS_STATUS_SUCCESS);
    aclblasComplex one{1, 0};
    aclblasComplex zero{0, 0};
    RunIndependentChemmCalls(handle_, h, n, da, db, dc0, dc1);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(other), ACL_SUCCESS);
    std::vector<float> out(c.size());
    ASSERT_EQ(aclrtMemcpy(out.data(), out.size() * 4, dc0.ptr, out.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(out, b);
    ASSERT_EQ(aclrtMemcpy(out.data(), out.size() * 4, dc1.ptr, out.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < b.size(); ++i)
        EXPECT_EQ(out[i], 2 * b[i]);
    void* userWorkspace = nullptr;
    ASSERT_EQ(aclrtMalloc(&userWorkspace, 1024, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclblasSetWorkspace(h, userWorkspace, 1024), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasChemm(
            h, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, n, n, &one, (aclblasComplex*)da.ptr, n, (aclblasComplex*)db.ptr, n,
            &zero, (aclblasComplex*)dc1.ptr, n),
        ACLBLAS_STATUS_ALLOC_FAILED);
    ASSERT_EQ(aclblasDestroy(h), ACLBLAS_STATUS_SUCCESS);
    // Library destruction must leave caller-owned workspace alive.
    EXPECT_EQ(aclrtMemset(userWorkspace, 1024, 0, 1024), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(userWorkspace), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyStream(other), ACL_SUCCESS);
}

static aclblasComplex* OffsetDevicePointer(DeviceArray& deviceArray, int64_t offset)
{
    auto* base = (aclblasComplex*)deviceArray.ptr;
    return base + offset;
}

TEST_F(ChemmTest, OffsetPointers)
{
    const int n = 65;
    for (int offset : {1, 3, 63}) {
        std::vector<float> a(2 * (offset + n * n + 1), 123);
        std::vector<float> b = a;
        std::vector<float> c = a;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                size_t q = 2 * (offset + j * n + i);
                a[q] = i == j ? 1 : 0;
                a[q + 1] = 0;
                b[q] = float((i + j) % 13);
                b[q + 1] = float((i + 2 * j) % 7);
            }
        DeviceArray da(a);
        DeviceArray db(b);
        DeviceArray dc(c);
        aclblasComplex one{1, 0};
        aclblasComplex zero{0, 0};
        ASSERT_EQ(
            aclblasChemm(
                handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, n, n, &one, OffsetDevicePointer(da, offset), n,
                OffsetDevicePointer(db, offset), n, &zero, OffsetDevicePointer(dc, offset), n),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(c.data(), c.size() * 4, dc.ptr, c.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        EXPECT_EQ(c, b);
    }
}

TEST_F(ChemmTest, MaximumFinitePrecisionGuard)
{
    // 99% matched alone must not hide one arbitrarily wrong FLT_MAX result.
    std::vector<float> golden(200, 0);
    std::vector<float> actual(200, 0);
    for (size_t i = 0; i < 100; ++i)
        golden[2 * i] = actual[2 * i] = std::numeric_limits<float>::max();
    actual[0] = 0;
    EXPECT_FALSE(CheckComponent(actual, golden, 100, 1, 100, 0, "HARNESS_MAX_FINITE"));
}
