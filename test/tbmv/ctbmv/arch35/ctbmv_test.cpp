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
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "ctbmv_param.h"
#include "ctbmv_golden.h"
#include "ctbmv_test_util.h"
#include "ctbmv_npu_wrapper.h"

class CtbmvArch35Test : public BlasTest<CtbmvParam> {
protected:
    void RunPerformanceCases(bool independentInputs);
    void RunCsvPerformanceCases();
};

namespace {
struct CsvPerformanceCase {
    std::string id;
    int n;
    int k;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    int incx;
    double gpuMs;
};

bool ParseFillMode(const std::string& text, aclblasFillMode_t& value)
{
    if (text == "UPPER") {
        value = ACLBLAS_UPPER;
        return true;
    }
    if (text == "LOWER") {
        value = ACLBLAS_LOWER;
        return true;
    }
    return false;
}

bool ParseOperation(const std::string& text, aclblasOperation_t& value)
{
    if (text == "N") {
        value = ACLBLAS_OP_N;
        return true;
    }
    if (text == "T") {
        value = ACLBLAS_OP_T;
        return true;
    }
    if (text == "C") {
        value = ACLBLAS_OP_C;
        return true;
    }
    return false;
}

bool ParseDiagType(const std::string& text, aclblasDiagType_t& value)
{
    if (text == "NON_UNIT") {
        value = ACLBLAS_NON_UNIT;
        return true;
    }
    if (text == "UNIT") {
        value = ACLBLAS_UNIT;
        return true;
    }
    return false;
}

std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::string GetGpuBaselinePath()
{
    const std::string sourcePath = __FILE__;
    const size_t slash = sourcePath.find_last_of("/\\");
    return sourcePath.substr(0, slash) + "/gpu_baseline.csv";
}
// The four task-book performance shapes with their Ascend 950PR limits.
struct CtbmvPerformanceCase {
    int n;
    int k;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    double limitUs;
};

constexpr CtbmvPerformanceCase kCtbmvPerformanceCases[] = {
    {512, 8, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 17.48},
    {1024, 16, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 27.47},
    {2048, 32, ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 31.87},
    {4096, 64, ACLBLAS_LOWER, ACLBLAS_OP_C, ACLBLAS_UNIT, 45.83},
};

bool AllFinite(const std::vector<aclblasComplex>& values)
{
    for (const auto& value : values) {
        if (!std::isfinite(value.real) || !std::isfinite(value.imag)) {
            return false;
        }
    }
    return true;
}

// Builds one case banded matrix and input vector. The seeds are offset by the
// shape so every case gets its own data while staying reproducible across runs.
void BuildPerformanceInputs(int n, int k, aclblasFillMode_t uplo, int lda, int incx, unsigned seed,
    std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& x)
{
    const int kl = (uplo == ACLBLAS_UPPER) ? 0 : k;
    const int ku = (uplo == ACLBLAS_UPPER) ? k : 0;
    a = makeCtbmvComplexBanded(n, n, kl, ku, lda, "RANDOM_NORM_5_5", seed);
    x = makeCtbmvComplexStrided(n, incx, "RANDOM_NORM_5_5", seed + 1000U);
}

// Owns the device copies of one case matrix and vector. Release() reports
// whether both frees succeeded; the destructor frees whatever is still held, so
// a failing assertion in the caller cannot leak.
class DeviceStage {
public:
    DeviceStage() = default;
    ~DeviceStage() { (void)Release(); }
    DeviceStage(const DeviceStage&) = delete;
    DeviceStage& operator=(const DeviceStage&) = delete;

    bool Upload(const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& x)
    {
        dA_ = UploadOne(a.data(), a.size() * sizeof(aclblasComplex));
        dX_ = UploadOne(x.data(), x.size() * sizeof(aclblasComplex));
        return dA_ != nullptr && dX_ != nullptr;
    }
    const aclblasComplex* A() const { return static_cast<const aclblasComplex*>(dA_); }
    aclblasComplex* X() const { return static_cast<aclblasComplex*>(dX_); }

    bool Release()
    {
        bool ok = true;
        if (dX_ != nullptr) {
            ok = (aclrtFree(dX_) == ACL_SUCCESS) && ok;
            dX_ = nullptr;
        }
        if (dA_ != nullptr) {
            ok = (aclrtFree(dA_) == ACL_SUCCESS) && ok;
            dA_ = nullptr;
        }
        return ok;
    }

private:
    static void* UploadOne(const void* host, size_t bytes)
    {
        void* device = nullptr;
        if (aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return nullptr;
        }
        if (aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            (void)aclrtFree(device);
            return nullptr;
        }
        return device;
    }

    void* dA_ = nullptr;
    void* dX_ = nullptr;
};

// Reads the 200-case baseline CSV. On a malformed row the reason is written to
// error and the function returns false, so the caller can report it with the
// failing row still attached.
bool LoadPerformanceCases(const std::string& path, std::vector<CsvPerformanceCase>& cases, std::string& error)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        error = "cannot open " + path;
        return false;
    }
    std::string line;
    if (!std::getline(input, line)) {
        error = path + ": missing header";
        return false;
    }
    while (std::getline(input, line)) {
        const auto fields = SplitCsvLine(line);
        if (fields.size() != 8U) {
            error = "expected 8 fields: " + line;
            return false;
        }
        CsvPerformanceCase perfCase{};
        perfCase.id = fields[0];
        try {
            perfCase.n = std::stoi(fields[1]);
            perfCase.k = std::stoi(fields[2]);
            perfCase.incx = std::stoi(fields[6]);
            perfCase.gpuMs = std::stod(fields[7]);
        } catch (const std::exception&) {
            error = "bad numeric field: " + line;
            return false;
        }
        if (!ParseFillMode(fields[3], perfCase.uplo) || !ParseOperation(fields[4], perfCase.trans) ||
            !ParseDiagType(fields[5], perfCase.diag)) {
            error = "bad uplo, trans or diag: " + line;
            return false;
        }
        cases.push_back(perfCase);
    }
    return true;
}

// Warms up, then times timedIterations calls between two stream synchronisations
// and reports the total microseconds across them. The caller divides by its own
// compile-time iteration count, which keeps the divisor a constant instead of a
// parameter the checker has to assume may be zero. Allocation, the device copies
// and every accuracy check stay outside this interval.
template <typename LaunchFn>
bool MeasureTotalUs(
    aclrtStream stream, int warmupIterations, int timedIterations, const LaunchFn& launch, double& totalUs)
{
    for (int i = 0; i < warmupIterations; ++i) {
        if (launch(i) != ACLBLAS_STATUS_SUCCESS) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < timedIterations; ++i) {
        if (launch(warmupIterations + i) != ACLBLAS_STATUS_SUCCESS) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    const auto end = std::chrono::steady_clock::now();
    totalUs = std::chrono::duration<double, std::micro>(end - begin).count();
    return true;
}

// Checks every staged output - the warmup slices included - for finiteness and
// against the CPU reference, and emits the summary line the self-test logs
// quote. Returns whether all of them passed.
bool VerifyFinitePerformanceOutputs(const std::vector<aclblasComplex>& batchX,
    const std::vector<aclblasComplex>& golden, const CtbmvPerformanceCase& p, int vectorCount)
{
    bool accuracyPassed = true;
    for (int i = 0; i < vectorCount; ++i) {
        SCOPED_TRACE("vector=" + std::to_string(i));
        const auto first = batchX.begin() + static_cast<size_t>(i) * golden.size();
        const std::vector<aclblasComplex> output(first, first + golden.size());
        const std::string caseId = "FinitePerf_n" + std::to_string(p.n) + "_sample" + std::to_string(i);
        accuracyPassed =
            AllFinite(output) && VerifyCtbmvComplexVector(output, golden, p.n, 1, caseId) && accuracyPassed;
    }
    std::cout << "[FINITE][ctbmv] n=" << p.n << " k=" << p.k << " vectors=" << vectorCount
              << " finite=true accuracy=" << (accuracyPassed ? "PASS" : "FAIL") << std::endl;
    return accuracyPassed;
}

// Builds the UNIT-diagonal probe inputs for one (uplo, trans) pair: aFinite is
// untouched, aSpecial has only the stored diagonal replaced, and x/golden start
// from the same vector. Every off-diagonal operand stays finite, so a
// non-finite output can only come from reading the diagonal despite diag=UNIT.
void BuildUnitDiagonalInputs(float special, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    int lda, std::vector<aclblasComplex>& aFinite, std::vector<aclblasComplex>& aSpecial,
    std::vector<aclblasComplex>& x, std::vector<aclblasComplex>& golden)
{
    const int kl = uplo == ACLBLAS_UPPER ? 0 : k;
    const int ku = uplo == ACLBLAS_UPPER ? k : 0;
    aFinite = makeCtbmvComplexBanded(n, n, kl, ku, lda, "RANDOM_NORM_5_5",
        62000U + static_cast<unsigned>(uplo) + static_cast<unsigned>(trans));
    aSpecial = aFinite;
    x = makeCtbmvComplexStrided(n, 1, "RANDOM_NORM_5_5", 63000U);
    golden = x;

    const int diagonalRow = uplo == ACLBLAS_UPPER ? k : 0;
    for (int col = 0; col < n; ++col) {
        aSpecial[static_cast<size_t>(col) * lda + diagonalRow] = aclblasComplex{special, special};
    }
}
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Ctbmv, CtbmvArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<CtbmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtbmvParam>);

TEST_P(CtbmvArch35Test, CsvDriven) {
    const auto& p = GetParam();

    int kl = (p.uplo == ACLBLAS_UPPER) ? 0 : p.k;
    int ku = (p.uplo == ACLBLAS_UPPER) ? p.k : 0;

    // For invalid-lda cases (lda < k+1) generate with a safe storage lda so the banded
    // layout stays in-bounds; the operator is still handed p.lda and will reject it.
    const int genLda = std::max(p.lda, p.k + 1);
    auto a = makeCtbmvComplexBanded(p.n, p.n, kl, ku, genLda, p.aFill, p.randomSeed);
    auto x = makeCtbmvComplexStrided(p.n, p.incx, p.xFill, p.randomSeed + 1);
    std::vector<aclblasComplex> golden = x;

    const aclblasComplex* aPtr = a.empty() ? nullptr : a.data();
    aclblasComplex* xPtr = x.empty() ? nullptr : x.data();
    aclblasComplex* goldPtr = golden.empty() ? nullptr : golden.data();

    aclblasStatus_t ret = aclblasCtbmv_npu(
        CtbmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, p.k,
        aPtr, p.lda, xPtr, p.incx);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasStatus_t goldRet = aclblasCtbmv_cpu(
        CtbmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, p.k,
        aPtr, p.lda, goldPtr, p.incx);
    ASSERT_EQ(goldRet, ACLBLAS_STATUS_SUCCESS);

    EXPECT_TRUE(VerifyCtbmvComplexVector(x, golden, p.n, p.incx, p.caseName));
}

TEST_F(CtbmvArch35Test, TransposeUbBoundaries)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (int n : {65, 129, 257}) {
        for (int k : {31, 32, 64}) {
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (auto trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                    for (auto diag : {ACLBLAS_NON_UNIT, ACLBLAS_UNIT}) {
                        const std::string caseId = "TransposeUb_n" + std::to_string(n) + "_k" +
                            std::to_string(k) + "_u" + std::to_string(uplo) + "_t" +
                            std::to_string(trans) + "_d" + std::to_string(diag);
                        SCOPED_TRACE(caseId);
                        const int lda = k + 3;
                        const int kl = uplo == ACLBLAS_UPPER ? 0 : k;
                        const int ku = uplo == ACLBLAS_UPPER ? k : 0;
                        auto a = makeCtbmvComplexBanded(n, n, kl, ku, lda, "RANDOM_NORM_5_5", 950U + n + k);
                        auto x = makeCtbmvComplexStrided(n, 1, "RANDOM_NORM_5_5", 1950U + n + k);
                        for (int col = 0; col < n; ++col) {
                            const size_t base = static_cast<size_t>(col) * lda;
                            for (int bandRow = k + 1; bandRow < lda; ++bandRow) {
                                a[base + bandRow] = aclblasComplex{nan, inf};
                            }
                            if (diag == ACLBLAS_UNIT) {
                                const int bandRow = uplo == ACLBLAS_UPPER ? k : 0;
                                a[base + bandRow] = aclblasComplex{nan, inf};
                            }
                        }
                        auto golden = x;
                        ASSERT_EQ(aclblasCtbmv_cpu(handle_, uplo, trans, diag, n, k,
                            a.data(), lda, golden.data(), 1), ACLBLAS_STATUS_SUCCESS);
                        ASSERT_EQ(aclblasCtbmv_npu(handle_, uplo, trans, diag, n, k,
                            a.data(), lda, x.data(), 1), ACLBLAS_STATUS_SUCCESS);
                        // These poisoned locations are not operands; matching NaNs must not pass.
                        for (int row = 0; row < n; ++row) {
                            ASSERT_TRUE(std::isfinite(golden[row].real) && std::isfinite(golden[row].imag)) << row;
                            ASSERT_TRUE(std::isfinite(x[row].real) && std::isfinite(x[row].imag)) << row;
                        }
                        EXPECT_TRUE(VerifyCtbmvComplexVector(x, golden, n, 1, caseId));
                    }
                }
            }
        }
    }
}

TEST_F(CtbmvArch35Test, UnitDiagonalSpecialValue)
{
    constexpr int n = 33;
    constexpr int k = 5;
    constexpr int lda = k + 3;
    const float specialValues[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
    };
    int passedCases = 0;

    for (float special : specialValues) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                const std::string valueName = std::isnan(special) ? "NaN" : "Inf";
                const std::string caseId = "UnitDiag_" + valueName + "_n" + std::to_string(n) +
                    "_k" + std::to_string(k) + "_u" + std::to_string(uplo) + "_t" + std::to_string(trans);
                SCOPED_TRACE(caseId);

                std::vector<aclblasComplex> aFinite, aSpecial, x, golden;
                BuildUnitDiagonalInputs(special, uplo, trans, n, k, lda, aFinite, aSpecial, x, golden);
                ASSERT_TRUE(AllFinite(aFinite));
                ASSERT_TRUE(AllFinite(x));

                ASSERT_EQ(aclblasCtbmv_cpu(handle_, uplo, trans, ACLBLAS_UNIT, n, k, aFinite.data(), lda,
                    golden.data(), 1), ACLBLAS_STATUS_SUCCESS);
                ASSERT_EQ(aclblasCtbmv_npu(handle_, uplo, trans, ACLBLAS_UNIT, n, k, aSpecial.data(), lda, x.data(),
                    1), ACLBLAS_STATUS_SUCCESS);
                ASSERT_TRUE(AllFinite(golden));
                ASSERT_TRUE(AllFinite(x));
                ASSERT_TRUE(VerifyCtbmvComplexVector(x, golden, n, 1, caseId));
                ++passedCases;
            }
        }
    }
    std::cout << "[SPECIAL][ctbmv] UnitDiagonalSpecialValue cases=" << passedCases
              << " finite=true accuracy=PASS" << std::endl;
}

// A zero entry of x must still take part in the multiply-add. The reference
// performs X(J) = X(J)*A(diagonal) and X(I) = X(I) + TEMP*A(...) without ever
// testing TEMP for zero, so 0*Inf has to propagate as NaN; skipping the column
// would leave the output finite. incx=1 exercises the parallel UB kernel and
// incx=2 the serial Netlib one, which used to disagree on exactly this input.
TEST_F(CtbmvArch35Test, ZeroOperandSpecialValues)
{
    constexpr int n = 32;
    constexpr int k = 4;
    constexpr int lda = k + 1;

    for (int incx : {1, 2}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            SCOPED_TRACE("incx=" + std::to_string(incx) + " uplo=" + std::to_string(uplo));
            const int kl = uplo == ACLBLAS_UPPER ? 0 : k;
            const int ku = uplo == ACLBLAS_UPPER ? k : 0;
            auto a = makeCtbmvComplexBanded(n, n, kl, ku, lda, "VALUE_NORM_INF", 64000U + static_cast<unsigned>(incx));
            auto x = makeCtbmvComplexStrided(n, incx, "VALUE_NORM_0", 64001U);

            ASSERT_EQ(aclblasCtbmv_npu(handle_, uplo, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n, k, a.data(), lda,
                x.data(), incx), ACLBLAS_STATUS_SUCCESS);

            // Every element keeps meeting an Inf diagonal or an Inf coefficient
            // through a zero operand, so none of them may stay finite.
            int finiteCount = 0;
            for (int i = 0; i < n; ++i) {
                const size_t index = static_cast<size_t>(i) * static_cast<size_t>(incx);
                if (std::isfinite(x[index].real) && std::isfinite(x[index].imag)) {
                    ++finiteCount;
                }
            }
            EXPECT_EQ(finiteCount, 0) << "a zero x entry suppressed the 0*Inf product in its column";
        }
    }
}

void CtbmvArch35Test::RunPerformanceCases(bool independentInputs)
{
    constexpr int warmupIterations = 10;
    constexpr int timedIterations = 100;

    for (const auto& p : kCtbmvPerformanceCases) {
        SCOPED_TRACE("n=" + std::to_string(p.n) + " k=" + std::to_string(p.k));
        const int lda = p.k + 1;
        std::vector<aclblasComplex> a, x;
        BuildPerformanceInputs(p.n, p.k, p.uplo, lda, 1, 20260907U + p.n, a, x);

        // Give every warmup and measured call its own original x. Each slice is
        // still updated in place by the same public API on the same stream.
        const int vectorCount = independentInputs ? warmupIterations + timedIterations : 1;
        std::vector<aclblasComplex> batchX(x.size() * vectorCount);
        for (int i = 0; i < vectorCount; ++i) {
            std::copy(x.begin(), x.end(), batchX.begin() + static_cast<size_t>(i) * x.size());
        }
        std::vector<aclblasComplex> golden;
        if (independentInputs) {
            ASSERT_TRUE(AllFinite(a));
            ASSERT_TRUE(AllFinite(x));
            golden = x;
            ASSERT_EQ(aclblasCtbmv_cpu(handle_, p.uplo, p.trans, p.diag, p.n, p.k, a.data(), lda, golden.data(), 1),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_TRUE(AllFinite(golden));
        }

        DeviceStage stage;
        ASSERT_TRUE(stage.Upload(a, batchX));

        std::vector<aclblasComplex*> inputs(warmupIterations + timedIterations);
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = stage.X() + (independentInputs ? i * x.size() : 0);
        }
        const auto launch = [&](int iteration) {
            return aclblasCtbmv(handle_, p.uplo, p.trans, p.diag, p.n, p.k, stage.A(), lda, inputs[iteration], 1);
        };

        double totalUs = 0.0;
        ASSERT_TRUE(MeasureTotalUs(stream_, warmupIterations, timedIterations, launch, totalUs));
        const double averageUs = totalUs / timedIterations;
        std::cout << "[PERF][ctbmv] n=" << p.n << " k=" << p.k << " average_us=" << averageUs
                  << " limit_us=" << p.limitUs << std::endl;
        EXPECT_LE(averageUs, p.limitUs);

        // D2H, allocation, finite checks and golden comparisons are all outside
        // the measured interval. Verify every slice, including warmup outputs.
        if (independentInputs) {
            const size_t xBytes = batchX.size() * sizeof(aclblasComplex);
            ASSERT_EQ(aclrtMemcpy(batchX.data(), xBytes, stage.X(), xBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
            ASSERT_TRUE(VerifyFinitePerformanceOutputs(batchX, golden, p, vectorCount));
        }
        ASSERT_TRUE(stage.Release());
    }
}

void CtbmvArch35Test::RunCsvPerformanceCases()
{
    constexpr int warmupIterations = 10;
    constexpr int timedIterations = 100;

    std::vector<CsvPerformanceCase> cases;
    std::string error;
    ASSERT_TRUE(LoadPerformanceCases(GetGpuBaselinePath(), cases, error)) << error;
    ASSERT_EQ(cases.size(), 200U);

    for (const auto& p : cases) {
        SCOPED_TRACE(p.id);
        const int lda = p.k + 1;
        std::vector<aclblasComplex> a, x;
        BuildPerformanceInputs(p.n, p.k, p.uplo, lda, p.incx, 20260909U + p.n + p.k, a, x);

        DeviceStage stage;
        ASSERT_TRUE(stage.Upload(a, x));

        const auto launch = [&](int) {
            return aclblasCtbmv(handle_, p.uplo, p.trans, p.diag, p.n, p.k, stage.A(), lda, stage.X(), p.incx);
        };
        double totalUs = 0.0;
        ASSERT_TRUE(MeasureTotalUs(stream_, warmupIterations, timedIterations, launch, totalUs));
        const double averageUs = totalUs / timedIterations;

        const double gpuUs = p.gpuMs * 1000.0;
        std::cout << "[PERF_CSV][ctbmv] id=" << p.id << " n=" << p.n << " k=" << p.k << " gpu_ms=" << p.gpuMs
                  << " gpu_us=" << gpuUs << " average_us=" << averageUs << " execution=PASS" << std::endl;
        ASSERT_TRUE(stage.Release());
    }
}

TEST_F(CtbmvArch35Test, PerformanceTaskCases)
{
    RunPerformanceCases(false);
}

TEST_F(CtbmvArch35Test, PerformanceFiniteInputCases)
{
    RunPerformanceCases(true);
}

TEST_F(CtbmvArch35Test, PerformanceCsvCases)
{
    RunCsvPerformanceCases();
}
