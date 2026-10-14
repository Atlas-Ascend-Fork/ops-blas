/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "blas_test.h"
#include "ctrsm_golden.h"
#include "ctrsm_npu_wrapper.h"
#include "ctrsm_param.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"

namespace {

constexpr float CTRSM_PADDING_VALUE = -997.0f;
constexpr int CTRSM_PERF_WARMUP = 10;
constexpr int CTRSM_PERF_SAMPLES = 51;

float CtrsmSeedParameter(uint32_t seed, uint32_t salt, float lower, float upper)
{
    constexpr uint32_t buckets = 10001U;
    const uint32_t mixed = seed * 1664525U + 1013904223U + salt;
    const float fraction = static_cast<float>(mixed % buckets) / static_cast<float>(buckets - 1U);
    return lower + (upper - lower) * fraction;
}

class ComplexGenerator {
public:
    ComplexGenerator(const BlasFillMode& fill, uint32_t seed, bool varyNormalParameters)
        : fill_(fill),
          realRng_(seed == 0 ? 42U : seed),
          imagRng_((seed == 0 ? 42U : seed) + 7919U),
          uniformReal_(-std::abs(fill.val1), std::abs(fill.val2)),
          uniformImag_(-std::abs(fill.val1), std::abs(fill.val2)),
          normalReal_(
              varyNormalParameters ? CtrsmSeedParameter(seed, 17U, -5.0f, 5.0f) : 0.0f,
              varyNormalParameters ? CtrsmSeedParameter(seed, 31U, 0.1f, 2.0f) : 1.0f),
          normalImag_(
              varyNormalParameters ? CtrsmSeedParameter(seed, 47U, -5.0f, 5.0f) : 0.0f,
              varyNormalParameters ? CtrsmSeedParameter(seed, 61U, 0.1f, 2.0f) : 1.0f),
          useNormal_((seed & 1U) != 0U)
    {}

    aclblasComplex Next(size_t index)
    {
        if (fill_.pattern == BlasFillMode::P_ALTER) {
            const float real = (index & 1U) == 0U ? static_cast<float>(index + 1) : -static_cast<float>(index + 1);
            const float imag = (index & 1U) == 0U ? -real : real;
            return {real, imag};
        }
        if (fill_.pattern == BlasFillMode::P_EXTREME) {
            constexpr float values[] = {
                1.0f,
                0.0f,
                -1.0f,
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::min(),
                -std::numeric_limits<float>::max(),
                std::numeric_limits<float>::denorm_min()};
            constexpr size_t count = sizeof(values) / sizeof(values[0]);
            return {values[index % count], values[(index + 3U) % count]};
        }
        if (fill_.method == BlasFillMode::M_VALUE) {
            return {fill_.val1, fill_.val1};
        }
        if (fill_.method == BlasFillMode::M_INDEX) {
            const float value = static_cast<float>(index + 1);
            return {value, -value};
        }
        if (useNormal_) {
            return {normalReal_(realRng_), normalImag_(imagRng_)};
        }
        return {uniformReal_(realRng_), uniformImag_(imagRng_)};
    }

private:
    BlasFillMode fill_;
    std::mt19937 realRng_;
    std::mt19937 imagRng_;
    std::uniform_real_distribution<float> uniformReal_;
    std::uniform_real_distribution<float> uniformImag_;
    std::normal_distribution<float> normalReal_;
    std::normal_distribution<float> normalImag_;
    bool useNormal_;
};

void ScaleCtrsmUnitTriangle(const CtrsmParam& p, std::vector<aclblasComplex>& a, int order)
{
    if (p.diag == ACLBLAS_UNIT) {
        const float unitTriangularScale = 0.1f / static_cast<float>(std::max(order, 1));
        for (int col = 0; col < order; ++col) {
            for (int row = 0; row < order; ++row) {
                const bool referenced = p.uplo == ACLBLAS_UPPER ? row < col : row > col;
                if (referenced && row < p.lda) {
                    aclblasComplex& value = a[static_cast<size_t>(col) * p.lda + row];
                    value.real *= unitTriangularScale;
                    value.imag *= unitTriangularScale;
                }
            }
        }
    }
}

std::vector<aclblasComplex> MakeCtrsmA(const CtrsmParam& p)
{
    const int order = p.side == ACLBLAS_SIDE_RIGHT ? p.n : p.m;
    if (p.aFill.method == BlasFillMode::M_NULLPTR || order <= 0 || p.lda <= 0) {
        return {};
    }
    const aclblasComplex poison = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};
    std::vector<aclblasComplex> a(static_cast<size_t>(p.lda) * order, poison);
    ComplexGenerator generator(p.aFill, p.randomSeed, !p.performance);
    size_t generated = 0;
    for (int col = 0; col < order; ++col) {
        for (int row = 0; row < order; ++row) {
            const bool referenced = p.uplo == ACLBLAS_UPPER ? row <= col : row >= col;
            if (referenced && row < p.lda) {
                a[static_cast<size_t>(col) * p.lda + row] = generator.Next(generated++);
            }
        }
    }

    for (int index = 0; index < order && index < p.lda; ++index) {
        aclblasComplex& diagonal = a[static_cast<size_t>(index) * p.lda + index];
        if (p.diag == ACLBLAS_UNIT) {
            diagonal = poison;
        } else if (std::isfinite(diagonal.real) && std::isfinite(diagonal.imag)) {
            const float boost = std::max(5.0f, static_cast<float>(order));
            diagonal.real += diagonal.real >= 0.0f ? boost : -boost;
        }
    }
    ScaleCtrsmUnitTriangle(p, a, order);
    return a;
}

std::vector<aclblasComplex> MakeCtrsmB(const CtrsmParam& p)
{
    if (p.bFill.method == BlasFillMode::M_NULLPTR || p.m <= 0 || p.n <= 0 || p.ldb <= 0) {
        return {};
    }
    const aclblasComplex padding = {CTRSM_PADDING_VALUE, CTRSM_PADDING_VALUE};
    std::vector<aclblasComplex> b(static_cast<size_t>(p.ldb) * p.n, padding);
    ComplexGenerator generator(p.bFill, p.randomSeed + 1U, !p.performance);
    size_t generated = 0;
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m && row < p.ldb; ++row) {
            b[static_cast<size_t>(col) * p.ldb + row] = generator.Next(generated++);
        }
    }
    return b;
}

struct PrecisionStats {
    size_t count = 0;
    size_t matched = 0;
    double maxAbsError = 0.0;
    double maxLimitRatio = 0.0;
    float worstOutput = 0.0f;
    float worstGolden = 0.0f;
    bool absoluteLimitPassed = true;
};

void AccumulatePrecision(float output, float golden, PrecisionStats& stats)
{
    ++stats.count;
    if (output == golden || (std::isnan(output) && std::isnan(golden))) {
        ++stats.matched;
        return;
    }
    if (!std::isfinite(output) || !std::isfinite(golden)) {
        stats.absoluteLimitPassed = false;
        return;
    }

    constexpr double rtol = 0.0009765625;       // 2^-10
    constexpr double atol = 0.0000152587890625; // 2^-16
    const double difference = std::abs(static_cast<double>(output) - golden);
    stats.maxAbsError = std::max(stats.maxAbsError, difference);
    if (difference <= atol + rtol * std::abs(static_cast<double>(golden))) {
        ++stats.matched;
    }
    const double elementLimit = std::max(1e-2, 32.0 * getUlpsAt(std::abs(static_cast<double>(golden)), 23, -126));
    if (!(elementLimit > 0.0)) {
        stats.absoluteLimitPassed = false;
        return;
    }
    const double limitRatio = difference / elementLimit;
    if (limitRatio > stats.maxLimitRatio) {
        stats.maxLimitRatio = limitRatio;
        stats.worstOutput = output;
        stats.worstGolden = golden;
    }
    if (difference > elementLimit) {
        stats.absoluteLimitPassed = false;
    }
}

bool VerifyCtrsmPrecision(
    const CtrsmParam& p, const std::vector<aclblasComplex>& output, const std::vector<aclblasComplex>& golden)
{
    PrecisionStats realStats;
    PrecisionStats imagStats;
    size_t reportedMismatches = 0;
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldb + row;
            const size_t matchedBefore = realStats.matched + imagStats.matched;
            const bool absolutePassedBefore = realStats.absoluteLimitPassed && imagStats.absoluteLimitPassed;
            AccumulatePrecision(output[offset].real, golden[offset].real, realStats);
            AccumulatePrecision(output[offset].imag, golden[offset].imag, imagStats);
            const bool absoluteFailed =
                absolutePassedBefore && (!realStats.absoluteLimitPassed || !imagStats.absoluteLimitPassed);
            if ((realStats.matched + imagStats.matched != matchedBefore + 2 || absoluteFailed) &&
                reportedMismatches < 8) {
                std::cout << "[" << p.caseName << "] mismatch row=" << row << " col=" << col << " output=("
                          << output[offset].real << "," << output[offset].imag << ") golden=(" << golden[offset].real
                          << "," << golden[offset].imag << ")" << std::endl;
                ++reportedMismatches;
            }
        }
    }
    const double realRatio = realStats.count == 0 ? 1.0 : static_cast<double>(realStats.matched) / realStats.count;
    const double imagRatio = imagStats.count == 0 ? 1.0 : static_cast<double>(imagStats.matched) / imagStats.count;
    std::cout << "[" << p.caseName << "] real matched_ratio=" << realRatio
              << ", max_abs_error=" << realStats.maxAbsError << ", max_limit_ratio=" << realStats.maxLimitRatio
              << ", worst=(" << realStats.worstOutput << "," << realStats.worstGolden
              << "); imag matched_ratio=" << imagRatio << ", max_abs_error=" << imagStats.maxAbsError
              << ", max_limit_ratio=" << imagStats.maxLimitRatio << ", worst=(" << imagStats.worstOutput << ","
              << imagStats.worstGolden << ")" << std::endl;
    return realRatio >= 0.99 && imagRatio >= 0.99 && realStats.absoluteLimitPassed && imagStats.absoluteLimitPassed;
}

bool VerifyCtrsmPadding(
    const CtrsmParam& p, const std::vector<aclblasComplex>& output, const std::vector<aclblasComplex>& input)
{
    if (p.ldb <= p.m) {
        return true;
    }
    for (int col = 0; col < p.n; ++col) {
        for (int row = p.m; row < p.ldb; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldb + row;
            if (output[offset].real != input[offset].real || output[offset].imag != input[offset].imag) {
                return false;
            }
        }
    }
    return true;
}

struct CtrsmGpuBaseline {
    int m;
    int n;
    aclblasSideMode_t side;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    double gpuUs;

    explicit CtrsmGpuBaseline(const csv_map& row)
        : m(std::stoi(ReadMap(row, "m"))),
          n(std::stoi(ReadMap(row, "n"))),
          side(parseSideMode("ACLBLAS_SIDE_" + ReadMap(row, "side"))),
          uplo(parseFillMode("ACLBLAS_" + ReadMap(row, "uplo"))),
          trans(parseOpTrans("ACLBLAS_OP_" + ReadMap(row, "trans"))),
          diag(parseDiagType("ACLBLAS_" + ReadMap(row, "diag"))),
          gpuUs(std::stod(ReadMap(row, "gpu_ms")) * 1000.0)
    {}
};

double CtrsmGpuTimeUs(const CtrsmParam& p)
{
    const std::string sourcePath = __FILE__;
    static const auto baseline = GetCasesFromCsv<CtrsmGpuBaseline>(
        sourcePath.substr(0, sourcePath.find_last_of("/\\") + 1) + "gpu_baseline.csv");
    for (const auto& row : baseline) {
        if (p.m == row.m && p.n == row.n && p.side == row.side && p.uplo == row.uplo && p.trans == row.trans &&
            p.diag == row.diag) {
            return row.gpuUs;
        }
    }
    return 0.0;
}

struct CtrsmPerformanceResources {
    void* deviceA = nullptr;
    void* deviceB = nullptr;
    void* deviceBInput = nullptr;
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    ~CtrsmPerformanceResources()
    {
        if (startEvent != nullptr) {
            aclrtDestroyEvent(startEvent);
        }
        if (endEvent != nullptr) {
            aclrtDestroyEvent(endEvent);
        }
        if (deviceA != nullptr) {
            aclrtFree(deviceA);
        }
        if (deviceB != nullptr) {
            aclrtFree(deviceB);
        }
        if (deviceBInput != nullptr) {
            aclrtFree(deviceBInput);
        }
    }
};

aclblasStatus_t InvokeCtrsmPerformance(
    aclblasHandle_t handle, const CtrsmParam& p, CtrsmPerformanceResources& resources)
{
    return aclblasCtrsm(
        handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha,
        static_cast<const aclblasComplex*>(resources.deviceA), p.lda, static_cast<aclblasComplex*>(resources.deviceB),
        p.ldb);
}

aclblasStatus_t WarmupCtrsmPerformance(
    aclblasHandle_t handle, aclrtStream stream, const CtrsmParam& p, CtrsmPerformanceResources& resources,
    size_t bBytes, int warmupCount)
{
    for (int sample = 0; sample < warmupCount; ++sample) {
        if (aclrtMemcpyAsync(
                resources.deviceB, bBytes, resources.deviceBInput, bBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) !=
            ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        const auto status = InvokeCtrsmPerformance(handle, p, resources);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return aclrtSynchronizeStream(stream) == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
}

aclblasStatus_t MeasureCtrsmPerformance(
    aclblasHandle_t handle, aclrtStream stream, const CtrsmParam& p, CtrsmPerformanceResources& resources,
    size_t bBytes, int sampleCount, double& averageUs)
{
    double elapsedUs = 0.0;
    for (int sample = 0; sample < sampleCount; ++sample) {
        if (aclrtMemcpyAsync(
                resources.deviceB, bBytes, resources.deviceBInput, bBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) !=
                ACL_SUCCESS ||
            aclrtRecordEvent(resources.startEvent, stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        const auto status = InvokeCtrsmPerformance(handle, p, resources);
        if (status != ACLBLAS_STATUS_SUCCESS || aclrtRecordEvent(resources.endEvent, stream) != ACL_SUCCESS ||
            aclrtSynchronizeEvent(resources.endEvent) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        float sampleMs = 0.0f;
        if (aclrtEventElapsedTime(&sampleMs, resources.startEvent, resources.endEvent) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        elapsedUs += static_cast<double>(sampleMs) * 1000.0;
    }
    if (sampleCount > 0) {
        averageUs = elapsedUs / sampleCount;
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ACLBLAS_STATUS_INVALID_VALUE;
}

aclblasStatus_t RunCtrsmPerformance(
    aclblasHandle_t handle, aclrtStream stream, const CtrsmParam& p, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& b, int warmupCount, int sampleCount, double& averageUs)
{
    if (warmupCount < 0 || sampleCount <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int order = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    const size_t aBytes = static_cast<size_t>(p.lda) * order * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(p.ldb) * p.n * sizeof(aclblasComplex);
    CtrsmPerformanceResources resources;
    if (CtrsmAllocCopy(&resources.deviceA, a.data(), aBytes) != ACL_SUCCESS ||
        CtrsmAllocCopy(&resources.deviceB, b.data(), bBytes) != ACL_SUCCESS ||
        CtrsmAllocCopy(&resources.deviceBInput, b.data(), bBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const auto status = WarmupCtrsmPerformance(handle, stream, p, resources, bBytes, warmupCount);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (aclrtCreateEvent(&resources.startEvent) != ACL_SUCCESS ||
        aclrtCreateEvent(&resources.endEvent) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    return MeasureCtrsmPerformance(handle, stream, p, resources, bBytes, sampleCount, averageUs);
}

void CheckCtrsmPerformance(
    aclblasHandle_t handle, aclrtStream stream, const CtrsmParam& p, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& bInput)
{
    const aclblasComplex* alpha = p.alphaNull ? nullptr : &p.alpha;
    double averageUs = 0.0;
    const double gpuUs = CtrsmGpuTimeUs(p);
    ASSERT_TRUE(std::isfinite(gpuUs) && gpuUs > 0.0) << "Missing or invalid GPU baseline: " << p.caseName;
    const int warmupCount = CTRSM_PERF_WARMUP;
    const int sampleCount = CTRSM_PERF_SAMPLES;
    ASSERT_EQ(
        RunCtrsmPerformance(handle, stream, p, a, bInput, warmupCount, sampleCount, averageUs), ACLBLAS_STATUS_SUCCESS);
    std::cout << "[CTRSM_PERF] case=" << p.caseName << " average_us=" << averageUs << " warmup=" << warmupCount
              << " samples=" << sampleCount;
    ASSERT_TRUE(std::isfinite(averageUs) && averageUs > 0.0);
    const double ratio = gpuUs / averageUs;
    std::cout << " gpu_us=" << gpuUs << " gpu_npu_ratio=" << ratio << " limit_us=" << gpuUs / 0.4 << std::endl;
    EXPECT_GT(ratio, 0.4) << "Performance gate failed: " << p.caseName;

    // Keep correctness outside the timed region so optimizations cannot bypass the precision gate.
    std::vector<aclblasComplex> output = bInput;
    ASSERT_EQ(
        aclblasCtrsmNpu(
            handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, alpha, a.empty() ? nullptr : a.data(), p.lda,
            output.data(), p.ldb),
        ACLBLAS_STATUS_SUCCESS);
    std::vector<aclblasComplex> golden = bInput;
    ASSERT_EQ(
        aclblasCtrsmCpu(
            handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, alpha, a.empty() ? nullptr : a.data(), p.lda,
            golden.data(), p.ldb),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(VerifyCtrsmPrecision(p, output, golden));
    EXPECT_TRUE(VerifyCtrsmPadding(p, output, bInput));
}

} // namespace

class CtrsmArch35Test : public BlasTest<CtrsmParam> {};

void CheckCtrsmNumericalInput(
    aclblasHandle_t handle, const CtrsmParam& p, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& input)
{
    auto output = input;
    auto golden = input;
    ASSERT_EQ(
        aclblasCtrsmCpu(
            handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha, a.data(), p.lda, golden.data(), p.ldb),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCtrsmNpu(
            handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha, a.data(), p.lda, output.data(), p.ldb),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(VerifyCtrsmPrecision(p, output, golden));
    EXPECT_TRUE(VerifyCtrsmPadding(p, output, input));
}

TEST_F(CtrsmArch35Test, NumericalPathBoundaries)
{
    for (const auto shape : {std::pair<int, int>{2, 1}, {63, 32}, {64, 31}, {64, 32}, {65, 33}}) {
        for (int scenario = 0; scenario < 3; ++scenario) {
            CtrsmParam p(csv_map{});
            p.caseName = "TC_NUMERIC_" + std::to_string(shape.first) + "_" + std::to_string(shape.second) + "_" +
                         std::to_string(scenario);
            SCOPED_TRACE(p.caseName);
            p.side = ACLBLAS_SIDE_LEFT;
            p.uplo = scenario == 2 ? ACLBLAS_UPPER : ACLBLAS_LOWER;
            p.trans = ACLBLAS_OP_N;
            p.diag = scenario == 0 ? ACLBLAS_NON_UNIT : ACLBLAS_UNIT;
            p.m = shape.first;
            p.n = shape.second;
            p.lda = p.m;
            p.ldb = p.m;
            p.alpha = {1.0f, 0.0f};
            std::vector<aclblasComplex> a(static_cast<size_t>(p.lda) * p.m, {0.0f, 0.0f});
            std::vector<aclblasComplex> b(static_cast<size_t>(p.ldb) * p.n, {0.0f, 0.0f});
            for (int row = 0; row < p.m; ++row) {
                a[static_cast<size_t>(row) * p.lda + row] = {scenario == 0 ? 1e20f : 1.0f, 0.0f};
            }
            if (scenario == 0) {
                std::fill(b.begin(), b.end(), aclblasComplex{1e20f, 0.0f});
            } else if (scenario == 1) {
                a[1] = {std::numeric_limits<float>::infinity(), 0.0f};
            } else {
                for (int col = 0; col < p.n; ++col) {
                    b[static_cast<size_t>(col) * p.ldb + p.m - 1] = {std::numeric_limits<float>::infinity(), 1.0f};
                }
            }
            CheckCtrsmNumericalInput(handle_, p, a, b);
        }
    }
}

std::vector<CtrsmParam> CtrsmSpecialValueRegressionCases()
{
    std::vector<CtrsmParam> result;
    const auto officialCases = GetCasesFromCsv<CtrsmParam>(ReplaceFileExtension2Csv(__FILE__));
    for (const auto& original : officialCases) {
        if (original.caseName != "TC_FL_110" && original.caseName != "TC_FL_116" && original.caseName != "TC_FL_117") {
            continue;
        }
        for (uint32_t seed : {7U, 42U, 20260911U, 20260912U}) {
            CtrsmParam regression = original;
            regression.caseName = "TC_REG_" + original.caseName + "_seed_" + std::to_string(seed);
            regression.randomSeed = seed;
            result.push_back(regression);
        }
    }
    return result;
}

TEST_F(CtrsmArch35Test, NullHandle)
{
    const aclblasComplex alpha = {1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCtrsm(
            nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, 1, &alpha, nullptr, 1,
            nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CtrsmArch35Test, RejectUnrepresentableBlockedLayout)
{
    const aclblasComplex alpha = {1.0f, 0.0f};
    CtrsmPerformanceResources resources;
    ASSERT_EQ(aclrtMalloc(&resources.deviceA, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&resources.deviceB, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    // Tiny backing buffers are intentional: these dimensions must be rejected before any launch.
    for (const auto side : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
        for (const int rhs : {1073741761, 2147483585, std::numeric_limits<int>::max()}) {
            const int m = side == ACLBLAS_SIDE_LEFT ? 64 : rhs;
            const int n = side == ACLBLAS_SIDE_LEFT ? rhs : 64;
            EXPECT_EQ(
                aclblasCtrsm(
                    handle_, side, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, m, n, &alpha,
                    static_cast<const aclblasComplex*>(resources.deviceA), 64,
                    static_cast<aclblasComplex*>(resources.deviceB), m),
                ACLBLAS_STATUS_INVALID_VALUE);
        }
        const int order = std::numeric_limits<int>::max();
        const int m = side == ACLBLAS_SIDE_LEFT ? order : 32;
        const int n = side == ACLBLAS_SIDE_LEFT ? 32 : order;
        EXPECT_EQ(
            aclblasCtrsm(
                handle_, side, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, m, n, &alpha,
                static_cast<const aclblasComplex*>(resources.deviceA), order,
                static_cast<aclblasComplex*>(resources.deviceB), m),
            ACLBLAS_STATUS_INVALID_VALUE);
    }
    EXPECT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
}

std::vector<CtrsmParam> CtrsmTailRegressionCases()
{
    std::vector<CtrsmParam> result;
    // Exercise every 32B remainder, plus the review's larger tail-split examples.
    const std::pair<int, int> shapes[] = {{65, 65}, {66, 66},    {67, 67},   {68, 68},   {69, 69}, {70, 70},
                                          {71, 71}, {255, 1025}, {257, 255}, {255, 511}, {63, 32}, {64, 31},
                                          {64, 32}, {65, 33},    {127, 63},  {128, 64},  {129, 65}};
    for (const auto& shape : shapes) {
        for (const auto side : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
            for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (const auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                    for (const auto diag : {ACLBLAS_NON_UNIT, ACLBLAS_UNIT}) {
                        CtrsmParam p(csv_map{});
                        const auto index = result.size();
                        p.caseName = "TC_TAIL_" + std::to_string(index);
                        p.side = side;
                        p.uplo = uplo;
                        p.trans = trans;
                        p.diag = diag;
                        p.m = side == ACLBLAS_SIDE_LEFT ? shape.first : shape.second;
                        p.n = side == ACLBLAS_SIDE_LEFT ? shape.second : shape.first;
                        p.lda = shape.first + 3;
                        p.ldb = p.m + 5;
                        p.alpha = index % 2 == 0 ? aclblasComplex{0.75f, -0.5f} : aclblasComplex{-0.75f, 0.5f};
                        p.randomSeed = 20261009U + static_cast<uint32_t>(index);
                        p.expectResult = ACLBLAS_STATUS_SUCCESS;
                        result.push_back(p);
                    }
                }
            }
        }
    }
    return result;
}

std::vector<aclblasComplex> MakeCtrsmDiagonal(const CtrsmParam& p, const aclblasComplex& diagonal)
{
    auto a = MakeCtrsmA(p);
    const int order = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    for (int col = 0; col < order; ++col) {
        for (int row = 0; row < order; ++row) {
            const bool active = p.uplo == ACLBLAS_UPPER ? row <= col : row >= col;
            if (active && (row != col || p.diag != ACLBLAS_UNIT)) {
                a[static_cast<size_t>(col) * p.lda + row] = row == col ? diagonal : aclblasComplex{0.0f, 0.0f};
            }
        }
    }
    return a;
}

std::vector<aclblasComplex> MakeCtrsmConstantB(const CtrsmParam& p, const aclblasComplex& value)
{
    auto b = MakeCtrsmB(p);
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            b[static_cast<size_t>(col) * p.ldb + row] = value;
        }
    }
    return b;
}

TEST_F(CtrsmArch35Test, ScaledDiagonalAllOperations)
{
    for (auto p : CtrsmTailRegressionCases()) {
        const int order = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
        if (order > 129) {
            continue;
        }
        for (float scale : {1e20f, 1e-20f}) {
            // Keep NON_UNIT diagonals far from zero, as required by the task's input policy.
            const float diagonal = std::max(scale, static_cast<float>(order) + 5.0f);
            p.caseName += scale > 1.0f ? "_large" : "_small";
            SCOPED_TRACE(p.caseName);
            const auto a = MakeCtrsmDiagonal(p, {diagonal, diagonal});
            const auto b = MakeCtrsmConstantB(p, {scale, -scale});
            CheckCtrsmNumericalInput(handle_, p, a, b);
        }
    }
}

TEST_F(CtrsmArch35Test, OffsetBuffersAndUserWorkspace)
{
    constexpr size_t workspaceBytes = 32U * 1024U * 1024U;
    CtrsmPerformanceResources resources;
    ASSERT_EQ(aclrtMalloc(&resources.deviceBInput, workspaceBytes + 32, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    auto* workspace = static_cast<uint8_t*>(resources.deviceBInput) + 32;
    EXPECT_EQ(aclblasSetWorkspace(handle_, nullptr, workspaceBytes), ACLBLAS_STATUS_INVALID_VALUE);
    for (size_t available : {size_t(32), workspaceBytes}) {
        ASSERT_EQ(aclblasSetWorkspace(handle_, workspace, available), ACLBLAS_STATUS_SUCCESS);
        CtrsmParam p = CtrsmTailRegressionCases().front();
        const aclblasComplex guard = {123.0f, -456.0f};
        auto a = MakeCtrsmA(p);
        auto b = MakeCtrsmB(p);
        auto golden = b;
        ASSERT_EQ(
            aclblasCtrsmCpu(
                handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha, a.data(), p.lda, golden.data(), p.ldb),
            ACLBLAS_STATUS_SUCCESS);
        a.insert(a.begin(), guard);
        b.insert(b.begin(), guard);
        b.push_back(guard);
        const size_t bBytes = b.size() * sizeof(aclblasComplex);
        ASSERT_EQ(CtrsmAllocCopy(&resources.deviceA, a.data(), a.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(CtrsmAllocCopy(&resources.deviceB, b.data(), bBytes), ACL_SUCCESS);
        const auto status = aclblasCtrsm(
            handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha,
            static_cast<aclblasComplex*>(resources.deviceA) + 1, p.lda,
            static_cast<aclblasComplex*>(resources.deviceB) + 1, p.ldb);
        EXPECT_EQ(status, available == workspaceBytes ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_ALLOC_FAILED);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(b.data(), bBytes, resources.deviceB, bBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        EXPECT_EQ(b.front().real, guard.real);
        EXPECT_EQ(b.front().imag, guard.imag);
        EXPECT_EQ(b.back().real, guard.real);
        EXPECT_EQ(b.back().imag, guard.imag);
        if (status == ACLBLAS_STATUS_SUCCESS) {
            EXPECT_TRUE(VerifyCtrsmPrecision(p, std::vector<aclblasComplex>(b.begin() + 1, b.end() - 1), golden));
        }
        ASSERT_EQ(aclrtFree(resources.deviceA), ACL_SUCCESS);
        resources.deviceA = nullptr;
        ASSERT_EQ(aclrtFree(resources.deviceB), ACL_SUCCESS);
        resources.deviceB = nullptr;
    }
    ASSERT_EQ(aclblasSetStream(handle_, stream_), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CtrsmArch35Test, NonFiniteAllOperations)
{
    for (auto p : CtrsmTailRegressionCases()) {
        const int order = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
        if (order != 63 && order != 64) {
            continue;
        }
        for (const aclblasComplex alpha : {aclblasComplex{1.0f, 0.0f}, {-0.75f, 0.5f}}) {
            p.alpha = alpha;
            SCOPED_TRACE(p.caseName);
            const float infinity = std::numeric_limits<float>::infinity();
            for (float special : {infinity, -infinity, std::numeric_limits<float>::quiet_NaN(), 0.0f}) {
                SCOPED_TRACE(special);
                auto a = MakeCtrsmDiagonal(p, {static_cast<float>(order) + 5.0f, 0.0f});
                auto b = MakeCtrsmConstantB(p, {special, special == 0.0f ? 0.0f : 1.0f});
                if (special == 0.0f) {
                    a[p.uplo == ACLBLAS_UPPER ? p.lda : 1] = {infinity, 0.0f};
                }
                CheckCtrsmNumericalInput(handle_, p, a, b);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Ctrsm, CtrsmArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CtrsmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtrsmParam>);

INSTANTIATE_TEST_SUITE_P(
    CtrsmRegression, CtrsmArch35Test, ::testing::ValuesIn(CtrsmSpecialValueRegressionCases()),
    PrintCaseInfoString<CtrsmParam>);

INSTANTIATE_TEST_SUITE_P(
    CtrsmTailRegression, CtrsmArch35Test, ::testing::ValuesIn(CtrsmTailRegressionCases()),
    PrintCaseInfoString<CtrsmParam>);

TEST_P(CtrsmArch35Test, CsvDriven)
{
    const CtrsmParam& p = GetParam();
    std::vector<aclblasComplex> a = MakeCtrsmA(p);
    std::vector<aclblasComplex> bInput = MakeCtrsmB(p);
    const aclblasComplex* alpha = p.alphaNull ? nullptr : &p.alpha;

    if (p.performance) {
        CheckCtrsmPerformance(handle_, stream_, p, a, bInput);
        return;
    }

    std::vector<aclblasComplex> output = bInput;
    aclblasStatus_t status = aclblasCtrsmNpu(
        handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, alpha, a.empty() ? nullptr : a.data(), p.lda,
        output.empty() ? nullptr : output.data(), p.ldb);
    ASSERT_EQ(status, p.expectResult);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    std::vector<aclblasComplex> golden = bInput;
    ASSERT_EQ(
        aclblasCtrsmCpu(
            handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, alpha, a.empty() ? nullptr : a.data(), p.lda,
            golden.empty() ? nullptr : golden.data(), p.ldb),
        ACLBLAS_STATUS_SUCCESS);
    if (p.m == 0 || p.n == 0) {
        return;
    }
    EXPECT_TRUE(VerifyCtrsmPrecision(p, output, golden));
    EXPECT_TRUE(VerifyCtrsmPadding(p, output, bInput));
}
