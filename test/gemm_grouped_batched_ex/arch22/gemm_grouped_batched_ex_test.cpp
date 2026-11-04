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
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <atomic>
#include <thread>
#include "blas_test.h"
#include "aclblas_handle_internal.h"
#include "../gemm_grouped_batched_ex_param.h"
#include "../gemm_grouped_batched_ex_golden.h"
#include "verify.h"

class GemmGrouped22Test : public BlasTest<GemmGroupedBatchedExParam> {};
namespace {
void CheckAcl(aclError error)
{
    if (error != ACL_SUCCESS)
        throw std::runtime_error("ACL error " + std::to_string(error));
}
struct DeviceBuffer {
    void* ptr = nullptr;
    void* allocation = nullptr;
    explicit DeviceBuffer(size_t bytes, size_t offset = 0)
    {
        CheckAcl(aclrtMalloc(&allocation, std::max<size_t>(bytes, 32) + offset, ACL_MEM_MALLOC_HUGE_FIRST));
        ptr = static_cast<uint8_t*>(allocation) + offset;
    }
    ~DeviceBuffer()
    {
        if (allocation)
            aclrtFree(allocation);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    void Put(const void* data, size_t bytes)
    {
        if (bytes)
            CheckAcl(aclrtMemcpy(ptr, bytes, data, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void Get(void* data, size_t bytes)
    {
        if (bytes)
            CheckAcl(aclrtMemcpy(data, bytes, ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
};
struct Matrix {
    int group;
    std::vector<float> a, b, c, result, golden;
    std::unique_ptr<DeviceBuffer> da, db, dc;
};
uint32_t Bits(float v)
{
    uint32_t bits;
    std::copy_n(reinterpret_cast<const unsigned char*>(&v), sizeof(v), reinterpret_cast<unsigned char*>(&bits));
    return bits;
}
int EnvInt(const char* key, int fallback)
{
    const char* v = std::getenv(key);
    return v ? std::stoi(v) : fallback;
}
std::vector<GemmGroupedBatchedExParam> Cases()
{
    const char* csv = std::getenv("GROUPED_CSV");
    if (csv)
        return GetCasesFromCsv<GemmGroupedBatchedExParam>(csv);
    const auto mainPath = ReplaceFileExtension2Csv(__FILE__);
    auto cases = GetCasesFromCsv<GemmGroupedBatchedExParam>(mainPath);
    const auto supplementalPath = mainPath.substr(0, mainPath.find_last_of("/\\") + 1) + "supplemental.csv";
    const auto supplemental = GetCasesFromCsv<GemmGroupedBatchedExParam>(supplementalPath);
    cases.insert(cases.end(), supplemental.begin(), supplemental.end());
    return cases;
}
aclblasStatus_t Call(
    aclblasHandle_t h, const GemmGroupedBatchedExParam& p, void* ap, void* bp, void* cp, const void* alpha,
    const void* beta)
{
    return aclblasGemmGroupedBatchedEx(
        p.description.find("handle_null") != std::string::npos ? nullptr : h, p.Arg(p.transaArray, "transaArray"),
        p.Arg(p.transbArray, "transbArray"), p.Arg(p.mArray, "mArray"), p.Arg(p.nArray, "nArray"),
        p.Arg(p.kArray, "kArray"), alpha,
        ReadMap(p.raw, "aarray_fill") == "NULLPTR" ? nullptr : static_cast<const void* const*>(ap), p.Atype,
        p.Arg(p.ldaArray, "ldaArray"),
        ReadMap(p.raw, "barray_fill") == "NULLPTR" ? nullptr : static_cast<const void* const*>(bp), p.Btype,
        p.Arg(p.ldbArray, "ldbArray"), beta,
        ReadMap(p.raw, "carray_fill") == "NULLPTR" ? nullptr : static_cast<void* const*>(cp), p.Ctype,
        p.Arg(p.ldcArray, "ldcArray"), p.groupCount, p.Arg(p.groupSizeArray, "groupSize"), p.computeType);
}
std::vector<float> InputMatrix(int rows, int cols, int ld, const std::string& fill, uint32_t seed)
{
    if (fill.rfind("GAUSS_NORM", 0) != 0) {
        auto data = makeBlasMatrix(rows, cols, ld, fill, seed);
        if (data.empty() && ld > 0 && cols > 0)
            data.assign(static_cast<size_t>(ld) * cols, kBlasSentinel);
        return data;
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> mean(-5.0f, 5.0f), sigma(.1f, 2.0f);
    const float mu = mean(rng), sd = sigma(rng);
    std::normal_distribution<float> normal(mu, sd);
    std::vector<float> data(static_cast<size_t>(ld) * cols, kBlasSentinel);
    for (int col = 0; col < cols; ++col)
        for (int row = 0; row < rows; ++row)
            data[static_cast<size_t>(col) * ld + row] = normal(rng);
    std::cout << std::setprecision(12) << "NORMAL|seed=" << seed << "|mean=" << mu << "|sigma=" << sd << std::endl;
    return data;
}
void GoldenColumns(Matrix& matrix, const GemmGroupedBatchedExParam& p, int first, int columns)
{
    const int g = matrix.group;
    const size_t bOffset = p.transbArray[g] == ACLBLAS_OP_N ? static_cast<size_t>(first) * p.ldbArray[g] : first;
    const float* b = matrix.b.empty() ? nullptr : matrix.b.data() + bOffset;
    float* c = matrix.golden.empty() ? nullptr : matrix.golden.data() + static_cast<size_t>(first) * p.ldcArray[g];
    GemmGroupedBatchedCpuOne(
        p.transaArray[g], p.transbArray[g], p.mArray[g], columns, p.kArray[g], p.alphaArray[g], p.betaArray[g],
        matrix.a.data(), p.ldaArray[g], b, p.ldbArray[g], c, p.ldcArray[g], p.Ctype);
}

void GoldenAll(std::vector<Matrix>& matrices, const GemmGroupedBatchedExParam& p)
{
    struct Work {
        size_t matrix;
        int first, columns;
    };
    std::vector<Work> jobs;
    for (size_t i = 0; i < matrices.size(); ++i) {
        auto& matrix = matrices[i];
        matrix.golden = matrix.c;
        const int n = p.nArray[matrix.group];
        for (int first = 0; first < n; first += 256)
            jobs.push_back({i, first, std::min(256, n - first)});
    }
    std::atomic<size_t> next{0};
    const size_t workers = std::min<size_t>(std::max(1, EnvInt("GROUPED_GOLDEN_THREADS", 8)), jobs.size());
    std::vector<std::thread> threads;
    for (size_t i = 0; i < workers; ++i)
        threads.emplace_back([&]() {
            for (size_t index = next.fetch_add(1); index < jobs.size(); index = next.fetch_add(1)) {
                const auto& work = jobs[index];
                GoldenColumns(matrices[work.matrix], p, work.first, work.columns);
            }
        });
    for (auto& thread : threads)
        thread.join();
}

Matrix PrepareMatrix(const GemmGroupedBatchedExParam& p, int g, uint32_t instance)
{
    Matrix matrix;
    matrix.group = g;
    const int ar = p.transaArray[g] == ACLBLAS_OP_N ? p.mArray[g] : p.kArray[g],
              ac = p.transaArray[g] == ACLBLAS_OP_N ? p.kArray[g] : p.mArray[g];
    const int br = p.transbArray[g] == ACLBLAS_OP_N ? p.kArray[g] : p.nArray[g],
              bc = p.transbArray[g] == ACLBLAS_OP_N ? p.nArray[g] : p.kArray[g];
    matrix.a = InputMatrix(ar, ac, p.ldaArray[g], ReadMap(p.raw, "aarray_fill"), p.randomSeed + instance * 3);
    matrix.b = InputMatrix(br, bc, p.ldbArray[g], ReadMap(p.raw, "barray_fill"), p.randomSeed + instance * 3 + 1);
    matrix.c = InputMatrix(
        p.mArray[g], p.nArray[g], p.ldcArray[g], ReadMap(p.raw, "carray_fill"), p.randomSeed + instance * 3 + 2);
    if (p.description.find("scale_subnormals") != std::string::npos)
        for (size_t z = 0; z < matrix.c.size(); ++z)
            matrix.c[z] = std::numeric_limits<float>::denorm_min() * static_cast<float>(z % 7 + 1);
    matrix.da = std::make_unique<DeviceBuffer>(matrix.a.size() * 4, EnvInt("GROUPED_OFFSET", 0) * 4);
    matrix.db = std::make_unique<DeviceBuffer>(matrix.b.size() * 4, EnvInt("GROUPED_OFFSET", 0) * 4);
    matrix.dc = std::make_unique<DeviceBuffer>(matrix.c.size() * 4, EnvInt("GROUPED_OFFSET", 0) * 4);
    matrix.da->Put(matrix.a.data(), matrix.a.size() * 4);
    matrix.db->Put(matrix.b.data(), matrix.b.size() * 4);
    matrix.dc->Put(matrix.c.data(), matrix.c.size() * 4);
    return matrix;
}

VerifyConfig CaseVerifyConfig(const GemmGroupedBatchedExParam& p)
{
    VerifyConfig cfg;
    cfg.mereThreshold = p.mereThreshold > 0.0 ? p.mereThreshold : std::ldexp(1.0, -13);
    cfg.mareMultiplier = p.mareMultiplier > 0.0 ? p.mareMultiplier : 10.0;
    const auto mode = ReadMap(p.raw, "precision_mode", "MIXED_TOLERANCE");
    if (mode == "MERE_MARE") {
        cfg.mode = PrecisionMode::MERE_MARE;
    } else if (mode == "MIXED_TOLERANCE") {
        cfg.mode = PrecisionMode::MIXED_TOLERANCE;
    } else {
        throw std::invalid_argument("unsupported precision_mode: " + mode);
    }
    // Taskbook FLOAT32 contract overrides the generic dtype defaults.
    cfg.mixedAtol = parseDouble(ReadMap(p.raw, "mixed_atol", "0.0001220703125"));
    cfg.mixedRtol = parseDouble(ReadMap(p.raw, "mixed_rtol", "0.0001220703125"));
    cfg.mixedRequiredMatchedRatio = parseDouble(ReadMap(p.raw, "mixed_required_matched_ratio", "0.99"));
    cfg.mixedMaxAbsErrorLimit = parseDouble(ReadMap(p.raw, "mixed_max_abs_error_limit", "0.01"));
    cfg.mixedMantissaBits = 23;
    cfg.mixedEmin = -126;
    return cfg;
}

bool EquivalentNonfinite(float actual, float golden)
{
    return !std::isfinite(actual) && !std::isfinite(golden) &&
           (std::isnan(actual) || std::isnan(golden) || std::signbit(actual) == std::signbit(golden));
}

void VerifyMatrix(const Matrix& matrix, const GemmGroupedBatchedExParam& p)
{
    const int g = matrix.group;
    std::vector<float> actualFinite, goldenFinite;
    size_t paddingErrors = 0, exactErrors = 0, specialErrors = 0;
    const bool exact = p.alphaArray[g] == 0.0f || (p.kArray[g] == 0 && p.betaArray[g] == 1.0f);
    for (int col = 0; col < p.nArray[g]; ++col) {
        for (int row = 0; row < p.ldcArray[g]; ++row) {
            const size_t index = static_cast<size_t>(col) * p.ldcArray[g] + row;
            const float actual = matrix.result[index], golden = matrix.golden[index];
            if (row >= p.mArray[g]) {
                paddingErrors += Bits(actual) != Bits(matrix.c[index]);
                continue;
            }
            if (exact)
                exactErrors += Bits(actual) != Bits(golden);
            if (!std::isfinite(actual) || !std::isfinite(golden)) {
                specialErrors += !EquivalentNonfinite(actual, golden);
                continue;
            }
            actualFinite.push_back(actual);
            goldenFinite.push_back(golden);
        }
    }
    EXPECT_EQ(paddingErrors, 0U);
    EXPECT_EQ(exactErrors, 0U);
    EXPECT_EQ(specialErrors, 0U);
    // Special-value pairs are checked separately and excluded from the finite ratio.
    EXPECT_TRUE(
        Verifier::verifyVector(
            actualFinite.data(), goldenFinite.data(), actualFinite.size(), 1, CaseVerifyConfig(p),
            p.caseName + "_group_" + std::to_string(g)));
}

void CheckNegative(const GemmGroupedBatchedExParam& p, aclblasHandle_t handle)
{
    DeviceBuffer dummy(32), ap(32), bp(32), cp(32);
    void* pointers[] = {dummy.ptr};
    ap.Put(pointers, sizeof(pointers));
    bp.Put(pointers, sizeof(pointers));
    cp.Put(pointers, sizeof(pointers));
    auto status =
        Call(handle, p, ap.ptr, bp.ptr, cp.ptr, p.Arg(p.alphaArray, "alphaArray"), p.Arg(p.betaArray, "betaArray"));
    EXPECT_EQ(status, p.expectResult);
    ::testing::Test::RecordProperty("actual_status", status);
    std::cout << "RESULT|" << p.caseName << "|API|status=" << status << "|expected=" << p.expectResult << std::endl;
}

struct PositiveCase {
    const GemmGroupedBatchedExParam& p;
    aclblasHandle_t handle_;
    aclrtStream stream_;
    std::vector<Matrix> matrices;
    std::vector<void*> av, bv, cv;
    std::unique_ptr<DeviceBuffer> ap, bp, cp, deviceAlpha, deviceBeta;
    const void* alpha = nullptr;
    const void* beta = nullptr;
    void* pinnedAlpha = nullptr;
    void* pinnedBeta = nullptr;

    PositiveCase(const GemmGroupedBatchedExParam& params, aclblasHandle_t handle, aclrtStream stream)
        : p(params), handle_(handle), stream_(stream)
    {}

    void PrepareMatrices()
    {
        uint32_t instance = 0;
        for (int g = 0; g < p.groupCount; ++g) {
            for (int i = 0; i < p.groupSizeArray[g]; ++i, ++instance) {
                auto matrix = PrepareMatrix(p, g, instance);
                const bool unreadable = p.description.find("unreadable_ab") != std::string::npos;
                av.push_back(unreadable ? nullptr : matrix.da->ptr);
                bv.push_back(unreadable ? nullptr : matrix.db->ptr);
                cv.push_back(matrix.dc->ptr);
                matrices.push_back(std::move(matrix));
            }
        }
        ap = std::make_unique<DeviceBuffer>(av.size() * sizeof(void*));
        bp = std::make_unique<DeviceBuffer>(bv.size() * sizeof(void*));
        cp = std::make_unique<DeviceBuffer>(cv.size() * sizeof(void*));
        ap->Put(av.data(), av.size() * sizeof(void*));
        bp->Put(bv.data(), bv.size() * sizeof(void*));
        cp->Put(cv.data(), cv.size() * sizeof(void*));
    }
    void PrepareScalars()
    {
        deviceAlpha = std::make_unique<DeviceBuffer>(p.alphaArray.size() * 4);
        deviceBeta = std::make_unique<DeviceBuffer>(p.betaArray.size() * 4);
        const bool deviceScalars = EnvInt("GROUPED_DEVICE_SCALARS", 0) != 0;
        alpha = p.Arg(p.alphaArray, "alphaArray");
        beta = p.Arg(p.betaArray, "betaArray");
        if (deviceScalars && p.groupCount > 0) {
            std::vector<float> staleAlpha(p.alphaArray.size(), 0.0f), staleBeta(p.betaArray.size(), 1.0f);
            deviceAlpha->Put(staleAlpha.data(), staleAlpha.size() * 4);
            deviceBeta->Put(staleBeta.data(), staleBeta.size() * 4);
            CheckAcl(aclrtMallocHost(&pinnedAlpha, p.alphaArray.size() * 4));
            CheckAcl(aclrtMallocHost(&pinnedBeta, p.betaArray.size() * 4));
            std::copy_n(
                static_cast<const unsigned char*>(alpha), p.alphaArray.size() * sizeof(float),
                static_cast<unsigned char*>(pinnedAlpha));
            std::copy_n(
                static_cast<const unsigned char*>(beta), p.betaArray.size() * sizeof(float),
                static_cast<unsigned char*>(pinnedBeta));
            CheckAcl(aclrtMemcpyAsync(
                deviceAlpha->ptr, p.alphaArray.size() * 4, pinnedAlpha, p.alphaArray.size() * 4,
                ACL_MEMCPY_HOST_TO_DEVICE, stream_));
            CheckAcl(aclrtMemcpyAsync(
                deviceBeta->ptr, p.betaArray.size() * 4, pinnedBeta, p.betaArray.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE,
                stream_));
            alpha = deviceAlpha->ptr;
            beta = deviceBeta->ptr;
        }
    }
    void Run()
    {
        PrepareMatrices();
        PrepareScalars();
        ASSERT_EQ(Invoke(), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ReleasePinned();
        GoldenAll(matrices, p);
        for (size_t instance = 0; instance < matrices.size(); ++instance) {
            auto& matrix = matrices[instance];
            matrix.result.resize(matrix.c.size());
            matrix.dc->Get(matrix.result.data(), matrix.result.size() * 4);
            VerifyMatrix(matrix, p);
        }
        std::cout << "RESULT|" << p.caseName << "|NUMERIC|workspace_bytes=" << GetEffectiveWorkspaceSize(handle_)
                  << std::endl;
    }

    aclblasStatus_t Invoke() { return Call(handle_, p, ap->ptr, bp->ptr, cp->ptr, alpha, beta); }

    void ReleasePinned()
    {
        if (pinnedAlpha) {
            CheckAcl(aclrtFreeHost(pinnedAlpha));
            pinnedAlpha = nullptr;
        }
        if (pinnedBeta) {
            CheckAcl(aclrtFreeHost(pinnedBeta));
            pinnedBeta = nullptr;
        }
    }
};
} // namespace
INSTANTIATE_TEST_SUITE_P(
    GemmGrouped22, GemmGrouped22Test, ::testing::ValuesIn(Cases()), PrintCaseInfoString<GemmGroupedBatchedExParam>);
TEST_P(GemmGrouped22Test, CsvDriven)
{
    const auto& p = GetParam();
    if (!p.lengthsValid) {
        EXPECT_EQ(p.expectResult, ACLBLAS_STATUS_INVALID_VALUE);
        RecordProperty("validation_kind", "CSV_SCHEMA");
        std::cout << "RESULT|" << p.caseName << "|CSV_SCHEMA|REJECTED_LENGTH_MISMATCH" << std::endl;
        return;
    }
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        CheckNegative(p, handle_);
        return;
    }
    PositiveCase run(p, handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(run.Run());
}
