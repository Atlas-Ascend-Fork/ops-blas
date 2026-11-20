/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Independent Csyrkx GTest acceptance runner; no legacy syrkx implementation used. */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "blas_test.h"
#include "csyrkx_golden.h"
#include "fill.h"

namespace csyrkx_test {
constexpr size_t kGuardElements = 32;
constexpr int kColumnChunk = 64;

inline aclblasComplex Sentinel(size_t index)
{
    // Different real/imag and signed zeros detect accidental copies/conjugation.
    return {index % 2 ? -0.0f : -991.25f, index % 3 ? 777.5f : 0.0f};
}

std::vector<aclblasComplex> Matrix(int rows, int cols, int ld, const std::string& token, uint32_t seed)
{
    if (token == "NULLPTR")
        return {};
    // Negative cases still receive real device allocations, never host tensor
    // pointers. Allocate a safe span even for intentionally invalid leading dims.
    const int safeLd = std::max({1, rows, ld});
    const size_t count = static_cast<size_t>(safeLd) * std::max(1, cols) + kGuardElements;
    std::vector<aclblasComplex> matrix(count);
    for (size_t i = 0; i < count; ++i)
        matrix[i] = Sentinel(i);
    std::mt19937 realRng(seed);
    std::mt19937 imagRng(seed ^ 0x9e3779b9U);
    BlasFillMode fill(token);
    auto realGen = createGenerator(fill, realRng);
    auto imagGen = createGenerator(fill, imagRng);
    size_t sequence = 0;
    for (int col = 0; col < cols; ++col)
        for (int row = 0; row < rows; ++row) {
            const size_t offset = static_cast<size_t>(col) * safeLd + row;
            matrix[offset] = {realGen->at(sequence), imagGen->at(sequence)};
            ++sequence;
        }
    return matrix;
}

void CheckAcl(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(const std::vector<aclblasComplex>& host) { Upload(host); }
    ~DeviceBuffer()
    {
        if (data_)
            aclrtFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    aclblasComplex* Data() const { return static_cast<aclblasComplex*>(data_); }
    size_t Size() const { return bytes_ / sizeof(aclblasComplex); }
    void Allocate(size_t bytes)
    {
        if (data_ || !bytes)
            throw std::runtime_error("invalid DeviceBuffer allocation");
        CheckAcl(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        bytes_ = bytes;
    }
    void Upload(const std::vector<aclblasComplex>& host)
    {
        if (host.empty())
            return;
        const size_t bytes = host.size() * sizeof(aclblasComplex);
        if (!data_)
            Allocate(bytes);
        if (bytes != bytes_)
            throw std::runtime_error("upload size mismatch");
        CheckAcl(aclrtMemcpy(data_, bytes_, host.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    }
    std::vector<aclblasComplex> Read(size_t start, size_t count) const
    {
        if (start > Size() || count > Size() - start)
            throw std::runtime_error("read out of range");
        std::vector<aclblasComplex> result(count);
        if (count)
            CheckAcl(
                aclrtMemcpy(
                    result.data(), count * sizeof(aclblasComplex), Data() + start, count * sizeof(aclblasComplex),
                    ACL_MEMCPY_DEVICE_TO_HOST),
                "D2H");
        return result;
    }
    void Snapshot(const DeviceBuffer& source)
    {
        if (!source.bytes_)
            return;
        Allocate(source.bytes_);
        CheckAcl(aclrtMemcpy(data_, bytes_, source.data_, bytes_, ACL_MEMCPY_DEVICE_TO_DEVICE), "snapshot D2D");
    }

private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
};

std::vector<aclblasComplex> MatrixA(const Param& p)
{
    return Matrix(
        p.trans == ACLBLAS_OP_N ? p.n : p.k, p.trans == ACLBLAS_OP_N ? p.k : p.n, p.lda, p.aFill, p.randomSeed);
}

struct CaseData {
    std::vector<aclblasComplex> a, b, c;
    DeviceBuffer da, db, dc;
    explicit CaseData(const Param& p)
        : a(MatrixA(p)),
          b(Matrix(
              p.trans == ACLBLAS_OP_N ? p.n : p.k, p.trans == ACLBLAS_OP_N ? p.k : p.n, p.ldb, p.bFill,
              p.randomSeed + 1U)),
          c(Matrix(p.n, p.n, p.ldc, p.cFill, p.randomSeed + 2U)),
          da(a),
          db(b),
          dc(c)
    {}
};

aclblasStatus_t Launch(aclblasHandle_t handle, const Param& p, const CaseData& data)
{
    // No wrapper-side validation or synthesized status: all negative cases
    // reach the actual public API with the original CSV values.
    return aclblasCsyrkx(
        handle, p.uplo, p.trans, p.n, p.k, p.alphaNull ? nullptr : &p.alpha, data.da.Data(), p.lda, data.db.Data(),
        p.ldb, p.betaNull ? nullptr : &p.beta, data.dc.Data(), p.ldc);
}

void ExpectUnchanged(const DeviceBuffer& device, const std::vector<aclblasComplex>& original, const char* name)
{
    constexpr size_t chunk = 65536;
    ASSERT_EQ(device.Size(), original.size());
    for (size_t first = 0; first < original.size(); first += chunk) {
        const size_t count = std::min(chunk, original.size() - first);
        const auto actual = device.Read(first, count);
        ASSERT_EQ(std::memcmp(actual.data(), original.data() + first, count * sizeof(aclblasComplex)), 0)
            << name << " was modified in chunk beginning at complex element " << first;
    }
}

void ExpectEqualDevices(const DeviceBuffer& actual, const DeviceBuffer& expected)
{
    constexpr size_t chunk = 65536;
    ASSERT_EQ(actual.Size(), expected.Size());
    for (size_t first = 0; first < actual.Size(); first += chunk) {
        const size_t count = std::min(chunk, actual.Size() - first);
        const auto a = actual.Read(first, count);
        const auto b = expected.Read(first, count);
        ASSERT_EQ(std::memcmp(a.data(), b.data(), count * sizeof(aclblasComplex)), 0)
            << "bitwise determinism/T=C failure near complex element " << first;
    }
}

void AccumulateCorrectChunk(
    const Param& p, const CaseData& data, int first, int columns, const std::vector<aclblasComplex>& golden,
    const std::vector<aclblasComplex>& actual, Metrics& real, Metrics& imag, uint64_t& preservedFailures,
    size_t& firstPreservedFailure)
{
    for (int col = 0; col < columns; ++col)
        for (int row = 0; row < p.ldc; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldc + row;
            const size_t originalOffset = static_cast<size_t>(first + col) * p.ldc + row;
            if (row < p.n && p.Selected(row, first + col)) {
                real.Add(Component(actual[offset], 0), Component(golden[offset], 0), row, first + col, 0);
                imag.Add(Component(actual[offset], 1), Component(golden[offset], 1), row, first + col, 1);
                continue;
            }
            if (std::memcmp(&actual[offset], &data.c[originalOffset], sizeof(aclblasComplex)) != 0) {
                if (!preservedFailures)
                    firstPreservedFailure = originalOffset;
                ++preservedFailures;
            }
        }
}

void ExpectCorrect(const Param& p, const CaseData& data)
{
    if (p.n == 0) {
        ExpectUnchanged(data.dc, data.c, "n=0 C");
        return;
    }
    ASSERT_FALSE(data.c.empty());
    Metrics real;
    Metrics imag;
    uint64_t preservedFailures = 0;
    size_t firstPreservedFailure = 0;
    for (int first = 0; first < p.n; first += kColumnChunk) {
        const int columns = std::min(kColumnChunk, p.n - first);
        const auto golden = GoldenColumns(p, data.a, data.b, data.c, first, columns);
        const auto actual = data.dc.Read(static_cast<size_t>(first) * p.ldc, golden.size());
        AccumulateCorrectChunk(
            p, data, first, columns, golden, actual, real, imag, preservedFailures, firstPreservedFailure);
    }
    const size_t matrixSize = static_cast<size_t>(p.n) * p.ldc;
    const auto guard = data.dc.Read(matrixSize, data.c.size() - matrixSize);
    EXPECT_EQ(std::memcmp(guard.data(), data.c.data() + matrixSize, guard.size() * sizeof(aclblasComplex)), 0)
        << "C trailing guard modified";
    EXPECT_EQ(preservedFailures, 0U) << "non-uplo triangle/padding changed, first index=" << firstPreservedFailure;
    EXPECT_TRUE(real.Passed()) << "real: " << real.Describe();
    EXPECT_TRUE(imag.Passed()) << "imag: " << imag.Describe();
    std::cout << "CSYRKX_ACCURACY," << p.caseName << ",real," << real.Describe() << '\n'
              << "CSYRKX_ACCURACY," << p.caseName << ",imag," << imag.Describe() << '\n';
}

void RunCorrectness(aclblasHandle_t handle, aclrtStream stream, const Param& p)
{
    SCOPED_TRACE(p.caseName + ": " + p.description);
    CaseData data(p);
    const auto status = Launch(handle, p, data);
    ASSERT_EQ(status, p.expectResult);
    CheckAcl(aclrtSynchronizeStream(stream), "synchronize after API");
    if (status != ACLBLAS_STATUS_SUCCESS) {
        ExpectUnchanged(data.dc, data.c, "error-return C");
    } else {
        ExpectCorrect(p, data);
        if ((p.k == 0 || csyrkx_test::Zero(p.alpha)) && csyrkx_test::One(p.beta))
            ExpectUnchanged(data.dc, data.c, "identity update C");
        DeviceBuffer firstOutput;
        firstOutput.Snapshot(data.dc);
        data.dc.Upload(data.c); // SAME original C, not the previous output.
        ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(stream), "determinism synchronize");
        ExpectEqualDevices(data.dc, firstOutput);
        if (p.trans == ACLBLAS_OP_C && p.n > 0) {
            Param transpose = p;
            transpose.trans = ACLBLAS_OP_T;
            data.dc.Upload(data.c);
            ASSERT_EQ(Launch(handle, transpose, data), ACLBLAS_STATUS_SUCCESS);
            CheckAcl(aclrtSynchronizeStream(stream), "T=C synchronize");
            ExpectEqualDevices(data.dc, firstOutput);
        }
    }
    ExpectUnchanged(data.da, data.a, "A including padding/guard");
    ExpectUnchanged(data.db, data.b, "B including padding/guard");
}

class EventPair {
public:
    aclrtEvent begin = nullptr, end = nullptr;
    ~EventPair()
    {
        if (begin)
            aclrtDestroyEvent(begin);
        if (end)
            aclrtDestroyEvent(end);
    }
    void Init()
    {
        CheckAcl(aclrtCreateEvent(&begin), "create start event");
        CheckAcl(aclrtCreateEvent(&end), "create end event");
    }
};

void RunPerformance(aclblasHandle_t handle, aclrtStream stream, const Param& p, bool profileOnly)
{
    ASSERT_TRUE(p.IsPerformance());
    ASSERT_EQ(p.expectResult, ACLBLAS_STATUS_SUCCESS);
    // All supplied PF rows have beta=0. Thus each of the 56 calls has the
    // same effective input without a copy or host preparation in timed regions.
    ASSERT_TRUE(Zero(p.beta));
    CaseData data(p);
    EventPair events;
    if (!profileOnly)
        events.Init();
    std::vector<double> samples;
    std::cout << "CSYRKX_PERF_BEGIN," << p.caseName << ",warmup=5,measurements=51\n";
    for (int iteration = 0; iteration < 56; ++iteration) {
        if (iteration >= 5 && !profileOnly)
            CheckAcl(aclrtRecordEvent(events.begin, stream), "record begin");
        ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS) << "iteration=" << iteration;
        if (iteration >= 5 && !profileOnly)
            CheckAcl(aclrtRecordEvent(events.end, stream), "record end");
        CheckAcl(aclrtSynchronizeStream(stream), "performance synchronize");
        if (iteration >= 5 && !profileOnly) {
            float ms = 0.0f;
            CheckAcl(aclrtEventElapsedTime(&ms, events.begin, events.end), "elapsed time");
            ASSERT_TRUE(std::isfinite(ms) && ms >= 0.0f);
            samples.push_back(static_cast<double>(ms) * 1000.0);
        }
    }
    if (!profileOnly) {
        ASSERT_EQ(samples.size(), 51U);
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        std::cout << std::setprecision(12) << "CSYRKX_PERF," << p.caseName
                  << ",source=acl_event,warmup=5,samples=51,mean_us=" << mean << '\n';
    }
    std::cout << "CSYRKX_PERF_END," << p.caseName << ",successful_calls=56\n";
}

std::string Mode()
{
    const char* mode = std::getenv("CSYRKX_MODE");
    const std::string result = mode ? mode : "accuracy";
    if (result != "accuracy" && result != "performance" && result != "profile")
        throw std::runtime_error("CSYRKX_MODE must be accuracy, performance, or profile");
    return result;
}
} // namespace csyrkx_test

class CsyrkxIndependentTest : public BlasTest<csyrkx_test::Param> {};

INSTANTIATE_TEST_SUITE_P(
    Csyrkx, CsyrkxIndependentTest, ::testing::ValuesIn(csyrkx_test::LoadCases(__FILE__)),
    [](const ::testing::TestParamInfo<csyrkx_test::Param>& info) { return info.param.caseName; });

TEST_P(CsyrkxIndependentTest, CsvDriven)
{
    const auto& p = GetParam();
    const auto mode = csyrkx_test::Mode();
    if (mode == "accuracy") {
        csyrkx_test::RunCorrectness(handle_, stream_, p);
    } else {
        if (!p.IsPerformance())
            GTEST_SKIP() << "performance/profile mode only executes TC_PF";
        const char* skipGolden = std::getenv("CSYRKX_SKIP_GOLDEN");
        if (skipGolden && std::string(skipGolden) == "0") {
            // Optional preflight; do not enable under msprof because it adds API calls.
            ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunCorrectness(handle_, stream_, p));
            ASSERT_FALSE(HasFailure());
        }
        csyrkx_test::RunPerformance(handle_, stream_, p, mode == "profile");
    }
}

TEST_F(CsyrkxIndependentTest, NullHandle)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    csyrkx_test::Param p;
    csyrkx_test::CaseData data(p);
    EXPECT_EQ(csyrkx_test::Launch(nullptr, p, data), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    csyrkx_test::CheckAcl(aclrtSynchronizeStream(stream_), "null handle synchronize");
    csyrkx_test::ExpectUnchanged(data.dc, data.c, "null handle C");
}

TEST_F(CsyrkxIndependentTest, ValidationBeforeNoOp)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    for (int variant = 0; variant < 8; ++variant) {
        csyrkx_test::Param p;
        p.n = 0;
        p.k = 0;
        p.lda = p.ldb = p.ldc = 1;
        p.caseName = "validation_before_noop_" + std::to_string(variant);
        p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        if (variant == 0)
            p.alphaNull = true;
        if (variant == 1)
            p.betaNull = true;
        if (variant == 2)
            p.uplo = static_cast<aclblasFillMode_t>(999);
        if (variant == 3)
            p.trans = static_cast<aclblasOperation_t>(999);
        if (variant == 4)
            p.lda = 0;
        if (variant == 5)
            p.ldb = 0;
        if (variant == 6)
            p.ldc = 0;
        if (variant == 7)
            p.k = -1;
        csyrkx_test::RunCorrectness(handle_, stream_, p);
    }
}

namespace csyrkx_test {
template <typename Verify>
void VerifyOverflowingInputSpans(
    aclblasHandle_t handle, aclblasFillMode_t uplo, DeviceBuffer& a, DeviceBuffer& b, DeviceBuffer& c,
    const aclblasComplex& zero, const aclblasComplex& one, const aclblasComplex& riskyAlpha, Verify& verify,
    bool& completed)
{
    const int limit = std::numeric_limits<int>::max();
    for (bool scalarRisk : {false, true})
        for (int input = 0; input < 3; ++input) {
            SCOPED_TRACE("input=" + std::to_string(input) + " scalarRisk=" + std::to_string(scalarRisk));
            const int lda = input == 2 ? 2 : limit;
            const int ldb = input == 1 ? 2 : limit;
            const auto& alpha = scalarRisk ? riskyAlpha : one;
            ASSERT_NO_FATAL_FAILURE(verify(
                aclblasCsyrkx(
                    handle, uplo, ACLBLAS_OP_N, 2, limit, &alpha, a.Data(), lda, b.Data(), ldb, &zero, c.Data(), 2),
                ACLBLAS_STATUS_INVALID_VALUE));
        }
    completed = true;
}
} // namespace csyrkx_test

TEST_F(CsyrkxIndependentTest, AddressableTensorSpansBeforeLaunch)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    using namespace csyrkx_test;
    // Never construct CaseData with impossible dimensions: every pointer below
    // is a real allocation containing only four complex values plus its guard.
    const auto original = Matrix(2, 2, 2, "VALUE_NORM_1", 701);
    DeviceBuffer a(original);
    DeviceBuffer b(original);
    DeviceBuffer c(original);
    const int limit = std::numeric_limits<int>::max();
    const int hugeN = (1 << 30) + 1;
    const aclblasComplex zero = {0.0f, 0.0f};
    const aclblasComplex one = {1.0f, 0.0f};
    const aclblasComplex riskyAlpha = {std::ldexp(1.0f, 21), 0.0f};
    auto verify = [&](aclblasStatus_t actual, aclblasStatus_t expected) {
        ASSERT_EQ(actual, expected);
        CheckAcl(aclrtSynchronizeStream(stream_), "addressable span synchronize");
        ExpectUnchanged(c, original, "span validation C including guard");
        ExpectUnchanged(a, original, "span validation A including guard");
        ExpectUnchanged(b, original, "span validation B including guard");
    };
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        SCOPED_TRACE("span validation uplo=" + std::to_string(static_cast<int>(uplo)));
        // Last diagonal complex index is 2^61, hence its byte offset is 2^64.
        // Both the scalar-risk ordered route and the alpha0 scale-only route
        // must reject the referenced C span before any kernel can use it.
        ASSERT_NO_FATAL_FAILURE(verify(
            aclblasCsyrkx(
                handle_, uplo, ACLBLAS_OP_N, hugeN, 1, &riskyAlpha, a.Data(), hugeN, b.Data(), hugeN, &zero, c.Data(),
                limit),
            ACLBLAS_STATUS_INVALID_VALUE));
        ASSERT_NO_FATAL_FAILURE(verify(
            aclblasCsyrkx(
                handle_, uplo, ACLBLAS_OP_N, hugeN, 1, &zero, nullptr, hugeN, nullptr, hugeN, &zero, c.Data(), limit),
            ACLBLAS_STATUS_INVALID_VALUE));
        // Isolate A and B as well as rejecting both overflowing spans together.
        // n=2,k=INT_MAX,ld=INT_MAX exceeds SIZE_MAX bytes on a 64-bit host;
        // ld=2 remains addressable and is never accessed because the other input fails.
        bool completed = false;
        VerifyOverflowingInputSpans(handle_, uplo, a, b, c, zero, one, riskyAlpha, verify, completed);
        if (!completed)
            return;
        // The early no-op must not validate spans or pointers of unused inputs.
        ASSERT_NO_FATAL_FAILURE(verify(
            aclblasCsyrkx(
                handle_, uplo, ACLBLAS_OP_N, 2, limit, &zero, nullptr, limit, nullptr, limit, &one, c.Data(), 2),
            ACLBLAS_STATUS_SUCCESS));
        // n=0 still validates scalar/enumeration/leading-dimension arguments,
        // but no tensor pointer is referenced even with enormous logical K.
        ASSERT_NO_FATAL_FAILURE(verify(
            aclblasCsyrkx(
                handle_, uplo, ACLBLAS_OP_N, 0, limit, &riskyAlpha, nullptr, limit, nullptr, limit, &zero, nullptr, 1),
            ACLBLAS_STATUS_SUCCESS));
    }
}

TEST_F(CsyrkxIndependentTest, BetaZeroIgnoresPoisonedC)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
            for (const char* fill : {"VALUE_NORM_NAN", "VALUE_NORM_INF"}) {
                csyrkx_test::Param p;
                p.trans = trans;
                p.uplo = uplo;
                p.beta = {0.0f, 0.0f};
                p.cFill = fill;
                csyrkx_test::RunCorrectness(handle_, stream_, p);
            }
}

TEST_F(CsyrkxIndependentTest, ZeroKAndAlphaDoNotReadUnusedInputs)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
        for (int variant = 0; variant < 4; ++variant) {
            csyrkx_test::Param p;
            p.uplo = uplo;
            p.aFill = p.bFill = "NULLPTR";
            p.beta = variant % 2 ? aclblasComplex{0.0f, 0.0f} : aclblasComplex{1.0f, 0.0f};
            p.cFill = "VALUE_NORM_NAN";
            if (variant < 2) {
                p.k = 0;
                p.alpha = {std::numeric_limits<float>::infinity(), 0.0f};
            } else
                p.alpha = {0.0f, 0.0f};
            csyrkx_test::RunCorrectness(handle_, stream_, p);
        }
}

TEST_F(CsyrkxIndependentTest, GaussianComplexComponents)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            csyrkx_test::Param p;
            p.trans = trans;
            p.uplo = uplo;
            p.aFill = p.bFill = p.cFill = "GAUSS_NORM_5_2";
            csyrkx_test::RunCorrectness(handle_, stream_, p);
        }
}

TEST_F(CsyrkxIndependentTest, LongKChunkTails)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    using namespace csyrkx_test;
    // 48 combinations, independent of CSV ids. Cover a single element, partial
    // tiles, and both sides of the 64-column boundary without a costly large-n
    // reference. Both K values exceed 2048 and leave tails in 128-element chunks.
    for (int n : {1, 17, 64, 65})
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
                for (int betaKind = 0; betaKind < 2; ++betaKind) {
                    Param p;
                    p.n = n;
                    p.k = (n + betaKind) % 2 ? 2051 : 2049;
                    p.trans = trans;
                    p.uplo = uplo;
                    const int storedRows = trans == ACLBLAS_OP_N ? n : p.k;
                    p.lda = storedRows + 3;
                    p.ldb = storedRows + 5;
                    p.ldc = n + 7;
                    p.alpha = {0.625f, -0.375f};
                    p.beta = betaKind ? aclblasComplex{-0.25f, 0.5f} : aclblasComplex{0.0f, 0.0f};
                    p.cFill = betaKind ? "RANDOM_NORM_5_5" : (n == 64 ? "VALUE_NORM_INF" : "VALUE_NORM_NAN");
                    p.randomSeed = 271828U + static_cast<uint32_t>(n);
                    p.caseName = "longk_n" + std::to_string(n) + "_k" + std::to_string(p.k) +
                                 (trans == ACLBLAS_OP_N ? "_N" : (trans == ACLBLAS_OP_T ? "_T" : "_C")) +
                                 (uplo == ACLBLAS_UPPER ? "_U" : "_L") + "_beta" + std::to_string(betaKind);
                    p.description = "long-K N/T/C, padded strides, K-partition tails";
                    // Netlib comparison, all padding/triangle guards, same-input
                    // repeat bitwise equality and OP_C=OP_T are checked here.
                    ASSERT_NO_FATAL_FAILURE(RunCorrectness(handle_, stream_, p));
                }
}

TEST_F(CsyrkxIndependentTest, PaddedLongKTailMemcheck)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    using namespace csyrkx_test;
    // Consecutive padded n=1 cases reproduce cross-core partial-block writes
    // while retaining both long-K tails and workspace reuse under memcheck.
    for (int betaKind = 0; betaKind < 2; ++betaKind) {
        Param p;
        p.n = 1;
        p.k = betaKind == 0 ? 2051 : 2049;
        p.lda = p.n + 3;
        p.ldb = p.n + 5;
        p.ldc = p.n + 7;
        p.alpha = {0.625f, -0.375f};
        p.beta = betaKind ? aclblasComplex{-0.25f, 0.5f} : aclblasComplex{0.0f, 0.0f};
        p.cFill = betaKind ? "RANDOM_NORM_5_5" : "VALUE_NORM_NAN";
        p.randomSeed = 271829U;
        p.caseName = "padded_longk_" + std::to_string(p.k);
        ASSERT_NO_FATAL_FAILURE(RunCorrectness(handle_, stream_, p));
    }
}

namespace csyrkx_test {
void CheckFiniteCancellation(
    aclblasHandle_t handle, aclrtStream stream, int n, float alpha, aclblasFillMode_t uplo, bool& completed)
{
    Param p;
    p.n = n;
    p.k = 4;
    p.lda = n + 3;
    p.ldb = n + 5;
    p.ldc = n + 7;
    p.uplo = uplo;
    p.alpha = {alpha, 0.0f};
    p.beta = {0.0f, 0.0f};
    p.aFill = p.bFill = "VALUE_NORM_0";
    p.cFill = "VALUE_NORM_NAN";
    p.caseName = "finite_cancellation_n" + std::to_string(n) + "_alpha" + std::to_string(alpha);
    SCOPED_TRACE(p.caseName + " uplo=" + std::to_string(static_cast<int>(uplo)));
    CaseData data(p);
    for (int col = 0; col < p.k; ++col)
        for (int row = 0; row < p.n; ++row) {
            const float sign = col % 2 ? -1.0f : 1.0f;
            data.a[static_cast<size_t>(col) * p.lda + row] = {1.0e38f, sign * 1.0e38f};
            data.b[static_cast<size_t>(col) * p.ldb + row] = {1.0f, sign};
        }
    data.da.Upload(data.a);
    data.db.Upload(data.b);
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "finite cancellation synchronize");
    ExpectCorrect(p, data);
    const auto actual = data.dc.Read(0, static_cast<size_t>(p.ldc) * p.n);
    for (int col = 0; col < p.n; ++col)
        for (int row = 0; row < p.n; ++row) {
            if (!p.Selected(row, col))
                continue;
            const auto& z = actual[static_cast<size_t>(col) * p.ldc + row];
            EXPECT_EQ(Component(z, 0), 0.0f) << "real row=" << row << " col=" << col;
            EXPECT_EQ(Component(z, 1), 0.0f) << "imag row=" << row << " col=" << col;
        }
    DeviceBuffer first;
    first.Snapshot(data.dc);
    data.dc.Upload(data.c);
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "finite cancellation determinism");
    ExpectEqualDevices(data.dc, first);
    ExpectUnchanged(data.da, data.a, "cancellation A");
    ExpectUnchanged(data.db, data.b, "cancellation B");
    completed = true;
}

void VerifyBetaFirstGolden(const Param& p, const std::vector<aclblasComplex>& golden, int n, float x, bool& completed)
{
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row) {
            if (!p.Selected(row, col))
                continue;
            const auto& z = golden[static_cast<size_t>(col) * p.ldc + row];
            ASSERT_TRUE(std::isfinite(Component(z, 0)) && std::isfinite(Component(z, 1)));
            ASSERT_EQ(Component(z, 0), x);
            ASSERT_EQ(Component(z, 1), 0.0f);
        }
    completed = true;
}

void ExpectBetaFirstOutput(const Param& p, const std::vector<aclblasComplex>& actual, int n, float x)
{
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row) {
            if (!p.Selected(row, col))
                continue;
            const auto& z = actual[static_cast<size_t>(col) * p.ldc + row];
            EXPECT_EQ(Component(z, 0), x) << "real row=" << row << " col=" << col;
            EXPECT_EQ(Component(z, 1), 0.0f) << "imag row=" << row << " col=" << col;
        }
}

void CheckFiniteBetaFirst(aclblasHandle_t handle, aclrtStream stream, int n, aclblasFillMode_t uplo, bool& completed)
{
    Param p;
    p.n = n;
    p.k = 2;
    p.trans = ACLBLAS_OP_N;
    p.uplo = uplo;
    p.lda = n + 3;
    p.ldb = n + 5;
    p.ldc = n + 7;
    p.alpha = p.beta = {1.0f, 0.0f};
    p.aFill = p.bFill = p.cFill = "VALUE_NORM_0";
    p.caseName = "finite_beta_first_n" + std::to_string(n);
    SCOPED_TRACE(p.caseName + " uplo=" + std::to_string(static_cast<int>(uplo)));
    CaseData data(p);
    const float x = 2.0e38f;
    for (int col = 0; col < p.k; ++col)
        for (int row = 0; row < n; ++row) {
            data.a[static_cast<size_t>(col) * p.lda + row] = {x, 0.0f};
            data.b[static_cast<size_t>(col) * p.ldb + row] = {1.0f, 0.0f};
        }
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row)
            data.c[static_cast<size_t>(col) * p.ldc + row] = {-x, 0.0f};
    // Assert the ACTUAL linked Netlib golden is finite before launching NPU;
    // no emulated/fp64 reference substitutes for this cblas_cgemm call.
    const auto golden = GoldenColumns(p, data.a, data.b, data.c, 0, n);
    bool goldenCompleted = false;
    VerifyBetaFirstGolden(p, golden, n, x, goldenCompleted);
    if (!goldenCompleted)
        return;
    data.da.Upload(data.a);
    data.db.Upload(data.b);
    data.dc.Upload(data.c);
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "beta-first cancellation synchronize");
    ExpectCorrect(p, data); // includes opposite triangle, padding and 32-element guard
    const auto actual = data.dc.Read(0, static_cast<size_t>(p.ldc) * n);
    ExpectBetaFirstOutput(p, actual, n, x);
    DeviceBuffer first;
    first.Snapshot(data.dc);
    data.dc.Upload(data.c);
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "beta-first cancellation determinism");
    ExpectEqualDevices(data.dc, first);
    ExpectUnchanged(data.da, data.a, "beta-first A");
    ExpectUnchanged(data.db, data.b, "beta-first B");
    completed = true;
}
} // namespace csyrkx_test

TEST_F(CsyrkxIndependentTest, FiniteComplexCancellationDespiteRealPartialOverflow)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    // Each COMPLEX product has real=0 and imag=+/-2e38, whose sequential
    // sum is finite zero. Separately accumulating rr/ii over four terms
    // overflows both partial sums and incorrectly yields Inf-Inf => NaN.
    for (int n : {1, 33})
        for (float alpha : {1.0f, 1.0e-38f})
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                bool completed = false;
                csyrkx_test::CheckFiniteCancellation(handle_, stream_, n, alpha, uplo, completed);
                if (!completed)
                    return;
            }
    // N-mode Netlib initializes beta*C BEFORE accumulating K terms:
    // (-X) + X + X = X, whereas sum-first X+X overflows before adding -X.
    // Extend this existing regression without changing its name or CSV cases.
    for (int n : {1, 33})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            bool completed = false;
            csyrkx_test::CheckFiniteBetaFirst(handle_, stream_, n, uplo, completed);
            if (!completed)
                return;
        }
}

namespace csyrkx_test {
void FillMixedOverflow(const Param& p, CaseData& data, bool normal, bool betaZero)
{
    for (int logical = 0; logical < p.n; ++logical)
        for (int l = 0; l < p.k; ++l) {
            const float sign = l % 2 ? -1.0f : 1.0f;
            const bool exceptional = logical == 0 || logical == p.n - 1;
            const size_t ap =
                normal ? static_cast<size_t>(l) * p.lda + logical : static_cast<size_t>(logical) * p.lda + l;
            const size_t bp =
                normal ? static_cast<size_t>(l) * p.ldb + logical : static_cast<size_t>(logical) * p.ldb + l;
            data.a[ap] = exceptional ? aclblasComplex{1.0e38f, sign * 1.0e38f} : aclblasComplex{2.0f, sign * 3.0f};
            data.b[bp] = {1.0f, sign};
        }
    if (!betaZero) {
        for (int col = 0; col < p.n; ++col)
            for (int row = 0; row < p.n; ++row)
                data.c[static_cast<size_t>(col) * p.ldc + row] = {4.0f + 0.25f * (row % 3), 2.0f + 0.5f * (col % 5)};
    }
}

void VerifyMixedOverflowGolden(
    const Param& p, const CaseData& data, bool normal, bool betaZero, const std::vector<int>& probeColumns,
    std::vector<std::vector<aclblasComplex>>& probeGolden, bool& completed)
{
    for (int col : probeColumns) {
        probeGolden.push_back(GoldenColumns(p, data.a, data.b, data.c, col, 1));
        const auto& golden = probeGolden.back();
        for (int row = 0; row < p.n; ++row) {
            if (!p.Selected(row, col))
                continue;
            const bool exceptional = row == 0 || row == p.n - 1;
            const auto& old = data.c[static_cast<size_t>(col) * p.ldc + row];
            const float betaReal = betaZero ? 0.0f : -0.25f * Component(old, 0) - 0.5f * Component(old, 1);
            const float betaImag = betaZero ? 0.0f : -0.25f * Component(old, 1) + 0.5f * Component(old, 0);
            ASSERT_TRUE(std::isfinite(Component(golden[row], 0)) && std::isfinite(Component(golden[row], 1)))
                << "fixture must have finite Netlib output row=" << row << " col=" << col;
            ASSERT_EQ(Component(golden[row], 0), betaReal + (exceptional ? 0.0f : -2.0f));
            // N starts with beta*C: huge imaginary terms lose its
            // small imaginary component before cancelling to zero.
            // T/C add beta*C only after the zero dot product.
            ASSERT_EQ(Component(golden[row], 1), normal && exceptional ? 0.0f : betaImag);
        }
    }
    completed = true;
}

void ExpectMixedOverflowAnchors(
    const Param& p, const CaseData& data, const std::vector<int>& probeColumns,
    const std::vector<std::vector<aclblasComplex>>& probeGolden)
{
    for (size_t probe = 0; probe < probeColumns.size(); ++probe) {
        const int col = probeColumns[probe];
        const auto actual = data.dc.Read(static_cast<size_t>(col) * p.ldc, p.n);
        for (int row = 0; row < p.n; ++row) {
            if (!p.Selected(row, col))
                continue;
            EXPECT_EQ(Component(actual[row], 0), Component(probeGolden[probe][row], 0))
                << "real row=" << row << " col=" << col;
            EXPECT_EQ(Component(actual[row], 1), Component(probeGolden[probe][row], 1))
                << "imag row=" << row << " col=" << col;
        }
    }
}

Param MakeMixedOverflowParam(bool betaZero, aclblasOperation_t trans, aclblasFillMode_t uplo)
{
    Param p;
    p.n = betaZero ? 67 : 2051;
    p.k = 4;
    p.trans = trans;
    p.uplo = uplo;
    const bool normal = trans == ACLBLAS_OP_N;
    const int storedRows = normal ? p.n : p.k;
    p.lda = storedRows + 3;
    p.ldb = storedRows + 5;
    p.ldc = p.n + 7;
    p.alpha = {0.5f, 0.0f};
    p.beta = betaZero ? aclblasComplex{0.0f, 0.0f} : aclblasComplex{-0.25f, 0.5f};
    p.aFill = p.bFill = "VALUE_NORM_0";
    p.cFill = betaZero ? "VALUE_NORM_NAN" : "VALUE_NORM_0";
    p.caseName = "mixed_overflow_n" + std::to_string(p.n) + (normal ? "_N" : (trans == ACLBLAS_OP_T ? "_T" : "_C")) +
                 (uplo == ACLBLAS_UPPER ? "_U" : "_L") + (betaZero ? "_beta0" : "_beta_complex");
    return p;
}

void RunMixedOverflow(
    aclblasHandle_t handle, aclrtStream stream, bool betaZero, aclblasOperation_t trans, aclblasFillMode_t uplo,
    bool& completed)
{
    Param p = MakeMixedOverflowParam(betaZero, trans, uplo);
    const bool normal = trans == ACLBLAS_OP_N;
    SCOPED_TRACE(p.caseName);
    CaseData data(p);
    FillMixedOverflow(p, data, normal, betaZero);
    // Keep exact linked-Netlib anchors so sparse bad lanes cannot
    // hide behind the aggregate 99% metric. The full selected
    // column crosses 2048, and n=67 also exercises a short VF tail.
    const std::vector<int> probeColumns = {0, p.n / 2, p.n - 1};
    std::vector<std::vector<aclblasComplex>> probeGolden;
    bool goldenCompleted = false;
    VerifyMixedOverflowGolden(p, data, normal, betaZero, probeColumns, probeGolden, goldenCompleted);
    if (!goldenCompleted)
        return;
    data.da.Upload(data.a);
    data.db.Upload(data.b);
    data.dc.Upload(data.c);
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "mixed overflow synchronize");
    ExpectCorrect(p, data); // real Netlib, opposite triangle, ldc padding and trailing guard
    ExpectMixedOverflowAnchors(p, data, probeColumns, probeGolden);
    DeviceBuffer first;
    first.Snapshot(data.dc);
    data.dc.Upload(data.c); // repeat from ORIGINAL C, never the updated output
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "mixed overflow determinism");
    ExpectEqualDevices(data.dc, first);
    if (trans == ACLBLAS_OP_C) {
        Param transpose = p;
        transpose.trans = ACLBLAS_OP_T;
        data.dc.Upload(data.c);
        ASSERT_EQ(Launch(handle, transpose, data), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(stream), "mixed overflow T=C");
        ExpectEqualDevices(data.dc, first);
    }
    ExpectUnchanged(data.da, data.a, "mixed overflow A including padding/guard");
    ExpectUnchanged(data.db, data.b, "mixed overflow B including padding/guard");
    completed = true;
}
} // namespace csyrkx_test

TEST_F(CsyrkxIndependentTest, MixedOverflowChunkRepair)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    // Row 0 and row n-1 overflow the independent real-product planes, but
    // their ordered complex dot products are finite zero. Ordinary rows have
    // dot=-4+0i. At n=2051 the full selected column contains both exceptional
    // endpoints in DIFFERENT 2048-element chunks, each mixed with finite lanes.
    for (bool betaZero : {false, true})
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                bool completed = false;
                csyrkx_test::RunMixedOverflow(handle_, stream_, betaZero, trans, uplo, completed);
                if (!completed)
                    return;
            }
}

TEST_F(CsyrkxIndependentTest, MixedOverflowChunkRepairLarge)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            bool completed = false;
            csyrkx_test::RunMixedOverflow(handle_, stream_, false, trans, uplo, completed);
            ASSERT_TRUE(completed);
        }
}

TEST_F(CsyrkxIndependentTest, MixedOverflowChunkRepairSmall)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            bool completed = false;
            csyrkx_test::RunMixedOverflow(handle_, stream_, true, trans, uplo, completed);
            ASSERT_TRUE(completed);
        }
}

namespace csyrkx_test {
void VerifyNormalExtremeGolden(
    const Param& p, const std::vector<aclblasComplex>& golden, int n, int variant, bool& completed)
{
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row) {
            if (!p.Selected(row, col))
                continue;
            const auto& z = golden[static_cast<size_t>(col) * p.ldc + row];
            ASSERT_TRUE(std::isinf(Component(z, 0)) && !std::signbit(Component(z, 0)));
            if (variant) {
                ASSERT_TRUE(std::isnan(Component(z, 1)));
            } else {
                ASSERT_EQ(Component(z, 1), 0.0f);
            }
        }
    completed = true;
}

void RunNormalExtremeScaling(
    aclblasHandle_t handle, aclrtStream stream, int variant, int n, aclblasFillMode_t uplo, bool& completed)
{
    const float inf = std::numeric_limits<float>::infinity();
    Param p;
    p.n = n;
    p.k = 1;
    p.trans = ACLBLAS_OP_N;
    p.uplo = uplo;
    p.lda = n + 3;
    p.ldb = n + 5;
    p.ldc = n + 7;
    p.alpha = variant ? aclblasComplex{2.0e38f, 0.0f} : aclblasComplex{1.0f, 0.0f};
    p.beta = variant ? aclblasComplex{0.0f, 0.0f} : aclblasComplex{1.0f, 0.0f};
    p.aFill = p.bFill = p.cFill = "VALUE_NORM_0";
    p.caseName = "normal_extreme_variant" + std::to_string(variant) + "_n" + std::to_string(n);
    SCOPED_TRACE(p.caseName + " uplo=" + std::to_string(static_cast<int>(uplo)));
    CaseData data(p);
    for (int row = 0; row < n; ++row) {
        data.a[row] = {variant ? 1.0e-38f : 1.0f, 0.0f};
        data.b[row] = {variant ? 2.0f : 1.0f, 0.0f};
    }
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row)
            data.c[static_cast<size_t>(col) * p.ldc + row] = {variant ? 0.0f : inf, 0.0f};
    const auto golden = GoldenColumns(p, data.a, data.b, data.c, 0, n);
    bool goldenCompleted = false;
    VerifyNormalExtremeGolden(p, golden, n, variant, goldenCompleted);
    if (!goldenCompleted)
        return;
    data.da.Upload(data.a);
    data.db.Upload(data.b);
    data.dc.Upload(data.c);
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "normal extreme scaling synchronize");
    ExpectCorrect(p, data); // official nonfinite equivalence, padding and guard checks
    DeviceBuffer first;
    first.Snapshot(data.dc);
    data.dc.Upload(data.c);
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "normal extreme scaling determinism");
    ExpectEqualDevices(data.dc, first);
    ExpectUnchanged(data.da, data.a, "extreme scaling A");
    ExpectUnchanged(data.db, data.b, "extreme scaling B");
    completed = true;
}
} // namespace csyrkx_test

TEST_F(CsyrkxIndependentTest, NormalTransposeExtremeScalingOrder)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    // CPU-only linked Netlib probe confirmed beta1 preserves (Inf,0), while
    // N-mode alpha*B overflow produces (Inf,NaN) before multiplying small A.
    for (int variant = 0; variant < 2; ++variant)
        for (int n : {1, 33}) {
            if (variant == 1 && n != 1)
                continue; // only requested small-direct extreme shape
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                bool completed = false;
                csyrkx_test::RunNormalExtremeScaling(handle_, stream_, variant, n, uplo, completed);
                if (!completed)
                    return;
            }
        }
}

namespace csyrkx_test {
// These columns straddle both the 64-column golden slices and 2048-lane repair
// boundaries. Small cases cover every column; large cases retain exact anchors
// in addition to ExpectCorrect's full selected-triangle Netlib comparison.
std::vector<int> RiskProbeColumns(int n)
{
    std::vector<int> columns;
    if (n <= 65) {
        for (int col = 0; col < n; ++col)
            columns.push_back(col);
    } else {
        for (int col : {0, 1, 31, 32, 63, 64, n / 2, 2047, 2048, n - 1})
            if (col < n)
                columns.push_back(col);
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    }
    return columns;
}

void ExpectCustomRiskCase(aclblasHandle_t handle, aclrtStream stream, const Param& p, CaseData& data)
{
    data.da.Upload(data.a);
    data.db.Upload(data.b);
    data.dc.Upload(data.c);
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "custom risk synchronize");
    ExpectCorrect(p, data); // full real Netlib, opposite triangle, padding and guard
    for (int col : RiskProbeColumns(p.n)) {
        const auto golden = GoldenColumns(p, data.a, data.b, data.c, col, 1);
        const auto actual = data.dc.Read(static_cast<size_t>(col) * p.ldc, p.n);
        for (int row = 0; row < p.n; ++row) {
            if (!p.Selected(row, col))
                continue;
            for (int component = 0; component < 2; ++component) {
                const float expected = Component(golden[row], component);
                const float value = Component(actual[row], component);
                if (std::isfinite(expected)) {
                    // These binary-exact fixtures have no finite rounding error.
                    // In particular, a second beta update cannot hide in 1% outliers.
                    EXPECT_EQ(value, expected) << "row=" << row << " col=" << col << " component=" << component;
                } else {
                    EXPECT_TRUE(NonFiniteEquivalent(value, expected))
                        << "row=" << row << " col=" << col << " component=" << component << " actual=" << value
                        << " golden=" << expected;
                }
            }
        }
    }
    DeviceBuffer first;
    first.Snapshot(data.dc);
    data.dc.Upload(data.c); // replay from ORIGINAL C, never from beta-updated C
    ASSERT_EQ(Launch(handle, p, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream), "custom risk determinism");
    ExpectEqualDevices(data.dc, first);
    if (p.trans == ACLBLAS_OP_C) {
        Param transpose = p;
        transpose.trans = ACLBLAS_OP_T;
        data.dc.Upload(data.c);
        ASSERT_EQ(Launch(handle, transpose, data), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(stream), "custom risk T=C");
        ExpectEqualDevices(data.dc, first);
    }
    ExpectUnchanged(data.da, data.a, "custom risk A including padding/guard");
    ExpectUnchanged(data.db, data.b, "custom risk B including padding/guard");
}
} // namespace csyrkx_test

namespace csyrkx_test {
constexpr size_t RISK_ALIGNMENT = 512;
constexpr uint32_t RISK_SLOT_WORDS = 8;

struct RiskWorkspace {
    const std::vector<aclblasComplex> initial;
    DeviceBuffer buffer;
    uint8_t* borrowed;
    size_t padding;
    aclblasHandle_t handle;
    aclrtStream stream;

    RiskWorkspace(aclblasHandle_t h, aclrtStream s, size_t elements)
        : initial(elements, aclblasComplex{-77.0f, 99.0f}), buffer(initial),
          borrowed(reinterpret_cast<uint8_t*>(buffer.Data()) + sizeof(aclblasComplex)),
          padding((RISK_ALIGNMENT - reinterpret_cast<uintptr_t>(borrowed) % RISK_ALIGNMENT) % RISK_ALIGNMENT),
          handle(h), stream(s)
    {}

    // Restore the suite's shared handle before the borrowed buffer is freed.
    ~RiskWorkspace() { aclblasSetStream(handle, stream); }

    size_t Capacity() const { return (initial.size() - 2) * sizeof(aclblasComplex); }
};

Param BuildRiskParam(int k)
{
    Param p;
    p.n = 65;
    p.k = k;
    p.lda = p.ldb = p.ldc = p.n;
    p.trans = ACLBLAS_OP_N;
    p.uplo = ACLBLAS_UPPER;
    p.alpha = {1.0f, 0.0f};
    p.beta = {0.0f, 0.0f};
    p.aFill = p.bFill = p.cFill = "VALUE_NORM_0";
    return p;
}

std::vector<uint32_t> ReadRiskWords(const RiskWorkspace& workspace, size_t riskOffset, size_t words)
{
    std::vector<uint32_t> flags(words);
    const size_t bytes = words * sizeof(uint32_t);
    CheckAcl(
        aclrtMemcpy(
            flags.data(), bytes, workspace.borrowed + workspace.padding + riskOffset, bytes,
            ACL_MEMCPY_DEVICE_TO_HOST),
        "read producer risk words");
    return flags;
}

void ExpectRiskWorkspaceGuards(const RiskWorkspace& workspace, size_t end)
{
    const auto first = workspace.buffer.Read(0, 1);
    const auto last = workspace.buffer.Read(workspace.initial.size() - 1, 1);
    EXPECT_EQ(std::memcmp(first.data(), workspace.initial.data(), sizeof(aclblasComplex)), 0);
    EXPECT_EQ(std::memcmp(last.data(), &workspace.initial.back(), sizeof(aclblasComplex)), 0);
    uint8_t guard[16];
    const size_t offset = sizeof(aclblasComplex) + end;
    ASSERT_LE(offset + sizeof(guard), workspace.initial.size() * sizeof(aclblasComplex));
    CheckAcl(
        aclrtMemcpy(guard, sizeof(guard), workspace.borrowed + end, sizeof(guard), ACL_MEMCPY_DEVICE_TO_HOST),
        "read workspace end guard");
    EXPECT_EQ(
        std::memcmp(guard, reinterpret_cast<const uint8_t*>(workspace.initial.data()) + offset, sizeof(guard)), 0);
}

void ExpectShortRiskWorkspace(const RiskWorkspace& workspace, size_t needed)
{
    ASSERT_EQ(aclblasSetWorkspace(workspace.handle, workspace.borrowed, needed - 1), ACLBLAS_STATUS_SUCCESS);
    const Param p = BuildRiskParam(1);
    CaseData data(p);
    ASSERT_EQ(Launch(workspace.handle, p, data), ACLBLAS_STATUS_ALLOC_FAILED);
    CheckAcl(aclrtSynchronizeStream(workspace.stream), "short workspace synchronize");
    ExpectUnchanged(data.dc, data.c, "short workspace C");
    ExpectUnchanged(workspace.buffer, workspace.initial, "short workspace");
}

void RunInputRiskSample(const RiskWorkspace& workspace, size_t riskOffset, int index, float sample)
{
    Param p = BuildRiskParam(1);
    // Risky input separates alpha*(A*B) from A*(alpha*B) by one FP32 ULP.
    const float alpha = index == 2 ? 1.40256929397583f : 1.0f;
    const float b = index == 2 ? 1.4233397245407104f : 1.0f;
    p.alpha = {alpha, 0.0f};
    p.caseName = "input_risk_threshold_" + std::to_string(index);
    SCOPED_TRACE(p.caseName);
    CaseData data(p);
    for (int row = 0; row < p.n; ++row) {
        data.a[row] = {sample, 0.0f};
        data.b[row] = {b, 0.0f};
    }
    if (index == 2) {
        volatile float scaledB = alpha * b;
        volatile float product = sample * b;
        volatile float ordered = sample * scaledB;
        volatile float reordered = alpha * product;
        ASSERT_NE(ordered, reordered);
    }
    ASSERT_NO_FATAL_FAILURE(ExpectCustomRiskCase(workspace.handle, workspace.stream, p, data));
    const auto flag = ReadRiskWords(workspace, riskOffset, 1);
    EXPECT_EQ(flag[0] != 0, index == 2);
    ASSERT_NO_FATAL_FAILURE(ExpectRiskWorkspaceGuards(workspace, workspace.padding + riskOffset + RISK_ALIGNMENT));
}

void RunInputRiskNine(const RiskWorkspace& workspace, float sample)
{
    // Only the second producer is risky; all three slots have 32-byte ownership.
    ASSERT_EQ(
        aclblasSetWorkspace(workspace.handle, workspace.borrowed, workspace.Capacity()), ACLBLAS_STATUS_SUCCESS);
    Param p = BuildRiskParam(9);
    p.alpha = {1.40256929397583f, 0.0f};
    p.caseName = "single_risky_producer";
    CaseData data(p);
    data.a[500] = {sample, 0.0f};
    for (int row = 0; row < p.n; ++row)
        data.b[7 * p.ldb + row] = {1.4233397245407104f, 0.0f};
    ASSERT_NO_FATAL_FAILURE(ExpectCustomRiskCase(workspace.handle, workspace.stream, p, data));
    constexpr size_t multiRiskOffset = 4 * 2560 + 4 * 80 * 80 * sizeof(float);
    const auto flags = ReadRiskWords(workspace, multiRiskOffset, 3 * RISK_SLOT_WORDS);
    for (size_t i = 0; i < 3; ++i)
        for (size_t word = 0; word < RISK_SLOT_WORDS; ++word)
            EXPECT_EQ(flags[i * RISK_SLOT_WORDS + word], i == 1 && word == 0 ? 1U : 0U);
}

void RunInputRiskThreshold(aclblasHandle_t handle, aclrtStream stream, bool runOne, bool runNine)
{
    // n=65, k=1: four aligned inputs, four 80x80 products, one risk slot.
    constexpr size_t riskOffset = 4 * RISK_ALIGNMENT + 4 * 80 * 80 * sizeof(float);
    constexpr size_t workspaceElements = 16384;
    RiskWorkspace workspace(handle, stream, workspaceElements);
    const size_t needed = workspace.padding + riskOffset + RISK_ALIGNMENT;
    ASSERT_LE(needed + 3 * sizeof(aclblasComplex), workspaceElements * sizeof(aclblasComplex));
    const float threshold = std::ldexp(1.0f, 20);
    const float samples[] = {
        std::nextafter(threshold, 0.0f), threshold, std::nextafter(threshold, std::numeric_limits<float>::infinity())};
    if (runOne) {
        ASSERT_NO_FATAL_FAILURE(ExpectShortRiskWorkspace(workspace, needed));
        ASSERT_EQ(aclblasSetWorkspace(handle, workspace.borrowed, needed), ACLBLAS_STATUS_SUCCESS);
        for (int index = 0; index < 3; ++index)
            ASSERT_NO_FATAL_FAILURE(RunInputRiskSample(workspace, riskOffset, index, samples[index]));
    }
    if (runNine)
        ASSERT_NO_FATAL_FAILURE(RunInputRiskNine(workspace, samples[2]));
}
} // namespace csyrkx_test

TEST_F(CsyrkxIndependentTest, InputRiskThresholdBoundary)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunInputRiskThreshold(handle_, stream_, true, true));
}

TEST_F(CsyrkxIndependentTest, InputRiskThresholdK1)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunInputRiskThreshold(handle_, stream_, true, false));
}

TEST_F(CsyrkxIndependentTest, InputRiskThresholdK9)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunInputRiskThreshold(handle_, stream_, false, true));
}

namespace csyrkx_test {
void RunNonfiniteComplexTerms(aclblasHandle_t handle, aclrtStream stream, int n)
{
    const float huge = std::ldexp(1.0f, 127);
    // rr/ii/ri/ir each cancel to zero across K, but each complex term's
    // imaginary part overflows to +/-Inf BEFORE cancellation in real Netlib.
    // Cover tiny direct, ordered, and generic Cube shapes without path overrides.
    // n257 also publishes adjacent safe/unsafe flags from multiple split cores.
    for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            Param p;
            p.n = n;
            p.k = 2;
            p.trans = trans;
            p.uplo = uplo;
            const bool normal = trans == ACLBLAS_OP_N;
            p.lda = (normal ? n : p.k) + 3;
            p.ldb = (normal ? n : p.k) + 5;
            p.ldc = n + 7;
            p.alpha = {1.0f, 0.0f};
            p.beta = {0.0f, 0.0f};
            p.aFill = p.bFill = "VALUE_NORM_0";
            p.cFill = "VALUE_NORM_NAN";
            p.caseName = "nonfinite_complex_terms_n" + std::to_string(n);
            SCOPED_TRACE(
                p.caseName + " trans=" + std::to_string(static_cast<int>(trans)) +
                " uplo=" + std::to_string(static_cast<int>(uplo)));
            CaseData data(p);
            for (int logical = 0; logical < n; ++logical)
                for (int l = 0; l < p.k; ++l) {
                    const size_t ap =
                        normal ? static_cast<size_t>(l) * p.lda + logical : static_cast<size_t>(logical) * p.lda + l;
                    const size_t bp =
                        normal ? static_cast<size_t>(l) * p.ldb + logical : static_cast<size_t>(logical) * p.ldb + l;
                    const float value = l ? -huge : huge;
                    data.a[ap] = {1.0f, 1.0f};
                    data.b[bp] = {value, value};
                }
            const auto golden = GoldenColumns(p, data.a, data.b, data.c, 0, n);
            for (int col = 0; col < n; ++col)
                for (int row = 0; row < n; ++row) {
                    if (!p.Selected(row, col))
                        continue;
                    ASSERT_FALSE(std::isfinite(Component(golden[static_cast<size_t>(col) * p.ldc + row], 1)))
                        << "fixture must expose nonfinite ordered complex terms";
                }
            ASSERT_NO_FATAL_FAILURE(ExpectCustomRiskCase(handle, stream, p, data));
        }
}
} // namespace csyrkx_test

TEST_F(CsyrkxIndependentTest, NonfiniteComplexTermsDespiteFiniteRealPlanes)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    for (int n : {1, 33, 65, 257})
        ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunNonfiniteComplexTerms(handle_, stream_, n));
}

TEST_F(CsyrkxIndependentTest, NonfiniteComplexTermsN1)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunNonfiniteComplexTerms(handle_, stream_, 1));
}

TEST_F(CsyrkxIndependentTest, NonfiniteComplexTermsN33)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunNonfiniteComplexTerms(handle_, stream_, 33));
}

TEST_F(CsyrkxIndependentTest, NonfiniteComplexTermsN65)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunNonfiniteComplexTerms(handle_, stream_, 65));
}

TEST_F(CsyrkxIndependentTest, NonfiniteComplexTermsN257)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    ASSERT_NO_FATAL_FAILURE(csyrkx_test::RunNonfiniteComplexTerms(handle_, stream_, 257));
}

TEST_F(CsyrkxIndependentTest, AlphaBeforeBOverflowDespiteFiniteFinalScale)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    using namespace csyrkx_test;
    // N-mode Netlib scales B before A*B: 2^30 * 2^100 overflows,
    // although A*B followed by alpha would produce the finite value 2^30.
    for (int n : {1, 33, 65})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            Param p;
            p.n = n;
            p.k = 1;
            p.trans = ACLBLAS_OP_N;
            p.uplo = uplo;
            p.lda = n + 3;
            p.ldb = n + 5;
            p.ldc = n + 7;
            p.alpha = {std::ldexp(1.0f, 30), 0.0f};
            p.beta = {0.0f, 0.0f};
            p.aFill = p.bFill = "VALUE_NORM_0";
            p.cFill = "VALUE_NORM_NAN";
            p.caseName = "alpha_before_b_overflow_n" + std::to_string(n);
            SCOPED_TRACE(p.caseName + " uplo=" + std::to_string(static_cast<int>(uplo)));
            CaseData data(p);
            for (int row = 0; row < n; ++row) {
                data.a[row] = {std::ldexp(1.0f, -100), 0.0f};
                data.b[row] = {std::ldexp(1.0f, 100), 0.0f};
            }
            const auto golden = GoldenColumns(p, data.a, data.b, data.c, 0, n);
            for (int col = 0; col < n; ++col)
                for (int row = 0; row < n; ++row) {
                    if (!p.Selected(row, col))
                        continue;
                    const auto& z = golden[static_cast<size_t>(col) * p.ldc + row];
                    ASSERT_TRUE(std::isinf(Component(z, 0)) && !std::signbit(Component(z, 0)));
                    ASSERT_TRUE(std::isnan(Component(z, 1)));
                }
            ASSERT_NO_FATAL_FAILURE(ExpectCustomRiskCase(handle_, stream_, p, data));
        }
}

namespace csyrkx_test {
void FillMixedOriginalC(const Param& p, CaseData& data, int n, bool normal)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int logical = 0; logical < n; ++logical)
        data.a[normal ? logical : static_cast<size_t>(logical) * p.lda] = {1.0f, 0.0f};
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row) {
            auto& z = data.c[static_cast<size_t>(col) * p.ldc + row];
            switch ((row + col) % 8) {
                case 0:
                    z = {inf, 0.0f};
                    break;
                case 1:
                    z = {nan, 0.0f};
                    break;
                case 2:
                    z = {std::ldexp(1.0f, 110), -std::ldexp(1.0f, 109)};
                    break;
                default:
                    z = {4.0f + 0.25f * (row % 3), 2.0f + 0.5f * (col % 5)};
                    break;
            }
        }
}

void VerifyMixedOriginalCGolden(const Param& p, const CaseData& data, int n, bool normal, bool betaOne, bool& completed)
{
    size_t ordinary = 0;
    size_t unsafe = 0;
    size_t largeFinite = 0;
    for (int col : RiskProbeColumns(n)) {
        const auto golden = GoldenColumns(p, data.a, data.b, data.c, col, 1);
        for (int row = 0; row < n; ++row) {
            if (!p.Selected(row, col))
                continue;
            const auto& original = data.c[static_cast<size_t>(col) * p.ldc + row];
            const auto& z = golden[row];
            const int kind = (row + col) % 8;
            if (normal && betaOne) {
                ASSERT_EQ(std::memcmp(&z, &original, sizeof(aclblasComplex)), 0)
                    << "N-mode zero B and beta1 must preserve original C bits";
            }
            if (kind < 2) {
                ++unsafe;
                ASSERT_FALSE(std::isfinite(Component(z, 0)));
            } else {
                if (kind == 2)
                    ++largeFinite;
                else
                    ++ordinary;
                const float re = Component(original, 0);
                const float im = Component(original, 1);
                ASSERT_TRUE(std::isfinite(Component(z, 0)) && std::isfinite(Component(z, 1)));
                ASSERT_EQ(Component(z, 0), betaOne ? re : 0.25f * re - 0.5f * im);
                ASSERT_EQ(Component(z, 1), betaOne ? im : 0.25f * im + 0.5f * re);
            }
        }
    }
    ASSERT_GT(ordinary, 0U);
    ASSERT_GT(unsafe, 0U);
    ASSERT_GT(largeFinite, 0U);
    completed = true;
}

void RunMixedOriginalC(
    aclblasHandle_t handle, aclrtStream stream, int n, aclblasOperation_t trans, aclblasFillMode_t uplo, bool betaOne,
    bool& completed)
{
    Param p;
    p.n = n;
    p.k = 1;
    p.trans = trans;
    p.uplo = uplo;
    const bool normal = trans == ACLBLAS_OP_N;
    p.lda = (normal ? n : p.k) + 3;
    p.ldb = (normal ? n : p.k) + 5;
    p.ldc = n + 7;
    p.alpha = {1.0f, 0.0f};
    p.beta = betaOne ? aclblasComplex{1.0f, 0.0f} : aclblasComplex{0.25f, 0.5f};
    p.aFill = p.bFill = p.cFill = "VALUE_NORM_0";
    p.caseName = "mixed_original_c_n" + std::to_string(n) + (betaOne ? "_beta1" : "_beta_complex");
    SCOPED_TRACE(
        p.caseName + " trans=" + std::to_string(static_cast<int>(trans)) +
        " uplo=" + std::to_string(static_cast<int>(uplo)));
    CaseData data(p);
    FillMixedOriginalC(p, data, n, normal);
    bool goldenCompleted = false;
    VerifyMixedOriginalCGolden(p, data, n, normal, betaOne, goldenCompleted);
    if (!goldenCompleted)
        return;
    ASSERT_NO_FATAL_FAILURE(ExpectCustomRiskCase(handle, stream, p, data));
    completed = true;
}
} // namespace csyrkx_test

TEST_F(CsyrkxIndependentTest, MixedOriginalCRiskAndSingleBetaUpdate)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    // B=0 makes N-mode Netlib skip every product. With beta=1 it preserves
    // original C, including (Inf,0), rather than evaluating generic beta*C.
    // Unsafe and ordinary C coexist inside each selected repair chunk; the
    // n=2051 cases cross 2048 lanes and the n=65 cases cross a short tail.
    for (int n : {65, 2051})
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
                for (bool betaOne : {true, false}) {
                    bool completed = false;
                    csyrkx_test::RunMixedOriginalC(handle_, stream_, n, trans, uplo, betaOne, completed);
                    if (!completed)
                        return;
                }
}

TEST_F(CsyrkxIndependentTest, MixedOriginalCRiskSmallBetaOne)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    bool completed = false;
    csyrkx_test::RunMixedOriginalC(handle_, stream_, 65, ACLBLAS_OP_N, ACLBLAS_UPPER, true, completed);
    ASSERT_TRUE(completed);
}

TEST_F(CsyrkxIndependentTest, MixedOriginalCRiskLargeComplexBeta)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    bool completed = false;
    csyrkx_test::RunMixedOriginalC(handle_, stream_, 2051, ACLBLAS_OP_T, ACLBLAS_LOWER, false, completed);
    ASSERT_TRUE(completed);
}

TEST(CsyrkxAcceptanceRules, TrimmedGoldenMatchesDenseNetlibBitwise)
{
    using namespace csyrkx_test;
    // No ACL initialization: an independent full dense Netlib call verifies
    // every retained tile value and every original padding/triangle bit.
    for (int n : {1, 7, 17, 67})
        for (int k : {0, 4, 13})
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
                for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
                    for (int betaKind = 0; betaKind < 3; ++betaKind) {
                        Param p;
                        p.n = n;
                        p.k = k;
                        p.trans = trans;
                        p.uplo = uplo;
                        p.lda = (trans == ACLBLAS_OP_N ? n : k) + 3;
                        p.ldb = (trans == ACLBLAS_OP_N ? n : k) + 5;
                        p.ldc = n + 7;
                        p.beta = betaKind == 0 ? aclblasComplex{0.0f, 0.0f} :
                                 betaKind == 1 ? aclblasComplex{1.0f, 0.0f} :
                                                 aclblasComplex{0.25f, -0.75f};
                        const bool normal = trans == ACLBLAS_OP_N;
                        const auto a = Matrix(normal ? n : k, normal ? k : n, p.lda, p.aFill, 701);
                        const auto b = Matrix(normal ? n : k, normal ? k : n, p.ldb, p.bFill, 907);
                        const auto original = Matrix(n, n, p.ldc, p.cFill, 1031);
                        auto dense = original;
                        cblas_cgemm(
                            CblasColMajor, normal ? CblasNoTrans : CblasTrans, normal ? CblasTrans : CblasNoTrans, n, n,
                            k, &p.alpha, a.data(), p.lda, b.data(), p.ldb, &p.beta, dense.data(), p.ldc);
                        for (int col = 0; col < n; ++col)
                            for (int row = 0; row < n; ++row)
                                if (!p.Selected(row, col)) {
                                    const size_t offset = static_cast<size_t>(col) * p.ldc + row;
                                    dense[offset] = original[offset];
                                }
                        for (int first = 0; first < n; first += kColumnChunk) {
                            const auto tile =
                                GoldenColumns(p, a, b, original, first, std::min(kColumnChunk, n - first));
                            ASSERT_EQ(
                                std::memcmp(
                                    tile.data(), dense.data() + static_cast<size_t>(first) * p.ldc,
                                    tile.size() * sizeof(aclblasComplex)),
                                0)
                                << "n=" << n << " k=" << k << " trans=" << static_cast<int>(trans)
                                << " uplo=" << static_cast<int>(uplo) << " betaKind=" << betaKind << " first=" << first;
                        }
                    }
}

TEST(CsyrkxAcceptanceRules, MaxErrorPointDefinesHardLimit)
{
    using namespace csyrkx_test;
    Metrics unrelatedLargeGolden;
    for (int i = 0; i < 100; ++i)
        unrelatedLargeGolden.Add(1.0e20f, 1.0e20f, i, 0, 0);
    unrelatedLargeGolden.Add(0.02f, 0.0f, 123, 7, 1);
    EXPECT_GE(static_cast<double>(unrelatedLargeGolden.matched) / unrelatedLargeGolden.finite, 0.99);
    EXPECT_FALSE(unrelatedLargeGolden.Passed()); // unrelated huge golden cannot inflate the cap
    EXPECT_EQ(unrelatedLargeGolden.maxPointGolden, 0.0f);
    EXPECT_EQ(unrelatedLargeGolden.maxPointRow, 123);
    EXPECT_EQ(unrelatedLargeGolden.maxPointCol, 7);
    EXPECT_EQ(unrelatedLargeGolden.maxPointComponent, 1);
    EXPECT_DOUBLE_EQ(unrelatedLargeGolden.MaxAbsLimit(), 0.01);

    Metrics official;
    for (int i = 0; i < 99; ++i)
        official.Add(1.0f, 1.0f, i, 0, 0);
    official.Add(100000.125f, 100000.0f, 9, 11, 1); // largest error=.125, ULP=.0078125, cap=.25
    official.Add(0.02f, 0.0f, 0, 0, 0); // stricter per-point rule fails; official global max-point rule passes
    EXPECT_EQ(official.hardFailures, 1U);
    EXPECT_TRUE(official.Passed());
    EXPECT_DOUBLE_EQ(official.maxAbs, 0.125);
    EXPECT_DOUBLE_EQ(official.MaxAbsLimit(), 0.25);
    EXPECT_EQ(official.maxPointGolden, 100000.0f);
    EXPECT_EQ(official.maxPointActual, 100000.125f);
    EXPECT_EQ(official.maxPointRow, 9);
    EXPECT_EQ(official.maxPointCol, 11);
    EXPECT_EQ(official.maxPointComponent, 1);
    EXPECT_NE(official.Describe().find("max_point_actual=100000.125"), std::string::npos);
    EXPECT_GT(official.maxUlp, 32.0); // remains independent diagnostic, not another gate

    // Equal maximum absolute errors conservatively choose the smaller ULP,
    // regardless of traversal order, rather than keeping an easier large-scale point.
    for (bool reverse : {false, true}) {
        Metrics tied;
        for (int i = 0; i < 100; ++i)
            tied.Add(1.0f, 1.0f, i, 0, 0);
        if (reverse)
            tied.Add(0.25f, 0.0f, 2, 3, 0);
        tied.Add(100000.25f, 100000.0f, 8, 9, 0);
        if (!reverse)
            tied.Add(0.25f, 0.0f, 2, 3, 0);
        EXPECT_FALSE(tied.Passed());
        EXPECT_EQ(tied.maxPointGolden, 0.0f);
        EXPECT_EQ(tied.maxPointRow, 2);
        EXPECT_DOUBLE_EQ(tied.MaxAbsLimit(), 0.01);
    }
}

// Host-only checks guard against a verifier that accidentally accepts bad NPU
// output; these do not initialize ACL or depend on the operator implementation.
TEST(CsyrkxAcceptanceRules, NonFiniteAndTolerance)
{
    using namespace csyrkx_test;
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(NonFiniteEquivalent(nan, nan));
    EXPECT_TRUE(NonFiniteEquivalent(nan, inf));
    EXPECT_TRUE(NonFiniteEquivalent(-inf, nan));
    EXPECT_TRUE(NonFiniteEquivalent(-inf, -inf));
    EXPECT_FALSE(NonFiniteEquivalent(inf, -inf));
    EXPECT_FALSE(NonFiniteEquivalent(0.0f, nan));
    Metrics bad;
    bad.Add(0.0f, inf, 0, 0, 0);
    EXPECT_FALSE(bad.Passed());
    Metrics finite;
    for (int i = 0; i < 99; ++i)
        finite.Add(1.0f, 1.0f, 0, 0, 0);
    finite.Add(1.001f, 1.0f, 0, 0, 0);
    EXPECT_TRUE(finite.Passed()); // exactly 99%, hard cap still satisfied
    finite.Add(1.001f, 1.0f, 0, 0, 0);
    EXPECT_FALSE(finite.Passed());
    EXPECT_TRUE(std::isfinite(Ulp(std::numeric_limits<float>::max())));
}

namespace csyrkx_test {
struct WideRiskLayout {
    uint64_t elements;
    uint32_t maxBlocks;
    uint64_t riskOffset;
    size_t complexCount;
};

WideRiskLayout BuildWideRiskLayout()
{
    constexpr uint32_t n = 65;
    constexpr uint32_t k = 64;
    const uint64_t elements = uint64_t(n) * k;
    const uint32_t maxBlocks = static_cast<uint32_t>((2 * elements + 511) / 512);
    auto align = [](uint64_t bytes) { return (bytes + RISK_ALIGNMENT - 1) / RISK_ALIGNMENT * RISK_ALIGNMENT; };
    const uint64_t paddedN = (n + 15) / 16 * 16;
    const uint64_t riskOffset = 4 * align(elements * sizeof(float)) + 4 * align(paddedN * paddedN * sizeof(float));
    const size_t bytes = static_cast<size_t>(
        riskOffset + align(uint64_t(maxBlocks) * RISK_SLOT_WORDS * sizeof(uint32_t)) + 2 * RISK_ALIGNMENT + 64);
    return {elements, maxBlocks, riskOffset, (bytes + sizeof(aclblasComplex) - 1) / sizeof(aclblasComplex)};
}

void CalibrateRiskProducers(
    const RiskWorkspace& workspace, const WideRiskLayout& layout, const Param& safe, CaseData& data, uint32_t& blocks)
{
    ASSERT_EQ(Launch(workspace.handle, safe, data), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(workspace.stream), "calibrate risk producer count");
    const auto marker = ReadRiskWords(workspace, layout.riskOffset, layout.maxBlocks * RISK_SLOT_WORDS);
    const auto* original = reinterpret_cast<const uint8_t*>(workspace.initial.data()) + sizeof(aclblasComplex) +
                           workspace.padding + layout.riskOffset;
    blocks = 0;
    bool sawSentinel = false;
    for (uint32_t i = 0; i < layout.maxBlocks; ++i) {
        const auto* current = reinterpret_cast<const uint8_t*>(marker.data() + i * RISK_SLOT_WORDS);
        bool allZero = true;
        for (uint32_t word = 0; word < RISK_SLOT_WORDS; ++word)
            allZero &= marker[i * RISK_SLOT_WORDS + word] == 0;
        if (allZero) {
            ASSERT_FALSE(sawSentinel);
            ++blocks;
        } else {
            ASSERT_EQ(
                std::memcmp(
                    current, original + i * RISK_SLOT_WORDS * sizeof(uint32_t), RISK_SLOT_WORDS * sizeof(uint32_t)),
                0);
            sawSentinel = true;
        }
    }
}

std::vector<uint32_t> AdaptiveRiskTargets(const WideRiskLayout& layout, uint32_t blocks, uint32_t& chunk)
{
    constexpr uint32_t splitChunkLimit = 4096;
    if (blocks == 0)
        throw std::runtime_error("no calibrated risk producers");
    const uint64_t perBlock = (2 * layout.elements + blocks - 1) / blocks;
    chunk = std::min<uint64_t>(splitChunkLimit, ((perBlock + 63) / 64) * 64);
    if (chunk == 0)
        throw std::runtime_error("empty adaptive risk chunk");
    const uint64_t packedChunks = (layout.elements + chunk - 1) / chunk;
    const uint32_t lastActive = std::min<uint64_t>(blocks - 1, 2 * packedChunks - 1);
    std::vector<uint32_t> targets = {0, std::min(7U, lastActive), std::min(8U, lastActive), lastActive};
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    std::cout << "ADAPTIVE_RISK blocks=" << blocks << " chunk=" << chunk << " packedChunks=" << packedChunks
              << " boundary8=" << (lastActive >= 8) << std::endl;
    return targets;
}

void ExpectAdaptiveRiskWords(const RiskWorkspace& workspace, size_t offset, uint32_t blocks, uint32_t slot)
{
    const auto flags = ReadRiskWords(workspace, offset, blocks * RISK_SLOT_WORDS);
    for (uint32_t i = 0; i < blocks; ++i)
        for (uint32_t word = 0; word < RISK_SLOT_WORDS; ++word)
            EXPECT_EQ(flags[i * RISK_SLOT_WORDS + word], i == slot && word == 0 ? 1U : 0U)
                << "slot=" << i << " word=" << word;
    const size_t riskBytes = size_t(blocks) * RISK_SLOT_WORDS * sizeof(uint32_t);
    ASSERT_NO_FATAL_FAILURE(ExpectRiskWorkspaceGuards(workspace, workspace.padding + offset + riskBytes));
}

void RunAdaptiveRiskSlot(
    const RiskWorkspace& workspace, const WideRiskLayout& layout, uint32_t blocks, uint32_t chunk, uint32_t slot)
{
    SCOPED_TRACE("risk producer slot=" + std::to_string(slot));
    Param p = BuildRiskParam(64);
    p.caseName = "adaptive_single_risk_slot_" + std::to_string(slot);
    if (chunk == 0)
        throw std::runtime_error("empty adaptive risk chunk");
    const uint64_t packedChunks = (layout.elements + chunk - 1) / chunk;
    const bool riskB = slot >= packedChunks;
    p.alpha = {riskB ? 0.9857491254806519f : 1.40256929397583f, 0.0f};
    const float other = riskB ? 0.726273775100708f : 1.4233397245407104f;
    const float threshold = std::nextafter(std::ldexp(1.0f, 20), std::numeric_limits<float>::infinity());
    const size_t index = static_cast<size_t>(riskB ? slot - packedChunks : slot) * chunk;
    ASSERT_LT(index, layout.elements);
    ASSERT_GT(p.n, 0);
    const int row = static_cast<int>(index % p.n);
    CaseData data(p);
    data.a[index] = {riskB ? other : threshold, 0.0f};
    data.b[index] = {riskB ? threshold : other, 0.0f};
    volatile float a = data.a[index].real;
    volatile float b = data.b[index].real;
    volatile float alpha = p.alpha.real;
    volatile float scaledB = alpha * b;
    volatile float product = a * b;
    volatile float ordered = a * scaledB;
    volatile float reordered = alpha * product;
    ASSERT_NE(ordered, reordered);
    ASSERT_NO_FATAL_FAILURE(ExpectCustomRiskCase(workspace.handle, workspace.stream, p, data));
    const auto actual = data.dc.Read(static_cast<size_t>(row) * p.ldc + row, 1);
    EXPECT_EQ(actual[0].real, ordered) << "exact ordered fallback";
    ASSERT_NO_FATAL_FAILURE(ExpectAdaptiveRiskWords(workspace, layout.riskOffset, blocks, slot));
}
} // namespace csyrkx_test

TEST_F(CsyrkxIndependentTest, WideRiskSlotsAdaptive)
{
    if (csyrkx_test::Mode() != "accuracy")
        GTEST_SKIP();
    using namespace csyrkx_test;
    const auto layout = BuildWideRiskLayout();
    RiskWorkspace workspace(handle_, stream_, layout.complexCount);
    const size_t probeBytes = size_t(layout.maxBlocks) * RISK_SLOT_WORDS * sizeof(uint32_t);
    ASSERT_LE(workspace.padding + layout.riskOffset + probeBytes, workspace.Capacity());
    ASSERT_EQ(aclblasSetWorkspace(handle_, workspace.borrowed, workspace.Capacity()), ACLBLAS_STATUS_SUCCESS);
    // Keep calibration inputs alive while every selected producer is tested.
    const Param safe = BuildRiskParam(64);
    CaseData safeData(safe);
    uint32_t blocks = 0;
    ASSERT_NO_FATAL_FAILURE(CalibrateRiskProducers(workspace, layout, safe, safeData, blocks));
    ASSERT_GT(blocks, 0U);
    uint32_t chunk = 0;
    const auto targets = AdaptiveRiskTargets(layout, blocks, chunk);
    for (uint32_t slot : targets)
        ASSERT_NO_FATAL_FAILURE(RunAdaptiveRiskSlot(workspace, layout, blocks, chunk, slot));
}
