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
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "crot_param.h"
#include "crot_golden.h"
#include "crot_npu_wrapper.h"

class CrotTest : public BlasTest<CrotParam> {};

// ---------------------------------------------------------------- fixed tests

TEST_F(CrotTest, NullHandle)
{
    aclblasComplex x[5] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
    aclblasComplex y[5] = {{6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}};
    const float c = 0.6f;
    const aclblasComplex s{0.8f, 0.1f};
    aclblasStatus_t ret = aclblasCrot(nullptr, 5, x, 1, y, 1, &c, &s);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CrotTest, NullHandleTakesPrecedenceOverQuickReturn)
{
    EXPECT_EQ(aclblasCrot(nullptr, 0, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// arch22 contract (issue #827): n<0 and zero / INT32_MIN strides are rejected, unlike arch35
// which accepts them. These expectations are the declared per-arch difference.
TEST_F(CrotTest, NegativeNRejected)
{
    EXPECT_EQ(
        aclblasCrot(CrotTest::handle_, -1, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CrotTest, ZeroStrideRejected)
{
    aclblasComplex x[8] = {};
    aclblasComplex y[8] = {};
    const float c = 0.6f;
    const aclblasComplex s{0.8f, 0.1f};
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, 8, x, 0, y, 1, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, 8, x, 1, y, 0, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, 8, x, 0, y, 0, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
}

// The operator rejects a strided offset that does not fit in int32 with
// ACLBLAS_STATUS_INVALID_VALUE. Called directly with tiny device buffers so the guard
// itself is exercised without the huge allocation the wrapper would attempt.
TEST_F(CrotTest, StridedOffsetOverflowRejected)
{
    constexpr int n = 3;
    void* dx = nullptr;
    void* dy = nullptr;
    ASSERT_EQ(aclrtMalloc(&dx, n * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dy, n * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    const float c = 0.6f;
    const aclblasComplex s{0.8f, 0.1f};
    auto* px = static_cast<aclblasComplex*>(dx);
    auto* py = static_cast<aclblasComplex*>(dy);

    EXPECT_EQ(aclblasCrot(CrotTest::handle_, n, px, INT32_MIN, py, 1, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, n, px, 1, py, INT32_MIN, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, n, px, INT32_MAX, py, 1, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, n, px, 1, py, INT32_MAX, &c, &s), ACLBLAS_STATUS_INVALID_VALUE);
    // Legal boundary: n=1 keeps the access span at a single element.
    EXPECT_EQ(aclblasCrot(CrotTest::handle_, 1, px, INT32_MAX, py, 1, &c, &s), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(CrotTest::stream_), ACL_SUCCESS);

    aclrtFree(dx);
    aclrtFree(dy);
}

// ------------------------------------------------------------- CSV data setup

static constexpr uint32_t kYSeedOffset = 7919U; // prime, distinct from imag offset 1000

static inline std::vector<aclblasComplex> MakeCrotBuffer(int n, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || n <= 0) {
        return {};
    }
    const int64_t absInc = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    std::vector<float> realPart;
    std::vector<float> imagPart;
    if (absInc <= 1) {
        realPart = makeBlasArray(static_cast<int64_t>(n), fill, seed);
        imagPart = makeBlasArray(static_cast<int64_t>(n), fill, seed + 1000U);
    } else {
        realPart = makeBlasStrided(n, inc, fill, seed);
        imagPart = makeBlasStrided(n, inc, fill, seed + 1000U);
    }
    const size_t sz = std::min(realPart.size(), imagPart.size());
    std::vector<aclblasComplex> data(sz);
    for (size_t i = 0; i < sz; i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

static inline int CrotAbsInc(int inc)
{
    const int64_t v = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    if (v == 0) {
        return 1;
    }
    return static_cast<int>(v);
}

// A stride that the operator must reject (zero / INT32_MIN / int32 span overflow).
// For such cases the test must not allocate the full span (which can be tens of billions
// of elements); it uses a one-element dummy buffer instead (task book section 3.5).
static inline bool CrotStrideInvalid(int n, int inc)
{
    if (inc == 0 || inc == INT32_MIN) {
        return true;
    }
    const int64_t absInc = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    return static_cast<int64_t>(n - 1) * absInc > INT32_MAX;
}

static inline bool CrotBufferHasSpecial(const std::vector<aclblasComplex>& buf, int n, int inc, bool wantNaN)
{
    if (buf.empty()) {
        return false;
    }
    const int step = CrotAbsInc(inc);
    for (int i = 0; i < n; i++) {
        const aclblasComplex& v = buf[static_cast<size_t>(i) * static_cast<size_t>(step)];
        const bool hit =
            wantNaN ? (std::isnan(v.real) || std::isnan(v.imag)) : (std::isinf(v.real) || std::isinf(v.imag));
        if (hit) {
            return true;
        }
    }
    return false;
}

static inline bool CrotAllNaNAt(const aclblasComplex* buf, int n, int inc)
{
    if (buf == nullptr) {
        return false;
    }
    const int step = CrotAbsInc(inc);
    for (int i = 0; i < n; i++) {
        const aclblasComplex& v = buf[static_cast<size_t>(i) * static_cast<size_t>(step)];
        if (!std::isnan(v.real) || !std::isnan(v.imag)) {
            return false;
        }
    }
    return true;
}

static inline void CrotExtractComponent(const aclblasComplex* buf, int n, int inc, bool imag, std::vector<float>& out)
{
    const int step = CrotAbsInc(inc);
    out.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        const aclblasComplex& v = buf[static_cast<size_t>(i) * static_cast<size_t>(step)];
        out[static_cast<size_t>(i)] = imag ? v.imag : v.real;
    }
}

struct CrotHostInputs {
    std::vector<aclblasComplex> xHost;
    std::vector<aclblasComplex> yHost;
    aclblasComplex* xPtr = nullptr;
    aclblasComplex* yPtr = nullptr;
    bool inputHasNaN = false;
    bool inputHasInf = false;
};

static inline void BuildHostInputs(const CrotParam& p, CrotHostInputs& in)
{
    in.xHost = MakeCrotBuffer(p.n, p.incx, p.x, p.randomSeed);
    in.yHost = MakeCrotBuffer(p.n, p.incy, p.y, p.randomSeed + kYSeedOffset);

    in.xPtr = in.xHost.empty() ? nullptr : in.xHost.data();
    in.yPtr = in.yHost.empty() ? nullptr : in.yHost.data();

    in.inputHasNaN =
        CrotBufferHasSpecial(in.xHost, p.n, p.incx, true) || CrotBufferHasSpecial(in.yHost, p.n, p.incy, true);
    in.inputHasInf =
        CrotBufferHasSpecial(in.xHost, p.n, p.incx, false) || CrotBufferHasSpecial(in.yHost, p.n, p.incy, false);
}

static inline aclblasStatus_t ExecuteOperator(
    aclblasHandle_t handle, aclrtStream stream, const CrotParam& p, const CrotHostInputs& in)
{
    const aclblasComplex sVal{p.sReal, p.sImag};
    return aclblasCrot_npu(
        handle, p.n, in.xPtr, p.incx, in.yPtr, p.incy, p.c, sVal, p.csPtrMode, p.cIsNull, p.sIsNull, stream);
}

static inline void VerifyTolerance(
    const CrotParam& p, const CrotHostInputs& in, const std::vector<aclblasComplex>& goldenX,
    const std::vector<aclblasComplex>& goldenY)
{
    VerifyConfig xCfg;
    applyMixedTolerance(xCfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(p.n));
    VerifyConfig yCfg;
    applyMixedTolerance(yCfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(p.n));

    std::vector<float> outXRe;
    std::vector<float> outXIm;
    std::vector<float> outYRe;
    std::vector<float> outYIm;
    std::vector<float> goldXRe;
    std::vector<float> goldXIm;
    std::vector<float> goldYRe;
    std::vector<float> goldYIm;
    CrotExtractComponent(in.xPtr, p.n, p.incx, false, outXRe);
    CrotExtractComponent(in.xPtr, p.n, p.incx, true, outXIm);
    CrotExtractComponent(in.yPtr, p.n, p.incy, false, outYRe);
    CrotExtractComponent(in.yPtr, p.n, p.incy, true, outYIm);
    CrotExtractComponent(goldenX.data(), p.n, p.incx, false, goldXRe);
    CrotExtractComponent(goldenX.data(), p.n, p.incx, true, goldXIm);
    CrotExtractComponent(goldenY.data(), p.n, p.incy, false, goldYRe);
    CrotExtractComponent(goldenY.data(), p.n, p.incy, true, goldYIm);

    EXPECT_TRUE(Verifier::verifyVector(
        outXRe.data(), goldXRe.data(), static_cast<size_t>(p.n), 1, xCfg, p.caseName + "_x_real"));
    EXPECT_TRUE(Verifier::verifyVector(
        outXIm.data(), goldXIm.data(), static_cast<size_t>(p.n), 1, xCfg, p.caseName + "_x_imag"));
    EXPECT_TRUE(Verifier::verifyVector(
        outYRe.data(), goldYRe.data(), static_cast<size_t>(p.n), 1, yCfg, p.caseName + "_y_real"));
    EXPECT_TRUE(Verifier::verifyVector(
        outYIm.data(), goldYIm.data(), static_cast<size_t>(p.n), 1, yCfg, p.caseName + "_y_imag"));
}

static inline void VerifyResult(aclblasHandle_t handle, const CrotParam& p, const CrotHostInputs& in)
{
    std::vector<aclblasComplex> goldenX = MakeCrotBuffer(p.n, p.incx, p.x, p.randomSeed);
    std::vector<aclblasComplex> goldenY = MakeCrotBuffer(p.n, p.incy, p.y, p.randomSeed + kYSeedOffset);
    const aclblasComplex sVal{p.sReal, p.sImag};
    const float* cArg = p.cIsNull ? nullptr : &p.c;
    const aclblasComplex* sArg = p.sIsNull ? nullptr : &sVal;
    aclblasCrot_cpu(handle, p.n, goldenX.data(), p.incx, goldenY.data(), p.incy, cArg, sArg);

    if (in.inputHasNaN) {
        EXPECT_TRUE(CrotAllNaNAt(in.xPtr, p.n, p.incx));
        EXPECT_TRUE(CrotAllNaNAt(in.yPtr, p.n, p.incy));
        return;
    }
    // Inf inputs are covered by the shared FP32 mixed tolerance; the golden may hold
    // NaN from 0*Inf, so a NaN<->NaN / same-sign-Inf comparison is not needed here.
    VerifyTolerance(p, in, goldenX, goldenY);
}

// Direct call for negative stride cases (zero / INT32_MIN / span overflow): a one-element
// dummy device buffer is used so the operator returns INVALID_VALUE before touching data,
// without the multi-billion-element allocation the normal path would attempt.
static inline void CheckInvalidStrideCase(const CrotParam& p, aclblasHandle_t handle)
{
    void* dx = nullptr;
    void* dy = nullptr;
    ASSERT_EQ(aclrtMalloc(&dx, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dy, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    const float c = p.cIsNull ? 0.0f : p.c;
    const aclblasComplex sVal{p.sReal, p.sImag};
    const float* cArg = p.cIsNull ? nullptr : &c;
    const aclblasComplex* sArg = p.sIsNull ? nullptr : &sVal;
    const aclblasStatus_t ret = aclblasCrot(
        handle, p.n, static_cast<aclblasComplex*>(dx), p.incx, static_cast<aclblasComplex*>(dy), p.incy, cArg, sArg);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    aclrtFree(dx);
    aclrtFree(dy);
}

// ------------------------------------------------------------- performance

static inline void RunCrotPerfCase(const CrotParam& p, aclblasHandle_t handle, aclrtStream stream)
{
    const int n = p.n;
    const size_t count = static_cast<size_t>(n);
    const size_t bytes = count * sizeof(aclblasComplex);

    std::vector<aclblasComplex> xHost(count);
    std::vector<aclblasComplex> yHost(count);
    for (size_t i = 0; i < count; i++) {
        xHost[i] = aclblasComplex{static_cast<float>(i % 17U) * 0.5f, static_cast<float>(i % 23U) * 0.25f};
        yHost[i] = aclblasComplex{static_cast<float>(i % 29U) * 0.5f, static_cast<float>(i % 31U) * 0.25f};
    }

    void* dx = nullptr;
    void* dy = nullptr;
    ASSERT_EQ(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dx, bytes, xHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dy, bytes, yHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    const float c = p.c;
    const aclblasComplex s{p.sReal, p.sImag};
    auto* px = static_cast<aclblasComplex*>(dx);
    auto* py = static_cast<aclblasComplex*>(dy);

    constexpr int kWarmup = 5;
    constexpr int kBatch = 10;
    for (int i = 0; i < kWarmup; i++) {
        ASSERT_EQ(aclblasCrot(handle, n, px, 1, py, 1, &c, &s), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    aclrtEvent startEvent = nullptr;
    aclrtEvent stopEvent = nullptr;
    ASSERT_EQ(aclrtCreateEvent(&startEvent), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&stopEvent), ACL_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(startEvent, stream), ACL_SUCCESS);
    for (int i = 0; i < kBatch; i++) {
        ASSERT_EQ(aclblasCrot(handle, n, px, 1, py, 1, &c, &s), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtRecordEvent(stopEvent, stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    float elapsedMs = 0.0f;
    ASSERT_EQ(aclrtEventElapsedTime(&elapsedMs, startEvent, stopEvent), ACL_SUCCESS);
    const double avgUs = static_cast<double>(elapsedMs) * 1000.0 / kBatch;
    std::cout << "[PERF] case=" << p.caseName << " avg_us=" << avgUs << std::endl;

    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(stopEvent);
    aclrtFree(dx);
    aclrtFree(dy);
}

// ------------------------------------------------------------- CSV driven

TEST_P(CrotTest, CsvDriven)
{
    const auto& p = GetParam();

    // Performance cases: timed and reported, not compared against the CPU golden.
    if (p.caseName.rfind("TC_PF_", 0) == 0) {
        RunCrotPerfCase(p, CrotTest::handle_, CrotTest::stream_);
        return;
    }

    // Negative stride cases use a dummy buffer (the operator rejects the stride before
    // reading data); the normal wrapper path would attempt a huge allocation.
    if (CrotStrideInvalid(p.n, p.incx) || CrotStrideInvalid(p.n, p.incy)) {
        CheckInvalidStrideCase(p, CrotTest::handle_);
        return;
    }

    CrotHostInputs inputs;
    BuildHostInputs(p, inputs);

    aclblasStatus_t ret = ExecuteOperator(CrotTest::handle_, CrotTest::stream_, p, inputs);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (p.n <= 0 || inputs.xPtr == nullptr || inputs.yPtr == nullptr) {
        return;
    }
    VerifyResult(CrotTest::handle_, p, inputs);
}

static std::vector<CrotParam> LoadAndValidateCrotCases(const std::string& csvPath)
{
    constexpr size_t kExpectedTotal = 1200;
    constexpr size_t kExpectedPerfCases = 200;
    std::vector<CrotParam> cases = GetCasesFromCsv<CrotParam>(csvPath);
    if (cases.size() != kExpectedTotal) {
        throw std::runtime_error(
            "aclblasCrot CSV case count mismatch, expected " + std::to_string(kExpectedTotal) + " but got " +
            std::to_string(cases.size()));
    }
    std::unordered_set<std::string> names;
    size_t perfCases = 0;
    for (const auto& item : cases) {
        if (!names.insert(item.caseName).second) {
            throw std::runtime_error("aclblasCrot CSV has a duplicate case_name " + item.caseName);
        }
        if (item.caseName.rfind("TC_PF_", 0) == 0) {
            ++perfCases;
        }
    }
    if (perfCases != kExpectedPerfCases) {
        throw std::runtime_error(
            "aclblasCrot CSV performance case count mismatch, expected " + std::to_string(kExpectedPerfCases) +
            " but got " + std::to_string(perfCases));
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Crot, CrotTest, ::testing::ValuesIn(LoadAndValidateCrotCases(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CrotParam>);
