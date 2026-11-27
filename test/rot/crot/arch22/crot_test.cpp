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
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"
#include "npu_event_timer.h"
#include "crot_golden.h"
#include "crot_npu_wrapper.h"

// Crot-specific fields are mapped by name after the shared loader reads each row.
struct CrotA2Param {
    std::string caseName;
    int n;
    BlasFillMode x;
    int incx;
    BlasFillMode y;
    int incy;
    float c;
    float sReal;
    float sImag;
    bool cIsNull;
    bool sIsNull;
    aclblasStatus_t expectResult;
    uint32_t randomSeed;
    std::string csPtrMode;
    int xOffset;
    int yOffset;
    std::string scalarFill;

    explicit CrotA2Param(const csv_map& row)
        : caseName(row.at("case_name")),
          n(std::stoi(row.at("n"))),
          x(parseFill(row.at("x"))),
          incx(std::stoi(row.at("incx"))),
          y(parseFill(row.at("y"))),
          incy(std::stoi(row.at("incy"))),
          c(row.at("c") == "null" ? 0.0f : std::stof(row.at("c"))),
          sReal(row.at("s_real") == "null" ? 0.0f : std::stof(row.at("s_real"))),
          sImag(row.at("s_imag") == "null" ? 0.0f : std::stof(row.at("s_imag"))),
          cIsNull(row.at("c") == "null"),
          sIsNull(row.at("s_real") == "null" || row.at("s_imag") == "null"),
          randomSeed(static_cast<uint32_t>(std::stoul(row.at("random_seed")))),
          csPtrMode(row.at("cs_ptr_mode")),
          xOffset(std::stoi(row.at("x_offset"))),
          yOffset(std::stoi(row.at("y_offset"))),
          scalarFill(ReadMap(row, "scalar_fill"))
    {
        if (row.at("expect_result") == "ACLBLAS_STATUS_SUCCESS") {
            expectResult = ACLBLAS_STATUS_SUCCESS;
        } else if (row.at("expect_result") == "ACLBLAS_STATUS_INVALID_VALUE") {
            expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        } else {
            throw std::runtime_error("Unsupported Crot expected status: " + row.at("expect_result"));
        }
    }
};

inline std::ostream& operator<<(std::ostream& out, const CrotA2Param& p) { return out << p.caseName; }

static std::vector<CrotA2Param> ReadCrotCases()
{
    auto cases = GetCasesFromCsv<CrotA2Param>(ReplaceFileExtension2Csv(__FILE__));
    if (cases.size() != 2536) {
        throw std::runtime_error("Expected 2536 Crot CSV cases, got " + std::to_string(cases.size()));
    }
    return cases;
}

class CrotTest : public BlasTest<CrotA2Param> {};

namespace {
constexpr size_t GUARD = 8; // guard on both sides; default payload remains 32B aligned
constexpr aclblasComplex SENTINEL{12345.5f, -7654.25f};

size_t PhysicalIndex(int i, int n, int inc)
{
    const int64_t magnitude = inc < 0 ? -static_cast<int64_t>(inc) : inc;
    return static_cast<size_t>(inc < 0 ? n - 1 - i : i) * magnitude;
}

struct HostVector {
    std::vector<aclblasComplex> data;
    size_t start;
    bool isNull;

    HostVector(const CrotA2Param& p, bool isX)
    {
        const int inc = isX ? p.incx : p.incy;
        const auto fill = isX ? p.x : p.y;
        const uint32_t seed = p.randomSeed + (isX ? 0U : 7919U);
        isNull = fill.method == BlasFillMode::M_NULLPTR;
        start = GUARD + static_cast<size_t>(isX ? p.xOffset : p.yOffset);
        // Error and quick-return cases must never allocate from an invalid stride.
        const bool numeric = p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0;
        const size_t span = numeric ? PhysicalIndex(0, p.n, -std::abs(inc)) + 1 : 1;
        data.assign(start + span + GUARD, SENTINEL);
        if (numeric && !isNull) {
            const auto real = makeBlasArray(p.n, fill, seed);
            const auto imag = makeBlasArray(p.n, fill, seed + 1000U);
            for (int i = 0; i < p.n; ++i) {
                data[start + PhysicalIndex(i, p.n, inc)] = {real[i], imag[i]};
            }
        }
    }

    size_t Bytes() const { return data.size() * sizeof(aclblasComplex); }
};

// Task section 3.2 permits NaN <-> Inf but never opposite signed infinities,
// or any finite/nonfinite mismatch. Nonfinite values are excluded from ratios.
bool VerifyComponent(
    const HostVector& actual, const HostVector& golden, int n, int inc, bool imag, const std::string& label)
{
    std::vector<float> finiteActual;
    std::vector<float> finiteGolden;
    size_t special = 0;
    size_t specialPassed = 0;
    for (int i = 0; i < n; ++i) {
        const size_t index = actual.start + PhysicalIndex(i, n, inc);
        const float a = imag ? actual.data[index].imag : actual.data[index].real;
        const float g = imag ? golden.data[index].imag : golden.data[index].real;
        if (std::isfinite(a) && std::isfinite(g)) {
            finiteActual.push_back(a);
            finiteGolden.push_back(g);
        } else {
            ++special;
            if (!std::isfinite(a) && !std::isfinite(g) && (std::isnan(a) || std::isnan(g) || a == g)) {
                ++specialPassed;
            }
        }
    }
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, static_cast<const float*>(nullptr), finiteActual.size());
    const bool finiteOk =
        finiteActual.empty() ||
        Verifier::verifyVector(finiteActual.data(), finiteGolden.data(), finiteActual.size(), 1, cfg, label);
    std::cout << "[" << label << "] nonfinite_passed=" << specialPassed << "/" << special << std::endl;
    return finiteOk && special == specialPassed;
}

void VerifyUntouched(const HostVector& actual, const HostVector& original, int n, int inc, bool numeric)
{
    const size_t magnitude = numeric ? static_cast<size_t>(std::abs(inc)) : 1;
    const size_t span = numeric ? static_cast<size_t>(n - 1) * magnitude + 1 : 0;
    for (size_t i = 0; i < actual.data.size(); ++i) {
        const bool active =
            numeric && i >= actual.start && i - actual.start < span && (i - actual.start) % magnitude == 0;
        if (!active) {
            ASSERT_EQ(std::memcmp(&actual.data[i], &original.data[i], sizeof(aclblasComplex)), 0)
                << "guard/gap modified at physical element " << i;
        }
    }
}

void PrepareScalars(const CrotA2Param& p, float& c, aclblasComplex& s)
{
    c = p.c;
    s = {p.sReal, p.sImag};
    if (!p.scalarFill.empty()) {
        c = makeBlasArray(1, p.scalarFill, p.randomSeed + 101U)[0];
        s.real = makeBlasArray(1, p.scalarFill, p.randomSeed + 211U)[0];
        s.imag = makeBlasArray(1, p.scalarFill, p.randomSeed + 307U)[0];
    }
}

template <typename Launch, typename Reset>
void RunPerformanceCase(const CrotA2Param& p, aclrtStream stream, const Launch& launch, const Reset& reset)
{
    if (std::getenv("CROT_PROFILE") != nullptr) {
        // msprof performs its own five warmups. Only ten real calls are
        // made here, so --launch-count=10 cannot include an extra probe call.
        for (int sample = 0; sample < 10; ++sample) {
            ASSERT_TRUE(reset());
            ASSERT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        }
        std::cout << "[PROFILE] case=" << p.caseName << " successful_samples=10" << std::endl;
        return;
    }
    std::vector<double> samples;
    ASSERT_EQ(
        blas_test::MeasureNpuEvent(
            stream, 5, 10, reset, [&]() { return launch() == ACLBLAS_STATUS_SUCCESS; }, samples),
        0);
    std::cout << "[PERF] case=" << p.caseName << " avg_us=" << blas_test::MeanUs(samples)
              << " source=aclrtEvent samples=10" << std::endl;
}

void VerifyCaseOutputs(
    aclblasHandle_t handle, const CrotA2Param& p, const HostVector& hx, const HostVector& hy,
    const HostVector& originalX, const HostVector& originalY, float c, const aclblasComplex& s)
{
    const bool numeric = p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0;
    VerifyUntouched(hx, originalX, p.n, p.incx, numeric);
    VerifyUntouched(hy, originalY, p.n, p.incy, numeric);
    if (!numeric) {
        return;
    }
    HostVector goldenX = originalX;
    HostVector goldenY = originalY;
    // The shared golden is used only for valid positive-n calculations; arch22
    // return-code expectations come directly from its own CSV and fixed tests.
    ASSERT_EQ(
        aclblasCrot_cpu(
            handle, p.n, goldenX.data.data() + goldenX.start, p.incx, goldenY.data.data() + goldenY.start, p.incy, &c,
            &s),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(VerifyComponent(hx, goldenX, p.n, p.incx, false, p.caseName + "_x_real"));
    EXPECT_TRUE(VerifyComponent(hx, goldenX, p.n, p.incx, true, p.caseName + "_x_imag"));
    EXPECT_TRUE(VerifyComponent(hy, goldenY, p.n, p.incy, false, p.caseName + "_y_real"));
    EXPECT_TRUE(VerifyComponent(hy, goldenY, p.n, p.incy, true, p.caseName + "_y_imag"));
}

void RunCase(aclblasHandle_t handle, aclrtStream stream, const CrotA2Param& p)
{
    HostVector hx(p, true);
    HostVector hy(p, false);
    const HostVector originalX = hx;
    const HostVector originalY = hy;
    float c;
    aclblasComplex s;
    PrepareScalars(p, c, s);

    CrotNpuWrapper buffers(stream);
    void* dx = nullptr;
    void* dy = nullptr;
    ASSERT_EQ(buffers.Upload(hx.data.data(), hx.Bytes(), &dx), ACL_SUCCESS);
    ASSERT_EQ(buffers.Upload(hy.data.data(), hy.Bytes(), &dy), ACL_SUCCESS);
    auto* x = hx.isNull ? nullptr : static_cast<aclblasComplex*>(dx) + hx.start;
    auto* y = hy.isNull ? nullptr : static_cast<aclblasComplex*>(dy) + hy.start;
    const float* cArg = p.cIsNull ? nullptr : &c;
    const aclblasComplex* sArg = p.sIsNull ? nullptr : &s;
    if (!p.cIsNull && (p.csPtrMode == "device" || p.csPtrMode == "mixed_c_device")) {
        void* dc = nullptr;
        ASSERT_EQ(buffers.Upload(&c, sizeof(c), &dc), ACL_SUCCESS);
        cArg = static_cast<const float*>(dc);
    }
    if (!p.sIsNull && (p.csPtrMode == "device" || p.csPtrMode == "mixed")) {
        void* ds = nullptr;
        ASSERT_EQ(buffers.Upload(&s, sizeof(s), &ds), ACL_SUCCESS);
        sArg = static_cast<const aclblasComplex*>(ds);
    }
    const auto launch = [&]() { return aclblasCrot(handle, p.n, x, p.incx, y, p.incy, cArg, sArg); };
    const bool performance = p.caseName.rfind("TC_PF_", 0) == 0;
    if (performance) {
        const auto reset = [&]() {
            return aclrtMemcpy(dx, hx.Bytes(), hx.data.data(), hx.Bytes(), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                   aclrtMemcpy(dy, hy.Bytes(), hy.data.data(), hy.Bytes(), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
        };
        ASSERT_NO_FATAL_FAILURE(RunPerformanceCase(p, stream, launch, reset));
        if (std::getenv("CROT_PROFILE") != nullptr) {
            return;
        }
    } else {
        ASSERT_EQ(launch(), p.expectResult);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(hx.data.data(), hx.Bytes(), dx, hx.Bytes(), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(hy.data.data(), hy.Bytes(), dy, hy.Bytes(), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    VerifyCaseOutputs(handle, p, hx, hy, originalX, originalY, c, s);
}
} // namespace

TEST_P(CrotTest, CsvDriven) { RunCase(handle_, stream_, GetParam()); }

TEST_F(CrotTest, ValidationPrecedence)
{
    for (int n : {-1, 0, 1}) {
        EXPECT_EQ(
            aclblasCrot(nullptr, n, nullptr, 0, nullptr, INT32_MIN, nullptr, nullptr),
            ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    EXPECT_EQ(aclblasCrot(handle_, -1, nullptr, 0, nullptr, INT32_MIN, nullptr, nullptr), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasCrot(handle_, 0, nullptr, 0, nullptr, INT32_MIN, nullptr, nullptr), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CrotTest, DeviceScalarsFollowStream)
{
    CrotNpuWrapper buffers(stream_);
    std::vector<aclblasComplex> hx(65, {2.0f, -3.0f});
    std::vector<aclblasComplex> hy(65, {4.0f, 5.0f});
    const size_t bytes = hx.size() * sizeof(aclblasComplex);
    float c = 1.0f;
    aclblasComplex s{0.5f, -0.25f};
    void* dx = nullptr;
    void* dy = nullptr;
    void* dc = nullptr;
    void* ds = nullptr;
    ASSERT_EQ(buffers.Upload(hx.data(), bytes, &dx), ACL_SUCCESS);
    ASSERT_EQ(buffers.Upload(hy.data(), bytes, &dy), ACL_SUCCESS);
    ASSERT_EQ(buffers.Upload(&c, sizeof(c), &dc), ACL_SUCCESS);
    ASSERT_EQ(buffers.Upload(&s, sizeof(s), &ds), ACL_SUCCESS);
    // Queued updates, without synchronizing before the operator call.
    ASSERT_EQ(aclrtMemsetAsync(dc, sizeof(c), 0, sizeof(c), stream_), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemsetAsync(ds, sizeof(s), 0, sizeof(s), stream_), ACL_SUCCESS);
    ASSERT_EQ(
        aclblasCrot(
            handle_, 65, static_cast<aclblasComplex*>(dx), 1, static_cast<aclblasComplex*>(dy), 1,
            static_cast<float*>(dc), static_cast<aclblasComplex*>(ds)),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(hx.data(), bytes, dx, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(hy.data(), bytes, dy, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < hx.size(); ++i) {
        EXPECT_EQ(hx[i].real, 0.0f);
        EXPECT_EQ(hx[i].imag, 0.0f);
        EXPECT_EQ(hy[i].real, 0.0f);
        EXPECT_EQ(hy[i].imag, 0.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    CrotA2, CrotTest, ::testing::ValuesIn(ReadCrotCases()),
    [](const ::testing::TestParamInfo<CrotA2Param>& info) { return info.param.caseName; });
