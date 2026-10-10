/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "blas_test.h"
#include "cdotc_golden.h"
#include "cdotc_npu_wrapper.h"
#include "cdotc_param.h"
#include "cdotc_test_utils.h"
#include "csv_loader.h"
#include "verify.h"

namespace {

constexpr int PERF_WARMUP_COUNT = 10;
constexpr int PERF_SAMPLE_COUNT = 100;

class TimelineEvent {
public:
    TimelineEvent() = default;
    TimelineEvent(const TimelineEvent&) = delete;
    TimelineEvent& operator=(const TimelineEvent&) = delete;
    ~TimelineEvent()
    {
        if (event_ != nullptr) {
            aclrtDestroyEvent(event_);
        }
    }

    aclError Create() { return aclrtCreateEventWithFlag(&event_, ACL_EVENT_TIME_LINE); }

    aclrtEvent Get() const { return event_; }

private:
    aclrtEvent event_ = nullptr;
};

bool IsPerformanceCase(const std::string& caseName) { return caseName.rfind("TC_PF", 0) == 0; }

void RunPerformanceCase(
    aclblasHandle_t handle, aclrtStream stream, const CdotcParam& parameter, const aclblasComplex* x,
    const aclblasComplex* y)
{
    aclblasComplex result{123.0f, -456.0f};
    CdotcDeviceBuffers device;
    ASSERT_EQ(device.Init(parameter.n, x, parameter.incx, y, parameter.incy, &result), ACLBLAS_STATUS_SUCCESS);
    for (int i = 0; i < PERF_WARMUP_COUNT; ++i) {
        ASSERT_EQ(
            aclblasCdotc(handle, parameter.n, device.X(), parameter.incx, device.Y(), parameter.incy, device.Result()),
            ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    TimelineEvent startEvent;
    TimelineEvent endEvent;
    ASSERT_EQ(startEvent.Create(), ACL_SUCCESS);
    ASSERT_EQ(endEvent.Create(), ACL_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(startEvent.Get(), stream), ACL_SUCCESS);
    for (int i = 0; i < PERF_SAMPLE_COUNT; ++i) {
        ASSERT_EQ(
            aclblasCdotc(handle, parameter.n, device.X(), parameter.incx, device.Y(), parameter.incy, device.Result()),
            ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtRecordEvent(endEvent.Get(), stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    float elapsedMs = 0.0f;
    ASSERT_EQ(aclrtEventElapsedTime(&elapsedMs, startEvent.Get(), endEvent.Get()), ACL_SUCCESS);
    const double averageUs = static_cast<double>(elapsedMs) * 1000.0 / PERF_SAMPLE_COUNT;
    std::cout << std::fixed << std::setprecision(6) << "[CDOTC_PERF] case=" << parameter.caseName
              << " n=" << parameter.n << " incx=" << parameter.incx << " incy=" << parameter.incy
              << " avg_us=" << averageUs << " samples=" << PERF_SAMPLE_COUNT << std::endl;
}

} // namespace

class CdotcArch35Test : public BlasTest<CdotcParam> {};

TEST_F(CdotcArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasCdotc_npu(nullptr, nullptr, 1, nullptr, 1, nullptr, 1, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CdotcArch35Test, ZeroLengthNullResult)
{
    aclblasStatus_t ret = aclblasCdotc_npu(handle_, stream_, 0, nullptr, 1, nullptr, 1, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Cdotc, CdotcArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CdotcParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CdotcParam>);

TEST_P(CdotcArch35Test, CsvDriven)
{
    const CdotcParam& p = GetParam();
    std::vector<aclblasComplex> xHost = MakeCdotcComplexStrided(p.n, p.incx, p.x, p.distribution, p.randomSeed);
    std::vector<aclblasComplex> yHost = MakeCdotcComplexStrided(p.n, p.incy, p.y, p.distribution, p.randomSeed + 2000U);
    const aclblasComplex* x = xHost.empty() ? nullptr : xHost.data();
    const aclblasComplex* y = yHost.empty() ? nullptr : yHost.data();

    aclblasComplex result{123.0f, -456.0f};
    aclblasComplex* resultPtr = p.resultFill.method == BlasFillMode::M_NULLPTR ? nullptr : &result;

    if (IsPerformanceCase(p.caseName)) {
        ASSERT_EQ(p.expectResult, ACLBLAS_STATUS_SUCCESS);
        RunPerformanceCase(handle_, stream_, p, x, y);
        return;
    }

    aclblasStatus_t ret = aclblasCdotc_npu(handle_, stream_, p.n, x, p.incx, y, p.incy, resultPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != p.expectResult || p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    aclblasComplex golden{};
    ASSERT_EQ(aclblasCdotc_cpu(handle_, p.n, x, p.incx, y, p.incy, &golden), ACLBLAS_STATUS_SUCCESS);
    VerifyConfig realConfig;
    VerifyConfig imagConfig;
    applyMixedTolerance(realConfig, ACL_FLOAT, golden.real);
    applyMixedTolerance(imagConfig, ACL_FLOAT, golden.imag);
    // Preserve every significant float digit in failure logs so precision
    // diagnostics do not introduce an additional text-rounding error.
    std::cout << std::setprecision(9);
    EXPECT_TRUE(Verifier::verifyScalar(result.real, golden.real, realConfig, p.caseName + "/real"));
    EXPECT_TRUE(Verifier::verifyScalar(result.imag, golden.imag, imagConfig, p.caseName + "/imag"));
}
