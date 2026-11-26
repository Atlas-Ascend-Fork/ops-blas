/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "blas_test.h"
#include "cann_ops_blas.h"

// CANN 9.1.0 does not expose ACL_HIFLOAT4, while the shared CSV parser
// references the token for newer toolkit versions. The token is unused by
// isamin; map it to the latest FP4 dtype available in 9.1.0.
#ifndef ACL_HIFLOAT4
#define ACL_HIFLOAT4 ACL_FLOAT4_E1M2
#endif

#include "csv_loader.h"
#include "device.h"
#include "fill.h"
#include "isamin_golden.h"
#include "isamin_param.h"
#include "verify.h"

namespace {

constexpr int ISAMIN_PERF_WARMUP = 5;
constexpr int ISAMIN_PERF_SAMPLES = 20;
constexpr int ISAMIN_PROFILE_SAMPLES = 15;

struct IsaminPerfEvents {
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;

    ~IsaminPerfEvents()
    {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (stop != nullptr) {
            aclrtDestroyEvent(stop);
        }
    }
};

std::string IsaminSpecialTag(const IsaminParam& param)
{
    constexpr const char* prefix = "special=";
    if (param.description.rfind(prefix, 0) != 0) {
        return {};
    }
    const size_t end = param.description.find(';');
    return param.description.substr(
        std::char_traits<char>::length(prefix), end - std::char_traits<char>::length(prefix));
}

bool IsaminWantsNull(const IsaminParam& param) { return IsaminSpecialTag(param) == "nullptr"; }

std::vector<float> MakeIsaminHostInput(const IsaminParam& param, int64_t inputLength)
{
    const std::string special = IsaminSpecialTag(param);
    if (special == "nullptr") {
        return {};
    }
    if (special == "zero") {
        return makeBlasArray(inputLength, "VALUE_NORM_0", param.randomSeed);
    }
    if (special == "alter") {
        return makeBlasArray(inputLength, "RANDOM_ALTER", param.randomSeed);
    }
    if (special == "index") {
        return makeBlasArray(inputLength, "INDEX", param.randomSeed);
    }
    if (special == "extreme") {
        return makeBlasArray(inputLength, "RANDOM_EXTREME", param.randomSeed);
    }
    if (special == "inf") {
        return makeBlasArray(inputLength, "VALUE_NORM_INF", param.randomSeed);
    }
    if (special == "nan_all") {
        return makeBlasArray(inputLength, "VALUE_NORM_NAN", param.randomSeed);
    }

    std::vector<float> host = makeBlasArray(inputLength, param.x, param.randomSeed);
    if (special == "nan_inf" && !host.empty()) {
        for (int i = 0; i < param.n; ++i) {
            host[static_cast<size_t>(i) * param.incx] = std::numeric_limits<float>::infinity();
        }
        host[0] = std::numeric_limits<float>::quiet_NaN();
        if (param.n >= 3) {
            host[2] = -std::numeric_limits<float>::infinity();
        }
        return host;
    }
    if (special == "nan_max" && !host.empty()) {
        for (int i = 0; i < param.n; ++i) {
            host[static_cast<size_t>(i) * param.incx] = std::numeric_limits<float>::max();
        }
        host[0] = std::numeric_limits<float>::quiet_NaN();
        return host;
    }
    if (special == "nan_first" && !host.empty()) {
        host[0] = std::numeric_limits<float>::quiet_NaN();
    }
    return host;
}

std::unique_ptr<DeviceBuffer> MakeIsaminDeviceInput(const std::vector<float>& host, bool wantsNull)
{
    if (wantsNull) {
        return nullptr;
    }
    auto device = std::make_unique<DeviceBuffer>(host.size() * sizeof(float));
    device->copyFromHost(host.data(), host.size() * sizeof(float));
    return device;
}

void RunIsaminNegativeCase(aclblasHandle_t handle, aclrtStream stream, const IsaminParam& param)
{
    std::vector<float> dummy{1.0F};
    auto deviceX = MakeIsaminDeviceInput(dummy, IsaminWantsNull(param));
    DeviceBuffer deviceResult(sizeof(int32_t));

    const aclblasStatus_t status = aclblasIsamin(
        handle, param.n, deviceX ? deviceX->floatPtr() : nullptr, param.incx,
        static_cast<int32_t*>(deviceResult.ptr()));
    EXPECT_EQ(status, param.expectResult) << "[" << param.caseName << "] unexpected return code";
    if (status == ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    }
}

void RunIsaminAccuracyCase(aclblasHandle_t handle, aclrtStream stream, const IsaminParam& param)
{
    if (param.expectResult != ACLBLAS_STATUS_SUCCESS) {
        RunIsaminNegativeCase(handle, stream, param);
        return;
    }

    const bool quickReturn = param.n == 0 || param.incx < 1;
    const int64_t inputLength =
        quickReturn ? 1 : static_cast<int64_t>(1) + static_cast<int64_t>(param.n - 1) * param.incx;
    const std::vector<float> hostX = MakeIsaminHostInput(param, inputLength);
    const bool wantsNull = IsaminWantsNull(param);
    auto deviceX = MakeIsaminDeviceInput(hostX, wantsNull);
    DeviceBuffer deviceResult(sizeof(int32_t));

    const aclblasStatus_t status = aclblasIsamin(
        handle, param.n, deviceX ? deviceX->floatPtr() : nullptr, param.incx,
        static_cast<int32_t*>(deviceResult.ptr()));
    ASSERT_EQ(status, param.expectResult) << "[" << param.caseName << "] unexpected return code";
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS) << aclGetRecentErrMsg();

    int result = -1;
    deviceResult.copyToHost(&result, sizeof(result));
    const int golden = quickReturn ? 0 : isamin_golden(param.n, hostX.data(), param.incx);
    EXPECT_EQ(result, golden) << "[" << param.caseName << "] NPU=" << result << " golden=" << golden;
}

aclblasStatus_t InvokeIsaminPerformance(
    aclblasHandle_t handle, const IsaminParam& param, DeviceBuffer& deviceX, DeviceBuffer& deviceResult)
{
    return aclblasIsamin(handle, param.n, deviceX.floatPtr(), param.incx, static_cast<int32_t*>(deviceResult.ptr()));
}

void RunIsaminProfiledPerformanceCase(
    aclblasHandle_t handle, aclrtStream stream, const IsaminParam& param, const std::vector<float>& hostX,
    DeviceBuffer& deviceX, DeviceBuffer& deviceResult)
{
    for (int i = 0; i < ISAMIN_PERF_WARMUP; ++i) {
        ASSERT_EQ(InvokeIsaminPerformance(handle, param, deviceX, deviceResult), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    for (int i = 0; i < ISAMIN_PROFILE_SAMPLES; ++i) {
        ASSERT_EQ(InvokeIsaminPerformance(handle, param, deviceX, deviceResult), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    int result = -1;
    deviceResult.copyToHost(&result, sizeof(result));
    EXPECT_EQ(result, isamin_golden(param.n, hostX.data(), param.incx))
        << "[" << param.caseName << "] profile case returned a wrong index";
    std::cout << "[PROFILE] case=" << param.caseName << " n=" << param.n << " incx=" << param.incx
              << " warmup=" << ISAMIN_PERF_WARMUP << " samples=" << ISAMIN_PROFILE_SAMPLES
              << " metric=msprof_op_task_duration" << std::endl;
}

void RunIsaminPerformanceCase(aclblasHandle_t handle, aclrtStream stream, const IsaminParam& param)
{
    ASSERT_GT(param.n, 0) << "[" << param.caseName << "] performance cases require n > 0";
    ASSERT_EQ(param.incx, 1) << "[" << param.caseName << "] performance cases require incx = 1";

    const std::vector<float> hostX = makeBlasArray(param.n, param.x, param.randomSeed);
    DeviceBuffer deviceX(hostX.size() * sizeof(float));
    deviceX.copyFromHost(hostX.data(), hostX.size() * sizeof(float));
    DeviceBuffer deviceResult(sizeof(int32_t));

    if (std::getenv("ISAMIN_PROFILE") != nullptr) {
        RunIsaminProfiledPerformanceCase(handle, stream, param, hostX, deviceX, deviceResult);
        return;
    }

    for (int i = 0; i < ISAMIN_PERF_WARMUP; ++i) {
        ASSERT_EQ(InvokeIsaminPerformance(handle, param, deviceX, deviceResult), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    IsaminPerfEvents events;
    ASSERT_EQ(aclrtCreateEvent(&events.start), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&events.stop), ACL_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(events.start, stream), ACL_SUCCESS);
    for (int i = 0; i < ISAMIN_PERF_SAMPLES; ++i) {
        ASSERT_EQ(InvokeIsaminPerformance(handle, param, deviceX, deviceResult), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtRecordEvent(events.stop, stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeEvent(events.stop), ACL_SUCCESS);

    float elapsedMs = 0.0F;
    ASSERT_EQ(aclrtEventElapsedTime(&elapsedMs, events.start, events.stop), ACL_SUCCESS);
    const float averageUs = elapsedMs * 1000.0F / static_cast<float>(ISAMIN_PERF_SAMPLES);
    ASSERT_GT(averageUs, 0.0F);

    int result = -1;
    deviceResult.copyToHost(&result, sizeof(result));
    EXPECT_EQ(result, isamin_golden(param.n, hostX.data(), param.incx))
        << "[" << param.caseName << "] performance case returned a wrong index";
    std::cout << "[PERF] case=" << param.caseName << " n=" << param.n << " incx=" << param.incx
              << " warmup=" << ISAMIN_PERF_WARMUP << " samples=" << ISAMIN_PERF_SAMPLES << " avg_us=" << std::fixed
              << std::setprecision(6) << averageUs << std::endl;
}

} // namespace

class IsaminTest : public BlasTest<IsaminParam> {};

TEST_F(IsaminTest, NullHandle)
{
    int result = 0;
    EXPECT_EQ(aclblasIsamin(nullptr, 5, nullptr, 1, &result), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(IsaminTest, NullResult)
{
    EXPECT_EQ(aclblasIsamin(handle_, 0, nullptr, 1, nullptr), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasIsamin(handle_, 5, nullptr, 0, nullptr), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasIsamin(handle_, 5, nullptr, 1, nullptr), ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(IsaminTest, QuickReturnWritesZero)
{
    DeviceBuffer result(sizeof(int32_t));
    ASSERT_EQ(aclrtMemset(result.ptr(), sizeof(int32_t), 0xFF, sizeof(int32_t)), ACL_SUCCESS);
    EXPECT_EQ(aclblasIsamin(handle_, 0, nullptr, 1, static_cast<int32_t*>(result.ptr())), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    int value = -1;
    result.copyToHost(&value, sizeof(value));
    EXPECT_EQ(value, 0);
}

TEST_F(IsaminTest, CsvDistribution)
{
    const auto cases = GetCasesFromCsv<IsaminParam>(ReplaceFileExtension2Csv(__FILE__));
    int uniform = 0;
    int gaussian = 0;
    int performance = 0;
    for (const auto& item : cases) {
        if (item.caseName.rfind("TC_PF_", 0) == 0) {
            ++performance;
            continue;
        }
        if (item.x.method == BlasFillMode::M_RANDOM) {
            ++uniform;
        } else if (item.x.method == BlasFillMode::M_GAUSS) {
            ++gaussian;
        }
    }
    EXPECT_EQ(uniform, 500);
    EXPECT_EQ(gaussian, 500);
    EXPECT_EQ(performance, 200);
}

INSTANTIATE_TEST_SUITE_P(
    Isamin, IsaminTest, ::testing::ValuesIn(GetCasesFromCsv<IsaminParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<IsaminParam>);

TEST_P(IsaminTest, CsvDriven)
{
    const auto& param = GetParam();
    if (param.caseName.rfind("TC_PF_", 0) == 0) {
        RunIsaminPerformanceCase(handle_, stream_, param);
    } else {
        RunIsaminAccuracyCase(handle_, stream_, param);
    }
}
