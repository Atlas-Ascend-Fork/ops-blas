/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include "acl/acl.h"
#include "../cscal_test_support.h"
namespace {
using namespace cscaltest;
inline void Acl(aclError e, const char* op)
{
    Require(e == ACL_SUCCESS, std::string(op) + " failed: " + std::to_string(e));
}
inline void Blas(aclblasStatus_t e, const char* op)
{
    Require(e == ACLBLAS_STATUS_SUCCESS, std::string(op) + ": " + Status(e));
}
class Runtime {
public:
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    int device = 0;
    bool init = false, selected = false;
    Runtime()
    {
        device = Int(Env("CSCAL_DEVICE", Env("ASCEND_DEVICE_ID", "0")));
        try {
            Acl(aclInit(nullptr), "aclInit");
            init = true;
            Acl(aclrtSetDevice(device), "aclrtSetDevice");
            selected = true;
            Acl(aclrtCreateStream(&stream), "aclrtCreateStream");
            Blas(aclblasCreate(&handle), "aclblasCreate");
            Blas(aclblasSetStream(handle, stream), "aclblasSetStream");
        } catch (...) {
            Cleanup();
            throw;
        }
    }
    ~Runtime() { Cleanup(); }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

private:
    void Cleanup() noexcept
    {
        if (stream) {
            // issue #605: 收尾同步失败仅记录，不在清理路径抛出
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
                std::cerr << "cscal_test: cleanup aclrtSynchronizeStream failed" << std::endl;
            }
        }
        if (handle) {
            aclblasDestroy(handle);
            handle = nullptr;
        }
        if (stream) {
            aclrtDestroyStream(stream);
            stream = nullptr;
        }
        if (selected) {
            aclrtResetDevice(device);
            selected = false;
        }
        if (init) {
            aclFinalize();
            init = false;
        }
    }
};
class Buffer {
public:
    void* data = nullptr;
    size_t bytes = 0;
    aclrtStream stream;
    Buffer(size_t n, aclrtStream s) : bytes(n), stream(s)
    {
        Acl(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~Buffer()
    {
        if (data) {
            // issue #605: 收尾同步失败仅记录，不在析构路径抛出
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
                std::cerr << "cscal_test: buffer aclrtSynchronizeStream failed" << std::endl;
            }
            aclrtFree(data);
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
class Event {
public:
    aclrtEvent e = nullptr;
    Event() { Acl(aclrtCreateEventWithFlag(&e, ACL_EVENT_TIME_LINE), "create timeline event"); }
    ~Event()
    {
        if (e)
            aclrtDestroyEvent(e);
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
template <typename Reset, typename Invoke>
std::vector<double> MeasureDeviceTimes(
    aclrtStream stream, int samples, int warmup, const Reset& reset, const Invoke& invoke)
{
    for (int i = 0; i < warmup; ++i) {
        reset();
        Blas(invoke(), "warmup Cscal");
    }
    Acl(aclrtSynchronizeStream(stream), "warmup sync");
    Event start, end;
    std::vector<double> times;
    times.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        // Reset precedes start event on the SAME stream, outside the timed interval.
        // This prevents alpha accumulation/overflow and defines a warm-reset cache regime.
        reset();
        Acl(aclrtRecordEvent(start.e, stream), "record start");
        Blas(invoke(), "timed Cscal");
        Acl(aclrtRecordEvent(end.e, stream), "record end");
        Acl(aclrtSynchronizeStream(stream), "timing sync");
        float ms = 0;
        Acl(aclrtEventElapsedTime(&ms, start.e, end.e), "event elapsed");
        Require(std::isfinite(ms) && ms > 0, "invalid/nonpositive device time");
        times.push_back(double(ms) * 1000.0);
    }
    return times;
}
Result RunCase(Runtime& rt, const Case& c)
{
    auto input = MakeInput(c);
    auto golden = MakeGolden(c, input);
    const size_t bytes = input.data.size() * sizeof(aclblasComplex);
    Buffer work(bytes, rt.stream);
    Acl(aclrtMemcpy(work.data, bytes, input.data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D input");
    auto x = reinterpret_cast<aclblasComplex*>(work.data) + input.prefix;
    auto invoke = [&] {
        return aclblasCscal(
            c.nullHandle ? nullptr : rt.handle, c.n, c.nullAlpha ? nullptr : &c.alpha, c.nullX ? nullptr : x, c.incx);
    };
    const auto status = invoke();
    Acl(aclrtSynchronizeStream(rt.stream), "synchronize correctness invocation");
    std::vector<aclblasComplex> output(input.data.size());
    Acl(aclrtMemcpy(output.data(), bytes, work.data, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H output");
    auto result = Check(c, input, golden, output, status);
    result.bytes = bytes;
    if (!c.performance || !result.pass)
        return result;
    Require(WillWrite(c), "refuse timing a no-op/negative case");
    const int samples = Int(Env("CSCAL_SAMPLES", "64")), warmup = Int(Env("CSCAL_WARMUP", "10"));
    Require(samples > 50 && samples <= 100000 && warmup >= 1 && warmup <= 10000, "invalid warmup/sample count");
    Buffer pristine(bytes, rt.stream);
    result.bytes += bytes;
    Acl(aclrtMemcpy(pristine.data, bytes, input.data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D pristine");
    auto reset = [&] {
        Acl(aclrtMemcpyAsync(work.data, bytes, pristine.data, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream),
            "D2D reset");
    };
    auto times = MeasureDeviceTimes(rt.stream, samples, warmup, reset, invoke);
    Acl(aclrtMemcpy(output.data(), bytes, work.data, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H timed output");
    auto finalCheck = Check(c, input, golden, output, ACLBLAS_STATUS_SUCCESS);
    result.pass &= finalCheck.pass;
    result.real = finalCheck.real;
    result.imag = finalCheck.imag;
    result.guards &= finalCheck.guards;
    result.zeroExact &= finalCheck.zeroExact;
    result.samples = samples;
    result.avgUs = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::sort(times.begin(), times.end());
    result.minUs = times.front();
    result.maxUs = times.back();
    result.medianUs =
        times.size() % 2 ? times[times.size() / 2] : (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2;
    return result;
}
class CscalCsvTest : public ::testing::TestWithParam<Case> {
protected:
    static std::unique_ptr<Runtime> runtime;
    static bool poisoned;
    static void SetUpTestSuite() { ASSERT_NO_THROW(runtime.reset(new Runtime)); }
    static void TearDownTestSuite() { runtime.reset(); }
};
std::unique_ptr<Runtime> CscalCsvTest::runtime;
bool CscalCsvTest::poisoned = false;

TEST_P(CscalCsvTest, CsvDriven)
{
    ASSERT_NE(runtime, nullptr);
    ASSERT_FALSE(poisoned) << "An earlier runtime error invalidated this stream";
    const auto& c = GetParam();
    Result r;
    try {
        r = RunCase(*runtime, c);
    } catch (const std::exception& e) {
        r.pass = false;
        r.error = e.what();
        poisoned = true;
    }
    const auto report = Env("CSCAL_REPORT", "");
    if (!report.empty())
        WriteResult(report, c, r);
    RecordProperty("case_name", c.id);
    RecordProperty("real_matched_ratio", std::to_string(r.real.Ratio()));
    RecordProperty("imag_matched_ratio", std::to_string(r.imag.Ratio()));
    EXPECT_TRUE(r.pass) << c.id << ": " << r.error << " status=" << r.actualStatus << " guards=" << r.guards
                        << " matched=" << r.real.Ratio() << '/' << r.imag.Ratio();
}

// The official verifier extracts the final /TC_* name from each GTest result.
INSTANTIATE_TEST_SUITE_P(
    Cscal950, CscalCsvTest,
    ::testing::ValuesIn(
        LoadCases(Env("CSCAL_CASES", std::string(__FILE__).substr(0, std::string(__FILE__).rfind('.')) + ".csv"))),
    [](const ::testing::TestParamInfo<Case>& info) { return info.param.id; });
} // namespace
