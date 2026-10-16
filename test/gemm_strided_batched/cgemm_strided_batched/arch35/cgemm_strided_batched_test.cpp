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
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "aclblas_handle_internal.h"
#include "blas_test.h"
#include "cgemm_strided_batched_golden.h"
#include "cgemm_strided_batched_npu_wrapper.h"
#include "cgemm_strided_batched_param.h"
#include "csv_loader.h"
#include "device.h"
#include "verify.h"

namespace cgemm_strided_batched_test {

// Keep the task's component-wise precision thresholds and memory guards unchanged.
constexpr int ABSOLUTE_TOLERANCE_EXPONENT = -16;
constexpr int RELATIVE_TOLERANCE_EXPONENT = -10;
constexpr int FP32_FRACTION_BITS = 23;
constexpr int FP32_MIN_NORMAL_EXPONENT = -126;
constexpr double MIN_ABSOLUTE_ERROR_LIMIT = 0.01;
constexpr int MAX_ERROR_ULPS = 32;
constexpr double MIN_MATCH_RATIO = 0.99;
constexpr size_t GUARD_BUFFER_BYTES = 4096;
constexpr uint8_t GUARD_BYTE_PATTERN = 0xa5;
constexpr int WORKSPACE_POISON_PATTERN = 0x3f;
constexpr int BENCHMARK_WARMUP_COUNT = 10;
constexpr double MICROSECONDS_PER_MILLISECOND = 1000.0;
constexpr int REPORT_SIGNIFICANT_DIGITS = 10;

struct ComponentResult {
    size_t count = 0;
    size_t matched = 0;
    size_t exactMismatches = 0;
    size_t nonfiniteMismatches = 0;
    size_t limitViolations = 0;
    double maxError = 0;
    void Add(float actual, float expected)
    {
        ++count;
        if (std::memcmp(&actual, &expected, sizeof(float)) != 0)
            ++exactMismatches;
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            bool equal = (std::isnan(actual) && std::isnan(expected)) || (std::isinf(actual) && actual == expected);
            matched += equal;
            nonfiniteMismatches += !equal;
            return;
        }
        double error = std::abs(double(actual) - expected);
        maxError = std::max(maxError, error);
        matched += error <= std::ldexp(1.0, ABSOLUTE_TOLERANCE_EXPONENT) +
                                std::ldexp(1.0, RELATIVE_TOLERANCE_EXPONENT) * std::abs(double(expected));
        double ulp = getUlpsAt(double(expected), FP32_FRACTION_BITS, FP32_MIN_NORMAL_EXPONENT);
        limitViolations += error > std::max(MIN_ABSOLUTE_ERROR_LIMIT, MAX_ERROR_ULPS * ulp);
    }
    double Ratio() const { return count ? double(matched) / count : 1; }
    bool Pass(bool exact) const
    {
        return nonfiniteMismatches == 0 &&
               (exact ? exactMismatches == 0 : Ratio() >= MIN_MATCH_RATIO && limitViolations == 0);
    }
};

// The fixture shares its handle across cases. Restore library ownership before
// freeing a temporary user buffer, including on an ASSERT early return.
struct WorkspaceRestore {
    aclblasHandle_t handle = nullptr;
    ~WorkspaceRestore()
    {
        if (handle != nullptr)
            EXPECT_EQ(ResetToDefaultWorkspace(handle), ACLBLAS_STATUS_SUCCESS);
    }
};

struct EventPair {
    aclrtEvent begin = nullptr, end = nullptr;
    ~EventPair()
    {
        if (end != nullptr)
            EXPECT_EQ(aclrtDestroyEvent(end), ACL_SUCCESS);
        if (begin != nullptr)
            EXPECT_EQ(aclrtDestroyEvent(begin), ACL_SUCCESS);
    }
};

void CheckNoWrite(const Invocation& q, aclrtStream stream, aclblasStatus_t expected)
{
    DeviceBuffer dummy(GUARD_BUFFER_BYTES);
    std::vector<uint8_t> guard(GUARD_BUFFER_BYTES, GUARD_BYTE_PATTERN);
    std::vector<uint8_t> after(GUARD_BUFFER_BYTES);
    dummy.copyFromHost(guard.data(), guard.size());
    auto ptr = reinterpret_cast<Complex*>(dummy.ptr());
    ASSERT_EQ(q.Run(ptr, ptr, ptr), expected);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    dummy.copyToHost(after.data(), after.size());
    EXPECT_EQ(guard, after);
}

class NumericalCase {
public:
    NumericalCase(const Invocation& call, aclrtStream suppliedStream) : q(call), stream(suppliedStream) {}

    void Prepare()
    {
        const auto& p = q.p;
        distribution = p.S("distribution");
        if (const char* configured = std::getenv("CGEMM_SB_DISTRIBUTION"))
            distribution = configured;
        if (distribution.empty())
            distribution = "package";
        bool normal = distribution == "normal" || (distribution == "mixed" && p.I("random_seed") % 2 != 0);
        bool preservePackage = distribution == "package";
        a = std::make_unique<Matrix>(
            q.ta == ACLBLAS_OP_N ? q.m : q.k, q.ta == ACLBLAS_OP_N ? q.k : q.m, q.lda, q.sa, q.batch, p.S("a_fill"),
            p.I("random_seed"), p.I("a_offset"), normal, preservePackage);
        b = std::make_unique<Matrix>(
            q.tb == ACLBLAS_OP_N ? q.k : q.n, q.tb == ACLBLAS_OP_N ? q.n : q.k, q.ldb, q.sb, q.batch, p.S("b_fill"),
            p.I("random_seed") + 1, p.I("b_offset"), normal, preservePackage);
        c = std::make_unique<Matrix>(
            q.m, q.n, q.ldc, q.sc, q.batch, p.S("c_fill"), p.I("random_seed") + 2, p.I("c_offset"), normal,
            preservePackage);
        ASSERT_NO_FATAL_FAILURE(InjectNonfinite());
        golden = c->host;
        exact = q.alpha.real == 0 && q.alpha.imag == 0;
        if (p.I("poison_workspace") != 0) {
            poisonedWorkspace = std::make_unique<DeviceBuffer>(GUARD_BUFFER_BYTES);
            ASSERT_EQ(
                aclblasSetWorkspace(q.handle, poisonedWorkspace->ptr(), GUARD_BUFFER_BYTES), ACLBLAS_STATUS_SUCCESS);
            restoreWorkspace.handle = q.handle;
            ASSERT_EQ(
                aclrtMemset(poisonedWorkspace->ptr(), GUARD_BUFFER_BYTES, WORKSPACE_POISON_PATTERN, GUARD_BUFFER_BYTES),
                ACL_SUCCESS);
        }
    }

    void Execute()
    {
        ASSERT_EQ(q.Run(a->Ptr(), b->Ptr(), c->Ptr()), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        actual.resize(c->host.size());
        c->device->copyToHost(actual.data(), actual.size() * sizeof(Complex));
        CalculateGolden();
        CheckPrecision();
    }

    void Benchmark()
    {
        if (q.p.caseName.find("TC_PF_") != 0 && !std::getenv("CGEMM_SB_BENCH_ALL"))
            return;
        for (int i = 0; i < BENCHMARK_WARMUP_COUNT; ++i) {
            c->device->copyFromHost(c->host.data(), c->host.size() * sizeof(Complex));
            ASSERT_EQ(q.Run(a->Ptr(), b->Ptr(), c->Ptr()), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        }
        EventPair events;
        ASSERT_EQ(aclrtCreateEvent(&events.begin), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateEvent(&events.end), ACL_SUCCESS);
        constexpr int samples = 60;
        for (int i = 0; i < samples; ++i) {
            c->device->copyFromHost(c->host.data(), c->host.size() * sizeof(Complex));
            ASSERT_EQ(aclrtRecordEvent(events.begin, stream), ACL_SUCCESS);
            ASSERT_EQ(q.Run(a->Ptr(), b->Ptr(), c->Ptr()), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtRecordEvent(events.end, stream), ACL_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeEvent(events.end), ACL_SUCCESS);
            float ms = 0;
            ASSERT_EQ(aclrtEventElapsedTime(&ms, events.begin, events.end), ACL_SUCCESS);
            avgUs += ms * MICROSECONDS_PER_MILLISECOND / samples;
        }
        std::cout << "PERFORMANCE " << q.p.caseName << " avg_us=" << avgUs << " samples=" << samples << std::endl;
    }

    void WriteReport() const
    {
        const char* path = std::getenv("CGEMM_SB_RESULTS");
        if (!path)
            return;
        bool empty = !std::ifstream(path).good();
        std::ofstream report(path, std::ios::app);
        if (empty)
            report << "case_name,distribution,m,n,k,batch,real_ratio,imag_ratio,real_max,imag_max,precision_pass,avg_"
                      "us,workspace_bytes,matrix_device_bytes,transA,transB,lda,ldb,ldc,strideA,strideB,strideC,"
                      "alpha_real,alpha_imag,beta_real,beta_imag,real_limit_violations,imag_limit_violations,"
                      "nonfinite_mismatches,exact_mismatches,padding_changes\n";
        report << std::setprecision(REPORT_SIGNIFICANT_DIGITS) << q.p.caseName << ',' << distribution << ',' << q.m
               << ',' << q.n << ',' << q.k << ',' << q.batch << ',' << real.Ratio() << ',' << imag.Ratio() << ','
               << real.maxError << ',' << imag.maxError << ','
               << (real.Pass(exact) && imag.Pass(exact) && paddingChanges == 0) << ',' << avgUs << ','
               << GetEffectiveWorkspaceSize(q.handle) << ','
               << a->device->size() + b->device->size() + c->device->size() << ',' << q.ta << ',' << q.tb << ','
               << q.lda << ',' << q.ldb << ',' << q.ldc << ',' << q.sa << ',' << q.sb << ',' << q.sc << ','
               << q.alpha.real << ',' << q.alpha.imag << ',' << q.beta.real << ',' << q.beta.imag << ','
               << real.limitViolations << ',' << imag.limitViolations << ','
               << real.nonfiniteMismatches + imag.nonfiniteMismatches << ','
               << (exact ? real.exactMismatches + imag.exactMismatches : 0) << ',' << paddingChanges << '\n';
        ASSERT_TRUE(report.good());
    }

private:
    void InjectNonfinite()
    {
        const auto& p = q.p;
        if (p.S("inject_matrix").empty())
            return;
        Matrix& target = p.S("inject_matrix") == "A" ? *a : *b;
        int64_t stride = p.S("inject_matrix") == "A" ? q.sa : q.sb;
        int batchIndex = p.I("inject_batch");
        ASSERT_GE(batchIndex, 0);
        ASSERT_LT(batchIndex, q.batch);
        float value = p.S("inject_value") == "NaN" ? std::numeric_limits<float>::quiet_NaN() :
                                                     std::numeric_limits<float>::infinity();
        target.host.at(target.base + batchIndex * stride).real = value;
        target.device->copyFromHost(target.host.data(), target.host.size() * sizeof(Complex));
    }

    void CalculateGolden()
    {
        ASSERT_EQ(
            aclblasCgemmStridedBatched_cpu(
                q.handle, static_cast<aclblasOperation_t>(q.ta), static_cast<aclblasOperation_t>(q.tb), q.m, q.n, q.k,
                &q.alpha, a->host.data() + a->base, q.lda, q.sa, b->host.data() + b->base, q.ldb, q.sb, &q.beta,
                golden.data() + c->base, q.ldc, q.sc, q.batch),
            ACLBLAS_STATUS_SUCCESS);
    }

    void CheckPrecision()
    {
        std::vector<bool> isOutput(golden.size(), false);
        for (int batch = 0; batch < q.batch; ++batch)
            for (int j = 0; j < q.n; ++j)
                for (int i = 0; i < q.m; ++i) {
                    size_t index = c->base + batch * q.sc + int64_t(j) * q.ldc + i;
                    isOutput[index] = true;
                    real.Add(actual[index].real, golden[index].real);
                    imag.Add(actual[index].imag, golden[index].imag);
                }
        for (size_t i = 0; i < golden.size(); ++i)
            if (!isOutput[i])
                paddingChanges += std::memcmp(&actual[i], &golden[i], sizeof(Complex)) != 0;
        std::cout << std::setprecision(REPORT_SIGNIFICANT_DIGITS) << "PRECISION " << q.p.caseName
                  << " real_ratio=" << real.Ratio() << " real_max=" << real.maxError << " imag_ratio=" << imag.Ratio()
                  << " imag_max=" << imag.maxError << " real_limits=" << real.limitViolations
                  << " imag_limits=" << imag.limitViolations
                  << " nonfinite=" << real.nonfiniteMismatches + imag.nonfiniteMismatches
                  << " exact_mismatches=" << (exact ? real.exactMismatches + imag.exactMismatches : 0)
                  << " padding_changes=" << paddingChanges << std::endl;
        if (!real.Pass(exact) || !imag.Pass(exact) || paddingChanges != 0)
            DumpFailure();
        EXPECT_TRUE(real.Pass(exact));
        EXPECT_TRUE(imag.Pass(exact));
        EXPECT_EQ(paddingChanges, 0);
    }

    void DumpFailure() const
    {
        const char* prefix = std::getenv("CGEMM_SB_DUMP");
        if (!prefix)
            return;
        auto dump = [&](const char* label, const std::vector<Complex>& values) {
            std::ofstream file(std::string(prefix) + "-" + q.p.caseName + "-" + label + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(Complex));
            EXPECT_TRUE(file.good());
        };
        dump("a", a->host);
        dump("b", b->host);
        dump("c", c->host);
        dump("golden", golden);
        dump("actual", actual);
        std::cout << "DUMP_BASE " << a->base << ',' << b->base << ',' << c->base << std::endl;
    }

    const Invocation& q;
    aclrtStream stream;
    std::string distribution;
    std::unique_ptr<Matrix> a, b, c;
    std::vector<Complex> golden, actual;
    ComponentResult real, imag;
    size_t paddingChanges = 0;
    bool exact = false;
    double avgUs = 0;
    std::unique_ptr<DeviceBuffer> poisonedWorkspace;
    // Declared last so it restores the shared handle before the buffer is freed.
    WorkspaceRestore restoreWorkspace;
};

class CgemmSbTest : public BlasTest<Case> {};

TEST_F(CgemmSbTest, NullHandle)
{
    EXPECT_EQ(
        aclblasCgemmStridedBatched_npu(
            nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 1, 1, 1, nullptr, nullptr, 1, 1, nullptr, 1, 1, nullptr, nullptr, 1, 1,
            1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    CgemmSb, CgemmSbTest, ::testing::ValuesIn(GetCasesFromCsv<Case>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Case>);

TEST_P(CgemmSbTest, CsvDriven)
{
    const auto& p = GetParam();
    Invocation call(p, handle_);
    auto expected = parseStatus(p.S("expect_result"));
    if (expected != ACLBLAS_STATUS_SUCCESS || call.m == 0 || call.n == 0 || call.batch == 0) {
        CheckNoWrite(call, stream_, expected);
        return;
    }
    NumericalCase test(call, stream_);
    ASSERT_NO_FATAL_FAILURE(test.Prepare());
    ASSERT_NO_FATAL_FAILURE(test.Execute());
    ASSERT_NO_FATAL_FAILURE(test.Benchmark());
    test.WriteReport();
}
} // namespace cgemm_strided_batched_test
