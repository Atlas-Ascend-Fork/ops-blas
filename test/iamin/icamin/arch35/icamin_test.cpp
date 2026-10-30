/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "icamin_param.h"
#include "icamin_golden.h"
#include "icamin_npu_wrapper.h"

class IcaminArch35Test : public BlasTest<IcaminParam> {};

namespace {

bool PerfBenchEnabled()
{
    const char* env = std::getenv("ICAMIN_PERF_BENCH");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

// Task §3.3: NPU avg_us <= gpu_ms * 1000 / 0.4 (from test/iamin/icamin/gpu_baseline.csv).
constexpr double kPerfRatio = 0.4;

struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const
    {
        return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(static_cast<uint32_t>(p.second));
    }
};

std::string GpuBaselineCsvPath()
{
    // arch35/icamin_test.cpp → ../gpu_baseline.csv
    std::string testCsv = ReplaceFileExtension2Csv(__FILE__);
    const auto slash = testCsv.find_last_of("/\\");
    std::string archDir = (slash == std::string::npos) ? std::string(".") : testCsv.substr(0, slash);
    const auto slash2 = archDir.find_last_of("/\\");
    std::string opDir = (slash2 == std::string::npos) ? std::string(".") : archDir.substr(0, slash2);
    return opDir + "/gpu_baseline.csv";
}

const std::unordered_map<std::pair<int, int>, double, PairHash>& GpuBaselineMs()
{
    static std::unordered_map<std::pair<int, int>, double, PairHash> table;
    static bool loaded = false;
    if (loaded) {
        return table;
    }
    loaded = true;
    const std::string path = GpuBaselineCsvPath();
    std::ifstream in(path);
    if (!in) {
        std::cerr << "[PERF] WARN: cannot open gpu_baseline.csv at " << path << std::endl;
        return table;
    }
    std::string line;
    if (!std::getline(in, line)) {
        return table;
    }
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string id;
        std::string nStr;
        std::string incxStr;
        std::string msStr;
        if (!std::getline(ss, id, ',') || !std::getline(ss, nStr, ',') || !std::getline(ss, incxStr, ',') ||
            !std::getline(ss, msStr, ',')) {
            continue;
        }
        try {
            const int n = std::stoi(nStr);
            const int incx = std::stoi(incxStr);
            const double ms = std::stod(msStr);
            table[{n, incx}] = ms; // duplicate (n,incx) keep last (same ms for doc 4M repeats)
        } catch (...) {
            continue;
        }
    }
    std::cerr << "[PERF] loaded gpu_baseline entries=" << table.size() << " from " << path << std::endl;
    return table;
}

// Returns threshold_us, or <0 if no baseline.
double PerfThresholdUs(int n, int incx)
{
    const auto& table = GpuBaselineMs();
    const auto it = table.find({n, incx});
    if (it == table.end()) {
        return -1.0;
    }
    return it->second * 1000.0 / kPerfRatio;
}

std::vector<aclblasComplex> MakeIcaminHostX(const IcaminParam& p)
{
    const int absIncx = std::abs(p.incx);
    const int64_t xLen =
        (p.n > 0 && absIncx > 0) ? static_cast<int64_t>(1) + static_cast<int64_t>(p.n - 1) * absIncx : 0;
    const std::vector<float> flat = makeBlasArray(xLen * 2, p.x, p.randomSeed);
    std::vector<aclblasComplex> xHost(flat.size() / 2);
    for (size_t i = 0; i < xHost.size(); ++i) {
        xHost[i].real = flat[2 * i];
        xHost[i].imag = flat[2 * i + 1];
    }
    // Mixed-NaN overlays (CSV fill alone cannot place a single NaN): cblas expects
    // index 1 when the first abs1 is NaN, even if later elements are finite zeros.
    if (p.description.rfind("mix_nan_first_", 0) == 0 && !xHost.empty()) {
        xHost[0].real = NAN;
        xHost[0].imag = 0.0f;
    } else if (p.description.rfind("mix_nan_mid_", 0) == 0 && xHost.size() >= 2) {
        xHost[1].real = NAN;
        xHost[1].imag = 0.0f;
    }
    return xHost;
}

aclblasStatus_t RunIcaminPerfCase(aclblasHandle_t handle, const IcaminParam& p, const aclblasComplex* xPtr, int* result)
{
    constexpr int kWarmup = 10;
    constexpr int kIters = 100; // task: effective samples > 50
    double avgUs = 0.0;
    const aclblasStatus_t ret = aclblasIcamin_npu_bench(handle, p.n, xPtr, p.incx, result, kWarmup, kIters, &avgUs);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "[" << p.caseName << "] unexpected return code in perf bench";
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    const double thr = PerfThresholdUs(p.n, p.incx);
    std::cout << "[PERF] case=" << p.caseName << " n=" << p.n << " incx=" << p.incx << " warmup=" << kWarmup
              << " iters=" << kIters << " avg_us=" << avgUs;
    if (thr > 0.0) {
        const bool pass = avgUs <= thr;
        std::cout << " threshold_us=" << thr << " verdict=" << (pass ? "PASS" : "FAIL") << std::endl;
        if (!pass) {
            ADD_FAILURE() << "[" << p.caseName << "] perf avg_us=" << avgUs << " > threshold_us=" << thr;
        }
    } else {
        std::cout << " verdict=NO_REF" << std::endl;
        ADD_FAILURE() << "[" << p.caseName << "] missing gpu_baseline for n=" << p.n << " incx=" << p.incx;
    }
    return ret;
}

} // namespace

TEST_F(IcaminArch35Test, NullHandle)
{
    int result = 0;
    aclblasStatus_t ret = aclblasIcamin_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

INSTANTIATE_TEST_SUITE_P(
    Icamin, IcaminArch35Test, ::testing::ValuesIn(GetCasesFromCsv<IcaminParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<IcaminParam>);

TEST_P(IcaminArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    std::cout << "\n=== RUN  [" << p.caseName << "] n=" << p.n << " incx=" << p.incx
              << " x_method=" << static_cast<int>(p.x.method) << " desc=" << p.description << " ===" << std::endl;

    std::vector<aclblasComplex> xHost = MakeIcaminHostX(p);
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    if (p.resultIsNull) {
        const aclblasStatus_t ret = aclblasIcamin(IcaminArch35Test::handle_, p.n, xPtr, p.incx, nullptr);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
            << "[" << p.caseName << "] unexpected return code for null result";
        return;
    }

    const bool doBench = p.caseName.rfind("TC_PF_", 0) == 0 && PerfBenchEnabled() &&
                         p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0 && p.incx >= 1 && xPtr != nullptr;
    int result = 0;
    aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;
    if (doBench) {
        ret = RunIcaminPerfCase(handle_, p, xPtr, &result);
    } else {
        ret = aclblasIcamin_npu(IcaminArch35Test::handle_, p.n, xPtr, p.incx, &result);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
            << "[" << p.caseName << "] unexpected return code: got " << static_cast<int>(ret) << " expect "
            << static_cast<int>(p.expectResult);
    }
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    int golden = 0;
    // issue #667: CPU 参考实现返回码必须校验——golden 失效时断言会退化为与 0 比较
    const aclblasStatus_t goldenRet = aclblasIcamin_cpu(IcaminArch35Test::handle_, p.n, xHost.data(), p.incx, &golden);
    EXPECT_EQ(static_cast<int>(goldenRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS))
        << "[" << p.caseName << "] CPU golden failed: " << static_cast<int>(goldenRet);
    EXPECT_EQ(result, golden) << "[" << p.caseName << "] index mismatch: NPU=" << result << " golden=" << golden;
}
