/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ── sspr2 arch22 ST（GTest + CSV 驱动，NPU 后端） ───────────────────────────
// 执行流：CSV 装载 → fill+seed 生成 → aclblasSspr2_npu → 返回码断言
// → check 分派（GOLDEN/AP_UNCHANGED/NAN_PROPAGATE/NO_COMPUTE/STATUS_ONLY/
// FRAME_GUARD）→ 逐 case_id 输出 EVIDENCE 行。
// 组成：
//   Sspr2Arch22 —— 测试用例 CSV（19 列）逐条执行；
//   Sspr2Pf     —— TC_PF 性能用例（sspr2_test.csv，200 条，perf/verify 双模式）。

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sspr2_param.h"
#include "sspr2_golden.h"
#include "sspr2_npu_wrapper.h"

namespace {

// FRAME_GUARD 用例显式长度元数据（装载层长度断言配合）
const std::map<std::string, std::map<std::string, size_t>>& FrameGuardCases()
{
    static const std::map<std::string, std::map<std::string, size_t>> kGuard = {
        {"SSPR2-L2-016", {{"ap", 5}}},
    };
    return kGuard;
}

struct CaseBuffers {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> ap;
    std::vector<float> apOrig;
    const float* xPtr = nullptr;
    const float* yPtr = nullptr;
    float* apPtr = nullptr;
    const float* alphaPtr = nullptr;
};

CaseBuffers PrepareBuffers(const Sspr2Param& p)
{
    CaseBuffers b;
    b.x = sspr2_st::MakeStrided(p.core.n, p.core.incx, sspr2_st::StridedFill::Parse(p.core.xFill), p.core.seed);
    b.y = sspr2_st::MakeStrided(p.core.n, p.core.incy, sspr2_st::StridedFill::Parse(p.core.yFill), p.core.seed + 1);
    b.ap = sspr2_st::MakePacked(p.core.n, sspr2_st::StridedFill::Parse(p.core.apFill), p.core.seed + 2);
    b.apOrig = b.ap;
    b.xPtr = b.x.empty() ? nullptr : b.x.data();
    b.yPtr = b.y.empty() ? nullptr : b.y.data();
    b.apPtr = b.ap.empty() ? nullptr : b.ap.data();
    b.alphaPtr = p.core.alphaNull ? nullptr : &p.core.alpha;
    return b;
}

void PrintEvidence(const std::string& caseId, const std::string& ret, const std::string& check,
                   const std::string& result, const std::string& detail = "")
{
    std::cout << "EVIDENCE|" << caseId << "|ret=" << ret << "|check=" << check
              << "|result=" << result;
    if (!detail.empty()) {
        std::cout << "|detail=" << detail;
    }
    std::cout << std::endl;
}

} // namespace

class Sspr2Arch22Test : public BlasTest<Sspr2Param> {};

// ── TEST_F: handle=nullptr（CSV 无 handle 列，由测试工程构造） ──
TEST_F(Sspr2Arch22Test, NullHandle)
{
    float alpha = 1.0f;
    float x[16] = {0};
    float ap[136] = {0};
    aclblasStatus_t ret = aclblasSspr2_npu(
        nullptr, ACLBLAS_UPPER, 16, &alpha, x, 1, x, 1, ap);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
    PrintEvidence("SSPR2-L0-010-wrapper", "ACLBLAS_STATUS_HANDLE_IS_NULLPTR", "STATUS_ONLY", "PASS");
}

// 测试用例 CSV（19 列，与本文件同目录）
static std::vector<Sspr2Param> GetAllSspr2Cases()
{
    std::string dir;
    {
        std::string p(__FILE__);
        size_t s = p.rfind('/');
        if (s != std::string::npos) dir = p.substr(0, s + 1);
    }
    return GetCasesFromCsv<Sspr2Param>(dir + "sspr2_test_cases.csv");
}

INSTANTIATE_TEST_SUITE_P(
    Sspr2Arch22, Sspr2Arch22Test,
    ::testing::ValuesIn(GetAllSspr2Cases()),
    PrintCaseInfoString<Sspr2Param>);

TEST_P(Sspr2Arch22Test, CsvDriven)
{
    const Sspr2Param& p = GetParam();
    const auto& c = p.core;
    ASSERT_NE(c.check, sspr2_st::CheckKind::UNKNOWN) << "[" << c.caseId << "] unknown check kind";

    // FRAME_GUARD：装载层长度断言，不向算子下发
    if (c.check == sspr2_st::CheckKind::FRAME_GUARD) {
        auto it = FrameGuardCases().find(c.caseId);
        ASSERT_NE(it, FrameGuardCases().end()) << "[" << c.caseId << "] lacks guard metadata";
        for (const auto& kv : it->second) {
            const size_t expect = (kv.first == "ap") ? c.ApLen() : (kv.first == "x" ? c.XLen() : c.YLen());
            EXPECT_NE(kv.second, expect) << "[" << c.caseId << "] declared len unexpectedly matches";
        }
        PrintEvidence(c.caseId, "FRAME_GUARD(loader)", "FRAME_GUARD", "PASS");
        return;
    }

    CaseBuffers b = PrepareBuffers(p);
    // CSV 无 handle 列：expect=HANDLE_IS_NULLPTR 的用例（L0-010/L2-010）
    // 由测试工程构造 nullptr handle，其余用框架共享有效句柄。
    const bool expectNullHandle = (c.expectResult == "ACLBLAS_STATUS_HANDLE_IS_NULLPTR");
    const aclblasStatus_t ret = aclblasSspr2_npu(
        expectNullHandle ? nullptr : Sspr2Arch22Test::handle_, static_cast<aclblasFillMode_t>(c.uplo),
        c.n, b.alphaPtr, b.xPtr, c.incx, b.yPtr, c.incy, b.apPtr);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "[" << c.caseId << "] status mismatch: got " << sspr2_st::StatusName(ret);
    if (static_cast<int>(ret) != static_cast<int>(p.expectResult)) {
        PrintEvidence(c.caseId, sspr2_st::StatusName(ret), "STATUS", "FAIL");
        return;
    }

    const std::string retStr = sspr2_st::StatusName(ret);
    if (ret != ACLBLAS_STATUS_SUCCESS || c.check == sspr2_st::CheckKind::STATUS_ONLY) {
        PrintEvidence(c.caseId, retStr, "STATUS_ONLY", "PASS");
        return;
    }
    if (c.check == sspr2_st::CheckKind::NO_COMPUTE) {
        ASSERT_EQ(c.n, 0);
        PrintEvidence(c.caseId, retStr, "NO_COMPUTE", "PASS");
        return;
    }

    const size_t apLen = c.ApLen();
    if (c.check == sspr2_st::CheckKind::AP_UNCHANGED) {
        ASSERT_TRUE(sspr2_st::VerifyExactBits(b.apPtr, b.apOrig.data(), apLen))
            << "[" << c.caseId << "] AP changed under alpha=0 quick return";
        PrintEvidence(c.caseId, retStr, "AP_UNCHANGED", "PASS");
        return;
    }

    // golden：同输入调用内置 cblas_sspr2_ref（Netlib sspr2.f 语义）
    std::vector<float> golden = b.apOrig;
    Sspr2GoldenCompute(c.uplo, c.n, c.alphaNull ? 0.0f : c.alpha,
                       b.xPtr, c.incx, b.yPtr, c.incy, golden.data());
    auto r = sspr2_st::VerifyGoldenTolerance(b.apPtr, golden.data(), apLen, c.rtol, c.atol, 1e-2);
    ASSERT_TRUE(r.Pass(c.matchedRatio))
        << "[" << c.caseId << "] tol fail: mismatch=" << r.mismatchCount << " ratio=" << r.matchedRatio
        << " maxAbsErr=" << r.maxAbsErr;

    if (c.check == sspr2_st::CheckKind::NAN_PROPAGATE) {
        ASSERT_GT(sspr2_st::CountNan(b.apPtr, apLen), size_t{0});
        PrintEvidence(c.caseId, retStr, "NAN_PROPAGATE", "PASS");
        return;
    }
    PrintEvidence(c.caseId, retStr, "GOLDEN", "PASS");
}

// ═══ TC_PF 性能用例（sspr2_test.csv；环境变量 SSPR2_PF_MODE 切换，缺省 perf） ═══
//   perf   — 设备常驻数据：SetUpTestSuite 一次性生成并上传；单例 = warmup + 同步后
//            单次调用估计 est_us + 按预算折算采样数（samples = clamp(200us/est_us,
//            2, 64)；launch 异步，预算必须按同步后 kernel 耗时折算，否则采样队列
//            排水会吃掉整个单例 wall）+ 计时采样 + 1 次 stream 同步 → PERF avg_us；
//            无 D2H/golden/比对，仅状态断言。
//   verify — 单次功能执行（fill → wrapper H2D+kernel+sync+D2H → cblas golden）+
//            frame MERE/MARE 判据作 verdict（阈值按 CSV 列承接；kernel 更新链与
//            golden 逐位对齐，见 sspr2_golden.h）。--gtest_filter=*TC_PF* 选出本套件。

#include <chrono>
#include <cstdlib>

namespace {

struct Sspr2PfParam : public Sspr2Param {
    int pfIndex = -1; // 套件级设备常驻数据下标（GetPfCases 装载顺序）

    explicit Sspr2PfParam(const csv_map& map) : Sspr2Param(map)
    {
        core.level = "PF";
        // random_seed 列承接（19 列词表同义列为 seed；两者均落 core.seed）
        const std::string randomSeedStr = ReadMap(map, "random_seed", "");
        if (!randomSeedStr.empty()) {
            core.seed = parseUint(randomSeedStr);
            randomSeed = core.seed;
        }
    }
};

std::vector<Sspr2PfParam> GetPfCases()
{
    std::string dir;
    {
        std::string p(__FILE__);
        size_t s = p.rfind('/');
        if (s != std::string::npos) dir = p.substr(0, s + 1);
    }
    auto cases = GetCasesFromCsv<Sspr2PfParam>(dir + "sspr2_test.csv");
    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].pfIndex = static_cast<int>(i);
    }
    return cases;
}

bool PfVerifyMode()
{
    const char* m = std::getenv("SSPR2_PF_MODE");
    return (m != nullptr && std::string(m) == "verify");
}

// ── perf 模式：套件级设备常驻数据（与 GetPfCases 同序） ──
struct PfDeviceData {
    void* dX = nullptr;
    void* dY = nullptr;
    void* dAP = nullptr;
    size_t xBytes = 0;
    size_t yBytes = 0;
    size_t apBytes = 0;
    bool ready = false;
};

std::vector<PfDeviceData>& PfDeviceStore()
{
    static std::vector<PfDeviceData> store;
    return store;
}

// AMORT 均摊均值（除数保护：samples == 0 前置守卫 + std::max(1, samples) 包裹除数）。
// 可达输入下 samples 已被 clamp 至 [2, 64]（RunPfPerfCase 内 max(2, min(kMaxSamples,
// ...))），故 max(1, samples) == samples，与直接除法逐位一致。
double PfAmortUs(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1,
                 int samples)
{
    if (samples == 0) {
        return 0.0;
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0 /
        std::max(1, samples);
}

} // namespace

class Sspr2Arch22PfTest : public BlasTest<Sspr2PfParam> {
protected:
    static void SetUpTestSuite()
    {
        BlasTest<Sspr2PfParam>::SetUpTestSuite();
        auto& store = PfDeviceStore();
        if (PfVerifyMode() || !store.empty()) {
            return; // verify 模式无设备常驻数据；重复进入复用
        }
        const auto cases = GetPfCases();
        store.resize(cases.size());
        size_t readyCount = 0;
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto& c = cases[i].core;
            PfDeviceData& d = store[i];
            if (c.n <= 0) {
                continue;
            }
            const auto xf = sspr2_st::StridedFill::Parse(c.xFill);
            const auto yf = sspr2_st::StridedFill::Parse(c.yFill);
            const auto apf = sspr2_st::StridedFill::Parse(c.apFill);
            std::vector<float> x = sspr2_st::MakeStrided(c.n, c.incx, xf, c.seed);
            std::vector<float> y = sspr2_st::MakeStrided(c.n, c.incy, yf, c.seed + 1);
            std::vector<float> ap = sspr2_st::MakePacked(c.n, apf, c.seed + 2);
            d.xBytes = x.size() * sizeof(float);
            d.yBytes = y.size() * sizeof(float);
            d.apBytes = ap.size() * sizeof(float);
            if (d.xBytes == 0 || d.yBytes == 0 || d.apBytes == 0) {
                continue;
            }
            if (aclrtMalloc(&d.dX, d.xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
                aclrtMemcpy(d.dX, d.xBytes, x.data(), d.xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                std::cout << "[PF-SETUP] x upload failed at case " << i << std::endl;
                continue;
            }
            if (aclrtMalloc(&d.dY, d.yBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
                aclrtMemcpy(d.dY, d.yBytes, y.data(), d.yBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                std::cout << "[PF-SETUP] y upload failed at case " << i << std::endl;
                continue;
            }
            if (aclrtMalloc(&d.dAP, d.apBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
                aclrtMemcpy(d.dAP, d.apBytes, ap.data(), d.apBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                std::cout << "[PF-SETUP] ap upload failed at case " << i << std::endl;
                continue;
            }
            d.ready = true;
            ++readyCount;
        }
        std::cout << "[PF-SETUP] device-resident data ready: " << readyCount << "/" << cases.size()
                  << " cases (perf mode)" << std::endl;
    }

    static void TearDownTestSuite()
    {
        auto& store = PfDeviceStore();
        for (auto& d : store) {
            if (d.dX != nullptr) {
                aclrtFree(d.dX);
                d.dX = nullptr;
            }
            if (d.dY != nullptr) {
                aclrtFree(d.dY);
                d.dY = nullptr;
            }
            if (d.dAP != nullptr) {
                aclrtFree(d.dAP);
                d.dAP = nullptr;
            }
        }
        store.clear();
        BlasTest<Sspr2PfParam>::TearDownTestSuite();
    }

    // ── perf 模式兜底路径 ──
    // 负向用例（期望非 SUCCESS，如 null_alpha）同样走 wrapper 路径：perf 模式的
    // 设备常驻调用恒以非空标量地址传 alpha，空指针类 INVALID_VALUE 语义须由
    // PrepareBuffers 的 nullptr 语义承接，否则状态断言结构性不匹配。
    static void RunPfWrapperFallback(const Sspr2PfParam& p)
    {
        const auto& c = p.core;
        CaseBuffers b = PrepareBuffers(p);
        const aclblasStatus_t ret = aclblasSspr2_npu(
            Sspr2Arch22PfTest::handle_, static_cast<aclblasFillMode_t>(c.uplo),
            c.n, b.alphaPtr, b.xPtr, c.incx, b.yPtr, c.incy, b.apPtr);
        ASSERT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
            << "[" << c.caseId << "] status mismatch: got " << sspr2_st::StatusName(ret);
        PrintEvidence(c.caseId, sspr2_st::StatusName(ret), "PERF", "PASS", "mode=wrapper");
    }

    // ── 估计 + 计时采样 + PERF 上报（预算折算/采样循环/打印序固定） ──
    template <typename Call>
    static void PfTimedSample(const sspr2_st::Sspr2Case& c, const std::string& retName, Call call)
    {
        constexpr int kMaxSamples = 64;         // 采样上限 64
        constexpr double kWallBudgetUs = 200.0; // 单例 wall 预算（微秒）

        // 预热 #2 即估计调用（同步后单次 wall = kernel + 一次同步开销）：launch 为
        // 异步，预算必须按同步后的 kernel 耗时折算，否则采样队列排水（drain）会
        // 吃掉整个单例 wall。
        const auto w0 = std::chrono::steady_clock::now();
        (void)call();
        aclrtSynchronizeStream(Sspr2Arch22PfTest::stream_);
        const auto w1 = std::chrono::steady_clock::now();
        const double estUs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count() / 1000.0;

        int samples = static_cast<int>(kWallBudgetUs / std::max(estUs, 1.0));
        samples = std::max(2, std::min(kMaxSamples, samples));

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < samples; ++i) {
            (void)call();
        }
        aclrtSynchronizeStream(Sspr2Arch22PfTest::stream_);
        const auto t1 = std::chrono::steady_clock::now();
        const double avgUs = PfAmortUs(t0, t1, samples);
        std::cout << "[" << c.caseId << "] PERF avg_us=" << avgUs << " (samples=" << samples
                  << ", warmup=2, est_us=" << estUs << ", n=" << c.n << ", uplo="
                  << (c.uplo == 121 ? "UPPER" : "LOWER") << ")" << std::endl;
        PrintEvidence(c.caseId, retName, "PERF", "PASS",
                      std::string("avg_us=") + std::to_string(avgUs) + " samples=" + std::to_string(samples));
    }

    // ── perf 模式单例（设备常驻 → warmup → 计时采样 → 同步） ──
    static void RunPfPerfCase(const Sspr2PfParam& p)
    {
        const auto& c = p.core;
        const float alphaHost = c.alphaNull ? 0.0f : c.alpha;
        auto call = [&]() {
            return aclblasSspr2(Sspr2Arch22PfTest::handle_, static_cast<aclblasFillMode_t>(c.uplo),
                c.n, &alphaHost, static_cast<const float*>(PfDeviceStore()[static_cast<size_t>(p.pfIndex)].dX),
                c.incx, static_cast<float*>(PfDeviceStore()[static_cast<size_t>(p.pfIndex)].dY), c.incy,
                static_cast<float*>(PfDeviceStore()[static_cast<size_t>(p.pfIndex)].dAP));
        };
        const auto& store = PfDeviceStore();
        const bool resident = (p.pfIndex >= 0) &&
                              (static_cast<size_t>(p.pfIndex) < store.size()) &&
                              store[static_cast<size_t>(p.pfIndex)].ready;
        // 兜底：设备数据缺失（分配失败/n<=0）或负向期望 → wrapper 单次调用，仅状态断言
        if (!resident ||
            static_cast<int>(p.expectResult) != static_cast<int>(ACLBLAS_STATUS_SUCCESS)) {
            RunPfWrapperFallback(p);
            return;
        }

        aclblasStatus_t ret = call();
        ASSERT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
            << "[" << c.caseId << "] status mismatch: got " << sspr2_st::StatusName(ret);
        if (static_cast<int>(ret) != static_cast<int>(p.expectResult)) {
            PrintEvidence(c.caseId, sspr2_st::StatusName(ret), "STATUS", "FAIL");
            return;
        }
        PfTimedSample(c, sspr2_st::StatusName(ret), call);
    }

    // ── verify 模式单例：完整功能执行 + frame MERE/MARE verdict ──
    static void RunPfVerifyCase(const Sspr2PfParam& p)
    {
        const auto& c = p.core;
        CaseBuffers b = PrepareBuffers(p);
        const aclblasStatus_t ret = aclblasSspr2_npu(
            Sspr2Arch22PfTest::handle_, static_cast<aclblasFillMode_t>(c.uplo),
            c.n, b.alphaPtr, b.xPtr, c.incx, b.yPtr, c.incy, b.apPtr);
        ASSERT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
            << "[" << c.caseId << "] status mismatch: got " << sspr2_st::StatusName(ret);
        if (static_cast<int>(ret) != static_cast<int>(p.expectResult)) {
            PrintEvidence(c.caseId, sspr2_st::StatusName(ret), "STATUS", "FAIL");
            return;
        }
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            PrintEvidence(c.caseId, sspr2_st::StatusName(ret), "STATUS_ONLY", "PASS");
            return;
        }

        std::vector<float> golden = b.apOrig;
        Sspr2GoldenCompute(c.uplo, c.n, c.alphaNull ? 0.0f : c.alpha,
                           b.xPtr, c.incx, b.yPtr, c.incy, golden.data());
        const size_t apLen = c.ApLen();

        // ── frame MERE/MARE verdict（test/frame/verify.h MereMareStrategy 语义，
        //    mere_threshold=2^-13 / mare_multiplier=10.0 按 CSV 列承接）──
        VerifyConfig cfg;
        cfg.mode = PrecisionMode::MERE_MARE;
        cfg.mereThreshold = (p.mereThreshold > 0.0) ? p.mereThreshold : (1.0 / 8192.0);
        cfg.mareMultiplier = (p.mareMultiplier > 0.0) ? p.mareMultiplier : 10.0;
        const bool mereOk = Verifier::verifyVector(b.apPtr, golden.data(), apLen, 1, cfg, c.caseId);

        // 19 列 GOLDEN 双指标容差：仅证据行（verdict 由 MERE/MARE 承担）
        auto r = sspr2_st::VerifyGoldenTolerance(b.apPtr, golden.data(), apLen, c.rtol, c.atol, 1e-2);
        PrintEvidence(c.caseId, sspr2_st::StatusName(ret), "GOLDEN_TOL",
                      r.Pass(c.matchedRatio) ? "PASS" : "FAIL",
                      "mismatch=" + std::to_string(r.mismatchCount) +
                          " maxAbsErr=" + std::to_string(r.maxAbsErr));

        ASSERT_TRUE(mereOk) << "[" << c.caseId << "] frame MERE/MARE verdict failed";
        PrintEvidence(c.caseId, sspr2_st::StatusName(ret), "MERE_MARE", "PASS");
    }
};

INSTANTIATE_TEST_SUITE_P(
    Sspr2Pf, Sspr2Arch22PfTest,
    ::testing::ValuesIn(GetPfCases()),
    PrintCaseInfoString<Sspr2PfParam>);

TEST_P(Sspr2Arch22PfTest, PerfDriven)
{
    const Sspr2PfParam& p = GetParam();
    if (PfVerifyMode()) {
        RunPfVerifyCase(p);
        return;
    }
    RunPfPerfCase(p);
}
