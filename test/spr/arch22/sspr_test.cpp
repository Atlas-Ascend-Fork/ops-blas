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
#include <cstdio>
#include <fstream>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sspr_param.h"
#include "sspr_golden.h"
#include "sspr_npu_wrapper.h"

/**
 * aclblasSspr（arch22 / Ascend910B3）ST 测试工程。
 *
 * - CSV 驱动：sspr_test.csv，GTest 参数化逐条执行；TC_PF 性能档用例逻辑
 *   与精度档一致（仅不比对精度、关注耗时）。
 * - golden：cblas（Netlib sspr，test/spr/sspr_golden.h -> aclblasSspr_cpu），
 *   AP 全数组验证（仅 uplo 三角对应打包位置参与，另一三角不存于 AP）。
 * - 精度口径：alpha=0 quick return EXACT（AP 与输入逐位一致，x 含 NaN/Inf 不影响）；
 *   其余按 CSV 双口径 mere_threshold=2^-13 / mare_multiplier=10.0（MERE/MARE 策略），
 *   CSV 缺省时回退 frame MIXED_TOLERANCE（rtol 2^-10 / atol 2^-16 / ratio 0.99 / 1e-2 或 32ULP）。
 */

class SsprArch22Test : public BlasTest<SsprParam> {};

TEST_F(SsprArch22Test, NullHandle)
{
    float alphaVal = 1.0f;
    aclblasStatus_t ret = aclblasSspr_npu(nullptr, ACLBLAS_UPPER, 4, &alphaVal, nullptr, 1, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// 校验 CSV 全量行均可被 ReadMap 解析（GetCasesFromCsv 会静默跳过字段数不足的行，
// 此处用文件数据行数 vs 加载条数比对，防止 CSV 行被静默丢弃）。
TEST_F(SsprArch22Test, CsvRowCountSanity)
{
    const std::string csvPath = ReplaceFileExtension2Csv(__FILE__);
    auto cases = GetCasesFromCsv<SsprParam>(csvPath);

    std::ifstream ifs(csvPath);
    ASSERT_TRUE(ifs.is_open()) << "Cannot open CSV: " << csvPath;
    size_t dataLines = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        dataLines++;
    }
    ASSERT_GE(dataLines, static_cast<size_t>(1)); // 扣除表头
    dataLines -= 1;

    EXPECT_EQ(cases.size(), dataLines)
        << "CSV rows silently skipped during parsing (field count mismatch?).";
}

INSTANTIATE_TEST_SUITE_P(
    Sspr, SsprArch22Test, ::testing::ValuesIn(GetCasesFromCsv<SsprParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsprParam>);

namespace {

// AP old-value construction from the CSV param.
std::vector<float> MakeApHost(const SsprParam& p)
{
    const bool useUpper = (p.uplo == ACLBLAS_UPPER);
    std::vector<float> apHost;
    if (p.ap.pattern == BlasFillMode::P_UPPER || p.ap.pattern == BlasFillMode::P_LOWER) {
        apHost = makeBlasPacked(p.n, p.ap, p.randomSeed);
    } else {
        apHost = makeBlasTriangular(p.n, useUpper, p.ap, p.randomSeed);
    }
    return apHost;
}

// Precision verification; the three caliber branches and their order are fixed.
void VerifyCsvResult(const SsprParam& p, const float* apPtr, const std::vector<float>& golden,
                     size_t apLen)
{
    VerifyConfig cfg;
    if (p.alpha == 0.0f && !p.alphaNull) {
        // alpha=0 quick return：invariant alpha_zero_identity，EXACT 逐位比对
        cfg.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(apPtr, golden.data(), apLen, 1, cfg, p.caseName));
    } else if (p.mereThreshold > 0.0 && p.mareMultiplier > 0.0) {
        // 测试工程双口径（CSV 列配置）：MERE < 2^-13 且 MARE < 10 * threshold
        cfg.mode = PrecisionMode::MERE_MARE;
        cfg.mereThreshold = p.mereThreshold;
        cfg.mareMultiplier = p.mareMultiplier;
        EXPECT_TRUE(Verifier::verifyVector(apPtr, golden.data(), apLen, 1, cfg, p.caseName));
    } else {
        // frame 缺省：MIXED_TOLERANCE（rtol 2^-10 / atol 2^-16 / ratio 0.99 / 1e-2 或 32ULP）
        applyMixedTolerance(cfg, ACL_FLOAT, golden.data(), apLen);
        EXPECT_TRUE(Verifier::verifyVector(apPtr, golden.data(), apLen, 1, cfg, p.caseName));
    }
}

} // namespace

TEST_P(SsprArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    const aclblasHandle_t handle = p.nullHandle ? nullptr : SsprArch22Test::handle_;

    // incx = INT_MIN：物理长度 1+(n-1)*|incx| 溢出，校验序中 incx 检查先于指针解引用，
    // 直接以空指针调用，期望 INVALID_VALUE。
    if (p.incx == INT_MIN) {
        const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
        aclblasStatus_t ret = aclblasSspr_npu(handle, p.uplo, p.n, alphaPtr, nullptr, p.incx, nullptr);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    const int absIncx = std::abs(p.incx);
    const size_t xLen = (p.n > 0) ? static_cast<size_t>((p.n - 1) * absIncx + 1) : 0;
    const size_t apLen = (p.n > 0) ? static_cast<size_t>(p.n) * (p.n + 1) / 2 : 0;

    // x：跨步向量（支持负步长反向遍历）；NULLPTR 填充返回空向量 -> 空指针
    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);

    // AP：打包三角旧值；若 fill 自带 UPPER/LOWER 打包 pattern 则按其布局
    std::vector<float> apHost = MakeApHost(p);
    std::vector<float> apOrig = apHost;

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* apPtr = apHost.empty() ? nullptr : apHost.data();
    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;

    aclblasStatus_t ret = aclblasSspr_npu(handle, p.uplo, p.n, alphaPtr, xPtr, p.incx, apPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (p.n == 0) {
        return; // quick return：AP / x 逐位不变（缓冲未被触碰）
    }

    // CPU golden：cblas（Netlib sspr），从 AP 旧值出发原地更新
    std::vector<float> golden = apOrig;
    if (!p.alphaNull && !xHost.empty() && !golden.empty()) {
        aclblasSspr_cpu(SsprArch22Test::handle_, p.uplo, p.n, &p.alpha, xHost.data(), p.incx, golden.data());
    }

    VerifyCsvResult(p, apPtr, golden, apLen);
}
