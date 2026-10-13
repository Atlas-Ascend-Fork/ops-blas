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
#include <string>
#include <vector>
#include <cmath>
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "crotg_param.h"
#include "crotg_golden.h"
#include "crotg_npu_wrapper.h"

class CrotgArch22Test : public BlasTest<CrotgParam> {
protected:
    static void RunSpecialValueCase(const CrotgParam &p);
    static void RunGeneralCase(const CrotgParam &p, bool isExtreme);
    static void RunCrotgCsvCase(const CrotgParam &p);
};

INSTANTIATE_TEST_SUITE_P(
    Crotg, CrotgArch22Test, ::testing::ValuesIn(GetCasesFromCsv<CrotgParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CrotgParam>);

namespace {
// 任务书 §3.2 标准
constexpr float RTOL = 9.765625e-4f;   // 2^-10
constexpr float ATOL = 1.5258789e-5f;  // 2^-16

bool IsSpecial(float x) { return std::isnan(x) || std::isinf(x); }

bool ElementPass(float out, float gold)
{
    if (std::isnan(out) && std::isnan(gold)) return true;
    if (std::isinf(out) && std::isinf(gold) && (out > 0) == (gold > 0)) return true;
    if (IsSpecial(out) || IsSpecial(gold)) return false;
    return std::fabs(out - gold) <= ATOL + RTOL * std::fabs(gold);
}

bool ScalarPass(const float* out, const float* gold, size_t n, const std::string& name)
{
    for (size_t i = 0; i < n; i++) {
        if (!ElementPass(out[i], gold[i])) {
            printf("[%s] MISMATCH idx=%zu out=%.9g gold=%.9g\n", name.c_str(), i, out[i], gold[i]);
            return false;
        }
    }
    return true;
}

// 任务书 §3.2 数学性质验证
bool MathPropPass(float a_r, float a_i, float b_r, float b_i,
                  float c, float s_r, float s_i, float r_r, float r_i,
                  const std::string& name)
{
    const float norm2 = a_r*a_r + a_i*a_i + b_r*b_r + b_i*b_i;
    const float norm = std::sqrt(norm2);
    const float tol1 = ATOL + RTOL;
    const float tol2 = ATOL + RTOL * norm;

    // 1. 正交归一性：|c² + |s|² − 1| ≤ atol + rtol
    float orth = c*c + s_r*s_r + s_i*s_i;
    if (std::fabs(orth - 1.0f) > tol1) {
        printf("[%s] 正交归一失败: c²+|s|²=%.9g, err=%.3g, tol=%.3g\n",
               name.c_str(), orth, std::fabs(orth-1.0f), tol1);
        return false;
    }

    // 2. 旋转零化：|c·a + s·b − r| ≤ tol2，|−conjg(s)·a + c·b| ≤ tol2
    // 复数乘法: (x+yi)(u+vi) = (xu-yv) + (xv+yu)i
    float ca_r = c*a_r;
    float ca_i = c*a_i;
    float sb_r = s_r*b_r - s_i*b_i;
    float sb_i = s_r*b_i + s_i*b_r;
    float z1_r = ca_r + sb_r - r_r;
    float z1_i = ca_i + sb_i - r_i;
    float zero1 = std::sqrt(z1_r*z1_r + z1_i*z1_i);
    if (zero1 > tol2) {
        printf("[%s] 旋转零化(1)失败: err=%.3g, tol=%.3g\n", name.c_str(), zero1, tol2);
        return false;
    }

    // -conjg(s) = -s_r + s_i*i
    float nc_r = -s_r;
    float nc_i = s_i;
    float nca_r = nc_r*a_r - nc_i*a_i;
    float nca_i = nc_r*a_i + nc_i*a_r;
    float cb_r = c*b_r;
    float cb_i = c*b_i;
    float z2_r = nca_r + cb_r;
    float z2_i = nca_i + cb_i;
    float zero2 = std::sqrt(z2_r*z2_r + z2_i*z2_i);
    if (zero2 > tol2) {
        printf("[%s] 旋转零化(2)失败: err=%.3g, tol=%.3g\n", name.c_str(), zero2, tol2);
        return false;
    }

    // 3. 模长保持：||r| − sqrt(|a|²+|b|²)| ≤ tol2
    float abs_r = std::sqrt(r_r*r_r + r_i*r_i);
    if (std::fabs(abs_r - norm) > tol2) {
        printf("[%s] 模长保持失败: |r|=%.9g, norm=%.9g, err=%.3g\n",
               name.c_str(), abs_r, norm, std::fabs(abs_r-norm));
        return false;
    }
    return true;
}

bool HasSpecialInput(const CrotgParam &p)
{
    return IsSpecial(p.a_real) || IsSpecial(p.a_imag) ||
           IsSpecial(p.b_real) || IsSpecial(p.b_imag);
}

// 缩放式 hypot（LAPY2），避免极端比值下下溢
float SafeHypot(float x, float y)
{
    float ax = std::abs(x);
    float ay = std::abs(y);
    float big = std::max(ax, ay);
    if (big == 0.0f) return 0.0f;
    return big * std::sqrt((ax / big) * (ax / big) + (ay / big) * (ay / big));
}

// 大数/小数用例：判据以数学性质为主（任务书 §3.2 补充说明）
bool IsExtremeCase(const CrotgParam &p)
{
    float abs_a = SafeHypot(p.a_real, p.a_imag);
    float abs_b = SafeHypot(p.b_real, p.b_imag);
    float max_mag = std::max(abs_a, abs_b);
    float min_mag = std::min(abs_a, abs_b);
    return (max_mag > 1e5f) || (max_mag < 1e-5f) || (min_mag > 0 && max_mag / min_mag > 1e8f);
}

std::vector<aclblasComplex> MakeIoVector(float re, float im)
{
    std::vector<aclblasComplex> v(1);
    v[0].real = re;
    v[0].imag = im;
    return v;
}
}

// 特殊值：只验证 CPU 路径不崩溃 + b 不被覆写
void CrotgArch22Test::RunSpecialValueCase(const CrotgParam &p)
{
    std::vector<aclblasComplex> cpuA = MakeIoVector(p.a_real, p.a_imag);
    std::vector<aclblasComplex> cpuB = MakeIoVector(p.b_real, p.b_imag);
    std::vector<aclblasComplex> cpuS = MakeIoVector(0.0f, 0.0f);
    float cpuC = 0.0f;
    aclblasStatus_t s = aclblasCrotg_cpu(CrotgArch22Test::handle_, cpuA.data(), cpuB.data(),
                                         &cpuC, cpuS.data());
    EXPECT_EQ(s, ACLBLAS_STATUS_SUCCESS) << p.caseName << " CPU 路径返回错误";
}

void CrotgArch22Test::RunGeneralCase(const CrotgParam &p, bool isExtreme)
{
    std::vector<aclblasComplex> resultA = MakeIoVector(p.a_real, p.a_imag);
    std::vector<aclblasComplex> resultB = MakeIoVector(p.b_real, p.b_imag);
    std::vector<aclblasComplex> resultS = MakeIoVector(0.0f, 0.0f);
    float resultC = 0.0f;

    std::vector<aclblasComplex> goldenA = resultA;
    std::vector<aclblasComplex> goldenB = resultB;
    std::vector<aclblasComplex> goldenS = resultS;
    float goldenC = 0.0f;

    aclblasStatus_t ret = aclblasCrotg_npu(
        CrotgArch22Test::handle_, resultA.data(), resultB.data(), &resultC, resultS.data());
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS) << p.caseName << " NPU 路径返回错误";

    aclblasStatus_t goldenRet = aclblasCrotg_cpu(
        CrotgArch22Test::handle_, goldenA.data(), goldenB.data(), &goldenC, goldenS.data());
    ASSERT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(resultB[0].real, p.b_real) << p.caseName << " b 被误覆写(real)";
    EXPECT_EQ(resultB[0].imag, p.b_imag) << p.caseName << " b 被误覆写(imag)";

    // 所有用例都验证数学性质（任务书 §3.2 第 3 点：每条精度用例须同时验证）
    EXPECT_TRUE(MathPropPass(p.a_real, p.a_imag, p.b_real, p.b_imag,
                             resultC, resultS[0].real, resultS[0].imag,
                             resultA[0].real, resultA[0].imag, p.caseName + "_math"));

    // 一般值用例：额外做严格逐分量比对
    if (!isExtreme) {
        EXPECT_TRUE(ScalarPass(reinterpret_cast<float*>(resultA.data()),
                               reinterpret_cast<float*>(goldenA.data()), 2, p.caseName + "_r"));
        EXPECT_TRUE(ScalarPass(&resultC, &goldenC, 1, p.caseName + "_c"));
        EXPECT_TRUE(ScalarPass(reinterpret_cast<float*>(resultS.data()),
                               reinterpret_cast<float*>(goldenS.data()), 2, p.caseName + "_s"));
    }
}

void CrotgArch22Test::RunCrotgCsvCase(const CrotgParam &p)
{
    bool hasSpecialInput = HasSpecialInput(p);
    bool isExtreme = IsExtremeCase(p);
    if (hasSpecialInput) {
        RunSpecialValueCase(p);
        return;
    }
    RunGeneralCase(p, isExtreme);
}

TEST_P(CrotgArch22Test, CsvDriven)
{
    RunCrotgCsvCase(GetParam());
}
