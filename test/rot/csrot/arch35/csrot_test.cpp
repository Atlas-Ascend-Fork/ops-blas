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
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "csrot_param.h"
#include "csrot_golden.h"
#include "csrot_npu_wrapper.h"

namespace {

// x / y 独立取样：y 在 x 的 randomSeed 上加固定偏移（同仓 cdotc_test 口径），
// 避免两向量逐元素相同——否则算子把 x/y 读写对象颠倒（swap 类缺陷）时，
// 完全相同的输入会令 golden 比对失去检出能力。
constexpr uint32_t CSROT_Y_SEED_OFFSET = 2000U;

// 复数 buffer 生成：fill.h 仅提供 float 生成器，复数按交错 float lane 生成——
// 生成 n 个复数的 2n 个 float lane（实部/虚部独立填充，与 gen_csv.py 设计一致），
// 再按 stride 语义散布到物理缓冲区：
//   |inc|<=1（含 inc==0）：span = n，数据连续排布（inc==0 时全部落在 element 0，
//                          与 Netlib/cblas 零步长"只旋转 element 0"一致）；
//   |inc|>1：span = (n-1)*|inc|+1，逻辑第 i 个元素落在物理 i*|inc|；
//   负步长：逻辑第 i 个元素落在物理 (n-1-i)*|inc|（从尾端起算）。
// x/y 列为 NULLPTR（BlasFillMode::M_NULLPTR）时返回空 vector，调用方传 nullptr。
std::vector<aclblasComplex> MakeCsrotBuffer(int n, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (n <= 0 || fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    const int64_t aInc = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    const int64_t span = (aInc <= 1) ? static_cast<int64_t>(n) : static_cast<int64_t>(n - 1) * aInc + 1;
    // 极端步长溢出用例（(n-1)*|inc| 超出 int32）：算子会在校验阶段以 INVALID_VALUE 拒绝，
    // 永不访问 x/y；此处返回极小占位 buffer（避免 ~16GB 分配 / std::vector 溢出），数据不参与比对。
    if (span <= 0 || span > static_cast<int64_t>(INT32_MAX)) {
        return std::vector<aclblasComplex>(static_cast<size_t>(n > 0 ? n : 1), aclblasComplex{0.0f, 0.0f});
    }
    std::vector<float> lanes = makeBlasArray(static_cast<int64_t>(n) * 2, fill, seed);
    std::vector<aclblasComplex> buf(static_cast<size_t>(span));
    for (int i = 0; i < n; ++i) {
        const int64_t idx = (inc < 0) ? static_cast<int64_t>(n - 1 - i) * aInc : static_cast<int64_t>(i) * aInc;
        buf[static_cast<size_t>(idx)] = aclblasComplex{lanes[2 * i], lanes[2 * i + 1]};
    }
    return buf;
}

// 逻辑第 i 个元素的物理下标（与算子 / Netlib 语义一致）
inline int64_t CsrotPhysIndex(int i, int n, int inc)
{
    const int absInc = (inc < 0) ? -inc : inc;
    return (inc < 0) ? static_cast<int64_t>(n - 1 - i) * absInc : static_cast<int64_t>(i) * absInc;
}

bool ContainsNaN(const aclblasComplex* buf, int n, int inc)
{
    for (int i = 0; i < n; ++i) {
        const aclblasComplex& e = buf[CsrotPhysIndex(i, n, inc)];
        if (std::isnan(e.real) || std::isnan(e.imag)) {
            return true;
        }
    }
    return false;
}

bool ContainsInf(const aclblasComplex* buf, int n, int inc)
{
    for (int i = 0; i < n; ++i) {
        const aclblasComplex& e = buf[CsrotPhysIndex(i, n, inc)];
        if (std::isinf(e.real) || std::isinf(e.imag)) {
            return true;
        }
    }
    return false;
}

// 按逻辑顺序抽取 touched 元素（stride / 负步长已归一），供复数验证接口使用
std::vector<aclblasComplex> ExtractLogical(const aclblasComplex* buf, int n, int inc)
{
    std::vector<aclblasComplex> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = buf[CsrotPhysIndex(i, n, inc)];
    }
    return out;
}

bool ExactEqual(const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b)
{
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].real != b[i].real || a[i].imag != b[i].imag) {
            return false;
        }
    }
    return true;
}

// NaN 传播逐元素比对：NaN 按同一下标在 x/y 间传播，与 golden 一致即通过
// （Inf 相减产生的 NaN 同样必须与 golden 一致，否则形成"输出全 NaN 即通过"的校验空洞）
void CheckCsrotNanPropagation(
    const CsrotParam& p, const aclblasComplex* xPtr, const aclblasComplex* yPtr,
    const std::vector<aclblasComplex>& goldenX, const std::vector<aclblasComplex>& goldenY)
{
    bool matchGolden = true;
    for (int i = 0; i < p.n; ++i) {
        const aclblasComplex& xe = xPtr[CsrotPhysIndex(i, p.n, p.incx)];
        const aclblasComplex& ye = yPtr[CsrotPhysIndex(i, p.n, p.incy)];
        const aclblasComplex& gx = goldenX[CsrotPhysIndex(i, p.n, p.incx)];
        const aclblasComplex& gy = goldenY[CsrotPhysIndex(i, p.n, p.incy)];
        auto nanEq = [](float a, float b) { return (std::isnan(a) && std::isnan(b)) || a == b; };
        if (!nanEq(xe.real, gx.real) || !nanEq(xe.imag, gx.imag) || !nanEq(ye.real, gy.real) ||
            !nanEq(ye.imag, gy.imag)) {
            matchGolden = false;
        }
    }
    std::cout << "[" << p.caseName << "] NaN-propagation check: matchGolden=" << matchGolden << std::endl;
    EXPECT_TRUE(matchGolden);
}

// 单位旋转不短路：x' = x, y' = y，与 golden 逐位相等
void CheckCsrotUnitRotation(
    const CsrotParam& p, const aclblasComplex* xPtr, const aclblasComplex* yPtr,
    const std::vector<aclblasComplex>& goldenX, const std::vector<aclblasComplex>& goldenY)
{
    EXPECT_TRUE(ExactEqual(ExtractLogical(xPtr, p.n, p.incx), ExtractLogical(goldenX.data(), p.n, p.incx)));
    EXPECT_TRUE(ExactEqual(ExtractLogical(yPtr, p.n, p.incy), ExtractLogical(goldenY.data(), p.n, p.incy)));
}

} // namespace

class CsrotTest : public BlasTest<CsrotParam> {};

TEST_F(CsrotTest, NullHandle)
{
    EXPECT_EQ(aclblasCsrot_npu(nullptr, 5, nullptr, 1, nullptr, 1, 0.6f, 0.8f), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsrotTest, NullHandleTakesPrecedenceOverQuickReturn)
{
    EXPECT_EQ(aclblasCsrot(nullptr, 0, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsrotTest, QuickReturnSkipsDataPointerValidation)
{
    EXPECT_EQ(aclblasCsrot(CsrotTest::handle_, 0, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasCsrot(CsrotTest::handle_, -1, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    Csrot, CsrotTest, ::testing::ValuesIn(GetCasesFromCsv<CsrotParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CsrotParam>);

TEST_P(CsrotTest, CsvDriven)
{
    const CsrotParam& p = GetParam();

    // Step 1: host 数据（x/y 同时是输入与输出，原地修改；y 的 seed 加固定偏移使两向量独立）
    std::vector<aclblasComplex> xHost = MakeCsrotBuffer(p.n, p.incx, p.x, p.randomSeed);
    std::vector<aclblasComplex> yHost = MakeCsrotBuffer(p.n, p.incy, p.y, p.randomSeed + CSROT_Y_SEED_OFFSET);
    aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    aclblasComplex* yPtr = yHost.empty() ? nullptr : yHost.data();
    const float* cArg = p.cIsNull ? nullptr : &p.c;
    const float* sArg = p.sIsNull ? nullptr : &p.s;

    // Step 2: NPU 执行（cs_ptr_mode 控制 c/s 指针落地位置，算子自动判定每个指针位置；
    //         cIsNull/sIsNull 透传 c/s 空指针负向用例）
    aclblasStatus_t ret = aclblasCsrot_npu(
        CsrotTest::handle_, p.n, xPtr, p.incx, yPtr, p.incy, p.c, p.s, p.csPtrMode, p.cIsNull, p.sIsNull);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != p.expectResult || p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return; // 负向用例：只校验返回码
    }
    if (p.n <= 0) {
        return; // n<=0 为 no-op，x/y 不被修改
    }
    // 性能用例（TC_PF_ 前缀）仅做下发 smoke + 返回码校验，不做 golden 比对：
    // 避免 200 条大 n（最大 4M）用例各自再生成两份 4M golden 把精度套件放大一个
    // 数量级；与 kernel 级计时（csrot_benchmark）职责分离。大尺寸正确性由精度套件
    // TC_SQ（最大 1M，已覆盖多核 tiling）承担；精度回归可用 GTEST_FILTER='-*TC_PF*' 单独跑。
    if (p.caseName.rfind("TC_PF", 0) == 0) {
        return;
    }

    // Step 3: CPU golden（对原始数据的新鲜拷贝执行，netlib csrot.f 公式逐通道独立计算）
    std::vector<aclblasComplex> goldenX = MakeCsrotBuffer(p.n, p.incx, p.x, p.randomSeed);
    std::vector<aclblasComplex> goldenY = MakeCsrotBuffer(p.n, p.incy, p.y, p.randomSeed + CSROT_Y_SEED_OFFSET);
    ASSERT_EQ(
        aclblasCsrot_cpu(CsrotTest::handle_, p.n, goldenX.data(), p.incx, goldenY.data(), p.incy, cArg, sArg),
        ACLBLAS_STATUS_SUCCESS);

    // Step 4: 特殊值分支
    if (ContainsNaN(xPtr, p.n, p.incx) || ContainsNaN(yPtr, p.n, p.incy)) {
        CheckCsrotNanPropagation(p, xPtr, yPtr, goldenX, goldenY);
        return;
    }

    if (p.c == 1.0f && p.s == 0.0f && (ContainsInf(xPtr, p.n, p.incx) || ContainsInf(yPtr, p.n, p.incy))) {
        CheckCsrotUnitRotation(p, xPtr, yPtr, goldenX, goldenY);
        return;
    }

    // Step 5: 复数精度验证——实部/虚部分开统计（CSV 列 mere_threshold=2^-13,
    // mare_multiplier=10；框架内 FP32 生态口径，Inf/NaN 等值元素自动跳过）
    std::vector<aclblasComplex> outX = ExtractLogical(xPtr, p.n, p.incx);
    std::vector<aclblasComplex> goldX = ExtractLogical(goldenX.data(), p.n, p.incx);
    std::vector<aclblasComplex> outY = ExtractLogical(yPtr, p.n, p.incy);
    std::vector<aclblasComplex> goldY = ExtractLogical(goldenY.data(), p.n, p.incy);
    std::cout << std::setprecision(9);
    EXPECT_TRUE(Verifier::verifyMereMareComplexFloat(
        outX.data(), goldX.data(), static_cast<size_t>(p.n), p.mereThreshold, p.mareMultiplier, 0.0,
        p.caseName + "_x"));
    EXPECT_TRUE(Verifier::verifyMereMareComplexFloat(
        outY.data(), goldY.data(), static_cast<size_t>(p.n), p.mereThreshold, p.mareMultiplier, 0.0,
        p.caseName + "_y"));
}

// 用例构成断言：防止"替换式插入"静默删减任务书原始 1200 条验收用例。
// 任务书 §3.5 要求执行原始 CSV 全部 1200 条；本 CSV 仅在尾部追加 6 条 TC_EXS 补充，
// 总数 1206 = 1200 任务（含 TC_PF 200）+ 6 补充。任何"删除原始用例以腾位"的改动都会
// 使 total / TC_EXS / TC_PF 计数偏离，从而被 CI 捕获。
TEST_F(CsrotTest, CsvCompositionAssert)
{
    const auto cases = GetCasesFromCsv<CsrotParam>(ReplaceFileExtension2Csv(__FILE__));
    int total = static_cast<int>(cases.size());
    int pf = 0, exs = 0;
    for (const auto& c : cases) {
        if (c.caseName.rfind("TC_PF", 0) == 0) {
            ++pf;
        } else if (c.caseName.rfind("TC_EXS", 0) == 0) {
            ++exs;
        }
    }
    EXPECT_EQ(total, 1206) << "CSV 总数应为 1206（1200 任务 + 6 补充）";
    EXPECT_EQ(pf, 200) << "TC_PF 性能用例应为 200";
    EXPECT_EQ(exs, 6) << "TC_EXS 补充用例应为 6";
    EXPECT_EQ(total - pf - exs, 1000) << "原始 1000 条精度任务用例必须一条不少";
}
