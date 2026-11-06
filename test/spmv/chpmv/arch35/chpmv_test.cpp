/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "blas_test.h"
#include "../chpmv_param.h"
#include "../chpmv_golden.h"
#include "../chpmv_test_utils.h"
#include "chpmv_npu_wrapper.h"

using chpmv_test::Complex;
using chpmv_test::Param;

class ChpmvArch35Test : public BlasTest<Param> {
protected:
    static void SetUpTestSuite()
    {
        BlasTest<Param>::SetUpTestSuite();
        ASSERT_NO_THROW((void)chpmv_test::Oracle::Instance());
    }
};

INSTANTIATE_TEST_SUITE_P(
    Chpmv, ChpmvArch35Test, ::testing::ValuesIn(chpmv_test::Cases(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Param>);

namespace {
struct StreamRestore {
    aclblasHandle_t handle;
    aclrtStream original, alternate;
    ~StreamRestore()
    {
        aclrtSynchronizeStream(alternate);
        aclblasSetStream(handle, original);
        aclrtDestroyStream(alternate);
    }
};

void ConfigureNumericalInput(const Param& p, std::vector<Complex>& ap, std::vector<Complex>& x)
{
    using namespace chpmv_test;
    size_t requiredAp = p.numericMode != 0 ? 1U : 0U;
    size_t requiredX = requiredAp;
    if (p.numericMode >= 3) {
        requiredAp = 2;
    }
    if (p.numericMode >= 5) {
        requiredAp = p.uplo == ACLBLAS_UPPER ? PackedCount(64) + 1 : 65;
        requiredX = 65;
    }
    ASSERT_GE(ap.size(), requiredAp);
    ASSERT_GE(x.size(), requiredX);
    if (p.numericMode != 0) {
        ap[0].real = p.numericMode == 1 ? 1e6f : 1e38f;
        const float small =
            p.numericMode == 1 ? std::numeric_limits<float>::denorm_min() : std::numeric_limits<float>::min();
        x[0] = {small, -small};
        if (p.numericMode >= 3) {
            ap[0].real = 0.0f;
            ap[1] = {1e6f, 0};
            const float subnormal = std::numeric_limits<float>::denorm_min();
            x[0] = {subnormal, -subnormal};
            if (p.numericMode >= 5) {
                const size_t index = p.uplo == ACLBLAS_UPPER ? PackedCount(64) : 64;
                ap[1] = {0, 0};
                ap[index] = {1e6f, 0};
                if (p.uplo == ACLBLAS_LOWER) {
                    x[0] = {0, 0};
                    x[64] = {subnormal, -subnormal};
                }
            }
        }
        std::cout << "NUMERICAL mode=" << p.numericMode << " diagonal=" << ap[0].real << " x=" << std::hexfloat
                  << x[0].real << ',' << x[0].imag << std::defaultfloat << '\n';
    }
}

void VerifyOutput(const Param& p, const std::vector<Complex>& actual, const std::vector<Complex>& golden)
{
    for (bool imaginary : {false, true}) {
        const std::string component = imaginary ? "imag" : "real";
        const auto result = chpmv_test::VerifyComponent(actual, golden, p.n, p.incy, imaginary, p.caseName + component);
        ::testing::Test::RecordProperty(component + "_finite", std::to_string(result.finite));
        ::testing::Test::RecordProperty(component + "_special", std::to_string(result.special));
        ::testing::Test::RecordProperty(component + "_special_matched", std::to_string(result.specialMatched));
        ::testing::Test::RecordProperty(component + "_verified", result.passed ? "true" : "false");
        EXPECT_TRUE(result.passed) << component;
    }
}

void VerifyStrideHoles(const Param& p, const std::vector<Complex>& actual, const std::vector<Complex>& y)
{
    // 步长空隙逐字节检查，不纳入数值匹配率。
    const uint64_t stride = p.incy < 0 ? -int64_t(p.incy) : int64_t(p.incy);
    for (size_t i = 0; i < y.size(); ++i) {
        if (i % stride != 0)
            EXPECT_EQ(std::memcmp(&actual[i], &y[i], sizeof(Complex)), 0) << "gap " << i;
    }
}
} // namespace

TEST_P(ChpmvArch35Test, CsvDriven)
{
    using namespace chpmv_test;
    const auto& p = GetParam();
    RecordProperty("n", p.n);
    RecordProperty("expected_status", int(p.expectResult));
    std::cout << "PARAM " << p.caseName << " n=" << p.n << " uplo=" << int(p.uplo) << " incx=" << p.incx
              << " incy=" << p.incy << " alpha=" << p.alpha.real << ',' << p.alpha.imag << " beta=" << p.beta.real
              << ',' << p.beta.imag << " seed=" << p.randomSeed << " ap=" << p.apFill << " x=" << p.xFill
              << " y=" << p.yFill << '\n';
    auto ap = Generate(PackedCount(p.n), p.apFill, p.randomSeed);
    auto x = Generate(Count(p.n, p.incx), p.xFill, p.randomSeed + 1);
    auto y = Generate(Count(p.n, p.incy), p.yFill, p.randomSeed + 2);
    ASSERT_NO_FATAL_FAILURE(ConfigureNumericalInput(p, ap, x));
    if (!ap.empty() && (p.uplo == ACLBLAS_UPPER || p.uplo == ACLBLAS_LOWER)) {
        for (int i = 0; i < p.n; ++i)
            ap[Diagonal(i, p.n, p.uplo)].imag = p.poisonDiagonal ? (i % 2 ? NAN : INFINITY) : 0.0f;
    }
    GuardedDevice dAP(ap, p.prefix), dX(x, p.prefix), dY(y, p.prefix);
    aclblasStatus_t status = ACLBLAS_STATUS_INTERNAL_ERROR;
    ASSERT_NO_FATAL_FAILURE(RunChpmvNpu(p, handle_, stream_, dAP, dX, dY, status));
    RecordProperty("actual_status", int(status));
    ASSERT_EQ(status, p.expectResult);
    ASSERT_TRUE(dAP.GuardsIntact());
    ASSERT_TRUE(dX.GuardsIntact());
    ASSERT_TRUE(dY.GuardsIntact());
    ASSERT_TRUE(dAP.Equals(ap));
    ASSERT_TRUE(dX.Equals(x));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        ASSERT_TRUE(dY.Equals(y));
        return;
    }
    if (p.n == 0) {
        RecordProperty("comparison", "legal_empty_output");
        return;
    }
    auto actual = y;
    dY.Read(actual);
    auto golden = y;
    const auto* a = p.nullAlpha ? nullptr : &p.alpha;
    const auto* b = p.nullBeta ? nullptr : &p.beta;
    Oracle::Instance().run(102, int(p.uplo), p.n, a, ap.data(), x.data(), p.incx, b, golden.data(), p.incy);
    VerifyOutput(p, actual, golden);
    VerifyStrideHoles(p, actual, y);
}

TEST_F(ChpmvArch35Test, ValidationOrder)
{
    const auto invalid = static_cast<aclblasFillMode_t>(999);
    EXPECT_EQ(
        aclblasChpmv(nullptr, invalid, -1, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChpmv(handle_, invalid, -1, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasChpmv(nullptr, invalid, 0, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChpmv(handle_, invalid, 0, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasChpmv(nullptr, ACLBLAS_UPPER, 1, nullptr, nullptr, nullptr, 1, nullptr, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(ChpmvArch35Test, InvalidUploBeforeOtherArguments)
{
    const auto invalid = static_cast<aclblasFillMode_t>(999);
    EXPECT_EQ(
        aclblasChpmv(handle_, invalid, 1, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
        ACLBLAS_STATUS_INVALID_ENUM);
}

TEST_F(ChpmvArch35Test, RepeatedInputsAndStream)
{
    using namespace chpmv_test;
    for (const int n : {33, 65, 257, 513}) {
        std::cout << "LIFECYCLE n=" << n << std::endl;
        std::vector<Complex> ap(PackedCount(n), {1, 0}), x(n, {1, 0}), y(n, {0, 0});
        GuardedDevice dAP(ap, 17), dX(x, 17), dY(y, 17);
        aclrtStream alternate = nullptr;
        ASSERT_EQ(aclrtCreateStream(&alternate), ACL_SUCCESS);
        StreamRestore restore{handle_, stream_, alternate};
        ASSERT_EQ(aclblasSetStream(handle_, alternate), ACLBLAS_STATUS_SUCCESS);
        Complex alpha{1, 0}, beta{0, 0};
        ASSERT_EQ(
            aclblasChpmv(handle_, ACLBLAS_UPPER, n, &alpha, dAP.ptr, dX.ptr, 1, &beta, dY.ptr, 1),
            ACLBLAS_STATUS_SUCCESS);
        alpha = {9, 9};
        beta = {7, 7};
        ASSERT_EQ(aclrtSynchronizeStream(alternate), ACL_SUCCESS);
        dY.Read(y);
        for (const auto& z : y) {
            EXPECT_EQ(z.real, n);
            EXPECT_EQ(z.imag, 0);
        }
        for (int i = 0; i < n; ++i)
            ap[Diagonal(i, n, ACLBLAS_UPPER)] = {2, NAN};
        for (auto& z : x)
            z = {2, 0};
        dAP.Write(ap);
        dX.Write(x);
        alpha = {1, 0};
        beta = {0, 0};
        ASSERT_EQ(
            aclblasChpmv(handle_, ACLBLAS_UPPER, n, &alpha, dAP.ptr, dX.ptr, 1, &beta, dY.ptr, 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(alternate), ACL_SUCCESS);
        dY.Read(y);
        for (const auto& z : y) {
            EXPECT_EQ(z.real, 2 * (n + 1));
            EXPECT_EQ(z.imag, 0);
        }
        EXPECT_TRUE(dAP.GuardsIntact() && dX.GuardsIntact() && dY.GuardsIntact());
        EXPECT_TRUE(dAP.Equals(ap) && dX.Equals(x));
    }
}

TEST(ChpmvComparator, RejectsInjectedErrors)
{
    using namespace chpmv_test;
    std::vector<Complex> g(100, {0, 0}), a = g;
    EXPECT_TRUE(VerifyComponent(a, g, 100, 1, false).passed);
    a[0].real = 0.001f;
    EXPECT_TRUE(VerifyComponent(a, g, 100, 1, false).passed);
    a[1].real = a[2].real = 0.001f;
    EXPECT_FALSE(VerifyComponent(a, g, 100, 1, false).passed);
    a = g;
    a[0].real = 0.02f;
    a[99].real = g[99].real = 1e20f;
    EXPECT_FALSE(VerifyComponent(a, g, 100, 1, false).passed);
    a = g;
    a[0].real = INFINITY;
    EXPECT_FALSE(VerifyComponent(a, g, 100, 1, false).passed);
    g.assign(1, {INFINITY, 0});
    a.assign(1, {-INFINITY, 0});
    EXPECT_FALSE(VerifyComponent(a, g, 1, 1, false).passed);
    a[0].real = NAN;
    EXPECT_TRUE(VerifyComponent(a, g, 1, 1, false).passed);
    g[0].real = NAN;
    a[0].real = 1;
    EXPECT_FALSE(VerifyComponent(a, g, 1, 1, false).passed);
    g.assign(1, {0, 0});
    a.assign(1, {0.0001f, 0});
    EXPECT_FALSE(VerifyComponent(a, g, 1, 1, false).passed);
    g[0].real = 1;
    a[0].real = 1.0005f;
    EXPECT_FALSE(VerifyComponent(a, g, 1, 1, false).passed);
    g.assign(100, {INFINITY, 0});
    a = g;
    g[0].real = 0;
    a[0].real = 0.001f;
    EXPECT_FALSE(VerifyComponent(a, g, 100, 1, false).passed);
}

TEST_F(ChpmvArch35Test, GuardAndReadonlySelfCheck)
{
    using namespace chpmv_test;
    const std::vector<Complex> original(3, {1, 2});
    GuardedDevice device(original, 17);
    EXPECT_TRUE(device.GuardsIntact() && device.Equals(original));
    const uint32_t bad = 0;
    GuardedDevice::Check(
        aclrtMemcpy(reinterpret_cast<uint8_t*>(device.ptr) - 4, 4, &bad, 4, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_FALSE(device.GuardsIntact());
    const Complex changed{7, 9};
    GuardedDevice::Check(
        aclrtMemcpy(device.ptr, sizeof(Complex), &changed, sizeof(Complex), ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_FALSE(device.Equals(original));
    RecordProperty("framework_selfcheck", "guard_and_readonly_fault_injection");
}
