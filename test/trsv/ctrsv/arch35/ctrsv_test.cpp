/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "device.h"
#include "ctrsv_param.h"
#include "ctrsv_cases.h"
#include "ctrsv_npu_wrapper.h"
#include "blas_test.h"

using namespace ctrsv_test;

namespace {
std::vector<CtrsvParam> Cases()
{
    auto original = GetCasesFromCsv<CtrsvParam>(ReplaceFileExtension2Csv(__FILE__));
    auto result = original;
    for (auto p : original) {
        if (p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0 && p.A.method == BlasFillMode::M_RANDOM &&
            p.x.method == BlasFillMode::M_RANDOM && p.A.pattern == BlasFillMode::P_NORM &&
            p.x.pattern == BlasFillMode::P_NORM) {
            p.normal = true;
            p.caseName = "TC_NORMAL_" + p.caseName;
            result.push_back(p);
        }
    }
    for (int mode = 0; mode < 2; ++mode) {
        for (int combo = 0; combo < 12; ++combo) {
            auto p = original.front();
            p.caseName = std::string(mode ? "TC_SMALL_WORKSPACE_" : "TC_GENERAL_") + std::to_string(combo);
            p.n = mode ? 513 : 4097;
            p.lda = p.n + 3;
            p.incx = combo % 2 ? -3 : 2;
            p.uplo = combo < 6 ? ACLBLAS_UPPER : ACLBLAS_LOWER;
            p.trans = static_cast<aclblasOperation_t>(ACLBLAS_OP_N + (combo % 6) / 2);
            p.diag = combo % 2 ? ACLBLAS_UNIT : ACLBLAS_NON_UNIT;
            p.smallWorkspace = mode != 0;
            result.push_back(p);
        }
    }
    return result;
}
} // namespace

class CtrsvTest : public BlasTest<CtrsvParam> {
protected:
    static aclblasStatus_t Invoke(const CtrsvParam& p, const void* a, void* x) { return CtrsvInvoke(handle_, p, a, x); }
};

INSTANTIATE_TEST_SUITE_P(Ctrsv, CtrsvTest, ::testing::ValuesIn(Cases()), PrintCaseInfoString<CtrsvParam>);

// Keep direct API checks under the same prefix used by ctrsv:ALL.
class CtrsvDirectTest : public CtrsvTest {
protected:
    void CheckExplicit(
        CtrsvParam p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& input, bool exact = true)
    {
        auto golden = input, output = input, afterA = a;
        CtrsvGolden(p, a, golden);
        CtrsvDeviceInput deviceInput(a, input);
        ASSERT_EQ(deviceInput.Run(handle_, stream_, p), ACLBLAS_STATUS_SUCCESS);
        deviceInput.Read(afterA, output);
        EXPECT_EQ(std::memcmp(a.data(), afterA.data(), a.size() * sizeof(aclblasComplex)), 0);
        for (size_t i = 0; i < output.size(); ++i) {
            SCOPED_TRACE(i);
            if (exact) {
                EXPECT_EQ(output[i].real, golden[i].real);
                EXPECT_EQ(output[i].imag, golden[i].imag);
            } else {
                EXPECT_FLOAT_EQ(output[i].real, golden[i].real);
                EXPECT_FLOAT_EQ(output[i].imag, golden[i].imag);
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(Ctrsv, CtrsvDirectTest, ::testing::Values(Cases().front()));

TEST_P(CtrsvDirectTest, NullHandle)
{
    EXPECT_EQ(
        aclblasCtrsv(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 0, nullptr, 1, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_P(CtrsvDirectTest, ZeroSizeValidation)
{
    EXPECT_EQ(
        aclblasCtrsv(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 0, nullptr, 1, nullptr, -3),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasCtrsv(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 0, nullptr, 0, nullptr, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCtrsv(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 0, nullptr, 1, nullptr, 0),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_P(CtrsvDirectTest, MisalignedComplexPointers)
{
    for (int combo = 0; combo < 12; ++combo) {
        auto p = Cases().front();
        p.caseName = "TC_OFFSET_" + std::to_string(combo);
        p.n = 129;
        p.lda = 132;
        p.incx = combo % 2 ? -3 : 2;
        p.uplo = combo < 6 ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        p.trans = static_cast<aclblasOperation_t>(ACLBLAS_OP_N + (combo % 6) / 2);
        p.diag = combo % 2 ? ACLBLAS_UNIT : ACLBLAS_NON_UNIT;
        auto a = MakeA(p), input = MakeX(p), golden = input;
        CtrsvGolden(p, a, golden);
        a.insert(a.begin(), SENTINEL);
        a.push_back(SENTINEL);
        input.insert(input.begin(), SENTINEL);
        input.push_back(SENTINEL);
        auto output = input, afterA = a;
        DeviceBuffer da(a.size() * sizeof(aclblasComplex)), dx(input.size() * sizeof(aclblasComplex));
        da.copyFromHost(a.data(), da.size());
        dx.copyFromHost(input.data(), dx.size());
        ASSERT_EQ(
            Invoke(p, da.bytePtr() + sizeof(aclblasComplex), dx.bytePtr() + sizeof(aclblasComplex)),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        dx.copyToHost(output.data(), dx.size());
        da.copyToHost(afterA.data(), da.size());
        EXPECT_EQ(std::memcmp(a.data(), afterA.data(), da.size()), 0);
        EXPECT_EQ(std::memcmp(&output.front(), &SENTINEL, sizeof(SENTINEL)), 0);
        EXPECT_EQ(std::memcmp(&output.back(), &SENTINEL, sizeof(SENTINEL)), 0);
        std::vector<aclblasComplex> logical(output.begin() + 1, output.end() - 1);
        EXPECT_TRUE(CheckPrecision(p, logical, golden));
        for (size_t i = 0; i < logical.size(); ++i) {
            if (i % std::abs(p.incx) != 0) {
                EXPECT_EQ(std::memcmp(&logical[i], &SENTINEL, sizeof(SENTINEL)), 0);
            }
        }
    }
}

TEST_P(CtrsvDirectTest, DivisionProductUnderflow)
{
    auto p = GetParam();
    p.n = p.lda = 2;
    p.incx = 1;
    p.uplo = ACLBLAS_LOWER;
    p.trans = ACLBLAS_OP_N;
    p.diag = ACLBLAS_NON_UNIT;
    for (int exponent : {-120, -119, -118}) {
        SCOPED_TRACE(exponent);
        CheckExplicit(p, {{0x1p-40f, 0}, {1, 0}, {0, 0}, {0x1p-100f, 0}}, {{std::ldexp(1.0f, exponent), 0}, {0, 0}});
    }
}

TEST_P(CtrsvDirectTest, DivisionComponentRange)
{
    auto p = GetParam();
    p.n = p.lda = p.incx = 1;
    p.uplo = ACLBLAS_LOWER;
    p.diag = ACLBLAS_NON_UNIT;
    for (int trans = ACLBLAS_OP_N; trans <= ACLBLAS_OP_C; ++trans) {
        p.trans = static_cast<aclblasOperation_t>(trans);
        for (int exponent : {-120, -100, -40, 0, 40, 100, 127}) {
            for (bool imaginary : {false, true}) {
                SCOPED_TRACE(::testing::Message() << trans << ":" << exponent << ":" << imaginary);
                const float diagonal = std::ldexp(1.0f, exponent);
                const aclblasComplex a = imaginary ? aclblasComplex{0, diagonal} : aclblasComplex{diagonal, 0};
                CheckExplicit(p, {a}, {{0x1p127f, 0x1p-23f}});
                CheckExplicit(p, {a}, {{0x1p-23f, -0x1p127f}});
            }
        }
    }
}

TEST_P(CtrsvDirectTest, DivisionMixedComponents)
{
    auto p = GetParam();
    p.n = p.lda = p.incx = 1;
    p.uplo = ACLBLAS_LOWER;
    p.trans = ACLBLAS_OP_N;
    p.diag = ACLBLAS_NON_UNIT;
    const int exponents[] = {-149, -120, -60, -1, 1, 60, 127};
    for (int ae : exponents) {
        for (int be : exponents) {
            for (int ce : exponents) {
                for (int de : exponents) {
                    SCOPED_TRACE(::testing::Message() << ae << ":" << be << ":" << ce << ":" << de);
                    CheckExplicit(
                        p, {{std::ldexp(1.0f, ce), -std::ldexp(1.0f, de)}},
                        {{std::ldexp(1.0f, ae), std::ldexp(1.0f, be)}}, false);
                }
            }
        }
    }
}

TEST_P(CtrsvTest, CsvDriven)
{
    const auto& p = GetParam();
    std::unique_ptr<DeviceBuffer> smallWorkspace;
    if (p.smallWorkspace) {
        smallWorkspace = std::make_unique<DeviceBuffer>(64);
        ASSERT_EQ(aclblasSetWorkspace(handle_, smallWorkspace->ptr(), 64), ACLBLAS_STATUS_SUCCESS);
    }
    struct RestoreWorkspace {
        aclblasHandle_t handle_;
        aclrtStream stream_;
        ~RestoreWorkspace() { EXPECT_EQ(aclblasSetStream(handle_, stream_), ACLBLAS_STATUS_SUCCESS); }
    } restore{handle_, stream_};
    auto a = MakeA(p), input = MakeX(p);
    auto output = input, golden = input;
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0) {
        EXPECT_EQ(Invoke(p, a.empty() ? nullptr : a.data(), output.empty() ? nullptr : output.data()), p.expectResult);
        return;
    }
    const size_t aBytes = a.size() * sizeof(aclblasComplex);
    CtrsvDeviceInput deviceInput(a, input);
    ASSERT_EQ(deviceInput.Run(handle_, stream_, p), ACLBLAS_STATUS_SUCCESS);
    std::vector<aclblasComplex> aAfter(a.size());
    deviceInput.Read(aAfter, output);
    CtrsvGolden(p, a, golden);
    EXPECT_TRUE(CheckPrecision(p, output, golden));
    const size_t stride = std::abs(int64_t(p.incx));
    for (size_t i = 0; i < output.size(); ++i) {
        if (i % stride != 0) {
            ASSERT_EQ(std::memcmp(&output[i], &input[i], sizeof(aclblasComplex)), 0) << "x padding overwritten";
        }
    }
    EXPECT_EQ(std::memcmp(a.data(), aAfter.data(), aBytes), 0) << "A modified";
}
