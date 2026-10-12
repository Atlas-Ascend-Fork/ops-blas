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
#include <cstdint>
#include <random>
#include <gtest/gtest.h>

#define CTRSV_SCALAR_INLINE inline
#include "../../../../blas/trsv/arch22/ctrsv_kernel_utils.h"

extern "C" __complex__ float __divsc3(float, float, float, float);
extern "C" __complex__ float __mulsc3(float, float, float, float);

namespace {
constexpr uint32_t DIVISION_EDGES[] = {0U,          0x80000000U, 1U,          0x80000001U, 0x7fffffU,
                                       0x800000U,   0x3f800000U, 0xbf800000U, 0x7f7fffffU, 0xff7fffffU,
                                       0x7f800000U, 0xff800000U, 0x7fc00000U};
constexpr uint32_t PRODUCT_EDGES[] = {0U,          0x80000000U, 1U,          0x800000U,   0x3f800000U, 0xbf800000U,
                                      0x7f7fffffU, 0xff7fffffU, 0x7f800000U, 0xff800000U, 0x7fc00000U};

bool SameFloat(float actual, float expected)
{
    return ctrsv_netlib::Bits(actual) == ctrsv_netlib::Bits(expected) || (std::isnan(actual) && std::isnan(expected));
}

void CheckDivision(uint32_t ar, uint32_t ai, uint32_t br, uint32_t bi)
{
    using namespace ctrsv_netlib;
    const Complex a{Value(ar), Value(ai)}, b{Value(br), Value(bi)};
    const auto expected = __divsc3(a.real, a.imag, b.real, b.imag);
    const auto actual = DivideFullRange(a, b);
    ASSERT_TRUE(SameFloat(actual.real, __real__ expected) && SameFloat(actual.imag, __imag__ expected))
        << std::hex << "inputs=" << ar << "," << ai << "," << br << "," << bi << " actual=" << Bits(actual.real) << ","
        << Bits(actual.imag) << " expected=" << Bits(__real__ expected) << "," << Bits(__imag__ expected);
}

void CheckDivisionDenominators(uint32_t ar, uint32_t ai)
{
    for (auto br : DIVISION_EDGES) {
        for (auto bi : DIVISION_EDGES)
            ASSERT_NO_FATAL_FAILURE(CheckDivision(ar, ai, br, bi));
    }
}

void CheckProduct(uint32_t ar, uint32_t ai, uint32_t br, uint32_t bi)
{
    using namespace ctrsv_netlib;
    const Complex a{Value(ar), Value(ai)}, b{Value(br), Value(bi)};
    const auto expected = __mulsc3(a.real, a.imag, b.real, b.imag);
    const auto actual = RecoverProduct(a, b);
    ASSERT_TRUE(SameFloat(actual.real, __real__ expected) && SameFloat(actual.imag, __imag__ expected))
        << std::hex << ar << "," << ai << "," << br << "," << bi;
}

void CheckProductFactors(uint32_t ar, uint32_t ai)
{
    for (auto br : PRODUCT_EDGES) {
        for (auto bi : PRODUCT_EDGES)
            ASSERT_NO_FATAL_FAILURE(CheckProduct(ar, ai, br, bi));
    }
}
} // namespace

TEST(CtrsvScalarArithmetic, FullRangeDivision)
{
    for (auto ar : DIVISION_EDGES) {
        for (auto ai : DIVISION_EDGES)
            ASSERT_NO_FATAL_FAILURE(CheckDivisionDenominators(ar, ai));
    }
    std::mt19937 random(20260916U);
    for (int i = 0; i < 100000; ++i) {
        const uint32_t ar = random(), ai = random(), br = random(), bi = random();
        ASSERT_NO_FATAL_FAILURE(CheckDivision(ar, ai, br, bi));
    }
}

TEST(CtrsvScalarArithmetic, ComplexMultiplyRecovery)
{
    for (auto ar : PRODUCT_EDGES) {
        for (auto ai : PRODUCT_EDGES)
            ASSERT_NO_FATAL_FAILURE(CheckProductFactors(ar, ai));
    }
}
