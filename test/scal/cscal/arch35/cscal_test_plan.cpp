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
#include "../../../../blas/scal/arch35/cscal_plan.h"

namespace {
TEST(CscalPlanTest, CeilingDivisionDoesNotOverflow)
{
    EXPECT_EQ(cscal950::CeilDiv<256>(0), 0u);
    EXPECT_EQ(cscal950::CeilDiv<256>(1), 1u);
    EXPECT_EQ(cscal950::CeilDiv<256>(256), 1u);
    EXPECT_EQ(cscal950::CeilDiv<256>(257), 2u);
    EXPECT_EQ(cscal950::CeilDiv<256>(UINT64_MAX), UINT64_MAX / 256 + 1);
    EXPECT_EQ(cscal950::CeilDiv<1>(UINT64_MAX), UINT64_MAX);
}

TEST(CscalPlanTest, BitClassificationPreservesSpecialValues)
{
    const uint32_t patterns[] = {0u,          0x80000000u, 1u,          0x80000001u, 0x3f800000u, 0x7f7fffffu,
                                 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00001u};
    for (uint32_t bits : patterns) {
        float value = 0.0f;
        std::copy_n(
            reinterpret_cast<const unsigned char*>(&bits), sizeof(value), reinterpret_cast<unsigned char*>(&value));
        EXPECT_EQ(cscal950::FloatBits(value), bits);
        const bool finite = (bits & 0x7f800000u) != 0x7f800000u;
        EXPECT_EQ(cscal950::IsFiniteAlphaComponent(value), finite);
        CscalTilingData plan{};
        ASSERT_TRUE(cscal950::MakePlan(257, 1, value, 1.0f, 40, plan));
        EXPECT_EQ(bool(plan.finiteAlpha), finite);
        ASSERT_TRUE(cscal950::MakePlan(257, 1, 1.0f, value, 40, plan));
        EXPECT_EQ(bool(plan.finiteAlpha), finite);
    }
}

TEST(CscalPlanTest, PartitionsCoverTailsAndRejectInvalidCores)
{
    for (int n : {1, 7, 8, 255, 256, 257, 2047, 2048, 2049, 8193}) {
        for (int incx : {1, 3}) {
            CscalTilingData plan{};
            ASSERT_TRUE(cscal950::MakePlan(n, incx, 2.0f, -3.0f, 40, plan));
            uint64_t end = 0;
            for (uint32_t core = 0; core < plan.coreCount; ++core) {
                const auto slice = cscal950::Partition(plan, core);
                EXPECT_EQ(slice.start, end);
                EXPECT_GT(slice.count, 0u);
                end += slice.count;
            }
            EXPECT_EQ(end, uint64_t(n));
            EXPECT_EQ(cscal950::Partition(plan, plan.coreCount).count, 0u);
        }
    }
    CscalTilingData plan{};
    EXPECT_FALSE(cscal950::MakePlan(1, 1, 2.0f, -3.0f, 0, plan));
    EXPECT_EQ(cscal950::Partition(plan, 0).count, 0u);
}
} // namespace
