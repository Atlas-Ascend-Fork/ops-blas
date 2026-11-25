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

#include "blasLtMatmul_golden.h"
#include "verify.h"

namespace {

TEST(LtMatmulBf16GoldenHostTest, RoundsFiniteValuesToNearestEven)
{
    struct Sample {
        float input;
        uint16_t expected;
    };
    const Sample samples[] = {
        {1.006f, 0x3f81},       // Above the midpoint: round up.
        {1.01171875f, 0x3f82}, // Odd lower mantissa at the midpoint: round up.
        {1.00390625f, 0x3f80}, // Even lower mantissa at the midpoint: round down.
        {1.002f, 0x3f80},      // Below the midpoint: round down.
        {1.0f, 0x3f80},
        {1.0078125f, 0x3f81},
    };
    for (const auto& sample : samples) {
        EXPECT_EQ(fp32ToBF16(sample.input), sample.expected) << sample.input;
        EXPECT_EQ(fp32ToBF16(-sample.input), static_cast<uint16_t>(sample.expected | 0x8000)) << -sample.input;
    }
}

TEST(LtMatmulBf16GoldenHostTest, AcceptsRoundedOutputWithCsvPrecisionConfig)
{
    // FP32 A/B and BF16 D retain these defaults in the arch35 CSV test.
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    for (float sign : {1.0f, -1.0f}) {
        const float golden[] = {
            bf16ToFP32(fp32ToBF16(sign * 1.006f)),
            bf16ToFP32(fp32ToBF16(sign * 1.01171875f)),
        };
        const float correct[] = {sign * 1.0078125f, sign * 1.015625f};
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_TRUE(Verifier::verifyVector(&correct[i], &golden[i], 1, 1, cfg, "rounded_bf16"));
        }
    }
}

TEST(LtMatmulBf16GoldenHostTest, RejectsTruncatedOutputWithCsvPrecisionConfig)
{
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    for (float sign : {1.0f, -1.0f}) {
        const float golden[] = {
            bf16ToFP32(fp32ToBF16(sign * 1.006f)),
            bf16ToFP32(fp32ToBF16(sign * 1.01171875f)),
        };
        const float truncated[] = {sign * 1.0f, sign * 1.0078125f};
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_FALSE(Verifier::verifyVector(&truncated[i], &golden[i], 1, 1, cfg, "truncated_bf16"));
        }
    }
}

} // namespace
