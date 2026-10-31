/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cmath>

#include "acl/acl.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers: MIXED_TOLERANCE thresholds (mixed_tolerance_standard.md)
//   per-element max_abs_error_limit = max(fixed_value, 32 * ULP)
//   ULP(x) = 2^(floor(log2|x|) - mantissaBits)  for finite x != 0
//   ULP(0) = 2^(emin - mantissaBits)              (subnormal ULP)
// ═══════════════════════════════════════════════════════════════════════════════
inline double getUlpsAt(double magnitude, int mantissaBits, int emin)
{
    double absMag = std::abs(magnitude);
    if (absMag == 0.0)
        return std::pow(2.0, static_cast<double>(emin - mantissaBits));
    int exp = 0;
    std::frexp(absMag, &exp);
    return std::ldexp(1.0, std::max(exp - 1, emin) - mantissaBits);
}

struct MixedToleranceDefaults {
    double rtol;
    double atol;
    double maxAbsErrorLimitFixed;
    int mantissaBits;
    int emin;
};

inline MixedToleranceDefaults getMixedToleranceDefaults(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16:
            return {0.0009765625, 0.0009765625, 1e-1, 10, -14};
        case ACL_BF16:
            return {0.0078125, 0.0078125, 1e-1, 7, -126};
        case ACL_FLOAT:
            return {0.0001220703125, 0.0001220703125, 1e-2, 23, -126};
        case ACL_FLOAT8_E4M3FN:
            return {0.125, 0.125, 1.0, 3, -6};
        case ACL_FLOAT8_E5M2:
            return {0.25, 0.25, 1.0, 2, -14};
        default:
            return {0.0009765625, 0.0009765625, 1e-1, 10, -14};
    }
}
