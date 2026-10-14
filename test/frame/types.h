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

#include <cstdint>

enum class PrecisionMode { ABS, REL, COMBINED, MERE_MARE, EXACT, INTEGER, MIXED_TOLERANCE };

struct VerifyConfig {
    PrecisionMode mode = PrecisionMode::ABS;
    double absTol = 1e-4;
    double relTol = 1e-5;
    double mereThreshold = 1.0 / 8192.0;
    double mareMultiplier = 10.0;
    double epsilonForRel = 1e-7;
    double mixedAtol = 1e-3;
    double mixedRtol = 1e-3;
    double mixedRequiredMatchedRatio = 0.99;
    double mixedMaxAbsErrorLimit = 1e-1;
    int mixedMantissaBits = 23;
    int mixedEmin = -126;
};
