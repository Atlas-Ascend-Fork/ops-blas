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
#include <vector>
#include "verify.h"

struct CgetrfComponentVerdict {
    bool mixed;
    bool relative;
    bool Passed() const { return mixed && relative; }
};

inline bool CgetrfShouldCheckLu(int goldenInfo)
{
    // The independent CPU result controls skipping, never the tested device's info.
    // Integer info/pivots are checked separately, including every singular matrix.
    return goldenInfo <= 0;
}

inline CgetrfComponentVerdict CgetrfVerifyComponent(
    const float* actual, const float* golden, size_t count, const std::string& name)
{
    VerifyConfig mixed;
    applyMixedTolerance(mixed, ACL_FLOAT, golden, count);
    VerifyConfig relative;
    relative.mode = PrecisionMode::MERE_MARE;
    relative.mereThreshold = 1.0 / 8192.0;
    relative.mareMultiplier = 10.0;
    // Always execute both checkers, even if the first fails; preserve independent evidence.
    const bool mixedPass = Verifier::verifyVector(actual, golden, count, 1, mixed, name + "_mixed");
    const bool relativePass = Verifier::verifyVector(actual, golden, count, 1, relative, name + "_relative");
    std::cout << "CGETRF_COMPONENT_CHECK," << name << ',' << mixedPass << ',' << relativePass << std::endl;
    return {mixedPass, relativePass};
}
