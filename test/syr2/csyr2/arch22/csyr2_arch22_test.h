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

#include "blas_test.h"
#include "csyr2_param.h"

using Csyr2Arch22Param = csyr2_arch22_test::Csyr2Param;

class Csyr2Arch22Test : public BlasTest<Csyr2Arch22Param> {
protected:
    void RunPerformance(
        const Csyr2Arch22Param& p, const aclblasComplex* xPtr, const aclblasComplex* yPtr, aclblasComplex* aPtr);

    void CheckResult(
        const Csyr2Arch22Param& p, const std::vector<aclblasComplex>& aHost, const std::vector<aclblasComplex>& aOrig,
        const std::vector<aclblasComplex>& xHost, const std::vector<aclblasComplex>& yHost);
};
