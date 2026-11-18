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

#include "blas_test.h"
#include "chemv_param.h"
#include "chemv_golden.h"

// Reuse the repository's shared ACL guard and runtime device selection, including
// when this source is linked into the combined blas_all_test runner.
class ChemvArch35Test : public BlasTest<ChemvParam> {
protected:
    void SetUp() override
    {
        static const bool referenceVerified = chemv_test::VerifyReferenceBinding();
        ASSERT_TRUE(referenceVerified);
    }
};
