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
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct ChemmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int64_t m = 0;
    int64_t n = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    bool nullAlpha = false;
    BlasFillMode aFill;
    std::string aFillSpec;
    bool aNormal = false;
    int64_t lda = 0;
    BlasFillMode bFill;
    std::string bFillSpec;
    bool bNormal = false;
    int64_t ldb = 0;
    float betaReal = 0.0f;
    float betaImag = 0.0f;
    bool nullBeta = false;
    BlasFillMode cFill;
    std::string cFillSpec;
    bool cNormal = false;
    int64_t ldc = 0;

    ChemmParam(const csv_map& map);
};
