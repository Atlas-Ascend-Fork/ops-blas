/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sgeam_tiling_data.h
 * \brief Tiling data definition for aclblasSgeam (arch22 / dav-2201).
 */

#pragma once

#include <cstdint>
#include "cann_ops_blas_common.h"

constexpr uint32_t SGEAM_ARCH22_TILE = 4096U;
constexpr uint32_t SGEAM_ARCH22_WIN_MAX = 4096U;

struct SgeamTilingData {
    uint32_t m;
    uint32_t n;
    float alpha;
    uint32_t alphaIsZero;
    aclblasOperation_t opA;
    uint32_t lda;
    float beta;
    uint32_t betaIsZero;
    aclblasOperation_t opB;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t usedCoreNum;
    uint32_t perCoreN;
    uint32_t remainder;
};
