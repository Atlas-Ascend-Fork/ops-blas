/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgbmv_tiling_data.h
 * \brief host/kernel shared tiling data of aclblasCgbmv
 */

#pragma once

#include <cstdint>

constexpr uint32_t CGBMV_TRANS_N = 0;
constexpr uint32_t CGBMV_TRANS_T = 1;
constexpr uint32_t CGBMV_TRANS_C = 2;

constexpr uint32_t CGBMV_MODE_GBMV = 0;  // y = alpha * op(A) * x + beta * y
constexpr uint32_t CGBMV_MODE_SCALE = 1; // alpha == 0: y = beta * y, A and x are not read
// A device-resident alpha/beta is only known inside the kernel, which re-derives mode/betaMode from it.

constexpr uint32_t CGBMV_BETA_ZERO = 0;    // old y is not read
constexpr uint32_t CGBMV_BETA_ONE = 1;     // old y is added without a multiplication
constexpr uint32_t CGBMV_BETA_GENERAL = 2; // old y is multiplied by beta

constexpr uint32_t CGBMV_WARP_SIZE = 32;
constexpr uint32_t CGBMV_N_MAX_THREADS = 1024;   // trans-N block size (32 * slices) upper bound
constexpr uint32_t CGBMV_MIN_COLS_PER_SLICE = 4; // slices are doubled only while each keeps this many columns
constexpr uint32_t CGBMV_N_STEPPED_MIN_COLS = 9; // trans-N slices this long use the stepped inner loop

struct CgbmvTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t kl;
    uint32_t ku;
    uint32_t lda;
    uint32_t trans;         // CGBMV_TRANS_*
    uint32_t mode;          // CGBMV_MODE_*
    uint32_t betaMode;      // CGBMV_BETA_*
    uint32_t numThreads;    // threads per block
    uint32_t groupSize;     // trans T/C: lanes cooperating on one output element (power of 2, <= warp size)
    uint32_t slices;        // trans N: warps sharing the column window of one 32-row tile (power of 2)
    uint32_t nStepped;      // trans N: 1 = stepped-offset inner loop (long slices), 0 = per-column addressing
    uint32_t alphaIsDevice; // 1: alpha lives on the device, the kernel reads it from GM (alphaR/I are 0)
    uint32_t betaIsDevice;  // 1: beta lives on the device, the kernel reads it from GM (betaR/I are 0)
    float alphaR;
    float alphaI;
    float betaR;
    float betaI;
    int64_t incx;
    int64_t incy;
};
