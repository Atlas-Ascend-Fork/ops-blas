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
 * \file ssyrk_tiling_data.h
 * \brief Tiling data structures for aclblasSsyrk (arch35).
 *        Shared by host side (const ref) and kernel side (by value).
 */

#pragma once

#include <cstdint>

// [general] DAV_3510 hardware parameters for FP32 Cube MAC array
constexpr uint32_t SYRK_ARCH35_BASE_M = 16;
constexpr uint32_t SYRK_ARCH35_BASE_N = 16;
constexpr uint32_t SYRK_ARCH35_BASE_K = 8;
constexpr uint32_t SYRK_ARCH35_FIXPIPE_N_ALIGN = 8;
constexpr uint32_t SYRK_ARCH35_DEFAULT_TILE_M = 128;
constexpr uint32_t SYRK_ARCH35_DEFAULT_TILE_N = 128;
constexpr uint32_t SYRK_ARCH35_DEFAULT_TILE_K_CHUNK = 256;
constexpr uint32_t SYRK_ARCH35_FP32_SIZE = sizeof(float);
// [general] DAV_3510 on-chip memory capacities
constexpr uint32_t SYRK_ARCH35_L1_SIZE_BYTES = 512 * 1024;
constexpr uint32_t SYRK_ARCH35_L1_BUF_NUM = 2;
constexpr uint32_t SYRK_ARCH35_L1_USAGE_RATIO_NUM = 9;
constexpr uint32_t SYRK_ARCH35_L1_USAGE_RATIO_DEN = 10;
// [general] SIMD mode parameters for AIV kernels
constexpr uint32_t SYRK_ARCH35_SCALE_BLOCK = 64;   // scale block size (rows/cols per tile)
constexpr uint32_t SYRK_ARCH35_ELEMENTS_PER_BLOCK = 8; // 32 bytes / sizeof(float)

// Phase 1: GEMM Tiling (AIC-only, tensor_api)
struct SyrkGemmTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileKChunk;
    uint32_t tempRowStride;
    uint8_t isTransN;
};

// Phase 2: Scale Tiling (AIV-only, SIMD)
struct SyrkScaleTilingData {
    uint32_t n;
    uint32_t ldc;
    uint32_t tempRowStride;
    uint32_t rowsPerCore;
    float alphaVal;
    float betaVal;
    uint8_t uploMode;
    uint8_t isAlphaZero;
    uint8_t isKZero;
    uint8_t isBetaZero;
};
