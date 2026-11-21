/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * csscal_tiling_data.h - Tiling data structure for aclblasCsscal
 */

#pragma once

#include <cstdint>

constexpr uint32_t CSSCAL_MAX_CORE_NUM = 64;
constexpr uint32_t CSSCAL_ALIGN_UNIT = 32 / sizeof(float);  // Partition granularity: 8 complex values (64 bytes)
constexpr uint32_t CSSCAL_UB_SIZE = 248 * 1024;  // UB size in bytes (arch35: 248KB)

// One in-place UB region. Changing the buffer count alone does not add pipelining.
constexpr uint32_t CSSCAL_UB_BUF_COUNT = 1;
// 单片复数元素数的上限。25008 = 4M 用例每核负载 75024 的三分之一，
// 4M 因此走 3 片等长编译期定长完整块、无运行期尾片。
// 扫描实测（2026-09-13，预热5+60采样 EVENT_TIME 均值，us）：
//   0(=31744): 2M 29.0 / 4M 48.0~48.2（基线，3 轮簇稳定）
//   28672:     2M 28.7 / 4M 47.1
//   24576:     2M 28.3 / 4M 49.8（1296 元素小尾片反噬）
//   16384:     2M 30.5 / 4M 49.2
//   25008:     2M 28.3 / 4M 46.9 <- 当前，4M 比基线 −2.4%（3 轮簇不重叠）
// 机制未完全坐实（「等长」「无尾片」「片长变小」三者在此扫描中耦合）。
// 代价：n 远大于 4M 时片数按 25008 计（小于 UB 容量 31744），单请求略小；
// 三个验收规模均不劣于 cap=0。1M 每核 18720 < 25008，路径完全不受影响。
constexpr uint32_t CSSCAL_MAX_TILE = 25008;
constexpr uint32_t CSSCAL_UB_TILE_CAPACITY =
    CSSCAL_UB_SIZE / (CSSCAL_UB_BUF_COUNT * 2 * sizeof(float)) / CSSCAL_ALIGN_UNIT * CSSCAL_ALIGN_UNIT;
constexpr uint32_t CSSCAL_TILE_CAPACITY =
    CSSCAL_MAX_TILE > 0 && CSSCAL_MAX_TILE < CSSCAL_UB_TILE_CAPACITY ?
    CSSCAL_MAX_TILE : CSSCAL_UB_TILE_CAPACITY;
static_assert(CSSCAL_TILE_CAPACITY > 0 && CSSCAL_TILE_CAPACITY % CSSCAL_ALIGN_UNIT == 0,
              "Full tiles must be nonempty and transfer aligned");

struct CsscalTilingData {
    uint32_t totalN;       // 总复数元素个数
    uint32_t perCoreN;     // 每核处理的复数元素数
    uint32_t remainder;     // 最后一个核的余数
    uint32_t tileSize;     // UB 切分大小（复数元素数）
    float alpha;           // 缩放因子

    int64_t incx;          // 步长
    uint32_t useCoreNum;   // 实际使用的核数
    // SIMT 路径的每核分片：由 blockIdx 在 kernel 内推导，避免传 64 元素数组。
    uint32_t baseCount;    // 每核基础元素数 = n / useCoreNum
    uint32_t remain;       // 前 remain 个核各多分 1 个 = n % useCoreNum
    uint32_t nthreads;     // SIMT 线程数
};

// Kernel launch function - implemented in csscal_kernel.cpp
void csscal_kernel_do(uint8_t* x, uint8_t* workSpace, const CsscalTilingData& tiling,
                      uint32_t numBlocks, void* stream);
