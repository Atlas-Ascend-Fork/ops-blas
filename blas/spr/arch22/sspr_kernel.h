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
#include "sspr_tiling_data.h"

// Kernel launcher: dispatches to the compile-time template branch (TilingKey_0 incx==1 fast path /
// TilingKey_1 general path) selected by tiling.incx, and launches on the given stream.
// The optional geometry-descriptor workspace address travels inside tiling.descAddr
// (tiling.descMode == 1); the desc argument is a placeholder and is not dereferenced.
// x is const - the kernel treats it strictly READ-ONLY. No device code writes through it.
void sspr_kernel_do(const uint8_t* x, uint8_t* ap, uint8_t* desc, const SsprTilingData& tiling,
                    uint32_t numBlocks, void* stream);

// R18.1 largeMode launch wrapper (column-ownership tile path, fused OUT of sspr_kernel.cpp
// into sspr_kernel_large.cpp so the shared-kernel codegen stays at the pre-R18 head):
// sspr_kernel_do routes tiling.largeMode == 1 here. The __global__ entry
// (sspr_kernel_large) stays inside its kernel TU - GM_ADDR is a kernel-side-only type, so
// this host-visible header carries only the wrapper. numBlocks = tiling.useCoreNum (host
// fills 40 parts on the eligible band); x is const / strictly READ-ONLY, same contract.
void sspr_kernel_large_do(const uint8_t* x, uint8_t* ap, const SsprTilingData& tiling,
                          uint32_t numBlocks, void* stream);
