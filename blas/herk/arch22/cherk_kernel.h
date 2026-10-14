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
#include "cherk_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void cherk_arch22_kernel_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream);

// Each launch wrapper is implemented beside its device kernel in an independent translation unit.
void cherk_arch22_small_vector_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream);
void cherk_arch22_outer_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR metadata, const CherkArch22TilingData& tiling,
    void* stream);
void cherk_arch22_pack_do(GM_ADDR a, GM_ADDR workspace, const CherkArch22TilingData& tiling, void* stream);
void cherk_arch22_direct_pack_do(
    GM_ADDR a, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData& tiling, void* stream);
void cherk_arch22_compute_do(
    GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData& tiling,
    void* stream);
void cherk_arch22_direct_compute_do(
    GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData& tiling,
    void* stream);

void cherk_arch22_square64_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream);

void cherk_arch22_square128_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream);

void cherk_arch22_direct_mix_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream);

#ifdef CHERK_ARCH22_DEVICE_IMPL
struct CherkCopyEvents {
    event_t vToMte2;
    event_t mte2ToV;
    event_t vToMte3;
    event_t mte3ToV;
};
#endif
