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
 * \file sspr2_kernel.h
 * \brief Declaration of the AIV kernel launcher for aclblasSspr2
 *        (arch22 / Ascend910B3, DAV_2201). Shared by sspr2_host.cpp and
 *        sspr2_kernel.cpp (GM_ADDR contract).
 *
 *        Single AIV kernel, single launch, no GM workspace: host validates
 *        parameters + computes the tiling struct, then launches
 *        sspr2_kernel<<<numBlocks, nullptr, stream>>> on the handle stream.
 */

#pragma once

#include <cstdint>
#include "sspr2_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// AIV single kernel (KERNEL_TYPE_AIV_ONLY). numBlocks = tiling.usedCoreNum
// (= min(GetAivCoreCount(), n), host-computed, never hard-coded). The tiling
// struct is passed by value; per-core segment ranges are derived in-kernel
// from GetBlockIdx() via Sspr2CoreSegStart.
void sspr2_kernel_do(
    GM_ADDR x, GM_ADDR y, GM_ADDR ap, const Sspr2TilingData& tiling, uint32_t numBlocks, void* stream);
