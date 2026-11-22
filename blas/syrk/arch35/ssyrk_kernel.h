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
 * \file ssyrk_kernel.h
 * \brief Declaration of kernel launchers for aclblasSsyrk (arch35).
 *        Shared by host.cpp and kernel.cpp.
 */

#pragma once

#include <cstdint>
#include "acl/acl.h"
#include "ssyrk_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

aclError syrk_gemm_kernel_do(
    GM_ADDR gmA, GM_ADDR gmTemp, const SyrkGemmTilingData& tiling, uint32_t numBlocks, void* stream);

aclError syrk_scale_kernel_do(
    GM_ADDR gmTemp, GM_ADDR gmC, const SyrkScaleTilingData& tiling, uint32_t numBlocks, void* stream);
