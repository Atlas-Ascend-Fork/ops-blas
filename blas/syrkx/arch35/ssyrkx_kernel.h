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
 * \file ssyrkx_kernel.h
 * \brief Declaration of kernel launchers for aclblasSsyrkx (arch35).
 *        Shared by host.cpp and kernel.cpp.
 */

#pragma once

#include <cstdint>
#include "acl/acl.h"
#include "ssyrkx_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

aclError ssyrkx_gemm_kernel_do(
    GM_ADDR gmB, GM_ADDR gmA, GM_ADDR gmTemp, const SsyrkxGemmTilingData& tiling, uint32_t numBlocks, void* stream);

aclError ssyrkx_scale_kernel_do(
    GM_ADDR gmTemp, GM_ADDR gmC, const SsyrkxScaleTilingData& tiling, uint32_t numBlocks, void* stream);
