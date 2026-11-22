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
 * \file ssyr2k_kernel.h
 * \brief Declaration of kernel launchers for aclblasSsyr2k (arch35).
 *        Shared by host.cpp and kernel.cpp.
 */

#pragma once

#include <cstdint>
#include "acl/acl.h"
#include "ssyr2k_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

aclError ssyr2k_gemm_kernel_do(
    GM_ADDR gmLeft, GM_ADDR gmRight, GM_ADDR gmTemp, const Ssyr2kGemmTilingData& tiling, uint32_t numBlocks,
    void* stream);

aclError ssyr2k_scale_kernel_do(
    GM_ADDR gmTemp1, GM_ADDR gmTemp2, GM_ADDR gmC, const Ssyr2kScaleTilingData& tiling, uint32_t numBlocks,
    void* stream);
