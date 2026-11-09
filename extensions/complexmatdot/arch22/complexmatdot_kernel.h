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
 * \file complexmatdot_kernel.h
 * \brief Declaration of the kernel launcher for aclblasComplexMatDot (arch22).
 *        Shared by host.cpp and kernel.cpp.
 */

#pragma once

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// issue #862: kernel_do 返回发射错误码（对齐 chpmv kernel_do 口径）
aclError complex_mat_dot_kernel_do(
    GM_ADDR matx, GM_ADDR maty, GM_ADDR aug, GM_ADDR result, GM_ADDR tilingGm, uint32_t numBlocks, void* stream);
