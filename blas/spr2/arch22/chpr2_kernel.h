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
 * \file chpr2_kernel.h
 * \brief Kernel launcher declaration for aclblasChpr2 (arch22).
 */

#pragma once

#include <cstdint>
#include "acl/acl.h"
#include "chpr2_tiling_data.h"

aclError chpr2_kernel_do(
    const uint8_t* x, const uint8_t* y, uint8_t* ap, const Chpr2TilingData& tiling, uint32_t numBlocks, void* stream);
