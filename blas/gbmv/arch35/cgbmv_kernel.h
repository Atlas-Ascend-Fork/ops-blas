/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgbmv_kernel.h
 * \brief kernel launcher declaration of aclblasCgbmv
 */

#pragma once

#include <cstdint>
#include "cgbmv_tiling_data.h"

// alphaGm/betaGm: device pointers of alpha/beta when tiling.alphaIsDevice/betaIsDevice is set, otherwise nullptr.
void cgbmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* alphaGm, uint8_t* betaGm, uint32_t numBlocks,
    const CgbmvTilingData& tiling, void* stream);
