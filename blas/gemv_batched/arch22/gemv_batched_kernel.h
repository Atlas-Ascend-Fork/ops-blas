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
 * \file gemv_batched_kernel.h
 * \brief Kernel launch declaration for aclblasGemvBatched (arch22 / dav-2201).
 */

#pragma once

#include <cstdint>
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "gemv_batched_tiling_data.h"

aclError gemv_batched_kernel_do(
    uint8_t* aArray, uint8_t* xArray, uint8_t* yArray, const GemvBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream);
