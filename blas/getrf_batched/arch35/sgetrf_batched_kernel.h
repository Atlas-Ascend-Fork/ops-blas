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
 * \file sgetrf_batched_kernel.h
 * \brief Declares kernel_do for batched single-precision LU factorization (SIMT).
 */

#pragma once

#include <cstdint>
#include "acl/acl.h"
#include "sgetrf_batched_tiling_data.h"

aclError sgetrf_batched_kernel_do(
    uint8_t* aarray, uint8_t* pivotArray, uint8_t* infoArray, const SgetrfBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream);
