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

#include "kernel_operator.h"
#include "cdotc_tiling_data.h"

constexpr uint32_t CDOTC_MAX_PARALLEL_CORE_NUM = 48;

void CdotcKernelDo(
    GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, uint32_t numBlocks, const CdotcTilingData& tiling,
    void* stream);
