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
#include "kernel_operator.h"

namespace cube_load {
__aicore__ inline void InitUnitFilter(AscendC::LoadData3DParamsV2<float>& params)
{
    params.strideW = 1U;
    params.strideH = 1U;
    params.filterW = 1U;
    params.filterH = 1U;
    params.dilationFilterW = 1U;
    params.dilationFilterH = 1U;
    params.filterSizeW = false;
    params.filterSizeH = false;
}
} // namespace cube_load
