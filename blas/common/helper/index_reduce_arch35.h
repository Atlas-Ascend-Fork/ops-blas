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

// Writes one (value, index) pair to the per-core workspace slot used by
// arch35 iamax/iamin reduction kernels. The index is stored as its float
// bit pattern so the final reduce kernel can move the slot with DataCopy.
__aicore__ inline void WriteIndexReduceWorkspace(
    AscendC::LocalTensor<float> outLocal, const AscendC::GlobalTensor<float>& wsGM, uint32_t blockIdx,
    float bestVal, uint32_t bestIdx)
{
    outLocal.SetValue(0, bestVal);
    outLocal.SetValue(1, *reinterpret_cast<float*>(&bestIdx));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);

    AscendC::DataCopyParams writeParams{1, static_cast<uint16_t>(2 * sizeof(float)), 0, 0};
    AscendC::DataCopyPad(wsGM[blockIdx * 2], outLocal, writeParams);
}
