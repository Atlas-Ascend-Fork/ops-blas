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
namespace LtMatmul22Device {
template <AscendC::HardEvent EVENT>
__aicore__ inline void SyncPipe()
{
    // Event zero is private to these queue-free kernels; every set is consumed.
    AscendC::SetFlag<EVENT>(EVENT_ID0);
    AscendC::WaitFlag<EVENT>(EVENT_ID0);
}
__aicore__ inline uint32_t MinOf(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }
} // namespace LtMatmul22Device
