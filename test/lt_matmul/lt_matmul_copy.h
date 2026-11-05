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

#include <cstddef>
#include <securec.h>
#include <stdexcept>

namespace LtMatmul22Test {

inline void CopyBytes(void* destination, size_t capacity, const void* source, size_t bytes)
{
    if (memcpy_s(destination, capacity, source, bytes) != EOK) {
        throw std::runtime_error("Test byte copy exceeds its destination or has an invalid source");
    }
}

} // namespace LtMatmul22Test
