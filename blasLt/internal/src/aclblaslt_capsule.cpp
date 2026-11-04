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
 * \file aclblaslt_capsule.cpp
 * \brief Impl-into-capsule packing helper implementation.
 */

#include "aclblaslt_capsule.h"

#include "host_utils.h"

#include <cstddef>

aclblasStatus_t MatPackTransformImpl(void* capsule, size_t capsuleBytes, const void* impl, size_t implBytes)
{
    // 与 host_utils.h 的 PackImplIntoCapsule 函数体曾逐字重复。本文件已 include
    // host_utils.h，故直接委托，只保留符号以维持既有调用点不变。
    return PackImplIntoCapsule(capsule, capsuleBytes, impl, implBytes);
}
