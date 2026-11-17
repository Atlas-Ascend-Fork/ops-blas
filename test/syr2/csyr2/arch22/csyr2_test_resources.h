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
#include <cstdio>
#include "cann_ops_blas.h"

inline void Csyr2CleanupStatus(aclError status, const char* operation) noexcept
{
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "Csyr2 cleanup %s status=%d\n", operation, static_cast<int>(status));
    }
}

struct Csyr2ReviewBuffers {
    void* x = nullptr;
    void* y = nullptr;
    void* a = nullptr;
    void* alpha = nullptr;
    aclrtStream stream = nullptr;

    Csyr2ReviewBuffers() = default;
    Csyr2ReviewBuffers(const Csyr2ReviewBuffers&) = delete;
    Csyr2ReviewBuffers& operator=(const Csyr2ReviewBuffers&) = delete;
    ~Csyr2ReviewBuffers()
    {
        // Also drain work on an assertion/launch failure before releasing inputs.
        if (stream != nullptr) {
            Csyr2CleanupStatus(aclrtSynchronizeStream(stream), "stream");
        }
        if (x != nullptr) {
            Csyr2CleanupStatus(aclrtFree(x), "x");
        }
        if (y != nullptr) {
            Csyr2CleanupStatus(aclrtFree(y), "y");
        }
        if (a != nullptr) {
            Csyr2CleanupStatus(aclrtFree(a), "A");
        }
        if (alpha != nullptr) {
            Csyr2CleanupStatus(aclrtFree(alpha), "alpha");
        }
    }
};
