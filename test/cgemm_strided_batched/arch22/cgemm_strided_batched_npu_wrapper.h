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
#include <algorithm>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

static aclblasStatus_t CopyToDevice(void** out, const std::vector<float>& host, size_t offset = 0)
{
    *out = nullptr;
    if (host.empty()) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    std::vector<float> guarded;
    if (offset != 0) {
        guarded.assign(host.size() + 2U * offset + 16U, 123.25f);
        std::copy(host.begin(), host.end(), guarded.begin() + 2U * offset);
    }
    const auto& source = offset == 0 ? host : guarded;
    const size_t bytes = source.size() * sizeof(float);
    if (aclrtMalloc(out, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*out, bytes, source.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        (void)aclrtFree(*out);
        *out = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

struct CgemmDeviceBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    CgemmDeviceBuffers() = default;
    CgemmDeviceBuffers(const CgemmDeviceBuffers&) = delete;
    CgemmDeviceBuffers& operator=(const CgemmDeviceBuffers&) = delete;
    ~CgemmDeviceBuffers()
    {
        if (a != nullptr) {
            (void)aclrtFree(a);
        }
        if (b != nullptr) {
            (void)aclrtFree(b);
        }
        if (c != nullptr) {
            (void)aclrtFree(c);
        }
    }
};
