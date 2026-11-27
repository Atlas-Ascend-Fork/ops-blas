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
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

// Own every allocation until the handle stream has completed, including failed assertions.
class CrotNpuWrapper {
public:
    explicit CrotNpuWrapper(aclrtStream stream) : stream_(stream) {}
    CrotNpuWrapper(const CrotNpuWrapper&) = delete;
    CrotNpuWrapper& operator=(const CrotNpuWrapper&) = delete;
    ~CrotNpuWrapper()
    {
        aclrtSynchronizeStream(stream_);
        for (void* pointer : allocations_) {
            aclrtFree(pointer);
        }
    }

    aclError Upload(const void* host, size_t bytes, void** device)
    {
        const aclError result = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (result != ACL_SUCCESS) {
            return result;
        }
        allocations_.push_back(*device);
        return aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

private:
    aclrtStream stream_;
    std::vector<void*> allocations_;
};
