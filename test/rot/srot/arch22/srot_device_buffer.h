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
#include <stdexcept>
#include <string>
#include "acl/acl.h"

namespace srot_arch22_test {
inline void CheckAcl(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

class SrotDeviceBuffer {
public:
    explicit SrotDeviceBuffer(size_t elements) : bytes_(elements * sizeof(float))
    {
        CheckAcl(aclrtMalloc(&pointer_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~SrotDeviceBuffer()
    {
        if (pointer_ != nullptr)
            aclrtFree(pointer_);
    }
    SrotDeviceBuffer(const SrotDeviceBuffer&) = delete;
    SrotDeviceBuffer& operator=(const SrotDeviceBuffer&) = delete;
    float* Data(size_t offset = 0) const { return static_cast<float*>(pointer_) + offset; }
    void Write(const float* data)
    {
        CheckAcl(aclrtMemcpy(pointer_, bytes_, data, bytes_, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    }
    void Read(float* data) { CheckAcl(aclrtMemcpy(data, bytes_, pointer_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST), "D2H"); }

private:
    void* pointer_ = nullptr;
    size_t bytes_;
};

} // namespace srot_arch22_test
