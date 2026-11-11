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

namespace chbmv_test {
class Allocation {
public:
    ~Allocation()
    {
        if (ptr_ != nullptr) {
            aclrtFree(ptr_);
        }
    }
    aclError Upload(const std::vector<aclblasComplex>& data)
    {
        if (data.empty()) {
            return ACL_SUCCESS;
        }
        bytes_ = data.size() * sizeof(aclblasComplex);
        const aclError status = aclrtMalloc(&ptr_, bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
        return status == ACL_SUCCESS ? aclrtMemcpy(ptr_, bytes_, data.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE) :
                                       status;
    }
    aclError Download(std::vector<aclblasComplex>& data) const
    {
        return aclrtMemcpy(data.data(), bytes_, ptr_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    aclblasComplex* Ptr() const { return static_cast<aclblasComplex*>(ptr_); }

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};
} // namespace chbmv_test
