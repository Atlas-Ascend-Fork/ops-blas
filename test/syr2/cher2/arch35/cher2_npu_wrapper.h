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
#include <securec.h>
#include <stdexcept>
#include <vector>
#include "cann_ops_blas.h"

namespace cher2_test {
inline void CheckAcl(aclError status)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error("ACL error " + std::to_string(status));
    }
}

// Guard bytes detect writes beyond the allocation. A 68-byte guard also
// exercises the public complex ABI's four-byte alignment.
class Cher2NpuWrapper {
public:
    explicit Cher2NpuWrapper(const std::vector<aclblasComplex>& host, size_t guard = 64)
        : bytes_(host.size() * sizeof(aclblasComplex)), guard_(guard)
    {
        if (host.empty()) {
            return;
        }
        std::vector<uint8_t> initial(bytes_ + 2 * guard_, 0xa5);
        if (memcpy_s(initial.data() + guard_, initial.size() - guard_, host.data(), bytes_) != EOK) {
            throw std::runtime_error("Device buffer initialization copy failed");
        }
        CheckAcl(aclrtMalloc(&base_, initial.size(), ACL_MEM_MALLOC_HUGE_FIRST));
        auto status = aclrtMemcpy(base_, initial.size(), initial.data(), initial.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != ACL_SUCCESS) {
            aclrtFree(base_);
            base_ = nullptr;
            CheckAcl(status);
        }
    }
    ~Cher2NpuWrapper()
    {
        if (base_) {
            aclrtFree(base_);
        }
    }
    Cher2NpuWrapper(const Cher2NpuWrapper&) = delete;
    Cher2NpuWrapper& operator=(const Cher2NpuWrapper&) = delete;
    aclblasComplex* Data() const
    {
        return base_ ? reinterpret_cast<aclblasComplex*>(static_cast<uint8_t*>(base_) + guard_) : nullptr;
    }
    void Read(std::vector<aclblasComplex>& host) const
    {
        if (!base_) {
            return;
        }
        std::vector<uint8_t> result(bytes_ + 2 * guard_);
        CheckAcl(aclrtMemcpy(result.data(), result.size(), base_, result.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        for (size_t i = 0; i < guard_; ++i) {
            if (result[i] != 0xa5 || result[guard_ + bytes_ + i] != 0xa5) {
                throw std::runtime_error("Device buffer guard overwritten");
            }
        }
        if (memcpy_s(host.data(), host.size() * sizeof(aclblasComplex), result.data() + guard_, bytes_) != EOK) {
            throw std::runtime_error("Device buffer read copy failed");
        }
    }

private:
    size_t guard_;
    void* base_ = nullptr;
    size_t bytes_;
};
} // namespace cher2_test
