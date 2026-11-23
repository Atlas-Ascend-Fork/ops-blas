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
#include <string>
#include <vector>
#include "cann_ops_blas.h"

namespace cher2_arch22_test {

class Cher2NpuWrapper {
public:
    explicit Cher2NpuWrapper(const std::vector<aclblasComplex>& host, size_t guard = 64)
        : guard_(guard), bytes_(host.size() * sizeof(aclblasComplex))
    {
        if (host.empty()) {
            return;
        }
        std::vector<uint8_t> initial(bytes_ + 2U * guard_, 0xa5);
        if (memcpy_s(initial.data() + guard_, initial.size() - guard_, host.data(), bytes_) != EOK) {
            ok_ = false;
            err_ = "Device buffer initialization copy failed";
            return;
        }
        aclError status = aclrtMalloc(&base_, initial.size(), ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != ACL_SUCCESS) {
            ok_ = false;
            err_ = "aclrtMalloc failed: " + std::to_string(status);
            return;
        }
        status = aclrtMemcpy(base_, initial.size(), initial.data(), initial.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != ACL_SUCCESS) {
            aclrtFree(base_);
            base_ = nullptr;
            ok_ = false;
            err_ = "aclrtMemcpy H2D failed: " + std::to_string(status);
        }
    }

    ~Cher2NpuWrapper()
    {
        if (base_ != nullptr) {
            aclrtFree(base_);
        }
    }

    Cher2NpuWrapper(const Cher2NpuWrapper&) = delete;
    Cher2NpuWrapper& operator=(const Cher2NpuWrapper&) = delete;

    bool Ok() const { return ok_; }

    const std::string& Error() const { return err_; }

    aclblasComplex* Data() const
    {
        return (base_ != nullptr) ? reinterpret_cast<aclblasComplex*>(static_cast<uint8_t*>(base_) + guard_) : nullptr;
    }

    bool Read(std::vector<aclblasComplex>& host, std::string& err) const
    {
        if (!ok_) {
            err = err_;
            return false;
        }
        if (base_ == nullptr) {
            return true;
        }
        std::vector<uint8_t> result(bytes_ + 2U * guard_);
        const aclError status =
            aclrtMemcpy(result.data(), result.size(), base_, result.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        if (status != ACL_SUCCESS) {
            err = "aclrtMemcpy D2H failed: " + std::to_string(status);
            return false;
        }
        for (size_t i = 0; i < guard_; ++i) {
            if (result[i] != 0xa5 || result[guard_ + bytes_ + i] != 0xa5) {
                err = "Device buffer guard overwritten";
                return false;
            }
        }
        if (memcpy_s(host.data(), host.size() * sizeof(aclblasComplex), result.data() + guard_, bytes_) != EOK) {
            err = "Device buffer read copy failed";
            return false;
        }
        return true;
    }

private:
    size_t guard_;
    void* base_ = nullptr;
    size_t bytes_;
    bool ok_ = true;
    std::string err_;
};

} // namespace cher2_arch22_test
