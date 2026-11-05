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

#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace chpr35 {
inline void CheckAcl(aclError status)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error("CHPR ACL operation failed: " + std::to_string(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::vector<aclblasComplex>& src) : bytes_(src.size() * sizeof(aclblasComplex))
    {
        if (bytes_ == 0) {
            return;
        }
        CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&ptr_), bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
        const auto status = aclrtMemcpy(ptr_, bytes_, src.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != ACL_SUCCESS) {
            aclrtFree(ptr_);
            ptr_ = nullptr;
            CheckAcl(status);
        }
    }
    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) {
            aclrtFree(ptr_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    aclblasComplex* get() const { return ptr_; }
    std::vector<aclblasComplex> Read() const
    {
        std::vector<aclblasComplex> out(bytes_ / sizeof(aclblasComplex));
        if (bytes_ != 0) {
            CheckAcl(aclrtMemcpy(out.data(), bytes_, ptr_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        return out;
    }

private:
    aclblasComplex* ptr_ = nullptr;
    size_t bytes_ = 0;
};

inline size_t PackedSize(int n)
{
    if (n <= 0) {
        return 0;
    }
    return static_cast<size_t>(n) * (static_cast<size_t>(n) + 1) / 2;
}
inline size_t VectorSize(int n, int incx)
{
    if (n <= 0) {
        return 0;
    }
    const int64_t stride = incx < 0 ? -static_cast<int64_t>(incx) : incx;
    return 1 + static_cast<size_t>(n - 1) * static_cast<size_t>(stride);
}
inline size_t DiagonalIndex(int j, int n, aclblasFillMode_t uplo)
{
    const size_t col = static_cast<size_t>(j);
    return uplo == ACLBLAS_UPPER ? col * (col + 1) / 2 + col : col * (2 * static_cast<size_t>(n) - col + 1) / 2;
}
} // namespace chpr35
