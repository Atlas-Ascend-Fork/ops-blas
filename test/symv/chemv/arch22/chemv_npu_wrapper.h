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
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

constexpr size_t CHEMV_GUARD_ELEMENTS = 16U;

class ChemvGuardedBuffer {
public:
    explicit ChemvGuardedBuffer(const std::vector<aclblasComplex>& hostData)
    {
        if (hostData.empty()) {
            return;
        }
        original_.assign(hostData.size() + 2U * CHEMV_GUARD_ELEMENTS, aclblasComplex{-713.25f, 859.5f});
        std::copy(hostData.begin(), hostData.end(), original_.begin() + CHEMV_GUARD_ELEMENTS);
        bytes_ = original_.size() * sizeof(aclblasComplex);
        aclError ret = aclrtMalloc(&devPtr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS || devPtr_ == nullptr) {
            throw std::runtime_error("aclrtMalloc failed in ChemvGuardedBuffer");
        }
        ret = aclrtMemcpy(devPtr_, bytes_, original_.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            aclrtFree(devPtr_);
            devPtr_ = nullptr;
            throw std::runtime_error("aclrtMemcpy H2D failed in ChemvGuardedBuffer");
        }
    }

    ~ChemvGuardedBuffer()
    {
        if (devPtr_ != nullptr) {
            aclrtFree(devPtr_);
            devPtr_ = nullptr;
        }
    }

    ChemvGuardedBuffer(const ChemvGuardedBuffer&) = delete;
    ChemvGuardedBuffer& operator=(const ChemvGuardedBuffer&) = delete;

    aclblasComplex* Data() const
    {
        return devPtr_ != nullptr ? static_cast<aclblasComplex*>(devPtr_) + CHEMV_GUARD_ELEMENTS : nullptr;
    }

    std::vector<aclblasComplex> ReadToHost() const
    {
        std::vector<aclblasComplex> out = original_;
        if (devPtr_ != nullptr) {
            aclError ret = aclrtMemcpy(out.data(), bytes_, devPtr_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpy D2H failed in ChemvGuardedBuffer");
            }
        }
        return out;
    }

    bool ElementUnchanged(const std::vector<aclblasComplex>& readback, size_t rawIndex) const
    {
        return std::memcmp(&readback.at(rawIndex), &original_.at(rawIndex), sizeof(aclblasComplex)) == 0;
    }

    bool AllUnchanged() const
    {
        if (devPtr_ == nullptr) {
            return true;
        }
        const auto readback = ReadToHost();
        return readback.size() == original_.size() &&
               std::memcmp(readback.data(), original_.data(), bytes_) == 0;
    }

private:
    std::vector<aclblasComplex> original_;
    void* devPtr_ = nullptr;
    size_t bytes_ = 0;
};

class ChemvScalarDeviceBuffer {
public:
    explicit ChemvScalarDeviceBuffer(const aclblasComplex& val, aclrtStream stream)
    {
        if (aclrtMalloc(&ptr_, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMalloc failed for device scalar");
        }
        if (aclrtMemcpyAsync(ptr_, sizeof(aclblasComplex), &val, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE, stream) !=
            ACL_SUCCESS) {
            aclrtFree(ptr_);
            ptr_ = nullptr;
            throw std::runtime_error("aclrtMemcpyAsync failed for device scalar");
        }
    }

    ~ChemvScalarDeviceBuffer()
    {
        if (ptr_ != nullptr) {
            aclrtFree(ptr_);
            ptr_ = nullptr;
        }
    }

    ChemvScalarDeviceBuffer(const ChemvScalarDeviceBuffer&) = delete;
    ChemvScalarDeviceBuffer& operator=(const ChemvScalarDeviceBuffer&) = delete;

    const aclblasComplex* Data() const { return static_cast<const aclblasComplex*>(ptr_); }

private:
    void* ptr_ = nullptr;
};
