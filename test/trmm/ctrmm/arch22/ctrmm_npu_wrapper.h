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
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "ctrmm_param.h"

class CtrmmDeviceBuffer {
public:
    explicit CtrmmDeviceBuffer(const std::vector<aclblasComplex>& input, size_t offset = 0)
        : count_(std::max<size_t>(input.size(), 1)), prefix_(8 + offset)
    {
        original_.assign(prefix_ + count_ + 8, {713.25f, -917.5f});
        std::copy(input.begin(), input.end(), original_.begin() + prefix_);
        Check(aclrtMalloc(&base_, original_.size() * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST));
        Check(aclrtMemcpy(
            base_, original_.size() * sizeof(aclblasComplex), original_.data(),
            original_.size() * sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ~CtrmmDeviceBuffer()
    {
        if (base_) {
            aclrtFree(base_);
        }
    }
    CtrmmDeviceBuffer(const CtrmmDeviceBuffer&) = delete;
    CtrmmDeviceBuffer& operator=(const CtrmmDeviceBuffer&) = delete;
    aclblasComplex* Data()
    {
        return static_cast<aclblasComplex*>(base_) + prefix_;
    }
    std::vector<aclblasComplex> Read()
    {
        auto actual = original_;
        Check(aclrtMemcpy(
            actual.data(), actual.size() * sizeof(aclblasComplex), base_, actual.size() * sizeof(aclblasComplex),
            ACL_MEMCPY_DEVICE_TO_HOST));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (i >= prefix_ && i < prefix_ + count_) {
                continue;
            }
            if (actual[i].real != original_[i].real || actual[i].imag != original_[i].imag) {
                throw std::runtime_error("Device buffer guard overwritten");
            }
        }
        return {actual.begin() + prefix_, actual.begin() + prefix_ + count_};
    }
    static void Check(aclError status)
    {
        if (status != ACL_SUCCESS) {
            throw std::runtime_error("ACL error " + std::to_string(status));
        }
    }

private:
    void* base_ = nullptr;
    size_t count_;
    size_t prefix_;
    std::vector<aclblasComplex> original_;
};

class CtrmmEvent {
public:
    CtrmmEvent()
    {
        CtrmmDeviceBuffer::Check(aclrtCreateEvent(&event_));
    }
    ~CtrmmEvent()
    {
        if (event_) {
            aclrtDestroyEvent(event_);
        }
    }
    CtrmmEvent(const CtrmmEvent&) = delete;
    CtrmmEvent& operator=(const CtrmmEvent&) = delete;
    aclrtEvent Get() const
    {
        return event_;
    }

private:
    aclrtEvent event_ = nullptr;
};
