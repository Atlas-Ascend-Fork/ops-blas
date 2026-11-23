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
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "../chpr2_param.h"

namespace chpr2_arch22_test {

constexpr size_t CHPR2_GUARD = 4;
constexpr Complex CHPR2_CANARY = {12345.0f, -54321.0f};

class Chpr2NpuWrapper {
public:
    explicit Chpr2NpuWrapper(const std::vector<Complex>& input)
        : size_(input.size()), original_(size_ + 2U * CHPR2_GUARD, CHPR2_CANARY)
    {
        std::copy(input.begin(), input.end(), original_.begin() + CHPR2_GUARD);
        Check(aclrtMalloc(reinterpret_cast<void**>(&base_), Bytes(), ACL_MEM_MALLOC_NORMAL_ONLY), "malloc");
        try {
            Check(aclrtMemcpy(base_, Bytes(), original_.data(), Bytes(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
        } catch (...) {
            aclrtFree(base_);
            throw;
        }
    }

    ~Chpr2NpuWrapper()
    {
        if (base_ != nullptr) {
            aclrtFree(base_);
        }
    }

    Chpr2NpuWrapper(const Chpr2NpuWrapper&) = delete;
    Chpr2NpuWrapper& operator=(const Chpr2NpuWrapper&) = delete;

    Complex* Data() { return base_ + CHPR2_GUARD; }

    std::vector<Complex> Read(bool unchanged = false)
    {
        auto result = original_;
        Check(aclrtMemcpy(result.data(), Bytes(), base_, Bytes(), ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        EXPECT_EQ(std::memcmp(result.data(), original_.data(), CHPR2_GUARD * sizeof(Complex)), 0) << "leading guard";
        EXPECT_EQ(
            std::memcmp(
                result.data() + CHPR2_GUARD + size_, original_.data() + CHPR2_GUARD + size_,
                CHPR2_GUARD * sizeof(Complex)),
            0)
            << "trailing guard";
        if (unchanged) {
            EXPECT_EQ(std::memcmp(result.data(), original_.data(), Bytes()), 0) << "read-only/no-op data changed";
        }
        return {result.begin() + CHPR2_GUARD, result.end() - CHPR2_GUARD};
    }

private:
    static void Check(aclError status, const char* name)
    {
        if (status != ACL_SUCCESS) {
            throw std::runtime_error(std::string(name) + " failed: " + std::to_string(status));
        }
    }

    size_t Bytes() const { return original_.size() * sizeof(Complex); }

    size_t size_;
    Complex* base_ = nullptr;
    std::vector<Complex> original_;
};

} // namespace chpr2_arch22_test
