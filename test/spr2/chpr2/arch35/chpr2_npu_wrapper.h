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

constexpr size_t GUARD = 4;
constexpr Complex CANARY = {12345.0f, -54321.0f};
class Chpr2NpuWrapper {
public:
    explicit Chpr2NpuWrapper(const std::vector<Complex>& input)
        : size_(input.size()), original_(size_ + 2 * GUARD, CANARY)
    {
        std::copy(input.begin(), input.end(), original_.begin() + GUARD);
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
        if (base_)
            aclrtFree(base_);
    }
    Chpr2NpuWrapper(const Chpr2NpuWrapper&) = delete;
    Chpr2NpuWrapper& operator=(const Chpr2NpuWrapper&) = delete;
    Complex* Data() { return base_ + GUARD; }
    std::vector<Complex> Read(bool unchanged = false)
    {
        auto result = original_;
        Check(aclrtMemcpy(result.data(), Bytes(), base_, Bytes(), ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        EXPECT_EQ(std::memcmp(result.data(), original_.data(), GUARD * sizeof(Complex)), 0) << "leading guard";
        EXPECT_EQ(
            std::memcmp(result.data() + GUARD + size_, original_.data() + GUARD + size_, GUARD * sizeof(Complex)), 0)
            << "trailing guard";
        if (unchanged)
            EXPECT_EQ(std::memcmp(result.data(), original_.data(), Bytes()), 0) << "read-only/no-op data changed";
        return {result.begin() + GUARD, result.end() - GUARD};
    }

private:
    static void Check(aclError status, const char* name)
    {
        if (status != ACL_SUCCESS)
            throw std::runtime_error(std::string(name) + " failed: " + std::to_string(status));
    }
    size_t Bytes() const { return original_.size() * sizeof(Complex); }
    size_t size_;
    Complex* base_ = nullptr;
    std::vector<Complex> original_;
};
