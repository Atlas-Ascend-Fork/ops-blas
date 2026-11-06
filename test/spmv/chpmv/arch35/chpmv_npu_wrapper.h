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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "../chpmv_param.h"
#include "device.h"

namespace chpmv_test {
class GuardedDevice {
    std::unique_ptr<DeviceBuffer> buffer_;
    size_t bytes_ = 0, prefix_ = 0;
    static constexpr uint32_t guard_ = 0x4a16b33d;

public:
    Complex* ptr = nullptr;
    GuardedDevice(const std::vector<Complex>& data, size_t prefix)
        : bytes_(data.size() * sizeof(Complex)), prefix_(prefix * sizeof(Complex))
    {
        if (data.empty())
            return;
        buffer_ = std::make_unique<DeviceBuffer>(bytes_ + prefix_ + 128);
        std::vector<uint32_t> guard((prefix_ + 128) / 4, guard_);
        Check(aclrtMemcpy(buffer_->ptr(), prefix_, guard.data(), prefix_, ACL_MEMCPY_HOST_TO_DEVICE));
        ptr = reinterpret_cast<Complex*>(buffer_->bytePtr() + prefix_);
        Check(aclrtMemcpy(ptr, bytes_, data.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
        Check(aclrtMemcpy(buffer_->bytePtr() + prefix_ + bytes_, 128, guard.data(), 128, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    static void Check(aclError status)
    {
        if (status != ACL_SUCCESS)
            throw std::runtime_error("ACL transfer failed: " + std::to_string(status));
    }
    void Read(std::vector<Complex>& data) const
    {
        if (ptr)
            Check(aclrtMemcpy(data.data(), bytes_, ptr, bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    void Write(const std::vector<Complex>& data)
    {
        if (!ptr || data.size() * sizeof(Complex) != bytes_)
            throw std::runtime_error("device buffer size mismatch");
        Check(aclrtMemcpy(ptr, bytes_, data.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    bool GuardsIntact() const
    {
        if (!ptr)
            return true;
        std::vector<uint32_t> head(prefix_ / 4), tail(32);
        Check(aclrtMemcpy(head.data(), prefix_, buffer_->ptr(), prefix_, ACL_MEMCPY_DEVICE_TO_HOST));
        Check(aclrtMemcpy(tail.data(), 128, buffer_->bytePtr() + prefix_ + bytes_, 128, ACL_MEMCPY_DEVICE_TO_HOST));
        return std::all_of(head.begin(), head.end(), [](uint32_t x) { return x == guard_; }) &&
               std::all_of(tail.begin(), tail.end(), [](uint32_t x) { return x == guard_; });
    }
    bool Equals(const std::vector<Complex>& original) const
    {
        if (!ptr)
            return original.empty();
        std::vector<uint8_t> chunk(std::min<size_t>(bytes_, 1024 * 1024));
        for (size_t at = 0; at < bytes_; at += chunk.size()) {
            const size_t length = std::min(chunk.size(), bytes_ - at);
            Check(aclrtMemcpy(
                chunk.data(), chunk.size(), reinterpret_cast<const uint8_t*>(ptr) + at, length,
                ACL_MEMCPY_DEVICE_TO_HOST));
            if (std::memcmp(chunk.data(), reinterpret_cast<const uint8_t*>(original.data()) + at, length))
                return false;
        }
        return true;
    }
};

inline void RunChpmvNpu(
    const Param& p, aclblasHandle_t handle, aclrtStream stream, const chpmv_test::GuardedDevice& dAP,
    const chpmv_test::GuardedDevice& dX, const chpmv_test::GuardedDevice& dY, aclblasStatus_t& status)
{
    const auto* a = p.nullAlpha ? nullptr : &p.alpha;
    const auto* b = p.nullBeta ? nullptr : &p.beta;
    const char* profile = std::getenv("CHPMV_PROFILE_REPEAT");
    const int repeat = profile ? std::stoi(profile) : 1;
    ASSERT_GT(repeat, 0);
    if (profile) {
        ASSERT_EQ(p.caseName.find("TC_PF_"), size_t(0));
        ASSERT_EQ(p.beta.real, 0.0f);
        ASSERT_EQ(p.beta.imag, 0.0f);
    }
    for (int iteration = 0; iteration < repeat; ++iteration) {
        status = aclblasChpmv(handle, p.uplo, p.n, a, dAP.ptr, dX.ptr, p.incx, b, dY.ptr, p.incy);
        if (status != p.expectResult || status != ACLBLAS_STATUS_SUCCESS)
            break;
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
}

} // namespace chpmv_test
