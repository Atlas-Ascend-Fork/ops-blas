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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "../chpmv_param.h"
#include "device.h"

namespace chpmv_arch22_test {

using chpmv_test::Complex;

class ChpmvNpuWrapper {
public:
    Complex* ptr = nullptr;

    ChpmvNpuWrapper(const std::vector<Complex>& data, size_t prefix)
        : bytes_(data.size() * sizeof(Complex)), prefix_(prefix * sizeof(Complex))
    {
        if (data.empty()) {
            return;
        }
        buffer_ = std::make_unique<DeviceBuffer>(bytes_ + prefix_ + 128U);
        std::vector<uint32_t> guard((prefix_ + 128U) / 4U, GUARD_);
        Check(aclrtMemcpy(buffer_->ptr(), prefix_, guard.data(), prefix_, ACL_MEMCPY_HOST_TO_DEVICE));
        ptr = reinterpret_cast<Complex*>(buffer_->bytePtr() + prefix_);
        Check(aclrtMemcpy(ptr, bytes_, data.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
        Check(aclrtMemcpy(buffer_->bytePtr() + prefix_ + bytes_, 128U, guard.data(), 128U, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    static void Check(aclError status)
    {
        if (status != ACL_SUCCESS) {
            throw std::runtime_error("ACL transfer failed: " + std::to_string(status));
        }
    }

    void Read(std::vector<Complex>& data) const
    {
        if (ptr != nullptr) {
            Check(aclrtMemcpy(data.data(), bytes_, ptr, bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
        }
    }

    bool GuardsIntact() const
    {
        if (ptr == nullptr) {
            return true;
        }
        std::vector<uint32_t> head(prefix_ / 4U);
        std::vector<uint32_t> tail(32U);
        Check(aclrtMemcpy(head.data(), prefix_, buffer_->ptr(), prefix_, ACL_MEMCPY_DEVICE_TO_HOST));
        Check(aclrtMemcpy(tail.data(), 128U, buffer_->bytePtr() + prefix_ + bytes_, 128U, ACL_MEMCPY_DEVICE_TO_HOST));
        return std::all_of(head.begin(), head.end(), [](uint32_t v) { return v == GUARD_; }) &&
               std::all_of(tail.begin(), tail.end(), [](uint32_t v) { return v == GUARD_; });
    }

    bool Equals(const std::vector<Complex>& original) const
    {
        if (ptr == nullptr) {
            return original.empty();
        }
        std::vector<uint8_t> chunk(std::min<size_t>(bytes_, 1024U * 1024U));
        for (size_t at = 0; at < bytes_; at += chunk.size()) {
            const size_t length = std::min(chunk.size(), bytes_ - at);
            Check(aclrtMemcpy(
                chunk.data(), chunk.size(), reinterpret_cast<const uint8_t*>(ptr) + at, length,
                ACL_MEMCPY_DEVICE_TO_HOST));
            if (std::memcmp(chunk.data(), reinterpret_cast<const uint8_t*>(original.data()) + at, length) != 0) {
                return false;
            }
        }
        return true;
    }

private:
    std::unique_ptr<DeviceBuffer> buffer_;
    size_t bytes_ = 0;
    size_t prefix_ = 0;
    static constexpr uint32_t GUARD_ = 0x4a16b33d;
};

inline void RunChpmvNpu(
    const chpmv_test::Param& p, aclblasHandle_t handle, aclrtStream stream, const ChpmvNpuWrapper& dAP,
    const ChpmvNpuWrapper& dX, const ChpmvNpuWrapper& dY, aclblasStatus_t& status)
{
    const auto* a = p.nullAlpha ? nullptr : &p.alpha;
    const auto* b = p.nullBeta ? nullptr : &p.beta;
    status = aclblasChpmv(handle, p.uplo, p.n, a, dAP.ptr, dX.ptr, p.incx, b, dY.ptr, p.incy);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
}

} // namespace chpmv_arch22_test
