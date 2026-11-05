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
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclblaslt_algo_heuristic.h"
#include "aclblaslt_layout_impl.h"
#include "lt_matmul_param.h"
#include "lt_matmul_copy.h"

namespace LtMatmul22Test {
inline void AclCheck(aclError result, const char* operation)
{
    if (result != ACL_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": ACL error " + std::to_string(result));
}

inline void BlasCheck(aclblasStatus_t result, const char* operation)
{
    if (result != ACLBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": BLAS status " + std::to_string(result));
}

constexpr size_t GUARD_BYTES = 128;
constexpr uint8_t GUARD_BYTE = 0x5a;
constexpr float PADDING_SENTINEL = -19023.25f;

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if (base_ != nullptr)
            aclrtFree(base_);
    }

    void Initialize(size_t bytes, const void* initial = nullptr)
    {
        bytes_ = bytes;
        if (bytes == 0)
            return;
        AclCheck(aclrtMalloc(&base_, bytes + 2 * GUARD_BYTES, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        AclCheck(
            aclrtMemset(base_, bytes + 2 * GUARD_BYTES, GUARD_BYTE, bytes + 2 * GUARD_BYTES), "aclrtMemset guards");
        if (initial != nullptr)
            AclCheck(aclrtMemcpy(Data(), bytes, initial, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D matrix");
    }

    void* Data() const { return base_ == nullptr ? nullptr : static_cast<uint8_t*>(base_) + GUARD_BYTES; }
    size_t Bytes() const { return bytes_; }
    void Download(std::vector<float>& output) const
    {
        if (bytes_ != 0)
            AclCheck(
                aclrtMemcpy(output.data(), output.size() * sizeof(float), Data(), bytes_, ACL_MEMCPY_DEVICE_TO_HOST),
                "D2H matrix");
    }
    void CheckGuards() const
    {
        if (base_ == nullptr)
            return;
        std::array<uint8_t, GUARD_BYTES> guard{};
        for (size_t offset : {size_t(0), bytes_ + GUARD_BYTES}) {
            AclCheck(
                aclrtMemcpy(
                    guard.data(), guard.size(), static_cast<uint8_t*>(base_) + offset, guard.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST),
                "D2H guards");
            if (!std::all_of(guard.begin(), guard.end(), [](uint8_t byte) { return byte == GUARD_BYTE; })) {
                throw std::runtime_error("Device allocation canary changed at offset " + std::to_string(offset));
            }
        }
    }

private:
    void* base_ = nullptr;
    size_t bytes_ = 0;
};

struct Descriptors {
    aclblasLtMatmulDesc_t compute = nullptr;
    std::array<aclblasLtMatrixLayout_t, 4> layout{};
    aclblasLtMatmulPreference_t preference = nullptr;
    Descriptors() = default;
    Descriptors(const Descriptors&) = delete;
    Descriptors& operator=(const Descriptors&) = delete;
    ~Descriptors()
    {
        for (auto item : layout)
            if (item != nullptr)
                aclblasLtMatrixLayoutDestroy(item);
        if (compute != nullptr)
            aclblasLtMatmulDescDestroy(compute);
        if (preference != nullptr)
            aclblasLtMatmulPreferenceDestroy(preference);
    }

    void InitializeLayouts(const Param& p)
    {
        const auto rows = p.Rows(), cols = p.Cols();
        for (size_t i = 0; i < layout.size(); ++i) {
            BlasCheck(
                aclblasLtMatrixLayoutCreate(
                    &layout[i], p.dtype[i], static_cast<uint64_t>(rows[i]), static_cast<uint64_t>(cols[i]),
                    std::max<int64_t>(0, p.ld[i])),
                "LayoutCreate");
            BlasCheck(
                aclblasLtMatrixLayoutSetAttribute(layout[i], ACLBLASLT_MATRIX_LAYOUT_ORDER, &p.order, sizeof(p.order)),
                "set order");
            BlasCheck(
                aclblasLtMatrixLayoutSetAttribute(
                    layout[i], ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &p.batch, sizeof(p.batch)),
                "set batchCount");
            BlasCheck(
                aclblasLtMatrixLayoutSetAttribute(
                    layout[i], ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &p.stride, sizeof(p.stride)),
                "set stride");
            if (p.ld[i] < 0) {
                aclblasLtMatrixLayoutImpl impl{};
                CopyBytes(&impl, sizeof(impl), layout[i], sizeof(impl));
                impl.ld = p.ld[i];
                CopyBytes(layout[i], sizeof(impl), &impl, sizeof(impl));
            }
        }
    }

    void Initialize(const Param& p)
    {
        BlasCheck(
            aclblasLtMatmulDescCreate(&compute, static_cast<aclblasComputeType_t>(p.computeType), p.scaleType),
            "MatmulDescCreate");
        BlasCheck(
            aclblasLtMatmulDescSetAttribute(compute, ACLBLASLT_MATMUL_DESC_TRANSA, &p.transA, sizeof(p.transA)),
            "set transA");
        BlasCheck(
            aclblasLtMatmulDescSetAttribute(compute, ACLBLASLT_MATMUL_DESC_TRANSB, &p.transB, sizeof(p.transB)),
            "set transB");
        if (p.pointerMode == ACLBLASLT_POINTER_MODE_HOST || p.pointerMode == ACLBLASLT_POINTER_MODE_DEVICE) {
            BlasCheck(
                aclblasLtMatmulDescSetAttribute(
                    compute, ACLBLASLT_MATMUL_DESC_POINTER_MODE, &p.pointerMode, sizeof(p.pointerMode)),
                "set pointerMode");
        } else {
            // White-box malformed-capsule negative: the public setter rejects 999 before Matmul.
            // Do not substitute that setter result for the CSV's actual Matmul invocation.
            aclblasLtMatmulDescImpl impl{};
            CopyBytes(&impl, sizeof(impl), compute, sizeof(impl));
            impl.pointerMode = p.pointerMode;
            CopyBytes(compute, sizeof(impl), &impl, sizeof(impl));
        }
        InitializeLayouts(p);
        BlasCheck(aclblasLtMatmulPreferenceCreate(&preference), "PreferenceCreate");
        BlasCheck(
            aclblasLtMatmulPreferenceSetAttribute(
                preference, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p.preferenceBytes, sizeof(p.preferenceBytes)),
            "set workspace preference");
    }
};

} // namespace LtMatmul22Test
