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
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace triangle_test {

// Confirm that the retained Host baseline is the byte-exact pre-call Device snapshot.
inline aclblasStatus_t VerifyDeviceMatrixSnapshot(const void* device, const void* host, std::size_t bytes)
{
    // No output buffer is uploaded for null-pointer and empty-matrix cases.
    if (bytes == 0 || host == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (device == nullptr) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    constexpr std::size_t kSnapshotChunkBytes = 8 * 1024 * 1024;
    std::vector<unsigned char> snapshot(std::min(bytes, kSnapshotChunkBytes));
    const auto* deviceBytes = static_cast<const unsigned char*>(device);
    const auto* hostBytes = static_cast<const unsigned char*>(host);
    for (std::size_t offset = 0; offset < bytes;) {
        const auto count = std::min(bytes - offset, snapshot.size());
        if (aclrtMemcpy(snapshot.data(), snapshot.size(), deviceBytes + offset, count, ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        if (std::memcmp(snapshot.data(), hostBytes + offset, count) != 0) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        offset += count;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace triangle_test
