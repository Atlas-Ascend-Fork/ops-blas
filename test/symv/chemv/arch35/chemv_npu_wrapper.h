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

#include <memory>
#include <vector>
#include "cann_ops_blas.h"
#include "chemv_param.h"
#include "device.h"

namespace chemv_test {
struct DeviceData {
    std::unique_ptr<DeviceBuffer> a, x, y;
    static std::unique_ptr<DeviceBuffer> Copy(const std::vector<aclblasComplex>& v)
    {
        if (v.empty()) {
            return nullptr;
        }
        auto b = std::make_unique<DeviceBuffer>(v.size() * sizeof(v[0]));
        b->copyFromHost(v.data(), v.size() * sizeof(v[0]));
        return b;
    }
    DeviceData(
        const std::vector<aclblasComplex>& hostA, const std::vector<aclblasComplex>& hostX,
        const std::vector<aclblasComplex>& hostY)
        : a(Copy(hostA)), x(Copy(hostX)), y(Copy(hostY))
    {}
    aclblasStatus_t Invoke(aclblasHandle_t handle, const ChemvParam& p) const
    {
        return aclblasChemv(
            handle, p.uplo, p.n, p.nullAlpha ? nullptr : &p.alpha, Ptr(a), p.lda, Ptr(x), p.incx,
            p.nullBeta ? nullptr : &p.beta, Ptr(y), p.incy);
    }
    static aclError Synchronize(aclrtStream stream) { return aclrtSynchronizeStream(stream); }
    aclError CopyResult(aclrtStream stream, std::vector<aclblasComplex>& hostY) const
    {
        const auto status = Synchronize(stream);
        if (status != ACL_SUCCESS) {
            return status;
        }
        if (y) {
            y->copyToHost(hostY.data(), hostY.size() * sizeof(hostY[0]));
        }
        return ACL_SUCCESS;
    }
    static aclblasComplex* Ptr(const std::unique_ptr<DeviceBuffer>& b)
    {
        return b ? static_cast<aclblasComplex*>(b->ptr()) : nullptr;
    }
};
} // namespace chemv_test
