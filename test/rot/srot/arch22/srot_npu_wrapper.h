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

#include <string>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "srot_device_buffer.h"

namespace srot_arch22_test {
constexpr size_t GUARD_ELEMENTS = 8;
constexpr float GUARD_VALUE = -123456.75f;

// The caller provides a complete guarded physical span, or a guarded single-element
// dummy for invalid arguments. Scalar memory location is independent for c and s.
inline aclblasStatus_t Execute(
    aclblasHandle_t handle, aclrtStream stream, int n, int incx, int incy, std::vector<float>& x, std::vector<float>& y,
    float c, float s, const std::string& mode, bool nullX, bool nullY, bool nullC = false, bool nullS = false,
    int iterations = 1)
{
    SrotDeviceBuffer dx(x.size());
    SrotDeviceBuffer dy(y.size());
    SrotDeviceBuffer dc(1);
    SrotDeviceBuffer ds(1);
    dx.Write(x.data());
    dy.Write(y.data());
    dc.Write(&c);
    ds.Write(&s);
    const bool cDevice = mode == "device" || mode == "mixed_reverse";
    const bool sDevice = mode == "device" || mode == "mixed";
    const float* cp = nullC ? nullptr : cDevice ? dc.Data() : &c;
    const float* sp = nullS ? nullptr : sDevice ? ds.Data() : &s;
    aclblasStatus_t result = ACLBLAS_STATUS_SUCCESS;
    for (int i = 0; i < iterations; ++i) {
        result = aclblasSrot(
            handle, n, nullX ? nullptr : dx.Data(GUARD_ELEMENTS), incx, nullY ? nullptr : dy.Data(GUARD_ELEMENTS), incy,
            cp, sp);
        if (result != ACLBLAS_STATUS_SUCCESS)
            break;
    }
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    dx.Read(x.data());
    dy.Read(y.data());
    return result;
}
} // namespace srot_arch22_test
