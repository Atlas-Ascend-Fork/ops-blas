/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "ctrsv_param.h"
#include "device.h"

namespace ctrsv_test {
inline aclblasStatus_t CtrsvInvoke(aclblasHandle_t handle, const CtrsvParam& p, const void* a, void* x)
{
    return aclblasCtrsv(
        handle, p.uplo, p.trans, p.diag, p.n, static_cast<const aclblasComplex*>(a), p.lda,
        static_cast<aclblasComplex*>(x), p.incx);
}

// DeviceBuffer provides RAII allocation and checked H2D/D2H transfers.
// Keep device storage alive until the bound stream has completed.
class CtrsvDeviceInput {
public:
    CtrsvDeviceInput(const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& x)
        : a_(a.size() * sizeof(aclblasComplex)), x_(x.size() * sizeof(aclblasComplex))
    {
        a_.copyFromHost(a.data(), a_.size());
        x_.copyFromHost(x.data(), x_.size());
    }
    aclblasStatus_t Run(aclblasHandle_t handle, aclrtStream stream, const CtrsvParam& p)
    {
        auto status = CtrsvInvoke(handle, p, a_.ptr(), x_.ptr());
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        return aclrtSynchronizeStream(stream) == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    void Read(std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& x)
    {
        a_.copyToHost(a.data(), a_.size());
        x_.copyToHost(x.data(), x_.size());
    }

private:
    DeviceBuffer a_;
    DeviceBuffer x_;
};
} // namespace ctrsv_test
