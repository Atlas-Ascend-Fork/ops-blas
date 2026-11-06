/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclblaslt_handle.cpp
 * \brief Public C API: handle lifecycle (create / destroy).
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_handle_impl.h"
#include "aclblaslt_layout_impl.h"
#include "host_utils.h"

#include <acl/acl.h>

extern "C" {

aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* lightHandle)
{
    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    _aclblaslt_handle* h = nullptr;
    auto st = AllocHandle(&h);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    int32_t deviceId = 0;
    aclError aclRet = aclrtGetDevice(&deviceId);
    if (aclRet != ACL_SUCCESS) {
        delete h;
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    aclrtContext currentCtx = nullptr;
    aclRet = aclrtGetCurrentContext(&currentCtx);
    if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
        delete h;
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    h->deviceId = deviceId;
    h->context = currentCtx;
    h->defaultStream = nullptr;
    // issue #822: 不再预分配 32MB internalWorkspace（全仓无读写点，实际 workspace
    // 由调用方按 heuristic 结果分配传入），同步移除仅初始化/销毁的死资源成员。
    h->initialized = true;
    *lightHandle = h;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t lightHandle)
{
    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = lightHandle;
    if (h->magic != ACLBLASLT_HANDLE_MAGIC) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    h->initialized = false;
    return FreeHandle(h);
}

} // extern "C"
