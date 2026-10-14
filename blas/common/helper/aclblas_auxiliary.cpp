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
 * \file aclblas_auxiliary.cpp
 * \brief Auxiliary functions for ops-blas handle lifecycle, stream, workspace, and version info.
 */

#include "cann_ops_blas.h"
#include "aclblas_handle_internal.h"
#include "aclblas_version.h"

#include <new>

extern "C" {

aclblasStatus_t aclblasCreate(aclblasHandle_t* handle)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // Prevent overwriting an existing pointer to avoid memory leaks.
    if (*handle != nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = new (std::nothrow) _aclblas_handle();
    if (h == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t st = AllocateLibraryWorkspace(h, ACLBLAS_DEFAULT_WORKSPACE_SIZE);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        delete h;
        return st;
    }

    *handle = h;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasDestroy(aclblasHandle_t handle)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto* h = handle;

    const aclblasStatus_t syncStatus = SynchronizeHandleStream(h);
    if (syncStatus != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasDestroy", "stream synchronization failed before handle destruction.");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (h->cherk_arch22_metadata != nullptr) {
        aclrtFree(h->cherk_arch22_metadata);
        h->cherk_arch22_metadata = nullptr;
    }
    if (h->cher2k_interleave_offsets != nullptr) {
        aclrtFree(h->cher2k_interleave_offsets);
        h->cher2k_interleave_offsets = nullptr;
    }
    FreeLibraryWorkspace(h);

    h->stream = nullptr;

    delete h;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetStream(aclblasHandle_t handle, aclrtStream stream)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto* h = handle;

    const aclblasStatus_t syncStatus = SynchronizeHandleStream(h);
    if (syncStatus != ACLBLAS_STATUS_SUCCESS) {
        return syncStatus;
    }

    h->stream = stream;

    // Reset to the library workspace when switching streams (cuBLAS-compatible behavior).
    return ResetToDefaultWorkspace(h);
}

aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream* stream)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (stream == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    *stream = h->stream;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetWorkspace(aclblasHandle_t handle, void* workspace, size_t workspaceSize)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto* h = handle;

    if (workspace == nullptr || workspaceSize == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Grow-only policy: keep the current user workspace when the new size is not larger.
    if (h->workspace_owner == AclblasWorkspaceOwner::User && workspaceSize <= h->workspace_size) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasStatus_t syncStatus = SynchronizeHandleStream(h);
    if (syncStatus != ACLBLAS_STATUS_SUCCESS) {
        return syncStatus;
    }

    if (h->workspace_owner == AclblasWorkspaceOwner::Library && h->workspace != nullptr) {
        h->library_workspace_size = h->workspace_size;
        FreeLibraryWorkspace(h);
    }

    h->workspace = workspace;
    h->workspace_size = workspaceSize;
    h->workspace_owner = AclblasWorkspaceOwner::User;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetVersion(aclblasHandle_t handle, int* version)
{
    (void)handle;
    if (version == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    *version = ACLBLAS_VERSION;
    return ACLBLAS_STATUS_SUCCESS;
}

} // extern "C"
