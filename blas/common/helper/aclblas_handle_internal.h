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
 * \file aclblas_handle_internal.h
 * \brief Internal ops-blas handle structure (not exposed to users).
 */

#pragma once

#include <acl/acl.h>
#include <algorithm>
#include <cstddef>
#include "log/log.h"

#include "cann_ops_blas_common.h"

/** @brief Default initial workspace size: 32 MiB. */
constexpr size_t ACLBLAS_DEFAULT_WORKSPACE_SIZE = 32U * 1024U * 1024U;

/** @brief Maximum workspace size: 2 GiB. Allocations beyond this fail. */
constexpr size_t ACLBLAS_MAX_WORKSPACE_SIZE = 2ULL * 1024ULL * 1024ULL * 1024ULL;

/** @brief Ownership of the active workspace buffer. */
enum class AclblasWorkspaceOwner {
    Library, /**< Allocated and freed by the library; may grow in place. */
    User,    /**< Injected by the user via aclblasSetWorkspace; never freed by the library. */
};

/**
 * @brief Internal ops-blas handle structure.
 *
 * Maintains a single active workspace buffer. When the user switches back to the
 * library workspace, the library re-allocates using library_workspace_size rather
 * than keeping two device buffers resident.
 *
 * This structure is visible only inside implementation files. Public APIs use
 * a forward-declared opaque pointer (struct _aclblas_handle*).
 */
struct _aclblas_handle {
    /* ========== Stream ========== */
    aclrtStream stream = nullptr;

    /* ========== Active workspace ========== */
    void* workspace = nullptr;
    size_t workspace_size = 0;
    AclblasWorkspaceOwner workspace_owner = AclblasWorkspaceOwner::Library;

    /** Size of the last library-managed workspace buffer (preserved across user switches). */
    size_t library_workspace_size = 0;

    // Immutable CHERK transpose/interleave/mask metadata. It is initialized
    // lazily and shared by every AIV block instead of being rebuilt per launch.
    void* cherk_arch22_metadata = nullptr;

    // Pointer-location metadata only; scalar values are never cached.  Tiny
    // Cher2k kernels read device alpha/beta on every launch so in-place scalar
    // updates retain BLAS semantics without repeated pointer queries.
    bool cher2k_scalar_location_valid = false;
    const void* cher2k_scalar_alpha_ptr = nullptr;
    const void* cher2k_scalar_beta_ptr = nullptr;
    bool cher2k_scalar_alpha_device = false;
    bool cher2k_scalar_beta_device = false;

    // Immutable planar-to-AoS gather metadata shared by Cher2k fast paths.
    // It is initialized lazily on first use and released with the handle.
    void* cher2k_interleave_offsets = nullptr;
};

/**
 * @brief Returns the pointer of the currently active workspace.
 */
inline void* GetEffectiveWorkspace(const _aclblas_handle* h)
{
    if (h == nullptr) {
        return nullptr;
    }
    return h->workspace;
}

/**
 * @brief Returns the size in bytes of the currently active workspace.
 */
inline size_t GetEffectiveWorkspaceSize(const _aclblas_handle* h)
{
    if (h == nullptr) {
        return 0;
    }
    return h->workspace_size;
}

/**
 * @brief check the size in bytes of the currently active workspace.
 */
inline bool CheckEffectiveWorkspaceSize(const _aclblas_handle* h, size_t workSize)
{
    size_t availableBytes = GetEffectiveWorkspaceSize(h);
    OP_CHECK_IF(
        availableBytes < workSize,
        OP_LOGE(
            "aclblasHandle",
            "workspace required %zu bytes, but only %zu bytes available. "
            "Please call aclblasSetWorkspace with size >= %zu bytes",
            workSize, availableBytes, workSize),
        return false);
    return true;
}

inline aclblasStatus_t SynchronizeHandleStream(_aclblas_handle* h)
{
    if (h == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclError aclRet = aclrtSynchronizeStream(h->stream);
    return aclRet == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
}

inline void FreeLibraryWorkspace(_aclblas_handle* h)
{
    if (h == nullptr || h->workspace_owner != AclblasWorkspaceOwner::Library || h->workspace == nullptr) {
        return;
    }
    aclrtFree(h->workspace);
    h->workspace = nullptr;
    h->workspace_size = 0;
}

inline aclblasStatus_t AllocateLibraryWorkspace(_aclblas_handle* h, size_t size)
{
    if (h == nullptr || size == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    void* ptr = nullptr;
    const aclError aclRet = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    h->workspace = ptr;
    h->workspace_size = size;
    h->library_workspace_size = size;
    h->workspace_owner = AclblasWorkspaceOwner::Library;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ResetToDefaultWorkspace(_aclblas_handle* h)
{
    if (h == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (h->workspace_owner == AclblasWorkspaceOwner::Library && h->workspace != nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasStatus_t syncStatus = SynchronizeHandleStream(h);
    if (syncStatus != ACLBLAS_STATUS_SUCCESS) {
        return syncStatus;
    }

    h->workspace = nullptr;
    h->workspace_size = 0;
    h->workspace_owner = AclblasWorkspaceOwner::Library;

    const size_t allocSize = h->library_workspace_size > 0 ? h->library_workspace_size : ACLBLAS_DEFAULT_WORKSPACE_SIZE;
    return AllocateLibraryWorkspace(h, allocSize);
}

/**
 * @brief Ensures the active library workspace is at least @p requiredSize bytes.
 *
 * User-owned workspace is never reallocated; requests beyond the current size fail.
 * Library-owned workspace grows in place via free + malloc with a doubling policy.
 */
inline aclblasStatus_t EnsureDefaultWorkspace(_aclblas_handle* h, size_t requiredSize)
{
    if (h == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (h->workspace_owner == AclblasWorkspaceOwner::User) {
        if (requiredSize > h->workspace_size) {
            OP_LOGE(
                "aclblasHandle", "user workspace too small: required=%zu, available=%zu", requiredSize,
                h->workspace_size);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (requiredSize <= h->workspace_size) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (requiredSize > ACLBLAS_MAX_WORKSPACE_SIZE) {
        OP_LOGE(
            "aclblasHandle", "workspace required %zu bytes exceeds maximum limit %zu bytes", requiredSize,
            ACLBLAS_MAX_WORKSPACE_SIZE);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t syncStatus = SynchronizeHandleStream(h);
    if (syncStatus != ACLBLAS_STATUS_SUCCESS) {
        return syncStatus;
    }

    const size_t doubledSize = h->workspace_size > 0 ? h->workspace_size * 2 : ACLBLAS_DEFAULT_WORKSPACE_SIZE;
    const size_t newSize = std::min(std::max(requiredSize, doubledSize), ACLBLAS_MAX_WORKSPACE_SIZE);

    OP_LOGW(
        "aclblasHandle", "library workspace (%zu bytes) insufficient, expanding to %zu bytes", h->workspace_size,
        newSize);

    FreeLibraryWorkspace(h);

    const aclblasStatus_t allocStatus = AllocateLibraryWorkspace(h, newSize);
    if (allocStatus != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasHandle", "workspace expansion to %zu bytes failed", newSize);
        return allocStatus;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
