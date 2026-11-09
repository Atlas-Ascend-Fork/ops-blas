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
 * \file aclblaslt_matmul_preference.cpp
 * \brief Public C API: matmul heuristic-search preference CRUD.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_layout_impl.h"
#include "host_utils.h"
#include "aclblaslt_logger_impl.h"

#include <algorithm>
#include <cstdint>
#include <new>

namespace {

// cuBLASLt 的 *Init 语义是「重置一个先前已分配的描述符」（"initializes ... in a previously
// allocated one"），因此 Create → Init（重置复用）→ Destroy 是合法模式。仓内 Init 若无条件以
// NSDMI 全新 impl 覆写胶囊，会把 heapOwned 从 1 翻回 0，此后 Destroy 的来源防护永久拒绝该堆
// 胶囊——既泄漏又给出意外的 INVALID_VALUE。故 magic 有效且 heapOwned==1 时保留来源位、只重置
// 属性；调用方存储（magic 不匹配或来源位为 0）仍得到 heapOwned=0，Destroy 照旧拒绝。
//
// 读取前提（公开契约，见 include/cann_ops_blasLt.h 中 Init 的 @param[in,out] pref）：调用方
// 存储入口须为确定值——零初始化，或 Create/Init 产物胶囊。本 helper 会读回该存储前
// sizeof(impl) 字节；未初始化（不确定值）存储属调用方违约（UB）。这与本文件「读不确定值亦属
// UB」的纪律一致：写入侧由 impl 值初始化保证，读取侧由公开契约保证。门控顺序固定：magic 先行、
// 再判 heapOwned，局部缓冲值初始化；全零字节必不匹配 magic，故零初始化存储必走纯重置路径
// （返回 0），只有真正的 Create 产物才保留来源位——自然垃圾误标堆来源需恰好命中 32 位 magic
// 且 heapOwned 非零，且前提已是违约输入。
uint32_t CarriedHeapOwnedFlag(aclblasLtMatmulPreference_t pref)
{
    aclblasLtMatmulPreferenceImpl existing{};
    if (CheckedMemcpyS(&existing, sizeof(existing), pref, sizeof(existing)) != ACLBLAS_STATUS_SUCCESS) {
        return 0U;
    }
    if (existing.magic != ACLBLASLT_PREFERENCE_MAGIC || existing.heapOwned == 0U) {
        return 0U;
    }
    return existing.heapOwned;
}

} // namespace

extern "C" {

aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref)
{
    if (pref == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    *pref = nullptr;
    auto* capsule = new (std::nothrow) aclblasLtMatmulPreferenceOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclblasStatus_t copyStatus = CheckedMemsetS(capsule, sizeof(*capsule), sizeof(*capsule));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }

    // 值初始化（与 Init 路径一致）：消除 magic 与 padding 字节的不确定值，令 Create 产物胶囊
    // 内容确定；heapOwned 置 1 标记堆来源（Destroy 运行期防护的依据，SetAttribute 会透传）。
    aclblasLtMatmulPreferenceImpl impl{};
    impl.heapOwned = 1U;
    copyStatus = PackImplIntoCapsule(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }

    *pref = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceInit(aclblasLtMatmulPreference_t pref)
{
    if (pref == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // 值初始化：impl 会被整体 memcpy 进调用方胶囊，未初始化的 magic 与 padding 字节会让胶囊内容
    // 不确定（读不确定值亦属 UB）。与 aclblaslt_transform_desc.cpp 的 InitTransformDescCapsule 对齐。
    aclblasLtMatmulPreferenceImpl impl{};
    // 对 Create 产物调用 Init 是「纯重置」：属性回到默认值，堆来源位保留，Destroy 资格不丢失。
    impl.heapOwned = CarriedHeapOwnedFlag(pref);
    return PackImplIntoCapsule(pref, sizeof(*pref), &impl, sizeof(impl));
}

aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref)
{
    if (pref == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* capsule = reinterpret_cast<aclblasLtMatmulPreferenceOpaque_t*>(pref);
    // Provenance guard: an Init() product lives in caller storage; deleting such a pointer is
    // UB and previously crashed only after this function had already returned SUCCESS.
    // heapOwned is set by Create() and round-trips Set/GetAttribute. (No magic check here,
    // unlike aclblasLtMatrixTransformDescDestroy: that file already validates the magic in
    // UnpackTransformDesc, so its Destroy mirrors an existing pattern; this file has none.)
    aclblasLtMatmulPreferenceImpl impl{};
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), pref, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (impl.heapOwned == 0U) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulPreferenceDestroy",
            "capsule was not allocated by aclblasLtMatmulPreferenceCreate (heapOwned=0); "
            "Init-produced preferences must not be passed to Destroy");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    delete capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(
    aclblasLtMatmulPreference_t pref, aclblasLtMatmulPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    if (pref == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulPreferenceImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), pref, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    switch (attr) {
        case ACLBLASLT_MATMUL_PREF_SEARCH_MODE: {
            if (sizeInBytes != sizeof(uint32_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            uint32_t v = 0;
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(v));
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            // 0=heuristic, 1=exhaustive, 2=fast
            if (v > 2) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.searchMode = v;
            break;
        }

        case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: {
            if (sizeInBytes != sizeof(size_t) && sizeInBytes != sizeof(uint64_t)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            size_t v = 0;
            const size_t copyBytes = std::min(sizeInBytes, sizeof(v));
            copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, copyBytes);
            if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
                return copyStatus;
            }
            if (v > INT64_MAX) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.maxWorkspaceBytes = v;
            break;
        }

        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    copyStatus = CheckedMemcpyS(pref, sizeof(*pref), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceGetAttribute(
    aclblasLtMatmulPreference_t pref, aclblasLtMatmulPreferenceAttribute_t attr, void* buf, size_t sizeInBytes,
    size_t* sizeWritten)
{
    if (pref == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulPreferenceImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), pref, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    size_t requiredSize = 0;
    const void* srcPtr = nullptr;

    switch (attr) {
        case ACLBLASLT_MATMUL_PREF_SEARCH_MODE:
            requiredSize = sizeof(impl.searchMode);
            srcPtr = &impl.searchMode;
            break;

        case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            requiredSize = sizeof(impl.maxWorkspaceBytes);
            srcPtr = &impl.maxWorkspaceBytes;
            break;

        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    return CopyAttributeOut(buf, sizeInBytes, srcPtr, requiredSize, sizeWritten);
}

} // extern "C"
