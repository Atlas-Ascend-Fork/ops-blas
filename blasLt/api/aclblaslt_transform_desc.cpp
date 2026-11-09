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
 * \file aclblaslt_transform_desc.cpp
 * \brief Public C API: matrix transform descriptor CRUD.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_capsule.h"
#include "host_utils.h"
#include "matrix_transform_engine.h"
#include "aclblaslt_logger_impl.h"

#include <cstdint>
#include <new>

namespace {

// Shared Set/Get prologue: reject null args, copy the capsule into a stack impl, and reject a
// corrupted / foreign descriptor via the magic check. Get 路径不要求 buf 非空——探测模式
// （sizeInBytes==0、buf 为空、sizeWritten 回写所需大小）由 CopyAttributeOut 统一处理（issue #664）。
aclblasStatus_t UnpackTransformDesc(
    const aclblasLtMatrixTransformDesc_t transformDesc, const void* buf, aclblasLtMatrixTransformDescImpl& impl,
    bool requireBuf)
{
    if (transformDesc == nullptr || (requireBuf && buf == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), transformDesc, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (impl.magic != ACLBLASLT_TRANSFORM_DESC_MAGIC) {
        return ACLBLAS_STATUS_INVALID_VALUE; // corrupted / foreign descriptor
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// cuBLASLt 的 *Init 语义是「重置一个先前已分配的描述符」（"initializes ... in a previously
// allocated one"），故 Create → Init（重置复用）→ Destroy 是合法模式。Init 若无条件以 NSDMI
// 默认值覆写胶囊，会把 heapOwned 从 1 翻回 0，此后 Destroy 的来源防护永久拒绝该堆胶囊——既
// 泄漏又给出意外的 INVALID_VALUE。magic 有效且 heapOwned==1 时保留来源位、只重置属性；调用方
// 存储（magic 不匹配或来源位为 0）仍得到 heapOwned=0，Destroy 照旧拒绝。与
// aclblaslt_matmul_preference.cpp 的同名 helper 口径一致。
//
// 读取前提（公开契约，见 include/cann_ops_blasLt.h 中 Init 的 \param[in,out] transformDesc）：
// 调用方存储入口须为确定值——零初始化，或 Create/Init 产物胶囊。本 helper 会读回该存储前
// sizeof(impl) 字节；未初始化（不确定值）存储属调用方违约（UB）。门控顺序固定：magic 先行、
// 再判 heapOwned，局部缓冲值初始化；全零字节必不匹配 magic，故零初始化存储必走纯重置路径
// （返回 0），只有真正的 Create 产物才保留来源位——自然垃圾误标堆来源需恰好命中 32 位 magic
// 且 heapOwned 非零，且前提已是违约输入。
uint32_t CarriedHeapOwnedFlag(const aclblasLtMatrixTransformDescOpaque_t* capsule)
{
    aclblasLtMatrixTransformDescImpl existing{};
    if (CheckedMemcpyS(&existing, sizeof(existing), capsule, sizeof(existing)) != ACLBLAS_STATUS_SUCCESS) {
        return 0U;
    }
    if (existing.magic != ACLBLASLT_TRANSFORM_DESC_MAGIC || existing.heapOwned == 0U) {
        return 0U;
    }
    return existing.heapOwned;
}

// In-place initializer of a caller-allocated transform desc capsule. Mirrors the Create path's
// Impl packing (magic + scaleType + in-place trans), with the NSDMI defaults (pointerMode = host)
// picked up identically, so an Init capsule carries the same attribute values as a Create +
// SetAttribute one (the heapOwned provenance flag is carried over rather than reset, see above).
aclblasStatus_t InitTransformDescCapsule(aclblasLtMatrixTransformDescOpaque_t* capsule, aclDataType scaleType)
{
    // magic and scaleType have no NSDMI and must be assigned explicitly; pointerMode and
    // transA/transB keep their NSDMI defaults (host / N), matching the Create path attribute-wise.
    aclblasLtMatrixTransformDescImpl impl;
    impl.magic = ACLBLASLT_TRANSFORM_DESC_MAGIC;
    impl.scaleType = scaleType;
    impl.heapOwned = CarriedHeapOwnedFlag(capsule);
    static_assert(sizeof(impl) <= sizeof(*capsule),
                  "aclblasLtMatrixTransformDescImpl too large, not fit in capsule!");
    return PackImplIntoCapsule(capsule, sizeof(*capsule), &impl, sizeof(impl));
}

// Set 期值域守卫。集中在此（而非内联在 switch 里）既让 SetAttribute 保持「prologue + 分派」的
// 扁平形态、远离函数行数上限，也让 TRANSA/TRANSB 两个同构分支共用一个谓词。

// issue #788: scaleType 在 Set 阶段可覆盖，但值域须为变换引擎可消费的精度
// （FP32/INT32/FP8 路径用 FLOAT/INT32 scale，FP4 路径用 BF16/FLOAT，见
// matrix_transform_host.cpp 的 MatScaleTypeMatchesPath），拒绝越界 aclDataType 直写
// （如 ACL_DOUBLE），避免成功构造后 transform 阶段才失败。
bool IsStorableScaleType(int32_t value)
{
    return value == static_cast<int32_t>(ACL_FLOAT) || value == static_cast<int32_t>(ACL_BF16) ||
           value == static_cast<int32_t>(ACL_INT32);
}

// TRANSA/TRANSB 的 Set 期白名单，与变换引擎运行期的 MatIsValidOp 同域（N/T/C，C 按 T 处理，
// 本库无复数 dtype）：越界取值在 Set 阶段就拒绝，消除「成功构造、Transform 阶段才失败」，与
// SCALE_TYPE 的收紧口径（issue #788）和 MatrixLayout ORDER 的 Set 期白名单（issue #783）拉齐。
bool IsStorableTransformOp(int32_t value)
{
    return value == static_cast<int32_t>(ACLBLAS_OP_N) || value == static_cast<int32_t>(ACLBLAS_OP_T) ||
           value == static_cast<int32_t>(ACLBLAS_OP_C);
}

// Set 路径的属性分派：校验值域后落盘。拒绝码按属性区分——值越界用 INVALID_VALUE（本仓 Set 期
// 值域校验的既有口径），未实现的 pointer mode 与未识别属性用 NOT_SUPPORTED。
aclblasStatus_t ApplyTransformDescAttribute(
    aclblasLtMatrixTransformDescAttribute_t attr, int32_t value, aclblasLtMatrixTransformDescImpl& impl)
{
    switch (attr) {
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
            if (!IsStorableScaleType(value)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.scaleType = static_cast<aclDataType>(value);
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
            // cuBLASLt 对齐：本库的标量执行链恒按主机指针解引用 alpha/beta（MatResolveScalars），
            // 只实现 HOST 指针模式。非 HOST 取值在 Set 阶段就响亮拒绝（NOT_SUPPORTED，与
            // MatTransformLaunch 的运行期守卫同返回码），而不是记录后静默把设备地址当主机指针
            // 解读——那会产生错误系数甚至段错误，且错误现场远离根因。
            if (value != static_cast<int32_t>(ACLBLASLT_POINTER_MODE_HOST)) {
                return ACLBLAS_STATUS_NOT_SUPPORTED;
            }
            impl.pointerMode = value;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
            if (!IsStorableTransformOp(value)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.transA = static_cast<aclblasOperation_t>(value);
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
            if (!IsStorableTransformOp(value)) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            impl.transB = static_cast<aclblasOperation_t>(value);
            return ACLBLAS_STATUS_SUCCESS;
        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
}

} // namespace

extern "C" {

aclblasStatus_t aclblasLtMatrixTransformDescCreate(aclblasLtMatrixTransformDesc_t* transformDesc, aclDataType scaleType)
{
    if (transformDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *transformDesc = nullptr;

    auto* capsule = new (std::nothrow) aclblasLtMatrixTransformDescOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasLtMatrixTransformDescImpl impl;
    impl.magic = ACLBLASLT_TRANSFORM_DESC_MAGIC;
    impl.scaleType = scaleType;
    impl.heapOwned = 1U;  // heap provenance: this capsule may be Destroy()ed

    aclblasStatus_t copyStatus = MatPackTransformImpl(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }

    *transformDesc = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

// Parameter face mirrors cuBLASLt 13.x cublasLtMatrixTransformDescInit(transformDesc, scaleType):
// by value, two parameters. transa/transb are NOT Init parameters there (nor here) -- they are set
// afterwards via aclblasLtMatrixTransformDescSetAttribute(TRANSA/TRANSB), the same path Create()
// users take. The handle is passed by value like the repo's aclblasLtMatmulDescInit /
// aclblasLtMatmulPreferenceInit (all mirroring cuBLASLt's *Init convention); a null handle is a
// NULL pointer, so a single check covers it.
aclblasStatus_t aclblasLtMatrixTransformDescInit(
    aclblasLtMatrixTransformDesc_t transformDesc, aclDataType scaleType)
{
    if (transformDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return InitTransformDescCapsule(transformDesc, scaleType);
}

aclblasStatus_t aclblasLtMatrixTransformDescDestroy(const aclblasLtMatrixTransformDesc_t transformDesc)
{
    if (transformDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* capsule = reinterpret_cast<aclblasLtMatrixTransformDescOpaque_t*>(transformDesc);
    // Provenance guard: an Init() product lives in caller storage (stack / static / member);
    // deleting such a pointer is UB (heap metadata corruption / SIGABRT) and previously the
    // crash happened only after this function had already returned SUCCESS. The heapOwned flag
    // set by Create() distinguishes the two provenances and round-trips SetAttribute. Also
    // rejects corrupted / foreign capsules via the magic check (mirrors UnpackTransformDesc).
    aclblasLtMatrixTransformDescImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), capsule, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (impl.magic != ACLBLASLT_TRANSFORM_DESC_MAGIC) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatrixTransformDescDestroy",
            "descriptor failed the magic check (corrupted / foreign capsule)");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (impl.heapOwned == 0U) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatrixTransformDescDestroy",
            "capsule was not allocated by aclblasLtMatrixTransformDescCreate (heapOwned=0); "
            "Init-produced descriptors must not be passed to Destroy");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    delete capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixTransformDescSetAttribute(
    aclblasLtMatrixTransformDesc_t transformDesc, aclblasLtMatrixTransformDescAttribute_t attr, const void* buf,
    size_t sizeInBytes)
{
    aclblasLtMatrixTransformDescImpl impl;
    aclblasStatus_t copyStatus = UnpackTransformDesc(transformDesc, buf, impl, true);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int32_t v = 0;
    copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(int32_t));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    // 校验顺序保持「判空 + magic → size → attr」不变：attr 识别与值域校验都在 size 之后，
    // 故非法 size 搭配未识别 attr 仍返回 INVALID_VALUE。
    copyStatus = ApplyTransformDescAttribute(attr, v, impl);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    return MatPackTransformImpl(transformDesc, sizeof(*transformDesc), &impl, sizeof(impl));
}

aclblasStatus_t aclblasLtMatrixTransformDescGetAttribute(
    aclblasLtMatrixTransformDesc_t transformDesc, aclblasLtMatrixTransformDescAttribute_t attr, void* buf,
    size_t sizeInBytes, size_t* sizeWritten)
{
    aclblasLtMatrixTransformDescImpl impl;
    aclblasStatus_t copyStatus = UnpackTransformDesc(transformDesc, buf, impl, false);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    const void* srcPtr = nullptr;
    switch (attr) {
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
            srcPtr = &impl.scaleType;
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
            srcPtr = &impl.pointerMode;
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
            srcPtr = &impl.transA;
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
            srcPtr = &impl.transB;
            break;
        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    const size_t requiredSize = sizeof(int32_t);
    return CopyAttributeOut(buf, sizeInBytes, srcPtr, requiredSize, sizeWritten);
}

} // extern "C"
