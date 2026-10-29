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
 * \file aclblaslt_layout_impl.h
 * \brief Internal capsule-backed impl structs for MatrixLayout / MatmulDesc / MatmulPreference,
 *        plus their capsule magics and shared defaults. Not installed.
 */

#pragma once

#include "cann_ops_blasLt.h"

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

constexpr uint32_t ACLBLASLT_LAYOUT_MAGIC = 0xACBB1234;
constexpr uint32_t ACLBLASLT_DESC_MAGIC = 0xACBC1234;
constexpr uint32_t ACLBLASLT_PREFERENCE_MAGIC = 0xACBD1234;

constexpr size_t DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024;
constexpr size_t BIAS_PTR_STORAGE_BYTES = sizeof(void*);

struct aclblasLtMatrixLayoutImpl {
    uint32_t magic;
    aclDataType type;
    uint64_t rows;
    uint64_t cols;
    int64_t ld;
    aclblasLtOrder_t order = ACLBLASLT_ORDER_COL;
    int32_t batchCount = 1;
    int64_t stridedBatchOffset = 0;
};
static_assert(
    sizeof(aclblasLtMatrixLayoutImpl) <= sizeof(aclblasLtMatrixLayoutOpaque_t),
    "Impl of aclblasLtMatrixLayout must fit in capsule!");

struct aclblasLtMatmulDescImpl {
    uint32_t magic;
    aclblasComputeType_t computeType;
    aclDataType scaleType;
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
    const void* bias = nullptr;
    aclDataType biasDataType = ACL_DT_UNDEFINED;
    const void* scaleA = nullptr;
    const void* scaleB = nullptr;
    int32_t pointerMode = ACLBLASLT_POINTER_MODE_HOST;
    // aclblasLtMatmulMatrixScale_t values (kept as raw int32 until the public
    // enum lands in the header); persisted so GetAttributes round-trips Set.
    int32_t aScaleMode = 0;
    int32_t bScaleMode = 0;
};
static_assert(
    sizeof(aclblasLtMatmulDescImpl) <= sizeof(aclblasLtMatmulDescOpaque_t),
    "Impl of aclblasLtMatmulDesc must fit in capsule!");

struct aclblasLtMatmulPreferenceImpl {
    uint32_t magic = ACLBLASLT_PREFERENCE_MAGIC;
    uint32_t searchMode = 0;
    size_t maxWorkspaceBytes = DEFAULT_WORKSPACE_SIZE;
    int32_t maxResults = 3;
    bool allowMixedPrecision = true;
    bool allowSplitK = true;
    // tiling
    uint32_t preferredL0M = 0;
    uint32_t preferredL0N = 0;
    uint32_t preferredL0K = 0;
    // Scheduling
    bool preferPingpong = false;
    bool preferDoubleBuffer = false;
    float minEfficiency = 0.5f;
};
static_assert(
    sizeof(aclblasLtMatmulPreferenceImpl) <= sizeof(aclblasLtMatmulPreferenceOpaque_t),
    "Impl of aclblasLtMatmulPreference must fit in capsule!");
