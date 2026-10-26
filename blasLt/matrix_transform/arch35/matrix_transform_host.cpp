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
 * \file matrix_transform_host.cpp
 * \brief Host side of aclblasLtMatrixTransform (ascend950 / arch35): tiling construction,
 *        dtype/order/dimension validation, index-table and workspace management, the FP4
 *        three-phase pipeline and the device kernel launches.
 */

#include "matrix_transform_engine.h"

#include <acl/acl.h>
#include <cstdint>
#include <vector>

#include "matrix_transform_get_tiling.h"
#include "matrix_transform_kernel.h"
#include "matrix_transform_perm_table.h"

void matrix_transform_get_tiling(
    uint32_t rows, uint32_t cols, uint32_t lda, uint32_t ldb, uint32_t ldc, uint8_t orderA, uint8_t orderB,
    uint8_t orderC, uint8_t opA, uint8_t opB, uint8_t hasB, uint32_t alphaBits, uint32_t betaBits,
    MatrixTransformTilingData& tilingData)
{
    tilingData = {};
    tilingData.rows = rows;
    tilingData.cols = cols;

    tilingData.lda = lda;
    tilingData.ldb = ldb;
    tilingData.ldc = ldc;

    tilingData.alphaBits = alphaBits;
    tilingData.betaBits = betaBits;

    tilingData.orderA = orderA;
    tilingData.orderB = orderB;
    tilingData.orderC = orderC;
    tilingData.opA = opA;
    tilingData.opB = opB;
    tilingData.hasB = hasB;
}

namespace {

// dtype codes (MT_DTYPE_FP32..MT_DTYPE_FP4) are defined once in matrix_transform_tiling_data.h and
// shared with the device kernel routing. Unsupported low-precision dtypes (HiF8 / FP8_E8M0 /
// FP4_E1M2) map to MT_DTYPE_INVALID so the dtype validation intercepts them with
// ACLBLAS_STATUS_NOT_SUPPORTED (design 1.3.A section 4.2).
inline uint8_t MatDtypeCode(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return MT_DTYPE_FP32;
        case ACL_FLOAT16:
            return MT_DTYPE_FP16;
        case ACL_BF16:
            return MT_DTYPE_BF16;
        case ACL_INT8:
            return MT_DTYPE_INT8;
        case ACL_INT32:
            return MT_DTYPE_INT32;
        case ACL_FLOAT8_E4M3FN:
            return MT_DTYPE_FP8_E4M3;
        case ACL_FLOAT8_E5M2:
            return MT_DTYPE_FP8_E5M2;
        case ACL_FLOAT4_E2M1:
            return MT_DTYPE_FP4;
        default:
            return MT_DTYPE_INVALID;
    }
}

// Byte width of a transform dtype's storage. FP8 = 1 byte/element; FP4 = packed 2 elements/byte,
// returned as 1 here for linear element math but sized as ceil(count/2) bytes where packing matters
// (ND workspaces and ld bounds use the packed helpers below).
inline size_t MatDtypeBytes(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
        case ACL_INT32:
            return 4U;
        case ACL_FLOAT16:
        case ACL_BF16:
            return 2U;
        case ACL_INT8:
        case ACL_FLOAT8_E4M3FN:
        case ACL_FLOAT8_E5M2:
        case ACL_FLOAT4_E2M1:
            return 1U;
        default:
            return 0U;
    }
}

// dtype code domain predicates. The float scale path is FP32/FP16/BF16; the integer path is
// INT8/INT32; FP8 (E4M3FN/E5M2) and FP4 (E2M1) are their own scale paths (float / bfloat16_t).
inline bool MatIsFloatCode(uint8_t code) { return code <= MT_DTYPE_BF16; }

inline bool MatIsIntCode(uint8_t code) { return code == MT_DTYPE_INT8 || code == MT_DTYPE_INT32; }

inline bool MatIsFp8Code(uint8_t code) { return code == MT_DTYPE_FP8_E4M3 || code == MT_DTYPE_FP8_E5M2; }

inline bool MatIsFp4Code(uint8_t code) { return code == MT_DTYPE_FP4; }

// FP4 packed leading dimension / byte count: two logical elements per byte (ceil).
inline uint64_t MatFp4PackedLd(uint64_t logicalLd) { return (logicalLd + 1U) / 2U; }

// Number of independent leading-dim blocks in an FP4 physical buffer (mirrors the test wrapper
// ltTransformNumBlocks): COL packs along columns (cols blocks), ROW along rows (rows blocks), and
// COL32 / complex orders pack 32-column groups (ceil(cols/32) blocks).
inline uint32_t MatFp4NumBlocks(aclblasLtOrder_t order, uint64_t physRows, uint64_t physCols)
{
    if (order == ACLBLASLT_ORDER_COL) {
        return static_cast<uint32_t>(physCols);
    }
    if (order == ACLBLASLT_ORDER_ROW) {
        return static_cast<uint32_t>(physRows);
    }
    return static_cast<uint32_t>((physCols + 31U) / 32U); // COL32 / COL4_4R2_8C / COL32_2R_4R4
}

// Valid order enum (COL/ROW/COL32/COL4_4R2_8C/COL32_2R_4R4).
inline bool MatIsValidOrder(aclblasLtOrder_t order)
{
    return order == ACLBLASLT_ORDER_COL || order == ACLBLASLT_ORDER_ROW || order == ACLBLASLT_ORDER_COL32 ||
           order == ACLBLASLT_ORDER_COL4_4R2_8C || order == ACLBLASLT_ORDER_COL32_2R_4R4;
}

// Complex quantization layout (INT8/INT32 only, integer path).
inline bool MatIsComplexOrder(aclblasLtOrder_t order)
{
    return order == ACLBLASLT_ORDER_COL4_4R2_8C || order == ACLBLASLT_ORDER_COL32_2R_4R4;
}

// Composite tile row count for a complex order (8 for COL4_4R2_8C, 32 for COL32_2R_4R4).
inline uint32_t MatComplexTileRowsHost(aclblasLtOrder_t order)
{
    return (order == ACLBLASLT_ORDER_COL4_4R2_8C) ? 8U : 32U;
}

inline uint8_t MatOpCode(aclblasOperation_t op)
{
    return (op == ACLBLAS_OP_N) ? 0U : 1U; // C is treated as T (no complex dtype)
}

// op enum validity: only N / T / C are supported (C is treated as T, no complex dtype).
inline bool MatIsValidOp(aclblasOperation_t op)
{
    return op == ACLBLAS_OP_N || op == ACLBLAS_OP_T || op == ACLBLAS_OP_C;
}

// Lower bound (element count) of a leading dimension for an order given physical rows/cols.
// COL: ld >= rows; ROW: ld >= cols; COL32/complex: ld >= group stride (rows padded over the tile).
inline uint64_t MatLeadingDimLowerBound(aclblasLtOrder_t order, uint64_t physRows, uint64_t physCols)
{
    if (order == ACLBLASLT_ORDER_COL) {
        return physRows;
    }
    if (order == ACLBLASLT_ORDER_ROW) {
        return physCols;
    }
    if (order == ACLBLASLT_ORDER_COL32) {
        return physRows * 32U; // one 32-col group stride
    }
    const uint64_t tileRows = MatComplexTileRowsHost(order);
    const uint64_t numRowBlocks = (physRows + tileRows - 1U) / tileRows;
    return numRowBlocks * tileRows * 32U; // group stride spans all row blocks
}

// Logical (rows x cols) of an input matrix after op, column-major basis.
inline void MatLogicalDims(const MatTransformLayout* layout, aclblasOperation_t op, uint64_t& rows, uint64_t& cols)
{
    if (op == ACLBLAS_OP_N) {
        rows = layout->rows;
        cols = layout->cols;
    } else {
        rows = layout->cols;
        cols = layout->rows;
    }
}

// Scale path of a dtype code (only meaningful for valid codes):
//   FP32 path = float; INT32 path = int32; FP8 path = float; FP4 path = bfloat16_t.
inline uint8_t MatScalePathOfCode(uint8_t code)
{
    if (MatIsIntCode(code)) {
        return MT_SCALE_PATH_INT32;
    }
    if (MatIsFp8Code(code)) {
        return MT_SCALE_PATH_FP8;
    }
    if (MatIsFp4Code(code)) {
        return MT_SCALE_PATH_FP4;
    }
    return MT_SCALE_PATH_FP32; // FP32 / FP16 / BF16
}

// Validate that the descriptor scaleType is consistent with the dtype scale path:
//   FP32 path -> FP32 scaleType; INT32 path -> INT32 scaleType; FP8 path -> FP32 scaleType;
//   FP4 path -> BF16 scaleType (FP32 also tolerated since the FP4 main pass computes in float).
inline bool MatScaleTypeMatchesPath(uint8_t scalePath, aclDataType scaleType)
{
    switch (scalePath) {
        case MT_SCALE_PATH_INT32:
            return scaleType == ACL_INT32;
        case MT_SCALE_PATH_FP8:
            return scaleType == ACL_FLOAT;
        case MT_SCALE_PATH_FP4:
            return scaleType == ACL_BF16 || scaleType == ACL_FLOAT;
        default: // MT_SCALE_PATH_FP32
            return scaleType == ACL_FLOAT;
    }
}

// Compute domain of a dtype code, used to enforce cross-dtype legality. FP32/FP16/BF16/FP8/FP4 all
// share the floating compute domain (cast to float/bf16); INT8/INT32 share the int32 domain.
constexpr uint8_t MT_DOMAIN_FLOAT = 0; // FP32 / FP16 / BF16 / FP8 / FP4
constexpr uint8_t MT_DOMAIN_INT = 1;   // INT8 / INT32
inline uint8_t MatComputeDomain(uint8_t code) { return MatIsIntCode(code) ? MT_DOMAIN_INT : MT_DOMAIN_FLOAT; }

// Validate dtype domain consistency and order/dtype legality.
// Rules: dtype in the supported set; A/B/C share one compute domain (FP8/FP4 cross float is legal,
// but no float<->int cross, and FP8 and FP4 cannot mix in one call); every order is a valid enum;
// complex quantization orders are supported on the INT8/INT32, FP8 and FP4 paths but not on the
// pure floating path (FP32/FP16/BF16 with no fp8 / fp4 participant).
inline aclblasStatus_t MatCheckDtypeAndOrder(
    uint8_t codeA, uint8_t codeB, uint8_t codeC, bool hasB, aclblasLtOrder_t orderA, aclblasLtOrder_t orderB,
    aclblasLtOrder_t orderC)
{
    if (codeA == MT_DTYPE_INVALID || codeC == MT_DTYPE_INVALID || (hasB && codeB == MT_DTYPE_INVALID)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const uint8_t domA = MatComputeDomain(codeA);
    const uint8_t domC = MatComputeDomain(codeC);
    if (domA != domC || (hasB && MatComputeDomain(codeB) != domA)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED; // cross-domain (float<->int) conversion is out of scope
    }
    // issue #585: 内核仅实例化 A/B 同 dtype 组合（MT_DTYPE_LIST 中每个实例的 A/B dtype code
    // 一致），混合 A/B dtype 组合此前会落入 default 分支静默不启动内核却返回 SUCCESS，
    // 此处在 host 校验收紧为与内核实例一致。
    if (hasB && codeB != codeA) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const bool anyFp8 = MatIsFp8Code(codeA) || MatIsFp8Code(codeC) || (hasB && MatIsFp8Code(codeB));
    const bool anyFp4 = MatIsFp4Code(codeA) || MatIsFp4Code(codeC) || (hasB && MatIsFp4Code(codeB));
    if (anyFp8 && anyFp4) {
        return ACLBLAS_STATUS_NOT_SUPPORTED; // FP8 and FP4 use incompatible storage / scale domains
    }
    if (!MatIsValidOrder(orderA) || !MatIsValidOrder(orderC) || (hasB && !MatIsValidOrder(orderB))) {
        return ACLBLAS_STATUS_NOT_SUPPORTED; // illegal order enum
    }
    const bool anyComplex =
        MatIsComplexOrder(orderA) || MatIsComplexOrder(orderC) || (hasB && MatIsComplexOrder(orderB));
    if (anyComplex && domA == MT_DOMAIN_FLOAT && !anyFp8 && !anyFp4) {
        return ACLBLAS_STATUS_NOT_SUPPORTED; // complex layouts are not supported on the pure float path
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Leading-dim lower bound in the layout's storage units. FP4 stores its ld in packed bytes
// (ceil(elements/2)), so the element bound is converted to packed bytes; every other dtype keeps
// the element-count bound.
inline uint64_t MatLeadingDimLowerBoundStored(const MatTransformLayout* layout)
{
    const uint64_t elemBound = MatLeadingDimLowerBound(layout->order, layout->rows, layout->cols);
    return MatIsFp4Code(MatDtypeCode(layout->type)) ? MatFp4PackedLd(elemBound) : elemBound;
}

// Validate that every participating layout's leading dimension meets its order lower bound.
inline aclblasStatus_t MatCheckLeadingDims(
    const MatTransformLayout* aLayout, const MatTransformLayout* bLayout, const MatTransformLayout* cLayout, bool hasB)
{
    if (aLayout->ld < static_cast<int64_t>(MatLeadingDimLowerBoundStored(aLayout))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (cLayout->ld < static_cast<int64_t>(MatLeadingDimLowerBoundStored(cLayout))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (hasB && bLayout->ld < static_cast<int64_t>(MatLeadingDimLowerBoundStored(bLayout))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

struct MatScalars {
    uint8_t scalePath = MT_SCALE_PATH_FP32;
    uint32_t alphaBits = 0U;
    uint32_t betaBits = 0U;
    bool hasB = false;
};

struct MatCodes {
    uint8_t codeA = MT_DTYPE_INVALID;
    uint8_t codeB = MT_DTYPE_INVALID;
    uint8_t codeC = MT_DTYPE_INVALID;
    aclblasLtOrder_t orderB = ACLBLASLT_ORDER_COL;
};

// Validate op enum, batchCount, dtype/order combination and leading dimensions in one pass,
// resolving the dtype codes and effective B order for the launch.
inline aclblasStatus_t MatValidateConfig(
    const aclblasLtMatrixTransformDescImpl* desc, const MatTransformLayout* aLayout, const MatTransformLayout* bLayout,
    const MatTransformLayout* cLayout, bool hasB, MatCodes& codes)
{
    if (!MatIsValidOp(desc->transA) || (hasB && !MatIsValidOp(desc->transB))) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (aLayout->batchCount > 1 || cLayout->batchCount > 1 || (hasB && bLayout->batchCount > 1)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    codes.codeA = MatDtypeCode(aLayout->type);
    codes.codeC = MatDtypeCode(cLayout->type);
    codes.codeB = hasB ? MatDtypeCode(bLayout->type) : codes.codeA;
    codes.orderB = hasB ? bLayout->order : aLayout->order;
    aclblasStatus_t check = MatCheckDtypeAndOrder(
        codes.codeA, codes.codeB, codes.codeC, hasB, aLayout->order, codes.orderB, cLayout->order);
    if (check != ACLBLAS_STATUS_SUCCESS) {
        return check;
    }
    return MatCheckLeadingDims(aLayout, bLayout, cLayout, hasB);
}

// Decide whether B participates (beta != 0) given the scale path. INT32 reads the raw int32 bits;
// every other path (FP32 / FP16 / BF16 / FP8 / FP4) reads beta as a float.
inline bool MatBetaIsZero(uint8_t scalePath, const void* beta, uint32_t betaBits)
{
    if (scalePath == MT_SCALE_PATH_INT32) {
        return betaBits == 0U;
    }
    return *reinterpret_cast<const float*>(beta) == 0.0f;
}

// issue #586: matmul 侧对维度有 IsRepresentableAsUint32 前置校验，transform 侧同样依赖
// uint32 tiling；rows/cols/ld 超过 2^32-1 时高位会被静默截断、内核按错误形状执行，入口显式拒绝。
constexpr uint64_t MT_MAX_UINT32 = 0xFFFFFFFFULL;

inline aclblasStatus_t MatCheckDimRanges(
    const MatTransformLayout* aLayout, const MatTransformLayout* bLayout, const MatTransformLayout* cLayout, bool hasB,
    uint64_t rows, uint64_t cols)
{
    if (rows > MT_MAX_UINT32 || cols > MT_MAX_UINT32) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (aLayout->ld > static_cast<int64_t>(MT_MAX_UINT32) || cLayout->ld > static_cast<int64_t>(MT_MAX_UINT32) ||
        (hasB && bLayout->ld > static_cast<int64_t>(MT_MAX_UINT32))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Validate that op(A) and (when present) op(B) logical dims match C. Returns no-op flag via outRows/outCols=0.
inline aclblasStatus_t MatValidateDims(
    const MatTransformLayout* aLayout, const MatTransformLayout* bLayout, const aclblasLtMatrixTransformDescImpl* desc,
    bool hasB, uint64_t rows, uint64_t cols)
{
    uint64_t rowsA = 0U;
    uint64_t colsA = 0U;
    MatLogicalDims(aLayout, desc->transA, rowsA, colsA);
    if (rowsA != rows || colsA != cols) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (hasB) {
        uint64_t rowsB = 0U;
        uint64_t colsB = 0U;
        MatLogicalDims(bLayout, desc->transB, rowsB, colsB);
        if (rowsB != rows || colsB != cols) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline uint32_t MatResolveVectorCores(int32_t deviceId)
{
    int64_t vecCoreNum = 0;
    aclError aclRet = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vecCoreNum);
    if (aclRet != ACL_SUCCESS || vecCoreNum <= 0) {
        vecCoreNum = 8; // fallback
    }
    return static_cast<uint32_t>(vecCoreNum);
}

// Resolve scalePath, alpha/beta bits and hasB; B / B-layout presence is checked by the caller.
inline void MatResolveScalars(uint8_t scalePath, const void* alpha, const void* beta, MatScalars& sc)
{
    sc.scalePath = scalePath;
    // alpha / beta are passed as float for the FP32 / FP16 / BF16 / FP8 / FP4 scale types (the FP4
    // main pass computes in float on the bf16 template, so float scalars feed it directly) and as
    // int32 for the integer path; we always copy the 4-byte scalar bit pattern verbatim.
    sc.alphaBits = *reinterpret_cast<const uint32_t*>(alpha);
    if (beta != nullptr) {
        sc.betaBits = *reinterpret_cast<const uint32_t*>(beta);
        sc.hasB = !MatBetaIsZero(scalePath, beta, sc.betaBits);
    }
}

// Pick the complex order that drives the group-path in-tile permutation, or COL when none.
inline aclblasLtOrder_t MatPickComplexOrder(
    aclblasLtOrder_t orderA, aclblasLtOrder_t orderB, aclblasLtOrder_t orderC, bool hasB)
{
    if (MatIsComplexOrder(orderC)) {
        return orderC;
    }
    if (MatIsComplexOrder(orderA)) {
        return orderA;
    }
    if (hasB && MatIsComplexOrder(orderB)) {
        return orderB;
    }
    return ACLBLASLT_ORDER_COL;
}

// Allocate and upload the column-major in-tile element-offset table for a complex order to GM.
// Returns the device pointer (caller frees), or nullptr when no complex order participates.
inline void* MatCreateIndexTable(aclblasLtOrder_t complexOrder)
{
    if (!MatIsComplexOrder(complexOrder)) {
        return nullptr;
    }
    const uint32_t count = 32U * 32U; // fixed 32x32 column-major tile (with padding scratch slots)
    std::vector<uint32_t> table(count, 0U);
    if (MatBuildTileColMajorTable(static_cast<uint8_t>(complexOrder), table.data(), count) != count) {
        return nullptr;
    }
    void* devTable = nullptr;
    const size_t bytes = static_cast<size_t>(count) * sizeof(uint32_t);
    if (aclrtMalloc(&devTable, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return nullptr;
    }
    if (aclrtMemcpy(devTable, bytes, table.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(devTable);
        return nullptr;
    }
    return devTable;
}

// GM workspaces backing one transform launch: the pivot in-tile offset table, the per-operand
// de-layout offset tables, and the de-layouted column-major ND buffers for op=T complex inputs.
struct MatTransformWorkspaces {
    void* idxTable = nullptr;  // pivot complex order column-major table (group path / output scatter)
    void* idxTableA = nullptr; // orderA de-layout table (A complex + op=T)
    void* idxTableB = nullptr; // orderB de-layout table (B complex + op=T)
    void* ndA = nullptr;       // de-layouted column-major A workspace
    void* ndB = nullptr;       // de-layouted column-major B workspace
};

inline void MatFreeTransformWorkspaces(MatTransformWorkspaces& ws)
{
    if (ws.idxTable != nullptr) {
        aclrtFree(ws.idxTable);
        ws.idxTable = nullptr;
    }
    if (ws.idxTableA != nullptr) {
        aclrtFree(ws.idxTableA);
        ws.idxTableA = nullptr;
    }
    if (ws.idxTableB != nullptr) {
        aclrtFree(ws.idxTableB);
        ws.idxTableB = nullptr;
    }
    if (ws.ndA != nullptr) {
        aclrtFree(ws.ndA);
        ws.ndA = nullptr;
    }
    if (ws.ndB != nullptr) {
        aclrtFree(ws.ndB);
        ws.ndB = nullptr;
    }
}

// Allocate a column-major ND workspace (physRows * physCols elements of dtypeBytes) for a staged
// op=T complex input. Returns the device pointer (caller frees), or nullptr on failure.
inline void* MatAllocNdWorkspace(uint64_t physRows, uint64_t physCols, size_t dtypeBytes)
{
    const size_t bytes = static_cast<size_t>(physRows) * static_cast<size_t>(physCols) * dtypeBytes;
    if (bytes == 0U) {
        return nullptr;
    }
    void* dev = nullptr;
    if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return nullptr;
    }
    return dev;
}

// Allocate the pivot table plus the per-operand de-layout tables and ND workspaces required by
// the requested launch. On any failure the caller frees whatever was already allocated.
inline aclblasStatus_t MatAllocTransformWorkspaces(
    aclblasLtOrder_t complexOrder, const MatTransformLayout* aLayout, const MatTransformLayout* bLayout, bool delayoutA,
    bool delayoutB, uint8_t codeA, uint8_t codeB, MatTransformWorkspaces& ws)
{
    if (MatIsComplexOrder(complexOrder)) {
        ws.idxTable = MatCreateIndexTable(complexOrder);
        if (ws.idxTable == nullptr) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    if (delayoutA) {
        ws.idxTableA = MatCreateIndexTable(aLayout->order);
        ws.ndA = MatAllocNdWorkspace(aLayout->rows, aLayout->cols, MatDtypeBytes(aLayout->type));
        if (ws.idxTableA == nullptr || ws.ndA == nullptr) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    if (delayoutB) {
        ws.idxTableB = MatCreateIndexTable(bLayout->order);
        ws.ndB = MatAllocNdWorkspace(bLayout->rows, bLayout->cols, MatDtypeBytes(bLayout->type));
        if (ws.idxTableB == nullptr || ws.ndB == nullptr) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    (void)codeA;
    (void)codeB;
    return ACLBLAS_STATUS_SUCCESS;
}

// Build tiling, log launch parameters and dispatch the device kernel.
inline aclblasStatus_t MatLaunch(
    const void* A, const void* B, void* C, const MatTransformLayout* aLayout, const MatTransformLayout* bLayout,
    const MatTransformLayout* cLayout, const aclblasLtMatrixTransformDescImpl* desc, const MatScalars& sc,
    aclblasLtOrder_t orderB, uint8_t codeA, uint8_t codeB, uint8_t codeC, uint64_t rows, uint64_t cols,
    uint32_t numBlocks, aclrtStream stream)
{
    MatrixTransformTilingData tiling;
    matrix_transform_get_tiling(
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), static_cast<uint32_t>(aLayout->ld),
        sc.hasB ? static_cast<uint32_t>(bLayout->ld) : 0U, static_cast<uint32_t>(cLayout->ld),
        static_cast<uint8_t>(aLayout->order), static_cast<uint8_t>(orderB), static_cast<uint8_t>(cLayout->order),
        MatOpCode(desc->transA), MatOpCode(desc->transB), sc.hasB ? 1U : 0U, sc.alphaBits, sc.betaBits, tiling);

    // Complex input + op=T staging: a complex physical input is first de-layouted (op=N) into a
    // plain column-major ND GM workspace, then the main pass reads it as a COL linear input and
    // applies op=T. This organises the transpose at the logical ND level (single-tile transpose
    // cannot cover the op=T cross-tile remap) and matches golden's de-layout->applyOp order.
    const bool delayoutA = MatIsComplexOrder(aLayout->order) && desc->transA != ACLBLAS_OP_N;
    const bool delayoutB = sc.hasB && MatIsComplexOrder(orderB) && desc->transB != ACLBLAS_OP_N;
    tiling.needDelayoutA = delayoutA ? 1U : 0U;
    tiling.needDelayoutB = delayoutB ? 1U : 0U;
    tiling.physRowsA = static_cast<uint32_t>(aLayout->rows);
    tiling.physColsA = static_cast<uint32_t>(aLayout->cols);
    tiling.physRowsB = sc.hasB ? static_cast<uint32_t>(bLayout->rows) : 0U;
    tiling.physColsB = sc.hasB ? static_cast<uint32_t>(bLayout->cols) : 0U;

    const aclblasLtOrder_t complexOrder = MatPickComplexOrder(aLayout->order, orderB, cLayout->order, sc.hasB);
    MatTransformWorkspaces ws;
    aclblasStatus_t wsStatus =
        MatAllocTransformWorkspaces(complexOrder, aLayout, bLayout, delayoutA, delayoutB, codeA, codeB, ws);
    if (wsStatus != ACLBLAS_STATUS_SUCCESS) {
        MatFreeTransformWorkspaces(ws);
        return wsStatus;
    }

    uint8_t* aDev = reinterpret_cast<uint8_t*>(const_cast<void*>(A));
    uint8_t* bDev = reinterpret_cast<uint8_t*>(const_cast<void*>(B));
    uint8_t* cDev = reinterpret_cast<uint8_t*>(C);
    uint8_t* idxDev = reinterpret_cast<uint8_t*>(ws.idxTable);
    uint8_t* idxADev = reinterpret_cast<uint8_t*>(ws.idxTableA);
    uint8_t* idxBDev = reinterpret_cast<uint8_t*>(ws.idxTableB);
    uint8_t* ndADev = reinterpret_cast<uint8_t*>(ws.ndA);
    uint8_t* ndBDev = reinterpret_cast<uint8_t*>(ws.ndB);

    if (delayoutA || delayoutB) {
        // Phase 1: de-layout the complex op=T input(s) into the column-major ND workspace(s).
        MatrixTransformTilingData delayoutTiling = tiling;
        delayoutTiling.phase = MT_PHASE_DELAYOUT;
        matrix_transform_kernel_do(
            aDev, bDev, cDev, idxDev, idxADev, idxBDev, ndADev, ndBDev, codeA, codeB, codeC, delayoutTiling, numBlocks,
            stream);
        aclrtSynchronizeStream(stream); // de-layout must finish before the main phase reads ND

        // Rewrite the staged operand to a COL linear input read from its ND workspace; the main
        // phase then applies op=T over the logical ND matrix.
        if (delayoutA) {
            aDev = ndADev;
            tiling.orderA = MT_ORDER_COL;
            tiling.lda = tiling.physRowsA;
        }
        if (delayoutB) {
            bDev = ndBDev;
            tiling.orderB = MT_ORDER_COL;
            tiling.ldb = tiling.physRowsB;
        }
    }

    // Phase 2 (or the only phase): the main transform.
    tiling.phase = MT_PHASE_MAIN;
    matrix_transform_kernel_do(
        aDev, bDev, cDev, idxDev, idxADev, idxBDev, ndADev, ndBDev, codeA, codeB, codeC, tiling, numBlocks, stream);
    if (ws.idxTable != nullptr || ws.idxTableA != nullptr || ws.idxTableB != nullptr || ws.ndA != nullptr ||
        ws.ndB != nullptr) {
        aclrtSynchronizeStream(stream); // ensure the kernel consumed the workspaces before free
    }
    MatFreeTransformWorkspaces(ws);
    return ACLBLAS_STATUS_SUCCESS;
}

// bf16 ND workspaces of the FP4 pipeline (one per fp4 operand). numBlocks / ldBf16 size each
// workspace to mirror the operand's physical layout (ld = 2*packedLd element stride).
struct MatFp4Workspaces {
    void* ndA = nullptr;
    void* ndB = nullptr;
    void* ndC = nullptr;
    uint64_t ldBf16A = 0U;
    uint64_t ldBf16B = 0U;
    uint64_t ldBf16C = 0U;
    uint32_t numBlocksA = 0U;
    uint32_t numBlocksB = 0U;
    uint32_t numBlocksC = 0U;
};

inline void MatFreeFp4Workspaces(MatFp4Workspaces& ws)
{
    if (ws.ndA != nullptr) {
        aclrtFree(ws.ndA);
        ws.ndA = nullptr;
    }
    if (ws.ndB != nullptr) {
        aclrtFree(ws.ndB);
        ws.ndB = nullptr;
    }
    if (ws.ndC != nullptr) {
        aclrtFree(ws.ndC);
        ws.ndC = nullptr;
    }
}

// Allocate the bf16 ND workspaces for whichever of A / B / C are fp4.
inline aclblasStatus_t MatAllocFp4Workspaces(
    const MatTransformLayout* aLayout, const MatTransformLayout* bLayout, const MatTransformLayout* cLayout, bool fp4A,
    bool fp4B, bool fp4C, bool hasB, MatFp4Workspaces& ws)
{
    ws.ldBf16A = 2U * static_cast<uint64_t>(aLayout->ld);
    ws.ldBf16B = hasB ? 2U * static_cast<uint64_t>(bLayout->ld) : 0U;
    ws.ldBf16C = 2U * static_cast<uint64_t>(cLayout->ld);
    ws.numBlocksA = MatFp4NumBlocks(aLayout->order, aLayout->rows, aLayout->cols);
    ws.numBlocksB = hasB ? MatFp4NumBlocks(bLayout->order, bLayout->rows, bLayout->cols) : 0U;
    ws.numBlocksC = MatFp4NumBlocks(cLayout->order, cLayout->rows, cLayout->cols);
    ws.ndA = fp4A ? MatAllocNdWorkspace(ws.numBlocksA, ws.ldBf16A, sizeof(uint16_t)) : nullptr;
    ws.ndB = fp4B ? MatAllocNdWorkspace(ws.numBlocksB, ws.ldBf16B, sizeof(uint16_t)) : nullptr;
    ws.ndC = fp4C ? MatAllocNdWorkspace(ws.numBlocksC, ws.ldBf16C, sizeof(uint16_t)) : nullptr;
    if ((fp4A && ws.ndA == nullptr) || (fp4B && ws.ndB == nullptr) || (fp4C && ws.ndC == nullptr)) {
        MatFreeFp4Workspaces(ws);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Build the base FP4 tiling shared by the unpack / repack phases (carries dims, packed lds, orders).
inline MatrixTransformTilingData MatBuildFp4Tiling(
    const MatTransformLayout* aLayout, const MatTransformLayout* bLayout, const MatTransformLayout* cLayout,
    const aclblasLtMatrixTransformDescImpl* desc, const MatScalars& sc, aclblasLtOrder_t orderB, uint64_t rows,
    uint64_t cols)
{
    MatrixTransformTilingData tiling;
    matrix_transform_get_tiling(
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), static_cast<uint32_t>(aLayout->ld),
        sc.hasB ? static_cast<uint32_t>(bLayout->ld) : 0U, static_cast<uint32_t>(cLayout->ld),
        static_cast<uint8_t>(aLayout->order), static_cast<uint8_t>(orderB), static_cast<uint8_t>(cLayout->order),
        MatOpCode(desc->transA), MatOpCode(desc->transB), sc.hasB ? 1U : 0U, sc.alphaBits, sc.betaBits, tiling);
    return tiling;
}

// Phase 2 of the FP4 pipeline: run the bf16 main transform over the unpacked bf16 ND inputs into the
// bf16 ND C output. An fp4 operand reads / writes its bf16 ND workspace; a non-fp4 operand keeps its
// real dtype / buffer / ld so the FP4<->float cross cases run as one bf16-or-float transform.
inline aclblasStatus_t MatFp4MainLaunch(
    const void* A, const void* B, void* C, const MatTransformLayout* aLayout, const MatTransformLayout* bLayout,
    const MatTransformLayout* cLayout, const aclblasLtMatrixTransformDescImpl* desc, const MatScalars& sc,
    aclblasLtOrder_t orderB, uint8_t codeA, uint8_t codeB, uint8_t codeC, bool fp4A, bool fp4B, bool fp4C,
    const MatFp4Workspaces& ws, uint64_t rows, uint64_t cols, uint32_t numBlocks, aclrtStream stream)
{
    MatTransformLayout bf16A = *aLayout;
    MatTransformLayout bf16B = sc.hasB ? *bLayout : *aLayout;
    MatTransformLayout bf16C = *cLayout;
    bf16A.type = fp4A ? ACL_BF16 : aLayout->type;
    bf16B.type = (sc.hasB && fp4B) ? ACL_BF16 : (sc.hasB ? bLayout->type : ACL_BF16);
    bf16C.type = fp4C ? ACL_BF16 : cLayout->type;
    bf16A.ld = fp4A ? static_cast<int64_t>(ws.ldBf16A) : aLayout->ld;
    bf16B.ld = (sc.hasB && fp4B) ? static_cast<int64_t>(ws.ldBf16B) : (sc.hasB ? bLayout->ld : 0);
    bf16C.ld = fp4C ? static_cast<int64_t>(ws.ldBf16C) : cLayout->ld;
    // Every fp4 dtype code maps to BF16 for the main pass (the fp4 operands are bf16 ND workspaces);
    // this also covers the single-input case where codeB mirrors codeA (fp4) but B is unused.
    const uint8_t mainCodeA = MatIsFp4Code(codeA) ? MT_DTYPE_BF16 : codeA;
    const uint8_t mainCodeB = MatIsFp4Code(codeB) ? MT_DTYPE_BF16 : codeB;
    const uint8_t mainCodeC = MatIsFp4Code(codeC) ? MT_DTYPE_BF16 : codeC;
    void* mainA = fp4A ? ws.ndA : const_cast<void*>(A);
    void* mainB = (sc.hasB && fp4B) ? ws.ndB : const_cast<void*>(B);
    void* mainC = fp4C ? ws.ndC : C;
    return MatLaunch(
        mainA, sc.hasB ? mainB : nullptr, mainC, &bf16A, sc.hasB ? &bf16B : nullptr, &bf16C, desc, sc, orderB,
        mainCodeA, mainCodeB, mainCodeC, rows, cols, numBlocks, stream);
}

// FP4 (fp4x2) cannot be moved or permuted at byte granularity for non-contiguous layouts, so the
// whole transform runs in the bf16 domain (design 1.3.A section 3.8.3):
//   UNPACK : packed fp4x2 inputs -> bf16 ND workspaces (logical layout, ld = 2*packedLd)
//   MAIN   : the regular bf16 transform (MatLaunch) over the bf16 ND inputs -> a bf16 ND C output
//   REPACK : bf16 ND C output -> packed fp4x2 C
// The three phases launch on one stream (stream order serialises them); the host synchronises once
// before freeing the bf16 ND workspaces. The main pass reuses the full bf16 engine (linear /
// complex / op=T de-layout), so every order / op / cross combination is handled there unchanged.
inline aclblasStatus_t MatLaunchFp4(
    const void* A, const void* B, void* C, const MatTransformLayout* aLayout, const MatTransformLayout* bLayout,
    const MatTransformLayout* cLayout, const aclblasLtMatrixTransformDescImpl* desc, const MatScalars& sc,
    aclblasLtOrder_t orderB, uint8_t codeA, uint8_t codeB, uint8_t codeC, uint64_t rows, uint64_t cols,
    uint32_t numBlocks, aclrtStream stream)
{
    const bool fp4A = MatIsFp4Code(codeA);
    const bool fp4B = sc.hasB && MatIsFp4Code(codeB);
    const bool fp4C = MatIsFp4Code(codeC);

    MatFp4Workspaces ws;
    aclblasStatus_t wsStatus = MatAllocFp4Workspaces(aLayout, bLayout, cLayout, fp4A, fp4B, fp4C, sc.hasB, ws);
    if (wsStatus != ACLBLAS_STATUS_SUCCESS) {
        return wsStatus;
    }

    uint8_t* aDev = reinterpret_cast<uint8_t*>(const_cast<void*>(A));
    uint8_t* bDev = reinterpret_cast<uint8_t*>(const_cast<void*>(B));
    uint8_t* cDev = reinterpret_cast<uint8_t*>(C);

    // Phase 1: unpack the packed fp4x2 inputs into their bf16 ND workspaces.
    MatrixTransformTilingData unpackTiling = MatBuildFp4Tiling(aLayout, bLayout, cLayout, desc, sc, orderB, rows, cols);
    unpackTiling.phase = MT_PHASE_FP4_UNPACK;
    unpackTiling.fp4IsA = fp4A ? 1U : 0U;
    unpackTiling.fp4IsB = fp4B ? 1U : 0U;
    unpackTiling.fp4NumBlocksA = ws.numBlocksA;
    unpackTiling.fp4NumBlocksB = ws.numBlocksB;
    matrix_transform_kernel_do(
        aDev, bDev, cDev, nullptr, nullptr, nullptr, reinterpret_cast<uint8_t*>(ws.ndA),
        reinterpret_cast<uint8_t*>(ws.ndB), MT_DTYPE_FP4, MT_DTYPE_FP4, MT_DTYPE_FP4, unpackTiling, numBlocks, stream);

    // Phase 2: the bf16 main transform.
    aclblasStatus_t mainStatus = MatFp4MainLaunch(
        A, B, C, aLayout, bLayout, cLayout, desc, sc, orderB, codeA, codeB, codeC, fp4A, fp4B, fp4C, ws, rows, cols,
        numBlocks, stream);
    if (mainStatus != ACLBLAS_STATUS_SUCCESS) {
        MatFreeFp4Workspaces(ws);
        return mainStatus;
    }

    // Phase 3: repack the bf16 ND C output into the packed fp4x2 C buffer.
    if (fp4C) {
        MatrixTransformTilingData repackTiling = unpackTiling;
        repackTiling.phase = MT_PHASE_FP4_REPACK;
        repackTiling.fp4IsA = 0U;
        repackTiling.fp4IsB = 0U;
        repackTiling.fp4IsC = 1U;
        repackTiling.fp4NumBlocksC = ws.numBlocksC;
        matrix_transform_kernel_do(
            reinterpret_cast<uint8_t*>(ws.ndC), nullptr, cDev, nullptr, nullptr, nullptr, nullptr, nullptr,
            MT_DTYPE_FP4, MT_DTYPE_FP4, MT_DTYPE_FP4, repackTiling, numBlocks, stream);
    }

    aclrtSynchronizeStream(stream); // ensure the pipeline consumed the bf16 ND workspaces before free
    MatFreeFp4Workspaces(ws);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t MatTransformLaunch(
    int32_t deviceId, const aclblasLtMatrixTransformDescImpl* desc, const void* alpha, const void* A,
    const MatTransformLayout* aLayout, const void* beta, const void* B, const MatTransformLayout* bLayoutIn,
    bool bLayoutValid, void* C, const MatTransformLayout* cLayout, uint64_t rows, uint64_t cols, aclrtStream stream)
{
    // Scale path: FP4 wins when any operand is fp4 (its packed pipeline drives the whole call), then
    // FP8 when any operand is fp8, otherwise the anchor (A) dtype's path. The descriptor scaleType
    // must match: FP8 requires FP32 scaleType, FP4 requires BF16 (FP32 tolerated), integer requires
    // INT32. Cross-dtype legality (one compute domain, no FP8/FP4 mix) is enforced in MatValidateConfig.
    const uint8_t codeAEntry = MatDtypeCode(aLayout->type);
    const uint8_t codeCEntry = MatDtypeCode(cLayout->type);
    if (codeAEntry == MT_DTYPE_INVALID) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const bool anyFp4Entry = MatIsFp4Code(codeAEntry) || MatIsFp4Code(codeCEntry);
    const bool anyFp8Entry = MatIsFp8Code(codeAEntry) || MatIsFp8Code(codeCEntry);
    const uint8_t scalePath =
        anyFp4Entry ? MT_SCALE_PATH_FP4 : (anyFp8Entry ? MT_SCALE_PATH_FP8 : MatScalePathOfCode(codeAEntry));
    if (!MatScaleTypeMatchesPath(scalePath, desc->scaleType)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    MatScalars sc;
    MatResolveScalars(scalePath, alpha, beta, sc);
    if (sc.hasB && (B == nullptr || bLayoutIn == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (C == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const MatTransformLayout* bLayout = sc.hasB ? bLayoutIn : nullptr;
    if (bLayout != nullptr && !bLayoutValid) {
        return ACLBLAS_STATUS_INVALID_VALUE; // corrupted / foreign layout
    }
    aclblasStatus_t dimStatus = MatValidateDims(aLayout, bLayout, desc, sc.hasB, rows, cols);
    if (dimStatus != ACLBLAS_STATUS_SUCCESS) {
        return dimStatus;
    }
    // issue #586: 维度超限（> 2^32-1）在写 tiling 前拒绝，避免静默截断
    aclblasStatus_t rangeStatus = MatCheckDimRanges(aLayout, bLayout, cLayout, sc.hasB, rows, cols);
    if (rangeStatus != ACLBLAS_STATUS_SUCCESS) {
        return rangeStatus;
    }

    MatCodes codes;
    aclblasStatus_t cfgStatus = MatValidateConfig(desc, aLayout, bLayout, cLayout, sc.hasB, codes);
    if (cfgStatus != ACLBLAS_STATUS_SUCCESS) {
        return cfgStatus;
    }

    const uint32_t vecCores = MatResolveVectorCores(deviceId);
    // FP4 runs the dedicated unpack -> bf16 transform -> repack pipeline; every other path goes
    // straight through the templated engine (FP8 reuses the int8 byte movement with a float scale).
    if (sc.scalePath == MT_SCALE_PATH_FP4) {
        return MatLaunchFp4(
            A, B, C, aLayout, bLayout, cLayout, desc, sc, codes.orderB, codes.codeA, codes.codeB, codes.codeC, rows,
            cols, vecCores, stream);
    }
    return MatLaunch(
        A, B, C, aLayout, bLayout, cLayout, desc, sc, codes.orderB, codes.codeA, codes.codeB, codes.codeC, rows, cols,
        vecCores, stream);
}
