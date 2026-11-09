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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"
#include "blasLtMatrixTransform_param.h"
#include "blasLtMatrixTransform_golden.h"

// ── physical buffer element count for an order (COL/ROW/COL32 + complex layouts) ──
// For tiled orders the count is groups*ld; ld carries the tile stride (>= rows*tileColspan),
// so groups*ld covers every group-padded tile. Mirrors the offset map in golden.
inline int64_t ltTransformPhysCount(aclblasLtOrder_t order, int rows, int cols, int ld)
{
    switch (order) {
        case ACLBLASLT_ORDER_COL:
            return static_cast<int64_t>(ld) * cols;                  // col*ld + row
        case ACLBLASLT_ORDER_ROW:
            return static_cast<int64_t>(ld) * rows;                  // row*ld + col
        case ACLBLASLT_ORDER_COL32:
        case ACLBLASLT_ORDER_COL4_4R2_8C:
        case ACLBLASLT_ORDER_COL32_2R_4R4: {
            int groups = (cols + 31) / 32;                           // 32-col groups, ld = group stride
            return static_cast<int64_t>(groups) * ld;
        }
        default:
            return static_cast<int64_t>(ld) * cols;
    }
}

// ── number of independent leading-dim blocks of length `ld` in the physical buffer ──
// physCount == numBlocks * ld. FP4 packs 2 logical elements per byte WITHIN each ld-block, so the
// packed byte buffer is numBlocks * packedLd(ld), matching the device layout (layout ld = packed
// bytes). Pairing inside a block keeps the packed nibble adjacency = physical-offset adjacency.
inline int ltTransformNumBlocks(aclblasLtOrder_t order, int rows, int cols)
{
    switch (order) {
        case ACLBLASLT_ORDER_COL:           return cols;             // ld blocks along columns
        case ACLBLASLT_ORDER_ROW:           return rows;             // ld blocks along rows
        case ACLBLASLT_ORDER_COL32:
        case ACLBLASLT_ORDER_COL4_4R2_8C:
        case ACLBLASLT_ORDER_COL32_2R_4R4:  return (cols + 31) / 32; // 32-col groups
        default:                            return cols;
    }
}

// ── FP16 (binary16) host conversion ──
inline uint16_t ltFloatToFp16(float v)
{
    uint32_t x;
    std::memcpy(&x, &v, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF)
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (exp >= 0x1F)
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t halfMant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t halfWay = 1u << (shift - 1);
        if (rem > halfWay || (rem == halfWay && (halfMant & 1))) halfMant++;
        return static_cast<uint16_t>(sign | halfMant);
    }
    uint32_t halfMant = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (halfMant & 1))) {
        halfMant++;
        if (halfMant == 0x400u) { halfMant = 0; exp++; }
    }
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
}

inline float ltFp16ToFloat(uint16_t h)
{
    uint32_t hsign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t hexp = (h >> 10) & 0x1F;
    uint32_t hmant = h & 0x3FFu;
    uint32_t f;
    if (hexp == 0) {
        if (hmant == 0) { f = hsign; }
        else {
            int e = -1;
            do { hmant <<= 1; e++; } while ((hmant & 0x400u) == 0);
            hmant &= 0x3FFu;
            f = hsign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (hmant << 13);
        }
    } else if (hexp == 0x1F) {
        f = hsign | 0x7F800000u | (hmant << 13);
    } else {
        f = hsign | (static_cast<uint32_t>(hexp - 15 + 127) << 23) | (hmant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

inline uint16_t ltFloatToBf16(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

inline float ltBf16ToFloat(uint16_t b)
{
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

// pack a host float buffer into device-dtype bytes
inline std::vector<uint8_t> ltPackDtype(const std::vector<float>& src, aclDataType dt)
{
    std::vector<uint8_t> bytes(src.size() * transformDtypeSize(dt));
    for (size_t i = 0; i < src.size(); i++) {
        switch (dt) {
            case ACL_FLOAT: {
                float v = src[i];
                std::memcpy(&bytes[i * 4], &v, 4);
                break;
            }
            case ACL_INT32: {
                int32_t v = static_cast<int32_t>(std::llround(src[i]));
                std::memcpy(&bytes[i * 4], &v, 4);
                break;
            }
            case ACL_FLOAT16: {
                uint16_t v = ltFloatToFp16(src[i]);
                std::memcpy(&bytes[i * 2], &v, 2);
                break;
            }
            case ACL_BF16: {
                uint16_t v = ltFloatToBf16(src[i]);
                std::memcpy(&bytes[i * 2], &v, 2);
                break;
            }
            case ACL_INT8: {
                int v = static_cast<int>(std::llround(src[i]));
                v = std::max(-128, std::min(127, v));
                bytes[i] = static_cast<uint8_t>(static_cast<int8_t>(v));
                break;
            }
            case ACL_FLOAT8_E4M3FN:
                bytes[i] = ltFloatToFp8E4m3(src[i]);   // RINT + saturate, golden-shared
                break;
            case ACL_FLOAT8_E5M2:
                bytes[i] = ltFloatToFp8E5m2(src[i]);
                break;
            default: {
                float v = src[i];
                std::memcpy(&bytes[i * 4], &v, 4);
                break;
            }
        }
    }
    return bytes;
}

// ── FP4 packed pack: logical-element float buffer → packed fp4x2 bytes (2 elements/byte). ──
// Packs WITHIN each ld-block (numBlocks blocks of length `ld`): logical element k in a block goes
// to byte k/2, nibble k%2 (low nibble first, matching the device fp4x2 packed layout). The packed
// byte ld = (ld+1)/2; padding nibble of an odd ld block is zero.
inline std::vector<uint8_t> ltPackFp4(const std::vector<float>& src, int ld, int numBlocks)
{
    int packedLd = fp4PackedLd(ld);
    std::vector<uint8_t> bytes(static_cast<size_t>(numBlocks) * packedLd, 0u);
    for (int blk = 0; blk < numBlocks; blk++) {
        for (int k = 0; k < ld; k++) {
            size_t srcIdx = static_cast<size_t>(blk) * ld + k;
            if (srcIdx >= src.size()) break;
            uint8_t nib = ltFloatToFp4E2m1(src[srcIdx]);  // bf16→fp4 RINT done on caller-quantised data
            size_t byteIdx = static_cast<size_t>(blk) * packedLd + (k / 2);
            if (k % 2 == 0) bytes[byteIdx] = (bytes[byteIdx] & 0xF0u) | (nib & 0x0Fu);
            else            bytes[byteIdx] = (bytes[byteIdx] & 0x0Fu) | static_cast<uint8_t>((nib & 0x0Fu) << 4);
        }
    }
    return bytes;
}

// ── FP4 packed unpack: packed fp4x2 bytes → logical-element float buffer (count elements). ──
inline std::vector<float> ltUnpackFp4(const std::vector<uint8_t>& bytes, int ld, int numBlocks, size_t count)
{
    int packedLd = fp4PackedLd(ld);
    std::vector<float> out(count, 0.0f);
    for (int blk = 0; blk < numBlocks; blk++) {
        for (int k = 0; k < ld; k++) {
            size_t dstIdx = static_cast<size_t>(blk) * ld + k;
            if (dstIdx >= count) break;
            size_t byteIdx = static_cast<size_t>(blk) * packedLd + (k / 2);
            if (byteIdx >= bytes.size()) break;
            uint8_t nib = (k % 2 == 0) ? (bytes[byteIdx] & 0x0Fu)
                                       : static_cast<uint8_t>((bytes[byteIdx] >> 4) & 0x0Fu);
            out[dstIdx] = ltFp4E2m1ToFloat(nib);
        }
    }
    return out;
}

// unpack device-dtype bytes back to host float
inline std::vector<float> ltUnpackDtype(const std::vector<uint8_t>& bytes, aclDataType dt, size_t count)
{
    std::vector<float> out(count, 0.0f);
    for (size_t i = 0; i < count; i++) {
        switch (dt) {
            case ACL_FLOAT: { float v; std::memcpy(&v, &bytes[i * 4], 4); out[i] = v; break; }
            case ACL_INT32: { int32_t v; std::memcpy(&v, &bytes[i * 4], 4); out[i] = static_cast<float>(v); break; }
            case ACL_FLOAT16: { uint16_t v; std::memcpy(&v, &bytes[i * 2], 2); out[i] = ltFp16ToFloat(v); break; }
            case ACL_BF16: { uint16_t v; std::memcpy(&v, &bytes[i * 2], 2); out[i] = ltBf16ToFloat(v); break; }
            case ACL_INT8: { int8_t v = static_cast<int8_t>(bytes[i]); out[i] = static_cast<float>(v); break; }
            case ACL_FLOAT8_E4M3FN: { out[i] = ltFp8E4m3ToFloat(bytes[i]); break; }
            case ACL_FLOAT8_E5M2:   { out[i] = ltFp8E5m2ToFloat(bytes[i]); break; }
            default: { float v; std::memcpy(&v, &bytes[i * 4], 4); out[i] = v; break; }
        }
    }
    return out;
}

// ── descriptor context ──
struct LtTransformNpuCtx {
    aclblasLtMatrixTransformDesc_t transformDesc = nullptr;
    aclblasLtMatrixLayout_t Adesc = nullptr;
    aclblasLtMatrixLayout_t Bdesc = nullptr;
    aclblasLtMatrixLayout_t Cdesc = nullptr;
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
};

inline void ltDestroyNpuCtx(LtTransformNpuCtx& c)
{
    if (c.dA) { aclrtFree(c.dA); c.dA = nullptr; }
    if (c.dB) { aclrtFree(c.dB); c.dB = nullptr; }
    if (c.dC) { aclrtFree(c.dC); c.dC = nullptr; }
    if (c.transformDesc) { aclblasLtMatrixTransformDescDestroy(c.transformDesc); c.transformDesc = nullptr; }
    if (c.Adesc) { aclblasLtMatrixLayoutDestroy(c.Adesc); c.Adesc = nullptr; }
    if (c.Bdesc) { aclblasLtMatrixLayoutDestroy(c.Bdesc); c.Bdesc = nullptr; }
    if (c.Cdesc) { aclblasLtMatrixLayoutDestroy(c.Cdesc); c.Cdesc = nullptr; }
}

inline aclblasStatus_t ltCreateLayout(
    aclblasLtMatrixLayout_t* desc, aclDataType dt, int rows, int cols, int ld, aclblasLtOrder_t order)
{
    // FP4 layout ld is given to the descriptor in PACKED bytes ((ld+1)/2), while the CSV/golden
    // carry logical-element ld (cast 唯一对端约定, §3 CSV 取值约定). Other dtypes pass ld unchanged.
    int64_t layoutLd = isFp4TransformDtype(dt) ? static_cast<int64_t>(fp4PackedLd(ld))
                                               : static_cast<int64_t>(ld);
    aclblasStatus_t ret = aclblasLtMatrixLayoutCreate(
        desc, dt, static_cast<uint64_t>(rows), static_cast<uint64_t>(cols), layoutLd);
    if (ret != ACLBLAS_STATUS_SUCCESS) return ret;
    int32_t ord = static_cast<int32_t>(order);
    return aclblasLtMatrixLayoutSetAttribute(*desc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &ord, sizeof(int32_t));
}

// physical (pre-op) rows/cols of A given logical op-applied dims
inline void ltPhysDims(int opRows, int opCols, aclblasOperation_t op, int& physRows, int& physCols)
{
    if (op == ACLBLAS_OP_N) { physRows = opRows; physCols = opCols; }
    else { physRows = opCols; physCols = opRows; }
}

// ── NPU wrapper ──
// physA/physB are physical float buffers laid out per orderA/orderB (pre-op physical dims).
// On success, devNdC receives the device output de-layouted to logical column-major (rowsC×colsC).
inline aclblasStatus_t aclblasLtMatrixTransform_npu(
    aclblasLtHandle_t lightHandle, aclrtStream stream,
    const LtMatrixTransformParam& p,
    const std::vector<float>& physA, const std::vector<float>& physB,
    std::vector<float>& devNdC)
{
    if (p.handleNull || lightHandle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;

    LtTransformNpuCtx ctx;
    aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;

    // transform descriptor (may be intentionally null)
    if (!p.transformDescNull) {
        ret = aclblasLtMatrixTransformDescCreate(&ctx.transformDesc, p.scaleType);
        if (ret != ACLBLAS_STATUS_SUCCESS) { ltDestroyNpuCtx(ctx); return ret; }
        int32_t ta = static_cast<int32_t>(p.transA);
        int32_t tb = static_cast<int32_t>(p.transB);
        // TRANSA/TRANSB 的值域在 SetAttribute 阶段校验（白名单 N/T/C），失败须像 ltCreateLayout
        // 传播 ORDER 校验失败一样向上传播，否则越界取值会被静默丢弃、描述符留在默认 OP_N。
        ret = aclblasLtMatrixTransformDescSetAttribute(
            ctx.transformDesc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &ta, sizeof(int32_t));
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            ltDestroyNpuCtx(ctx);
            return ret;
        }
        ret = aclblasLtMatrixTransformDescSetAttribute(
            ctx.transformDesc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &tb, sizeof(int32_t));
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            ltDestroyNpuCtx(ctx);
            return ret;
        }
    }

    // physical pre-op dims
    int physRowsA, physColsA, physRowsB, physColsB;
    ltPhysDims(p.rowsA, p.colsA, p.transA, physRowsA, physColsA);
    ltPhysDims(p.rowsB, p.colsB, p.transB, physRowsB, physColsB);

    const bool hasB = !p.BIsNull;  // B present unless explicitly omitted (beta=0 single-input)

    // layouts (Adesc/Cdesc may be intentionally null)
    if (!p.AdescNull) {
        ret = ltCreateLayout(&ctx.Adesc, p.dtypeA, physRowsA, physColsA, p.lda, p.orderA);
        if (ret != ACLBLAS_STATUS_SUCCESS) { ltDestroyNpuCtx(ctx); return ret; }
    }
    if (hasB && !p.BdescNull) {
        ret = ltCreateLayout(&ctx.Bdesc, p.dtypeB, physRowsB, physColsB, p.ldb, p.orderB);
        if (ret != ACLBLAS_STATUS_SUCCESS) { ltDestroyNpuCtx(ctx); return ret; }
    }
    if (!p.CdescNull) {
        ret = ltCreateLayout(&ctx.Cdesc, p.dtypeC, p.rowsC, p.colsC, p.ldc, p.orderC);
        if (ret != ACLBLAS_STATUS_SUCCESS) { ltDestroyNpuCtx(ctx); return ret; }
        if (p.batchCount != 1) {
            int32_t bc = p.batchCount;  // batchCount>1 → host校验应返回 NOT_SUPPORTED (first batch)
            aclblasLtMatrixLayoutSetAttribute(
                ctx.Cdesc, ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(int32_t));
        }
    }

    const bool emptyMat = (p.rowsC == 0 || p.colsC == 0);

    // FP4 packs 2 logical elements per byte within each ld-block; other dtypes use elem-size bytes.
    // ltPackDtype/ltUnpackDtype handle FP8 (1 byte) inline; ltPackFp4/ltUnpackFp4 own the FP4 path.
    auto packBuf = [&](const std::vector<float>& src, aclDataType dt,
                       aclblasLtOrder_t order, int physRows, int physCols, int ld) -> std::vector<uint8_t> {
        if (isFp4TransformDtype(dt))
            return ltPackFp4(src, ld, ltTransformNumBlocks(order, physRows, physCols));
        return ltPackDtype(src, dt);
    };

    // device buffers (skip on empty matrix / null-injection / nullptr data)
    std::vector<uint8_t> aBytes, bBytes;
    if (!emptyMat && !p.ANull && !physA.empty()) {
        aBytes = packBuf(physA, p.dtypeA, p.orderA, physRowsA, physColsA, p.lda);
        if (aclrtMalloc(&ctx.dA, aBytes.size(), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ltDestroyNpuCtx(ctx); return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclrtMemcpy(ctx.dA, aBytes.size(), aBytes.data(), aBytes.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (!emptyMat && hasB && !physB.empty()) {
        bBytes = packBuf(physB, p.dtypeB, p.orderB, physRowsB, physColsB, p.ldb);
        if (aclrtMalloc(&ctx.dB, bBytes.size(), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ltDestroyNpuCtx(ctx); return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclrtMemcpy(ctx.dB, bBytes.size(), bBytes.data(), bBytes.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    int64_t cCount = emptyMat ? 0 : ltTransformPhysCount(p.orderC, p.rowsC, p.colsC, p.ldc);
    // FP4 output byte size = numBlocks * packedLd(ldc); other dtypes = cCount * elemSize.
    size_t cBytes;
    if (isFp4TransformDtype(p.dtypeC)) {
        int cBlocks = ltTransformNumBlocks(p.orderC, p.rowsC, p.colsC);
        cBytes = emptyMat ? 0 : static_cast<size_t>(cBlocks) * fp4PackedLd(p.ldc);
    } else {
        cBytes = static_cast<size_t>(cCount) * transformDtypeSize(p.dtypeC);
    }
    if (!emptyMat && !p.CIsNull && cBytes > 0) {
        if (aclrtMalloc(&ctx.dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ltDestroyNpuCtx(ctx); return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclrtMemset(ctx.dC, cBytes, 0, cBytes);
    }

    float a = p.alpha, b = p.beta;
    const void* alphaPtr = p.alphaNull ? nullptr : &a;
    const void* betaPtr = p.betaNull ? nullptr : &b;
    // integer path interprets alpha/beta as int32 (scaleType=INT32)
    int32_t ai = static_cast<int32_t>(std::llround(p.alpha));
    int32_t bi = static_cast<int32_t>(std::llround(p.beta));
    if (p.scaleType == ACL_INT32) {
        alphaPtr = p.alphaNull ? nullptr : &ai;
        betaPtr = p.betaNull ? nullptr : &bi;
    }

    const void* Bdev = hasB ? ctx.dB : nullptr;

    ret = aclblasLtMatrixTransform(
        lightHandle, ctx.transformDesc,
        alphaPtr, ctx.dA, ctx.Adesc,
        betaPtr, Bdev, ctx.Bdesc,
        ctx.dC, ctx.Cdesc, stream);

    if (stream != nullptr) aclrtSynchronizeStream(stream);
    else aclrtSynchronizeDevice();

    if (ret == ACLBLAS_STATUS_SUCCESS && !emptyMat && ctx.dC != nullptr) {
        std::vector<uint8_t> cHost(cBytes);
        aclrtMemcpy(cHost.data(), cBytes, ctx.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        std::vector<float> physC;
        if (isFp4TransformDtype(p.dtypeC)) {
            int cBlocks = ltTransformNumBlocks(p.orderC, p.rowsC, p.colsC);
            physC = ltUnpackFp4(cHost, p.ldc, cBlocks, static_cast<size_t>(cCount));
        } else {
            physC = ltUnpackDtype(cHost, p.dtypeC, static_cast<size_t>(cCount));
        }
        devNdC = ltTransformDeLayout(physC, p.orderC, p.rowsC, p.colsC, p.ldc);
    }

    ltDestroyNpuCtx(ctx);
    return ret;
}

