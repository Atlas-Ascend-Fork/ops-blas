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
 * \file matrix_transform_kernel.cpp
 * \brief aclblasLtMatrixTransform kernel (ascend950 / arch35), SIMD membase.
 *        Two device paths:
 *          - Linear fast path: COL/ROW/COL32 with a per-output-column uniform stride
 *            (op=N/T via column-major address arithmetic), scaled add and cross-dtype cast.
 *          - General group-tile path: complex quantization layouts (COL4_4R2_8C /
 *            COL32_2R_4R4) and COL32 op=T, processed per 32-column composite tile with
 *            Gather (complex input) / Scatter (complex output) UB permutation. The in-tile
 *            element offsets come from the shared matrix_transform_perm_table.h single data
 *            source, so the device placement and the test golden de-layout agree byte-for-byte.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "matrix_transform_tiling_data.h"

using namespace AscendC;

// FP8 / FP4 low-precision dtypes. On arch35 (DAV_3510) these alias the compiler builtin
// quantized types; on the CPU-debug / non-3510 path they degrade to byte storage. The matrix
// transform kernel uses them only as template tags: the scaleType-domain Cast does the numeric
// work and the DMA layer moves the raw bytes via a reinterpret to int8_t / uint8_t (the device
// DataCopyPad / Gather / Scatter have no native fp8 / fp4 dtype, design 1.3.A section 3.8).
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using MatFp8E4m3 = float8_e4m3_t;
using MatFp8E5m2 = float8_e5m2_t;
using MatFp4x2 = float4_e2m1x2_t;
#else
struct MatFp8E4m3Tag {
    int8_t v;
};
struct MatFp8E5m2Tag {
    int8_t v;
};
struct MatFp4x2Tag {
    uint8_t v;
};
using MatFp8E4m3 = MatFp8E4m3Tag;
using MatFp8E5m2 = MatFp8E5m2Tag;
using MatFp4x2 = MatFp4x2Tag;
#endif

namespace {
// One UB column slot caps the per-iteration element count; a column longer than this
// is processed in row tiles. 4096 keeps double-buffered FP32 buffers well inside UB.
constexpr uint32_t MT_MAX_TILE_ELEMS = 4096;
constexpr int32_t MT_INT8_MIN = -128;
constexpr int32_t MT_INT8_MAX = 127;

// Compile-time tags for the low-precision dtypes (used to pick the unpack / repack Cast chains
// and the byte-reinterpret DMA). FP8 stores one element per byte (like INT8); FP4 packs two
// elements per byte (fp4x2), so its DMA byte type is uint8_t and its logical element count is
// twice the byte count.
template <typename T>
struct MatIsFp8 {
    static constexpr bool value =
        AscendC::Std::is_same<T, MatFp8E4m3>::value || AscendC::Std::is_same<T, MatFp8E5m2>::value;
};
template <typename T>
struct MatIsFp4 {
    static constexpr bool value = AscendC::Std::is_same<T, MatFp4x2>::value;
};

// Group-tile path constants: one composite tile is at most 32 rows x 32 cols (COL32_2R_4R4),
// so a single tile holds at most 1024 elements. COL4_4R2_8C tiles are 8x32 = 256 elements.
constexpr uint32_t MT_TILE_COLS = 32;
constexpr uint32_t MT_MAX_TILE_ROWS = 32;
constexpr uint32_t MT_MAX_GROUP_ELEMS = MT_MAX_TILE_ROWS * MT_TILE_COLS;

// Whether a (order, op) input read is contiguous (step 1) along a fixed output column.
__aicore__ inline bool MatInputContiguous(uint8_t order, uint8_t op)
{
    return (order == MT_ORDER_COL && op == MT_OP_N) || (order == MT_ORDER_ROW && op == MT_OP_T);
}

// Per-element physical step of an input column for (order, op), column-major basis.
// COL32 op=N walks a fixed 32-column group with a constant stride of 32 (linear-uniform);
// COL32 op=T is handled by a dedicated 2D-burst load, so its step here is unused.
__aicore__ inline uint32_t MatInputStep(uint8_t order, uint8_t op, uint32_t ld)
{
    if (order == MT_ORDER_COL32) {
        return (op == MT_OP_T) ? 1U : 32U;
    }
    return MatInputContiguous(order, op) ? 1U : ld;
}

// Physical base offset of input column c for (order, op), column-major basis.
__aicore__ inline uint64_t MatInputColBase(uint8_t order, uint8_t op, uint32_t ld, uint32_t c)
{
    if (order == MT_ORDER_COL32) {
        // op=N reads logical (r, c) at group*ld + r*32 + c%32; op=T reads A_preop(c, r) whose tile
        // base (row block 0) is c*32 (the 2D-burst load supplies the per-32-row-block ld stride).
        return (op == MT_OP_T) ? static_cast<uint64_t>(c) * 32U
                               : static_cast<uint64_t>(c / 32U) * ld + (c % 32U);
    }
    const bool scaledByLd = (order == MT_ORDER_COL && op == MT_OP_N) ||
                            (order == MT_ORDER_ROW && op == MT_OP_T);
    return scaledByLd ? static_cast<uint64_t>(c) * ld : static_cast<uint64_t>(c);
}

// Output column c base / step for C order, column-major basis (COL32 strides by 32 in a group).
__aicore__ inline uint32_t MatOutputStep(uint8_t order, uint32_t ldc)
{
    if (order == MT_ORDER_COL32) {
        return 32U;
    }
    return (order == MT_ORDER_COL) ? 1U : ldc;
}

__aicore__ inline uint64_t MatOutputColBase(uint8_t order, uint32_t ldc, uint32_t c)
{
    if (order == MT_ORDER_COL32) {
        return static_cast<uint64_t>(c / 32U) * ldc + (c % 32U);
    }
    return (order == MT_ORDER_COL) ? static_cast<uint64_t>(c) * ldc : static_cast<uint64_t>(c);
}

// True when the case needs the general group-tile Gather/Scatter path: a complex quantization
// order on either input or output. COL32 op=T stays on the linear path via a 2D-burst load.
// orderB only counts when B participates (hasB): otherwise a residual complex orderB (the Host
// fills orderB with A's order for the single-input case) would wrongly select the group path with
// a zero-row pivot (MatPickComplexOrderDev ignores orderB when hasB==0).
__aicore__ inline bool MatNeedsGroupPath(uint8_t orderA, uint8_t orderB, uint8_t orderC, uint8_t hasB)
{
    const bool complexA = orderA == MT_ORDER_COL4_4R2_8C || orderA == MT_ORDER_COL32_2R_4R4;
    const bool complexB = (hasB != 0U) &&
                          (orderB == MT_ORDER_COL4_4R2_8C || orderB == MT_ORDER_COL32_2R_4R4);
    const bool complexC = orderC == MT_ORDER_COL4_4R2_8C || orderC == MT_ORDER_COL32_2R_4R4;
    return complexA || complexB || complexC;
}

// DMA byte type for a tensor dtype: the device DataCopyPad / DataCopy have no native fp8 / fp4
// dtype, so fp8 moves as int8_t and fp4x2 (packed) as uint8_t; every other dtype moves natively.
template <typename T>
struct MatDmaByte {
    using type = T;
};
template <>
struct MatDmaByte<MatFp8E4m3> {
    using type = int8_t;
};
template <>
struct MatDmaByte<MatFp8E5m2> {
    using type = int8_t;
};
template <>
struct MatDmaByte<MatFp4x2> {
    using type = uint8_t;
};

// Reinterpret a UB / GM tensor to its DMA byte type (identity for native dtypes). fp8 / fp4 share
// the byte width of their DMA proxy (1 byte), so blockLen / stride byte math built on sizeof(T)
// stays exact after the reinterpret.
template <typename T>
__aicore__ inline LocalTensor<typename MatDmaByte<T>::type> MatAsDmaBytes(const LocalTensor<T>& t)
{
    return t.template ReinterpretCast<typename MatDmaByte<T>::type>();
}
template <typename T>
__aicore__ inline GlobalTensor<typename MatDmaByte<T>::type> MatAsDmaBytes(const GlobalTensor<T>& t)
{
    return t.template ReinterpretCast<typename MatDmaByte<T>::type>();
}

// Cast an input tile into the scaleType domain; same-width types are copied as-is. DataCopy moves
// whole 32-byte blocks, so the element count is rounded up to a block multiple (the buffers carry
// padding headroom); this keeps small tiles (e.g. single-row matrices) correct. fp8 / fp4 are never
// same-width as the scale domain, so they always take the Cast branch: fp8->float (CAST_NONE) and
// fp4x2->bfloat16_t (CAST_NONE, unpacking two elements per byte), per design 1.3.A section 3.8.1.
template <typename TIn, typename TScale>
__aicore__ inline void MatCastToScale(const LocalTensor<TScale>& dst, const LocalTensor<TIn>& src, uint32_t count)
{
    if constexpr (sizeof(TIn) == sizeof(TScale)) {
        constexpr uint32_t elemsPerBlock = 32U / sizeof(TScale);
        const uint32_t aligned = (count + elemsPerBlock - 1U) / elemsPerBlock * elemsPerBlock;
        DataCopy(dst, src.template ReinterpretCast<TScale>(), aligned);
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }
}

// True when an order needs in-tile element permutation (the two complex quantization layouts).
__aicore__ inline bool MatIsComplexOrder(uint8_t order)
{
    return order == MT_ORDER_COL4_4R2_8C || order == MT_ORDER_COL32_2R_4R4;
}

// Device-side composite tile row count (8 for COL4_4R2_8C, 32 for COL32_2R_4R4, else 0). Mirrors
// the host MatComplexTileRows; the permutation offsets themselves come from the Host-built GM table.
__aicore__ inline uint32_t MatTileRowsDev(uint8_t order)
{
    if (order == MT_ORDER_COL4_4R2_8C) {
        return 8U;
    }
    if (order == MT_ORDER_COL32_2R_4R4) {
        return 32U;
    }
    return 0U;
}

// Pick the complex order that drives the group-path in-tile permutation. Mirrors the Host
// MatPickComplexOrder priority (C -> A -> B, B only when present), so kernel tile sizing stays
// consistent with the Host-built index table. Returns a linear order when none participate.
__aicore__ inline uint8_t MatPickComplexOrderDev(uint8_t orderA, uint8_t orderB, uint8_t orderC, uint8_t hasB)
{
    if (MatIsComplexOrder(orderC)) {
        return orderC;
    }
    if (MatIsComplexOrder(orderA)) {
        return orderA;
    }
    if (hasB != 0U && MatIsComplexOrder(orderB)) {
        return orderB;
    }
    return orderC;
}

// Physical element offset of a complex-order tile's group base for output column group g.
__aicore__ inline uint64_t MatComplexGroupBase(uint32_t group, uint32_t rowBlock, uint8_t order, uint32_t ld)
{
    const uint32_t tileSize = MatTileRowsDev(order) * MT_TILE_COLS;
    return static_cast<uint64_t>(group) * ld + static_cast<uint64_t>(rowBlock) * tileSize;
}

// Convert an element-offset table (uint32) into byte offsets for Gather/Scatter. Muls has no
// uint32 overload, so the multiply runs in the int32 domain (offsets <= 1024*4 stay positive).
__aicore__ inline void MatElemOffsetToByte(
    const LocalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& src, uint32_t elemSize, uint32_t count)
{
    LocalTensor<int32_t> dstI = dst.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> srcI = src.ReinterpretCast<int32_t>();
    Muls(dstI, srcI, static_cast<int32_t>(elemSize), count);
}

// Physical element offset of logical element (row, col) under a linear order (COL/ROW/COL32).
__aicore__ inline uint64_t MatPhysOffsetLinear(uint8_t order, uint32_t row, uint32_t col, uint32_t ld)
{
    if (order == MT_ORDER_ROW) {
        return static_cast<uint64_t>(row) * ld + col;
    }
    if (order == MT_ORDER_COL32) {
        return static_cast<uint64_t>(col / 32U) * ld + static_cast<uint64_t>(row) * 32U + (col % 32U);
    }
    return static_cast<uint64_t>(col) * ld + row;  // COL
}

// Group-path tile-column read for a linear input order: returns base/step of the contiguous-in-rowInTile
// element offsets for output tile column (group, rowBlock, colInTile). Affine within one tile block.
__aicore__ inline void MatTileColInputLinear(
    uint8_t order, uint8_t op, uint32_t ld, uint32_t group, uint32_t rowBlock, uint32_t tileRows,
    uint32_t colInTile, uint64_t& base, uint32_t& step)
{
    const uint32_t outCol = group * MT_TILE_COLS + colInTile;
    const uint32_t outRow0 = rowBlock * tileRows;  // first output row in the tile block
    // op=N reads A[outRow, outCol]; op=T reads A[outCol, outRow] (transpose).
    const uint32_t a0 = (op == MT_OP_N) ? outRow0 : outCol;
    const uint32_t b0 = (op == MT_OP_N) ? outCol : outRow0;
    base = MatPhysOffsetLinear(order, a0, b0, ld);
    const uint32_t a1 = (op == MT_OP_N) ? (outRow0 + 1U) : outCol;
    const uint32_t b1 = (op == MT_OP_N) ? outCol : (outRow0 + 1U);
    step = static_cast<uint32_t>(MatPhysOffsetLinear(order, a1, b1, ld) - base);
}
}  // namespace

template <typename TinA, typename TinB, typename ToutC, typename TScale>
class MatrixTransformAIV {
public:
    __aicore__ inline MatrixTransformAIV() {}
    __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR idxGm, GM_ADDR idxAGm, GM_ADDR idxBGm,
                                GM_ADDR ndAGm, GM_ADDR ndBGm, const MatrixTransformTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadStrided(const GlobalTensor<TinA>& src, const LocalTensor<TinA>& dst, uint64_t base,
                                       uint32_t step, uint32_t count);
    __aicore__ inline void LoadCol32Transpose(const GlobalTensor<TinA>& src, const LocalTensor<TinA>& dst,
                                              uint64_t base, uint32_t count);
    __aicore__ inline void LoadStridedB(const GlobalTensor<TinB>& src, const LocalTensor<TinB>& dst, uint64_t base,
                                        uint32_t step, uint32_t count);
    __aicore__ inline void StoreStrided(uint64_t base, uint32_t step, uint32_t count);
    // Cast a scaleType tile into dtype T: fp8 round-to-nearest, the INT32->INT8 three-step chain
    // (int32->float->half->int8, bit-exact for values already in [-128,127]), or a single Cast.
    // Same-width copies and INT8 output saturation are handled by the callers.
    template <typename T>
    __aicore__ inline void CastScaleToDtype(const LocalTensor<T>& dst, const LocalTensor<TScale>& src, uint32_t count);
    __aicore__ inline void CastOutput(const LocalTensor<ToutC>& out, const LocalTensor<TScale>& acc, uint32_t count);
    __aicore__ inline void AccumulateB(const LocalTensor<TScale>& accA, uint32_t col, uint32_t rowOffset,
                                       uint32_t count);
    __aicore__ inline void ComputeTile(uint32_t col, uint32_t rowOffset, uint32_t count);
    __aicore__ inline void ProcessColumn(uint32_t col);
    // Single-row (rows == 1) contiguous-vector path on one core (avoids sub-DataBlock races).
    __aicore__ inline void ProcessSingleRow();
    // Linear fast path (COL/ROW/COL32 with uniform per-output-column stride).
    __aicore__ inline void ProcessLinear();
    // General group-tile path (complex layouts and COL32 op=T).
    __aicore__ inline void BuildTileOffsetTable();
    template <typename T>
    __aicore__ inline void LoadTileColumnMajor(const GlobalTensor<T>& src, const LocalTensor<T>& dst, uint8_t order,
                                               uint8_t op, uint32_t ld, uint32_t group, uint32_t rowBase,
                                               uint32_t blockRows, uint32_t blockCols);
    template <typename T>
    __aicore__ inline void GatherInputTile(const GlobalTensor<T>& gm, uint8_t order, uint32_t ld, uint32_t group,
                                           uint32_t rowBase, const LocalTensor<TScale>& acc);
    __aicore__ inline void StoreOutputTile(uint32_t group, uint32_t rowBlock, uint32_t blockRows, uint32_t blockCols,
                                           const LocalTensor<TScale>& accTile);
    __aicore__ inline void ProcessGroupTile(uint32_t group, uint32_t rowBlock, uint32_t blockRows, uint32_t blockCols);
    __aicore__ inline void ProcessGroups();
    // De-layout pass (complex input + op=T): gather the complex physical input into a logical
    // column-major ND GM workspace (op=N semantics) so the main pass can read it as a COL linear
    // input and apply op=T at the logical ND level. T is the operand's input dtype.
    __aicore__ inline void LoadDelayoutByteOffsets(GM_ADDR idxGm, const LocalTensor<uint32_t>& byteOff);
    template <typename T>
    __aicore__ inline void DelayoutCastToOperand(const LocalTensor<T>& outTile, const LocalTensor<TScale>& accTile);
    template <typename T>
    __aicore__ inline void StoreDelayoutTile(const GlobalTensor<T>& ndGm, const LocalTensor<T>& accSrc, uint32_t group,
                                             uint32_t rowBase, uint32_t physRows, uint32_t blockRows,
                                             uint32_t blockCols);
    template <typename T>
    __aicore__ inline void DelayoutOperand(const GlobalTensor<T>& srcGm, const GlobalTensor<T>& ndGm, uint8_t order,
                                           uint32_t ld, uint32_t physRows, uint32_t physCols, GM_ADDR idxGm);
    __aicore__ inline void DelayoutComplexInputs();
    // FP4 packed pipeline (only instantiated when TScale is bfloat16_t and the operand is fp4x2).
    // Unpack one fp4x2 operand block-by-block into the bf16 ND workspace (logical ld = 2*packedLd);
    // repack the bf16 ND C output block-by-block back into the packed fp4x2 C buffer.
    __aicore__ inline void Fp4UnpackOperand(GM_ADDR srcFp4Gm, GM_ADDR dstBf16Gm, uint32_t packedLd,
                                            uint32_t numBlocks);
    __aicore__ inline void Fp4RepackOutput(GM_ADDR srcBf16Gm, GM_ADDR dstFp4Gm, uint32_t packedLd,
                                           uint32_t numBlocks);
    __aicore__ inline void Fp4UnpackInputs();
    __aicore__ inline void Fp4RepackC();

    TPipe* pipe_;
    GM_ADDR idxGm_;
    GM_ADDR idxAGm_;
    GM_ADDR idxBGm_;
    GM_ADDR aGmRaw_{nullptr};   // raw GM pointers for the byte-level FP4 unpack / repack passes
    GM_ADDR bGmRaw_{nullptr};
    GM_ADDR cGmRaw_{nullptr};
    GM_ADDR ndAGmRaw_{nullptr};
    GM_ADDR ndBGmRaw_{nullptr};
    GlobalTensor<TinA> aGm_;
    GlobalTensor<TinB> bGm_;
    GlobalTensor<ToutC> cGm_;
    GlobalTensor<TinA> ndAGm_;
    GlobalTensor<TinB> ndBGm_;

    TQue<QuePosition::VECIN, 2> inQueueA_;
    TQue<QuePosition::VECIN, 2> inQueueB_;
    TQue<QuePosition::VECOUT, 2> outQueueC_;
    TBuf<QuePosition::VECCALC> calcBufA_;
    TBuf<QuePosition::VECCALC> calcBufB_;
    TBuf<QuePosition::VECCALC> castChainBuf_;  // int32->float->half chain for INT8 output

    // Group-tile path scratch buffers (allocated only when the group path is taken).
    TBuf<QuePosition::VECCALC> tileOffBuf_;     // uint32 in-tile element offset table + byte scratch
    TBuf<QuePosition::VECCALC> physTileBuf_;    // scaleType permuted physical tile (Scatter target)
    TBuf<QuePosition::VECCALC> gatherSrcBuf_;   // contiguous raw physical input tile (complex input)
    TBuf<QuePosition::VECCALC> gatherScaleBuf_; // scaleType cast of the raw tile before Gather
    bool groupPath_{false};
    uint32_t chainHalfBase_{0};  // half-domain start offset within castChainBuf_ (matches the allocated slotElems)

    MatrixTransformTilingData tiling_;
    uint32_t blockIdx_;
    uint32_t blockNum_;
};

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::Init(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR idxGm, GM_ADDR idxAGm, GM_ADDR idxBGm, GM_ADDR ndAGm,
    GM_ADDR ndBGm, const MatrixTransformTilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    idxGm_ = idxGm;
    idxAGm_ = idxAGm;
    idxBGm_ = idxBGm;
    tiling_ = tiling;
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();

    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TinA*>(aGm));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TinB*>(bGm));
    cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ToutC*>(cGm));
    ndAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TinA*>(ndAGm));
    ndBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TinB*>(ndBGm));
    aGmRaw_ = aGm;
    bGmRaw_ = bGm;
    cGmRaw_ = cGm;
    ndAGmRaw_ = ndAGm;
    ndBGmRaw_ = ndBGm;

    // The FP4 instance only runs the unpack / repack passes (the bf16 main pass runs on the bf16
    // template), so it just needs a byte input queue and a bf16-wide output staging buffer.
    if constexpr (AscendC::Std::is_same<TScale, bfloat16_t>::value &&
                  (MatIsFp4<TinA>::value || MatIsFp4<ToutC>::value)) {
        pipe_->InitBuffer(inQueueA_, 2, MT_MAX_TILE_ELEMS * sizeof(bfloat16_t));
        pipe_->InitBuffer(outQueueC_, 2, MT_MAX_TILE_ELEMS * sizeof(bfloat16_t));
        return;
    }

    // Raw-order group-path flag drives buffer sizing: any complex order (input or output, before
    // de-layout) needs the larger group-tile UB slots and the Gather/Scatter scratch. De-layout
    // staging always involves a complex input, so this flag already covers the de-layout pass.
    groupPath_ = MatNeedsGroupPath(tiling_.orderA, tiling_.orderB, tiling_.orderC, tiling_.hasB);
    const uint32_t slotElems = groupPath_ ? MT_MAX_GROUP_ELEMS : MT_MAX_TILE_ELEMS;

    // The de-layout pass casts a staged operand (TinA / TinB) through the output queue and loads
    // its raw tile into gatherSrcBuf_, so these buffers are sized for the widest input dtype too.
    constexpr uint32_t maxInBytes = (sizeof(TinA) > sizeof(TinB)) ? sizeof(TinA) : sizeof(TinB);
    constexpr uint32_t outSlotBytes =
        (sizeof(ToutC) > maxInBytes) ? sizeof(ToutC) : maxInBytes;

    pipe_->InitBuffer(inQueueA_, 2, slotElems * sizeof(TinA));
    pipe_->InitBuffer(outQueueC_, 2, slotElems * outSlotBytes);
    pipe_->InitBuffer(calcBufA_, slotElems * sizeof(TScale));
    if (tiling_.hasB != 0U) {
        pipe_->InitBuffer(inQueueB_, 2, slotElems * sizeof(TinB));
        pipe_->InitBuffer(calcBufB_, slotElems * sizeof(TScale));
    }
    // The INT32->INT8 narrowing chain (int32->float->half->int8) backs both the INT8 output path
    // and the de-layout pass writing an INT8 staged operand back to its ND workspace. Allocate it
    // whenever any participating INT8 tensor exists on the integer scale path.
    if constexpr (sizeof(TScale) == sizeof(int32_t) &&
                  (sizeof(ToutC) == sizeof(int8_t) || sizeof(TinA) == sizeof(int8_t) ||
                   sizeof(TinB) == sizeof(int8_t))) {
        pipe_->InitBuffer(castChainBuf_, slotElems * (sizeof(float) + sizeof(half)));
        chainHalfBase_ = slotElems;
    }
    if (groupPath_) {
        // First MT_MAX_GROUP_ELEMS entries hold the element-offset table, the next block holds the
        // byte-offset scratch (avoids overflowing the table on conversion). Gather/Scatter run in
        // the 4-byte scaleType domain (the device does not support 1-byte Gather/Scatter).
        pipe_->InitBuffer(tileOffBuf_, 2U * MT_MAX_GROUP_ELEMS * sizeof(uint32_t));
        pipe_->InitBuffer(physTileBuf_, MT_MAX_GROUP_ELEMS * sizeof(TScale));
        pipe_->InitBuffer(gatherSrcBuf_, MT_MAX_GROUP_ELEMS * maxInBytes);
        pipe_->InitBuffer(gatherScaleBuf_, MT_MAX_GROUP_ELEMS * sizeof(TScale));
    }
}

// Shared scaleType->dtype cast chain used by both the output cast and the de-layout operand cast.
// fp8 takes a direct CAST_RINT; an INT8 target on the int32 scale path uses the int32->float->half->
// int8 three-step chain (each step bit-exact for values in [-128,127]); every other narrowing or
// reinterpret-equivalent dtype takes a single CAST_RINT. Same-width plain copies and the INT8
// output saturation clamp are kept in the callers (only the output path saturates).
template <typename TinA, typename TinB, typename ToutC, typename TScale>
template <typename T>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::CastScaleToDtype(
    const LocalTensor<T>& dst, const LocalTensor<TScale>& src, uint32_t count)
{
    if constexpr (MatIsFp8<T>::value) {
        // float -> fp8 (scale domain = float): direct CAST_RINT round-to-nearest, matching the host
        // golden's float->fp8 quantisation. The fp8 branch must precede the INT32->INT8 branch below:
        // fp8's storage is 1 byte and the scale type is 4 bytes, which would otherwise alias that
        // branch's sizeof test (float and int32_t are both 4 bytes).
        Cast(dst, src, RoundMode::CAST_RINT, count);
    } else if constexpr (AscendC::Std::is_same<TScale, int32_t>::value && sizeof(T) == sizeof(int8_t)) {
        // INT32 -> INT8 three-step chain (each step is bit-exact for the [-128,127] value range).
        LocalTensor<float> chainF = castChainBuf_.Get<float>();
        LocalTensor<half> chainH = chainF[chainHalfBase_].template ReinterpretCast<half>();
        Cast(chainF, src, RoundMode::CAST_RINT, count);
        Cast(chainH, chainF, RoundMode::CAST_NONE, count);
        Cast(dst, chainH, RoundMode::CAST_RINT, count);
    } else {
        // FP32 -> FP16 / BF16, or INT32 -> (other same-path dtype): single Cast.
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::CastOutput(
    const LocalTensor<ToutC>& out, const LocalTensor<TScale>& acc, uint32_t count)
{
    if constexpr (AscendC::Std::is_same<TScale, int32_t>::value && sizeof(ToutC) == sizeof(int8_t)) {
        // INT32 -> INT8: saturate in the INT32 domain before the shared narrowing chain.
        Maxs(acc, acc, static_cast<TScale>(MT_INT8_MIN), count);
        Mins(acc, acc, static_cast<TScale>(MT_INT8_MAX), count);
        CastScaleToDtype<ToutC>(out, acc, count);
    } else if constexpr (!MatIsFp8<ToutC>::value && sizeof(ToutC) == sizeof(TScale) &&
                         sizeof(TScale) == sizeof(float)) {
        // FP32 in / FP32 out and INT32 in / INT32 out: no dtype change, plain copy. DataCopy moves
        // whole 32-byte blocks, so the count is rounded up to a block multiple (the out buffer carries
        // padding headroom); without this the sub-block tail (e.g. single-row matrices, count < 8) is
        // dropped and stale UB data is written out.
        constexpr uint32_t elemsPerBlock = 32U / sizeof(TScale);
        const uint32_t aligned = (count + elemsPerBlock - 1U) / elemsPerBlock * elemsPerBlock;
        DataCopy(out.template ReinterpretCast<TScale>(), acc, aligned);
    } else {
        CastScaleToDtype<ToutC>(out, acc, count);
    }
}

// COL32 op=T input read for output column c: within each 32-row block the elements are contiguous
// (32 elements at block*ld + c*32), blocks stride by ld. Loaded as a 2D burst (no per-element gap).
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::LoadCol32Transpose(
    const GlobalTensor<TinA>& src, const LocalTensor<TinA>& dst, uint64_t base, uint32_t count)
{
    using DmaT = typename MatDmaByte<TinA>::type;
    const uint16_t nBurst = static_cast<uint16_t>((count + 31U) / 32U);
    const uint32_t blockLen = ((count < 32U) ? count : 32U) * static_cast<uint32_t>(sizeof(TinA));
    const int64_t srcStride = (static_cast<int64_t>(tiling_.lda) - 32) * static_cast<int64_t>(sizeof(TinA));
    DataCopyExtParams params{nBurst, blockLen, srcStride, 0, 0};
    DataCopyPadExtParams<DmaT> pad{false, 0, 0, 0};
    DataCopyPad(MatAsDmaBytes(dst), MatAsDmaBytes(src)[base], params, pad);
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::LoadStrided(
    const GlobalTensor<TinA>& src, const LocalTensor<TinA>& dst, uint64_t base, uint32_t step, uint32_t count)
{
    using DmaT = typename MatDmaByte<TinA>::type;
    if (tiling_.orderA == MT_ORDER_COL32 && tiling_.opA == MT_OP_T) {
        LoadCol32Transpose(src, dst, base, count);
        return;
    }
    if (step == 1U) {
        DataCopyExtParams params{1U, static_cast<uint32_t>(count * sizeof(TinA)), 0, 0, 0};
        DataCopyPadExtParams<DmaT> pad{false, 0, 0, 0};
        DataCopyPad(MatAsDmaBytes(dst), MatAsDmaBytes(src)[base], params, pad);
        return;
    }
    // Gather count single-element blocks at GM stride into a contiguous UB buffer.
    // Compact mode packs the sub-32B blocks tightly in UB; Normal mode would pad each
    // block up to a 32B DataBlock and break the contiguous compute that follows.
    const int64_t srcStride = static_cast<int64_t>((step - 1U)) * static_cast<int64_t>(sizeof(TinA));
    DataCopyExtParams params{static_cast<uint16_t>(count), static_cast<uint32_t>(sizeof(TinA)), srcStride, 0, 0};
    DataCopyPadExtParams<DmaT> pad{false, 0, 0, 0};
    DataCopyPad<DmaT, PaddingMode::Compact>(MatAsDmaBytes(dst), MatAsDmaBytes(src)[base], params, pad);
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::LoadStridedB(
    const GlobalTensor<TinB>& src, const LocalTensor<TinB>& dst, uint64_t base, uint32_t step, uint32_t count)
{
    using DmaT = typename MatDmaByte<TinB>::type;
    if (step == 1U) {
        DataCopyExtParams params{1U, static_cast<uint32_t>(count * sizeof(TinB)), 0, 0, 0};
        DataCopyPadExtParams<DmaT> pad{false, 0, 0, 0};
        DataCopyPad(MatAsDmaBytes(dst), MatAsDmaBytes(src)[base], params, pad);
        return;
    }
    const int64_t srcStride = static_cast<int64_t>((step - 1U)) * static_cast<int64_t>(sizeof(TinB));
    DataCopyExtParams params{static_cast<uint16_t>(count), static_cast<uint32_t>(sizeof(TinB)), srcStride, 0, 0};
    DataCopyPadExtParams<DmaT> pad{false, 0, 0, 0};
    DataCopyPad<DmaT, PaddingMode::Compact>(MatAsDmaBytes(dst), MatAsDmaBytes(src)[base], params, pad);
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::StoreStrided(
    uint64_t base, uint32_t step, uint32_t count)
{
    using DmaT = typename MatDmaByte<ToutC>::type;
    LocalTensor<ToutC> out = outQueueC_.DeQue<ToutC>();
    if (step == 1U) {
        DataCopyExtParams params{1U, static_cast<uint32_t>(count * sizeof(ToutC)), 0, 0, 0};
        DataCopyPad(MatAsDmaBytes(cGm_)[base], MatAsDmaBytes(out), params);
    } else {
        // Scatter from a contiguous UB buffer to count single-element blocks at GM stride.
        // Compact mode reads the sub-32B UB blocks tightly; Normal mode would expect each
        // source block padded to a 32B DataBlock and read stale data between elements.
        const int64_t dstStride = static_cast<int64_t>((step - 1U)) * static_cast<int64_t>(sizeof(ToutC));
        DataCopyExtParams params{static_cast<uint16_t>(count), static_cast<uint32_t>(sizeof(ToutC)), 0, dstStride, 0};
        DataCopyPad<DmaT, PaddingMode::Compact>(MatAsDmaBytes(cGm_)[base], MatAsDmaBytes(out), params);
    }
    outQueueC_.FreeTensor(out);
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::AccumulateB(
    const LocalTensor<TScale>& accA, uint32_t col, uint32_t rowOffset, uint32_t count)
{
    const uint32_t stepB = MatInputStep(tiling_.orderB, tiling_.opB, tiling_.ldb);
    const uint64_t baseB = MatInputColBase(tiling_.orderB, tiling_.opB, tiling_.ldb, col) +
                           static_cast<uint64_t>(rowOffset) * stepB;
    LocalTensor<TinB> bIn = inQueueB_.AllocTensor<TinB>();
    LoadStridedB(bGm_, bIn, baseB, stepB, count);
    inQueueB_.EnQue(bIn);

    LocalTensor<TinB> bLoaded = inQueueB_.DeQue<TinB>();
    LocalTensor<TScale> accB = calcBufB_.Get<TScale>();
    MatCastToScale<TinB, TScale>(accB, bLoaded, count);
    inQueueB_.FreeTensor(bLoaded);
    const TScale beta = *reinterpret_cast<const TScale*>(&tiling_.betaBits);
    Muls(accB, accB, beta, count);
    Add(accA, accA, accB, count);
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::ComputeTile(
    uint32_t col, uint32_t rowOffset, uint32_t count)
{
    const uint32_t stepA = MatInputStep(tiling_.orderA, tiling_.opA, tiling_.lda);
    const uint64_t baseA = MatInputColBase(tiling_.orderA, tiling_.opA, tiling_.lda, col) +
                           static_cast<uint64_t>(rowOffset) * stepA;
    LocalTensor<TinA> aIn = inQueueA_.AllocTensor<TinA>();
    LoadStrided(aGm_, aIn, baseA, stepA, count);
    inQueueA_.EnQue(aIn);

    LocalTensor<TinA> aLoaded = inQueueA_.DeQue<TinA>();
    LocalTensor<TScale> accA = calcBufA_.Get<TScale>();
    MatCastToScale<TinA, TScale>(accA, aLoaded, count);
    inQueueA_.FreeTensor(aLoaded);
    const TScale alpha = *reinterpret_cast<const TScale*>(&tiling_.alphaBits);
    Muls(accA, accA, alpha, count);

    if (tiling_.hasB != 0U) {
        AccumulateB(accA, col, rowOffset, count);
    }

    LocalTensor<ToutC> out = outQueueC_.AllocTensor<ToutC>();
    CastOutput(out, accA, count);
    outQueueC_.EnQue(out);

    const uint32_t stepC = MatOutputStep(tiling_.orderC, tiling_.ldc);
    const uint64_t baseC = MatOutputColBase(tiling_.orderC, tiling_.ldc, col) +
                           static_cast<uint64_t>(rowOffset) * stepC;
    StoreStrided(baseC, stepC, count);
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::ProcessColumn(uint32_t col)
{
    const uint32_t rows = tiling_.rows;
    uint32_t rowOffset = 0U;
    while (rowOffset < rows) {
        uint32_t count = rows - rowOffset;
        if (count > MT_MAX_TILE_ELEMS) {
            count = MT_MAX_TILE_ELEMS;
        }
        ComputeTile(col, rowOffset, count);
        rowOffset += count;
    }
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::ProcessLinear()
{
    for (uint32_t col = blockIdx_; col < tiling_.cols; col += blockNum_) {
        ProcessColumn(col);
    }
}

// Single-row matrices (rows == 1) have a one-element-per-column output. Writing each column on a
// separate core would race on shared 32-byte DataBlocks, so the whole row is processed as one
// contiguous vector on a single core: the cols elements are gathered/scattered with the per-column
// stride and stored in one shot, avoiding sub-DataBlock multi-core writes.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::ProcessSingleRow()
{
    if (blockIdx_ != 0U) {
        return;
    }
    const uint32_t cols = tiling_.cols;
    const uint32_t stepA = static_cast<uint32_t>(MatInputColBase(tiling_.orderA, tiling_.opA, tiling_.lda, 1U) -
                                                 MatInputColBase(tiling_.orderA, tiling_.opA, tiling_.lda, 0U));
    LocalTensor<TinA> aIn = inQueueA_.AllocTensor<TinA>();
    LoadStrided(aGm_, aIn, MatInputColBase(tiling_.orderA, tiling_.opA, tiling_.lda, 0U), stepA, cols);
    inQueueA_.EnQue(aIn);
    LocalTensor<TinA> aLoaded = inQueueA_.DeQue<TinA>();
    LocalTensor<TScale> accA = calcBufA_.Get<TScale>();
    MatCastToScale<TinA, TScale>(accA, aLoaded, cols);
    inQueueA_.FreeTensor(aLoaded);
    const TScale alpha = *reinterpret_cast<const TScale*>(&tiling_.alphaBits);
    Muls(accA, accA, alpha, cols);
    if (tiling_.hasB != 0U) {
        const uint32_t stepB = static_cast<uint32_t>(MatInputColBase(tiling_.orderB, tiling_.opB, tiling_.ldb, 1U) -
                                                     MatInputColBase(tiling_.orderB, tiling_.opB, tiling_.ldb, 0U));
        LocalTensor<TinB> bIn = inQueueB_.AllocTensor<TinB>();
        LoadStridedB(bGm_, bIn, MatInputColBase(tiling_.orderB, tiling_.opB, tiling_.ldb, 0U), stepB, cols);
        inQueueB_.EnQue(bIn);
        LocalTensor<TinB> bLoaded = inQueueB_.DeQue<TinB>();
        LocalTensor<TScale> accB = calcBufB_.Get<TScale>();
        MatCastToScale<TinB, TScale>(accB, bLoaded, cols);
        inQueueB_.FreeTensor(bLoaded);
        const TScale beta = *reinterpret_cast<const TScale*>(&tiling_.betaBits);
        Muls(accB, accB, beta, cols);
        Add(accA, accA, accB, cols);
    }
    LocalTensor<ToutC> out = outQueueC_.AllocTensor<ToutC>();
    CastOutput(out, accA, cols);
    outQueueC_.EnQue(out);
    const uint32_t stepC = static_cast<uint32_t>(MatOutputColBase(tiling_.orderC, tiling_.ldc, 1U) -
                                                 MatOutputColBase(tiling_.orderC, tiling_.ldc, 0U));
    StoreStrided(MatOutputColBase(tiling_.orderC, tiling_.ldc, 0U), stepC, cols);
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::Process()
{
    if (tiling_.rows == 0U || tiling_.cols == 0U || blockNum_ == 0U) {
        return;
    }
    // De-layout phase (complex input + op=T): a dedicated launch gathers each complex physical
    // input into a plain column-major ND GM workspace (op=N semantics). The main phase is a
    // separate launch (host stream sync between them, no device cross-core sync) that reads the
    // workspace as a COL linear input and applies op=T, so the transpose is organised at the
    // logical ND level (single-tile transpose cannot cover the op=T cross-tile remap), matching
    // golden's de-layout->applyOp(transpose) order.
    if (tiling_.phase == MT_PHASE_DELAYOUT) {
        DelayoutComplexInputs();
        return;
    }
    // FP4 packed pipeline: the unpack / repack phases run on the fp4 template instance; the main
    // bf16 transform runs on the bf16 instance (host-orchestrated separate launches).
    if (tiling_.phase == MT_PHASE_FP4_UNPACK) {
        Fp4UnpackInputs();
        return;
    }
    if (tiling_.phase == MT_PHASE_FP4_REPACK) {
        Fp4RepackC();
        return;
    }
    if (!groupPath_ && tiling_.rows == 1U) {
        ProcessSingleRow();
        return;
    }
    if (groupPath_) {
        ProcessGroups();
    } else {
        ProcessLinear();
    }
}

// Load the operand's column-major in-tile offset table (built for its own complex order) and
// convert it to byte offsets in tbl[MT_MAX_GROUP_ELEMS..]. A MTE2->V flag pairs the table DMA load
// with the Muls so the conversion (and the dependent Gather) read the fully-loaded table.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::LoadDelayoutByteOffsets(
    GM_ADDR idxGm, const LocalTensor<uint32_t>& byteOff)
{
    LocalTensor<uint32_t> tbl = tileOffBuf_.Get<uint32_t>();
    GlobalTensor<uint32_t> idx;
    idx.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(idxGm));
    DataCopyExtParams tblParams{1U, MT_MAX_GROUP_ELEMS * static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> tblPad{false, 0, 0, 0};
    DataCopyPad(tbl, idx, tblParams, tblPad);
    const int32_t tblEid = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(tblEid);
    WaitFlag<HardEvent::MTE2_V>(tblEid);
    MatElemOffsetToByte(byteOff, tbl, static_cast<uint32_t>(sizeof(TScale)), MT_MAX_GROUP_ELEMS);
}

// Cast a de-layouted scaleType tile back to the operand dtype. Same-width is a plain copy; an INT8
// operand uses the int32->float->half->int8 chain (the device has no direct INT32->INT8 cast and the
// de-layout values are exact INT8, so each step is bit-exact for [-128,127]); other widths cast once.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
template <typename T>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::DelayoutCastToOperand(
    const LocalTensor<T>& outTile, const LocalTensor<TScale>& accTile)
{
    if constexpr (!MatIsFp8<T>::value && sizeof(T) == sizeof(TScale)) {
        // Same-width operand (e.g. FP32 / INT32): plain copy of the whole fixed-size tile.
        DataCopy(outTile.template ReinterpretCast<TScale>(), accTile, MT_MAX_GROUP_ELEMS);
    } else {
        // fp8 / INT8 / narrowing: the de-layout values are exact, so no output saturation is needed
        // (unlike CastOutput); the shared cast chain handles fp8, the INT8 chain and the single Cast.
        CastScaleToDtype<T>(outTile, accTile, MT_MAX_GROUP_ELEMS);
    }
}

// Store one de-layouted column-major tile (accSrc) to the COL ND workspace via the output queue:
// nd column (group*32 + colInTile) is contiguous, starting at base + rowBase. EnQue/DeQue pairs the
// vector cast with the DMA store and serialises against the next tile's compute (double-buffered).
template <typename TinA, typename TinB, typename ToutC, typename TScale>
template <typename T>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::StoreDelayoutTile(
    const GlobalTensor<T>& ndGm, const LocalTensor<T>& accSrc, uint32_t group, uint32_t rowBase, uint32_t physRows,
    uint32_t blockRows, uint32_t blockCols)
{
    outQueueC_.EnQue(accSrc);
    LocalTensor<T> outDeq = outQueueC_.DeQue<T>();
    for (uint32_t colInTile = 0U; colInTile < blockCols; ++colInTile) {
        const uint32_t ndCol = group * MT_TILE_COLS + colInTile;
        const uint64_t base = static_cast<uint64_t>(ndCol) * physRows + rowBase;
        LocalTensor<T> slot = outDeq[colInTile * MT_TILE_COLS];
        DataCopyExtParams sp{1U, blockRows * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(MatAsDmaBytes(ndGm)[base], MatAsDmaBytes(slot), sp);
    }
    outQueueC_.FreeTensor(outDeq);
}

// De-layout one complex operand: each core processes a grid-stride share of the operand's physical
// composite tiles, gathers each tile into a column-major scaleType buffer (op=N de-layout), casts
// it back to the operand dtype and writes it to the ND workspace at plain COL positions
// (nd[col*physRows + row]). The op=T transpose is applied later by the COL-order main pass.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
template <typename T>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::DelayoutOperand(
    const GlobalTensor<T>& srcGm, const GlobalTensor<T>& ndGm, uint8_t order, uint32_t ld, uint32_t physRows,
    uint32_t physCols, GM_ADDR idxGm)
{
    LocalTensor<uint32_t> byteOff = tileOffBuf_.Get<uint32_t>()[MT_MAX_GROUP_ELEMS];
    LoadDelayoutByteOffsets(idxGm, byteOff);

    const uint32_t tileRows = MatTileRowsDev(order);
    const uint32_t physCount = tileRows * MT_TILE_COLS;
    const uint32_t numGroups = (physCols + MT_TILE_COLS - 1U) / MT_TILE_COLS;
    const uint32_t numRowBlocks = (physRows + tileRows - 1U) / tileRows;
    const uint32_t totalTiles = numGroups * numRowBlocks;
    for (uint32_t t = blockIdx_; t < totalTiles; t += blockNum_) {
        const uint32_t group = t / numRowBlocks;
        const uint32_t rowBlock = t % numRowBlocks;
        const uint32_t rowBase = rowBlock * tileRows;
        uint32_t blockRows = physRows - rowBase;
        if (blockRows > tileRows) {
            blockRows = tileRows;
        }
        uint32_t blockCols = physCols - group * MT_TILE_COLS;
        if (blockCols > MT_TILE_COLS) {
            blockCols = MT_TILE_COLS;
        }
        // Gather the physical tile into column-major scaleType order (accTile[col*32 + row]).
        LocalTensor<T> rawTile = gatherSrcBuf_.Get<T>();
        const uint64_t gbase = MatComplexGroupBase(group, rowBlock, order, ld);
        DataCopyExtParams p{1U, physCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<typename MatDmaByte<T>::type> pad{false, 0, 0, 0};
        DataCopyPad(MatAsDmaBytes(rawTile), MatAsDmaBytes(srcGm)[gbase], p, pad);
        // gatherSrcBuf_ is a raw single-slot TBuf, so no EnQue/DeQue flag orders this MTE2 load
        // against the Vector pipe. Pair it with the Cast that consumes it (RAW)...
        const int32_t rawEid = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(rawEid);
        WaitFlag<HardEvent::MTE2_V>(rawEid);
        LocalTensor<TScale> physScale = gatherScaleBuf_.Get<TScale>();
        MatCastToScale<T, TScale>(physScale, rawTile, physCount);
        // ...and retire that read before the next loop iteration overwrites the same slot (WAR).
        const int32_t warEid = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(warEid);
        WaitFlag<HardEvent::V_MTE2>(warEid);
        LocalTensor<TScale> accTile = calcBufA_.Get<TScale>();
        Gather(accTile, physScale, byteOff, 0U, MT_MAX_GROUP_ELEMS);

        LocalTensor<T> outTile = outQueueC_.AllocTensor<T>();
        DelayoutCastToOperand<T>(outTile, accTile);
        StoreDelayoutTile<T>(ndGm, outTile, group, rowBase, physRows, blockRows, blockCols);
    }
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::DelayoutComplexInputs()
{
    if (tiling_.needDelayoutA != 0U) {
        DelayoutOperand<TinA>(aGm_, ndAGm_, tiling_.orderA, tiling_.lda, tiling_.physRowsA, tiling_.physColsA,
                              idxAGm_);
    }
    if (tiling_.needDelayoutB != 0U) {
        DelayoutOperand<TinB>(bGm_, ndBGm_, tiling_.orderB, tiling_.ldb, tiling_.physRowsB, tiling_.physColsB,
                              idxBGm_);
    }
}

// Unpack one packed fp4x2 operand into its bf16 ND workspace. The packed buffer (numBlocks blocks
// of packedLd bytes) and the bf16 ND (numBlocks blocks of 2*packedLd elements) are both physically
// contiguous and the fp4x2->bf16 Cast emits byte j -> elements {2j (low nibble), 2j+1 (high nibble)},
// matching the host ltUnpackFp4 convention, so the whole operand unpacks as one flat byte->bf16
// stream split into per-core chunks. The bf16 ND keeps the same physical placement as the fp4 input
// (logical ld = 2*packedLd), so the bf16 main pass reads it with the operand's order unchanged.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::Fp4UnpackOperand(
    GM_ADDR srcFp4Gm, GM_ADDR dstBf16Gm, uint32_t packedLd, uint32_t numBlocks)
{
    if constexpr (AscendC::Std::is_same<TScale, bfloat16_t>::value) {
        const uint64_t totalBytes = static_cast<uint64_t>(numBlocks) * packedLd;
        GlobalTensor<uint8_t> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcFp4Gm));
        GlobalTensor<bfloat16_t> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dstBf16Gm));
        // Each byte yields two bf16 elements; chunk so the bf16 result fits one calc slot.
        const uint32_t chunkBytes = MT_MAX_TILE_ELEMS / 2U;  // 2*chunkBytes <= MT_MAX_TILE_ELEMS bf16
        const uint64_t numChunks = (totalBytes + chunkBytes - 1U) / chunkBytes;
        for (uint64_t ch = blockIdx_; ch < numChunks; ch += blockNum_) {
            const uint64_t byteBase = ch * chunkBytes;
            uint32_t bytes = static_cast<uint32_t>(totalBytes - byteBase);
            if (bytes > chunkBytes) {
                bytes = chunkBytes;
            }
            LocalTensor<uint8_t> raw = inQueueA_.AllocTensor<uint8_t>();
            DataCopyExtParams lp{1U, bytes, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> lpad{false, 0, 0, 0};
            DataCopyPad(raw, src[byteBase], lp, lpad);
            inQueueA_.EnQue(raw);
            LocalTensor<uint8_t> rawDeq = inQueueA_.DeQue<uint8_t>();
            LocalTensor<bfloat16_t> bf = outQueueC_.AllocTensor<bfloat16_t>();
            Cast(bf, rawDeq.template ReinterpretCast<MatFp4x2>(), RoundMode::CAST_NONE, bytes * 2U);
            inQueueA_.FreeTensor(rawDeq);
            outQueueC_.EnQue(bf);
            LocalTensor<bfloat16_t> bfDeq = outQueueC_.DeQue<bfloat16_t>();
            DataCopyExtParams sp{1U, bytes * 2U * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(dst[byteBase * 2U], bfDeq, sp);
            outQueueC_.FreeTensor(bfDeq);
        }
    } else {
        (void)srcFp4Gm;
        (void)dstBf16Gm;
        (void)packedLd;
        (void)numBlocks;
    }
}

// Repack the bf16 ND C output back into the packed fp4x2 C buffer (inverse of Fp4UnpackOperand):
// each pair of bf16 elements {2j, 2j+1} casts to byte j of the packed output (CAST_RINT, matching
// the host ltFloatToFp4E2m1 round-half-to-even). The physical placement is identical to the bf16 ND
// output (logical ld = 2*packedLd), so the packed C buffer (numBlocks*packedLd bytes) is one flat
// bf16->byte stream split into per-core chunks.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::Fp4RepackOutput(
    GM_ADDR srcBf16Gm, GM_ADDR dstFp4Gm, uint32_t packedLd, uint32_t numBlocks)
{
    if constexpr (AscendC::Std::is_same<TScale, bfloat16_t>::value) {
        const uint64_t totalBytes = static_cast<uint64_t>(numBlocks) * packedLd;
        GlobalTensor<bfloat16_t> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(srcBf16Gm));
        GlobalTensor<uint8_t> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dstFp4Gm));
        const uint32_t chunkBytes = MT_MAX_TILE_ELEMS / 2U;  // reads 2*chunkBytes bf16 per chunk
        const uint64_t numChunks = (totalBytes + chunkBytes - 1U) / chunkBytes;
        for (uint64_t ch = blockIdx_; ch < numChunks; ch += blockNum_) {
            const uint64_t byteBase = ch * chunkBytes;
            uint32_t bytes = static_cast<uint32_t>(totalBytes - byteBase);
            if (bytes > chunkBytes) {
                bytes = chunkBytes;
            }
            LocalTensor<bfloat16_t> bf = inQueueA_.AllocTensor<bfloat16_t>();
            DataCopyExtParams lp{1U, bytes * 2U * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPadExtParams<bfloat16_t> lpad{false, 0, 0, 0};
            DataCopyPad(bf, src[byteBase * 2U], lp, lpad);
            inQueueA_.EnQue(bf);
            LocalTensor<bfloat16_t> bfDeq = inQueueA_.DeQue<bfloat16_t>();
            LocalTensor<uint8_t> raw = outQueueC_.AllocTensor<uint8_t>();
            Cast(raw.template ReinterpretCast<MatFp4x2>(), bfDeq, RoundMode::CAST_RINT, bytes * 2U);
            inQueueA_.FreeTensor(bfDeq);
            outQueueC_.EnQue(raw);
            LocalTensor<uint8_t> rawDeq = outQueueC_.DeQue<uint8_t>();
            DataCopyExtParams sp{1U, bytes, 0, 0, 0};
            DataCopyPad(dst[byteBase], rawDeq, sp);
            outQueueC_.FreeTensor(rawDeq);
        }
    } else {
        (void)srcBf16Gm;
        (void)dstFp4Gm;
        (void)packedLd;
        (void)numBlocks;
    }
}

// FP4 unpack phase: unpack each fp4x2 input operand (A, and B when present) into its bf16 ND
// workspace. aGm_ / bGm_ hold the packed fp4 inputs; ndAGm_ / ndBGm_ are the bf16 ND targets.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::Fp4UnpackInputs()
{
    if (tiling_.fp4IsA != 0U) {
        Fp4UnpackOperand(aGmRaw_, ndAGmRaw_, tiling_.lda, tiling_.fp4NumBlocksA);
    }
    if (tiling_.hasB != 0U && tiling_.fp4IsB != 0U) {
        Fp4UnpackOperand(bGmRaw_, ndBGmRaw_, tiling_.ldb, tiling_.fp4NumBlocksB);
    }
}

// FP4 repack phase: repack the bf16 ND C output (carried in aGm_) into the packed fp4x2 C buffer
// (cGm_). The host wires the bf16 ND C pointer into aGm_ for this phase.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::Fp4RepackC()
{
    if (tiling_.fp4IsC != 0U) {
        Fp4RepackOutput(aGmRaw_, cGmRaw_, tiling_.ldc, tiling_.fp4NumBlocksC);
    }
}

// Load the column-major in-tile element-offset table from the Host-prepared GM workspace into UB
// (data-independent, single data source per matrix_transform_perm_table.h, no SetValue: R1-safe).
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::BuildTileOffsetTable()
{
    // The Host table is a fixed 32x32 column-major tile (with padding scratch slots) regardless of
    // the complex order, so the device always loads MT_MAX_GROUP_ELEMS entries.
    LocalTensor<uint32_t> tbl = tileOffBuf_.Get<uint32_t>();
    GlobalTensor<uint32_t> idx;
    idx.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(idxGm_));
    DataCopyExtParams params{1U, MT_MAX_GROUP_ELEMS * static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
    DataCopyPad(tbl, idx, params, pad);
    // The table lands in a raw TBuf (not a TQue), so no EnQue/DeQue flag is inserted for it. Pair the
    // MTE2 load with the first Vector consumer (MatElemOffsetToByte in GatherInputTile / the Scatter
    // index build) explicitly, otherwise that consumer reads tbl before the DMA has landed.
    const int32_t tblEid = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(tblEid);
    WaitFlag<HardEvent::MTE2_V>(tblEid);
}

// Load one tile column (blockRows contiguous-in-rowInTile elements) of a linear input order into
// the column-major accumulator slot at [colInTile * 32]. The fixed 32-element column stride keeps
// every per-column UB load on a 32-byte boundary (required for INT8). base/step are affine.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
template <typename T>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::LoadTileColumnMajor(
    const GlobalTensor<T>& src, const LocalTensor<T>& dst, uint8_t order, uint8_t op, uint32_t ld, uint32_t group,
    uint32_t rowBase, uint32_t blockRows, uint32_t blockCols)
{
    const uint8_t pivotOrder =
        MatPickComplexOrderDev(tiling_.orderA, tiling_.orderB, tiling_.orderC, tiling_.hasB);
    const uint32_t tileRows = MatTileRowsDev(pivotOrder);
    const uint32_t rowBlock = rowBase / tileRows;
    for (uint32_t colInTile = 0U; colInTile < blockCols; ++colInTile) {
        uint64_t base = 0U;
        uint32_t step = 0U;
        MatTileColInputLinear(order, op, ld, group, rowBlock, tileRows, colInTile, base, step);
        LocalTensor<T> slot = dst[colInTile * MT_TILE_COLS];
        using DmaT = typename MatDmaByte<T>::type;
        if (step == 1U) {
            DataCopyExtParams p{1U, blockRows * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<DmaT> pad{false, 0, 0, 0};
            DataCopyPad(MatAsDmaBytes(slot), MatAsDmaBytes(src)[base], p, pad);
        } else {
            const int64_t srcStride = static_cast<int64_t>(step - 1U) * static_cast<int64_t>(sizeof(T));
            DataCopyExtParams p{static_cast<uint16_t>(blockRows), static_cast<uint32_t>(sizeof(T)), srcStride, 0, 0};
            DataCopyPadExtParams<DmaT> pad{false, 0, 0, 0};
            DataCopyPad<DmaT, PaddingMode::Compact>(MatAsDmaBytes(slot), MatAsDmaBytes(src)[base], p, pad);
        }
    }
}

// Gather a complex-order input tile (group, rowBlock) of operand gm (order, ld) into a column-major
// scaleType buffer. The raw tile is loaded contiguously (physical in-tile order), cast to the 4-byte
// scaleType domain, then Gather-permuted to logical column-major order (no 1-byte Gather on device).
// Shared by A and B (the only difference is the operand dtype / GM tensor / order / ld).
template <typename TinA, typename TinB, typename ToutC, typename TScale>
template <typename T>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::GatherInputTile(
    const GlobalTensor<T>& gm, uint8_t order, uint32_t ld, uint32_t group, uint32_t rowBase,
    const LocalTensor<TScale>& acc)
{
    const uint32_t tileRows = MatTileRowsDev(order);
    const uint32_t rowBlock = rowBase / tileRows;
    const uint32_t physCount = tileRows * MT_TILE_COLS;  // physical tile element count
    LocalTensor<T> rawTile = gatherSrcBuf_.Get<T>();
    const uint64_t gbase = MatComplexGroupBase(group, rowBlock, order, ld);
    DataCopyExtParams p{1U, physCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<typename MatDmaByte<T>::type> pad{false, 0, 0, 0};
    DataCopyPad(MatAsDmaBytes(rawTile), MatAsDmaBytes(gm)[gbase], p, pad);
    // gatherSrcBuf_ is a raw single-slot TBuf shared by the A and B gathers of the same tile (and by
    // every tile of the loop), so no EnQue/DeQue flag orders these MTE2 loads against the Vector
    // pipe. Pair the load with the Cast that consumes it (RAW)...
    const int32_t rawEid = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(rawEid);
    WaitFlag<HardEvent::MTE2_V>(rawEid);

    LocalTensor<TScale> physScale = gatherScaleBuf_.Get<TScale>();
    MatCastToScale<T, TScale>(physScale, rawTile, physCount);
    // ...and retire that read before the next gather (the other operand, or the next tile)
    // overwrites the same slot (WAR).
    const int32_t warEid = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(warEid);
    WaitFlag<HardEvent::V_MTE2>(warEid);
    LocalTensor<uint32_t> off = tileOffBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> byteOff = off[MT_MAX_GROUP_ELEMS];
    MatElemOffsetToByte(byteOff, off, static_cast<uint32_t>(sizeof(TScale)), MT_MAX_GROUP_ELEMS);
    Gather(acc, physScale, byteOff, 0U, MT_MAX_GROUP_ELEMS);
}

// Store the scaleType accumulator tile. Complex output Scatter-permutes the 4-byte accumulator
// into a physical tile (no 1-byte Scatter on device), casts it to the output dtype and DataCopyPad
// to GM; linear output casts each tile column and writes it with a strided copy.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::StoreOutputTile(
    uint32_t group, uint32_t rowBlock, uint32_t blockRows, uint32_t blockCols, const LocalTensor<TScale>& accTile)
{
    if (MatIsComplexOrder(tiling_.orderC)) {
        const uint32_t tileRows = MatTileRowsDev(tiling_.orderC);
        const uint32_t tileSize = tileRows * MT_TILE_COLS;
        (void)blockRows;
        (void)blockCols;
        LocalTensor<uint32_t> off = tileOffBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> byteOff = off[MT_MAX_GROUP_ELEMS];
        MatElemOffsetToByte(byteOff, off, static_cast<uint32_t>(sizeof(TScale)), MT_MAX_GROUP_ELEMS);
        LocalTensor<TScale> physScale = physTileBuf_.Get<TScale>();
        Scatter(physScale, accTile, byteOff, 0U, MT_MAX_GROUP_ELEMS);
        LocalTensor<ToutC> out = outQueueC_.AllocTensor<ToutC>();
        CastOutput(out, physScale, tileSize);
        outQueueC_.EnQue(out);
        LocalTensor<ToutC> outDeq = outQueueC_.DeQue<ToutC>();
        const uint64_t gbase = MatComplexGroupBase(group, rowBlock, tiling_.orderC, tiling_.ldc);
        DataCopyExtParams p{1U, tileSize * static_cast<uint32_t>(sizeof(ToutC)), 0, 0, 0};
        DataCopyPad(MatAsDmaBytes(cGm_)[gbase], MatAsDmaBytes(outDeq), p);
        outQueueC_.FreeTensor(outDeq);
        return;
    }
    // Linear output (complex input -> COL/ROW/COL32): logical row block size follows the driving
    // complex order (C -> A -> B), matching ProcessGroups' tile sizing.
    const uint8_t pivotOrder =
        MatPickComplexOrderDev(tiling_.orderA, tiling_.orderB, tiling_.orderC, tiling_.hasB);
    const uint32_t tileRows = MatTileRowsDev(pivotOrder);
    const uint32_t logicalRows = (tileRows != 0U) ? tileRows : MT_MAX_TILE_ROWS;
    LocalTensor<ToutC> out = outQueueC_.AllocTensor<ToutC>();
    CastOutput(out, accTile, MT_MAX_GROUP_ELEMS);
    outQueueC_.EnQue(out);
    LocalTensor<ToutC> tileOut = outQueueC_.DeQue<ToutC>();
    const uint32_t stepC = MatOutputStep(tiling_.orderC, tiling_.ldc);
    for (uint32_t colInTile = 0U; colInTile < blockCols; ++colInTile) {
        const uint32_t outCol = group * MT_TILE_COLS + colInTile;
        const uint64_t base = MatOutputColBase(tiling_.orderC, tiling_.ldc, outCol) +
                              static_cast<uint64_t>(rowBlock * logicalRows) * stepC;
        LocalTensor<ToutC> slot = tileOut[colInTile * MT_TILE_COLS];
        using DmaT = typename MatDmaByte<ToutC>::type;
        if (stepC == 1U) {
            DataCopyExtParams p{1U, blockRows * static_cast<uint32_t>(sizeof(ToutC)), 0, 0, 0};
            DataCopyPad(MatAsDmaBytes(cGm_)[base], MatAsDmaBytes(slot), p);
        } else {
            const int64_t dstStride = static_cast<int64_t>(stepC - 1U) * static_cast<int64_t>(sizeof(ToutC));
            DataCopyExtParams p{static_cast<uint16_t>(blockRows), static_cast<uint32_t>(sizeof(ToutC)), 0, dstStride, 0};
            DataCopyPad<DmaT, PaddingMode::Compact>(MatAsDmaBytes(cGm_)[base], MatAsDmaBytes(slot), p);
        }
    }
    outQueueC_.FreeTensor(tileOut);
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::ProcessGroupTile(
    uint32_t group, uint32_t rowBlock, uint32_t blockRows, uint32_t blockCols)
{
    const uint8_t pivotOrder =
        MatPickComplexOrderDev(tiling_.orderA, tiling_.orderB, tiling_.orderC, tiling_.hasB);
    const uint32_t tileRows = MatTileRowsDev(pivotOrder);
    const uint32_t rowBase = rowBlock * tileRows;
    // Compute over the full 32x32 column-major tile (fixed 32-element column stride). Padding slots
    // (stale data) only map to never-read padding output positions, so they need no zeroing.
    const uint32_t count = MT_MAX_GROUP_ELEMS;

    LocalTensor<TScale> accA = calcBufA_.Get<TScale>();
    if (MatIsComplexOrder(tiling_.orderA)) {
        GatherInputTile<TinA>(aGm_, tiling_.orderA, tiling_.lda, group, rowBase, accA);
    } else {
        LocalTensor<TinA> aIn = inQueueA_.AllocTensor<TinA>();
        LoadTileColumnMajor<TinA>(aGm_, aIn, tiling_.orderA, tiling_.opA, tiling_.lda, group, rowBase, blockRows,
                                  blockCols);
        inQueueA_.EnQue(aIn);
        LocalTensor<TinA> aLoaded = inQueueA_.DeQue<TinA>();
        MatCastToScale<TinA, TScale>(accA, aLoaded, count);
        inQueueA_.FreeTensor(aLoaded);
    }
    const TScale alpha = *reinterpret_cast<const TScale*>(&tiling_.alphaBits);
    Muls(accA, accA, alpha, count);

    if (tiling_.hasB != 0U) {
        LocalTensor<TScale> accB = calcBufB_.Get<TScale>();
        if (MatIsComplexOrder(tiling_.orderB)) {
            GatherInputTile<TinB>(bGm_, tiling_.orderB, tiling_.ldb, group, rowBase, accB);
        } else {
            LocalTensor<TinB> bIn = inQueueB_.AllocTensor<TinB>();
            LoadTileColumnMajor<TinB>(bGm_, bIn, tiling_.orderB, tiling_.opB, tiling_.ldb, group, rowBase, blockRows,
                                      blockCols);
            inQueueB_.EnQue(bIn);
            LocalTensor<TinB> bLoaded = inQueueB_.DeQue<TinB>();
            MatCastToScale<TinB, TScale>(accB, bLoaded, count);
            inQueueB_.FreeTensor(bLoaded);
        }
        const TScale beta = *reinterpret_cast<const TScale*>(&tiling_.betaBits);
        Muls(accB, accB, beta, count);
        Add(accA, accA, accB, count);
    }

    StoreOutputTile(group, rowBlock, blockRows, blockCols, accA);
}

// Iterate every output composite tile (column group x row block) assigned to this core.
template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformAIV<TinA, TinB, ToutC, TScale>::ProcessGroups()
{
    BuildTileOffsetTable();
    const uint8_t pivotOrder =
        MatPickComplexOrderDev(tiling_.orderA, tiling_.orderB, tiling_.orderC, tiling_.hasB);
    const uint32_t tileRows = MatTileRowsDev(pivotOrder);
    const uint32_t numGroups = (tiling_.cols + MT_TILE_COLS - 1U) / MT_TILE_COLS;
    const uint32_t numRowBlocks = (tiling_.rows + tileRows - 1U) / tileRows;
    const uint32_t totalTiles = numGroups * numRowBlocks;
    for (uint32_t t = blockIdx_; t < totalTiles; t += blockNum_) {
        const uint32_t group = t / numRowBlocks;
        const uint32_t rowBlock = t % numRowBlocks;
        uint32_t blockRows = tiling_.rows - rowBlock * tileRows;
        if (blockRows > tileRows) {
            blockRows = tileRows;
        }
        uint32_t blockCols = tiling_.cols - group * MT_TILE_COLS;
        if (blockCols > MT_TILE_COLS) {
            blockCols = MT_TILE_COLS;
        }
        ProcessGroupTile(group, rowBlock, blockRows, blockCols);
    }
}

template <typename TinA, typename TinB, typename ToutC, typename TScale>
__aicore__ inline void MatrixTransformLaunch(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR idxGm, GM_ADDR idxAGm, GM_ADDR idxBGm, GM_ADDR ndAGm,
    GM_ADDR ndBGm, const MatrixTransformTilingData& tiling)
{
    TPipe pipe;
    MatrixTransformAIV<TinA, TinB, ToutC, TScale> op;
    op.Init(aGm, bGm, cGm, idxGm, idxAGm, idxBGm, ndAGm, ndBGm, tiling, &pipe);
    op.Process();
}

// dtype codes shared with the Host router (matrix_transform_kernel_do):
//   0=FP32, 1=FP16, 2=BF16, 3=INT8, 4=INT32, 5=FP8_E4M3FN, 6=FP8_E5M2, 7=FP4_E2M1.
namespace {
constexpr uint32_t MatDtypeKey(uint8_t a, uint8_t b, uint8_t c)
{
    return (static_cast<uint32_t>(a) << 8U) | (static_cast<uint32_t>(b) << 4U) | static_cast<uint32_t>(c);
}
}  // namespace

// Single source of truth for the materialised dtype combinations. Each X(SUFFIX, TINA, TINB, TOUTC,
// TSCALE, CA, CB, CC) entry drives both the kernel instantiation and the router case, so adding a
// dtype combination is a one-line edit (no three-way drift). The three groups by scale path:
//   - float : A/B/C in {FP32, FP16, BF16}
//   - int32 : A/B/C in {INT8, INT32}
//   - float : FP8 {E4M3, E5M2} plus FP8<->float cross (fp8 moves as int8 bytes, casts in the scale
//             domain, reusing the linear / group / de-layout engine byte-for-byte like INT8)
//   - bf16  : FP4_E2M1 (packed fp4x2; dedicated unpack/transform/repack pipeline, single instance)
#define MT_DTYPE_LIST(X)                                                                                             \
    X(f32f32f32, float, float, float, float, MT_DTYPE_FP32, MT_DTYPE_FP32, MT_DTYPE_FP32)                            \
    X(f16f16f16, half, half, half, float, MT_DTYPE_FP16, MT_DTYPE_FP16, MT_DTYPE_FP16)                               \
    X(bf16bf16bf16, bfloat16_t, bfloat16_t, bfloat16_t, float, MT_DTYPE_BF16, MT_DTYPE_BF16, MT_DTYPE_BF16)          \
    X(f16f16f32, half, half, float, float, MT_DTYPE_FP16, MT_DTYPE_FP16, MT_DTYPE_FP32)                              \
    X(f32f32f16, float, float, half, float, MT_DTYPE_FP32, MT_DTYPE_FP32, MT_DTYPE_FP16)                             \
    X(bf16bf16f32, bfloat16_t, bfloat16_t, float, float, MT_DTYPE_BF16, MT_DTYPE_BF16, MT_DTYPE_FP32)                \
    X(f32f32bf16, float, float, bfloat16_t, float, MT_DTYPE_FP32, MT_DTYPE_FP32, MT_DTYPE_BF16)                      \
    X(bf16bf16f16, bfloat16_t, bfloat16_t, half, float, MT_DTYPE_BF16, MT_DTYPE_BF16, MT_DTYPE_FP16)                 \
    X(f16f16bf16, half, half, bfloat16_t, float, MT_DTYPE_FP16, MT_DTYPE_FP16, MT_DTYPE_BF16)                        \
    X(i32i32i32, int32_t, int32_t, int32_t, int32_t, MT_DTYPE_INT32, MT_DTYPE_INT32, MT_DTYPE_INT32)                 \
    X(i8i8i32, int8_t, int8_t, int32_t, int32_t, MT_DTYPE_INT8, MT_DTYPE_INT8, MT_DTYPE_INT32)                       \
    X(i32i32i8, int32_t, int32_t, int8_t, int32_t, MT_DTYPE_INT32, MT_DTYPE_INT32, MT_DTYPE_INT8)                    \
    X(i8i8i8, int8_t, int8_t, int8_t, int32_t, MT_DTYPE_INT8, MT_DTYPE_INT8, MT_DTYPE_INT8)                          \
    X(e4m3e4m3e4m3, MatFp8E4m3, MatFp8E4m3, MatFp8E4m3, float, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP8_E4M3) \
    X(e5m2e5m2e5m2, MatFp8E5m2, MatFp8E5m2, MatFp8E5m2, float, MT_DTYPE_FP8_E5M2, MT_DTYPE_FP8_E5M2, MT_DTYPE_FP8_E5M2) \
    X(e4m3e4m3e5m2, MatFp8E4m3, MatFp8E4m3, MatFp8E5m2, float, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP8_E5M2) \
    X(e5m2e5m2e4m3, MatFp8E5m2, MatFp8E5m2, MatFp8E4m3, float, MT_DTYPE_FP8_E5M2, MT_DTYPE_FP8_E5M2, MT_DTYPE_FP8_E4M3) \
    X(e4m3e4m3f16, MatFp8E4m3, MatFp8E4m3, half, float, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP16)           \
    X(e4m3e4m3bf16, MatFp8E4m3, MatFp8E4m3, bfloat16_t, float, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP8_E4M3, MT_DTYPE_BF16)    \
    X(e4m3e4m3f32, MatFp8E4m3, MatFp8E4m3, float, float, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP8_E4M3, MT_DTYPE_FP32)          \
    X(f32f32e4m3, float, float, MatFp8E4m3, float, MT_DTYPE_FP32, MT_DTYPE_FP32, MT_DTYPE_FP8_E4M3)                   \
    X(f16f16e4m3, half, half, MatFp8E4m3, float, MT_DTYPE_FP16, MT_DTYPE_FP16, MT_DTYPE_FP8_E4M3)                     \
    X(bf16bf16e4m3, bfloat16_t, bfloat16_t, MatFp8E4m3, float, MT_DTYPE_BF16, MT_DTYPE_BF16, MT_DTYPE_FP8_E4M3)       \
    X(e2m1e2m1e2m1, MatFp4x2, MatFp4x2, MatFp4x2, bfloat16_t, MT_DTYPE_FP4, MT_DTYPE_FP4, MT_DTYPE_FP4)

#define MT_DEFINE_KERNEL(SUFFIX, TINA, TINB, TOUTC, TSCALE, CA, CB, CC)                                         \
    __global__ __aicore__ void MatrixTransformKernel_##SUFFIX(                                                 \
        GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR idxGm, GM_ADDR idxAGm, GM_ADDR idxBGm, GM_ADDR ndAGm,   \
        GM_ADDR ndBGm, const MatrixTransformTilingData tiling)                                                 \
    {                                                                                                         \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                       \
        MatrixTransformLaunch<TINA, TINB, TOUTC, TSCALE>(                                                      \
            aGm, bGm, cGm, idxGm, idxAGm, idxBGm, ndAGm, ndBGm, tiling);                                       \
    }
MT_DTYPE_LIST(MT_DEFINE_KERNEL)
#undef MT_DEFINE_KERNEL

#define MT_LAUNCH(SUFFIX)                                                                       \
    MatrixTransformKernel_##SUFFIX<<<numBlocks, nullptr, stream>>>(                              \
        aGm, bGm, cGm, idxGm, idxAGm, idxBGm, ndAGm, ndBGm, tiling)

void matrix_transform_kernel_do(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR idxGm, GM_ADDR idxAGm, GM_ADDR idxBGm, GM_ADDR ndAGm,
    GM_ADDR ndBGm, uint8_t dtypeA, uint8_t dtypeB, uint8_t dtypeC, const MatrixTransformTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    const uint32_t key = MatDtypeKey(dtypeA, dtypeB, dtypeC);
    switch (key) {
#define MT_CASE(SUFFIX, TINA, TINB, TOUTC, TSCALE, CA, CB, CC) \
    case MatDtypeKey(CA, CB, CC): MT_LAUNCH(SUFFIX); break;
        MT_DTYPE_LIST(MT_CASE)
#undef MT_CASE
        default: break;
    }
}

#undef MT_LAUNCH
#undef MT_DTYPE_LIST
