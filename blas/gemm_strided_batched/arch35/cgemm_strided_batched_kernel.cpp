/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cgemm_strided_batched_cube.h"
#include "cgemm_strided_batched_vector.h"

// Vector packing primitives and compensated-input preparation.
namespace {

namespace {

// Real, imaginary and sum planes each have FP16 high/residual parts.

// Partial combination reads two product planes and the previous complex output.

template <class Register, class Mask>
__simd_callee__ inline void PackComponent(
    Register& value, __ubuf__ half* dst, uint32_t offset, uint32_t lowOffset, Mask mask)
{
    namespace M = AscendC::MicroAPI;
    static constexpr M::CastTrait down{
        M::RegLayout::ZERO, M::SatMode::NO_SAT, M::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr M::CastTrait up{
        M::RegLayout::ZERO, M::SatMode::UNKNOWN, M::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    M::RegTensor<half> high, low;
    M::RegTensor<float> rounded, residual;
    M::Cast<half, float, down>(high, value, mask);
    M::Cast<float, half, up>(rounded, high, mask);
    M::Sub(residual, value, rounded, mask);
    M::Cast<half, float, down>(low, residual, mask);
    M::DataCopy<half, M::StoreDist::DIST_PACK_B32>(dst + offset, high, mask);
    M::DataCopy<half, M::StoreDist::DIST_PACK_B32>(dst + offset + lowOffset, low, mask);
}
__simd_vf__ inline void MicroPackHalf(
    __ubuf__ float* src, __ubuf__ half* dst, __ubuf__ int32_t* maximum, uint32_t count, bool conjugate, uint32_t stride)
{
    namespace M = AscendC::MicroAPI;
    M::RegTensor<int32_t> maxReg;
    M::DataCopy(maxReg, maximum);
    M::RegTensor<uint32_t> mag, bits;
    auto all = M::CreateMask<uint32_t, M::MaskPattern::ALL>();
    M::Duplicate(mag, uint32_t(CGEMM_SB_FP32_ABS_MASK), all);
    for (uint16_t i = 0; i < (count + CGEMM_SB_FP32_LANES - 1) / CGEMM_SB_FP32_LANES; ++i) {
        uint32_t len = count - uint32_t(i) * CGEMM_SB_FP32_LANES;
        auto mask = M::UpdateMask<float>(len);
        M::RegTensor<float> real, imag, sum;
        M::DataCopy<float, M::PostLiteral::POST_MODE_UPDATE, M::LoadDist::DIST_DINTLV_B32>(
            real, imag, src, CGEMM_SB_COMPLEX_REGISTER_FLOATS);
        M::And(bits, reinterpret_cast<M::RegTensor<uint32_t>&>(real), mag, mask);
        M::Max<int32_t, M::MaskMergeMode::MERGING>(
            maxReg, maxReg, reinterpret_cast<M::RegTensor<int32_t>&>(bits), mask);
        M::And(bits, reinterpret_cast<M::RegTensor<uint32_t>&>(imag), mag, mask);
        M::Max<int32_t, M::MaskMergeMode::MERGING>(
            maxReg, maxReg, reinterpret_cast<M::RegTensor<int32_t>&>(bits), mask);
        if (conjugate)
            M::Muls(imag, imag, -1.0f, mask);
        M::Add(sum, real, imag, mask);
        PackComponent(real, dst, uint32_t(i) * CGEMM_SB_FP32_LANES, CGEMM_SB_THREE_M_PLANES * stride, mask);
        PackComponent(imag, dst, stride + uint32_t(i) * CGEMM_SB_FP32_LANES, CGEMM_SB_THREE_M_PLANES * stride, mask);
        PackComponent(sum, dst, 2 * stride + uint32_t(i) * CGEMM_SB_FP32_LANES, CGEMM_SB_THREE_M_PLANES * stride, mask);
    }
    M::DataCopy(maximum, maxReg, all);
}
template <uint32_t STRIDE>
__simd_vf__ inline void MicroCombineUnit(__ubuf__ float* src, __ubuf__ float* dst, uint32_t count)
{
    namespace M = AscendC::MicroAPI;
    for (uint16_t i = 0; i < (count + CGEMM_SB_FP32_LANES - 1) / CGEMM_SB_FP32_LANES; ++i) {
        uint32_t len = count - uint32_t(i) * CGEMM_SB_FP32_LANES;
        auto mask = M::UpdateMask<float>(len);
        M::RegTensor<float> rr, ii, sum, real, imag;
        M::DataCopy(rr, src + uint32_t(i) * CGEMM_SB_FP32_LANES);
        M::DataCopy(ii, src + STRIDE + uint32_t(i) * CGEMM_SB_FP32_LANES);
        M::DataCopy(sum, src + 2 * STRIDE + uint32_t(i) * CGEMM_SB_FP32_LANES);
        M::Sub(real, rr, ii, mask);
        M::Sub(imag, sum, rr, mask);
        M::Sub(imag, imag, ii, mask);
        M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(
            dst + uint32_t(i) * CGEMM_SB_COMPLEX_REGISTER_FLOATS, real, imag, mask);
    }
}
__aicore__ inline uint32_t ScalarMin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Tile in physical column order. Several short columns share one DMA.

__aicore__ inline uint64_t TileCount(uint32_t rows, uint32_t cols, uint32_t capacity)
{
    if (capacity == 0 || rows == 0 || cols == 0)
        return 0;
    uint32_t rowBlocks = (rows - 1) / capacity + 1;
    uint32_t columnGroup = rows <= capacity ? capacity / rows : 1;
    return uint64_t(rowBlocks) * ((cols + columnGroup - 1) / columnGroup);
}

__aicore__ inline ColumnTile DecodeTile(
    uint64_t index, uint32_t paddedRows, uint32_t paddedCols, uint32_t logicalRows, uint32_t logicalCols,
    uint32_t capacity)
{
    if (capacity == 0 || paddedRows == 0 || paddedCols == 0)
        return {};
    uint32_t rowBlocks = (paddedRows - 1) / capacity + 1;
    uint32_t columnGroup = paddedRows <= capacity ? capacity / paddedRows : 1;
    ColumnTile v{};
    v.row = (index % rowBlocks) * capacity;
    v.col = (index / rowBlocks) * columnGroup;
    v.rows = ScalarMin(capacity, paddedRows - v.row);
    v.cols = ScalarMin(columnGroup, paddedCols - v.col);
    v.validRows = v.row < logicalRows ? ScalarMin(v.rows, logicalRows - v.row) : 0;
    v.validCols = v.col < logicalCols ? ScalarMin(v.cols, logicalCols - v.col) : 0;
    return v;
}

__aicore__ inline void ReadColumns(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, const ColumnTile& v, uint32_t ld)
{
    // Initialize explicit matrix padding, including completely padded columns.
    AscendC::Duplicate(dst, 0.0f, v.rows * v.cols * 2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(PADDING_READY_EVENT);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(PADDING_READY_EVENT);
    if (v.validRows == 0 || v.validCols == 0)
        return;
    uint32_t alignedFloats =
        ((v.validRows * 2 + CGEMM_SB_FP32_PER_BLOCK - 1) / CGEMM_SB_FP32_PER_BLOCK) * CGEMM_SB_FP32_PER_BLOCK;
    AscendC::DataCopyExtParams cp{
        static_cast<uint16_t>(v.validCols), v.validRows * CGEMM_SB_COMPLEX_BYTES,
        (ld - v.validRows) * CGEMM_SB_COMPLEX_BYTES, (v.rows * 2 - alignedFloats) / CGEMM_SB_FP32_PER_BLOCK, 0};
    AscendC::DataCopyPadExtParams<float> pp{true, 0, static_cast<uint8_t>(alignedFloats - v.validRows * 2), 0};
    AscendC::DataCopyPad(dst, src, cp, pp);
}

__aicore__ inline PackGeometry MakePackGeometry(const CgemmSbTiling& t, bool isA, uint32_t capacity)
{
    PackGeometry g{};
    if (capacity == 0)
        return g;
    bool normal = (isA ? t.transA : t.transB) == ACLBLAS_OP_N;
    g.rows = isA ? t.m : t.k;
    g.cols = isA ? t.k : t.n;
    if (!normal) {
        uint32_t rows = g.rows;
        g.rows = g.cols;
        g.cols = rows;
    }
    g.paddedRows = isA ? (normal ? t.mp : t.kp) : (normal ? t.kp : t.np);
    g.paddedCols = isA ? (normal ? t.kp : t.mp) : (normal ? t.np : t.kp);
    g.ld = isA ? t.lda : t.ldb;
    g.stride = isA ? t.strideA : t.strideB;
    g.capacity = capacity;
    g.dense = t.densePack;
    g.matrix = uint64_t(g.paddedRows) * g.paddedCols;
    g.flat = g.dense && g.stride == int64_t(g.matrix);
    uint32_t batches = isA ? t.batchesA : t.batchesB;
    g.elements = g.matrix * (g.flat ? batches : 1);
    g.tiles = g.dense ? (g.elements + capacity - 1) / capacity : TileCount(g.paddedRows, g.paddedCols, capacity);
    g.total = g.tiles * (g.flat ? 1 : batches);
    return g;
}

__aicore__ inline void ReadPackTile(
    AscendC::LocalTensor<float> x, AscendC::GlobalTensor<float> source, const PackGeometry& g, uint64_t index,
    uint64_t& batch, uint64_t& offset, uint32_t& count)
{
    batch = index / g.tiles;
    offset = index % g.tiles * g.capacity;
    uint64_t remaining = (g.flat ? g.elements : g.matrix) - offset;
    count = remaining < g.capacity ? remaining : g.capacity;
    if (g.flat) {
        batch = offset / g.matrix;
        offset %= g.matrix;
    }
    int64_t src = int64_t(batch) * g.stride;
    if (g.dense) {
        AscendC::DataCopy(x, source[2 * (src + offset)], count * 2);
        return;
    }
    auto v = DecodeTile(index % g.tiles, g.paddedRows, g.paddedCols, g.rows, g.cols, g.capacity);
    count = v.rows * v.cols;
    offset = uint64_t(v.col) * g.paddedRows + v.row;
    src += uint64_t(v.col) * g.ld + v.row;
    ReadColumns(x, source[src * 2], v, g.ld);
}

template <uint32_t SIZE>
__aicore__ inline void FastHalfPacker<SIZE>::Run(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, CgemmSbTiling t, uint32_t core, uint32_t cores, AscendC::TPipe& pipe)
{
    pipe.InitBuffer(input, 1, 2 * SIZE * CGEMM_SB_FP32_BYTES);
    pipe.InitBuffer(output, 1, HALF_PACK_PLANES * SIZE * CGEMM_SB_FP16_BYTES);
    pipe.InitBuffer(flagq, 1, CGEMM_SB_DMA_BLOCK_BYTES);
    pipe.InitBuffer(scratch, CGEMM_SB_MAX_SCRATCH_BYTES);
    auto maxbuf = scratch.Get<int32_t>();
    AscendC::Duplicate(maxbuf, int32_t(0), CGEMM_SB_FP32_LANES);
    AscendC::GlobalTensor<float> ag;
    AscendC::GlobalTensor<float> bg;
    AscendC::GlobalTensor<float> wg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    auto ga = MakePackGeometry(t, true, SIZE);
    auto gb = MakePackGeometry(t, false, SIZE);
    for (uint64_t task = core; task < ga.total + gb.total; task += cores) {
        bool isA = task < ga.total;
        auto geometry = isA ? ga : gb;
        uint64_t batch = 0;
        uint64_t off = 0;
        uint32_t count = 0;
        auto x = input.AllocTensor<float>();
        ReadPackTile(x, isA ? ag : bg, geometry, isA ? task : task - ga.total, batch, off, count);
        input.EnQue(x);
        x = input.DeQue<float>();
        auto y = output.AllocTensor<half>();
        asc_vf_call<MicroPackHalf>(
            reinterpret_cast<__ubuf__ float*>(x.GetPhyAddr()), reinterpret_cast<__ubuf__ half*>(y.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(maxbuf.GetPhyAddr()), count,
            (isA ? t.transA : t.transB) == ACLBLAS_OP_C, SIZE);
        output.EnQue(y);
        y = output.DeQue<half>();
        AscendC::GlobalTensor<half> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wg.GetPhyAddr(isA ? 0 : t.bOffset)));
        AscendC::DataCopyExtParams copyPlanes{
            HALF_PACK_PLANES, count * CGEMM_SB_FP16_BYTES, (SIZE - count) / CGEMM_SB_FP16_PER_BLOCK,
            static_cast<uint32_t>((isA ? t.aSize : t.bSize) * CGEMM_SB_FP32_BYTES - count * CGEMM_SB_FP16_BYTES), 0};
        AscendC::DataCopyPad(dst[batch * geometry.matrix + off], y, copyPlanes);
        output.FreeTensor(y);
        input.FreeTensor(x);
    }
    PublishFlag(wg, t, core);
}

template <uint32_t SIZE>
__aicore__ inline void FastHalfPacker<SIZE>::PublishFlag(
    AscendC::GlobalTensor<float> wg, const CgemmSbTiling& t, uint32_t core)
{
    auto maxbuf = scratch.Get<int32_t>();
    AscendC::ReduceMax(
        maxbuf[CGEMM_SB_MAX_RESULT_OFFSET], maxbuf, maxbuf[CGEMM_SB_MAX_TEMP_OFFSET], CGEMM_SB_FP32_LANES);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    auto flag = flagq.AllocTensor<float>();
    AscendC::Duplicate(
        flag, maxbuf.GetValue(CGEMM_SB_MAX_RESULT_OFFSET) >= CGEMM_SB_HALF_FALLBACK_BITS ? 1.0f : 0.0f,
        CGEMM_SB_FLAG_STRIDE);
    flagq.EnQue(flag);
    flag = flagq.DeQue<float>();
    AscendC::DataCopy(wg[t.flagOffset + uint64_t(core) * CGEMM_SB_FLAG_STRIDE], flag, CGEMM_SB_FLAG_STRIDE);
    flagq.FreeTensor(flag);
}

template <uint32_t SIZE>
__simd_vf__ inline void MicroMergeComponent(
    __ubuf__ float* src, __ubuf__ float* dst, uint32_t count, uint32_t component)
{
    namespace M = AscendC::MicroAPI;
    auto mask = M::CreateMask<float, M::MaskPattern::ALL>();
    for (uint16_t off = 0; off < count; off += CGEMM_SB_FP32_LANES) {
        M::RegTensor<float> first, second, real, imag;
        M::DataCopy(first, src + off);
        M::DataCopy(second, src + SIZE + off);
        auto previous = src + CGEMM_SB_PARTIAL_PRODUCTS * SIZE + 2 * off;
        M::DataCopy<float, M::PostLiteral::POST_MODE_UPDATE, M::LoadDist::DIST_DINTLV_B32>(
            real, imag, previous, CGEMM_SB_COMPLEX_REGISTER_FLOATS);
        if (component == CGEMM_SB_REFERENCE_IMAG)
            M::Sub(real, first, second, mask);
        else
            M::Add(imag, first, second, mask);
        M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(dst + 2 * off, real, imag, mask);
    }
}

template <uint32_t SIZE, bool Partial, uint32_t OUTPUT_SLOTS>
__aicore__ inline void FastUnitCombiner<SIZE, Partial, OUTPUT_SLOTS>::Run(
    GM_ADDR c, GM_ADDR workspace, CgemmSbTiling t, uint32_t core, uint32_t cores, AscendC::TPipe& pipe,
    uint32_t component)
{
    pipe.InitBuffer(input, 1, (Partial ? PARTIAL_INPUT_PLANES : CGEMM_SB_THREE_M_PLANES) * SIZE * sizeof(float));
    pipe.InitBuffer(output, OUTPUT_SLOTS, 2 * SIZE * sizeof(float));
    AscendC::GlobalTensor<float> wg;
    AscendC::GlobalTensor<float> cg;
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    uint64_t matrix = uint64_t(t.mp) * t.np;
    bool flat = t.denseCombine && t.strideC == int64_t(matrix);
    uint32_t capacity = SIZE / t.mp;
    uint32_t columnTiles = (t.n + capacity - 1) / capacity;
    uint32_t columns = (t.n + columnTiles - 1) / columnTiles;
    uint64_t tiles = flat ? (matrix * t.batch + SIZE - 1) / SIZE : columnTiles;
    uint64_t total = tiles * (flat ? 1 : t.batch);
    for (uint64_t task = core; task < total; task += cores) {
        uint64_t batch = flat ? 0 : task / tiles;
        uint64_t off = flat ? task * SIZE : uint64_t(task % tiles) * columns * t.mp;
        uint32_t col = flat ? 0 : uint32_t(task % tiles) * columns;
        uint32_t cols = flat ? 0 : ScalarMin(columns, t.n - col);
        uint32_t count = flat ? uint32_t(matrix * t.batch - off < SIZE ? matrix * t.batch - off : SIZE) : t.mp * cols;
        uint64_t source = t.cOffset + batch * matrix + off;
        int64_t dest = flat ? int64_t(off) * 2 : (int64_t(batch) * t.strideC + int64_t(col) * t.ldc) * 2;
        auto x = input.AllocTensor<float>();
        AscendC::DataCopyExtParams read{
            Partial ? CGEMM_SB_PARTIAL_PRODUCTS : CGEMM_SB_THREE_M_PLANES, count * CGEMM_SB_FP32_BYTES,
            static_cast<uint32_t>((t.cSize - count) * CGEMM_SB_FP32_BYTES), (SIZE - count) / CGEMM_SB_FP32_PER_BLOCK,
            0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        AscendC::DataCopyPad(x, wg[source], read, noPad);
        if constexpr (Partial) {
            if (flat) {
                AscendC::DataCopy(x[CGEMM_SB_PARTIAL_PRODUCTS * SIZE], cg[dest], count * 2);
            } else {
                ReadCurrentColumns(x[CGEMM_SB_PARTIAL_PRODUCTS * SIZE], cg[dest], t, cols);
            }
        }
        input.EnQue(x);
        x = input.DeQue<float>();
        auto y = output.template AllocTensor<float>();
        ComputeTile(x, y, count, component);
        output.EnQue(y);
        y = output.template DeQue<float>();
        StoreTile(cg, y, t, flat, dest, count, cols);
        output.FreeTensor(y);
        input.FreeTensor(x);
    }
}

template <uint32_t SIZE, bool Partial, uint32_t OUTPUT_SLOTS>
__aicore__ inline void FastUnitCombiner<SIZE, Partial, OUTPUT_SLOTS>::ComputeTile(
    AscendC::LocalTensor<float> x, AscendC::LocalTensor<float> y, uint32_t count, uint32_t component)
{
    if constexpr (Partial) {
        asc_vf_call<MicroMergeComponent<SIZE>>(
            reinterpret_cast<__ubuf__ float*>(x.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(y.GetPhyAddr()), count,
            component);
    } else
        asc_vf_call<MicroCombineUnit<SIZE>>(
            reinterpret_cast<__ubuf__ float*>(x.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(y.GetPhyAddr()),
            count);
}

template <uint32_t SIZE, bool Partial, uint32_t OUTPUT_SLOTS>
__aicore__ inline void FastUnitCombiner<SIZE, Partial, OUTPUT_SLOTS>::ReadCurrentColumns(
    AscendC::LocalTensor<float> x, AscendC::GlobalTensor<float> c, const CgemmSbTiling& t, uint32_t cols)
{
    uint32_t aligned = (t.m * 2 + CGEMM_SB_FP32_PER_BLOCK - 1) / CGEMM_SB_FP32_PER_BLOCK * CGEMM_SB_FP32_PER_BLOCK;
    AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(cols), t.m * CGEMM_SB_COMPLEX_BYTES, (t.ldc - t.m) * CGEMM_SB_COMPLEX_BYTES,
        (t.mp * 2 - aligned) / CGEMM_SB_FP32_PER_BLOCK, 0};
    AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
    AscendC::DataCopyPad(x, c, copy, noPad);
}

template <uint32_t SIZE, bool Partial, uint32_t OUTPUT_SLOTS>
__aicore__ inline void FastUnitCombiner<SIZE, Partial, OUTPUT_SLOTS>::StoreTile(
    AscendC::GlobalTensor<float> cg, AscendC::LocalTensor<float> y, const CgemmSbTiling& t, bool flat, int64_t dest,
    uint32_t count, uint32_t cols)
{
    if (flat)
        AscendC::DataCopy(cg[dest], y, count * 2);
    else {
        uint32_t aligned = (t.m * 2 + CGEMM_SB_FP32_PER_BLOCK - 1) / CGEMM_SB_FP32_PER_BLOCK * CGEMM_SB_FP32_PER_BLOCK;
        AscendC::DataCopyExtParams write{
            static_cast<uint16_t>(cols), t.m * CGEMM_SB_COMPLEX_BYTES, (t.mp * 2 - aligned) / CGEMM_SB_FP32_PER_BLOCK,
            (t.ldc - t.m) * CGEMM_SB_COMPLEX_BYTES, 0};
        AscendC::DataCopyPad(cg[dest], y, write);
    }
}

} // namespace
} // namespace

// Vector packing, product combination and output writeback.
namespace {

namespace {
__aicore__ inline void CgemmStridedBatchedVectorImpl::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t, bool pack, bool mixed, AscendC::TPipe& pipe)
{
    t_ = t;
    core_ = AscendC::GetBlockIdx();
    cores_ = AscendC::GetBlockNum() * (mixed ? 2 : 1);
    a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    b_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    c_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    ws_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
    if (!pack && t_.threeM) {
        pipe.InitBuffer(input_, 2, CGEMM_SB_THREE_M_PLANES * TILE * sizeof(float));
        pipe.InitBuffer(output_, 2, 2 * TILE * sizeof(float));
        return;
    }
    pipe.InitBuffer(input_, 2, (pack ? CGEMM_SB_COMPONENT_PLANES : CGEMM_SB_GENERAL_PRODUCTS) * TILE * sizeof(float));
    pipe.InitBuffer(
        output_, pack ? 1 : 2,
        (pack && t_.compensatedHalf ? CGEMM_SB_HALF_PLANES : CGEMM_SB_THREE_M_PLANES) * TILE * sizeof(float));
    if (pack && t_.compensatedHalf)
        pipe.InitBuffer(halfOutput_, 1, CGEMM_SB_HALF_PLANES * TILE * sizeof(half));
    if (pack && t_.threeM && t_.k >= CGEMM_SB_ORDERED_MIN_K) {
        pipe.InitBuffer(moments_, CGEMM_SB_MOMENT_PLANES * TILE * sizeof(float));
        AscendC::Duplicate(moments_.Get<float>(), 0.0f, CGEMM_SB_MOMENT_PLANES * TILE);
    }
    if (!pack)
        pipe.InitBuffer(old_, 2, 2 * TILE * sizeof(float));
    pipe.InitBuffer(calc_, VECTOR_CALC_TILE_SLOTS * TILE * sizeof(float));
    pipe.InitBuffer(indices_, 2 * TILE * sizeof(uint32_t));
    even_ = indices_.Get<uint32_t>();
    odd_ = even_[TILE];
    auto evenInt = even_.ReinterpretCast<int32_t>();
    auto oddInt = odd_.ReinterpretCast<int32_t>();
    CreateVecIndex(evenInt, int32_t(0), TILE);
    ShiftLeft(even_, even_, uint32_t(CGEMM_SB_COMPLEX_BYTE_SHIFT), TILE);
    Adds(oddInt, evenInt, int32_t(CGEMM_SB_FP32_BYTES), TILE);
    AscendC::PipeBarrier<PIPE_ALL>();
    if (pack) {
        auto running = calc_.Get<int32_t>();
        auto mask = calc_.Get<uint32_t>()[VECTOR_PACK_MASK_SLOT * TILE];
        AscendC::Duplicate(running, int32_t(0), TILE * 2);
        AscendC::Duplicate(mask, uint32_t(CGEMM_SB_FP32_ABS_MASK), TILE * 2);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::Pack()
{
    auto ga = MakePackGeometry(t_, true, TILE);
    auto gb = MakePackGeometry(t_, false, TILE / t_.batchGroup);
    for (uint64_t task = core_; task < ga.total + gb.total; task += cores_) {
        bool isA = task < ga.total;
        auto geometry = isA ? ga : gb;
        uint64_t batch = 0;
        uint64_t offset = 0;
        uint64_t matrix = geometry.matrix;
        uint32_t count = 0;
        auto data = input_.AllocTensor<float>();
        ReadPackTile(data, isA ? a_ : b_, geometry, isA ? task : task - ga.total, batch, offset, count);
        input_.EnQue(data);
        data = input_.DeQue<float>();
        if (t_.compensatedHalf && t_.k < SHORT_HALF_K_LIMIT) {
            PackShortHalf(data, isA, batch, matrix, offset, count);
            input_.FreeTensor(data);
            continue;
        }
        auto out = output_.AllocTensor<float>();
        MakePlanes(data, out, isA, count);
        if (!isA && t_.batchGroup > 1) {
            MergeBPlanes(out, batch, count);
            count *= t_.batchGroup;
            matrix *= t_.batchGroup;
            offset *= t_.batchGroup;
        }
        if (t_.compensatedHalf) {
            StoreHalfPlanes(out, isA, batch, matrix, offset, count);
            input_.FreeTensor(data);
            output_.FreeTensor(out);
            continue;
        }
        output_.EnQue(out);
        out = output_.DeQue<float>();
        uint64_t dest = (isA ? 0 : t_.bOffset) + batch * matrix + offset;
        AscendC::DataCopy(ws_[dest], out, count);
        AscendC::DataCopy(ws_[dest + (isA ? t_.aSize : t_.bSize)], out[TILE], count);
        if (t_.threeM)
            AscendC::DataCopy(
                ws_[dest + CGEMM_SB_PLANE_SUM * (isA ? t_.aSize : t_.bSize)], out[CGEMM_SB_PLANE_SUM * TILE], count);
        input_.FreeTensor(data);
        output_.FreeTensor(out);
    }
    PublishMetadata();
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::Combine()
{
    uint64_t matrix = uint64_t(t_.mp) * t_.np;
    bool flat = t_.denseCombine && t_.strideC == int64_t(matrix);
    uint64_t elements = matrix * (flat ? t_.batch : 1);
    uint64_t tiles = t_.denseCombine ? (elements + TILE - 1) / TILE : TileCount(t_.mp, t_.n);
    for (uint64_t task = core_; task < tiles * (flat ? 1 : t_.batch); task += cores_) {
        uint64_t batch = task / tiles;
        uint64_t offset = (task % tiles) * TILE;
        uint32_t count = static_cast<uint32_t>(elements - offset < TILE ? elements - offset : TILE);
        if (flat) {
            batch = offset / matrix;
            offset %= matrix;
        }
        ColumnTile view{};
        if (!t_.denseCombine) {
            view = DecodeTile(task % tiles, t_.mp, t_.n, t_.m, t_.n);
            count = view.rows * view.cols;
            offset = uint64_t(view.col) * t_.mp + view.row;
        }
        uint64_t source = t_.cOffset + batch * matrix + offset;
        int64_t dest =
            (int64_t(batch) * t_.strideC + (t_.denseCombine ? offset : uint64_t(view.col) * t_.ldc + view.row)) * 2;
        auto data = input_.AllocTensor<float>();

        for (uint32_t p = 0; p < (t_.threeM ? CGEMM_SB_THREE_M_PLANES : CGEMM_SB_GENERAL_PRODUCTS); ++p)
            AscendC::DataCopy(data[p * TILE], ws_[source + p * t_.cSize], count);

        input_.EnQue(data);
        data = input_.DeQue<float>();
        auto out = output_.AllocTensor<float>();
        if (t_.threeM) {
            asc_vf_call<MicroCombineUnit<TILE>>(
                reinterpret_cast<__ubuf__ float*>(data.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(out.GetPhyAddr()), count);
        } else {
            CombineGeneral(data, out, view, dest, count);
        }
        output_.EnQue(out);
        out = output_.DeQue<float>();
        StoreCombinedTile(out, view, dest, count);
        input_.FreeTensor(data);
        output_.FreeTensor(out);
    }
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::StoreCombinedTile(
    AscendC::LocalTensor<float> out, const ColumnTile& view, int64_t dest, uint32_t count)
{
    if (t_.denseCombine)
        AscendC::DataCopy(c_[dest], out, count * 2);
    else {
        uint32_t alignedFloats =
            (view.validRows * 2 + CGEMM_SB_FP32_PER_BLOCK - 1) / CGEMM_SB_FP32_PER_BLOCK * CGEMM_SB_FP32_PER_BLOCK;
        AscendC::DataCopyExtParams cp{
            static_cast<uint16_t>(view.validCols), view.validRows * CGEMM_SB_COMPLEX_BYTES,
            (view.rows * 2 - alignedFloats) / CGEMM_SB_FP32_PER_BLOCK,
            (t_.ldc - view.validRows) * CGEMM_SB_COMPLEX_BYTES, 0};
        AscendC::DataCopyPad(c_[dest], out, cp);
    }
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::PackShortHalf(
    AscendC::LocalTensor<float> data, bool isA, uint64_t batch, uint64_t matrix, uint64_t offset, uint32_t count)
{
    auto packed = halfOutput_.AllocTensor<half>();
    asc_vf_call<MicroPackHalf>(
        reinterpret_cast<__ubuf__ float*>(data.GetPhyAddr()), reinterpret_cast<__ubuf__ half*>(packed.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(calc_.Get<int32_t>().GetPhyAddr()), count,
        (isA ? t_.transA : t_.transB) == ACLBLAS_OP_C, TILE);
    halfOutput_.EnQue(packed);
    packed = halfOutput_.DeQue<half>();
    for (uint32_t p = 0; p < CGEMM_SB_HALF_PLANES; ++p) {
        uint64_t base = (isA ? 0 : t_.bOffset) + p * (isA ? t_.aSize : t_.bSize);
        AscendC::GlobalTensor<half> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ws_.GetPhyAddr(base)));
        AscendC::DataCopy(target[batch * matrix + offset], packed[p * TILE], count);
    }
    halfOutput_.FreeTensor(packed);
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::MakePlanes(
    AscendC::LocalTensor<float> data, AscendC::LocalTensor<float> out, bool isA, uint32_t count)
{
    auto real = out;
    auto imag = out[TILE];
    Gather(real, data, even_, 0U, count);
    Gather(imag, data, odd_, 0U, count);
    if ((isA ? t_.transA : t_.transB) == ACLBLAS_OP_C)
        Muls(imag, imag, -1.0f, count);
    if (t_.threeM && t_.k >= CGEMM_SB_ORDERED_MIN_K) {
        auto sums = moments_.Get<float>()[(isA ? 0 : CGEMM_SB_COMPONENT_PLANES) * TILE];
        Add(sums, sums, real, count);
        Add(sums[TILE], sums[TILE], imag, count);
        if (isA)
            countA_ += count;
        else
            countB_ += count;
    }
    if (t_.threeM) {
        auto sum = out[CGEMM_SB_PLANE_SUM * TILE];
        Add(sum, real, imag, count);
    }
    {
        auto scratch = calc_.Get<uint32_t>()[VECTOR_PACK_ABS_SLOT * TILE];
        auto mask = calc_.Get<uint32_t>()[VECTOR_PACK_MASK_SLOT * TILE];
        auto sourceBits = data.ReinterpretCast<uint32_t>();
        AscendC::Duplicate(mask, uint32_t(CGEMM_SB_FP32_ABS_MASK), count * 2);
        And(scratch, sourceBits, mask, count * 2);
        auto absInts = scratch.ReinterpretCast<int32_t>();
        auto running = calc_.Get<int32_t>();
        Max(running, running, absInts, count * 2);
    }
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::MergeBPlanes(
    AscendC::LocalTensor<float> out, uint64_t batch, uint32_t count)
{
    // The B columns contain one nonzero diagonal block. Stacked A
    // batches then produce stacked C batches in one Cube tile.
    auto pos = calc_.Get<uint32_t>()[VECTOR_MERGE_POSITION_SLOT * TILE];
    auto cols = calc_.Get<uint32_t>()[VECTOR_MERGE_COLUMN_SLOT * TILE];
    CreateVecIndex(pos.ReinterpretCast<int32_t>(), int32_t(0), count);
    ShiftRight(
        cols, pos, uint32_t(t_.kp == CGEMM_SB_FRACTAL_EDGE ? MERGE_SMALL_DIM_SHIFT : MERGE_LARGE_DIM_SHIFT), count);
    Muls(cols.ReinterpretCast<int32_t>(), cols.ReinterpretCast<int32_t>(), int32_t((t_.batchGroup - 1) * t_.kp), count);
    Add(pos.ReinterpretCast<int32_t>(), pos.ReinterpretCast<int32_t>(), cols.ReinterpretCast<int32_t>(), count);
    auto batchIndex = calc_.Get<uint32_t>()[VECTOR_MERGE_BATCH_SLOT * TILE];
    CreateVecIndex(batchIndex.ReinterpretCast<int32_t>(), int32_t(0), count);
    uint32_t matrixShift = (t_.kp == CGEMM_SB_FRACTAL_EDGE ? MERGE_SMALL_DIM_SHIFT : MERGE_LARGE_DIM_SHIFT) +
                           (t_.np == CGEMM_SB_FRACTAL_EDGE ? MERGE_SMALL_DIM_SHIFT : MERGE_LARGE_DIM_SHIFT);
    ShiftRight(batchIndex, batchIndex, matrixShift, count);
    Adds(
        batchIndex.ReinterpretCast<int32_t>(), batchIndex.ReinterpretCast<int32_t>(), int32_t(batch % t_.batchGroup),
        count);
    uint32_t modShift = t_.batchGroup == MERGE_FOUR_BATCH_GROUP ? MERGE_FOUR_BATCH_SHIFT : MERGE_TWO_BATCH_SHIFT;
    ShiftLeft(batchIndex, batchIndex, modShift, count);
    ShiftRight(batchIndex, batchIndex, modShift, count);
    ShiftLeft(
        batchIndex, batchIndex,
        uint32_t(t_.kp == CGEMM_SB_FRACTAL_EDGE ? MERGE_SMALL_DIM_SHIFT : MERGE_LARGE_DIM_SHIFT), count);
    Add(pos.ReinterpretCast<int32_t>(), pos.ReinterpretCast<int32_t>(), batchIndex.ReinterpretCast<int32_t>(), count);
    ShiftLeft(pos, pos, uint32_t(CGEMM_SB_FP32_BYTE_SHIFT), count);
    auto values = calc_.Get<float>()[VECTOR_MERGE_VALUES_SLOT * TILE];
    for (uint32_t p = 0; p < CGEMM_SB_THREE_M_PLANES; ++p) {
        AscendC::DataCopy(values, out[p * TILE], count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(out[p * TILE], 0.0f, count * t_.batchGroup);
        Scatter(out[p * TILE], values, pos, 0U, count);
    }
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::StoreHalfPlanes(
    AscendC::LocalTensor<float> out, bool isA, uint64_t batch, uint64_t matrix, uint64_t offset, uint32_t count)
{
    auto packed = halfOutput_.AllocTensor<half>();
    for (uint32_t p = 0; p < CGEMM_SB_THREE_M_PLANES; ++p) {
        auto rounded = out[(p + CGEMM_SB_THREE_M_PLANES) * TILE];
        Cast(packed[p * TILE], out[p * TILE], AscendC::RoundMode::CAST_RINT, count);
        Cast(rounded, packed[p * TILE], AscendC::RoundMode::CAST_NONE, count);
        Sub(rounded, out[p * TILE], rounded, count);
        Cast(packed[(p + CGEMM_SB_THREE_M_PLANES) * TILE], rounded, AscendC::RoundMode::CAST_RINT, count);
    }
    halfOutput_.EnQue(packed);
    packed = halfOutput_.DeQue<half>();
    for (uint32_t p = 0; p < CGEMM_SB_HALF_PLANES; ++p) {
        uint64_t base = (isA ? 0 : t_.bOffset) + p * (isA ? t_.aSize : t_.bSize);
        auto target = reinterpret_cast<__gm__ half*>(ws_.GetPhyAddr(base));
        AscendC::GlobalTensor<half> halfWs;
        halfWs.SetGlobalBuffer(target);
        AscendC::DataCopy(halfWs[batch * matrix + offset], packed[p * TILE], count);
    }
    halfOutput_.FreeTensor(packed);
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::PublishMetadata()
{
    auto running = calc_.Get<int32_t>();
    auto maxValue = running[VECTOR_REDUCTION_RESULT_SLOT * TILE];
    auto shared = running[VECTOR_REDUCTION_SCRATCH_SLOT * TILE];
    AscendC::ReduceMax(maxValue, running, shared, TILE * 2);
    AscendC::PipeBarrier<PIPE_ALL>();
    // Absolute IEEE bits order finite positive magnitudes, infinity and NaN.
    auto flag = output_.AllocTensor<float>();
    AscendC::Duplicate(
        flag,
        maxValue.GetValue(0) >= (t_.compensatedHalf ? CGEMM_SB_HALF_FALLBACK_BITS : CGEMM_SB_FLOAT_FALLBACK_BITS) ?
            1.0f :
            0.0f,
        CGEMM_SB_FLAG_STRIDE);
    if (t_.threeM && t_.k >= CGEMM_SB_ORDERED_MIN_K) {
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t p = 0; p < CGEMM_SB_MOMENT_PLANES; ++p) {
            auto result = calc_.Get<float>()[VECTOR_REDUCTION_RESULT_SLOT * TILE];
            ReduceSum(
                result, moments_.Get<float>()[p * TILE], calc_.Get<float>()[VECTOR_REDUCTION_SCRATCH_SLOT * TILE],
                TILE);
            AscendC::PipeBarrier<PIPE_ALL>();
            flag.SetValue(p + CGEMM_SB_META_A_REAL, result.GetValue(0));
        }
        flag.SetValue(CGEMM_SB_META_A_COUNT, float(countA_));
        flag.SetValue(CGEMM_SB_META_B_COUNT, float(countB_));
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    output_.EnQue(flag);
    flag = output_.DeQue<float>();
    AscendC::DataCopy(ws_[t_.flagOffset + uint64_t(core_) * CGEMM_SB_FLAG_STRIDE], flag, CGEMM_SB_FLAG_STRIDE);
    output_.FreeTensor(flag);
}

__aicore__ inline void CgemmStridedBatchedVectorImpl::CombineGeneral(
    AscendC::LocalTensor<float> data, AscendC::LocalTensor<float> out, const ColumnTile& view, int64_t dest,
    uint32_t count)
{
    auto work = calc_.Get<float>();
    auto pr = work;
    auto pi = work[TILE];
    auto cr = work[VECTOR_COMBINE_REAL_SLOT * TILE];
    auto ci = work[VECTOR_COMBINE_IMAG_SLOT * TILE];
    auto temp = work[VECTOR_COMBINE_TEMP_SLOT * TILE];
    auto betaR = work[VECTOR_COMBINE_BETA_REAL_SLOT * TILE];
    auto betaI = work[VECTOR_COMBINE_BETA_IMAG_SLOT * TILE];
    auto aux = work[VECTOR_COMBINE_AUX_SLOT * TILE];

    Sub(pr, data, data[CGEMM_SB_PRODUCT_II * TILE], count);
    Add(pi, data[CGEMM_SB_PRODUCT_IR * TILE], data[CGEMM_SB_PRODUCT_RI * TILE], count);

    Muls(cr, pr, t_.ar, count);
    Muls(temp, pi, t_.ai, count);
    Sub(cr, cr, temp, count);
    Muls(ci, pi, t_.ar, count);
    Muls(temp, pr, t_.ai, count);
    Add(ci, ci, temp, count);
    if (t_.br != 0 || t_.bi != 0) {
        auto old = old_.AllocTensor<float>();
        if (t_.denseCombine)
            AscendC::DataCopy(old, c_[dest], count * 2);
        else
            ReadColumns(old, c_[dest], view, t_.ldc);
        old_.EnQue(old);
        old = old_.DeQue<float>();
        Gather(pr, old, even_, 0U, count);
        Gather(pi, old, odd_, 0U, count);
        Muls(betaR, pr, t_.br, count);
        Muls(aux, pi, t_.bi, count);
        Sub(betaR, betaR, aux, count);
        Muls(betaI, pi, t_.br, count);
        Muls(aux, pr, t_.bi, count);
        Add(betaI, betaI, aux, count);
        Add(cr, cr, betaR, count);
        Add(ci, ci, betaI, count);
        AscendC::PipeBarrier<PIPE_V>();
        old_.FreeTensor(old);
    }
    Scatter(out, cr, even_, 0U, count);
    Scatter(out, ci, odd_, 0U, count);
}

} // namespace

template <bool Mixed>
__aicore__ inline void CgemmSbRunExceptional(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t, uint32_t core, uint32_t cores)
{
    auto flags = reinterpret_cast<__gm__ float*>(ws) + t.flagOffset;
    if constexpr (!Mixed) {
        if (t.threeM && t.k >= CGEMM_SB_ORDERED_MIN_K) {
            uint32_t mask = CgemmSbReferenceMask(flags, t);
            if (CgemmSbPartialReference(mask, t)) {
                if (t.mp <= PARTIAL_TILE_ELEMENTS) {
                    AscendC::TPipe mergePipe;
                    FastUnitCombiner<PARTIAL_TILE_ELEMENTS, true> op;
                    op.Run(c, ws, t, core, cores, mergePipe, mask);
                } else {
                    asc_vf_call<CgemmSbMergeComponent>(
                        dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(c),
                        reinterpret_cast<__gm__ float*>(ws), t, mask, core, cores);
                }
            } else if (mask == CGEMM_SB_REFERENCE_EXCEPTIONAL) {
                asc_vf_call<CgemmSbUnitVector>(
                    dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a),
                    reinterpret_cast<__gm__ float*>(b), reinterpret_cast<__gm__ float*>(c), t, core, cores);
            }
            return;
        }
    }
    if (t.threeM && uint64_t(t.m) * t.n * t.batch <= CGEMM_SB_INT32_INDEX_MAX) {
        asc_vf_call<CgemmSbUnitVector>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), t, core, cores);
        return;
    }
    if (uint64_t(t.m) * t.n * t.batch <= CGEMM_SB_INT32_INDEX_MAX) {
        asc_vf_call<CgemmSbVector<uint32_t>>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), reinterpret_cast<__gm__ float*>(ws), t, CGEMM_SB_MODE_DIRECT, core,
            cores);
    } else {
        asc_vf_call<CgemmSbVector<uint64_t>>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), reinterpret_cast<__gm__ float*>(ws), t, CGEMM_SB_MODE_DIRECT, core,
            cores);
    }
    return;
}

__aicore__ inline bool UseFastHalfPacker(const CgemmSbTiling& t, bool pack)
{
    return pack && t.compensatedHalf && t.k < SHORT_HALF_K_LIMIT && t.mp <= FAST_TILE_ELEMENTS &&
           t.np <= FAST_TILE_ELEMENTS;
}

__aicore__ inline bool UseFastUnitCombiner(const CgemmSbTiling& t, bool pack)
{
    return !pack && t.threeM && t.mp <= FAST_TILE_ELEMENTS;
}

template <bool Mixed>
__aicore__ inline void CgemmSbRunVector(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t, bool pack)
{
    uint32_t core = AscendC::GetBlockIdx();
    uint32_t cores = AscendC::GetBlockNum() * (Mixed ? 2 : 1);

    if (!pack) {
        auto flags = reinterpret_cast<__gm__ float*>(ws) + t.flagOffset;
        bool exceptional = CgemmSbNeedsReference(flags, t);
        if (exceptional) {
            CgemmSbRunExceptional<Mixed>(a, b, c, ws, t, core, cores);
            return;
        }
    }
    AscendC::TPipe pipe;
    if (UseFastHalfPacker(t, pack)) {
        if (t.mp == FAST_CUBE_DIM && t.np == FAST_CUBE_DIM && t.kp == FAST_CUBE_DIM) {
            FastHalfPacker<FAST_TILE_ELEMENTS> op;
            op.Run(a, b, ws, t, core, cores, pipe);
        } else {
            FastHalfPacker<LARGE_FAST_TILE_ELEMENTS> op;
            op.Run(a, b, ws, t, core, cores, pipe);
        }
        return;
    }
    if (UseFastUnitCombiner(t, pack)) {
        uint64_t matrix = uint64_t(t.mp) * t.np;
        if (matrix > FAST_TILE_ELEMENTS && matrix <= LARGE_FAST_TILE_ELEMENTS) {
            FastUnitCombiner<LARGE_FAST_TILE_ELEMENTS, false, 1> op;
            op.Run(c, ws, t, core, cores, pipe);
        } else {
            FastUnitCombiner<FAST_TILE_ELEMENTS> op;
            op.Run(c, ws, t, core, cores, pipe);
        }
        return;
    }
    CgemmStridedBatchedVectorImpl op;
    op.Init(a, b, c, ws, t, pack, Mixed, pipe);
    if (pack)
        op.Pack();
    else
        op.Combine();
}

} // namespace

using namespace AscendC;

extern "C" __global__ __aicore__ void cgemm_sb_fused_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        CgemmSbRunVector<true>(a, b, c, ws, t, true);
        CrossCoreSetFlag<0, PIPE_MTE3>(CGEMM_SB_FUSED_VECTOR_READY);
        CrossCoreWaitFlag<0>(CGEMM_SB_FUSED_VECTOR_READY);
        CrossCoreSetFlag<2, PIPE_MTE3>(CGEMM_SB_FUSED_PACK_READY);
        CrossCoreWaitFlag<2>(CGEMM_SB_CUBE_DONE);
        CgemmSbRunVector<true>(a, b, c, ws, t, false);
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<2>(CGEMM_SB_FUSED_PACK_READY);
        CgemmSbRunCube(ws, t);
        CrossCoreSetFlag<0, PIPE_FIX>(CGEMM_SB_CUBE_BARRIER);
        CrossCoreWaitFlag<0>(CGEMM_SB_CUBE_BARRIER);
        CrossCoreSetFlag<2, PIPE_FIX>(CGEMM_SB_CUBE_DONE);
    }
}
void cgemm_sb_fused_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t)
{
    cgemm_sb_fused_kernel<<<blocks, nullptr, stream>>>(a, b, c, ws, t);
}

// Dense Vector kernel entry.
extern "C" __global__ __aicore__ void cgemm_sb_dense_vector_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t, bool pack)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemmSbRunVector<false>(a, b, c, ws, t, pack);
}
void cgemm_sb_dense_vector_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling& t,
    bool pack)
{
    cgemm_sb_dense_vector_kernel<<<blocks, nullptr, stream>>>(a, b, c, workspace, t, pack);
}

// Direct SIMT reductions, tiny matrices and input normalization.
// The fast unit-alpha path maintains four output columns per thread.
constexpr uint32_t UNIT_COLUMNS_PER_THREAD = 4;

template <int TA, int TB>
__simt_callee__ __aicore__ inline void CgemmSbTinyExceptional(
    __gm__ float* a, __gm__ float* b, const CgemmSbTiling& t, uint32_t batch, uint32_t row, uint32_t col, float& sr,
    float& si)
{
#pragma clang fp contract(off)
    uint32_t astep = TA == 0 ? 2 * t.lda : 2;
    uint32_t bstep = TB == 0 ? 2 : 2 * t.ldb;
    auto ap = a + 2 * (int64_t(batch) * t.strideA + (TA == 0 ? row : int64_t(row) * t.lda));
    auto bp = b + 2 * (int64_t(batch) * t.strideB + (TB == 0 ? int64_t(col) * t.ldb : col));
    sr = 0;
    si = 0;
    for (uint32_t k = 0; k < t.k; ++k) {
        float ar = ap[0];
        float ai = ap[1];
        float br = bp[0];
        float bi = bp[1];
        if constexpr (TA == 2)
            ai = -ai;
        if constexpr (TB == 2)
            bi = -bi;
        if constexpr (TA == 0) {
            float tr = br - 0.0f * bi;
            float ti = bi + 0.0f * br;
            br = tr;
            bi = ti;
        }
        CgemmSbAccumulateComplex(ar, ai, br, bi, sr, si);
        ap += astep;
        bp += bstep;
    }
    if constexpr (TA != 0) {
        float rr;
        float ri;
        CgemmSbScale(1.0f, 0.0f, sr, si, rr, ri);
        sr = rr;
        si = ri;
    }
}

template <int TA, int TB>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_SB_SIMT_LAUNCH_BOUND) inline void CgemmSbFastUnitFour(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, CgemmSbTiling t, uint32_t core, uint32_t cores)
{
    uint32_t matrix = t.m * ((t.n + UNIT_COLUMNS_PER_THREAD - 1) / UNIT_COLUMNS_PER_THREAD);
    for (uint32_t idx = core * blockDim.x + threadIdx.x; idx < matrix * t.batch; idx += cores * blockDim.x) {
        uint32_t batch = idx / matrix;
        uint32_t row = idx % t.m;
        uint32_t col = idx % matrix / t.m * UNIT_COLUMNS_PER_THREAD;
        auto ap = a + 2 * (int64_t(batch) * t.strideA + (TA == 0 ? row : int64_t(row) * t.lda));
        auto bp = b + 2 * (int64_t(batch) * t.strideB + (TB == 0 ? int64_t(col) * t.ldb : col));
        uint32_t astep = TA == 0 ? 2 * t.lda : 2;
        uint32_t bstep = TB == 0 ? 2 : 2 * t.ldb;
        float real[UNIT_COLUMNS_PER_THREAD] = {0, 0, 0, 0};
        float imag[UNIT_COLUMNS_PER_THREAD] = {0, 0, 0, 0};
        uint32_t columnStep = TB == 0 ? 2 * t.ldb : 2;
        for (uint32_t k = 0; k < t.k; ++k) {
            float ar = ap[0];
            float ai = ap[1];
            if constexpr (TA == 2)
                ai = -ai;
#pragma unroll
            for (uint32_t j = 0; j < UNIT_COLUMNS_PER_THREAD; ++j) {
                if (col + j >= t.n)
                    continue;
                float br = bp[j * columnStep];
                float bi = bp[j * columnStep + 1];
                if constexpr (TB == 2)
                    bi = -bi;
                real[j] = fmaf(ar, br, real[j]);
                real[j] = fmaf(-ai, bi, real[j]);
                imag[j] = fmaf(ar, bi, imag[j]);
                imag[j] = fmaf(ai, br, imag[j]);
            }
            ap += astep;
            bp += bstep;
        }
#pragma unroll
        for (uint32_t j = 0; j < UNIT_COLUMNS_PER_THREAD; ++j) {
            if (col + j >= t.n)
                continue;
            float sr = real[j];
            float si = imag[j];
            if (!isfinite(sr) || !isfinite(si)) {
                CgemmSbTinyExceptional<TA, TB>(a, b, t, batch, row, col + j, sr, si);
            }
            auto out = c + 2 * (int64_t(batch) * t.strideC + int64_t(col + j) * t.ldc + row);
            out[0] = sr;
            out[1] = si;
        }
    }
}

// Input normalization for ordered accumulation.
namespace {
constexpr uint32_t NORMAL_TILE = 64, NORMAL_PITCH = 68;
// A 64-lane FP32 register holds 32 interleaved complex rows.
constexpr uint32_t NORMAL_COMPLEX_ROWS_PER_REGISTER = 32;
constexpr uint32_t NORMAL_INPUT = NORMAL_PITCH * NORMAL_TILE * 2;
constexpr uint32_t NORMAL_OUTPUT = NORMAL_TILE * NORMAL_TILE * 2;

__simd_vf__ inline void NormalizeTile(__ubuf__ float* input, __ubuf__ float* output, uint32_t trans)
{
    namespace M = AscendC::MicroAPI;
    auto mask = M::CreateMask<uint32_t, M::MaskPattern::ALL>();
    M::RegTensor<uint32_t> lane, part, row, indexBase, index, sign, bits;
    M::Arange(reinterpret_cast<M::RegTensor<int32_t>&>(lane), int32_t(0));
    M::Duplicate(part, uint32_t(1), mask);
    M::And(part, lane, part, mask);
    M::ShiftRights(row, lane, int16_t(1), mask);
    M::Muls(indexBase, row, NORMAL_PITCH * 2, mask);
    M::Add(indexBase, indexBase, part, mask);
    M::ShiftLefts(sign, part, int16_t(CGEMM_SB_FP32_SIGN_SHIFT), mask);
    for (uint16_t i = 0; i < NORMAL_OUTPUT / CGEMM_SB_FP32_LANES; ++i) {
        M::RegTensor<float> value;
        if (trans == ACLBLAS_OP_N) {
            M::DataCopy(value, input + uint32_t(i / 2) * NORMAL_PITCH * 2 + uint32_t(i % 2) * CGEMM_SB_FP32_LANES);
        } else {
            M::Adds(
                index, indexBase,
                uint32_t(i % 2) * NORMAL_COMPLEX_ROWS_PER_REGISTER * NORMAL_PITCH * 2 + uint32_t(i / 2) * 2, mask);
            M::Gather(value, input, index, mask);
        }
        if (trans == ACLBLAS_OP_C) {
            M::Xor(bits, reinterpret_cast<M::RegTensor<uint32_t>&>(value), sign, mask);
            M::DataCopy(output + uint32_t(i) * CGEMM_SB_FP32_LANES, reinterpret_cast<M::RegTensor<float>&>(bits), mask);
        } else {
            M::DataCopy(output + uint32_t(i) * CGEMM_SB_FP32_LANES, value, mask);
        }
    }
}

__aicore__ inline uint32_t NormalizeCount(uint32_t extent, uint32_t offset)
{
    if (offset >= extent)
        return 0;
    return extent - offset < NORMAL_TILE ? extent - offset : NORMAL_TILE;
}

__aicore__ inline void StoreNormalizedTile(
    AscendC::GlobalTensor<float> output, AscendC::LocalTensor<float> out, uint32_t batch, uint32_t row0, uint32_t col0,
    uint32_t rows, uint32_t cols)
{
    using namespace AscendC;
    uint32_t copyRows = NormalizeCount(rows, row0);
    uint32_t copyCols = NormalizeCount(cols, col0);
    DataCopyExtParams copy{
        static_cast<uint16_t>(copyCols), copyRows * CGEMM_SB_COMPLEX_BYTES,
        (NORMAL_TILE -
         (copyRows + CGEMM_SB_COMPLEX_PER_BLOCK - 1) / CGEMM_SB_COMPLEX_PER_BLOCK * CGEMM_SB_COMPLEX_PER_BLOCK) /
            CGEMM_SB_COMPLEX_PER_BLOCK,
        (rows - copyRows) * CGEMM_SB_COMPLEX_BYTES, 0};
    DataCopyPad(output[2 * (uint64_t(batch) * rows * cols + uint64_t(col0) * rows + row0)], out, copy);
}

__aicore__ inline void LoadNormalizeInput(
    AscendC::LocalTensor<float> in, AscendC::GlobalTensor<float> input, int64_t offset, uint32_t rows, uint32_t cols,
    uint32_t ld)
{
    using namespace AscendC;
    DataCopyExtParams copy{
        static_cast<uint16_t>(cols), rows * CGEMM_SB_COMPLEX_BYTES, (ld - rows) * CGEMM_SB_COMPLEX_BYTES,
        (NORMAL_PITCH -
         (rows + CGEMM_SB_COMPLEX_PER_BLOCK - 1) / CGEMM_SB_COMPLEX_PER_BLOCK * CGEMM_SB_COMPLEX_PER_BLOCK) /
            CGEMM_SB_COMPLEX_PER_BLOCK,
        0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(in, input[2 * offset], copy, pad);
}

__aicore__ inline void NormalizeMatrix(GM_ADDR source, GM_ADDR destination, const CgemmSbTiling& t, bool left)
{
    using namespace AscendC;
    uint32_t rows = left ? t.orderedM : t.orderedK;
    uint32_t cols = left ? t.orderedK : t.orderedN;
    uint32_t validRows = left ? t.m : t.k;
    uint32_t validCols = left ? t.k : t.n;
    uint32_t trans = left ? t.transA : t.transB;
    uint32_t ld = left ? t.lda : t.ldb;
    int64_t stride = left ? t.strideA : t.strideB;
    uint32_t batches = left ? t.batchesA : t.batchesB;
    // Ordered accumulation reads only K >= prefix. The Cube prefix reads
    // the separate packed planes, so earlier normalized K tiles are unused.
    uint32_t firstKTile = CgemmSbPrefixK(t) / NORMAL_TILE;
    uint32_t firstRow = left ? 0 : firstKTile;
    uint32_t firstCol = left ? firstKTile : 0;
    uint32_t tileRows = (rows + NORMAL_TILE - 1) / NORMAL_TILE - firstRow;
    uint32_t tileCols = (cols + NORMAL_TILE - 1) / NORMAL_TILE - firstCol;
    uint64_t tiles = uint64_t(tileRows) * tileCols;
    GlobalTensor<float> input;
    GlobalTensor<float> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(destination));
    LocalTensor<float> in(TPosition::VECIN, 0, NORMAL_INPUT);
    LocalTensor<float> out(TPosition::VECOUT, NORMAL_INPUT * CGEMM_SB_FP32_BYTES, NORMAL_OUTPUT);
    for (uint64_t task = GetBlockIdx(); task < tiles * batches; task += GetBlockNum()) {
        uint32_t batch = task / tiles;
        uint32_t row0 = (task % tileRows + firstRow) * NORMAL_TILE;
        uint32_t col0 = (task % tiles / tileRows + firstCol) * NORMAL_TILE;
        uint32_t realRows = NormalizeCount(validRows, row0);
        uint32_t realCols = NormalizeCount(validCols, col0);
        uint32_t physicalRows = trans == ACLBLAS_OP_N ? realRows : realCols;
        uint32_t physicalCols = trans == ACLBLAS_OP_N ? realCols : realRows;
        Duplicate(in, 0.0f, NORMAL_INPUT);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        if (physicalRows != 0 && physicalCols != 0) {
            int64_t offset = int64_t(batch) * stride +
                             (trans == ACLBLAS_OP_N ? int64_t(col0) * ld + row0 : int64_t(row0) * ld + col0);
            LoadNormalizeInput(in, input, offset, physicalRows, physicalCols, ld);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        asc_vf_call<NormalizeTile>(
            reinterpret_cast<__ubuf__ float*>(in.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(out.GetPhyAddr()),
            trans);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        StoreNormalizedTile(output, out, batch, row0, col0, rows, cols);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}
} // namespace

extern "C" __global__ __aicore__ void cgemm_sb_normalize_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR ws, CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t mask = CgemmSbReferenceMask(reinterpret_cast<__gm__ float*>(ws) + t.flagOffset, t);
    if (mask == CGEMM_SB_REFERENCE_NONE || mask == CGEMM_SB_REFERENCE_EXCEPTIONAL)
        return;
    auto af = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ float*>(ws) + t.orderedAOffset);
    auto bf = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ float*>(ws) + t.orderedBOffset);
    NormalizeMatrix(a, af, t, true);
    NormalizeMatrix(b, bf, t, false);
}

void cgemm_sb_normalize_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR ws, const CgemmSbTiling& t)
{
    cgemm_sb_normalize_kernel<<<blocks, nullptr, stream>>>(a, b, ws, t);
}

// Direct Vector dispatch and kernel entry.
namespace {
template <int TA>
__aicore__ inline void DispatchTinyB(GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmSbTiling& t)
{
    if (t.transB == ACLBLAS_OP_N)
        asc_vf_call<CgemmSbFastUnitFour<TA, 0>>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), t, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    if (t.transB == ACLBLAS_OP_T)
        asc_vf_call<CgemmSbFastUnitFour<TA, 1>>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), t, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    if (t.transB == ACLBLAS_OP_C)
        asc_vf_call<CgemmSbFastUnitFour<TA, 2>>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), t, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}

__aicore__ inline void DispatchTiny(GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmSbTiling& t)
{
    if (t.transA == ACLBLAS_OP_N)
        DispatchTinyB<0>(a, b, c, t);
    else if (t.transA == ACLBLAS_OP_T)
        DispatchTinyB<1>(a, b, c, t);
    else
        DispatchTinyB<2>(a, b, c, t);
}
} // namespace

extern "C" __global__ __aicore__ void cgemm_sb_vector_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t, uint32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (mode == CGEMM_SB_MODE_TINY) {
        DispatchTiny(a, b, c, t);
        return;
    }
    if (mode == CGEMM_SB_MODE_UNIT) {
        asc_vf_call<CgemmSbUnitVector>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), t, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
        return;
    }
    if (uint64_t(t.m) * t.n * t.batch <= CGEMM_SB_INT32_INDEX_MAX) {
        asc_vf_call<CgemmSbVector<uint32_t>>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), reinterpret_cast<__gm__ float*>(ws), t, mode, AscendC::GetBlockIdx(),
            AscendC::GetBlockNum());
    } else {
        asc_vf_call<CgemmSbVector<uint64_t>>(
            dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(c), reinterpret_cast<__gm__ float*>(ws), t, mode, AscendC::GetBlockIdx(),
            AscendC::GetBlockNum());
    }
}

void cgemm_sb_vector_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling& t,
    uint32_t mode)
{
    if (mode == CGEMM_SB_MODE_PACK || mode == CGEMM_SB_MODE_COMBINE) {
        cgemm_sb_dense_vector_do(blocks, stream, a, b, c, workspace, t, mode == CGEMM_SB_MODE_PACK);
        return;
    }
    cgemm_sb_vector_kernel<<<blocks, nullptr, stream>>>(a, b, c, workspace, t, mode);
}
