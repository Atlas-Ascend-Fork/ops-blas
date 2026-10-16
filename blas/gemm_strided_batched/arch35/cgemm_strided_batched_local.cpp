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

// Local packing is private to the Cube execution paths below.
namespace {
// Split a SIMD lane index into the 16-wide local fractal coordinates.
constexpr uint32_t LOCAL_FRACTAL_LANE_MASK = 15;
constexpr int16_t LOCAL_FRACTAL_LANE_SHIFT = 4;
// A gather covers an 8-by-4 logical block; the A representation
// doubles its columns, while B swaps the K/column roles.
constexpr uint32_t LOCAL_PACK_K_BLOCK = 8;
constexpr uint32_t LOCAL_PACK_COLUMN_BLOCK = 4;
struct LocalPackShape {
    uint32_t mp, np, kp, trans, localInputPad, batchGroup;
};
template <class Register, class Mask>
__simd_callee__ inline void LocalPackIndices(
    LocalPackShape t, bool isA, Register& bits, Register& sign, Register& mag, uint32_t& stepK, uint32_t& stepC,
    Mask mask)
{
    namespace M = AscendC::MicroAPI;
    M::RegTensor<uint32_t> lane, lo, hi, row, col, part, tmp, neg, one;
    M::Arange(reinterpret_cast<M::RegTensor<int32_t>&>(lane), int32_t(0));
    M::Duplicate(one, uint32_t(LOCAL_FRACTAL_LANE_MASK), mask);
    M::And(lo, lane, one, mask);
    M::ShiftRights(hi, lane, int16_t(LOCAL_FRACTAL_LANE_SHIFT), mask);
    M::Duplicate(one, uint32_t(1), mask);
    uint32_t trans = t.trans;
    uint32_t ld =
        (isA ? (trans == ACLBLAS_OP_N ? t.mp : t.kp) : (trans == ACLBLAS_OP_N ? t.kp : t.np)) + t.localInputPad;
    M::Move(row, hi, mask);
    M::Move(col, lo, mask);
    if (isA) {
        M::Xor(part, row, col, mask);
        M::And(part, part, one, mask);
        M::And(neg, row, one, mask);
        M::Xor(neg, neg, one, mask);
        M::And(tmp, col, one, mask);
        M::And(neg, neg, tmp, mask);
        M::ShiftRights(row, row, int16_t(1), mask);
        M::ShiftRights(col, col, int16_t(1), mask);
    } else {
        M::And(part, col, one, mask);
        M::Duplicate(neg, uint32_t(0), mask);
        M::ShiftRights(tmp, col, int16_t(1), mask);
        M::Move(col, row, mask);
        M::Move(row, tmp, mask);
    }
    if (trans == ACLBLAS_OP_N) {
        M::Muls(bits, col, ld, mask);
        M::Add(bits, bits, row, mask);
    } else {
        M::Muls(bits, row, ld, mask);
        M::Add(bits, bits, col, mask);
    }
    M::ShiftLefts(bits, bits, int16_t(1), mask);
    M::Add(bits, bits, part, mask);
    if (trans == ACLBLAS_OP_C)
        M::Xor(neg, neg, part, mask);
    M::ShiftLefts(sign, neg, int16_t(CGEMM_SB_FP32_SIGN_SHIFT), mask);
    M::Or(bits, bits, sign, mask);
    stepK = isA ? (trans == ACLBLAS_OP_N ? CGEMM_SB_FRACTAL_EDGE * ld : CGEMM_SB_FRACTAL_EDGE) :
                  (trans == ACLBLAS_OP_N ? CGEMM_SB_FRACTAL_EDGE : CGEMM_SB_FRACTAL_EDGE * ld);
    stepC = isA ? (trans == ACLBLAS_OP_N ? LOCAL_PACK_COLUMN_BLOCK : LOCAL_PACK_COLUMN_BLOCK * ld) :
                  (trans == ACLBLAS_OP_N ? LOCAL_PACK_K_BLOCK * ld : LOCAL_PACK_K_BLOCK);
    M::Duplicate(mag, uint32_t(CGEMM_SB_FP32_ABS_MASK), mask);
    M::And(bits, bits, mag, mask);
}

template <bool Merge>
__simd_vf__ inline void PackLocal(
    __ubuf__ float* src, __ubuf__ float* dst, __ubuf__ int32_t* maxima, uint32_t size, uint32_t insize,
    uint32_t batches, LocalPackShape t, bool isA)
{
    namespace M = AscendC::MicroAPI;
    static constexpr M::CastTrait down{
        M::RegLayout::ZERO, M::SatMode::NO_SAT, M::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr M::CastTrait up{
        M::RegLayout::ZERO, M::SatMode::UNKNOWN, M::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    M::RegTensor<uint32_t> mag, index;
    M::RegTensor<int32_t> maxReg;
    auto mask = M::CreateMask<uint32_t, M::MaskPattern::ALL>();
    M::DataCopy(maxReg, maxima);
    M::RegTensor<uint32_t> bits, sign, tmp;
    uint32_t stepK = 0, stepC = 0, columns = isA ? 2 * t.mp : t.np;
    LocalPackIndices(t, isA, bits, sign, mag, stepK, stepC, mask);

    for (uint16_t batch = 0; batch < batches; ++batch) {
        for (uint16_t kblock = 0; kblock < t.kp / LOCAL_PACK_K_BLOCK; ++kblock) {
            for (uint16_t cblock = 0; cblock < columns / LOCAL_PACK_COLUMN_BLOCK; ++cblock) {
                uint32_t offset =
                    uint32_t(kblock) * columns * CGEMM_SB_FRACTAL_EDGE + uint32_t(cblock) * CGEMM_SB_FP32_LANES;
                M::Adds(index, bits, uint32_t(kblock) * stepK + uint32_t(cblock) * stepC, mask);
                M::RegTensor<float> value;
                M::Gather(value, src + uint32_t(batch) * insize, index, mask);
                M::Xor(tmp, reinterpret_cast<M::RegTensor<uint32_t>&>(value), sign, mask);
                M::RegTensor<half> high, low;
                M::RegTensor<float> rounded, residual;
                M::Cast<half, float, down>(high, reinterpret_cast<M::RegTensor<float>&>(tmp), mask);
                M::Cast<float, half, up>(rounded, high, mask);
                M::Sub(residual, reinterpret_cast<M::RegTensor<float>&>(tmp), rounded, mask);
                M::Cast<half, float, down>(low, residual, mask);
                uint32_t highOffset = uint32_t(batch) * size * 2 + offset, lowOffset = highOffset + size;
                if constexpr (Merge) {
                    highOffset =
                        isA ? uint32_t(batch) * size + offset :
                              (uint32_t(batch) * t.kp / LOCAL_PACK_K_BLOCK + kblock) * (t.np * t.batchGroup) *
                                      CGEMM_SB_FRACTAL_EDGE +
                                  (uint32_t(batch) * t.np / LOCAL_PACK_COLUMN_BLOCK + cblock) * CGEMM_SB_FP32_LANES;
                    lowOffset = highOffset + size * t.batchGroup;
                }
                M::DataCopy<half, M::StoreDist::DIST_PACK_B32>(
                    reinterpret_cast<__ubuf__ half*>(dst) + highOffset, high, mask);
                M::DataCopy<half, M::StoreDist::DIST_PACK_B32>(
                    reinterpret_cast<__ubuf__ half*>(dst) + lowOffset, low, mask);
                M::And(tmp, tmp, mag, mask);
                M::Max(maxReg, maxReg, reinterpret_cast<M::RegTensor<int32_t>&>(tmp), mask);
            }
        }
    }
    M::DataCopy(maxima, maxReg, mask);
}

struct LocalInputView {
    uint32_t trans, rows, cols, pitch, paddedRows, paddedCols, ld, inputSize, outputSize;
};

__aicore__ inline LocalInputView LocalInput(const CgemmSbTiling& t, bool isA, bool merge)
{
    LocalInputView v{};
    v.trans = isA ? t.transA : t.transB;
    bool normal = v.trans == ACLBLAS_OP_N;
    v.rows = isA ? (normal ? t.m : t.k) : (normal ? t.k : t.n);
    v.cols = isA ? (normal ? t.k : t.m) : (normal ? t.n : t.k);
    v.paddedRows = isA ? (normal ? t.mp : t.kp) : (normal ? t.kp : t.np);
    v.pitch = v.paddedRows + t.localInputPad;
    v.paddedCols = isA ? (normal ? t.kp : t.mp) : (normal ? t.np : t.kp);
    v.ld = isA ? t.lda : t.ldb;
    v.inputSize = 2 * v.pitch * v.paddedCols;
    v.outputSize = isA ? CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp :
                         CGEMM_SB_COMPONENT_PLANES * t.np * t.kp * (merge ? t.batchGroup : 1);
    return v;
}

__aicore__ inline void CopyLocalInput(
    AscendC::LocalTensor<float> x, AscendC::GlobalTensor<float> source, const LocalInputView& v, int64_t stride,
    uint32_t batch, uint32_t count, bool forceColumnCopy = false)
{
    // The extra pitch only separates UB banks; the gather never reads those holes.
    // Zero logical matrix padding, without clearing unused pitch-only storage.
    if (v.rows != v.paddedRows || v.cols != v.paddedCols) {
        AscendC::Duplicate(x, 0.0f, v.inputSize * count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    }
    uint32_t aligned = (v.rows * 2 + CGEMM_SB_FP32_PER_BLOCK - 1) / CGEMM_SB_FP32_PER_BLOCK * CGEMM_SB_FP32_PER_BLOCK;
    AscendC::DataCopyExtParams cp{
        static_cast<uint16_t>(v.cols), v.rows * CGEMM_SB_COMPLEX_BYTES, (v.ld - v.rows) * CGEMM_SB_COMPLEX_BYTES,
        (v.pitch * 2 - aligned) / CGEMM_SB_FP32_PER_BLOCK, 0};
    AscendC::DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(aligned - v.rows * 2), 0};
    if (!forceColumnCopy && v.rows == v.pitch && v.ld == v.pitch && stride == v.inputSize / 2 &&
        reinterpret_cast<uint64_t>(source.GetPhyAddr()) % CGEMM_SB_DMA_BLOCK_BYTES == 0)
        AscendC::DataCopy(x, source[2 * int64_t(batch) * stride], v.inputSize * count);
    else if (v.cols == v.paddedCols && stride == int64_t(v.ld) * v.cols) {
        cp.blockCount = static_cast<uint16_t>(v.cols * count);
        AscendC::DataCopyPad(x, source[2 * int64_t(batch) * stride], cp, pad);
    } else
        for (uint32_t j = 0; j < count; ++j)
            AscendC::DataCopyPad(x[j * v.inputSize], source[2 * int64_t(batch + j) * stride], cp, pad);
}

} // namespace

using namespace AscendC;
using namespace AscendC::Te;

// Single-batch local compensated Cube path.
namespace {

constexpr uint32_t LOCAL_AIVS_PER_AIC = 2;

struct LocalBatchRange {
    uint32_t first;
    uint32_t count;
    uint32_t rounds;
};

template <bool Merge>
__aicore__ inline LocalBatchRange LocalWorkerRange(const CgemmSbTiling& t, uint32_t worker, uint32_t workers)
{
    if constexpr (!Merge) {
        if (t.batchGroup == 1) {
            // Balance Cube work first, then split each Cube's contiguous
            // batches between its AIV pair. Both AIVs signal the same rounds.
            uint32_t cubes = workers / LOCAL_AIVS_PER_AIC;
            uint32_t cube = worker / LOCAL_AIVS_PER_AIC;
            uint32_t perCube = t.batch / cubes;
            uint32_t extra = t.batch % cubes;
            uint32_t count = perCube + (cube < extra ? 1U : 0U);
            uint32_t first = cube * perCube + (cube < extra ? cube : extra);
            uint32_t leading = (count + LOCAL_AIVS_PER_AIC - 1) / LOCAL_AIVS_PER_AIC;
            return worker % LOCAL_AIVS_PER_AIC == 0 ? LocalBatchRange{first, leading, leading} :
                                                      LocalBatchRange{first + leading, count - leading, leading};
        }
    }
    uint32_t perWorker = t.batch / workers;
    uint32_t extra = t.batch % workers;
    return {
        worker * perWorker + (worker < extra ? worker : extra), perWorker + (worker < extra ? 1U : 0U),
        perWorker + (worker / LOCAL_AIVS_PER_AIC * LOCAL_AIVS_PER_AIC < extra ? 1U : 0U)};
}

// A low-batch row slice owns one output. Fixed buffers and a full L0 operand
// bank avoid the queue and ping-pong bookkeeping needed by the batch pipeline.
// The two AIVs pack A and B independently into separate 256 KiB L1 regions.
__aicore__ inline void LocalSinglePack(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t)
{
    const bool isA = AscendC::GetBlockIdx() % 2 == 0;
    auto view = LocalInput(t, isA, false);
    uint32_t maxInput = view.inputSize;
    uint32_t maxOutput = view.outputSize;
    AscendC::LocalTensor<float> x(AscendC::TPosition::VECIN, 0, maxInput);
    AscendC::LocalTensor<float> y(AscendC::TPosition::VECOUT, maxInput * CGEMM_SB_FP32_BYTES, maxOutput);
    AscendC::LocalTensor<int32_t> maxima(
        AscendC::TPosition::VECCALC, (maxInput + maxOutput) * CGEMM_SB_FP32_BYTES, CGEMM_SB_MAX_STORAGE_ELEMENTS);
    AscendC::GlobalTensor<float> ag;
    AscendC::GlobalTensor<float> bg;
    AscendC::GlobalTensor<float> wg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
    AscendC::Duplicate(maxima, int32_t(0), CGEMM_SB_FP32_LANES);
    CopyLocalInput(x, isA ? ag : bg, view, isA ? t.strideA : t.strideB, 0, 1, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    asc_vf_call<PackLocal<false>>(
        reinterpret_cast<__ubuf__ float*>(x.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(y.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(maxima.GetPhyAddr()), view.outputSize, view.inputSize, 1,
        LocalPackShape{t.mp, t.np, t.kp, isA ? t.transA : t.transB, t.localInputPad, t.batchGroup}, isA);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::LocalTensor<float> l1(AscendC::TPosition::A1, isA ? 0 : CGEMM_SB_LOCAL_L1_REGION_BYTES, view.outputSize);
    AscendC::DataCopy(l1, y, view.outputSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::ReduceMax(
        maxima[CGEMM_SB_MAX_RESULT_OFFSET], maxima, maxima[CGEMM_SB_MAX_TEMP_OFFSET], CGEMM_SB_FP32_LANES);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    bool exceptional = maxima.GetValue(CGEMM_SB_MAX_RESULT_OFFSET) >= CGEMM_SB_HALF_FALLBACK_BITS;
    auto flag = maxima[CGEMM_SB_MAX_FLAG_OFFSET].ReinterpretCast<float>();
    AscendC::Duplicate(flag, exceptional ? 1.0f : 0.0f, CGEMM_SB_FLAG_STRIDE);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(wg[isA ? 0 : CGEMM_SB_FLAG_STRIDE], flag, CGEMM_SB_FLAG_STRIDE);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(CGEMM_SB_LOCAL_PACK_EVENT_BASE);
    AscendC::CrossCoreWaitFlag<2>(CGEMM_SB_CUBE_DONE);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    if (isA) {
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(wg);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(wg[CGEMM_SB_FLAG_STRIDE]);
        if (wg.GetValue(0) != 0.0f || wg.GetValue(CGEMM_SB_FLAG_STRIDE) != 0.0f)
            asc_vf_call<CgemmSbUnitVector>(
                dim3{CGEMM_SB_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a),
                reinterpret_cast<__gm__ float*>(b), reinterpret_cast<__gm__ float*>(c), t, 0, 1);
    }
}

template <typename Left, typename Right, typename Accumulator>
__aicore__ inline void LocalSingleProduct(
    Left al, Right bl, Accumulator cc, uint32_t cm, uint32_t ck, uint32_t cn, bool first, bool last)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    auto a0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, half>(0),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(cm, ck));
    // Retain both B planes when they fit in two 32 KiB banks. Wider slices
    // use the full 64 KiB bank and reload high B for the final product.
    bool retainB = uint64_t(ck) * cn * sizeof(half) <= CGEMM_HALF_HALF_L0_SIZE;
    uint64_t bOffset = !first && !last && retainB ? CGEMM_HALF_HALF_L0_SIZE : 0;
    auto b0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, half>(bOffset),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(ck, cn));
    if (!last) {
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{}), a0, al);
    }
    if (!last || !retainB) {
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{}), b0, bl);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::Te::MmadParams params{
        static_cast<uint16_t>(cm), static_cast<uint16_t>(cn), static_cast<uint16_t>(ck),
        static_cast<uint8_t>(last ? CGEMM_HALF_FINAL_ACCUMULATION : CGEMM_HALF_NON_FINAL_ACCUMULATION), first};
    AscendC::Te::Mmad(
        AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(params), cc, a0, b0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
}

__aicore__ inline void LocalSingleCube(GM_ADDR c, GM_ADDR ws, CgemmSbTiling t)
{
    AscendC::CrossCoreWaitFlag<2>(CGEMM_SB_LOCAL_PACK_EVENT_BASE);
    AscendC::GlobalTensor<float> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(flags);
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(flags[CGEMM_SB_FLAG_STRIDE]);
    if (flags.GetValue(0) == 0.0f && flags.GetValue(CGEMM_SB_FLAG_STRIDE) == 0.0f) {
        AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        uint32_t as = CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp;
        uint32_t bs = CGEMM_SB_COMPONENT_PLANES * t.np * t.kp;
        auto ah = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, half>(CGEMM_SB_LOCAL_L1_REGION_BYTES),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(
                t.np, 2 * t.kp));
        auto al = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, half>(CGEMM_SB_LOCAL_L1_REGION_BYTES + bs * 2),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(
                t.np, 2 * t.kp));
        auto bh = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, half>(0),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(
                2 * t.kp, 2 * t.mp));
        auto bl = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, half>(as * 2),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(
                2 * t.kp, 2 * t.mp));
        auto cc = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(
                t.np, 2 * t.mp));
        LocalSingleProduct(al, bh, cc, t.np, 2 * t.kp, 2 * t.mp, true, false);
        LocalSingleProduct(ah, bl, cc, t.np, 2 * t.kp, 2 * t.mp, false, false);
        LocalSingleProduct(ah, bh, cc, t.np, 2 * t.kp, 2 * t.mp, false, true);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        auto out = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ float*>(c)),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(t.n, 2 * t.ldc));
        auto outputSlice = out.Slice(AscendC::Te::MakeCoord(0, 0), AscendC::Te::MakeShape(t.n, 2 * t.m));
        auto accumulatorSlice = cc.Slice(AscendC::Te::MakeCoord(0, 0), AscendC::Te::MakeShape(t.n, 2 * t.m));
        AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{})
            .Call(outputSlice, accumulatorSlice, AscendC::Te::FixpipeParams{CGEMM_HALF_FINAL_ACCUMULATION});
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    }
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CGEMM_SB_CUBE_DONE);
}

} // namespace

// Grouped and sliced local compensated Cube paths.
namespace {
// Reduction scratch plus one DMA block of alignment headroom.
constexpr uint32_t LOCAL_REDUCTION_RESERVE_BYTES = 1056;
constexpr uint32_t LOCAL_WORKSPACE_BYTES_PER_AIC = 128;
// Four 64 KiB slots or two 128 KiB slots partition each L1 region.
constexpr uint32_t LOCAL_SMALL_SLOT_STAGES = 4;
constexpr uint32_t LOCAL_LARGE_SLOT_STAGES = 2;

template <bool Merge>
__aicore__ inline uint32_t LocalSlotBytes(const CgemmSbTiling& t)
{
    uint32_t as = CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp;
    uint32_t bs = CGEMM_SB_COMPONENT_PLANES * t.np * t.kp * (Merge ? t.batchGroup : 1);
    return uint64_t(as + bs) * t.batchGroup * CGEMM_SB_FP32_BYTES <= CGEMM_SB_LOCAL_SMALL_SLOT_BYTES ?
               CGEMM_SB_LOCAL_SMALL_SLOT_BYTES :
               CGEMM_SB_LOCAL_LARGE_SLOT_BYTES;
}

template <bool Merge>
class CgemmStridedBatchedLocal {
public:
    __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, CgemmSbTiling t, TPipe& pipe)
    {
        uint32_t core = t.localSlices ? GetBlockIdx() % 2 : GetBlockIdx();
        uint32_t cores = t.localSlices ? 2 : GetBlockNum() * 2;
        uint32_t sub = core % 2;
        uint32_t group = t.batchGroup;
        uint32_t as = CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp;
        uint32_t bs = CGEMM_SB_COMPONENT_PLANES * t.np * t.kp * (Merge ? group : 1);
        uint32_t stages = uint64_t(as + bs) * group * CGEMM_SB_FP32_BYTES > CGEMM_SB_LOCAL_LARGE_SLOT_BYTES ?
                              1 :
                              (LocalSlotBytes<Merge>(t) == CGEMM_SB_LOCAL_SMALL_SLOT_BYTES ? LOCAL_SMALL_SLOT_STAGES :
                                                                                             LOCAL_LARGE_SLOT_STAGES);
        InitBuffers(t, pipe);
        GlobalTensor<float> ag;
        GlobalTensor<float> bg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        uint32_t round = 0;
        auto range = LocalWorkerRange<Merge>(t, core, cores);
        uint32_t firstBase = range.first;
        uint32_t lastBatch = firstBase + range.count;
        uint32_t maxCount = range.rounds;
        auto maxbuf = scratch.Get<int32_t>();
        Duplicate(maxbuf, int32_t(0), CGEMM_SB_FP32_LANES);
        for (uint32_t base = 0; base < maxCount; base += group, ++round) {
            uint32_t slot = round % stages;
            uint32_t batch = firstBase + base;
            if (round >= stages)
                CrossCoreWaitFlag<2>(CGEMM_SB_LOCAL_RELEASE_EVENT_BASE + slot);
            if (batch < lastBatch) {
                uint32_t count = group < lastBatch - batch ? group : lastBatch - batch;
                PackOperand(t, true, ag, batch, count, slot, sub);
                PackOperand(t, false, bg, batch, count, slot, sub);
            }
            CrossCoreSetFlag<2, PIPE_MTE3>(CGEMM_SB_LOCAL_PACK_EVENT_BASE + slot);
        }
        for (uint32_t pending = round > stages ? round - stages : 0; pending < round; ++pending)
            CrossCoreWaitFlag<2>(CGEMM_SB_LOCAL_RELEASE_EVENT_BASE + pending % stages);
        // No output is still in flight after all PIPE_FIX releases. An
        // exceptional input triggers the ordered fallback for this AIV's
        // disjoint batches; normal inputs need just one scalar reduction.
        if (MaximumMagnitude() >= CGEMM_SB_HALF_FALLBACK_BITS) {
            for (uint32_t base = 0; base < maxCount; base += group) {
                uint32_t batch = firstBase + base;
                if (batch < lastBatch)
                    Fallback(a, b, c, t, batch, group < lastBatch - batch ? group : lastBatch - batch);
            }
        }
    }

private:
    __aicore__ inline int32_t MaximumMagnitude()
    {
        auto maxbuf = scratch.Get<int32_t>();
        ReduceMax(maxbuf[CGEMM_SB_MAX_RESULT_OFFSET], maxbuf, maxbuf[CGEMM_SB_MAX_TEMP_OFFSET], CGEMM_SB_FP32_LANES);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        return maxbuf.GetValue(CGEMM_SB_MAX_RESULT_OFFSET);
    }

    __aicore__ inline void InitBuffers(const CgemmSbTiling& t, TPipe& pipe)
    {
        uint32_t group = t.batchGroup;
        uint32_t as = CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp;
        uint32_t bs = CGEMM_SB_COMPONENT_PLANES * t.np * t.kp * (Merge ? group : 1);
        uint32_t ai =
            2 * ((t.transA == ACLBLAS_OP_N ? t.mp : t.kp) + t.localInputPad) * (t.transA == ACLBLAS_OP_N ? t.kp : t.mp);
        uint32_t bi =
            2 * ((t.transB == ACLBLAS_OP_N ? t.kp : t.np) + t.localInputPad) * (t.transB == ACLBLAS_OP_N ? t.np : t.kp);
        uint32_t maxInput = ai > bi ? ai : bi;
        uint32_t maxOutput = as > bs ? as : bs;
        uint32_t outputBuffers =
            (uint64_t(maxInput) + 2 * maxOutput) * group * CGEMM_SB_FP32_BYTES + LOCAL_REDUCTION_RESERVE_BYTES <=
                    CGEMM_SB_UB_BUDGET_BYTES ?
                2 :
                1;
        pipe.InitBuffer(input, 1, maxInput * group * CGEMM_SB_FP32_BYTES);
        pipe.InitBuffer(out, outputBuffers, maxOutput * group * CGEMM_SB_FP32_BYTES);
        pipe.InitBuffer(scratch, CGEMM_SB_MAX_SCRATCH_BYTES);
    }

    __aicore__ inline void PackOperand(
        const CgemmSbTiling& t, bool isA, GlobalTensor<float> source, uint32_t batch, uint32_t count, uint32_t slot,
        uint32_t sub)
    {
        auto view = LocalInput(t, isA, Merge);
        auto x = input.AllocTensor<float>();
        CopyLocalInput(x, source, view, isA ? t.strideA : t.strideB, batch, count);
        input.EnQue(x);
        x = input.DeQue<float>();
        auto y = out.AllocTensor<float>();
        if constexpr (Merge)
            Duplicate(y, 0.0f, view.outputSize * t.batchGroup);
        asc_vf_call<PackLocal<Merge>>(
            reinterpret_cast<__ubuf__ float*>(x.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(y.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(scratch.Get<int32_t>().GetPhyAddr()), view.outputSize, view.inputSize,
            count, LocalPackShape{t.mp, t.np, t.kp, view.trans, t.localInputPad, t.batchGroup}, isA);
        out.EnQue(y);
        y = out.DeQue<float>();
        LocalTensor<float> l1(
            TPosition::A1,
            uint64_t(sub) * CGEMM_SB_LOCAL_L1_REGION_BYTES + slot * LocalSlotBytes<Merge>(t) +
                (isA ? uint64_t(0) :
                       uint64_t(CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp * t.batchGroup * CGEMM_SB_FP32_BYTES)),
            view.outputSize * count);
        DataCopy(l1, y, view.outputSize * (Merge ? t.batchGroup : count));
        out.FreeTensor(y);
        input.FreeTensor(x);
    }

    __aicore__ inline void Fallback(GM_ADDR a, GM_ADDR b, GM_ADDR c, CgemmSbTiling t, uint32_t batch, uint32_t count)
    {
        t.batch = count;
        dim3 vectorThreads = {CGEMM_SB_SIMT_THREADS, 1, 1};
        asc_vf_call<CgemmSbUnitVector>(
            vectorThreads, reinterpret_cast<__gm__ float*>(a) + 2 * int64_t(batch) * t.strideA,
            reinterpret_cast<__gm__ float*>(b) + 2 * int64_t(batch) * t.strideB,
            reinterpret_cast<__gm__ float*>(c) + 2 * int64_t(batch) * t.strideC, t, 0, 1);
    }
    TQue<TPosition::VECIN, 1> input;
    TQue<TPosition::VECOUT, 2> out;
    TBuf<TPosition::VECCALC> scratch;
};
__aicore__ inline void LocalFullProduct(
    uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl, uint32_t cm, uint32_t cn, uint32_t ck, uint64_t cOffset)
{
    LocalTensor<half> aHigh(TPosition::A1, ah, cm * ck);
    LocalTensor<half> aLow(TPosition::A1, al, cm * ck);
    LocalTensor<half> bHigh(TPosition::A1, bh, ck * cn);
    LocalTensor<half> bLow(TPosition::A1, bl, ck * cn);
    LocalTensor<half> a0(TPosition::A2, 0, cm * ck);
    LocalTensor<half> a1(TPosition::A2, CGEMM_SB_L0_HALF_BANK_BYTES, cm * ck);
    LocalTensor<half> b0(TPosition::B2, 0, ck * cn);
    LocalTensor<half> b1(TPosition::B2, CGEMM_SB_L0_HALF_BANK_BYTES, ck * cn);
    LocalTensor<float> output(TPosition::CO1, cOffset, cm * cn);
    LoadData2DParamsV2 ap{
        0,
        0,
        uint16_t(cm / CGEMM_SB_FRACTAL_EDGE),
        uint16_t(ck / CGEMM_SB_FRACTAL_EDGE),
        int32_t(cm / CGEMM_SB_FRACTAL_EDGE),
        uint16_t(cm / CGEMM_SB_FRACTAL_EDGE),
        false,
        0};
    LoadData2DParamsV2 bp{
        0,
        0,
        uint16_t(cn / CGEMM_SB_FRACTAL_EDGE),
        uint16_t(ck / CGEMM_SB_FRACTAL_EDGE),
        int32_t(cn / CGEMM_SB_FRACTAL_EDGE),
        uint16_t(cn / CGEMM_SB_FRACTAL_EDGE),
        false,
        0};
    WaitFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(1);
    LoadData(a0, aLow, ap);
    LoadData(b0, bHigh, bp);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    AscendC::MmadParams params{uint16_t(cm), uint16_t(cn), uint16_t(ck), uint8_t(CGEMM_HALF_NON_FINAL_ACCUMULATION),
                               false,        true};
    AscendC::Mmad(output, a0, b0, params);
    LoadData(a1, aHigh, ap);
    LoadData(b1, bLow, bp);
    SetFlag<HardEvent::MTE1_M>(1);
    WaitFlag<HardEvent::MTE1_M>(1);
    params.cmatrixInitVal = false;
    AscendC::Mmad(output, a1, b1, params);
    params.unitFlag = CGEMM_HALF_FINAL_ACCUMULATION;
    AscendC::Mmad(output, a1, b0, params);
    SetFlag<HardEvent::M_MTE1>(0);
    SetFlag<HardEvent::M_MTE1>(1);
}

template <bool Merge>
__aicore__ inline void LocalCubeBatch(
    GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t, uint32_t batch, uint32_t core, uint32_t count, uint32_t sub,
    uint32_t slot, uint32_t j, uint64_t& l0, uint32_t& outputIndex)
{
    uint32_t group = t.batchGroup;
    uint32_t as = CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp;
    uint32_t bs = CGEMM_SB_COMPONENT_PLANES * t.np * t.kp * (Merge ? group : 1);
    uint32_t cm = t.np * (Merge ? group : 1);
    uint32_t ck = 2 * t.kp * (Merge ? group : 1);
    uint32_t planeGroup = Merge ? group : 1;
    uint32_t outputRows = Merge ? t.n * count : t.n;
    uint64_t ab = uint64_t(sub) * CGEMM_SB_LOCAL_L1_REGION_BYTES + slot * LocalSlotBytes<Merge>(t) +
                  as * group * CGEMM_SB_FP32_BYTES + j * bs * CGEMM_SB_FP32_BYTES;
    uint64_t bb =
        uint64_t(sub) * CGEMM_SB_LOCAL_L1_REGION_BYTES + slot * LocalSlotBytes<Merge>(t) + j * as * CGEMM_SB_FP32_BYTES;
    auto al = MakeTensor(
        MakeMemPtr<Location::L1, half>(ab),
        MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(cm, ck));
    auto alo = MakeTensor(
        MakeMemPtr<Location::L1, half>(ab + bs * planeGroup * 2),
        MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(cm, ck));
    auto bl = MakeTensor(
        MakeMemPtr<Location::L1, half>(bb),
        MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(ck, 2 * t.mp));
    auto blo = MakeTensor(
        MakeMemPtr<Location::L1, half>(bb + as * planeGroup * 2),
        MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(ck, 2 * t.mp));
    // The owning AIV corrects exceptional groups only after the PIPE_FIX
    // release below. Outputs are disjoint, so Cube needs no GM flag round trip.
    uint32_t outputSlot = outputIndex++ % 2;
    uint64_t cOffset = uint64_t(outputSlot) * CGEMM_SB_L0C_SLOT_BYTES;
    WaitFlag<HardEvent::FIX_M>(outputSlot);
    if (ck <= t.baseK) {
        LocalFullProduct(ab, ab + bs * planeGroup * 2, bb, bb + as * planeGroup * 2, cm, 2 * t.mp, ck, cOffset);
    } else {
        CgemmHalfL0MmadLoop(alo, bl, cm, ck, 2 * t.mp, 1, 0, t.baseK, l0, true, false, cOffset);
        CgemmHalfL0MmadLoop(al, blo, cm, ck, 2 * t.mp, 1, 0, t.baseK, l0, false, false, cOffset);
        CgemmHalfL0MmadLoop(al, bl, cm, ck, 2 * t.mp, 1, 0, t.baseK, l0, false, true, cOffset);
    }
    SetFlag<HardEvent::M_FIX>(0);
    WaitFlag<HardEvent::M_FIX>(0);
    auto out = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(c) + 2 * int64_t(batch) * t.strideC),
        MakeFrameLayout<NDExtLayoutPtn>(outputRows, 2 * t.ldc));
    auto cc = MakeTensor(
        MakeMemPtr<Location::L0C, float>(cOffset),
        MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<CGEMM_SB_FRACTAL_EDGE>>(cm, 2 * t.mp));
    MakeCopy(CopyL0C2GM{})
        .Call(
            out.Slice(MakeCoord(0, 0), MakeShape(outputRows, 2 * t.m)),
            cc.Slice(MakeCoord(0, 0), MakeShape(outputRows, 2 * t.m)),
            AscendC::Te::FixpipeParams{CGEMM_HALF_FINAL_ACCUMULATION});
    SetFlag<HardEvent::FIX_M>(outputSlot);
}

template <bool Merge>
__aicore__ inline void LocalCube(GM_ADDR c, GM_ADDR ws, CgemmSbTiling t)
{
    uint32_t core = t.localSlices ? 0 : GetBlockIdx();
    uint32_t cores = t.localSlices ? 1 : GetBlockNum();
    uint32_t group = t.batchGroup;
    uint32_t as = CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp;
    uint32_t bs = CGEMM_SB_COMPONENT_PLANES * t.np * t.kp * (Merge ? group : 1);
    uint32_t stages = uint64_t(as + bs) * group * CGEMM_SB_FP32_BYTES > CGEMM_SB_LOCAL_LARGE_SLOT_BYTES ?
                          1 :
                          (LocalSlotBytes<Merge>(t) == CGEMM_SB_LOCAL_SMALL_SLOT_BYTES ? LOCAL_SMALL_SLOT_STAGES :
                                                                                         LOCAL_LARGE_SLOT_STAGES);
    SetHF32Mode(HF32Mode::DISABLE);
    SetFlag<HardEvent::M_MTE1>(0);
    SetFlag<HardEvent::M_MTE1>(1);
    uint64_t l0 = 0;
    uint32_t round = 0;
    uint32_t outputIndex = 0;
    SetFlag<HardEvent::FIX_M>(0);
    SetFlag<HardEvent::FIX_M>(1);
    uint32_t maxCount = LocalWorkerRange<Merge>(t, core * LOCAL_AIVS_PER_AIC, cores * LOCAL_AIVS_PER_AIC).rounds;
    for (uint32_t base = 0; base < maxCount; base += group, ++round) {
        uint32_t slot = round % stages;
        CrossCoreWaitFlag<2>(CGEMM_SB_LOCAL_PACK_EVENT_BASE + slot);
        for (uint32_t sub = 0; sub < 2; ++sub) {
            uint32_t worker = core * 2 + sub;
            auto range = LocalWorkerRange<Merge>(t, worker, cores * LOCAL_AIVS_PER_AIC);
            uint32_t assigned = range.count;
            if (base >= assigned)
                continue;
            uint32_t count = group < assigned - base ? group : assigned - base;
            uint32_t batch = range.first + base;
            for (uint32_t j = 0; j < (Merge ? 1 : count); ++j) {
                LocalCubeBatch<Merge>(c, ws, t, batch + j, core, count, sub, slot, j, l0, outputIndex);
            }
        }
        CrossCoreSetFlag<2, PIPE_FIX>(CGEMM_SB_LOCAL_RELEASE_EVENT_BASE + slot);
    }
    WaitFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(1);
    WaitFlag<HardEvent::FIX_M>(0);
    WaitFlag<HardEvent::FIX_M>(1);
}
__aicore__ inline bool PrepareLocalSlice(
    GM_ADDR& a, GM_ADDR& b, GM_ADDR& c, GM_ADDR& ws, CgemmSbTiling& t, uint64_t task)
{
    uint32_t core = 0;
    if ASCEND_IS_AIV {
        core = GetBlockIdx() / 2;
    }
    if ASCEND_IS_AIC {
        core = GetBlockIdx();
    }
    if (task >= uint64_t(t.batch) * t.localSlices)
        return false;
    uint32_t batch = task / t.localSlices;
    uint32_t row = task % t.localSlices * t.localSliceRows;
    a += 2 * (int64_t(batch) * t.strideA + (t.transA == ACLBLAS_OP_N ? row : int64_t(row) * t.lda)) * sizeof(float);
    b += 2 * int64_t(batch) * t.strideB * sizeof(float);
    c += 2 * (int64_t(batch) * t.strideC + row) * sizeof(float);
    ws += core * LOCAL_WORKSPACE_BYTES_PER_AIC;
    t.m = t.m - row < t.localSliceRows ? t.m - row : t.localSliceRows;
    t.mp = (t.m + CGEMM_SB_FP32_PER_BLOCK - 1) / CGEMM_SB_FP32_PER_BLOCK * CGEMM_SB_FP32_PER_BLOCK;
    t.batch = 1;
    uint32_t width = t.np > 2 * t.mp ? t.np : 2 * t.mp;
    t.baseK = CGEMM_SB_L0_HALF_BANK_ELEMENTS / width / CGEMM_SB_FRACTAL_EDGE * CGEMM_SB_FRACTAL_EDGE;
    if (t.baseK > 2 * t.kp)
        t.baseK = 2 * t.kp;
    return true;
}
} // namespace
extern "C" __global__ __aicore__ void cgemm_sb_local_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (t.localSlices) {
        uint32_t core = GetBlockIdx();
        if ASCEND_IS_AIV {
            core /= 2;
        }
        for (uint64_t task = core; task < uint64_t(t.batch) * t.localSlices; task += GetBlockNum()) {
            GM_ADDR sliceA = a;
            GM_ADDR sliceB = b;
            GM_ADDR sliceC = c;
            GM_ADDR sliceWs = ws;
            CgemmSbTiling slice = t;
            if (!PrepareLocalSlice(sliceA, sliceB, sliceC, sliceWs, slice, task))
                return;
            if ASCEND_IS_AIV {
                LocalSinglePack(sliceA, sliceB, sliceC, sliceWs, slice);
            }
            if ASCEND_IS_AIC {
                LocalSingleCube(sliceC, sliceWs, slice);
            }
        }
    } else if (t.localBatchMerge) {
        if ASCEND_IS_AIV {
            TPipe pipe;
            CgemmStridedBatchedLocal<true> pack;
            pack.Run(a, b, c, ws, t, pipe);
        }
        if ASCEND_IS_AIC {
            LocalCube<true>(c, ws, t);
        }
    } else {
        if ASCEND_IS_AIV {
            TPipe pipe;
            CgemmStridedBatchedLocal<false> pack;
            pack.Run(a, b, c, ws, t, pipe);
        }
        if ASCEND_IS_AIC {
            LocalCube<false>(c, ws, t);
        }
    }
}
void cgemm_sb_local_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t)
{
    cgemm_sb_local_kernel<<<blocks, nullptr, stream>>>(a, b, c, ws, t);
}

// Ordered FP32 SIMD accumulation for the compensated suffix.
namespace {
constexpr uint32_t ORDERED_TILE_ROWS = 64;
constexpr uint32_t ORDERED_TILE_COLUMNS = 32;
constexpr uint32_t ORDERED_TILE_K = 128;
constexpr uint32_t ORDERED_FULL_COLUMNS_PER_CALL = 4;
constexpr uint32_t ORDERED_PARTIAL_COLUMNS_PER_CALL = 8;

// Microkernels explicitly unroll columns 0..3 (complex) or 0..7
// (one component); their template indices and register offsets
// below are column coordinates, not independent tuning values.

template <uint32_t KC, uint32_t Column, typename Mask, typename Register>
__simd_callee__ inline void CgemmSbFullColumn(
    Register& real, Register& imag, Register& ar, Register& ai, Register& nai, __ubuf__ float* b, uint32_t k,
    Mask& mask)
{
    namespace M = AscendC::MicroAPI;
    M::RegTensor<float> br, bi, product;
    M::DataCopy<float, M::LoadDist::DIST_BRC_B32>(br, b + 2 * (Column * KC + k));
    M::DataCopy<float, M::LoadDist::DIST_BRC_B32>(bi, b + 2 * (Column * KC + k) + 1);
    M::Mul(product, ar, br, mask);
    M::MulAddDst(product, nai, bi, mask);
    M::Add(real, real, product, mask);
    M::Mul(product, ar, bi, mask);
    M::MulAddDst(product, ai, br, mask);
    M::Add(imag, imag, product, mask);
}

template <uint32_t KC>
__simd_vf__ inline void CgemmSbOrderedMicro(__ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* c, uint32_t count)
{
    namespace M = AscendC::MicroAPI;
    auto mask = M::CreateMask<float, M::MaskPattern::ALL>();
    M::RegTensor<float> real0, imag0, real1, imag1, real2, imag2, real3, imag3;
    auto c0 = c + 0;
    M::DataCopy<float, M::PostLiteral::POST_MODE_UPDATE, M::LoadDist::DIST_DINTLV_B32>(
        real0, imag0, c0, CGEMM_SB_COMPLEX_REGISTER_FLOATS);
    auto c1 = c + CGEMM_SB_COMPLEX_REGISTER_FLOATS;
    M::DataCopy<float, M::PostLiteral::POST_MODE_UPDATE, M::LoadDist::DIST_DINTLV_B32>(
        real1, imag1, c1, CGEMM_SB_COMPLEX_REGISTER_FLOATS);
    auto c2 = c + 2 * CGEMM_SB_COMPLEX_REGISTER_FLOATS;
    M::DataCopy<float, M::PostLiteral::POST_MODE_UPDATE, M::LoadDist::DIST_DINTLV_B32>(
        real2, imag2, c2, CGEMM_SB_COMPLEX_REGISTER_FLOATS);
    auto c3 = c + 3 * CGEMM_SB_COMPLEX_REGISTER_FLOATS;
    M::DataCopy<float, M::PostLiteral::POST_MODE_UPDATE, M::LoadDist::DIST_DINTLV_B32>(
        real3, imag3, c3, CGEMM_SB_COMPLEX_REGISTER_FLOATS);
#pragma clang loop unroll(disable)
    for (uint16_t k = 0; k < count; ++k) {
        M::RegTensor<float> ar, ai, nai;
        auto src = a + uint32_t(k) * CGEMM_SB_COMPLEX_REGISTER_FLOATS;
        M::DataCopy<float, M::PostLiteral::POST_MODE_UPDATE, M::LoadDist::DIST_DINTLV_B32>(
            ar, ai, src, CGEMM_SB_COMPLEX_REGISTER_FLOATS);
        M::Muls(nai, ai, -1.0f, mask);
        CgemmSbFullColumn<KC, 0>(real0, imag0, ar, ai, nai, b, k, mask);
        CgemmSbFullColumn<KC, 1>(real1, imag1, ar, ai, nai, b, k, mask);
        CgemmSbFullColumn<KC, 2>(real2, imag2, ar, ai, nai, b, k, mask);
        CgemmSbFullColumn<KC, 3>(real3, imag3, ar, ai, nai, b, k, mask);
    }
    M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(c + 0, real0, imag0, mask);
    M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(c + CGEMM_SB_COMPLEX_REGISTER_FLOATS, real1, imag1, mask);
    M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(c + 2 * CGEMM_SB_COMPLEX_REGISTER_FLOATS, real2, imag2, mask);
    M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(c + 3 * CGEMM_SB_COMPLEX_REGISTER_FLOATS, real3, imag3, mask);
}

template <uint32_t KC, uint32_t Component, uint32_t Column, typename Mask, typename Register>
__simd_callee__ inline void CgemmSbPartialPair(
    Register& sum0, Register& sum1, Register& ar, Register& ai, __ubuf__ float* b, uint32_t k, Mask& mask)
{
    namespace M = AscendC::MicroAPI;
    M::RegTensor<float> br0, bi0, br1, bi1, p0, p1;
    M::DataCopy<float, M::LoadDist::DIST_BRC_B32>(br0, b + 2 * (Column * KC + k));
    M::DataCopy<float, M::LoadDist::DIST_BRC_B32>(bi0, b + 2 * (Column * KC + k) + 1);
    M::DataCopy<float, M::LoadDist::DIST_BRC_B32>(br1, b + 2 * ((Column + 1) * KC + k));
    M::DataCopy<float, M::LoadDist::DIST_BRC_B32>(bi1, b + 2 * ((Column + 1) * KC + k) + 1);
    if constexpr (Component == CGEMM_SB_REFERENCE_REAL) {
        M::Mul(p0, ar, br0, mask);
        M::Mul(p1, ar, br1, mask);
        M::MulAddDst(p0, ai, bi0, mask);
        M::MulAddDst(p1, ai, bi1, mask);
    } else {
        M::Mul(p0, ar, bi0, mask);
        M::Mul(p1, ar, bi1, mask);
        M::MulAddDst(p0, ai, br0, mask);
        M::MulAddDst(p1, ai, br1, mask);
    }
    M::Add(sum0, sum0, p0, mask);
    M::Add(sum1, sum1, p1, mask);
}

template <uint32_t KC, uint32_t Component>
__simd_vf__ inline void CgemmSbOrderedPartialMicro(
    __ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* c, uint32_t count)
{
    namespace M = AscendC::MicroAPI;
    auto mask = M::CreateMask<float, M::MaskPattern::ALL>();
    M::RegTensor<float> sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    M::DataCopy(sum0, c + 0);
    M::DataCopy(sum1, c + CGEMM_SB_FP32_LANES);
    M::DataCopy(sum2, c + CGEMM_SB_COMPLEX_REGISTER_FLOATS);
    M::DataCopy(sum3, c + 3 * CGEMM_SB_FP32_LANES);
    M::DataCopy(sum4, c + 2 * CGEMM_SB_COMPLEX_REGISTER_FLOATS);
    M::DataCopy(sum5, c + 5 * CGEMM_SB_FP32_LANES);
    M::DataCopy(sum6, c + 3 * CGEMM_SB_COMPLEX_REGISTER_FLOATS);
    M::DataCopy(sum7, c + 7 * CGEMM_SB_FP32_LANES);
#pragma clang loop unroll(disable)
    for (uint16_t k = 0; k < count; ++k) {
        M::RegTensor<float> ar, ai;
        auto src = a + uint32_t(k) * CGEMM_SB_COMPLEX_REGISTER_FLOATS;
        M::DataCopy<float, M::PostLiteral::POST_MODE_UPDATE, M::LoadDist::DIST_DINTLV_B32>(
            ar, ai, src, CGEMM_SB_COMPLEX_REGISTER_FLOATS);
        if constexpr (Component == CGEMM_SB_REFERENCE_REAL)
            M::Muls(ai, ai, -1.0f, mask);
        CgemmSbPartialPair<KC, Component, 0>(sum0, sum1, ar, ai, b, k, mask);
        CgemmSbPartialPair<KC, Component, 2>(sum2, sum3, ar, ai, b, k, mask);
        CgemmSbPartialPair<KC, Component, 4>(sum4, sum5, ar, ai, b, k, mask);
        CgemmSbPartialPair<KC, Component, 6>(sum6, sum7, ar, ai, b, k, mask);
    }
    M::StoreAlign(c + 0, sum0, mask);
    M::StoreAlign(c + CGEMM_SB_FP32_LANES, sum1, mask);
    M::StoreAlign(c + CGEMM_SB_COMPLEX_REGISTER_FLOATS, sum2, mask);
    M::StoreAlign(c + 3 * CGEMM_SB_FP32_LANES, sum3, mask);
    M::StoreAlign(c + 2 * CGEMM_SB_COMPLEX_REGISTER_FLOATS, sum4, mask);
    M::StoreAlign(c + 5 * CGEMM_SB_FP32_LANES, sum5, mask);
    M::StoreAlign(c + 3 * CGEMM_SB_COMPLEX_REGISTER_FLOATS, sum6, mask);
    M::StoreAlign(c + 7 * CGEMM_SB_FP32_LANES, sum7, mask);
}

template <uint32_t Component>
__simd_vf__ inline void CgemmSbExpandComponent(__ubuf__ float* input, __ubuf__ float* output, uint32_t count)
{
    namespace M = AscendC::MicroAPI;
    auto mask = M::CreateMask<float, M::MaskPattern::ALL>();
    for (uint16_t q = 0; q < count / CGEMM_SB_FP32_LANES; ++q) {
        M::RegTensor<float> value, zero;
        M::Duplicate(zero, 0.0f, mask);
        M::DataCopy(value, input + uint32_t(q) * CGEMM_SB_FP32_LANES);
        if constexpr (Component == CGEMM_SB_REFERENCE_REAL)
            M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(
                output + uint32_t(q) * CGEMM_SB_COMPLEX_REGISTER_FLOATS, value, zero, mask);
        else
            M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(
                output + uint32_t(q) * CGEMM_SB_COMPLEX_REGISTER_FLOATS, zero, value, mask);
    }
}

template <uint32_t MR, uint32_t NR, uint32_t KR>
__aicore__ inline void CgemmSbLoadOrderedTile(
    AscendC::LocalTensor<float> al, AscendC::LocalTensor<float> bl, AscendC::GlobalTensor<float> ag,
    AscendC::GlobalTensor<float> bg, const CgemmSbTiling& t, uint32_t batch, uint32_t row0, uint32_t col0, uint32_t k0,
    uint32_t slot)
{
    uint32_t count = t.k - k0 < KR ? t.k - k0 : KR;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
    AscendC::DataCopyExtParams ac{
        static_cast<uint16_t>(count), MR * CGEMM_SB_COMPLEX_BYTES, (t.lda - MR) * CGEMM_SB_COMPLEX_BYTES, 0, 0};
    AscendC::DataCopyExtParams bc{
        NR, count * CGEMM_SB_COMPLEX_BYTES, (t.ldb - count) * CGEMM_SB_COMPLEX_BYTES,
        (KR - (count + CGEMM_SB_COMPLEX_PER_BLOCK - 1) / CGEMM_SB_COMPLEX_PER_BLOCK * CGEMM_SB_COMPLEX_PER_BLOCK) /
            CGEMM_SB_COMPLEX_PER_BLOCK,
        0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(al, ag[2 * (int64_t(batch) * t.strideA + int64_t(k0) * t.lda + row0)], ac, pad);
    AscendC::DataCopyPad(bl, bg[2 * (int64_t(batch) * t.strideB + int64_t(col0) * t.ldb + k0)], bc, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}

template <uint32_t COLS>
__simd_vf__ inline void CgemmSbInitializeFullPrefix(
    __ubuf__ float* realPlanes, __ubuf__ float* imagPlanes, __ubuf__ float* output)
{
    namespace M = AscendC::MicroAPI;
    auto mask = M::CreateMask<float, M::MaskPattern::ALL>();
    for (uint16_t q = 0; q < COLS; ++q) {
        M::RegTensor<float> rr, ii, ri, ir, real, imag;
        M::DataCopy(rr, realPlanes + uint32_t(q) * CGEMM_SB_FP32_LANES);
        M::DataCopy(ii, realPlanes + COLS * CGEMM_SB_FP32_LANES + uint32_t(q) * CGEMM_SB_FP32_LANES);
        M::DataCopy(ri, imagPlanes + uint32_t(q) * CGEMM_SB_FP32_LANES);
        M::DataCopy(ir, imagPlanes + COLS * CGEMM_SB_FP32_LANES + uint32_t(q) * CGEMM_SB_FP32_LANES);
        M::Sub(real, rr, ii, mask);
        M::Add(imag, ri, ir, mask);
        M::StoreAlign<float, M::StoreDist::DIST_INTLV_B32>(
            output + uint32_t(q) * CGEMM_SB_COMPLEX_REGISTER_FLOATS, real, imag, mask);
    }
}

template <uint32_t MR, uint32_t NR>
__aicore__ inline void CgemmSbLoadFullPrefix(
    AscendC::LocalTensor<float> al, AscendC::LocalTensor<float> bl, AscendC::LocalTensor<float> cl, GM_ADDR workspace,
    const CgemmSbTiling& t, uint32_t batch, uint32_t row0, uint32_t col0)
{
    constexpr uint32_t CSIZE = MR * NR * 2;
    AscendC::GlobalTensor<float> wg;
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    uint64_t offset = t.cOffset + uint64_t(batch) * t.mp * t.np + uint64_t(col0) * t.mp + row0;
    uint32_t validRows = t.m - row0 < MR ? t.m - row0 : MR;
    uint32_t validCols = t.n - col0 < NR ? t.n - col0 : NR;
    AscendC::Duplicate(al, 0.0f, CSIZE);
    AscendC::Duplicate(bl, 0.0f, CSIZE);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(validCols), validRows * CGEMM_SB_FP32_BYTES, (t.mp - validRows) * CGEMM_SB_FP32_BYTES,
        (MR - (validRows + CGEMM_SB_FP32_PER_BLOCK - 1) / CGEMM_SB_FP32_PER_BLOCK * CGEMM_SB_FP32_PER_BLOCK) /
            CGEMM_SB_FP32_PER_BLOCK,
        0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(al, wg[offset], copy, pad);
    AscendC::DataCopyPad(al[MR * NR], wg[offset + CGEMM_SB_PRODUCT_II * t.cSize], copy, pad);
    AscendC::DataCopyPad(bl, wg[offset + CGEMM_SB_PRODUCT_IR * t.cSize], copy, pad);
    AscendC::DataCopyPad(bl[MR * NR], wg[offset + CGEMM_SB_PRODUCT_RI * t.cSize], copy, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    asc_vf_call<CgemmSbInitializeFullPrefix<NR>>(
        reinterpret_cast<__ubuf__ float*>(al.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(bl.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(cl.GetPhyAddr()));
}

template <uint32_t Components, uint32_t MR, uint32_t NR>
__aicore__ inline void CgemmSbLoadPartialPrefix(
    AscendC::LocalTensor<float> cl, GM_ADDR workspace, const CgemmSbTiling& t, uint32_t batch, uint32_t row0,
    uint32_t col0)
{
    constexpr uint32_t CSIZE = MR * NR * 2;
    AscendC::GlobalTensor<float> wg;
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    uint64_t offset =
        t.cOffset + CGEMM_SB_PARTIAL_PRODUCTS * t.cSize + uint64_t(batch) * t.mp * t.np + uint64_t(col0) * t.mp + row0;
    uint32_t validRows = t.m - row0 < MR ? t.m - row0 : MR;
    uint32_t validCols = t.n - col0 < NR ? t.n - col0 : NR;
    AscendC::Duplicate(cl, 0.0f, CSIZE);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(validCols), validRows * CGEMM_SB_FP32_BYTES, (t.mp - validRows) * CGEMM_SB_FP32_BYTES,
        (MR - (validRows + CGEMM_SB_FP32_PER_BLOCK - 1) / CGEMM_SB_FP32_PER_BLOCK * CGEMM_SB_FP32_PER_BLOCK) /
            CGEMM_SB_FP32_PER_BLOCK,
        0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(cl, wg[offset], copy, pad);
    AscendC::DataCopyPad(cl[MR * NR], wg[offset + CGEMM_SB_PRODUCT_II * t.cSize], copy, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    if constexpr (Components == CGEMM_SB_REFERENCE_REAL)
        AscendC::Sub(cl, cl, cl[MR * NR], MR * NR);
    else
        AscendC::Add(cl, cl, cl[MR * NR], MR * NR);
}

template <uint32_t Components, uint32_t MR, uint32_t NR, uint32_t KR>
__aicore__ inline void CgemmSbAccumulateOrdered(
    AscendC::LocalTensor<float> al, AscendC::LocalTensor<float> bl, AscendC::LocalTensor<float> cl,
    AscendC::GlobalTensor<float> ag, AscendC::GlobalTensor<float> bg, const CgemmSbTiling& t, uint32_t batch,
    uint32_t row0, uint32_t col0)
{
    constexpr uint32_t ASIZE = MR * KR * 2, BSIZE = NR * KR * 2;
    uint32_t prefix = CgemmSbPrefixK(t);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
    CgemmSbLoadOrderedTile<MR, NR, KR>(al, bl, ag, bg, t, batch, row0, col0, prefix, 0);
    for (uint32_t k0 = prefix; k0 < t.k; k0 += KR) {
        uint32_t slot = (k0 - prefix) / KR % 2;
        uint32_t count = t.k - k0 < KR ? t.k - k0 : KR;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
        if (k0 + KR < t.k)
            CgemmSbLoadOrderedTile<MR, NR, KR>(
                al[(1 - slot) * ASIZE], bl[(1 - slot) * BSIZE], ag, bg, t, batch, row0, col0, k0 + KR, 1 - slot);
        constexpr uint32_t NC =
            Components == CGEMM_SB_REFERENCE_BOTH ? ORDERED_FULL_COLUMNS_PER_CALL : ORDERED_PARTIAL_COLUMNS_PER_CALL;
        for (uint32_t j0 = 0; j0 < NR; j0 += NC) {
            auto ap = reinterpret_cast<__ubuf__ float*>(al.GetPhyAddr()) + slot * ASIZE;
            auto bp = reinterpret_cast<__ubuf__ float*>(bl.GetPhyAddr()) + slot * BSIZE + j0 * KR * 2;
            auto cp = reinterpret_cast<__ubuf__ float*>(cl.GetPhyAddr()) +
                      j0 * MR * (Components == CGEMM_SB_REFERENCE_BOTH ? 2 : 1);
            if constexpr (Components == CGEMM_SB_REFERENCE_BOTH)
                asc_vf_call<CgemmSbOrderedMicro<KR>>(ap, bp, cp, count);
            else
                asc_vf_call<CgemmSbOrderedPartialMicro<KR, Components>>(ap, bp, cp, count);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
}

template <uint32_t Components, uint32_t MR, uint32_t NR, uint32_t KR>
__aicore__ inline void CgemmSbStoreOrdered(
    AscendC::GlobalTensor<float> cg, AscendC::LocalTensor<float> cl, const CgemmSbTiling& t, uint32_t batch,
    uint32_t row0, uint32_t col0)
{
    constexpr uint32_t ASIZE = MR * KR * 2, BSIZE = NR * KR * 2, CSIZE = MR * NR * 2;
    auto out = cl;
    if constexpr (Components != CGEMM_SB_REFERENCE_BOTH) {
        AscendC::LocalTensor<float> interleaved(
            AscendC::TPosition::VECOUT, (2 * ASIZE + 2 * BSIZE + MR * NR) * CGEMM_SB_FP32_BYTES, CSIZE);
        asc_vf_call<CgemmSbExpandComponent<Components>>(
            reinterpret_cast<__ubuf__ float*>(cl.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(interleaved.GetPhyAddr()), MR * NR);
        out = interleaved;
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    uint32_t validRows = t.m - row0 < MR ? t.m - row0 : MR;
    uint32_t validCols = t.n - col0 < NR ? t.n - col0 : NR;
    AscendC::DataCopyExtParams cc{
        static_cast<uint16_t>(validCols), validRows * CGEMM_SB_COMPLEX_BYTES,
        (MR - (validRows + CGEMM_SB_COMPLEX_PER_BLOCK - 1) / CGEMM_SB_COMPLEX_PER_BLOCK * CGEMM_SB_COMPLEX_PER_BLOCK) /
            CGEMM_SB_COMPLEX_PER_BLOCK,
        (t.ldc - validRows) * CGEMM_SB_COMPLEX_BYTES, 0};
    AscendC::DataCopyPad(cg[2 * (int64_t(batch) * t.strideC + int64_t(col0) * t.ldc + row0)], out, cc);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}

template <uint32_t Components = CGEMM_SB_REFERENCE_BOTH>
__aicore__ inline void CgemmSbOrderedSimdNN(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling& t, uint32_t core, uint32_t cores)
{
    constexpr uint32_t MR = ORDERED_TILE_ROWS, NR = ORDERED_TILE_COLUMNS, KR = ORDERED_TILE_K;
    constexpr uint32_t ASIZE = MR * KR * 2, BSIZE = NR * KR * 2, CSIZE = MR * NR * 2;
    AscendC::LocalTensor<float> al(AscendC::TPosition::VECIN, 0, 2 * ASIZE);
    AscendC::LocalTensor<float> bl(AscendC::TPosition::VECIN, 2 * ASIZE * CGEMM_SB_FP32_BYTES, 2 * BSIZE);
    AscendC::LocalTensor<float> cl(AscendC::TPosition::VECOUT, (2 * ASIZE + 2 * BSIZE) * CGEMM_SB_FP32_BYTES, CSIZE);
    AscendC::GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    uint32_t rows = (t.m + MR - 1) / MR, cols = (t.n + NR - 1) / NR;
    uint64_t tiles = uint64_t(rows) * cols;
    for (uint64_t tile = core; tile < tiles * t.batch; tile += cores) {
        uint32_t batch = tile / tiles, row0 = tile % rows * MR, col0 = tile % tiles / rows * NR;
        if constexpr (Components == CGEMM_SB_REFERENCE_BOTH)
            CgemmSbLoadFullPrefix<MR, NR>(al, bl, cl, workspace, t, batch, row0, col0);
        else
            CgemmSbLoadPartialPrefix<Components, MR, NR>(cl, workspace, t, batch, row0, col0);
        CgemmSbAccumulateOrdered<Components, MR, NR, KR>(al, bl, cl, ag, bg, t, batch, row0, col0);
        CgemmSbStoreOrdered<Components, MR, NR, KR>(cg, cl, t, batch, row0, col0);
    }
}

} // namespace

extern "C" __global__ __aicore__ void cgemm_sb_cube_kernel(GM_ADDR workspace, CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CgemmSbRunCube(workspace, t);
}
void cgemm_sb_cube_do(uint32_t blocks, void* stream, GM_ADDR workspace, const CgemmSbTiling& t)
{
    cgemm_sb_cube_kernel<<<blocks, nullptr, stream>>>(workspace, t);
}

// Hybrid Cube prefix and ordered Vector suffix kernel entry.
extern "C" __global__ __aicore__ void cgemm_sb_hybrid_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        CgemmSbRunCube<true, true>(ws, t);
        CrossCoreSetFlag<0, PIPE_FIX>(CGEMM_SB_PREFIX_BARRIER);
        CrossCoreWaitFlag<0>(CGEMM_SB_PREFIX_BARRIER);
        CrossCoreSetFlag<2, PIPE_FIX>(CGEMM_SB_PREFIX_READY);
        CgemmSbRunCube<true>(ws, t);
        CrossCoreSetFlag<0, PIPE_FIX>(CGEMM_SB_CUBE_BARRIER);
        CrossCoreWaitFlag<0>(CGEMM_SB_CUBE_BARRIER);
        CrossCoreSetFlag<2, PIPE_FIX>(CGEMM_SB_CUBE_DONE);
    }
    if ASCEND_IS_AIV {
        uint32_t mask = CgemmSbReferenceMask(reinterpret_cast<__gm__ float*>(ws) + t.flagOffset, t);
        uint32_t core = GetBlockIdx(), cores = 2 * GetBlockNum();
        bool partial = CgemmSbPartialReference(mask, t);
        CrossCoreWaitFlag<2>(CGEMM_SB_PREFIX_READY);
        CgemmSbTiling normalized = t;
        GM_ADDR orderedA = a;
        GM_ADDR orderedB = b;
        if (t.normalizeOrdered) {
            orderedA = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ float*>(ws) + t.orderedAOffset);
            orderedB = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ float*>(ws) + t.orderedBOffset);
            normalized.lda = t.orderedM;
            normalized.ldb = t.orderedK;
            normalized.strideA = t.strideA == 0 ? 0 : int64_t(t.orderedM) * t.orderedK;
            normalized.strideB = t.strideB == 0 ? 0 : int64_t(t.orderedK) * t.orderedN;
            normalized.transA = normalized.transB = ACLBLAS_OP_N;
        }
        if (partial) {
            if (mask == CGEMM_SB_REFERENCE_REAL)
                CgemmSbOrderedSimdNN<CGEMM_SB_REFERENCE_REAL>(orderedA, orderedB, c, ws, normalized, core, cores);
            else
                CgemmSbOrderedSimdNN<CGEMM_SB_REFERENCE_IMAG>(orderedA, orderedB, c, ws, normalized, core, cores);
        } else if (mask != CGEMM_SB_REFERENCE_NONE) {
            if (mask != CGEMM_SB_REFERENCE_EXCEPTIONAL)
                CgemmSbOrderedSimdNN<CGEMM_SB_REFERENCE_BOTH>(orderedA, orderedB, c, ws, normalized, core, cores);
        }
        CrossCoreWaitFlag<2>(CGEMM_SB_CUBE_DONE);
    }
}

void cgemm_sb_hybrid_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t)
{
    cgemm_sb_hybrid_kernel<<<blocks, nullptr, stream>>>(a, b, c, ws, t);
}
