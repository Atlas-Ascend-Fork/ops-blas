/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/** Shared DAV_2201 vector layout helpers for the reachable Ctrsv paths. */
#include "ctrsv_kernel.h"
#include "ctrsv_kernel_utils.h"
#include "ctrsv_kernel_layout.h"
#include "ctrsv_generic_kernel.h"
#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/memory/mem.h"
#include "cann_ops_blas_common.h"

using namespace AscendC;
using ctrsv_layout::CtrsvIndexRamp64;
using ctrsv_layout::CtrsvTransposePlane;
using ctrsv_layout::MinU32;
using ctrsv_layout::SplitComplex;

/** Small solves keep each scalar in a full UB block, eliminating pivot broadcasts. */
namespace {
template <bool TRANSPOSE, bool CONJUGATE, bool UNIT, bool MICRO = false, uint32_t STATIC_N = 0U, uint32_t WIDTH = 16U>
class CtrsvTinyReplicatedKernel {
    static constexpr uint32_t PLANE = WIDTH * WIDTH;
    static constexpr uint32_t ROW = WIDTH * 8U;
    static constexpr uint32_t MATRIX = PLANE * 8U;
    static constexpr uint32_t FLOATS = 4U * PLANE + 2U * MATRIX + 11U * ROW + 64U;
    static_assert(WIDTH == 16U || WIDTH == 32U, "Unsupported replicated width");
    static_assert(FLOATS * sizeof(float) < 96U * 1024U, "Tiny replicated UB overflow");
    AsdopsBuffer<ArchType::ASCEND_V220> buffer_;
    GlobalTensor<float> matrix_, vector_;
    LocalTensor<float> aos_, compactReal_, compactImag_, ar_, ai_;
    LocalTensor<float> xr_, xi_, dr_, di_, norm_, pr_, pi_, qr_, qi_, out_, loadX_;
    LocalTensor<uint32_t> offsets_;
    uint32_t n_;
    bool upper_;
    bool orderedBoundary_ = false;

    __aicore__ inline uint32_t LogicalN() const { return STATIC_N == 0U ? n_ : STATIC_N; }
    // Included in CtrsvTinyReplicatedKernel: n<=16 fuses AoS split, transpose,
    // and eight-lane replication. Only real input rows are gathered.
    __aicore__ inline void MicroRowOffsets(uint32_t stride)
    {
        auto indices = offsets_.template ReinterpretCast<int32_t>();
        Duplicate(indices, int32_t{0}, 64U);
        PIPE_BARRIER(V);
        uint64_t rowBit0[2] = {0xff00ff00ff00ff00ULL, 0U};
        uint64_t rowBit1[2] = {0xffff0000ffff0000ULL, 0U};
        uint64_t rowBit2[2] = {0xffffffff00000000ULL, 0U};
        Adds(indices, indices, static_cast<int32_t>(stride), rowBit0, 1U, {1U, 1U, 8U, 8U});
        PIPE_BARRIER(V);
        Adds(indices, indices, static_cast<int32_t>(2U * stride), rowBit1, 1U, {1U, 1U, 8U, 8U});
        PIPE_BARRIER(V);
        Adds(indices, indices, static_cast<int32_t>(4U * stride), rowBit2, 1U, {1U, 1U, 8U, 8U});
        PIPE_BARRIER(V);
    }

    __aicore__ inline void MicroGather(LocalTensor<float> dst, LocalTensor<float> src, uint32_t base, uint32_t stride)
    {
        for (uint32_t row = 0; row < LogicalN(); row += 8U) {
            Gather(dst[row * 8U], src, offsets_, base + row * stride, MinU32(8U, LogicalN() - row) * 8U);
        }
    }

    __aicore__ inline void ConjugateMicro(uint32_t logicalN, uint32_t padded)
    {
        if constexpr (CONJUGATE) {
            // Conjugate the compact AoS input once, before matrix and diagonal
            // gathers replicate it. Even (real) lanes remain untouched.
            uint64_t imaginaryLanes[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
            const uint32_t fullRepeats = logicalN * padded * 2U / 64U;
            const uint32_t tail = logicalN * padded * 2U % 64U;
            if (fullRepeats != 0U) {
                Muls(aos_, aos_, -1.0f, imaginaryLanes, static_cast<uint8_t>(fullRepeats), {1U, 1U, 8U, 8U});
            }
            if (tail != 0U) {
                uint64_t tailMask[2] = {0xaaaaaaaaaaaaaaaaULL & ((1ULL << tail) - 1ULL), 0U};
                Muls(aos_[fullRepeats * 64U], aos_[fullRepeats * 64U], -1.0f, tailMask, 1U, {1U, 1U, 8U, 8U});
            }
            PIPE_BARRIER(V);
        }
    }

    __aicore__ inline void InitMicroDiagonal(uint32_t logicalN, uint32_t padded)
    {
        if constexpr (!UNIT) {
            const uint32_t diagStride = (padded + 1U) * 8U;
            if constexpr (STATIC_N == 8U) {
                Muls(
                    offsets_.template ReinterpretCast<int32_t>(), compactReal_.template ReinterpretCast<int32_t>(),
                    int32_t{9}, 64U);
                PIPE_BARRIER(V);
            } else {
                MicroRowOffsets(diagStride);
            }
            MicroGather(dr_, aos_, 0U, diagStride);
            MicroGather(di_, aos_, 4U, diagStride);
            PIPE_BARRIER(V);
            Mul(pr_, dr_, dr_, logicalN * 8U);
            Mul(pi_, di_, di_, logicalN * 8U);
            PIPE_BARRIER(V);
            Add(norm_, pr_, pi_, logicalN * 8U);
            PIPE_BARRIER(V);
        }
    }

    __aicore__ inline void LoadMicroOperands(uint32_t logicalN, uint32_t padded)
    {
        DataCopyPad(
            aos_, matrix_, DataCopyExtParams{static_cast<uint16_t>(logicalN), logicalN * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((padded - logicalN) * 2U), 0.0f});
        DataCopyPad(
            loadX_, vector_, DataCopyExtParams{1U, logicalN * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((padded - logicalN) * 2U), 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        orderedBoundary_ = ctrsv_layout::NeedsOrderedInput<UNIT>(aos_, loadX_, logicalN, padded, upper_ != TRANSPOSE);
    }

    __aicore__ inline void InitMicro()
    {
        if constexpr ((STATIC_N == 8U || STATIC_N == 16U) && TRANSPOSE) {
            InitPackedMicro();
            return;
        }
        const uint32_t logicalN = LogicalN();
        // MTE padding fully defines each aligned physical column; unlike
        // the 16x16 path, no Vector clear or pre-DMA dependency is needed.
        const uint32_t padded = (logicalN + 3U) & ~3U;
        LoadMicroOperands(logicalN, padded);
        if (orderedBoundary_)
            return;
        ConjugateMicro(logicalN, padded);
        const uint32_t stride = TRANSPOSE ? padded * 8U : 8U;
        if constexpr (STATIC_N == 8U) {
            MicroRowOffsets(8U);
            MicroGather(xr_, loadX_, 0U, 8U);
            MicroGather(xi_, loadX_, 4U, 8U);
            auto baseOffsets = compactReal_.template ReinterpretCast<int32_t>();
            Adds(baseOffsets, offsets_.template ReinterpretCast<int32_t>(), int32_t{0}, 64U);
            PIPE_BARRIER(V);
            if constexpr (TRANSPOSE) {
                Muls(offsets_.template ReinterpretCast<int32_t>(), baseOffsets, int32_t{8}, 64U);
                PIPE_BARRIER(V);
            }
        } else {
            MicroRowOffsets(stride);
        }
        for (uint32_t col = 0; col < logicalN; ++col) {
            const uint32_t base = col * (TRANSPOSE ? 8U : padded * 8U);
            MicroGather(ar_[col * ROW], aos_, base, stride);
            MicroGather(ai_[col * ROW], aos_, base + 4U, stride);
        }
        PIPE_BARRIER(V);
        if constexpr (STATIC_N != 8U) {
            MicroRowOffsets(8U);
            MicroGather(xr_, loadX_, 0U, 8U);
            MicroGather(xi_, loadX_, 4U, 8U);
            PIPE_BARRIER(V);
        }
        InitMicroDiagonal(logicalN, padded);
        // The common output gather reads 16 rows, so define inactive RHS only.
        if (logicalN < WIDTH) {
            Duplicate(xr_[logicalN * 8U], 0.0f, ROW - logicalN * 8U);
            Duplicate(xi_[logicalN * 8U], 0.0f, ROW - logicalN * 8U);
        }
        PIPE_BARRIER(V);
    }

    // Included in CtrsvTinyReplicatedKernel. AoS transpose and deinterleave share
    // one NCHWCONV issue; the resulting effective columns feed batched Brcb.
    template <uint32_t STRIDE, uint32_t BITS>
    __aicore__ inline void PackedIndexRamp(LocalTensor<int32_t> indices)
    {
        static_assert(BITS == 3U || BITS == 4U);
        Duplicate(indices, int32_t{0}, 64U);
        PIPE_BARRIER(V);
        constexpr uint64_t masks[] = {
            0xaaaaaaaaaaaaaaaaULL, 0xccccccccccccccccULL, 0xf0f0f0f0f0f0f0f0ULL, 0xff00ff00ff00ff00ULL};
#pragma unroll
        for (uint32_t bit = 0; bit < BITS; ++bit) {
            uint64_t mask[2] = {masks[bit], 0U};
            Adds(
                indices, indices, static_cast<int32_t>((1U << bit) * STRIDE), mask, 1U,
                UnaryRepeatParams{1U, 1U, 8U, 8U});
            PIPE_BARRIER(V);
        }
    }

    __aicore__ inline void InitPackedDiagonal()
    {
        constexpr uint32_t n = STATIC_N;

        if constexpr (!UNIT) {
            if constexpr (n == 8U) {
                PackedIndexRamp<(WIDTH + 1U) * 4U, 3U>(offsets_.template ReinterpretCast<int32_t>());
            } else {
                CtrsvIndexRamp64<(WIDTH + 1U) * 4U>(offsets_.template ReinterpretCast<int32_t>());
            }
            Gather(pr_, compactReal_, offsets_, 0U, n);
            Gather(pi_, compactImag_, offsets_, 0U, n);
            PIPE_BARRIER(V);
            Brcb(dr_, pr_, n / 8U, {1U, 8U});
            Brcb(di_, pi_, n / 8U, {1U, 8U});
            PIPE_BARRIER(V);
            Mul(pr_, dr_, dr_, n * 8U);
            Mul(pi_, di_, di_, n * 8U);
            PIPE_BARRIER(V);
            Add(norm_, pr_, pi_, n * 8U);
            PIPE_BARRIER(V);
        }
    }

    __aicore__ inline void InitPackedMicro()
    {
        static_assert(TRANSPOSE && (STATIC_N == 8U || STATIC_N == 16U));
        constexpr uint32_t n = STATIC_N;
        DataCopy(aos_, matrix_, 2U * n * n);
        DataCopy(loadX_, vector_, 2U * n);
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        orderedBoundary_ = ctrsv_layout::NeedsOrderedInput<UNIT>(aos_, loadX_, n, n, upper_ != TRANSPOSE);
        if (orderedBoundary_)
            return;

        uint64_t sources[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t targets[NCHW_CONV_ADDR_LIST_SIZE];
#pragma unroll
        for (uint32_t lane = 0; lane < 16U; ++lane) {
            // n=8 duplicates source columns only into inactive output rows.
            sources[lane] = reinterpret_cast<uint64_t>(aos_[(lane % n) * 2U * n].GetPhyAddr());
        }
#pragma unroll
        for (uint32_t row = 0; row < 4U; ++row) {
            targets[4U * row] = reinterpret_cast<uint64_t>(compactReal_[row * WIDTH].GetPhyAddr());
            targets[4U * row + 1U] = reinterpret_cast<uint64_t>(compactReal_[row * WIDTH + 8U].GetPhyAddr());
            targets[4U * row + 2U] = reinterpret_cast<uint64_t>(compactImag_[row * WIDTH].GetPhyAddr());
            targets[4U * row + 3U] = reinterpret_cast<uint64_t>(compactImag_[row * WIDTH + 8U].GetPhyAddr());
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams(false, false, n / 4U, 8U, 1U));
        // SplitComplex reads a complete 64-float issue even for a short RHS.
        Duplicate(loadX_[2U * n], 0.0f, 64U - 2U * n);
        PIPE_BARRIER(V);
        SplitComplex(pr_, pi_, loadX_, 32U);
        if constexpr (CONJUGATE) {
            Muls(compactImag_, compactImag_, -1.0f, n * WIDTH);
            PIPE_BARRIER(V);
        }
        Brcb(ar_, compactReal_, n * WIDTH / 8U, {1U, 8U});
        Brcb(ai_, compactImag_, n * WIDTH / 8U, {1U, 8U});
        Brcb(xr_, pr_, WIDTH / 8U, {1U, 8U});
        Brcb(xi_, pi_, WIDTH / 8U, {1U, 8U});
        PIPE_BARRIER(V);
        InitPackedDiagonal();
    }

    // Included in CtrsvTinyReplicatedKernel. Counter=16 activates two UB blocks.
    __aicore__ inline void PreparePairedPivotNumerator(uint32_t x)
    {
        static_assert(ROW == 128U, "Paired products require 128-float planes");
        // First issue: pr=xr*dr, qr=xi*dr. Second: pi=xr*di, qi=xi*di.
        // The imaginary products exchange scratch slots, not arithmetic order.
        const BinaryRepeatParams products{32U, 16U, 0U, 8U, 8U, 0U};
        const BinaryRepeatParams linear{1U, 1U, 1U, 8U, 8U, 8U};
        AscendC::SetVectorMask<float, MaskMode::COUNTER>(16U);
        Mul<float, false>(pr_, xr_[x], dr_[x], MASK_PLACEHOLDER, 1U, products);
        Mul<float, false>(pi_, xr_[x], di_[x], MASK_PLACEHOLDER, 1U, products);
        PIPE_BARRIER(V);
        AscendC::SetVectorMask<float, MaskMode::COUNTER>(8U);
        Add<float, false>(pr_, pr_, qi_, MASK_PLACEHOLDER, 1U, linear);
        Sub<float, false>(qr_, qr_, pi_, MASK_PLACEHOLDER, 1U, linear);
        PIPE_BARRIER(V);
    }

    __aicore__ inline void DividePairedPivot(uint32_t x)
    {
        static_assert(ROW == 128U, "Paired quotient requires 128-float planes");
        // Block 0: xr[x..x+7] = pr[0..7] / norm[x..x+7].
        // Block 1: xi[x..x+7] = qr[0..7] / norm[x..x+7].
        // xi-xr=ROW, qr-pr=2*ROW; strides are measured in eight floats.
        const BinaryRepeatParams paired{16U, 32U, 0U, 8U, 8U, 0U};
        AscendC::SetVectorMask<float, MaskMode::COUNTER>(16U);
        Div<float, false>(xr_[x], pr_, norm_[x], MASK_PLACEHOLDER, 1U, paired);
    }

    __aicore__ inline void TransposePlane(LocalTensor<float> dst, LocalTensor<float> src)
    {
        CtrsvTransposePlane<WIDTH>(dst, src);
    }

public:
    __aicore__ inline void LoadResidentSystem()
    {
        Duplicate(aos_, 0.0f, 2U * PLANE);
        Duplicate(loadX_, 0.0f, ROW);
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        const uint32_t padded = (n_ + 3U) & ~3U;
        DataCopyPad(
            aos_, matrix_, DataCopyExtParams{static_cast<uint16_t>(n_), n_ * 8U, 0U, (WIDTH - padded) / 4U, 0U},
            DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((padded - n_) * 2U), 0.0f});
        DataCopyPad(
            loadX_, vector_, DataCopyExtParams{1U, n_ * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((padded - n_) * 2U), 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        orderedBoundary_ = ctrsv_layout::NeedsOrderedInput<UNIT>(aos_, loadX_, n_, WIDTH, upper_ != TRANSPOSE);
        if (orderedBoundary_)
            return;
        SplitComplex(compactReal_, compactImag_, aos_, PLANE);
        SplitComplex(pr_, pi_, loadX_, 32U);
        if constexpr (CONJUGATE) {
            Muls(compactImag_, compactImag_, -1.0f, PLANE);
            PIPE_BARRIER(V);
        }
        if constexpr (TRANSPOSE) {
            TransposePlane(aos_, compactReal_);
            TransposePlane(aos_[PLANE], compactImag_);
            PIPE_BARRIER(V);
            // Both compact inputs are dead. Consume the transposed storage directly.
            compactReal_ = aos_;
            compactImag_ = aos_[PLANE];
        }
        Brcb(ar_, compactReal_, PLANE / 8U, {1U, 8U});
        Brcb(ai_, compactImag_, PLANE / 8U, {1U, 8U});
        Brcb(xr_, pr_, WIDTH / 8U, {1U, 8U});
        Brcb(xi_, pi_, WIDTH / 8U, {1U, 8U});
        PIPE_BARRIER(V);
        PrepareResidentDiagonal();
    }

    __aicore__ inline void PrepareResidentDiagonal()
    {
        if constexpr (UNIT)
            return;
        CtrsvIndexRamp64<(WIDTH + 1U) * 4U>(offsets_.template ReinterpretCast<int32_t>());
        Gather(pr_, compactReal_, offsets_, 0U, WIDTH);
        Gather(pi_, compactImag_, offsets_, 0U, WIDTH);
        PIPE_BARRIER(V);
        Brcb(dr_, pr_, WIDTH / 8U, {1U, 8U});
        Brcb(di_, pi_, WIDTH / 8U, {1U, 8U});
        PIPE_BARRIER(V);
        Mul(pr_, dr_, dr_, ROW);
        Mul(pi_, di_, di_, ROW);
        PIPE_BARRIER(V);
        Add(norm_, pr_, pi_, ROW);
        PIPE_BARRIER(V);
    }

    __aicore__ inline void Init(GM_ADDR matrix, GM_ADDR vector, uint32_t n, int32_t uplo)
    {
        n_ = n;
        upper_ = (uplo == ACLBLAS_UPPER) != TRANSPOSE;
        matrix_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrix));
        vector_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vector));
        aos_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        compactReal_ = aos_[2U * PLANE];
        compactImag_ = compactReal_[PLANE];
        ar_ = compactImag_[PLANE];
        ai_ = ar_[MATRIX];
        xr_ = ai_[MATRIX];
        xi_ = xr_[ROW];
        dr_ = xi_[ROW];
        di_ = dr_[ROW];
        norm_ = di_[ROW];
        pr_ = norm_[ROW];
        pi_ = pr_[ROW];
        qr_ = pi_[ROW];
        qi_ = qr_[ROW];
        out_ = qi_[ROW];
        loadX_ = out_[ROW];
        offsets_ = loadX_[ROW].template ReinterpretCast<uint32_t>();
        if constexpr (MICRO) {
            InitMicro();
            return;
        }
        LoadResidentSystem();
    }

    __aicore__ inline bool NeedsOrdered() const { return orderedBoundary_; }

    __aicore__ inline void SolveReplicatedPivot(uint32_t x)
    {
        const BinaryRepeatParams linear{1U, 1U, 1U, 8U, 8U, 8U};

        if constexpr (!UNIT) {
            if constexpr (STATIC_N == 8U && TRANSPOSE) {
                PreparePairedPivotNumerator(x);
            } else {
                AscendC::SetVectorMask<float, MaskMode::COUNTER>(8U);
                Mul<float, false>(pr_, xr_[x], dr_[x], MASK_PLACEHOLDER, 1U, linear);
                Mul<float, false>(pi_, xi_[x], di_[x], MASK_PLACEHOLDER, 1U, linear);
                Mul<float, false>(qr_, xi_[x], dr_[x], MASK_PLACEHOLDER, 1U, linear);
                Mul<float, false>(qi_, xr_[x], di_[x], MASK_PLACEHOLDER, 1U, linear);
                PIPE_BARRIER(V);
                Add<float, false>(pr_, pr_, pi_, MASK_PLACEHOLDER, 1U, linear);
                Sub<float, false>(qr_, qr_, qi_, MASK_PLACEHOLDER, 1U, linear);
                PIPE_BARRIER(V);
            }
            if constexpr (STATIC_N == 16U || (STATIC_N == 8U && TRANSPOSE)) {
                DividePairedPivot(x);
            } else {
                Div<float, false>(xr_[x], pr_, norm_[x], MASK_PLACEHOLDER, 1U, linear);
                Div<float, false>(xi_[x], qr_, norm_[x], MASK_PLACEHOLDER, 1U, linear);
            }
            PIPE_BARRIER(V);
        }
    }

    __aicore__ inline void Process()
    {
        const BinaryRepeatParams linear{1U, 1U, 1U, 8U, 8U, 8U};
        const BinaryRepeatParams pivotBroadcast{1U, 1U, 0U, 8U, 8U, 0U};
        SetMaskCount();
        for (uint32_t ordinal = 0; ordinal < LogicalN(); ++ordinal) {
            const uint32_t pivot = upper_ ? LogicalN() - ordinal - 1U : ordinal;
            const uint32_t x = pivot * 8U;
            SolveReplicatedPivot(x);
            const uint32_t first = upper_ ? 0U : pivot + 1U;
            const uint32_t count = upper_ ? pivot : LogicalN() - first;
            if (count == 0U)
                continue;
            const uint32_t start = first * 8U;
            const uint32_t a = pivot * ROW + start;
            AscendC::SetVectorMask<float, MaskMode::COUNTER>(count * 8U);
            // Each pivot already occupies an identical 8-lane block. Zero source
            // strides broadcast it directly; no per-pivot Brcb or scalar load.
            Mul<float, false>(pr_, ar_[a], xr_[x], MASK_PLACEHOLDER, 1U, pivotBroadcast);
            Mul<float, false>(pi_, ai_[a], xi_[x], MASK_PLACEHOLDER, 1U, pivotBroadcast);
            Mul<float, false>(qr_, ar_[a], xi_[x], MASK_PLACEHOLDER, 1U, pivotBroadcast);
            Mul<float, false>(qi_, ai_[a], xr_[x], MASK_PLACEHOLDER, 1U, pivotBroadcast);
            PIPE_BARRIER(V);
            Sub<float, false>(pr_, pr_, pi_, MASK_PLACEHOLDER, 1U, linear);
            Add<float, false>(qr_, qr_, qi_, MASK_PLACEHOLDER, 1U, linear);
            PIPE_BARRIER(V);
            Sub<float, false>(xr_[start], xr_[start], pr_, MASK_PLACEHOLDER, 1U, linear);
            Sub<float, false>(xi_[start], xi_[start], qr_, MASK_PLACEHOLDER, 1U, linear);
            PIPE_BARRIER(V);
        }
        SetMaskNorm();
        ResetMask();
        Store();
    }

    __aicore__ inline void Store()
    {
        // Interleave the first replica from the adjacent real/imaginary planes.
        auto interleave = offsets_.template ReinterpretCast<int32_t>();
        if constexpr (STATIC_N == 8U && TRANSPOSE) {
            PackedIndexRamp<16U, 4U>(interleave);
        } else {
            CtrsvIndexRamp64<16U>(interleave);
        }
        uint64_t oddLanes[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
        Adds(interleave, interleave, int32_t{ROW * sizeof(float) - 16U}, oddLanes, 1U, {1U, 1U, 8U, 8U});
        PIPE_BARRIER(V);
        constexpr uint32_t outputCount = STATIC_N == 8U && TRANSPOSE ? 16U : 2U * WIDTH;
        Gather(out_, xr_, offsets_, 0U, outputCount);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        DataCopyPad(vector_, out_, DataCopyExtParams{1U, LogicalN() * 8U, 0U, 0U, 0U});
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
    }
};
} // namespace

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT, typename TinyKernel>
__aicore__ inline void RunTinyKernel(TinyKernel& op, GM_ADDR matrix, GM_ADDR vector, uint32_t n, int32_t uplo)
{
    if (op.NeedsOrdered()) {
        const CtrsvTilingData td{
            static_cast<int32_t>(n),
            static_cast<int32_t>(n),
            1,
            uplo,
            TRANSPOSE ? (CONJUGATE ? ACLBLAS_OP_C : ACLBLAS_OP_T) : ACLBLAS_OP_N,
            UNIT ? ACLBLAS_UNIT : ACLBLAS_NON_UNIT};
        ctrsv_generic::Kernel<TRANSPOSE, CONJUGATE, UNIT> fallback;
        fallback.Init(matrix, vector, td);
        fallback.Process();
    } else {
        op.Process();
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT, uint32_t WIDTH = 16U, uint32_t STATIC_N = 0U>
__global__ __aicore__ __vector__ void ctrsv_kernel_tiny_replicated(
    GM_ADDR matrix, GM_ADDR vector, uint32_t n, int32_t uplo)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIC {
        return;
    }
    CtrsvTinyReplicatedKernel<TRANSPOSE, CONJUGATE, UNIT, false, STATIC_N, WIDTH> op;
    op.Init(matrix, vector, n, uplo);
    RunTinyKernel<TRANSPOSE, CONJUGATE, UNIT>(op, matrix, vector, n, uplo);
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT, uint32_t STATIC_N = 0U>
__global__ __aicore__ __vector__ void ctrsv_kernel_tiny_micro(GM_ADDR matrix, GM_ADDR vector, uint32_t n, int32_t uplo)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIC {
        return;
    }
    CtrsvTinyReplicatedKernel<TRANSPOSE, CONJUGATE, UNIT, true, STATIC_N> op;
    op.Init(matrix, vector, n, uplo);
    RunTinyKernel<TRANSPOSE, CONJUGATE, UNIT>(op, matrix, vector, n, uplo);
}

/** Single-owner exact solve with the entire small system resident in UB. */
namespace {
template <uint32_t WIDTH, bool TRANSPOSE, bool CONJUGATE, bool UNIT>
class CtrsvTinyKernel {
    static_assert(WIDTH == 16U || WIDTH == 32U || WIDTH == 64U, "Unsupported tiny tile");
    static constexpr uint32_t PLANE = WIDTH * WIDTH;
    static constexpr uint32_t LANES = 64U;
    static constexpr uint32_t FLOATS = 4U * PLANE + 15U * LANES;
    static_assert(FLOATS * sizeof(float) <= 72U * 1024U, "Tiny UB overflow");
    AsdopsBuffer<ArchType::ASCEND_V220> buffer_;
    GlobalTensor<float> matrix_, vector_;
    LocalTensor<float> aos_, ar_, ai_, xr_, xi_, dr_, di_, norm_;
    LocalTensor<float> br_, bi_, pr_, pi_, qr_, qi_, out_, planar_;
    LocalTensor<uint32_t> offsets_;
    uint32_t n_, lda_;
    bool upper_;
    bool orderedBoundary_ = false;

    __aicore__ inline void TransposePlane(LocalTensor<float> dst, LocalTensor<float> src)
    {
        CtrsvTransposePlane<WIDTH>(dst, src);
        PIPE_BARRIER(V);
    }

public:
    __aicore__ inline void LoadResidentSystem()
    {
        Duplicate(aos_, 0.0f, 2U * PLANE);
        Duplicate(planar_, 0.0f, 2U * LANES);
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        const uint32_t paddedRows = (n_ + 3U) & ~3U;
        DataCopyPad(
            aos_, matrix_,
            DataCopyExtParams{static_cast<uint16_t>(n_), n_ * 8U, (lda_ - n_) * 8U, (WIDTH - paddedRows) / 4U, 0U},
            DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((paddedRows - n_) * 2U), 0.0f});
        DataCopyPad(
            planar_, vector_, DataCopyExtParams{1U, n_ * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((paddedRows - n_) * 2U), 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        orderedBoundary_ = ctrsv_layout::NeedsOrderedInput<UNIT>(aos_, planar_, n_, WIDTH, upper_ != TRANSPOSE);
        if (orderedBoundary_)
            return;
        SplitComplex(ar_, ai_, aos_, PLANE);
        SplitComplex(xr_, xi_, planar_, LANES);
        if constexpr (CONJUGATE) {
            Muls(ai_, ai_, -1.0f, PLANE);
            PIPE_BARRIER(V);
        }
        if constexpr (TRANSPOSE) {
            // Elimination consumes effective columns, unlike row-dot GEMV.
            TransposePlane(aos_, ar_);
            TransposePlane(aos_[PLANE], ai_);
            Adds(ar_, aos_, 0.0f, PLANE);
            Adds(ai_, aos_[PLANE], 0.0f, PLANE);
            PIPE_BARRIER(V);
        }
        if constexpr (!UNIT) {
            CtrsvIndexRamp64<(WIDTH + 1U) * 4U>(offsets_.template ReinterpretCast<int32_t>());
            Gather(dr_, ar_, offsets_, 0U, WIDTH);
            Gather(di_, ai_, offsets_, 0U, WIDTH);
            PIPE_BARRIER(V);
            Mul(pr_, dr_, dr_, WIDTH);
            Mul(pi_, di_, di_, WIDTH);
            PIPE_BARRIER(V);
            Add(norm_, pr_, pi_, WIDTH);
            PIPE_BARRIER(V);
            // The diagonal is immutable. Divide it once in parallel rather
            // than issuing two dependent divisions for every solved pivot.
            Div(dr_, dr_, norm_, WIDTH);
            Div(di_, di_, norm_, WIDTH);
            PIPE_BARRIER(V);
        }
    }

    __aicore__ inline void Init(GM_ADDR matrix, GM_ADDR vector, uint32_t n, int32_t uplo)
    {
        n_ = n;
        lda_ = n;
        upper_ = (uplo == ACLBLAS_UPPER) != TRANSPOSE;
        matrix_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrix));
        vector_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vector));
        aos_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        ar_ = aos_[2U * PLANE];
        ai_ = ar_[PLANE];
        xr_ = ai_[PLANE];
        xi_ = xr_[LANES];
        dr_ = xi_[LANES];
        di_ = dr_[LANES];
        norm_ = di_[LANES];
        br_ = norm_[LANES];
        bi_ = br_[LANES];
        pr_ = bi_[LANES];
        pi_ = pr_[LANES];
        qr_ = pi_[LANES];
        qi_ = qr_[LANES];
        out_ = qi_[LANES];
        planar_ = out_[2U * LANES];
        // planar_ needs 128 floats; offsets share dead output staging during load.
        offsets_ = out_.template ReinterpretCast<uint32_t>();
        LoadResidentSystem();
    }

    __aicore__ inline bool NeedsOrdered() const { return orderedBoundary_; }

    __aicore__ inline void Store()
    {
        // xr_/xi_ are adjacent 64-lane planes. Form the interleave map
        // 0,64,1,65,... without the general Transpose metadata/workspace graph.
        auto interleave = norm_.template ReinterpretCast<int32_t>();
        CtrsvIndexRamp64<2U>(interleave);
        uint64_t oddLanes[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
        Adds(interleave, interleave, int32_t{254}, oddLanes, 1U, UnaryRepeatParams{1U, 1U, 8U, 8U});
        PIPE_BARRIER(V);
        Gather(out_, xr_, interleave.template ReinterpretCast<uint32_t>(), 0U, LANES);
        if constexpr (WIDTH == 64U) {
            Gather(out_[LANES], xr_, interleave.template ReinterpretCast<uint32_t>(), 128U, LANES);
        }
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        DataCopyPad(vector_, out_, DataCopyExtParams{1U, n_ * 8U, 0U, 0U, 0U});
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
    }

    __aicore__ inline void Process()
    {
        const BinaryRepeatParams repeat{1U, 1U, 1U, 8U, 8U, 8U};
        const uint64_t valid = n_ == 64U ? ~0ULL : (1ULL << n_) - 1ULL;
        for (uint32_t ordinal = 0; ordinal < n_; ++ordinal) {
            const uint32_t pivot = upper_ ? n_ - 1U - ordinal : ordinal;
            uint64_t pivotMask[2] = {1ULL << pivot, 0U};
            if constexpr (!UNIT) {
                Mul(pr_, xr_, dr_, pivotMask, 1U, repeat);
                Mul(pi_, xi_, di_, pivotMask, 1U, repeat);
                Mul(qr_, xi_, dr_, pivotMask, 1U, repeat);
                Mul(qi_, xr_, di_, pivotMask, 1U, repeat);
                PIPE_BARRIER(V);
                Add(xr_, pr_, pi_, pivotMask, 1U, repeat);
                Sub(xi_, qr_, qi_, pivotMask, 1U, repeat);
                PIPE_BARRIER(V);
            }
            const uint64_t active =
                upper_ ? (1ULL << pivot) - 1ULL : (pivot == 63U ? 0U : valid & (~0ULL << (pivot + 1U)));
            if (active == 0U)
                continue;
            Brcb(pr_, xr_[pivot & ~7U], 1U, {1U, 8U});
            Brcb(pi_, xi_[pivot & ~7U], 1U, {1U, 8U});
            PIPE_BARRIER(V);
            Brcb(br_, pr_[(pivot & 7U) * 8U], 1U, {1U, 8U});
            Brcb(bi_, pi_[(pivot & 7U) * 8U], 1U, {1U, 8U});
            PIPE_BARRIER(V);
            uint64_t mask[2] = {active, 0U};
            Mul(pr_, ar_[pivot * WIDTH], br_, mask, 1U, repeat);
            Mul(pi_, ai_[pivot * WIDTH], bi_, mask, 1U, repeat);
            Mul(qr_, ar_[pivot * WIDTH], bi_, mask, 1U, repeat);
            Mul(qi_, ai_[pivot * WIDTH], br_, mask, 1U, repeat);
            PIPE_BARRIER(V);
            Sub(pr_, pr_, pi_, mask, 1U, repeat);
            Add(qr_, qr_, qi_, mask, 1U, repeat);
            PIPE_BARRIER(V);
            Sub(xr_, xr_, pr_, mask, 1U, repeat);
            Sub(xi_, xi_, qr_, mask, 1U, repeat);
            PIPE_BARRIER(V);
        }
        Store();
    }
};
} // namespace

template <uint32_t WIDTH, bool TRANSPOSE, bool CONJUGATE, bool UNIT>
__global__ __aicore__ __vector__ void ctrsv_kernel_tiny(GM_ADDR matrix, GM_ADDR vector, uint32_t n, int32_t uplo)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIC {
        return;
    }
    CtrsvTinyKernel<WIDTH, TRANSPOSE, CONJUGATE, UNIT> op;
    op.Init(matrix, vector, n, uplo);
    RunTinyKernel<TRANSPOSE, CONJUGATE, UNIT>(op, matrix, vector, n, uplo);
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT>
void LaunchCtrsvTiny(uint8_t* matrix, uint8_t* vector, uint32_t n, int32_t uplo, void* stream)
{
    if (n == 8U) {
        ctrsv_kernel_tiny_micro<TRANSPOSE, CONJUGATE, UNIT, 8U><<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    } else if (n == 16U) {
        ctrsv_kernel_tiny_micro<TRANSPOSE, CONJUGATE, UNIT, 16U><<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    } else if (n <= 16U) {
        ctrsv_kernel_tiny_micro<TRANSPOSE, CONJUGATE, UNIT><<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    } else if (n == 32U) {
        ctrsv_kernel_tiny_replicated<TRANSPOSE, CONJUGATE, UNIT, 32U, 32U>
            <<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    } else if (n < 32U) {
        ctrsv_kernel_tiny_replicated<TRANSPOSE, CONJUGATE, UNIT, 32U><<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    } else {
        ctrsv_kernel_tiny<64U, TRANSPOSE, CONJUGATE, UNIT><<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    }
}

/** Launch only the paths reachable through the public API. */
void ctrsv_tiny_do(
    uint8_t* matrix, uint8_t* vector, uint32_t n, int32_t uplo, int32_t trans, int32_t diag, void* stream)
{
    if (trans == ACLBLAS_OP_N) {
        if (diag == ACLBLAS_UNIT)
            LaunchCtrsvTiny<false, false, true>(matrix, vector, n, uplo, stream);
        else
            LaunchCtrsvTiny<false, false, false>(matrix, vector, n, uplo, stream);
    } else if (trans == ACLBLAS_OP_T) {
        if (diag == ACLBLAS_UNIT)
            LaunchCtrsvTiny<true, false, true>(matrix, vector, n, uplo, stream);
        else
            LaunchCtrsvTiny<true, false, false>(matrix, vector, n, uplo, stream);
    } else {
        if (diag == ACLBLAS_UNIT)
            LaunchCtrsvTiny<true, true, true>(matrix, vector, n, uplo, stream);
        else
            LaunchCtrsvTiny<true, true, false>(matrix, vector, n, uplo, stream);
    }
}
