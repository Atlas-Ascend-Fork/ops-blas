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
#include "common/compute/simd.h"
#include "common/helper/kernel_utils.h"
#include "common/memory/mem.h"
#include "cann_ops_blas_common.h"

using namespace AscendC;
using ctrsv_layout::CtrsvIndexRamp64;
using ctrsv_layout::SplitComplex;

#define CTRSV_GENERIC_ENTRY(NAME, TRANSPOSE_VALUE, CONJUGATE_VALUE, UNIT_VALUE)                  \
    __global__ __aicore__ __vector__ void NAME(GM_ADDR matrix, GM_ADDR vector, GM_ADDR tiling)   \
    {                                                                                            \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                          \
        if ASCEND_IS_AIC {                                                                       \
            return;                                                                              \
        }                                                                                        \
        const auto td = reinterpret_cast<__gm__ const CtrsvTilingData*>(tiling);                 \
        const CtrsvTilingData metadata{td->n, td->lda, td->incx, td->uplo, td->trans, td->diag}; \
        ctrsv_generic::Kernel<TRANSPOSE_VALUE, CONJUGATE_VALUE, UNIT_VALUE> op;                  \
        op.Init(matrix, vector, metadata);                                                       \
        op.Process();                                                                            \
    }

CTRSV_GENERIC_ENTRY(ctrsv_kernel_generic_n, false, false, false)
CTRSV_GENERIC_ENTRY(ctrsv_kernel_generic_n_unit, false, false, true)
CTRSV_GENERIC_ENTRY(ctrsv_kernel_generic_t, true, false, false)
CTRSV_GENERIC_ENTRY(ctrsv_kernel_generic_t_unit, true, false, true)
CTRSV_GENERIC_ENTRY(ctrsv_kernel_generic_c, true, true, false)
CTRSV_GENERIC_ENTRY(ctrsv_kernel_generic_c_unit, true, true, true)

#undef CTRSV_GENERIC_ENTRY

/**
 * Block-wavefront path for aligned OP_T/OP_C non-unit solves on DAV_2201.
 *
 * One owner solves a 64-row diagonal block and publishes it through the shared
 * state vector.  All dependent blocks then consume that block in parallel.
 * Consequently every off-diagonal 64x64 panel is transferred and evaluated
 * exactly once.  This is an exact blocked triangular solve; unlike truncated
 * block-Jacobi it does not depend on diagonal dominance or iteration count.
 */
namespace {

constexpr uint32_t JACOBI_BLOCK = 64U;
constexpr uint32_t JACOBI_COMPLEX = 2U;
constexpr uint32_t JACOBI_BLOCK_FLOATS = JACOBI_BLOCK * JACOBI_BLOCK;
constexpr uint32_t JACOBI_BLOCK_AOS_FLOATS = JACOBI_BLOCK_FLOATS * JACOBI_COMPLEX;
constexpr uint32_t JACOBI_PRODUCT_FLOATS = JACOBI_BLOCK_FLOATS;
constexpr uint32_t JACOBI_TRANSPOSE_TMP_BYTES = 2048U;
// A DAV_2201 core can own more than one 64-row panel when panelCount exceeds
// the available AIVs (for example, 64 panels on the 4K case and 40 AIVs).
// Keep the complete panel state in UB so phase updates never overwrite the
// accumulator belonging to a different panel.  The bound also covers a
// reduced-core runtime without sending panel state through GM every phase.
// With 40 AIVs the 4K shape has 64 panels, so the first 24 AIVs own two
// panels.  The host keeps round-robin ownership; two resident RHS slots are
// sufficient for every supported shape and avoid a GM read/write per phase.
constexpr uint32_t JACOBI_MAX_PANELS_PER_CORE = 4U;

__aicore__ inline uint32_t JacobiMin(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG = false>
class CtrsvJacobiTcKernel {
public:
    template <bool PADDED = false>
    __aicore__ inline void PrepareGroupedInverse();
    template <bool PADDED>
    __aicore__ inline void BuildPreparedPanelInverse();
    __aicore__ inline void PrepareInverseGeometry();
    template <bool PADDED = false>
    __aicore__ inline void ProcessPreparedPanels();
    __aicore__ inline void LoadPreparedTailTileAsync(uint32_t targetStart, uint32_t sourceStart);
    __aicore__ inline void LoadPreparedTailDiagonal();
    __aicore__ inline void LoadPreparedTailRhs(LocalTensor<float> real, LocalTensor<float> imag);
    __aicore__ inline void PreparePreparedNSource();
    __aicore__ inline void ComputePreparedNUpdate(
        LocalTensor<float> destinationReal, LocalTensor<float> destinationImag);

    __aicore__ inline void InitPrepared(
        GM_ADDR matrix, GM_ADDR vector, GM_ADDR workspace, const CtrsvPreparedArgs& args)
    {
        matrixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrix));
        vectorGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vector));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        n_ = args.n;
        lda_ = args.lda;
        uplo_ = args.uplo;
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        blockCount_ = args.blocks;
        panelCount_ = (n_ + JACOBI_BLOCK - 1U) / JACOBI_BLOCK;
        blockStart_ = blockIdx_ * JACOBI_BLOCK;
        flagsOffset_ = args.flags / sizeof(float);
        pingOffset_ = args.ping / sizeof(float);
        diagonalRealOffset_ = args.diagonalReal / sizeof(float);
        diagonalImagOffset_ = args.diagonalImag / sizeof(float);
        InitBuffers();
    }

    template <bool PADDED = false>
    __aicore__ inline void ProcessSafePrepared(GM_ADDR matrix, GM_ADDR vector)
    {
        if (NeedsGenericArithmetic()) {
            if (blockIdx_ == 0U) {
                const CtrsvTilingData td{
                    static_cast<int32_t>(n_),
                    static_cast<int32_t>(lda_),
                    1,
                    uplo_,
                    TRANSPOSE ? (CONJUGATE ? ACLBLAS_OP_C : ACLBLAS_OP_T) : ACLBLAS_OP_N,
                    UNIT_DIAG ? ACLBLAS_UNIT : ACLBLAS_NON_UNIT};
                ctrsv_generic::Kernel<TRANSPOSE, CONJUGATE, UNIT_DIAG> fallback;
                fallback.Init(matrix, vector, td);
                fallback.Process();
            }
            return;
        }
        ProcessPreparedPanels<PADDED>();
    }

private:
    struct PreparedState {
        bool upper;
        uint32_t stateN;
        uint32_t localCount;
        bool residentInverse;
        LocalTensor<float> savedReal;
        LocalTensor<float> savedImag;
    };

    __aicore__ inline void NormalizeInverseRows();
    __aicore__ inline void InitializeInverseIdentity(LocalTensor<uint8_t> identity);
    template <bool PADDED>
    __aicore__ inline void InitializePreparedInverses(const PreparedState& state);
    template <bool PADDED>
    __aicore__ inline void InitializePanelRhs();
    template <bool PADDED>
    __aicore__ inline void PrefetchDependentPanel(const PreparedState& state, uint32_t firstOrdinal, uint32_t start);
    __aicore__ inline void PublishPreparedPivot(const PreparedState& state, uint32_t pivot, uint32_t start);
    template <bool PADDED>
    __aicore__ inline void UpdateDependentPanels(const PreparedState& state, uint32_t firstOrdinal, uint32_t start);
    __aicore__ inline void LoadPublishedSource(uint32_t start, uint32_t stateN);
    template <bool PADDED>
    __aicore__ inline void LoadDependentPanel(uint32_t start, bool prefetched);
    __aicore__ inline void UpdateInversePivot(uint32_t row, uint32_t first, uint32_t rows, uint32_t coefficient);
    __aicore__ inline void PrepareSplitPivotBatch(uint32_t ordinal, uint32_t localPivot);
    __aicore__ inline void BuildGroupedInverse();
    template <bool UPPER>
    __aicore__ inline void EliminatePreparedInverse();
    template <bool UPPER>
    __aicore__ inline void EliminateSplitInverse();
    __aicore__ inline void SolveSplitPrepared(
        LocalTensor<float> rhsReal, LocalTensor<float> rhsImag, LocalTensor<float> originalReal,
        LocalTensor<float> originalImag);
    __aicore__ inline void ComputePreparedHalf(
        LocalTensor<float> real, LocalTensor<float> imag, uint32_t target, uint32_t source);
    __aicore__ inline void SyncAllAiv()
    {
        // A single strictly paired mode-0 token keeps all phase barriers on
        // one ordered counter.  Rotating across multiple flag IDs is unsafe
        // on DAV_2201 because CrossCoreSetFlag does not guarantee ordering
        // between different IDs; a later phase can otherwise overtake the
        // wait protecting the current GM state.
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        FftsCrossCoreSync<PIPE_MTE3, 0>(0U);
        WaitFlagDev(0U);
    }

    __aicore__ inline bool InspectPanelInputs(uint32_t start, uint32_t count)
    {
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        const uint64_t diagonal = static_cast<uint64_t>(start) * (lda_ + 1U) * JACOBI_COMPLEX;
        DataCopyPad(
            matrixAos_, matrixGm_[diagonal], DataCopyExtParams{static_cast<uint16_t>(count), 8U, lda_ * 8U, 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, 6U, 0.0f});
        DataCopyPad(
            xAos_, vectorGm_[static_cast<uint64_t>(start) * JACOBI_COMPLEX],
            DataCopyExtParams{1U, count * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((8U - count * 2U % 8U) % 8U), 0.0f});
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        bool exceptional = false;
        for (uint32_t i = 0; i < count; ++i) {
            exceptional |=
                ctrsv_netlib::NeedsFullRangeDiagonal({matrixAos_.GetValue(i * 8U), matrixAos_.GetValue(i * 8U + 1U)});
            exceptional |= ctrsv_netlib::NeedsSoftwareProduct({xAos_.GetValue(i * 2U), xAos_.GetValue(i * 2U + 1U)});
        }
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        return exceptional;
    }

    __aicore__ inline bool NeedsGenericArithmetic()
    {
        bool exceptional = false;
        for (uint32_t panel = blockIdx_; panel < panelCount_; panel += blockCount_) {
            const uint32_t start = panel * JACOBI_BLOCK;
            exceptional |= InspectPanelInputs(start, JacobiMin(JACOBI_BLOCK, n_ - start));
        }
        constexpr uint32_t slotFloats = CTRSV_FLAG_SLOT_BYTES / sizeof(float);
        // Separate 32-byte DMA slots avoid concurrent partial-block stores.
        Duplicate(rowDotReal_, exceptional ? 1.0f : 0.0f, slotFloats);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        DataCopy(workspaceGm_[flagsOffset_ + blockIdx_ * slotFloats], rowDotReal_, slotFloats);
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        SyncAllAiv();
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        DataCopy(matrixAos_, workspaceGm_[flagsOffset_], blockCount_ * slotFloats);
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        for (uint32_t owner = 0; owner < blockCount_; ++owner)
            exceptional |= matrixAos_.GetValue(owner * slotFloats) != 0.0f;
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        return exceptional;
    }

    __aicore__ inline void InitScratchBuffers(uint32_t& offset)
    {
        productReal_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_PRODUCT_FLOATS * sizeof(float);
        productImag_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_PRODUCT_FLOATS * sizeof(float);
        rowDotReal_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK * sizeof(float);
        rowDotImag_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK * sizeof(float);
        xAos_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK_AOS_FLOATS / JACOBI_BLOCK * sizeof(float);
        xReal_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK * sizeof(float);
        xImag_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK * sizeof(float);
        signedX_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK * sizeof(float);
        invReal_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK * sizeof(float);
        invImag_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK * sizeof(float);
        gatherOffsets_ = buffer_.GetBuffer<BufferType::ASCEND_UB, int32_t>(offset);
        offset += JACOBI_BLOCK * sizeof(int32_t);
        planar_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK_AOS_FLOATS / JACOBI_BLOCK * sizeof(float);
        transposeTmp_ = buffer_.GetBuffer<BufferType::ASCEND_UB, uint8_t>(offset);
        offset += JACOBI_TRANSPOSE_TMP_BYTES;
        groupReal_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += 2U * CTRSV_GROUP_ROWS * sizeof(float);
        groupSourceReal_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += 2U * CTRSV_GROUP_ROWS * sizeof(float);
        groupAos_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += 2U * CTRSV_GROUP_ROWS * sizeof(float);
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t offset = 0U;
        matrixAos_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK_AOS_FLOATS * sizeof(float);
        matrixReal_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK_FLOATS * sizeof(float);
        matrixImag_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK_FLOATS * sizeof(float);
        diagonalReal_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK_FLOATS * sizeof(float);
        diagonalImag_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
        offset += JACOBI_BLOCK_FLOATS * sizeof(float);
        for (uint32_t slot = 0U; slot < JACOBI_MAX_PANELS_PER_CORE; ++slot) {
            panelAccReal_[slot] = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
            offset += JACOBI_BLOCK * sizeof(float);
            panelAccImag_[slot] = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(offset);
            offset += JACOBI_BLOCK * sizeof(float);
        }
        InitScratchBuffers(offset);
        static_assert(
            (2U * JACOBI_BLOCK_AOS_FLOATS + 2U * JACOBI_BLOCK_FLOATS + 11U * JACOBI_BLOCK + 2U * JACOBI_PRODUCT_FLOATS +
             2U * JACOBI_MAX_PANELS_PER_CORE * JACOBI_BLOCK) *
                        sizeof(float) +
                    JACOBI_BLOCK * sizeof(int32_t) + JACOBI_TRANSPOSE_TMP_BYTES +
                    6U * CTRSV_GROUP_ROWS * sizeof(float) <=
                160U * 1024U,
            "Ctrsv shared UB layout overlaps the prepared inverse cache");
        CtrsvIndexRamp64<(JACOBI_BLOCK + 1U) * sizeof(float)>(gatherOffsets_);
        SetMaskNorm();
        AscendC::SetVectorMask<float>(0U, static_cast<uint64_t>(-1));
    }

    __aicore__ inline bool EffectiveUpper() const
    {
        return TRANSPOSE ? ((uplo_ == ACLBLAS_UPPER) ^ true) : (uplo_ == ACLBLAS_UPPER);
    }

    __aicore__ inline void LoadInitialPanel(
        LocalTensor<float> destinationReal, LocalTensor<float> destinationImag, uint32_t count)
    {
        LoadInitialPanel(destinationReal, destinationImag, count, blockStart_);
    }

    __aicore__ inline void LoadInitialPanel(
        LocalTensor<float> destinationReal, LocalTensor<float> destinationImag, uint32_t count, uint32_t globalStart)
    {
        const uint64_t offset = static_cast<uint64_t>(globalStart) * JACOBI_COMPLEX;
        DataCopyPad(
            xAos_, vectorGm_[offset],
            DataCopyExtParams{1, static_cast<uint32_t>(count * JACOBI_COMPLEX * sizeof(float)), 0U, 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        // SplitComplex uses a full vector issue and may write padded lanes;
        // the caller supplies a destination with enough tail room.  Most
        // importantly, use the requested count rather than the enclosing
        // 64-row panel so a 16-row slice cannot consume the next slice.
        SplitComplex(destinationReal, destinationImag, xAos_, count);
    }

    __aicore__ inline void LoadDiagonalPanel(uint32_t count)
    {
        const uint64_t matrixOffset =
            (static_cast<uint64_t>(blockStart_) + static_cast<uint64_t>(blockStart_) * lda_) * JACOBI_COMPLEX;
        const uint32_t rowBytes = count * JACOBI_COMPLEX * sizeof(float);
        DataCopyPad(
            matrixAos_, matrixGm_[matrixOffset],
            DataCopyExtParams{
                static_cast<uint16_t>(count), rowBytes,
                static_cast<uint32_t>((lda_ - count) * JACOBI_COMPLEX * sizeof(float)), 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        SplitComplex(matrixReal_, matrixImag_, matrixAos_, count * count);
        if constexpr (CONJUGATE) {
            Muls(matrixImag_, matrixImag_, -1.0f, static_cast<int32_t>(count * count));
            PIPE_BARRIER(V);
        }
        if constexpr (!TRANSPOSE) {
            // OP_N receives column-major source columns.  Convert the dense
            // 64x64 panel once in UB so the vector residual engine consumes
            // contiguous effective rows, exactly like the CGEMV transpose
            // path.  No GM repack or scalar element access is needed.
            TransposePlane(diagonalReal_, matrixReal_, count);
            Adds(matrixReal_, diagonalReal_, 0.0f, static_cast<int32_t>(count * count));
            TransposePlane(diagonalImag_, matrixImag_, count);
            Adds(matrixImag_, diagonalImag_, 0.0f, static_cast<int32_t>(count * count));
            PIPE_BARRIER(V);
        }
    }

    __aicore__ inline void TransposePlane(LocalTensor<float> destination, LocalTensor<float> source, uint32_t count)
    {
        constexpr uint32_t floatsPerBlock = 8U;
        constexpr uint32_t rowChunk = 2U * floatsPerBlock;
        const uint32_t repeatTimes = count / floatsPerBlock;
        for (uint32_t row = 0U; row < count; row += rowChunk) {
            uint64_t sourceList[NCHW_CONV_ADDR_LIST_SIZE];
            uint64_t destinationList[NCHW_CONV_ADDR_LIST_SIZE];
#pragma unroll
            for (uint32_t lane = 0U; lane < NCHW_CONV_ADDR_LIST_SIZE; ++lane) {
                sourceList[lane] = reinterpret_cast<uint64_t>(source[(row + lane) * count].GetPhyAddr());
            }
#pragma unroll
            for (uint32_t lane = 0U; lane < floatsPerBlock; ++lane) {
                destinationList[lane * 2U] = reinterpret_cast<uint64_t>(destination[lane * count + row].GetPhyAddr());
                destinationList[lane * 2U + 1U] =
                    reinterpret_cast<uint64_t>(destination[lane * count + row + floatsPerBlock].GetPhyAddr());
            }
            TransDataTo5HDParams params(
                false, false, static_cast<uint8_t>(repeatTimes), static_cast<uint16_t>(count),
                static_cast<uint16_t>(1U));
            TransDataTo5HD<float>(destinationList, sourceList, params);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ inline void BuildDiagonalInverse(uint32_t count, uint32_t validCount = JACOBI_BLOCK)
    {
        if constexpr (UNIT_DIAG) {
            Duplicate(invReal_, 1.0f, count);
            Duplicate(invImag_, 0.0f, count);
            PIPE_BARRIER(V);
            return;
        }
        Gather(invReal_, matrixReal_, gatherOffsets_.ReinterpretCast<uint32_t>(), 0U, count);
        Gather(invImag_, matrixImag_, gatherOffsets_.ReinterpretCast<uint32_t>(), 0U, count);
        PIPE_BARRIER(V);
        if (validCount < count) {
            // Padded equations are independent identity rows, never 0/0.
            uint64_t paddingMask[2] = {~0ULL << validCount, 0U};
            SetMaskNorm();
            Duplicate<float>(invReal_, 1.0f, paddingMask, 1U, 1U, 8U);
            Duplicate<float>(invImag_, 0.0f, paddingMask, 1U, 1U, 8U);
            PIPE_BARRIER(V);
            AscendC::SetVectorMask<float>(0U, ~0ULL);
        }
        LocalTensor<float> absReal = rowDotReal_;
        LocalTensor<float> absImag = rowDotImag_;
        Abs(absReal, invReal_, static_cast<int32_t>(count));
        Abs(absImag, invImag_, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        Max(signedX_, absReal, absImag, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        Div(xReal_, invReal_, signedX_, static_cast<int32_t>(count));
        Div(xImag_, invImag_, signedX_, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        Mul(invReal_, xReal_, xReal_, static_cast<int32_t>(count));
        Mul(invImag_, xImag_, xImag_, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        Add(invReal_, invReal_, invImag_, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        // Divide by the scaled norm first. Multiplying it by the original
        // maximum can overflow even when the complex reciprocal is finite.
        Div(xReal_, xReal_, invReal_, static_cast<int32_t>(count));
        Div(xImag_, xImag_, invReal_, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        Div(invReal_, xReal_, signedX_, static_cast<int32_t>(count));
        Div(invImag_, xImag_, signedX_, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        Muls(invImag_, invImag_, -1.0f, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
    }

    __aicore__ inline void LoadSourceFromLocal(
        LocalTensor<float> sourceReal, LocalTensor<float> sourceImag, uint32_t count)
    {
        Adds(xReal_, sourceReal, 0.0f, static_cast<int32_t>(count));
        Adds(xImag_, sourceImag, 0.0f, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
    }

    __aicore__ inline void LoadOffDiagonal(uint32_t chunkStart, uint32_t count)
    {
        const uint32_t rowBytes = count * JACOBI_COMPLEX * sizeof(float);
        // One complete panel transfer keeps the 64 source rows contiguous in
        // the UB staging layout.  The old four-issue schedule paid four DMA
        // setup/barrier costs for the same 16 KB panel and prevented MTE2/V
        // overlap with the following vector reduction.
        const uint64_t rowMatrixOffset = (TRANSPOSE ? static_cast<uint64_t>(blockStart_) * lda_ + chunkStart :
                                                      static_cast<uint64_t>(chunkStart) * lda_ + blockStart_) *
                                         JACOBI_COMPLEX;
        DataCopyPad(
            matrixAos_, matrixGm_[rowMatrixOffset],
            DataCopyExtParams{
                static_cast<uint16_t>(JACOBI_BLOCK), rowBytes,
                static_cast<uint32_t>((lda_ - count) * JACOBI_COMPLEX * sizeof(float)), 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        SplitComplex(matrixReal_, matrixImag_, matrixAos_, JACOBI_BLOCK * count);
        if constexpr (CONJUGATE) {
            Muls(matrixImag_, matrixImag_, -1.0f, static_cast<int32_t>(JACOBI_BLOCK * count));
            PIPE_BARRIER(V);
        }
    }

    __aicore__ inline void ComputeRows(
        LocalTensor<float> coefficientReal, LocalTensor<float> coefficientImag, uint32_t count)
    {
        ComputeRows(coefficientReal, coefficientImag, count, count);
    }

    __aicore__ inline void ComputeRows(
        LocalTensor<float> coefficientReal, LocalTensor<float> coefficientImag, uint32_t rowCount, uint32_t columnCount)
    {
        if (rowCount == JACOBI_BLOCK && columnCount == JACOBI_BLOCK) {
            ComputePreparedRows(coefficientReal, coefficientImag);
            return;
        }
        // Process at most 16 repeated rows per vector issue.  DAV_2201's
        // cadd_v repeat field is limited to the Cgemv-supported 16-row
        // schedule; issuing 64 repeats silently leaves later rows stale.
        SetMaskNorm();
        AscendC::SetVectorMask<float>(0U, static_cast<uint64_t>(-1));
        muls_v<ArchType::ASCEND_V220, float>(signedX_, xImag_, -1.0f, 1U, 1U, 1U, 8U, 8U);
        PIPE_BARRIER(V);
        for (uint32_t rowBase = 0U; rowBase < rowCount; rowBase += 16U) {
            const uint32_t rows = JacobiMin(16U, rowCount - rowBase);
            const uint32_t matrixOffset = rowBase * columnCount;
            mul_v<ArchType::ASCEND_V220, float>(
                productReal_, xReal_, coefficientReal[matrixOffset], static_cast<uint8_t>(rows), 1U, 1U, 1U, 8U, 0U,
                static_cast<uint8_t>(columnCount / 8U));
            mul_v<ArchType::ASCEND_V220, float>(
                productImag_, xImag_, coefficientReal[matrixOffset], static_cast<uint8_t>(rows), 1U, 1U, 1U, 8U, 0U,
                static_cast<uint8_t>(columnCount / 8U));
            PIPE_BARRIER(V);
            vmla(
                reinterpret_cast<__ubuf__ float*>(productReal_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(signedX_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(coefficientImag[matrixOffset].GetPhyAddr()),
                static_cast<uint8_t>(rows), 1U, 1U, 1U, 8U, 0U, static_cast<uint8_t>(columnCount / 8U));
            vmla(
                reinterpret_cast<__ubuf__ float*>(productImag_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(xReal_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(coefficientImag[matrixOffset].GetPhyAddr()),
                static_cast<uint8_t>(rows), 1U, 1U, 1U, 8U, 0U, static_cast<uint8_t>(columnCount / 8U));
            PIPE_BARRIER(V);
            cadd_v<ArchType::ASCEND_V220, float>(
                rowDotReal_[rowBase], productReal_, static_cast<uint8_t>(rows), 1U, 1U,
                static_cast<uint8_t>(columnCount / 8U));
            cadd_v<ArchType::ASCEND_V220, float>(
                rowDotImag_[rowBase], productImag_, static_cast<uint8_t>(rows), 1U, 1U,
                static_cast<uint8_t>(columnCount / 8U));
            // The next row group overwrites these same product planes.
            // Disjoint rowDot destinations do not protect pending reads.
            PIPE_BARRIER(V);
        }
    }
    // Included in CtrsvJacobiTcKernel. Product planes remain live through all reductions.
    __aicore__ inline void ComputePreparedRows(LocalTensor<float> coefficientReal, LocalTensor<float> coefficientImag)
    {
        SetMaskNorm();
        AscendC::SetVectorMask<float>(0U, ~0ULL);
        muls_v<ArchType::ASCEND_V220, float>(signedX_, xImag_, -1.0f, 1U, 1U, 1U, 8U, 8U);
        PIPE_BARRIER(V);
        mul_v<ArchType::ASCEND_V220, float>(
            productReal_, xReal_, coefficientReal, JACOBI_BLOCK, 1U, 1U, 1U, 8U, 0U, 8U);
        mul_v<ArchType::ASCEND_V220, float>(
            productImag_, xImag_, coefficientReal, JACOBI_BLOCK, 1U, 1U, 1U, 8U, 0U, 8U);
        PIPE_BARRIER(V);
        vmla(
            reinterpret_cast<__ubuf__ float*>(productReal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(signedX_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(coefficientImag.GetPhyAddr()), JACOBI_BLOCK, 1U, 1U, 1U, 8U, 0U, 8U);
        vmla(
            reinterpret_cast<__ubuf__ float*>(productImag_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(xReal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(coefficientImag.GetPhyAddr()), JACOBI_BLOCK, 1U, 1U, 1U, 8U, 0U, 8U);
        PIPE_BARRIER(V);
#pragma unroll
        for (uint32_t row = 0U; row < JACOBI_BLOCK; row += 16U) {
            cadd_v<ArchType::ASCEND_V220, float>(rowDotReal_[row], productReal_[row * JACOBI_BLOCK], 16U, 1U, 1U, 8U);
            cadd_v<ArchType::ASCEND_V220, float>(rowDotImag_[row], productImag_[row * JACOBI_BLOCK], 16U, 1U, 1U, 8U);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ inline void ComputeDot(
        LocalTensor<float> destinationReal, LocalTensor<float> destinationImag, uint32_t count)
    {
        ComputeRows(matrixReal_, matrixImag_, count);
        Sub(destinationReal, destinationReal, rowDotReal_, static_cast<int32_t>(count));
        Sub(destinationImag, destinationImag, rowDotImag_, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        AscendC::SetVectorMask<float>(0U, static_cast<uint64_t>(-1));
    }

    __aicore__ inline void ComputeDot(
        LocalTensor<float> destinationReal, LocalTensor<float> destinationImag, uint32_t rowCount, uint32_t columnCount)
    {
        ComputeRows(matrixReal_, matrixImag_, rowCount, columnCount);
        Sub(destinationReal, destinationReal, rowDotReal_, static_cast<int32_t>(rowCount));
        Sub(destinationImag, destinationImag, rowDotImag_, static_cast<int32_t>(rowCount));
        PIPE_BARRIER(V);
        AscendC::SetVectorMask<float>(0U, static_cast<uint64_t>(-1));
    }

    __aicore__ inline void StoreLocalResult(
        LocalTensor<float> sourceReal, LocalTensor<float> sourceImag, uint32_t count,
        uint32_t validCount = JACOBI_BLOCK)
    {
        Adds(planar_, sourceReal, 0.0f, static_cast<int32_t>(count));
        Adds(planar_[count], sourceImag, 0.0f, static_cast<int32_t>(count));
        PIPE_BARRIER(V);
        TransposeParamsExt params;
        params.nSize = 1U;
        params.cSize = JACOBI_COMPLEX;
        params.hSize = 1U;
        params.wSize = count;
        params.transposeType = TransposeType::TRANSPOSE_NCHW2NHWC;
        AscendC::Transpose(xAos_, planar_, transposeTmp_, params);
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        if (validCount < count) {
            DataCopyPad(
                vectorGm_[static_cast<uint64_t>(blockStart_) * JACOBI_COMPLEX], xAos_,
                DataCopyExtParams{1U, static_cast<uint32_t>(validCount * JACOBI_COMPLEX * sizeof(float)), 0U, 0U, 0U});
        } else {
            DataCopy(vectorGm_[static_cast<uint64_t>(blockStart_) * JACOBI_COMPLEX], xAos_, count * JACOBI_COMPLEX);
        }
        // This helper is also used to publish several panels from one AIV.
        // Drain MTE3 before the next iteration reuses xAos_ and EVENT_ID0;
        // a single final ALL barrier is insufficient for queued stores.
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

    __aicore__ inline uint32_t PanelSlot(uint32_t panel) const
    {
        // Prepared panels are assigned round-robin, matching the host block
        // count. The quotient is the compact local accumulator slot.
        return (panel - blockIdx_) / blockCount_;
    }

    AsdopsBuffer<ArchType::ASCEND_V220> buffer_;
    GlobalTensor<float> matrixGm_;
    GlobalTensor<float> vectorGm_;
    GlobalTensor<float> workspaceGm_;
    LocalTensor<float> matrixAos_;
    LocalTensor<float> matrixReal_;
    LocalTensor<float> matrixImag_;
    LocalTensor<float> diagonalReal_;
    LocalTensor<float> diagonalImag_;
    LocalTensor<float> panelAccReal_[JACOBI_MAX_PANELS_PER_CORE];
    LocalTensor<float> panelAccImag_[JACOBI_MAX_PANELS_PER_CORE];
    LocalTensor<float> productReal_;
    LocalTensor<float> productImag_;
    LocalTensor<float> rowDotReal_;
    LocalTensor<float> rowDotImag_;
    LocalTensor<float> xAos_;
    LocalTensor<float> xReal_;
    LocalTensor<float> xImag_;
    LocalTensor<float> signedX_;
    LocalTensor<float> invReal_;
    LocalTensor<float> invImag_;
    LocalTensor<int32_t> gatherOffsets_;
    LocalTensor<float> planar_;
    LocalTensor<uint8_t> transposeTmp_;
    LocalTensor<float> groupReal_;
    LocalTensor<float> groupSourceReal_;
    LocalTensor<float> groupAos_;
    uint32_t n_ = 0U;
    uint32_t lda_ = 0U;
    int32_t uplo_ = ACLBLAS_LOWER;
    uint32_t blockIdx_ = 0U;
    uint32_t blockCount_ = 1U;
    bool splitPrepared_ = false;
    uint32_t panelCount_ = 0U;
    uint32_t blockStart_ = 0U;
    uint32_t flagsOffset_ = 0U;
    uint32_t pingOffset_ = 0U;
    uint32_t diagonalRealOffset_ = 0U;
    uint32_t diagonalImagOffset_ = 0U;
};

} // namespace

/** Parallel native-panel inversion for the grouped T/C path on arch22. */
namespace {

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::PrepareInverseGeometry()
{
    // The diagonal gather offsets are exact, monotone FP32 coordinates.
    // Build col-row once per owner; the two packed masks survive all local
    // diagonal blocks in groupAos_, which is idle until the solve phase.
    constexpr uint32_t maskFloats = JACOBI_BLOCK_FLOATS / (8U * sizeof(float));
    static_assert(2U * maskFloats <= 2U * CTRSV_GROUP_ROWS, "Inverse masks exceed grouped scratch");
    Cast(xReal_, gatherOffsets_, RoundMode::CAST_NONE, JACOBI_BLOCK);
    PIPE_BARRIER(V);
    Brcb(productReal_, xReal_, 8U, {1U, 8U});
    PIPE_BARRIER(V);
    SetMaskNorm();
    AscendC::SetVectorMask<float>(0U, ~0ULL);
    sub_v<ArchType::ASCEND_V220, float>(matrixAos_, xReal_, productReal_, JACOBI_BLOCK, 1U, 1U, 0U, 8U, 0U, 1U);
    PIPE_BARRIER(V);
    Compares(groupAos_.template ReinterpretCast<uint8_t>(), matrixAos_, 0.0f, CMPMODE::EQ, JACOBI_BLOCK_FLOATS);
    Compares(
        groupAos_[maskFloats].template ReinterpretCast<uint8_t>(), matrixAos_, 0.0f,
        EffectiveUpper() ? CMPMODE::GT : CMPMODE::LT, JACOBI_BLOCK_FLOATS);
    PIPE_BARRIER(V);
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::NormalizeInverseRows()
{
    // Normalize all rows before the dependency chain: L = D^-1 A.
    // Elimination then starts from D^-1 and never rescales a pivot row.
    if constexpr (UNIT_DIAG) {
        TransposePlane(matrixAos_, matrixReal_, JACOBI_BLOCK);
        TransposePlane(matrixAos_[JACOBI_BLOCK_FLOATS], matrixImag_, JACOBI_BLOCK);
    } else {
        Brcb(productReal_, invReal_, 8U, {1U, 8U});
        Brcb(productImag_, invImag_, 8U, {1U, 8U});
        PIPE_BARRIER(V);
        SetMaskNorm();
        AscendC::SetVectorMask<float>(0U, static_cast<uint64_t>(-1));
        mul_v<ArchType::ASCEND_V220, float>(
            diagonalReal_, matrixReal_, productReal_, JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
        mul_v<ArchType::ASCEND_V220, float>(
            diagonalImag_, matrixImag_, productReal_, JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
        PIPE_BARRIER(V);
        vmla(
            reinterpret_cast<__ubuf__ float*>(diagonalImag_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(matrixReal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(productImag_.GetPhyAddr()), JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
        PIPE_BARRIER(V);
        Muls(productImag_, productImag_, -1.0f, JACOBI_BLOCK * 8U);
        PIPE_BARRIER(V);
        SetMaskNorm();
        AscendC::SetVectorMask<float>(0U, ~0ULL);
        vmla(
            reinterpret_cast<__ubuf__ float*>(diagonalReal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(matrixImag_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(productImag_.GetPhyAddr()), JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
        PIPE_BARRIER(V);
        TransposePlane(matrixAos_, diagonalReal_, JACOBI_BLOCK);
        TransposePlane(matrixAos_[JACOBI_BLOCK_FLOATS], diagonalImag_, JACOBI_BLOCK);
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::InitializeInverseIdentity(
    LocalTensor<uint8_t> identity)
{
    const BinaryRepeatParams dense{1U, 1U, 1U, 8U, 8U, 8U};

    Muls(matrixAos_, matrixAos_, -1.0f, JACOBI_BLOCK_FLOATS);
    // Original matrixReal_ is dead after transposition. Retain -imag(A/D)
    // here so each pivot consumes immutable signs without a negate barrier.
    Muls(matrixReal_, matrixAos_[JACOBI_BLOCK_FLOATS], -1.0f, JACOBI_BLOCK_FLOATS);
    Duplicate(diagonalReal_, 1.0f, JACOBI_BLOCK_FLOATS);
    Duplicate(diagonalImag_, 0.0f, JACOBI_BLOCK_FLOATS);
    PIPE_BARRIER(V);
    SetCmpMask(rowDotReal_.template ReinterpretCast<uint64_t>());
    PIPE_BARRIER(V);
    SetMaskNorm();
    AscendC::SetVectorMask<float>(0U, ~0ULL);
    Select(diagonalReal_, identity, diagonalReal_, JACOBI_BLOCK, dense);
    PIPE_BARRIER(V);
    if constexpr (!UNIT_DIAG) {
        Brcb(productImag_, invImag_, 8U, {1U, 8U});
        PIPE_BARRIER(V);
        SetMaskNorm();
        AscendC::SetVectorMask<float>(0U, static_cast<uint64_t>(-1));
        mul_v<ArchType::ASCEND_V220, float>(
            diagonalImag_, diagonalReal_, productImag_, JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
        PIPE_BARRIER(V);
        mul_v<ArchType::ASCEND_V220, float>(
            diagonalReal_, diagonalReal_, productReal_, JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
        PIPE_BARRIER(V);
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::BuildGroupedInverse()
{
    const bool upper = EffectiveUpper();
    constexpr uint32_t maskFloats = JACOBI_BLOCK_FLOATS / (8U * sizeof(float));
    const auto triangle = groupAos_[maskFloats].template ReinterpretCast<uint8_t>();
    const auto identity = groupAos_.template ReinterpretCast<uint8_t>();
    const BinaryRepeatParams dense{1U, 1U, 1U, 8U, 8U, 8U};
    // Explicit mode-1 setup avoids Select's hidden 8KB temporary UB region.
    // Selecting zero (not multiplying by it) also discards poisoned NaNs.
    Duplicate(rowDotReal_, 0.0f, JACOBI_BLOCK);
    PIPE_BARRIER(V);
    SetCmpMask(rowDotReal_.template ReinterpretCast<uint64_t>());
    PIPE_BARRIER(V);
    SetMaskNorm();
    AscendC::SetVectorMask<float>(0U, ~0ULL);
    Select(matrixReal_, triangle, matrixReal_, JACOBI_BLOCK, dense);
    Select(matrixImag_, triangle, matrixImag_, JACOBI_BLOCK, dense);
    PIPE_BARRIER(V);
    if (splitPrepared_) {
        // Save only the selected strict triangle: NaN * zero is not a mask.
        Adds(buffer_.GetBuffer<BufferType::ASCEND_UB, float>(160U * 1024U), matrixReal_, 0.0f, JACOBI_BLOCK_FLOATS);
        Adds(buffer_.GetBuffer<BufferType::ASCEND_UB, float>(176U * 1024U), matrixImag_, 0.0f, JACOBI_BLOCK_FLOATS);
        PIPE_BARRIER(V);
    }
    NormalizeInverseRows();
    InitializeInverseIdentity(identity);
    if (splitPrepared_) {
        if (upper)
            EliminateSplitInverse<true>();
        else
            EliminateSplitInverse<false>();
    } else {
        if (upper)
            EliminatePreparedInverse<true>();
        else
            EliminatePreparedInverse<false>();
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::UpdateInversePivot(
    uint32_t row, uint32_t first, uint32_t rows, uint32_t coefficient)
{
    vmla(
        reinterpret_cast<__ubuf__ float*>(diagonalReal_[first * JACOBI_BLOCK].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(diagonalReal_[row].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(productReal_[coefficient].GetPhyAddr()), rows, 1U, 1U, 0U, 8U, 0U, 1U);
    vmla(
        reinterpret_cast<__ubuf__ float*>(diagonalImag_[first * JACOBI_BLOCK].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(diagonalImag_[row].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(productReal_[coefficient].GetPhyAddr()), rows, 1U, 1U, 0U, 8U, 0U, 1U);
    PIPE_BARRIER(V);
    vmla(
        reinterpret_cast<__ubuf__ float*>(diagonalReal_[first * JACOBI_BLOCK].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(diagonalImag_[row].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(productImag_[coefficient].GetPhyAddr()), rows, 1U, 1U, 0U, 8U, 0U, 1U);
    vmla(
        reinterpret_cast<__ubuf__ float*>(diagonalImag_[first * JACOBI_BLOCK].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(diagonalReal_[row].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(matrixImag_[coefficient].GetPhyAddr()), rows, 1U, 1U, 0U, 8U, 0U, 1U);
    PIPE_BARRIER(V);
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool UPPER>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::EliminatePreparedInverse()
{
    SetMaskNorm();
    AscendC::SetVectorMask<float>(0U, ~0ULL);
    static_assert(JACOBI_PRODUCT_FLOATS >= 8U * JACOBI_BLOCK * 8U, "Prepared broadcast batch exceeds product plane");
    for (uint32_t batch = 0U; batch < JACOBI_BLOCK; batch += 8U) {
        // matrixImag_ is dead after transposition; all three coefficient
        // sources and both inverse outputs remain disjoint from this batch.
        const uint32_t batchRow = (UPPER ? JACOBI_BLOCK - batch - 8U : batch) * JACOBI_BLOCK;
        vbrcb(
            reinterpret_cast<__ubuf__ uint32_t*>(productReal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(matrixAos_[batchRow].GetPhyAddr()), 1U, 8U, 64U);
        vbrcb(
            reinterpret_cast<__ubuf__ uint32_t*>(productImag_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(matrixAos_[JACOBI_BLOCK_FLOATS + batchRow].GetPhyAddr()), 1U, 8U, 64U);
        vbrcb(
            reinterpret_cast<__ubuf__ uint32_t*>(matrixImag_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(matrixReal_[batchRow].GetPhyAddr()), 1U, 8U, 64U);
        PIPE_BARRIER(V);
#pragma unroll
        for (uint32_t lane = 0U; lane < 8U; ++lane) {
            const uint32_t ordinal = batch + lane;
            if (ordinal + 1U == JACOBI_BLOCK)
                break;
            const uint32_t pivot = UPPER ? JACOBI_BLOCK - 1U - ordinal : ordinal;
            const uint32_t row = pivot * JACOBI_BLOCK;
            const uint32_t first = UPPER ? 0U : pivot + 1U;
            const uint32_t rows = UPPER ? pivot : JACOBI_BLOCK - pivot - 1U;
            const uint32_t coefficient = (pivot & 7U) * JACOBI_BLOCK * 8U + first * 8U;
            UpdateInversePivot(row, first, rows, coefficient);
        }
    }
    ResetMask();
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool PADDED>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::BuildPreparedPanelInverse()
{
    SET_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    if constexpr (PADDED) {
        LoadPreparedTailDiagonal();
        BuildDiagonalInverse(JACOBI_BLOCK, JacobiMin(JACOBI_BLOCK, n_ - blockStart_));
    } else {
        LoadDiagonalPanel(JACOBI_BLOCK);
        BuildDiagonalInverse(JACOBI_BLOCK);
    }
    BuildGroupedInverse();
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool PADDED>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::PrepareGroupedInverse()
{
    PrepareInverseGeometry();
    for (uint32_t panel = blockIdx_; panel < panelCount_; panel += GetBlockNum()) {
        blockStart_ = panel * JACOBI_BLOCK;
        BuildPreparedPanelInverse<PADDED>();
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        const uint64_t offset = static_cast<uint64_t>(blockStart_) * JACOBI_BLOCK;
        DataCopy(workspaceGm_[diagonalRealOffset_ + offset], diagonalReal_, JACOBI_BLOCK_FLOATS);
        DataCopy(workspaceGm_[diagonalImagOffset_ + offset], diagonalImag_, JACOBI_BLOCK_FLOATS);
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
    }
}
} // namespace

/** Two independent 32-row diagonal inverses, with an owner-local coupling update. */
namespace {
template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::PrepareSplitPivotBatch(
    uint32_t ordinal, uint32_t localPivot)
{
    if ((ordinal & 3U) == 0U) {
#pragma unroll
        for (uint32_t half = 0U; half < 2U; ++half) {
            const uint32_t row = (half * 32U + (localPivot & ~3U)) * 64U;
            const uint32_t storage = half * 2048U;
            vbrcb(
                reinterpret_cast<__ubuf__ uint32_t*>(productReal_[storage].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(matrixAos_[row].GetPhyAddr()), 1U, 8U, 32U);
            vbrcb(
                reinterpret_cast<__ubuf__ uint32_t*>(productImag_[storage].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(matrixAos_[4096U + row].GetPhyAddr()), 1U, 8U, 32U);
            vbrcb(
                reinterpret_cast<__ubuf__ uint32_t*>(matrixImag_[storage].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(matrixReal_[row].GetPhyAddr()), 1U, 8U, 32U);
        }
        PIPE_BARRIER(V);
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool UPPER>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::EliminateSplitInverse()
{
    SetMaskNorm();
    AscendC::SetVectorMask<float>(0U, ~0ULL);
    for (uint32_t ordinal = 0U; ordinal < 31U; ++ordinal) {
        const uint32_t localPivot = UPPER ? 31U - ordinal : ordinal;
        const uint32_t first = UPPER ? 0U : localPivot + 1U;
        const uint32_t rows = UPPER ? localPivot : 31U - localPivot;
        PrepareSplitPivotBatch(ordinal, localPivot);

#pragma unroll
        for (uint32_t half = 0U; half < 2U; ++half) {
            const uint32_t row = (half * 32U + localPivot) * 64U;
            const uint32_t destination = (half * 32U + first) * 64U;
            const uint32_t coefficient = half * 2048U + (localPivot & 3U) * 512U + (half * 32U + first) * 8U;
            vmla(
                reinterpret_cast<__ubuf__ float*>(diagonalReal_[destination].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(diagonalReal_[row].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(productReal_[coefficient].GetPhyAddr()), rows, 1U, 1U, 0U, 8U, 0U,
                1U);
            vmla(
                reinterpret_cast<__ubuf__ float*>(diagonalImag_[destination].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(diagonalImag_[row].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(productReal_[coefficient].GetPhyAddr()), rows, 1U, 1U, 0U, 8U, 0U,
                1U);
        }
        PIPE_BARRIER(V);
#pragma unroll
        for (uint32_t half = 0U; half < 2U; ++half) {
            const uint32_t row = (half * 32U + localPivot) * 64U;
            const uint32_t destination = (half * 32U + first) * 64U;
            const uint32_t coefficient = half * 2048U + (localPivot & 3U) * 512U + (half * 32U + first) * 8U;
            vmla(
                reinterpret_cast<__ubuf__ float*>(diagonalReal_[destination].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(diagonalImag_[row].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(productImag_[coefficient].GetPhyAddr()), rows, 1U, 1U, 0U, 8U, 0U,
                1U);
            vmla(
                reinterpret_cast<__ubuf__ float*>(diagonalImag_[destination].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(diagonalReal_[row].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(matrixImag_[coefficient].GetPhyAddr()), rows, 1U, 1U, 0U, 8U, 0U, 1U);
        }
        PIPE_BARRIER(V);
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::ComputePreparedHalf(
    LocalTensor<float> real, LocalTensor<float> imag, uint32_t target, uint32_t source)
{
    Muls(signedX_, xImag_[source], -1.0f, 32U);
    PIPE_BARRIER(V);
    SetMaskNorm();
    AscendC::SetVectorMask<float>(0U, 0xffffffffULL);
    const uint32_t matrix = target * 64U + source;
    mul_v<ArchType::ASCEND_V220, float>(productReal_, xReal_[source], real[matrix], 32U, 1U, 1U, 1U, 8U, 0U, 8U);
    mul_v<ArchType::ASCEND_V220, float>(productImag_, xImag_[source], real[matrix], 32U, 1U, 1U, 1U, 8U, 0U, 8U);
    PIPE_BARRIER(V);
    vmla(
        reinterpret_cast<__ubuf__ float*>(productReal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(signedX_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(imag[matrix].GetPhyAddr()), 32U, 1U, 1U, 1U, 8U, 0U, 8U);
    vmla(
        reinterpret_cast<__ubuf__ float*>(productImag_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(xReal_[source].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(imag[matrix].GetPhyAddr()), 32U, 1U, 1U, 1U, 8U, 0U, 8U);
    PIPE_BARRIER(V);
#pragma unroll
    for (uint32_t row = 0U; row < 32U; row += 16U) {
        cadd_v<ArchType::ASCEND_V220, float>(rowDotReal_[target + row], productReal_[row * 64U], 16U, 1U, 1U, 8U);
        cadd_v<ArchType::ASCEND_V220, float>(rowDotImag_[target + row], productImag_[row * 64U], 16U, 1U, 1U, 8U);
    }
    PIPE_BARRIER(V);
    ResetMask();
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::SolveSplitPrepared(
    LocalTensor<float> rhsReal, LocalTensor<float> rhsImag, LocalTensor<float> originalReal,
    LocalTensor<float> originalImag)
{
    const uint32_t first = EffectiveUpper() ? 32U : 0U;
    const uint32_t second = 32U - first;
    LoadSourceFromLocal(rhsReal, rhsImag, 64U);
    ComputePreparedHalf(diagonalReal_, diagonalImag_, first, first);
    Adds(rhsReal[first], rowDotReal_[first], 0.0f, 32U);
    Adds(rhsImag[first], rowDotImag_[first], 0.0f, 32U);
    Adds(xReal_[first], rowDotReal_[first], 0.0f, 32U);
    Adds(xImag_[first], rowDotImag_[first], 0.0f, 32U);
    PIPE_BARRIER(V);
    // Only the opposite half consumes this dot. Its coefficients are strictly
    // off-diagonal, so neither the UNIT diagonal nor the unused triangle enters.
    ComputePreparedHalf(originalReal, originalImag, second, first);
    Sub(rhsReal[second], rhsReal[second], rowDotReal_[second], 32U);
    Sub(rhsImag[second], rhsImag[second], rowDotImag_[second], 32U);
    PIPE_BARRIER(V);
    LoadSourceFromLocal(rhsReal, rhsImag, 64U);
    ComputePreparedHalf(diagonalReal_, diagonalImag_, second, second);
    Adds(rhsReal[second], rowDotReal_[second], 0.0f, 32U);
    Adds(rhsImag[second], rowDotImag_[second], 0.0f, 32U);
    PIPE_BARRIER(V);
}
} // namespace

/** Native-panel owners with immutable publication and resident inverses. */
namespace {
template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool PADDED>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::InitializePreparedInverses(
    const PreparedState& state)
{
    if (state.residentInverse) {
        PrepareInverseGeometry();
        for (uint32_t slot = 0U; slot < state.localCount; ++slot) {
            blockStart_ = (blockIdx_ + slot * blockCount_) * JACOBI_BLOCK;
            BuildPreparedPanelInverse<PADDED>();
            if (slot + 1U < state.localCount) {
                Adds(state.savedReal, diagonalReal_, 0.0f, JACOBI_BLOCK_FLOATS);
                Adds(state.savedImag, diagonalImag_, 0.0f, JACOBI_BLOCK_FLOATS);
                PIPE_BARRIER(V);
            }
        }
    } else {
        // Reduced-core launches retain the existing GM inverse contract.
        // Only this owner consumes its inverse, so no extra global barrier
        // is needed between local preparation and the first pivot.
        PrepareGroupedInverse<PADDED>();
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool PADDED>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::InitializePanelRhs()
{
    for (uint32_t panel = blockIdx_; panel < panelCount_; panel += blockCount_) {
        blockStart_ = panel * JACOBI_BLOCK;
        // With 64 panels and 40 AIVs some owners initialize two slots.
        // SplitComplex must retire before the next DMA reuses xAos_.
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        if constexpr (PADDED) {
            LoadPreparedTailRhs(panelAccReal_[PanelSlot(panel)], panelAccImag_[PanelSlot(panel)]);
        } else {
            LoadInitialPanel(panelAccReal_[PanelSlot(panel)], panelAccImag_[PanelSlot(panel)], JACOBI_BLOCK);
        }
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool PADDED>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::PrefetchDependentPanel(
    const PreparedState& state, uint32_t firstOrdinal, uint32_t start)
{
    if (firstOrdinal < state.localCount) {
        const uint32_t slot = state.upper ? state.localCount - firstOrdinal - 1U : firstOrdinal;
        const uint32_t panel = blockIdx_ + slot * blockCount_;
        const uint64_t offset = (TRANSPOSE ? static_cast<uint64_t>(panel * JACOBI_BLOCK) * lda_ + start :
                                             static_cast<uint64_t>(start) * lda_ + panel * JACOBI_BLOCK) *
                                JACOBI_COMPLEX;
        // Matrix coefficients do not depend on the pivot solution. Start
        // the first full panel while the owner evaluates its inverse GEMV.
        // matrixAos_ is disjoint from the owner's inverse/product buffers.
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        if constexpr (PADDED) {
            LoadPreparedTailTileAsync(panel * JACOBI_BLOCK, start);
        } else {
            DataCopyPad(
                matrixAos_, matrixGm_[offset],
                DataCopyExtParams{
                    JACOBI_BLOCK, JACOBI_BLOCK * JACOBI_COMPLEX * sizeof(float),
                    (lda_ - JACOBI_BLOCK) * JACOBI_COMPLEX * sizeof(float), 0U, 0U},
                DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
        }
        SET_FLAG(MTE2, V, EVENT_ID1);
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::PublishPreparedPivot(
    const PreparedState& state, uint32_t pivot, uint32_t start)
{
    if (blockIdx_ == pivot % blockCount_) {
        const uint32_t slot = PanelSlot(pivot);
        blockStart_ = start;
        if (!state.residentInverse) {
            SET_FLAG(V, MTE2, EVENT_ID0);
            WAIT_FLAG(V, MTE2, EVENT_ID0);
            const uint64_t offset = static_cast<uint64_t>(start) * JACOBI_BLOCK;
            DataCopy(diagonalReal_, workspaceGm_[diagonalRealOffset_ + offset], JACOBI_BLOCK_FLOATS);
            DataCopy(diagonalImag_, workspaceGm_[diagonalImagOffset_ + offset], JACOBI_BLOCK_FLOATS);
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
        if (splitPrepared_) {
            SolveSplitPrepared(panelAccReal_[slot], panelAccImag_[slot], state.savedReal, state.savedImag);
        } else {
            LoadSourceFromLocal(panelAccReal_[slot], panelAccImag_[slot], JACOBI_BLOCK);
            const bool saved = state.residentInverse && state.localCount == 2U && slot == 0U;
            ComputeRows(saved ? state.savedReal : diagonalReal_, saved ? state.savedImag : diagonalImag_, JACOBI_BLOCK);
            Adds(panelAccReal_[slot], rowDotReal_, 0.0f, JACOBI_BLOCK);
            Adds(panelAccImag_[slot], rowDotImag_, 0.0f, JACOBI_BLOCK);
            PIPE_BARRIER(V);
        }
        // The workspace stays planar for its entire publication lifetime.
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        DataCopy(workspaceGm_[pingOffset_ + start], panelAccReal_[slot], JACOBI_BLOCK);
        DataCopy(workspaceGm_[pingOffset_ + state.stateN + start], panelAccImag_[slot], JACOBI_BLOCK);
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::LoadPublishedSource(
    uint32_t start, uint32_t stateN)
{
    SET_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    DataCopy(xReal_, workspaceGm_[pingOffset_ + start], JACOBI_BLOCK);
    DataCopy(xImag_, workspaceGm_[pingOffset_ + stateN + start], JACOBI_BLOCK);
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
    if constexpr (!TRANSPOSE) {
        PreparePreparedNSource();
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool PADDED>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::LoadDependentPanel(
    uint32_t start, bool prefetched)
{
    if (prefetched) {
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        SplitComplex(matrixReal_, matrixImag_, matrixAos_, JACOBI_BLOCK_FLOATS);
        if constexpr (CONJUGATE) {
            Muls(matrixImag_, matrixImag_, -1.0f, JACOBI_BLOCK_FLOATS);
            PIPE_BARRIER(V);
        }
    } else {
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        if constexpr (PADDED) {
            LoadPreparedTailTileAsync(blockStart_, start);
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            SplitComplex(matrixReal_, matrixImag_, matrixAos_, JACOBI_BLOCK_FLOATS);
            if constexpr (CONJUGATE) {
                Muls(matrixImag_, matrixImag_, -1.0f, JACOBI_BLOCK_FLOATS);
                PIPE_BARRIER(V);
            }
        } else {
            LoadOffDiagonal(start, JACOBI_BLOCK);
        }
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool PADDED>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::UpdateDependentPanels(
    const PreparedState& state, uint32_t firstOrdinal, uint32_t start)
{
    bool loaded = false;
    for (uint32_t ordinal = firstOrdinal; ordinal < state.localCount; ++ordinal) {
        const uint32_t slot = state.upper ? state.localCount - ordinal - 1U : ordinal;
        const uint32_t panel = blockIdx_ + slot * blockCount_;
        if (!loaded) {
            LoadPublishedSource(start, state.stateN);
            loaded = true;
        }
        blockStart_ = panel * JACOBI_BLOCK;
        LoadDependentPanel<PADDED>(start, ordinal == firstOrdinal);
        if constexpr (TRANSPOSE) {
            ComputeDot(panelAccReal_[slot], panelAccImag_[slot], JACOBI_BLOCK);
        } else {
            ComputePreparedNUpdate(panelAccReal_[slot], panelAccImag_[slot]);
        }
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
template <bool PADDED>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::ProcessPreparedPanels()
{
    const PreparedState state{
        EffectiveUpper(),
        PADDED ? panelCount_ * JACOBI_BLOCK : n_,
        (panelCount_ - 1U - blockIdx_) / blockCount_ + 1U,
        panelCount_ <= 2U * blockCount_,
        buffer_.GetBuffer<BufferType::ASCEND_UB, float>(160U * 1024U),
        buffer_.GetBuffer<BufferType::ASCEND_UB, float>(176U * 1024U)};
    splitPrepared_ = n_ <= 512U && state.localCount == 1U;
    InitializePreparedInverses<PADDED>(state);
    InitializePanelRhs<PADDED>();
    for (uint32_t phase = 0; phase < panelCount_; ++phase) {
        const uint32_t pivot = state.upper ? panelCount_ - phase - 1U : phase;
        const uint32_t start = pivot * JACOBI_BLOCK;
        uint32_t firstOrdinal = 0U;
        while (firstOrdinal < state.localCount) {
            const uint32_t slot = state.upper ? state.localCount - firstOrdinal - 1U : firstOrdinal;
            const uint32_t panel = blockIdx_ + slot * blockCount_;
            if (state.upper ? panel < pivot : panel > pivot)
                break;
            ++firstOrdinal;
        }
        PrefetchDependentPanel<PADDED>(state, firstOrdinal, start);
        PublishPreparedPivot(state, pivot, start);
        // Each panel is published once to its own address. No ACK or slot
        // reuse barrier is required; the next owner waits for its own update.
        SyncAllAiv();
        UpdateDependentPanels<PADDED>(state, firstOrdinal, start);
    }
    for (uint32_t panel = blockIdx_; panel < panelCount_; panel += blockCount_) {
        blockStart_ = panel * JACOBI_BLOCK;
        StoreLocalResult(
            panelAccReal_[PanelSlot(panel)], panelAccImag_[PanelSlot(panel)], JACOBI_BLOCK,
            PADDED ? JacobiMin(JACOBI_BLOCK, n_ - blockStart_) : JACOBI_BLOCK);
    }
}
} // namespace

__schedmode__(1) __global__ __mix__(0, 1) __aicore__
    void ctrsv_kernel_prepared_t(GM_ADDR matrix, GM_ADDR vector, GM_ADDR workspace, const CtrsvPreparedArgs args)
{
    AscendC::InitSocState();
    CtrsvJacobiTcKernel<true, false> op;
    op.InitPrepared(matrix, vector, workspace, args);
    op.ProcessSafePrepared(matrix, vector);
}

__schedmode__(1) __global__ __mix__(0, 1) __aicore__
    void ctrsv_kernel_prepared_c(GM_ADDR matrix, GM_ADDR vector, GM_ADDR workspace, const CtrsvPreparedArgs args)
{
    AscendC::InitSocState();
    CtrsvJacobiTcKernel<true, true> op;
    op.InitPrepared(matrix, vector, workspace, args);
    op.ProcessSafePrepared(matrix, vector);
}

/** Exact native-panel N solve; off-diagonal panels stay column-major. */
namespace {
template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::PreparePreparedNSource()
{
    // N never uses the grouped-T/C state. Each broadcast spans two adjacent
    // 256-float group buffers, separate from matrix DMA and inverse storage.
    static_assert(2U * CTRSV_GROUP_ROWS >= JACOBI_BLOCK * 8U, "N source broadcasts exceed grouped scratch");
    Muls(signedX_, xImag_, -1.0f, JACOBI_BLOCK);
    PIPE_BARRIER(V);
    Brcb(groupReal_, xReal_, 8U, {1U, 8U});
    Brcb(groupSourceReal_, xImag_, 8U, {1U, 8U});
    Brcb(groupAos_, signedX_, 8U, {1U, 8U});
    PIPE_BARRIER(V);
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::ComputePreparedNUpdate(
    LocalTensor<float> destinationReal, LocalTensor<float> destinationImag)
{
    // Every product column owns 64 contiguous target rows; a plane-fold tree
    // reduces columns without reducing or mixing the target-row dimension.
    LocalTensor<float> broadcastReal = groupReal_;
    LocalTensor<float> broadcastImag = groupSourceReal_;
    LocalTensor<float> broadcastNegImag = groupAos_;
    SetMaskNorm();
    AscendC::SetVectorMask<float>(0U, ~0ULL);
    mul_v<ArchType::ASCEND_V220, float>(productReal_, matrixReal_, broadcastReal, JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
    mul_v<ArchType::ASCEND_V220, float>(productImag_, matrixImag_, broadcastReal, JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
    PIPE_BARRIER(V);
    vmla(
        reinterpret_cast<__ubuf__ float*>(productImag_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(matrixReal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(broadcastImag.GetPhyAddr()), JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
    vmla(
        reinterpret_cast<__ubuf__ float*>(productReal_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(matrixImag_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(broadcastNegImag.GetPhyAddr()), JACOBI_BLOCK, 1U, 1U, 0U, 8U, 8U, 1U);
    PIPE_BARRIER(V);
    for (uint32_t half = JACOBI_BLOCK_FLOATS / 2U; half >= JACOBI_BLOCK; half /= 2U) {
        Add(productReal_, productReal_, productReal_[half], half);
        Add(productImag_, productImag_, productImag_[half], half);
        PIPE_BARRIER(V);
    }
    Sub(destinationReal, destinationReal, productReal_, JACOBI_BLOCK);
    Sub(destinationImag, destinationImag, productImag_, JACOBI_BLOCK);
    PIPE_BARRIER(V);
}
} // namespace

__schedmode__(1) __global__ __mix__(0, 1) __aicore__
    void ctrsv_kernel_prepared_n(GM_ADDR matrix, GM_ADDR vector, GM_ADDR workspace, const CtrsvPreparedArgs args)
{
    AscendC::InitSocState();
    CtrsvJacobiTcKernel<false, false> op;
    op.InitPrepared(matrix, vector, workspace, args);
    op.ProcessSafePrepared(matrix, vector);
}

/** Native 64x64 panels with tile-local padding, never a padded GM matrix. */
namespace {
template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::LoadPreparedTailTileAsync(
    uint32_t targetStart, uint32_t sourceStart)
{
    const uint32_t targetCount = JacobiMin(JACOBI_BLOCK, n_ - targetStart);
    const uint32_t sourceCount = JacobiMin(JACOBI_BLOCK, n_ - sourceStart);
    const uint32_t rowStart = TRANSPOSE ? sourceStart : targetStart;
    const uint32_t colStart = TRANSPOSE ? targetStart : sourceStart;
    const uint32_t rows = TRANSPOSE ? sourceCount : targetCount;
    const uint32_t cols = TRANSPOSE ? targetCount : sourceCount;
    if (rows != JACOBI_BLOCK || cols != JACOBI_BLOCK) {
        Duplicate(matrixAos_, 0.0f, JACOBI_BLOCK_AOS_FLOATS);
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
    }
    // GM strides are bytes; UB strides are 32-byte blocks after right padding.
    // The padded DMA width plus dstStride is always one 64-complex column.
    const uint32_t alignedRows = (rows + 3U) & ~3U;
    const uint64_t offset = (static_cast<uint64_t>(colStart) * lda_ + rowStart) * JACOBI_COMPLEX;
    const uint32_t rowBytes = rows * JACOBI_COMPLEX * sizeof(float);
    const uint32_t sourceStride = (lda_ - rows) * JACOBI_COMPLEX * sizeof(float);
    DataCopyPad(
        matrixAos_, matrixGm_[offset],
        DataCopyExtParams{static_cast<uint16_t>(cols), rowBytes, sourceStride, (JACOBI_BLOCK - alignedRows) / 4U, 0U},
        DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((alignedRows - rows) * JACOBI_COMPLEX), 0.0f});
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::LoadPreparedTailDiagonal()
{
    LoadPreparedTailTileAsync(blockStart_, blockStart_);
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
    SplitComplex(matrixReal_, matrixImag_, matrixAos_, JACOBI_BLOCK_FLOATS);
    if constexpr (CONJUGATE) {
        Muls(matrixImag_, matrixImag_, -1.0f, JACOBI_BLOCK_FLOATS);
        PIPE_BARRIER(V);
    }
    if constexpr (!TRANSPOSE) {
        TransposePlane(diagonalReal_, matrixReal_, JACOBI_BLOCK);
        Adds(matrixReal_, diagonalReal_, 0.0f, JACOBI_BLOCK_FLOATS);
        TransposePlane(diagonalImag_, matrixImag_, JACOBI_BLOCK);
        Adds(matrixImag_, diagonalImag_, 0.0f, JACOBI_BLOCK_FLOATS);
        PIPE_BARRIER(V);
    }
}

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG>
__aicore__ inline void CtrsvJacobiTcKernel<TRANSPOSE, CONJUGATE, UNIT_DIAG>::LoadPreparedTailRhs(
    LocalTensor<float> real, LocalTensor<float> imag)
{
    const uint32_t count = JacobiMin(JACOBI_BLOCK, n_ - blockStart_);
    if (count != JACOBI_BLOCK) {
        Duplicate(xAos_, 0.0f, JACOBI_BLOCK * JACOBI_COMPLEX);
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
    }
    const uint32_t floats = count * JACOBI_COMPLEX;
    DataCopyPad(
        xAos_, vectorGm_[static_cast<uint64_t>(blockStart_) * JACOBI_COMPLEX],
        DataCopyExtParams{1U, static_cast<uint32_t>(floats * sizeof(float)), 0U, 0U, 0U},
        DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((8U - floats % 8U) % 8U), 0.0f});
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
    SplitComplex(real, imag, xAos_, JACOBI_BLOCK);
}
} // namespace

#define CTRSV_PREPARED_TAIL_ENTRY(NAME, TRANSPOSE_VALUE, CONJUGATE_VALUE, UNIT_VALUE)    \
    __schedmode__(1) __global__ __mix__(0, 1) __aicore__ void NAME(                      \
        GM_ADDR matrix, GM_ADDR vector, GM_ADDR workspace, const CtrsvPreparedArgs args) \
    {                                                                                    \
        AscendC::InitSocState();                                                         \
        CtrsvJacobiTcKernel<TRANSPOSE_VALUE, CONJUGATE_VALUE, UNIT_VALUE> op;            \
        op.InitPrepared(matrix, vector, workspace, args);                                \
        op.ProcessSafePrepared<true>(matrix, vector);                                    \
    }

CTRSV_PREPARED_TAIL_ENTRY(ctrsv_kernel_prepared_tail_n, false, false, false)
CTRSV_PREPARED_TAIL_ENTRY(ctrsv_kernel_prepared_tail_t, true, false, false)
CTRSV_PREPARED_TAIL_ENTRY(ctrsv_kernel_prepared_tail_c, true, true, false)

#undef CTRSV_PREPARED_TAIL_ENTRY

void ctrsv_kernel_do(
    uint8_t* matrix, uint8_t* vector, uint8_t* workspace, int32_t trans, int32_t diag, bool prepared, bool tail,
    const CtrsvPreparedArgs& args, void* stream)
{
    if (prepared) {
        const uint32_t blocks = args.blocks;
        if (tail) {
            if (trans == ACLBLAS_OP_N) {
                ctrsv_kernel_prepared_tail_n<<<blocks, nullptr, stream>>>(matrix, vector, workspace, args);
            } else if (trans == ACLBLAS_OP_T) {
                ctrsv_kernel_prepared_tail_t<<<blocks, nullptr, stream>>>(matrix, vector, workspace, args);
            } else {
                ctrsv_kernel_prepared_tail_c<<<blocks, nullptr, stream>>>(matrix, vector, workspace, args);
            }
        } else if (trans == ACLBLAS_OP_N) {
            ctrsv_kernel_prepared_n<<<blocks, nullptr, stream>>>(matrix, vector, workspace, args);
        } else if (trans == ACLBLAS_OP_T) {
            ctrsv_kernel_prepared_t<<<blocks, nullptr, stream>>>(matrix, vector, workspace, args);
        } else {
            ctrsv_kernel_prepared_c<<<blocks, nullptr, stream>>>(matrix, vector, workspace, args);
        }
    } else if (trans == ACLBLAS_OP_N) {
        if (diag == ACLBLAS_UNIT)
            ctrsv_kernel_generic_n_unit<<<1, nullptr, stream>>>(matrix, vector, workspace);
        else
            ctrsv_kernel_generic_n<<<1, nullptr, stream>>>(matrix, vector, workspace);
    } else if (trans == ACLBLAS_OP_T) {
        if (diag == ACLBLAS_UNIT)
            ctrsv_kernel_generic_t_unit<<<1, nullptr, stream>>>(matrix, vector, workspace);
        else
            ctrsv_kernel_generic_t<<<1, nullptr, stream>>>(matrix, vector, workspace);
    } else {
        if (diag == ACLBLAS_UNIT)
            ctrsv_kernel_generic_c_unit<<<1, nullptr, stream>>>(matrix, vector, workspace);
        else
            ctrsv_kernel_generic_c<<<1, nullptr, stream>>>(matrix, vector, workspace);
    }
}
