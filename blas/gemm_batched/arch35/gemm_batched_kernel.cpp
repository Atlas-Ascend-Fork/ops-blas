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
 * \file gemm_batched_kernel.cpp
 * \brief Batched GEMM kernel for arch35 (DAV_3510).
 *
 * Kernel 1: GEMM (AIC-only, tensor_api) - op(A[i]) @ op(B[i]) → temp (FP32)
 * Kernel 2: Alpha/Beta post-process (AIV-only, low-level API) - alpha*temp + beta*C → C
 *
 * Cube uses tensor_api (AscendC::Te); vector uses low-level AscendC API.
 */

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#define ASCENDC_CUBE_ONLY
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "gemm_batched_kernel.h"
#include "gemm_batched_tiling_data.h"
#include "gemm_batched_cube_impl.h"

template <class LayoutA, class LayoutB, uint32_t DtypeId>
__aicore__ inline void GbProcessOneStridedTask(
    __gm__ uint8_t* aBase, __gm__ uint8_t* bBase, __gm__ uint8_t* cBase, uint64_t strideA, uint64_t strideB,
    uint64_t strideC, uint32_t taskId, const GemmBatchedGemmTilingData& tiling, uint32_t baseK, uint32_t kL1Iter)
{
    using DType = typename GbDtypeTraits<DtypeId>::type;
    const GbTaskCoordinates task = GbResolveTaskCoordinates(taskId, tiling);
    const GbTaskBounds bounds = GbResolveTaskBounds(task, tiling);
    __gm__ DType* aGm = reinterpret_cast<__gm__ DType*>(aBase) + static_cast<uint64_t>(task.batch) * strideA;
    __gm__ DType* bGm = reinterpret_cast<__gm__ DType*>(bBase) + static_cast<uint64_t>(task.batch) * strideB;
    __gm__ float* cGm = reinterpret_cast<__gm__ float*>(cBase) + static_cast<uint64_t>(task.batch) * strideC;

    auto gmATensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGm),
        MakeGmLayout<DType>(LayoutA{}, tiling.m, tiling.k, tiling.lda));
    auto gmBTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGm),
        MakeGmLayout<DType>(LayoutB{}, tiling.k, tiling.n, tiling.ldb));
    auto gmCTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGm),
        MakeGmLayout<DType>(AscendC::Te::NDExtLayoutPtn{}, tiling.m, tiling.n, tiling.ldc));
    GbProcessMNTiles<DtypeId>(
        gmATensor, gmBTensor, gmCTensor, bounds.mStart, bounds.mEnd, bounds.nStart, bounds.nEnd, tiling.tileM,
        tiling.tileN, tiling.k, tiling.tileKChunk, baseK, kL1Iter);
}

template <class LayoutA, class LayoutB, uint32_t DtypeId>
__aicore__ inline void gemm_batched_strided_gemm_kernel_impl(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, __gm__ uint8_t* infoArray, uint64_t strideA,
    uint64_t strideB, uint64_t strideC, GemmBatchedGemmTilingData tiling)
{
    GbInitializeCubePipeline<DtypeId>();
    using DType = typename GbDtypeTraits<DtypeId>::type;
    const uint32_t baseK = (sizeof(DType) == sizeof(float)) ? GEMM_BATCHED_FP32_L0_BASE_K : GEMM_BATCHED_FP16_L0_BASE_K;
    const uint32_t kL1Iter = (tiling.k + tiling.tileKChunk - 1) / tiling.tileKChunk;

    AscendC::GlobalTensor<int> statuses;
    statuses.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(infoArray));

    for (uint32_t taskId = AscendC::GetBlockIdx(); taskId < tiling.totalTasks; taskId += AscendC::GetBlockNum()) {
        const GbTaskCoordinates task = GbResolveTaskCoordinates(taskId, tiling);
        if (statuses.GetValue(task.batch) != 0) {
            continue;
        }
        GbProcessOneStridedTask<LayoutA, LayoutB, DtypeId>(
            a, b, c, strideA, strideB, strideC, taskId, tiling, baseK, kL1Iter);
    }

    GbFinalizeCubePipeline();
}

// ── extern "C" kernel entry points for cube GEMM ──
#define GEMM_BATCHED_GEMM_KERNEL_ENTRY(FUNC_NAME, LAYOUT_A, LAYOUT_B, DTYPE_ID)                                   \
    extern "C" __global__ __aicore__ __cube__ void FUNC_NAME(                                                     \
        __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* carray, GemmBatchedGemmTilingData tiling) \
    {                                                                                                             \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                                           \
        gemm_batched_gemm_kernel_impl<LAYOUT_A, LAYOUT_B, DTYPE_ID>(aarray, barray, carray, tiling);              \
    }

// FLOAT16 output entry: fixpipe casts the FP32 accumulator to half on copy-out.
#define GEMM_BATCHED_GEMM_HALFOUT_KERNEL_ENTRY(FUNC_NAME, LAYOUT_A, LAYOUT_B, DTYPE_ID)                           \
    extern "C" __global__ __aicore__ __cube__ void FUNC_NAME(                                                     \
        __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* carray, GemmBatchedGemmTilingData tiling) \
    {                                                                                                             \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                                           \
        gemm_batched_gemm_kernel_impl_halfout<LAYOUT_A, LAYOUT_B, DTYPE_ID>(aarray, barray, carray, tiling);      \
    }

GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp32_nn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp32_nt, AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, 0)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp32_tn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp32_tt, AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, 0)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp16_nn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 1)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp16_nt, AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, 1)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp16_tn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 1)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp16_tt, AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, 1)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_bf16_nn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 2)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_bf16_nt, AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, 2)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_bf16_tn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 2)
GEMM_BATCHED_GEMM_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_bf16_tt, AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, 2)

GEMM_BATCHED_GEMM_HALFOUT_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp16out_nn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 1)
GEMM_BATCHED_GEMM_HALFOUT_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp16out_nt, AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, 1)
GEMM_BATCHED_GEMM_HALFOUT_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp16out_tn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 1)
GEMM_BATCHED_GEMM_HALFOUT_KERNEL_ENTRY(
    gemm_batched_gemm_kernel_fp16out_tt, AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, 1)

extern "C" __global__ __aicore__ __cube__ void cgetri_strided_gemm_kernel_fp32_nn(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, __gm__ uint8_t* infoArray, uint64_t strideA,
    uint64_t strideB, uint64_t strideC, GemmBatchedGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_batched_strided_gemm_kernel_impl<AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0>(
        a, b, c, infoArray, strideA, strideB, strideC, tiling);
}

// ── Kernel 1 launcher ──
// Launches through aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs so the
// runtime status of the launch is returned to the caller: the <<<>>> statement
// form cannot be used as an rvalue, and thread-wide last-error state may contain
// unrelated runtime calls. Same pattern as the arch22 cgemm_batched launchers
// and chbmv_kernel_do.
#define DEFINE_GEMM_DTYPE_LAUNCHER(LAUNCHER_NAME, KERNEL_PREFIX)                                                      \
    static inline aclError LAUNCHER_NAME(                                                                             \
        uint32_t numBlocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* carray,                                    \
        const GemmBatchedGemmTilingData& tilingData)                                                                  \
    {                                                                                                                 \
        bool isTransA = tilingData.isTransA != 0;                                                                     \
        bool isTransB = tilingData.isTransB != 0;                                                                     \
        const void* symbol = nullptr;                                                                                 \
        if (isTransA && isTransB) {                                                                                   \
            symbol = reinterpret_cast<const void*>(&KERNEL_PREFIX##_tt);                                              \
        } else if (isTransA) {                                                                                        \
            symbol = reinterpret_cast<const void*>(&KERNEL_PREFIX##_tn);                                              \
        } else if (isTransB) {                                                                                        \
            symbol = reinterpret_cast<const void*>(&KERNEL_PREFIX##_nt);                                              \
        } else {                                                                                                      \
            symbol = reinterpret_cast<const void*>(&KERNEL_PREFIX##_nn);                                              \
        }                                                                                                             \
        aclrtFuncHandle function = nullptr;                                                                           \
        const aclError lookup = aclrtGetFuncBySymbol(symbol, &function);                                              \
        if (lookup != ACL_SUCCESS) {                                                                                  \
            return lookup;                                                                                            \
        }                                                                                                             \
        struct Arguments {                                                                                            \
            uint8_t* a;                                                                                               \
            uint8_t* b;                                                                                               \
            uint8_t* carray;                                                                                          \
            GemmBatchedGemmTilingData tiling;                                                                         \
        };                                                                                                            \
        Arguments args{a, b, carray, tilingData};                                                                     \
        static_assert(offsetof(Arguments, tiling) == 3U * sizeof(uint8_t*), "kernel ABI mismatch");                   \
        return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U); \
    }

DEFINE_GEMM_DTYPE_LAUNCHER(LaunchGemmFp32, gemm_batched_gemm_kernel_fp32)
DEFINE_GEMM_DTYPE_LAUNCHER(LaunchGemmFp16, gemm_batched_gemm_kernel_fp16)
DEFINE_GEMM_DTYPE_LAUNCHER(LaunchGemmBf16, gemm_batched_gemm_kernel_bf16)
DEFINE_GEMM_DTYPE_LAUNCHER(LaunchGemmFp16Out, gemm_batched_gemm_kernel_fp16out)
#undef DEFINE_GEMM_DTYPE_LAUNCHER

aclError gemm_batched_gemm_kernel_do(
    uint32_t numBlocks, void* stream, const uint8_t* aarray, const uint8_t* barray, uint8_t* carray,
    const GemmBatchedGemmTilingData& tilingData)
{
    uint8_t* a = const_cast<uint8_t*>(aarray);
    uint8_t* b = const_cast<uint8_t*>(barray);
    switch (tilingData.dtypeCase) {
        case GEMM_BATCHED_DTYPE_FP32:
            return LaunchGemmFp32(numBlocks, stream, a, b, carray, tilingData);
        case GEMM_BATCHED_DTYPE_FP16:
            return LaunchGemmFp16(numBlocks, stream, a, b, carray, tilingData);
        case GEMM_BATCHED_DTYPE_BF16:
            return LaunchGemmBf16(numBlocks, stream, a, b, carray, tilingData);
        default:
            return ACL_ERROR_INVALID_PARAM;
    }
}

// FLOAT16-input, FLOAT16-output GEMM launcher (fixpipe cast, no FP32 temporary).
aclError gemm_batched_gemm_kernel_halfout_do(
    uint32_t numBlocks, void* stream, const uint8_t* aarray, const uint8_t* barray, uint8_t* carray,
    const GemmBatchedGemmTilingData& tilingData)
{
    uint8_t* a = const_cast<uint8_t*>(aarray);
    uint8_t* b = const_cast<uint8_t*>(barray);
    return LaunchGemmFp16Out(numBlocks, stream, a, b, carray, tilingData);
}

void cgetri_strided_gemm_kernel_do(
    uint32_t numBlocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* c, uint8_t* infoArray,
    uint64_t strideA, uint64_t strideB, uint64_t strideC, const GemmBatchedGemmTilingData& tilingData)
{
    cgetri_strided_gemm_kernel_fp32_nn<<<numBlocks, nullptr, stream>>>(
        const_cast<uint8_t*>(a), const_cast<uint8_t*>(b), c, infoArray, strideA, strideB, strideC, tilingData);
}

// ============================================================================
// Kernel 2: Alpha/Beta post-process (AIV-only, low-level API)
// C[j*ldc+i] = alpha * temp[j*tempRowStride+i] + beta * C_orig[j*ldc+i]
// ============================================================================

template <typename DstType, typename SrcType>
struct GbIsSameType {
    static constexpr bool value = false;
};
template <typename T>
struct GbIsSameType<T, T> {
    static constexpr bool value = true;
};

constexpr int32_t GB_AB_TILE = 256;
constexpr uint32_t GB_ALIGN_BYTES = GEMM_BATCHED_UB_ALIGN_BYTES;

template <typename T>
__aicore__ inline void GbAlignedCRead(
    AscendC::LocalTensor<T>& dst, AscendC::GlobalTensor<T>& src, uint64_t cOffset, uint32_t cnt)
{
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(cnt * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src[cOffset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void GbAlignedCWrite(
    AscendC::GlobalTensor<T>& dst, AscendC::LocalTensor<T>& result, uint64_t cOffset, uint32_t cnt)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(cnt * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst[cOffset], result, copyParams);
}

__aicore__ inline void GbComputeColRange(int64_t totalCols, int64_t& startCol, int64_t& endCol)
{
    uint32_t blockNum = AscendC::GetBlockNum();
    uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockNum > 1) {
        int64_t colsPerCore = (totalCols + blockNum - 1) / blockNum;
        startCol = static_cast<int64_t>(blockIdx) * colsPerCore;
        endCol = startCol + colsPerCore;
        if (endCol > totalCols) {
            endCol = totalCols;
        }
        if (startCol >= totalCols) {
            startCol = 0;
            endCol = 0;
        }
    } else {
        startCol = 0;
        endCol = totalCols;
    }
}

template <typename Op>
__aicore__ inline void GbIterateColTiles(Op& op, int64_t startCol, int64_t endCol, int32_t n, int32_t m)
{
    if (startCol >= endCol) {
        return;
    }
    int32_t prevBatch = -1;
    for (int64_t globalCol = startCol; globalCol < endCol; globalCol++) {
        int32_t batchIdx = static_cast<int32_t>(globalCol / n);
        int32_t col = static_cast<int32_t>(globalCol % n);
        if (batchIdx != prevBatch) {
            if (prevBatch >= 0) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            op.OnBatchEnter(batchIdx);
            prevBatch = batchIdx;
        }
        int32_t rowOffset = 0;
        while (rowOffset < m) {
            int32_t count = m - rowOffset;
            if (count > GB_AB_TILE) {
                count = GB_AB_TILE;
            }
            op.ProcessTile(col, rowOffset, count);
            rowOffset += count;
        }
    }
    if (prevBatch >= 0) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename Derived, uint32_t DtypeId>
class GemmBatchedColOpBase {
protected:
    using CType = typename GbDtypeTraits<DtypeId>::type;
    static constexpr bool IS_FP32 = GbIsSameType<CType, float>::value;
    static constexpr uint32_t C_ELEM_SIZE = GbDtypeTraits<DtypeId>::elemSize;
    static constexpr uint32_t ALIGN_ELEMS = GB_ALIGN_BYTES / sizeof(CType);
    static constexpr AscendC::RoundMode OUTPUT_ROUND =
        (DtypeId == 1) ? AscendC::RoundMode::CAST_ROUND : AscendC::RoundMode::CAST_RINT;

    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int64_t totalCols_ = 0;
    int64_t startCol_ = 0;
    int64_t endCol_ = 0;
    __gm__ uint64_t* cPtrArr_ = nullptr;

    __aicore__ inline void InitColRange() { GbComputeColRange(totalCols_, startCol_, endCol_); }

    __aicore__ inline __gm__ CType* GetCGmBatch(int32_t batchIdx)
    {
        return reinterpret_cast<__gm__ CType*>(cPtrArr_[batchIdx]);
    }

public:
    __aicore__ inline void Process() { GbIterateColTiles(static_cast<Derived&>(*this), startCol_, endCol_, n_, m_); }
};

template <uint32_t DtypeId>
class GemmBatchedAlphaBetaOp : public GemmBatchedColOpBase<GemmBatchedAlphaBetaOp<DtypeId>, DtypeId> {
    using Base = GemmBatchedColOpBase<GemmBatchedAlphaBetaOp<DtypeId>, DtypeId>;
    using CType = typename Base::CType;

public:
    __aicore__ inline void Init(
        __gm__ uint8_t* tempAB, __gm__ uint8_t* carray, GemmBatchedAlphaBetaTilingData tiling, AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        this->m_ = tiling.m;
        this->n_ = tiling.n;
        this->ldc_ = tiling.ldc;
        tempRowStride_ = tiling.tempRowStride;
        alpha_ = tiling.alpha;
        beta_ = tiling.beta;
        hasBeta_ = tiling.hasBeta;
        this->totalCols_ = tiling.totalCols;
        tempBase_ = reinterpret_cast<__gm__ float*>(tempAB);
        this->cPtrArr_ = reinterpret_cast<__gm__ uint64_t*>(carray);

        pipe_->InitBuffer(tempQue_, 2, GB_AB_TILE * sizeof(float) + GB_ALIGN_BYTES);
        pipe_->InitBuffer(outQue_, 2, GB_AB_TILE * sizeof(CType) + GB_ALIGN_BYTES);
        pipe_->InitBuffer(cOrigQue_, 2, GB_AB_TILE * sizeof(CType) + GB_ALIGN_BYTES);
        if constexpr (Base::IS_FP32) {
            if (hasBeta_) {
                pipe_->InitBuffer(scaledCQue_, 2, GB_AB_TILE * sizeof(float) + GB_ALIGN_BYTES);
            }
        } else {
            pipe_->InitBuffer(resultQue_, 2, GB_AB_TILE * sizeof(float) + GB_ALIGN_BYTES);
            if (hasBeta_) {
                pipe_->InitBuffer(cF32Que_, 2, GB_AB_TILE * sizeof(float) + GB_ALIGN_BYTES);
                pipe_->InitBuffer(scaledCQue_, 2, GB_AB_TILE * sizeof(float) + GB_ALIGN_BYTES);
            }
        }

        this->InitColRange();
    }

    __aicore__ inline void OnBatchEnter(int32_t batchIdx)
    {
        __gm__ CType* cGmBatch = this->GetCGmBatch(batchIdx);
        cBufTotalElems_ = static_cast<uint64_t>(this->ldc_) * this->n_;
        cOrigGlobal_.SetGlobalBuffer(cGmBatch, cBufTotalElems_);
        cOutGlobal_.SetGlobalBuffer(cGmBatch, cBufTotalElems_);
        __gm__ float* tempBatchBase = tempBase_ + static_cast<uint64_t>(batchIdx) * this->n_ * tempRowStride_;
        tempGlobal_.SetGlobalBuffer(tempBatchBase, static_cast<uint64_t>(tempRowStride_) * this->n_);
    }

    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t cnt = static_cast<uint32_t>(count);
        uint64_t tempOffset = static_cast<uint64_t>(col) * tempRowStride_ + rowOffset;
        uint64_t cOffset = static_cast<uint64_t>(col) * this->ldc_ + rowOffset;

        AscendC::LocalTensor<float> tempLocal = tempQue_.AllocTensor<float>();
        AscendC::DataCopyPadExtParams<float> tempPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams tempCopy{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(tempLocal, tempGlobal_[tempOffset], tempCopy, tempPad);
        tempQue_.EnQue(tempLocal);
        tempLocal = tempQue_.DeQue<float>();

        if constexpr (Base::IS_FP32) {
            ProcessTileFp32(tempLocal, cnt, cOffset);
        } else {
            ProcessTileNonFp32(tempLocal, cnt, cOffset);
        }

        tempQue_.FreeTensor(tempLocal);
    }

    __aicore__ inline void ProcessTileFp32(AscendC::LocalTensor<float>& tempLocal, uint32_t cnt, uint64_t cOffset)
    {
        AscendC::LocalTensor<float> result = outQue_.AllocTensor<float>();
        AscendC::Muls(result, tempLocal, alpha_, cnt);

        if (hasBeta_) {
            AscendC::LocalTensor<float> cOrigLocal = cOrigQue_.AllocTensor<float>();
            GbAlignedCRead<float>(cOrigLocal, cOrigGlobal_, cOffset, cnt);
            cOrigQue_.EnQue(cOrigLocal);
            cOrigLocal = cOrigQue_.DeQue<float>();

            AscendC::LocalTensor<float> scaledC = scaledCQue_.AllocTensor<float>();
            AscendC::Muls(scaledC, cOrigLocal, beta_, cnt);
            AscendC::Add(result, result, scaledC, cnt);
            scaledCQue_.FreeTensor(scaledC);
            cOrigQue_.FreeTensor(cOrigLocal);
        }

        outQue_.EnQue(result);
        AscendC::LocalTensor<float> outReady = outQue_.DeQue<float>();
        GbAlignedCWrite<float>(cOutGlobal_, outReady, cOffset, cnt);
        outQue_.FreeTensor(outReady);
    }

    __aicore__ inline void ProcessTileNonFp32(AscendC::LocalTensor<float>& tempLocal, uint32_t cnt, uint64_t cOffset)
    {
        AscendC::LocalTensor<float> result = resultQue_.AllocTensor<float>();
        AscendC::Muls(result, tempLocal, alpha_, cnt);

        if (hasBeta_) {
            AscendC::LocalTensor<CType> cOrigLocal = cOrigQue_.AllocTensor<CType>();
            GbAlignedCRead<CType>(cOrigLocal, cOrigGlobal_, cOffset, cnt);
            cOrigQue_.EnQue(cOrigLocal);
            cOrigLocal = cOrigQue_.DeQue<CType>();

            AscendC::LocalTensor<float> cF32 = cF32Que_.AllocTensor<float>();
            AscendC::Cast(cF32, cOrigLocal, AscendC::RoundMode::CAST_NONE, cnt);

            AscendC::LocalTensor<float> scaledC = scaledCQue_.AllocTensor<float>();
            AscendC::Muls(scaledC, cF32, beta_, cnt);
            AscendC::Add(result, result, scaledC, cnt);

            cF32Que_.FreeTensor(cF32);
            scaledCQue_.FreeTensor(scaledC);
            cOrigQue_.FreeTensor(cOrigLocal);
        }

        AscendC::LocalTensor<CType> outLocal = outQue_.AllocTensor<CType>();
        AscendC::Cast(outLocal, result, Base::OUTPUT_ROUND, cnt);
        AscendC::PipeBarrier<PIPE_ALL>();
        resultQue_.FreeTensor(result);

        outQue_.EnQue(outLocal);
        AscendC::LocalTensor<CType> outReady = outQue_.DeQue<CType>();
        GbAlignedCWrite<CType>(cOutGlobal_, outReady, cOffset, cnt);
        outQue_.FreeTensor(outReady);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> tempQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> cOrigQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> resultQue_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> cF32Que_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> scaledCQue_;
    AscendC::GlobalTensor<float> tempGlobal_;
    AscendC::GlobalTensor<CType> cOrigGlobal_;
    AscendC::GlobalTensor<CType> cOutGlobal_;
    int32_t tempRowStride_ = 0;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    int32_t hasBeta_ = 0;
    uint64_t cBufTotalElems_ = 0;
    __gm__ float* tempBase_ = nullptr;
};

// ── extern "C" kernel entry points for alpha/beta post-process ──
#define GEMM_BATCHED_AB_KERNEL_ENTRY(FUNC_NAME, DTYPE_ID)                                      \
    extern "C" __global__ __aicore__ void FUNC_NAME(                                           \
        __gm__ uint8_t* tempAB, __gm__ uint8_t* carray, GemmBatchedAlphaBetaTilingData tiling) \
    {                                                                                          \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                        \
        AscendC::TPipe pipe;                                                                   \
        GemmBatchedAlphaBetaOp<DTYPE_ID> op;                                                   \
        op.Init(tempAB, carray, tiling, &pipe);                                                \
        op.Process();                                                                          \
    }

GEMM_BATCHED_AB_KERNEL_ENTRY(gemm_batched_alpha_beta_kernel_fp32, 0)
GEMM_BATCHED_AB_KERNEL_ENTRY(gemm_batched_alpha_beta_kernel_fp16, 1)
GEMM_BATCHED_AB_KERNEL_ENTRY(gemm_batched_alpha_beta_kernel_bf16, 2)

// AIV launchers follow the same status-returning host-args pattern as the GEMM
// launcher above (the <<<>>> statement form cannot be used as an rvalue).
static inline aclError LaunchAlphaBetaKernel(
    uint32_t numBlocks, void* stream, uint8_t* tempABPtr, uint8_t* carray, int dtypeCase,
    const GemmBatchedAlphaBetaTilingData& tilingData)
{
    const void* symbol = nullptr;
    if (dtypeCase == GEMM_BATCHED_DTYPE_FP32) {
        symbol = reinterpret_cast<const void*>(&gemm_batched_alpha_beta_kernel_fp32);
    } else if (dtypeCase == GEMM_BATCHED_DTYPE_FP16) {
        symbol = reinterpret_cast<const void*>(&gemm_batched_alpha_beta_kernel_fp16);
    } else if (dtypeCase == GEMM_BATCHED_DTYPE_BF16) {
        symbol = reinterpret_cast<const void*>(&gemm_batched_alpha_beta_kernel_bf16);
    } else {
        return ACL_ERROR_INVALID_PARAM;
    }
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(symbol, &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        uint8_t* tempAB;
        uint8_t* carray;
        GemmBatchedAlphaBetaTilingData tiling;
    };
    Arguments args{tempABPtr, carray, tilingData};
    static_assert(offsetof(Arguments, tiling) == 2U * sizeof(uint8_t*), "kernel ABI mismatch");
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}

aclError gemm_batched_alpha_beta_kernel_do(
    uint32_t numBlocks, void* stream, const uint8_t* tempAB, uint8_t* carray,
    const GemmBatchedAlphaBetaTilingData& tilingData)
{
    return LaunchAlphaBetaKernel(
        numBlocks, stream, const_cast<uint8_t*>(tempAB), carray, tilingData.dtypeCase, tilingData);
}

// ============================================================================
// Kernel 3: Early-exit (AIV-only, low-level API)
// Handles k=0 or alpha=0: C = beta * C_orig
// ============================================================================
template <uint32_t DtypeId>
class GemmBatchedEarlyExitOp : public GemmBatchedColOpBase<GemmBatchedEarlyExitOp<DtypeId>, DtypeId> {
    using Base = GemmBatchedColOpBase<GemmBatchedEarlyExitOp<DtypeId>, DtypeId>;
    using CType = typename Base::CType;

public:
    __aicore__ inline void Init(__gm__ uint8_t* carray, GemmBatchedAlphaBetaTilingData tiling, AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        this->m_ = tiling.m;
        this->n_ = tiling.n;
        this->ldc_ = tiling.ldc;
        beta_ = tiling.beta;
        this->totalCols_ = tiling.totalCols;
        this->cPtrArr_ = reinterpret_cast<__gm__ uint64_t*>(carray);

        pipe_->InitBuffer(inQue_, 2, GB_AB_TILE * sizeof(CType) + GB_ALIGN_BYTES);
        pipe_->InitBuffer(outQue_, 2, GB_AB_TILE * sizeof(CType) + GB_ALIGN_BYTES);
        if constexpr (!Base::IS_FP32) {
            pipe_->InitBuffer(calcQue_, 2, GB_AB_TILE * sizeof(float) + GB_ALIGN_BYTES);
        }

        this->InitColRange();
    }

    __aicore__ inline void OnBatchEnter(int32_t batchIdx)
    {
        __gm__ CType* cGmBatch = this->GetCGmBatch(batchIdx);
        cBufTotalElems_ = static_cast<uint64_t>(this->ldc_) * this->n_;
        cGlobal_.SetGlobalBuffer(cGmBatch, cBufTotalElems_);
    }

    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t cnt = static_cast<uint32_t>(count);
        uint64_t cOffset = static_cast<uint64_t>(col) * this->ldc_ + rowOffset;

        // beta == 0 must zero C without reading it, so stale Inf/NaN contents cannot
        // propagate (0 * Inf = NaN). Aligns with the alpha/beta kernel hasBeta==0 path.
        if (beta_ == 0.0f) {
            AscendC::LocalTensor<CType> zeroTile = outQue_.AllocTensor<CType>();
            if constexpr (Base::IS_FP32) {
                AscendC::Duplicate(zeroTile, 0.0f, cnt);
            } else {
                AscendC::LocalTensor<float> fp32Tile = calcQue_.AllocTensor<float>();
                AscendC::Duplicate(fp32Tile, 0.0f, cnt);
                AscendC::Cast(zeroTile, fp32Tile, Base::OUTPUT_ROUND, cnt);
                AscendC::PipeBarrier<PIPE_ALL>();
                calcQue_.FreeTensor(fp32Tile);
            }
            outQue_.EnQue(zeroTile);
            AscendC::LocalTensor<CType> zeroReady = outQue_.DeQue<CType>();
            GbAlignedCWrite<CType>(cGlobal_, zeroReady, cOffset, cnt);
            outQue_.FreeTensor(zeroReady);
            return;
        }

        AscendC::LocalTensor<CType> inTile = inQue_.AllocTensor<CType>();
        GbAlignedCRead<CType>(inTile, cGlobal_, cOffset, cnt);
        inQue_.EnQue(inTile);
        AscendC::LocalTensor<CType> procTile = inQue_.DeQue<CType>();

        if constexpr (Base::IS_FP32) {
            AscendC::LocalTensor<float> outTile = outQue_.AllocTensor<float>();
            AscendC::Muls(outTile, procTile, beta_, cnt);
            outQue_.EnQue(outTile);
            AscendC::LocalTensor<float> outReady = outQue_.DeQue<float>();
            GbAlignedCWrite<CType>(cGlobal_, outReady, cOffset, cnt);
            outQue_.FreeTensor(outReady);
        } else {
            AscendC::LocalTensor<float> fp32Tile = calcQue_.AllocTensor<float>();
            AscendC::Cast(fp32Tile, procTile, AscendC::RoundMode::CAST_NONE, cnt);
            AscendC::Muls(fp32Tile, fp32Tile, beta_, cnt);

            AscendC::LocalTensor<CType> outTile = outQue_.AllocTensor<CType>();
            AscendC::Cast(outTile, fp32Tile, Base::OUTPUT_ROUND, cnt);
            AscendC::PipeBarrier<PIPE_ALL>();
            calcQue_.FreeTensor(fp32Tile);

            outQue_.EnQue(outTile);
            AscendC::LocalTensor<CType> outReady = outQue_.DeQue<CType>();
            GbAlignedCWrite<CType>(cGlobal_, outReady, cOffset, cnt);
            outQue_.FreeTensor(outReady);
        }
        inQue_.FreeTensor(procTile);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> calcQue_;
    AscendC::GlobalTensor<CType> cGlobal_;
    uint64_t cBufTotalElems_ = 0;
    float beta_ = 0.0f;
};

// ── extern "C" kernel entry points for early-exit ──
#define GEMM_BATCHED_EE_KERNEL_ENTRY(FUNC_NAME, DTYPE_ID)                                                          \
    extern "C" __global__ __aicore__ void FUNC_NAME(__gm__ uint8_t* carray, GemmBatchedAlphaBetaTilingData tiling) \
    {                                                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                            \
        AscendC::TPipe pipe;                                                                                       \
        GemmBatchedEarlyExitOp<DTYPE_ID> op;                                                                       \
        op.Init(carray, tiling, &pipe);                                                                            \
        op.Process();                                                                                              \
    }

GEMM_BATCHED_EE_KERNEL_ENTRY(gemm_batched_early_exit_kernel_fp32, 0)
GEMM_BATCHED_EE_KERNEL_ENTRY(gemm_batched_early_exit_kernel_fp16, 1)
GEMM_BATCHED_EE_KERNEL_ENTRY(gemm_batched_early_exit_kernel_bf16, 2)

aclError gemm_batched_early_exit_kernel_do(
    uint32_t numBlocks, void* stream, uint8_t* carray, const GemmBatchedAlphaBetaTilingData& tilingData)
{
    const void* symbol = nullptr;
    if (tilingData.dtypeCase == GEMM_BATCHED_DTYPE_FP32) {
        symbol = reinterpret_cast<const void*>(&gemm_batched_early_exit_kernel_fp32);
    } else if (tilingData.dtypeCase == GEMM_BATCHED_DTYPE_FP16) {
        symbol = reinterpret_cast<const void*>(&gemm_batched_early_exit_kernel_fp16);
    } else if (tilingData.dtypeCase == GEMM_BATCHED_DTYPE_BF16) {
        symbol = reinterpret_cast<const void*>(&gemm_batched_early_exit_kernel_bf16);
    } else {
        return ACL_ERROR_INVALID_PARAM;
    }
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(symbol, &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        uint8_t* carray;
        GemmBatchedAlphaBetaTilingData tiling;
    };
    Arguments args{carray, tilingData};
    static_assert(offsetof(Arguments, tiling) == sizeof(uint8_t*), "kernel ABI mismatch");
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}

// ============================================================================
// Complex GEMM support kernels (AIV-only, low-level API)
//
// cgemm_batched is decomposed via the 4M algorithm:
//   A = Ar + j*Ai,  B = Br + j*Bi
//   op(A)*op(B) = (Ar*Br - Ai*Bi) + j*(Ar*Bi + Ai*Br)
//
// Kernel 4: Deinterleave — split interleaved complex matrix into real/imag float matrices.
// Kernel 5: Combine — combine four real GEMM results into final complex C with alpha/beta.
//
// Deinterleave/interleave uses vectorized Gather/Scatter with a pre-built
// byte-offset table (even/odd element byte offsets). The offset table is built
// once in Init via GbBuildGatherOffsetTable (Scalar base + Adds vector extension).
// ============================================================================

constexpr int32_t CGEMM_BATCHED_TILE = 256;
constexpr uint32_t CGEMM_BATCHED_ALIGN = GEMM_BATCHED_UB_ALIGN_BYTES;

// Build Gather/Scatter byte-offset tables using vectorized Adds instead of
// per-element SetValue.  evenOff[i] = i*2*sizeof(float), oddOff[i] = (i*2+1)*sizeof(float).
// Pattern: Scalar SetValue base group (8 int32 = 32B) + Adds vector extension.
// Uses int32_t for Adds (uint32_t not supported), then caller reinterprets as uint32_t.
__aicore__ inline void GbBuildGatherOffsetTable(
    AscendC::LocalTensor<int32_t>& evenOff, AscendC::LocalTensor<int32_t>& oddOff, uint32_t count)
{
    constexpr uint32_t VEC_GROUP = 8;                                   // 8 × int32 = 32B (Adds alignment)
    constexpr int32_t STRIDE = 2 * static_cast<int32_t>(sizeof(float)); // = 8
    for (uint32_t i = 0; i < VEC_GROUP; i++) {
        evenOff.SetValue(i, static_cast<int32_t>(i) * STRIDE);
    }
    uint32_t numGroups = count / VEC_GROUP;
    for (uint32_t g = 1; g < numGroups; g++) {
        AscendC::Adds(
            evenOff[g * VEC_GROUP], evenOff, static_cast<int32_t>(g * VEC_GROUP) * STRIDE,
            static_cast<int32_t>(VEC_GROUP));
    }
    AscendC::Adds(oddOff, evenOff, static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void GbInitGatherOffsetBuffer(
    AscendC::TPipe* pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& idxBuf, AscendC::LocalTensor<uint32_t>& evenOff,
    AscendC::LocalTensor<uint32_t>& oddOff)
{
    uint32_t idxBufSize = 2 * CGEMM_BATCHED_TILE * sizeof(uint32_t) + CGEMM_BATCHED_ALIGN;
    pipe->InitBuffer(idxBuf, idxBufSize);
    {
        auto evenOffI32 = idxBuf.GetWithOffset<int32_t>(CGEMM_BATCHED_TILE, 0);
        auto oddOffI32 = idxBuf.GetWithOffset<int32_t>(CGEMM_BATCHED_TILE, CGEMM_BATCHED_TILE * sizeof(uint32_t));
        GbBuildGatherOffsetTable(evenOffI32, oddOffI32, CGEMM_BATCHED_TILE);
    }
    evenOff = idxBuf.GetWithOffset<uint32_t>(CGEMM_BATCHED_TILE, 0);
    oddOff = idxBuf.GetWithOffset<uint32_t>(CGEMM_BATCHED_TILE, CGEMM_BATCHED_TILE * sizeof(uint32_t));
}

// Column-wise deinterleave: splits interleaved complex matrix into real/imag.
class CgemmBatchedDeinterleaveColOp {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* srcArray, __gm__ uint8_t* realArray, __gm__ uint8_t* imagArray,
        CgemmBatchedDeinterleaveTilingData tiling, AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        physRows_ = tiling.m;
        physCols_ = tiling.k;
        ld_ = tiling.lda;
        batchCount_ = tiling.batchCount;
        isConjugate_ = (tiling.isConjugate != 0);
        srcPtrArr_ = reinterpret_cast<__gm__ uint64_t*>(srcArray);
        realPtrArr_ = reinterpret_cast<__gm__ uint64_t*>(realArray);
        imagPtrArr_ = reinterpret_cast<__gm__ uint64_t*>(imagArray);

        uint32_t tileBytes = CGEMM_BATCHED_TILE * sizeof(float) + CGEMM_BATCHED_ALIGN;
        pipe_->InitBuffer(cplxInQue_, 2, tileBytes);
        pipe_->InitBuffer(realOutQue_, 2, tileBytes);
        pipe_->InitBuffer(imagOutQue_, 2, tileBytes);

        GbInitGatherOffsetBuffer(pipe_, idxBuf_, evenOff_, oddOff_);

        totalCols_ = static_cast<int64_t>(batchCount_) * physCols_;
        GbComputeColRange(totalCols_, startCol_, endCol_);
    }

    __aicore__ inline void Process() { GbIterateColTiles(*this, startCol_, endCol_, physCols_, physRows_); }

    __aicore__ inline void OnBatchEnter(int32_t batchIdx)
    {
        __gm__ aclblasComplex* cplxGm = reinterpret_cast<__gm__ aclblasComplex*>(srcPtrArr_[batchIdx]);
        __gm__ float* realGm = reinterpret_cast<__gm__ float*>(realPtrArr_[batchIdx]);
        __gm__ float* imagGm = reinterpret_cast<__gm__ float*>(imagPtrArr_[batchIdx]);
        uint64_t totalElems = static_cast<uint64_t>(ld_) * physCols_;
        cplxFloatGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cplxGm), totalElems * 2);
        realGlobal_.SetGlobalBuffer(realGm, totalElems);
        imagGlobal_.SetGlobalBuffer(imagGm, totalElems);
    }

    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t cnt = static_cast<uint32_t>(count);
        uint64_t baseFloatOffset = (static_cast<uint64_t>(col) * ld_ + rowOffset) * 2;

        AscendC::LocalTensor<float> cplxLocal = cplxInQue_.AllocTensor<float>();
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(cnt * 2 * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(cplxLocal, cplxFloatGlobal_[baseFloatOffset], copyParams, padParams);
        cplxInQue_.EnQue(cplxLocal);
        cplxLocal = cplxInQue_.DeQue<float>();

        AscendC::LocalTensor<float> realLocal = realOutQue_.AllocTensor<float>();
        AscendC::LocalTensor<float> imagLocal = imagOutQue_.AllocTensor<float>();
        AscendC::Gather(realLocal, cplxLocal, evenOff_, 0U, cnt);
        AscendC::Gather(imagLocal, cplxLocal, oddOff_, 0U, cnt);
        if (isConjugate_) {
            AscendC::Muls(imagLocal, imagLocal, -1.0f, cnt);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        cplxInQue_.FreeTensor(cplxLocal);

        uint64_t outOffset = static_cast<uint64_t>(col) * ld_ + rowOffset;
        AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> outPad{false, 0, 0, 0};

        realOutQue_.EnQue(realLocal);
        AscendC::LocalTensor<float> realReady = realOutQue_.DeQue<float>();
        GbAlignedCWrite(realGlobal_, realReady, outOffset, cnt);
        realOutQue_.FreeTensor(realReady);

        imagOutQue_.EnQue(imagLocal);
        AscendC::LocalTensor<float> imagReady = imagOutQue_.DeQue<float>();
        GbAlignedCWrite(imagGlobal_, imagReady, outOffset, cnt);
        imagOutQue_.FreeTensor(imagReady);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> cplxInQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> realOutQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> imagOutQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf_;
    AscendC::LocalTensor<uint32_t> evenOff_;
    AscendC::LocalTensor<uint32_t> oddOff_;
    AscendC::GlobalTensor<float> cplxFloatGlobal_;
    AscendC::GlobalTensor<float> realGlobal_;
    AscendC::GlobalTensor<float> imagGlobal_;
    __gm__ uint64_t* srcPtrArr_ = nullptr;
    __gm__ uint64_t* realPtrArr_ = nullptr;
    __gm__ uint64_t* imagPtrArr_ = nullptr;
    int32_t physRows_ = 0;
    int32_t physCols_ = 0;
    int32_t ld_ = 0;
    int32_t batchCount_ = 0;
    bool isConjugate_ = false;
    int64_t totalCols_ = 0;
    int64_t startCol_ = 0;
    int64_t endCol_ = 0;
};

extern "C" __global__ __aicore__ void cgemm_batched_deinterleave_col_kernel(
    __gm__ uint8_t* srcArray, __gm__ uint8_t* realArray, __gm__ uint8_t* imagArray,
    CgemmBatchedDeinterleaveTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    CgemmBatchedDeinterleaveColOp op;
    op.Init(srcArray, realArray, imagArray, tiling, &pipe);
    op.Process();
}

// ============================================================================
// Kernel 5: Combine — T1,T2,T3,T4 (real float, stride=tempRowStride) → complex C
// ============================================================================
class CgemmBatchedCombineOp {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* t1Array, __gm__ uint8_t* t2Array, __gm__ uint8_t* t3Array, __gm__ uint8_t* t4Array,
        __gm__ uint8_t* carray, CgemmBatchedCombineTilingData tiling, AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = tiling.m;
        n_ = tiling.n;
        ldc_ = tiling.ldc;
        tempRowStride_ = tiling.tempRowStride;
        alphaR_ = tiling.alphaReal;
        alphaI_ = tiling.alphaImag;
        betaR_ = tiling.betaReal;
        betaI_ = tiling.betaImag;
        hasBeta_ = tiling.hasBeta;
        batchCount_ = tiling.batchCount;
        t1PtrArr_ = reinterpret_cast<__gm__ uint64_t*>(t1Array);
        t2PtrArr_ = reinterpret_cast<__gm__ uint64_t*>(t2Array);
        t3PtrArr_ = reinterpret_cast<__gm__ uint64_t*>(t3Array);
        t4PtrArr_ = reinterpret_cast<__gm__ uint64_t*>(t4Array);
        cPtrArr_ = reinterpret_cast<__gm__ uint64_t*>(carray);

        uint32_t tileBytes = CGEMM_BATCHED_TILE * sizeof(float) + CGEMM_BATCHED_ALIGN;
        pipe_->InitBuffer(t1Que_, 2, tileBytes);
        pipe_->InitBuffer(t2Que_, 2, tileBytes);
        pipe_->InitBuffer(t3Que_, 2, tileBytes);
        pipe_->InitBuffer(t4Que_, 2, tileBytes);
        pipe_->InitBuffer(cOrigQue_, 2, CGEMM_BATCHED_TILE * 2 * sizeof(float) + CGEMM_BATCHED_ALIGN);
        pipe_->InitBuffer(prQue_, 2, tileBytes);
        pipe_->InitBuffer(piQue_, 2, tileBytes);
        pipe_->InitBuffer(crQue_, 2, tileBytes);
        pipe_->InitBuffer(ciQue_, 2, tileBytes);
        pipe_->InitBuffer(outQue_, 2, CGEMM_BATCHED_TILE * 2 * sizeof(float) + CGEMM_BATCHED_ALIGN);

        GbInitGatherOffsetBuffer(pipe_, idxBuf_, evenOff_, oddOff_);

        totalCols_ = tiling.totalCols;
        GbComputeColRange(totalCols_, startCol_, endCol_);
    }

    __aicore__ inline void Process() { GbIterateColTiles(*this, startCol_, endCol_, n_, m_); }

    __aicore__ inline void OnBatchEnter(int32_t batchIdx)
    {
        __gm__ float* t1Gm = reinterpret_cast<__gm__ float*>(t1PtrArr_[batchIdx]);
        __gm__ float* t2Gm = reinterpret_cast<__gm__ float*>(t2PtrArr_[batchIdx]);
        __gm__ float* t3Gm = reinterpret_cast<__gm__ float*>(t3PtrArr_[batchIdx]);
        __gm__ float* t4Gm = reinterpret_cast<__gm__ float*>(t4PtrArr_[batchIdx]);
        __gm__ aclblasComplex* cGm = reinterpret_cast<__gm__ aclblasComplex*>(cPtrArr_[batchIdx]);
        uint64_t tempTotal = static_cast<uint64_t>(tempRowStride_) * n_;
        uint64_t cTotal = static_cast<uint64_t>(ldc_) * n_;
        t1Global_.SetGlobalBuffer(t1Gm, tempTotal);
        t2Global_.SetGlobalBuffer(t2Gm, tempTotal);
        t3Global_.SetGlobalBuffer(t3Gm, tempTotal);
        t4Global_.SetGlobalBuffer(t4Gm, tempTotal);
        cFloatGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm), cTotal * 2);
    }

    __aicore__ inline void ApplyBeta(
        AscendC::LocalTensor<float>& cr, AscendC::LocalTensor<float>& ci, uint32_t cnt, uint64_t cFloatOffset,
        AscendC::DataCopyPadExtParams<float>& pad)
    {
        AscendC::LocalTensor<float> cOrig = cOrigQue_.AllocTensor<float>();
        AscendC::DataCopyExtParams cParams{1, static_cast<uint32_t>(cnt * 2 * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(cOrig, cFloatGlobal_[cFloatOffset], cParams, pad);
        cOrigQue_.EnQue(cOrig);
        cOrig = cOrigQue_.DeQue<float>();

        AscendC::LocalTensor<float> coR = prQue_.AllocTensor<float>();
        AscendC::LocalTensor<float> coI = piQue_.AllocTensor<float>();
        AscendC::Gather(coR, cOrig, evenOff_, 0U, cnt);
        AscendC::Gather(coI, cOrig, oddOff_, 0U, cnt);
        AscendC::PipeBarrier<PIPE_ALL>();
        cOrigQue_.FreeTensor(cOrig);

        AscendC::LocalTensor<float> sR = t1Que_.AllocTensor<float>();
        AscendC::LocalTensor<float> sI = t2Que_.AllocTensor<float>();
        AscendC::Muls(sR, coR, betaR_, cnt);
        AscendC::Muls(sI, coI, betaR_, cnt);
        AscendC::Add(cr, cr, sR, cnt);
        AscendC::Add(ci, ci, sI, cnt);
        AscendC::Muls(sR, coI, betaI_, cnt);
        AscendC::Muls(sI, coR, betaI_, cnt);
        AscendC::Sub(cr, cr, sR, cnt);
        AscendC::Add(ci, ci, sI, cnt);
        t1Que_.FreeTensor(sR);
        t2Que_.FreeTensor(sI);
        prQue_.FreeTensor(coR);
        piQue_.FreeTensor(coI);
    }

    __aicore__ inline void StoreOutput(
        AscendC::LocalTensor<float>& cr, AscendC::LocalTensor<float>& ci, uint32_t cnt, uint64_t cFloatOffset)
    {
        AscendC::LocalTensor<float> out = outQue_.AllocTensor<float>();
        AscendC::Scatter(out, cr, evenOff_, 0U, cnt);
        AscendC::Scatter(out, ci, oddOff_, 0U, cnt);
        AscendC::PipeBarrier<PIPE_ALL>();
        crQue_.FreeTensor(cr);
        ciQue_.FreeTensor(ci);

        outQue_.EnQue(out);
        AscendC::LocalTensor<float> outReady = outQue_.DeQue<float>();
        GbAlignedCWrite(cFloatGlobal_, outReady, cFloatOffset, cnt * 2);
        outQue_.FreeTensor(outReady);
    }

    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t cnt = static_cast<uint32_t>(count);
        uint64_t tempOffset = static_cast<uint64_t>(col) * tempRowStride_ + rowOffset;
        uint64_t cFloatOffset = (static_cast<uint64_t>(col) * ldc_ + rowOffset) * 2;

        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyExtParams inParams{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};

        AscendC::LocalTensor<float> t1 = t1Que_.AllocTensor<float>();
        GbAlignedCRead(t1, t1Global_, tempOffset, cnt);
        t1Que_.EnQue(t1);
        t1 = t1Que_.DeQue<float>();

        AscendC::LocalTensor<float> t2 = t2Que_.AllocTensor<float>();
        GbAlignedCRead(t2, t2Global_, tempOffset, cnt);
        t2Que_.EnQue(t2);
        t2 = t2Que_.DeQue<float>();

        AscendC::LocalTensor<float> t3 = t3Que_.AllocTensor<float>();
        GbAlignedCRead(t3, t3Global_, tempOffset, cnt);
        t3Que_.EnQue(t3);
        t3 = t3Que_.DeQue<float>();

        AscendC::LocalTensor<float> t4 = t4Que_.AllocTensor<float>();
        GbAlignedCRead(t4, t4Global_, tempOffset, cnt);
        t4Que_.EnQue(t4);
        t4 = t4Que_.DeQue<float>();

        AscendC::LocalTensor<float> pr = prQue_.AllocTensor<float>();
        AscendC::Sub(pr, t1, t2, cnt);
        AscendC::LocalTensor<float> pi = piQue_.AllocTensor<float>();
        AscendC::Add(pi, t3, t4, cnt);
        t1Que_.FreeTensor(t1);
        t2Que_.FreeTensor(t2);
        t3Que_.FreeTensor(t3);
        t4Que_.FreeTensor(t4);

        AscendC::LocalTensor<float> cr = crQue_.AllocTensor<float>();
        AscendC::LocalTensor<float> ci = ciQue_.AllocTensor<float>();
        AscendC::Muls(cr, pr, alphaR_, cnt);
        AscendC::LocalTensor<float> tmp = prQue_.AllocTensor<float>();
        AscendC::Muls(tmp, pi, alphaI_, cnt);
        AscendC::Sub(cr, cr, tmp, cnt);
        AscendC::Muls(ci, pi, alphaR_, cnt);
        AscendC::Muls(tmp, pr, alphaI_, cnt);
        AscendC::Add(ci, ci, tmp, cnt);
        prQue_.FreeTensor(tmp);
        prQue_.FreeTensor(pr);
        piQue_.FreeTensor(pi);

        if (hasBeta_) {
            ApplyBeta(cr, ci, cnt, cFloatOffset, pad);
        }
        StoreOutput(cr, ci, cnt, cFloatOffset);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> t1Que_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> t2Que_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> t3Que_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> t4Que_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> cOrigQue_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> prQue_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> piQue_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> crQue_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 2> ciQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf_;
    AscendC::LocalTensor<uint32_t> evenOff_;
    AscendC::LocalTensor<uint32_t> oddOff_;
    AscendC::GlobalTensor<float> t1Global_;
    AscendC::GlobalTensor<float> t2Global_;
    AscendC::GlobalTensor<float> t3Global_;
    AscendC::GlobalTensor<float> t4Global_;
    AscendC::GlobalTensor<float> cFloatGlobal_;
    __gm__ uint64_t* t1PtrArr_ = nullptr;
    __gm__ uint64_t* t2PtrArr_ = nullptr;
    __gm__ uint64_t* t3PtrArr_ = nullptr;
    __gm__ uint64_t* t4PtrArr_ = nullptr;
    __gm__ uint64_t* cPtrArr_ = nullptr;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int32_t tempRowStride_ = 0;
    float alphaR_ = 1.0f;
    float alphaI_ = 0.0f;
    float betaR_ = 0.0f;
    float betaI_ = 0.0f;
    int32_t hasBeta_ = 0;
    int32_t batchCount_ = 0;
    int64_t totalCols_ = 0;
    int64_t startCol_ = 0;
    int64_t endCol_ = 0;
};

extern "C" __global__ __aicore__ void cgemm_batched_combine_kernel(
    __gm__ uint8_t* t1Array, __gm__ uint8_t* t2Array, __gm__ uint8_t* t3Array, __gm__ uint8_t* t4Array,
    __gm__ uint8_t* carray, CgemmBatchedCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    CgemmBatchedCombineOp op;
    op.Init(t1Array, t2Array, t3Array, t4Array, carray, tiling, &pipe);
    op.Process();
}

void cgemm_batched_deinterleave_do(
    uint32_t numBlocks, void* stream, uint8_t* srcArray, uint8_t* realArray, uint8_t* imagArray,
    const CgemmBatchedDeinterleaveTilingData& tiling)
{
    cgemm_batched_deinterleave_col_kernel<<<numBlocks, nullptr, stream>>>(srcArray, realArray, imagArray, tiling);
}

void cgemm_batched_combine_do(
    uint32_t numBlocks, void* stream, uint8_t* t1Array, uint8_t* t2Array, uint8_t* t3Array, uint8_t* t4Array,
    uint8_t* carray, const CgemmBatchedCombineTilingData& tiling)
{
    cgemm_batched_combine_kernel<<<numBlocks, nullptr, stream>>>(t1Array, t2Array, t3Array, t4Array, carray, tiling);
}
