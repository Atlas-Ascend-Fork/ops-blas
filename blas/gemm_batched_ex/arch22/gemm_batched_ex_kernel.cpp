/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define ASCENDC_CUBE_ONLY
#include "acl/acl.h"
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "gemm_batched_ex_tiling.h"
#include "gemm_batched_ex_fp32.h"

using namespace AscendC;
namespace {
using GemmBatchedExFp32::AccumulateReference;
using GemmBatchedExFp32::FusedFloat;
using GemmBatchedExFp32::REFERENCE_CAPACITY;
using GemmBatchedExFp32::ReferenceScalars;
using GemmBatchedExFp32::StoreReference;
using GemmBatchedExFp32::SyncEvent;
using Matrix = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulShapeParams SHAPE{256, 256, 4096, 128, 128, 32};
constexpr MatmulFuncParams FUNC{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                true,  true};
constexpr MatmulBiasParams BIAS{false};
constexpr MatmulConfig CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(SHAPE, FUNC, BIAS);
constexpr MatmulShapeParams RIGHT_SHAPE{256, 256, 4096, 128, 128, 64};
constexpr MatmulConfig RIGHT_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(RIGHT_SHAPE, FUNC, BIAS);
template <bool T, bool NZ>
using Input = matmul::MatmulType<TPosition::GM, NZ ? CubeFormat::NZ : CubeFormat::ND, float, T>;
constexpr MatmulShapeParams RAW_SHAPE{128, 128, 4096, 128, 128, 64};
constexpr MatmulConfig RAW_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(RAW_SHAPE, FUNC, BIAS);
template <bool TA, bool TB, bool NZ>
constexpr MatmulApiStaticTiling TILING = matmul::GetMatmulApiTiling<Input<TB, NZ>, Input<TA, NZ>, Matrix, Matrix>(
    !NZ ? RAW_CONFIG : (TA && !TB ? RIGHT_CONFIG : CONFIG));
template <bool TA, bool TB, bool NZ>
using MM = matmul::MatmulImpl<Input<TB, NZ>, Input<TA, NZ>, Matrix, Matrix, TILING<TA, TB, NZ>>;

__aicore__ inline float Alpha(const GemmBatchedExArch22Tiling& t)
{
    return t.alphaDevice ? *reinterpret_cast<__gm__ float*>(t.alphaAddress) : t.alpha;
}

__aicore__ inline float Beta(const GemmBatchedExArch22Tiling& t)
{
    return t.betaDevice ? *reinterpret_cast<__gm__ float*>(t.betaAddress) : t.beta;
}

// Explicit register writes disable the compiler's automatic FFTS initialization.
// Both MIX entries therefore install the runtime address and restore it on exit.
__aicore__ inline void FinishCrossCore()
{
    PipeBarrier<PIPE_ALL>();
    SetSyncBaseAddr(0);
}

__aicore__ inline void BindMatrix(GlobalTensor<float>& matrix, const GlobalTensor<uint64_t>& pointers, uint32_t batch)
{
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pointers.GetValue(batch)));
}

// Each batch owns a 64-byte policy flag followed by per-core statistics.
// Selection uses only the live device matrices; batches are independent.
__aicore__ inline bool NonFiniteStatistic(float value)
{
    union Bits {
        float value;
        uint32_t bits;
    } bits;
    bits.value = value;
    return (bits.bits & 0x7f800000U) == 0x7f800000U;
}

__aicore__ inline bool ReferenceRequired(
    const GemmBatchedExArch22Tiling& t, uint32_t batch, uint32_t& previousBatch, bool& previousValue)
{
    if (t.policy == 0)
        return false;
    if (previousBatch != batch) {
        GlobalTensor<uint32_t> flags;
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(t.policy));
        const uint64_t offset = static_cast<uint64_t>(batch) * 16U;
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE>(flags[offset]);
        // Scalar instruction ordering is guaranteed by hardware.
        previousValue = flags.GetValue(offset) != 0;
        previousBatch = batch;
    }
    return previousValue;
}

__aicore__ inline void AddStatisticsTile(
    const GlobalTensor<float>& matrix, uint64_t offset, uint32_t count, const LocalTensor<float>& input,
    const LocalTensor<float>& absolute, const LocalTensor<float>& work, const LocalTensor<float>& reduction, float& sum,
    float& absoluteSum)
{
    const DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
    const DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    DataCopyPad(input, matrix[offset], copy, padding);
    SyncEvent<HardEvent::MTE2_V>();
    ReduceSum(reduction, input, work, static_cast<int32_t>(count));
    SyncEvent<HardEvent::V_S>();
    sum += reduction.GetValue(0);
    SyncEvent<HardEvent::S_V>();
    Abs(absolute, input, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    ReduceSum(reduction, absolute, work, static_cast<int32_t>(count));
    SyncEvent<HardEvent::V_S>();
    absoluteSum += reduction.GetValue(0);
    SyncEvent<HardEvent::S_V>();
}

__aicore__ inline void ReadStatistics(
    const GlobalTensor<float>& matrix, uint32_t rows, uint32_t columns, uint32_t leading,
    const LocalTensor<float>& input, const LocalTensor<float>& absolute, const LocalTensor<float>& work,
    const LocalTensor<float>& reduction, float& sum, float& absoluteSum, uint32_t cores)
{
    constexpr uint32_t COUNT = 4096;
    if (rows == leading) {
        const uint64_t elements = static_cast<uint64_t>(rows) * columns;
        for (uint64_t offset = static_cast<uint64_t>(GetBlockIdx()) * COUNT; offset < elements;
             offset += static_cast<uint64_t>(cores) * COUNT) {
            AddStatisticsTile(
                matrix, offset, elements - offset < COUNT ? elements - offset : COUNT, input, absolute, work, reduction,
                sum, absoluteSum);
        }
    } else {
        const uint64_t chunks = (static_cast<uint64_t>(rows) + COUNT - 1) / COUNT;
        for (uint64_t task = GetBlockIdx(); task < chunks * columns; task += cores) {
            const uint64_t row = task % chunks * COUNT;
            const uint64_t column = task / chunks;
            AddStatisticsTile(
                matrix, column * leading + row, rows - row < COUNT ? rows - row : COUNT, input, absolute, work,
                reduction, sum, absoluteSum);
        }
    }
}

__aicore__ inline bool ReadPolicyDecision(
    const GlobalTensor<float>& statistics, const LocalTensor<float>& input, uint32_t batch,
    const GemmBatchedExArch22Tiling& t)
{
    const DataCopyExtParams copy{1, t.vectorBlocks * 32U, 0, 0, 0};
    const DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    SyncEvent<HardEvent::S_MTE2>();
    DataCopyPad(input, statistics[static_cast<uint64_t>(batch) * t.vectorBlocks * 8U], copy, padding);
    SyncEvent<HardEvent::MTE2_S>();
    float sumA = 0;
    float absoluteA = 0;
    float sumB = 0;
    float absoluteB = 0;
    for (uint32_t core = 0; core < t.vectorBlocks; ++core) {
        sumA += input.GetValue(core * 8U);
        absoluteA += input.GetValue(core * 8U + 1);
        sumB += input.GetValue(core * 8U + 2);
        absoluteB += input.GetValue(core * 8U + 3);
    }
    const bool nonfinite = NonFiniteStatistic(sumA) || NonFiniteStatistic(absoluteA) || NonFiniteStatistic(sumB) ||
                           NonFiniteStatistic(absoluteB);
    const float magnitudeA = sumA < 0.0f ? -sumA : sumA;
    const float magnitudeB = sumB < 0.0f ? -sumB : sumB;
    const float ratioA = absoluteA > 0.0f ? magnitudeA / absoluteA : 0.0f;
    const float ratioB = absoluteB > 0.0f ? magnitudeB / absoluteB : 0.0f;
    const bool reference = nonfinite || ratioA * ratioB > 0x1p-12f;
    return reference;
}

__aicore__ inline void ScanPolicy(
    GM_ADDR a, GM_ADDR b, const GemmBatchedExArch22Tiling& t, const LocalTensor<float>& input,
    const LocalTensor<float>& absolute, const LocalTensor<float>& work, const LocalTensor<float>& output,
    const GlobalTensor<float>& statistics)
{
    GlobalTensor<uint64_t> ap;
    GlobalTensor<uint64_t> bp;
    ap.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(a));
    bp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(b));
    for (uint32_t batch = 0; batch < t.batchCount; ++batch) {
        GlobalTensor<float> ag;
        GlobalTensor<float> bg;
        BindMatrix(ag, ap, batch);
        BindMatrix(bg, bp, batch);
        float sumA = 0;
        float absoluteA = 0;
        float sumB = 0;
        float absoluteB = 0;
        if (Alpha(t) != 0.0f) {
            ReadStatistics(
                ag, t.transA ? t.k : t.m, t.transA ? t.m : t.k, t.lda, input, absolute, work, output, sumA, absoluteA,
                t.vectorBlocks);
            ReadStatistics(
                bg, t.transB ? t.n : t.k, t.transB ? t.k : t.n, t.ldb, input, absolute, work, output, sumB, absoluteB,
                t.vectorBlocks);
        }
        for (uint32_t i = 0; i < 8; ++i)
            output.SetValue(i, 0.0f);
        output.SetValue(0, sumA);
        output.SetValue(1, absoluteA);
        output.SetValue(2, sumB);
        output.SetValue(3, absoluteB);
        SyncEvent<HardEvent::S_MTE3>();
        const DataCopyExtParams store{1, 32U, 0, 0, 0};
        DataCopyPad(statistics[(static_cast<uint64_t>(batch) * t.vectorBlocks + GetBlockIdx()) * 8U], output, store);
        SyncEvent<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void ComputeReferenceTask(
    const GlobalTensor<uint64_t>& ap, const GlobalTensor<uint64_t>& bp, const GlobalTensor<uint64_t>& cp,
    const GemmBatchedExArch22Tiling& t, const ReferenceScalars& scalars, uint64_t row, uint64_t col, uint32_t batch,
    uint32_t tile, const LocalTensor<float>& sum, const LocalTensor<float>& input, const LocalTensor<float>& old,
    const LocalTensor<float>& work)
{
    const DataCopyPadExtParams<float> padding{true, 0, static_cast<uint8_t>(t.transA ? 7 : 0), 0};
    const uint32_t rows = t.m - row < tile ? t.m - row : tile;
    const uint32_t count = t.transA ? rows * 8 : rows;
    const DataCopyExtParams packed{1, rows * 4U, 0, 0, 0};
    GlobalTensor<float> ag;
    GlobalTensor<float> bg;
    GlobalTensor<float> cg;
    BindMatrix(ag, ap, batch);
    BindMatrix(bg, bp, batch);
    BindMatrix(cg, cp, batch);
    const uint64_t offset = col * t.ldc + row;
    Duplicate(sum, 0.0f, count);
    PipeBarrier<PIPE_V>();
    if (!t.transA && scalars.beta != 0.0f) {
        DataCopyPad(old, cg[offset], packed, padding);
        SyncEvent<HardEvent::MTE2_V>();
        Muls(sum, old, scalars.beta, count);
        PipeBarrier<PIPE_V>();
    }
    if (scalars.alpha != 0.0f && t.k != 0)
        AccumulateReference(ag, bg, row, col, rows, count, t, scalars, sum, input, work);
    StoreReference(cg, offset, rows, count, t, scalars, sum, old, work);
}

__global__ __aicore__ void GemmBatchedExStatistics(GM_ADDR a, GM_ADDR b, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> input;
    TBuf<TPosition::VECCALC> absolute;
    TBuf<TPosition::VECCALC> work;
    TBuf<TPosition::VECOUT> output;
    pipe.InitBuffer(input, 4096U * sizeof(float));
    pipe.InitBuffer(absolute, 4096U * sizeof(float));
    pipe.InitBuffer(work, 64U * sizeof(float));
    pipe.InitBuffer(output, 64U);
    GlobalTensor<float> statistics;
    statistics.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t.policy + static_cast<uint64_t>(t.batchCount) * 64U));
    ScanPolicy(a, b, t, input.Get<float>(), absolute.Get<float>(), work.Get<float>(), output.Get<float>(), statistics);
    PipeBarrier<PIPE_ALL>();
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline bool PublishReferencePolicy(
    const GemmBatchedExArch22Tiling& t, uint32_t batch, const LocalTensor<float>& input,
    const LocalTensor<uint32_t>& flag)
{
    if (t.policy == 0)
        return true;
    GlobalTensor<float> statistics;
    statistics.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t.policy + static_cast<uint64_t>(t.batchCount) * 64U));
    const bool reference = ReadPolicyDecision(statistics, input, batch, t);
    if (GetBlockIdx() == 0) {
        GlobalTensor<uint32_t> flags;
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(t.policy));
        for (uint32_t i = 0; i < 16; ++i)
            flag.SetValue(i, 0);
        flag.SetValue(0, reference ? 1U : 0U);
        SyncEvent<HardEvent::S_MTE3>();
        const DataCopyExtParams store{1, 64U, 0, 0, 0};
        DataCopyPad(flags[static_cast<uint64_t>(batch) * 16U], flag, store);
        SyncEvent<HardEvent::MTE3_S>();
    }
    return reference;
}

__global__ __aicore__ void GemmBatchedExReference(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t CAPACITY = REFERENCE_CAPACITY;
    const uint32_t tile = t.transA ? CAPACITY / 8 : CAPACITY;
    const uint64_t chunks = (static_cast<uint64_t>(t.m) + tile - 1) / tile;
    const uint64_t blocks = GetBlockNum();
    if (blocks == 0 || chunks == 0) {
        return;
    }
    const uint64_t block = GetBlockIdx();
    TPipe pipe;
    TBuf<TPosition::VECOUT> sumBuffer;
    TBuf<TPosition::VECIN> inputBuffer;
    TBuf<TPosition::VECIN> oldBuffer;
    TBuf<TPosition::VECOUT> flagsBuffer;
    TBuf<TPosition::VECCALC> mathBuffer;
    pipe.InitBuffer(sumBuffer, CAPACITY * sizeof(float));
    pipe.InitBuffer(inputBuffer, CAPACITY * sizeof(float));
    pipe.InitBuffer(oldBuffer, CAPACITY * sizeof(float));
    pipe.InitBuffer(flagsBuffer, 64U);
    pipe.InitBuffer(mathBuffer, (5 * CAPACITY + 64) * sizeof(float));
    GlobalTensor<uint64_t> ap;
    GlobalTensor<uint64_t> bp;
    GlobalTensor<uint64_t> cp;
    ap.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(a));
    bp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(b));
    cp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(c));
    const ReferenceScalars scalars{Alpha(t), Beta(t)};
    const uint64_t tasksPerBatch = chunks * t.n;
    for (uint32_t batch = 0; batch < static_cast<uint32_t>(t.batchCount); ++batch) {
        if (!PublishReferencePolicy(t, batch, inputBuffer.Get<float>(), flagsBuffer.Get<uint32_t>()))
            continue;
        const uint64_t batchFirst = batch * tasksPerBatch;
        const uint64_t begin = max(t.referenceFirst, batchFirst);
        const uint64_t end = min(t.referenceLast, batchFirst + tasksPerBatch);
        // Rotate the first task to keep (task - referenceFirst) assigned to the same block across batches.
        const uint64_t rotation = (begin - t.referenceFirst) % blocks;
        const uint64_t first = begin + (block >= rotation ? block - rotation : block + blocks - rotation);
        for (uint64_t globalTask = first; globalTask < end; globalTask += blocks) {
            const uint64_t task = globalTask - batchFirst;
            ComputeReferenceTask(
                ap, bp, cp, t, scalars, task % chunks * tile, task / chunks, batch, tile, sumBuffer.Get<float>(),
                inputBuffer.Get<float>(), oldBuffer.Get<float>(), mathBuffer.Get<float>());
        }
    }
    PipeBarrier<PIPE_ALL>();
    SetMaskNorm();
    ResetMask();
}

template <bool NZ>
__aicore__ inline void BindCubeInputs(
    GlobalTensor<float>& x, GlobalTensor<float>& y, __gm__ float* aPtr, __gm__ float* bPtr,
    const GemmBatchedExArch22Tiling& t, uint64_t row, uint64_t col)
{
    if constexpr (NZ) {
        uint64_t aRows = t.transA ? (t.m + 15LL) / 16 * 16 : t.k;
        uint64_t bRows = t.transB ? t.k : (t.n + 15LL) / 16 * 16;
        x.SetGlobalBuffer(bPtr + (t.transB ? row * bRows : row * 8));
        y.SetGlobalBuffer(aPtr + (t.transA ? col * 8 : col * aRows));
    } else {
        x.SetGlobalBuffer(bPtr + (t.transB ? row : row * t.ldb));
        y.SetGlobalBuffer(aPtr + (t.transA ? col * t.lda : col));
    }
}

template <bool TA, bool TB, bool NZ>
__aicore__ inline void IterateCubeReduction(
    MM<TA, TB, NZ>& mm, GlobalTensor<float>& x, GlobalTensor<float>& y, GlobalTensor<float>& z, __gm__ float* aPtr,
    __gm__ float* bPtr, const GemmBatchedExArch22Tiling& t, uint64_t row, uint64_t col, int tailM, int tailN)
{
    for (int32_t first = 0; first < t.k; first += 256) {
        int32_t count = min(256, t.k - first);
        x.SetGlobalBuffer(bPtr + (TB ? row + static_cast<uint64_t>(first) * t.ldb : row * t.ldb + first));
        y.SetGlobalBuffer(aPtr + (TA ? col * t.lda + first : col + static_cast<uint64_t>(first) * t.lda));
        mm.SetTensorA(x, TB);
        mm.SetTensorB(y, TA);
        mm.SetSingleShape(tailM, tailN, count);
        mm.IterateAll(z, first == 0 ? 0 : 1);
        if (first + count < t.k)
            mm.End();
    }
}

template <bool TA, bool TB, bool NZ>
__aicore__ inline void CubeTask(
    MM<TA, TB, NZ>& mm, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const GemmBatchedExArch22Tiling& t,
    uint64_t task)
{
    uint64_t rows = (t.n + t.tileM - 1LL) / t.tileM;
    uint64_t cols = (t.m + t.tileN - 1LL) / t.tileN;
    uint64_t batch = task / (rows * cols);
    uint64_t row = (task / cols % rows) * t.tileM;
    uint64_t col = (task % cols) * t.tileN;
    auto aPtr = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
    auto bPtr = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
    auto cPtr = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(c)[batch]);
    GlobalTensor<float> x;
    GlobalTensor<float> y;
    GlobalTensor<float> z;
    if (t.packed) {
        aPtr = reinterpret_cast<__gm__ float*>(workspace) + batch * t.packAElements;
        bPtr = reinterpret_cast<__gm__ float*>(workspace) + static_cast<uint64_t>(t.batchCount) * t.packAElements +
               batch * t.packBElements;
    }
    BindCubeInputs<NZ>(x, y, aPtr, bPtr, t, row, col);
    uint64_t stride = t.temporary ? t.productLd : t.ldc;
    auto out =
        t.temporary ? reinterpret_cast<__gm__ float*>(workspace) + t.productOffset + batch * t.productLd * t.n : cPtr;
    z.SetGlobalBuffer(out + row * stride + col);
    mm.SetOrgShape(
        t.transB ? t.ldb : t.n, t.transA ? t.m : t.lda, t.transB ? t.k : t.ldb, t.transA ? t.lda : t.k, stride);
    mm.SetTensorA(x, t.transB);
    mm.SetTensorB(y, t.transA);
    int tailM = t.n - row < t.tileM ? t.n - row : t.tileM;
    int tailN = t.m - col < t.tileN ? t.m - col : t.tileN;
    mm.SetSingleShape(tailM, tailN, t.k);
    if constexpr (!TA && !TB && !NZ) {
        if (t.k >= 4096) {
            IterateCubeReduction<TA, TB, NZ>(mm, x, y, z, aPtr, bPtr, t, row, col, tailM, tailN);
            return;
        }
    }
    if constexpr (TA && TB && !NZ) {
        if (t.temporary && t.n <= 64 && t.k >= 1024) {
            IterateCubeReduction<TA, TB, NZ>(mm, x, y, z, aPtr, bPtr, t, row, col, tailM, tailN);
            return;
        }
    }
    mm.IterateAll(z);
}

template <bool TA, bool TB, bool NZ>
__global__ __aicore__ void GemmBatchedExCube(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (Alpha(t) == 0.0f)
        return;
    TPipe pipe;
    MM<TA, TB, NZ> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const AscendC::tiling::TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    uint32_t policyBatch = UINT32_MAX;
    bool policyValue = false;
    const uint64_t batchTasks = ((t.n + t.tileM - 1LL) / t.tileM) * ((t.m + t.tileN - 1LL) / t.tileN);
    for (uint64_t task = GetBlockIdx(); task < t.tasks; task += GetBlockNum()) {
        if (ReferenceRequired(t, task / batchTasks, policyBatch, policyValue))
            continue;
        CubeTask<TA, TB, NZ>(mm, a, b, c, workspace, t, task);
        if constexpr (!NZ)
            mm.End();
    }
    if constexpr (NZ)
        mm.End();
    mm.SetHF32(false);
}

// Preserve each operand's physical orientation. Extra reduction coordinates are zero.
class Pack {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, GemmBatchedExArch22Tiling t)
    {
        a_ = reinterpret_cast<__gm__ uint64_t*>(a);
        b_ = reinterpret_cast<__gm__ uint64_t*>(b);
        workspace_ = reinterpret_cast<__gm__ float*>(workspace);
        t_ = t;
        pipe_.InitBuffer(in_, 1, ELEMENTS * 4);
        pipe_.InitBuffer(out_, 1, ELEMENTS * 4);
        pipe_.InitBuffer(pad_, ELEMENTS * 4);
    }
    __aicore__ inline TPipe& Pipe() { return pipe_; }
    __aicore__ inline void Process(bool onlyA = false)
    {
        Operand(
            a_, workspace_, t_.transA ? (t_.m + 15LL) / 16 * 16 : t_.k, t_.lda, t_.transA ? t_.m : t_.originalK,
            t_.transA ? t_.originalK : t_.m, t_.originalLda, t_.packAElements);
        if (onlyA)
            return;
        uint32_t aRows = t_.transA ? (t_.m + 15U) / 16 * 16 : t_.k;
        uint32_t aTasks = static_cast<uint64_t>(t_.batchCount) * (aRows / ROWS) * ((t_.lda + COLS - 1U) / COLS);
        Operand(
            b_, workspace_ + static_cast<uint64_t>(t_.batchCount) * t_.packAElements,
            t_.transB ? t_.k : (t_.n + 15LL) / 16 * 16, t_.ldb, t_.transB ? t_.originalK : t_.n,
            t_.transB ? t_.n : t_.originalK, t_.originalLdb, t_.packBElements, aTasks);
    }

private:
    __aicore__ inline void LoadTile(
        GlobalTensor<float> src, uint32_t rows, uint32_t width, uint32_t leading, uint32_t stride)
    {
        auto padded = pad_.Get<float>();
        PipeBarrier<PIPE_V>();
        Duplicate(padded, 0.0f, ROWS * stride);
        if (rows && width) {
            auto in = in_.AllocTensor<float>();
            DataCopyExtParams read{static_cast<uint16_t>(rows), width * 4U, (leading - width) * 4U, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, 0, 0};
            DataCopyPad(in, src, read, pad);
            in_.EnQue(in);
            in = in_.DeQue<float>();
            uint32_t inWidth = (width + 7) / 8 * 8;
            CopyRepeatParams repeat{1, 1, static_cast<uint16_t>(stride / 8), static_cast<uint16_t>(inWidth / 8)};
            PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < width; col += 64) {
                Copy(
                    padded[col], in[col], static_cast<uint64_t>(min(64U, width - col)), static_cast<uint8_t>(rows),
                    repeat);
            }
            in_.FreeTensor(in);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StoreTile(GlobalTensor<float> dst, uint32_t width, uint32_t height, uint32_t stride)
    {
        auto out = out_.AllocTensor<float>();
        CopyRepeatParams repeat{ROWS, 1, 1, static_cast<uint16_t>(stride / 8)};
        auto padded = pad_.Get<float>();
        for (uint32_t col = 0; col < width; col += 64) {
            Copy(out[col * ROWS], padded[col], static_cast<uint64_t>(min(64U, width - col)), ROWS, repeat);
        }
        out_.EnQue(out);
        out = out_.DeQue<float>();
        DataCopyExtParams write{static_cast<uint16_t>(width / 8), ROWS * 8 * 4U, 0, (height - ROWS) * 8 * 4U, 0};
        DataCopyPad(dst, out, write);
        out_.FreeTensor(out);
    }
    __aicore__ inline void Operand(
        __gm__ uint64_t* pointers, __gm__ float* output, int rows, int width, int realRows, int realWidth, int leading,
        uint64_t plane, uint32_t firstTask = 0)
    {
        uint32_t columns = (width + COLS - 1U) / COLS;
        uint32_t groups = rows / ROWS;
        uint32_t tasks = static_cast<uint64_t>(t_.batchCount) * groups * columns;
        uint32_t shift = firstTask % t_.vectorBlocks;
        uint32_t first = static_cast<uint32_t>(GetBlockIdx()) >= shift ?
                             static_cast<uint32_t>(GetBlockIdx()) - shift :
                             static_cast<uint32_t>(GetBlockIdx()) + t_.vectorBlocks - shift;
        for (uint32_t task = first; task < tasks; task += t_.vectorBlocks) {
            uint32_t batch = task / (columns * groups);
            uint32_t row = task / columns % groups * ROWS;
            uint32_t col = task % columns * COLS;
            uint32_t count = min(static_cast<uint32_t>(COLS), width - col);
            uint32_t validRows = row >= realRows ? 0 : min(static_cast<uint32_t>(ROWS), realRows - row);
            uint32_t validWidth = col >= realWidth ? 0 : min(static_cast<uint32_t>(count), realWidth - col);
            GlobalTensor<float> src;
            GlobalTensor<float> dst;
            src.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(pointers[batch]) + static_cast<uint64_t>(row) * leading + col);
            dst.SetGlobalBuffer(output + batch * plane + static_cast<uint64_t>(col) * rows + row * 8);
            uint32_t stride = count <= 128 ? 128 : COLS;
            LoadTile(src, validRows, validWidth, leading, stride);
            StoreTile(dst, count, rows, stride);
        }
    }
    static constexpr uint32_t ROWS = 16, COLS = 256, ELEMENTS = ROWS * COLS;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> in_;
    TQue<QuePosition::VECOUT, 1> out_;
    TBuf<TPosition::VECCALC> pad_;
    __gm__ uint64_t *a_, *b_;
    __gm__ float* workspace_;
    GemmBatchedExArch22Tiling t_;
};

__global__ __aicore__ void GemmBatchedExPack(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Pack op;
    op.Init(a, b, workspace, t);
    op.Process();
    ResetMask();
}

class Epilogue {
public:
    __aicore__ inline void Init(GM_ADDR c, GM_ADDR workspace, GemmBatchedExArch22Tiling t, bool scaleOnly)
    {
        t_ = t;
        c_ = reinterpret_cast<__gm__ uint64_t*>(c);
        workspace_ = workspace;
        alpha_ = Alpha(t);
        beta_ = Beta(t);
        scale_ = scaleOnly || alpha_ == 0.0f;
        pipe_.InitBuffer(in_, 1, TILE * sizeof(float));
        pipe_.InitBuffer(old_, 1, TILE * sizeof(float));
        pipe_.InitBuffer(out_, 1, TILE * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        if (scale_ && beta_ == 1.0f)
            return;
        uint64_t tiles = (t_.m + TILE - 1ULL) / TILE;
        uint64_t count = static_cast<uint64_t>(t_.batchCount) * t_.n * tiles;
        uint32_t policyBatch = UINT32_MAX;
        bool policyValue = false;
        for (uint64_t task = GetBlockIdx(); task < count; task += GetBlockNum()) {
            uint64_t batch = task / (tiles * t_.n);
            if (!scale_ && ReferenceRequired(t_, batch, policyBatch, policyValue))
                continue;
            uint64_t col = task / tiles % t_.n;
            uint64_t row = task % tiles * TILE;
            uint32_t size = t_.m - row < TILE ? t_.m - row : TILE;
            GlobalTensor<float> c;
            GlobalTensor<float> product;
            c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c_[batch]) + col * t_.ldc + row);
            if (!scale_)
                product.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(workspace_) + t_.productOffset +
                    (batch * t_.n + col) * t_.productLd + row);
            Tile(c, product, size);
        }
    }

private:
    __aicore__ inline void Tile(GlobalTensor<float> c, GlobalTensor<float> product, uint32_t count)
    {
        DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        auto x = in_.AllocTensor<float>();
        auto old = old_.AllocTensor<float>();
        if (!scale_)
            DataCopyPad(x, product, copy, pad);
        if (beta_ != 0.0f)
            DataCopyPad(old, c, copy, pad);
        in_.EnQue(x);
        old_.EnQue(old);
        x = in_.DeQue<float>();
        old = old_.DeQue<float>();
        auto out = out_.AllocTensor<float>();
        if (scale_) {
            if (beta_ == 0.0f)
                Duplicate(out, 0.0f, count);
            else
                Muls(out, old, beta_, count);
        } else {
            Muls(out, x, alpha_, count);
            if (beta_ != 0.0f) {
                Muls(old, old, beta_, count);
                PipeBarrier<PIPE_V>();
                Add(out, out, old, count);
            }
        }
        out_.EnQue(out);
        out = out_.DeQue<float>();
        DataCopyPad(c, out, copy);
        in_.FreeTensor(x);
        old_.FreeTensor(old);
        out_.FreeTensor(out);
    }
    static constexpr uint32_t TILE = 2048;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> in_, old_;
    TQue<QuePosition::VECOUT, 1> out_;
    GemmBatchedExArch22Tiling t_;
    __gm__ uint64_t* c_;
    GM_ADDR workspace_;
    float alpha_, beta_;
    bool scale_;
};
// Short NN reductions load both operands once. Each AIV owns one output
// column and preserves the sequential BLAS special-value semantics.
class SmallNN {
public:
    __aicore__ inline void Init(TPipe& pipe, const GemmBatchedExArch22Tiling& t)
    {
        alpha = Alpha(t);
        beta = Beta(t);
        lanes = (t.m + 7) / 8 * 8;
        pipe.InitBuffer(aq, 1, WIDTH * WIDTH * sizeof(float));
        pipe.InitBuffer(bq, 1, WIDTH * sizeof(float));
        pipe.InitBuffer(old, 1, WIDTH * sizeof(float));
        pipe.InitBuffer(output, 1, WIDTH * sizeof(float));
        pipe.InitBuffer(broadcastBuffer, WIDTH * 8 * sizeof(float));
        pipe.InitBuffer(coefficientBuffer, WIDTH * WIDTH * sizeof(float));
        broadcast = broadcastBuffer.Get<float>();
        coefficients = coefficientBuffer.Get<float>();
    }

    __aicore__ inline void Column(GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmBatchedExArch22Tiling& t, uint64_t task)
    {
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        const uint32_t batch = task / t.n;
        const uint32_t j = task % t.n;
        GlobalTensor<float> ag;
        GlobalTensor<float> bg;
        GlobalTensor<float> cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]));
        bg.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]) +
            static_cast<uint64_t>(j) * t.ldb);
        cg.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(c)[batch]) +
            static_cast<uint64_t>(j) * t.ldc);
        auto av = aq.AllocTensor<float>();
        auto bv = bq.AllocTensor<float>();
        auto cv = old.AllocTensor<float>();
        auto sum = output.AllocTensor<float>();
        DataCopyExtParams row{1, static_cast<uint32_t>(t.m) * 4, 0, 0, 0};
        if (beta != 0.0f) {
            DataCopyPad(cv, cg, row, pad);
            old.EnQue(cv);
            cv = old.DeQue<float>();
            Muls(sum, cv, beta, t.m);
        } else
            Duplicate(sum, 0.0f, t.m);
        PipeBarrier<PIPE_V>();
        if (alpha != 0.0f)
            Accumulate(t, ag, bg, av, bv, sum);
        aq.FreeTensor(av);
        bq.FreeTensor(bv);
        old.FreeTensor(cv);
        output.EnQue(sum);
        sum = output.DeQue<float>();
        DataCopyPad(cg, sum, row);
        output.FreeTensor(sum);
    }

private:
    __aicore__ inline void PrepareCoefficients(const GemmBatchedExArch22Tiling& t, const LocalTensor<float>& bv)
    {
        Brcb(broadcast, bv, (t.k + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        const uint32_t elements = t.k * lanes;
        CopyRepeatParams repeat{1, 0, static_cast<uint16_t>(lanes / 8), 1};
        Copy(coefficients, broadcast, static_cast<uint64_t>(lanes), static_cast<uint8_t>(t.k), repeat);
        PipeBarrier<PIPE_V>();
        if (alpha != 1.0f) {
            Muls(coefficients, coefficients, alpha, elements);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Accumulate(
        const GemmBatchedExArch22Tiling& t, const GlobalTensor<float>& ag, const GlobalTensor<float>& bg,
        LocalTensor<float>& av, LocalTensor<float>& bv, const LocalTensor<float>& sum)
    {
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyExtParams acopy{
            static_cast<uint16_t>(t.k), static_cast<uint32_t>(t.m) * 4, static_cast<uint32_t>(t.lda - t.m) * 4, 0, 0};
        DataCopyExtParams rightCopy{1, static_cast<uint32_t>(t.k) * 4, 0, 0, 0};
        DataCopyPad(av, ag, acopy, pad);
        DataCopyPad(bv, bg, rightCopy, pad);
        aq.EnQue(av);
        bq.EnQue(bv);
        av = aq.DeQue<float>();
        bv = bq.DeQue<float>();
        PrepareCoefficients(t, bv);
        SetMaskNorm();
        SetVectorMask<float>(0, (1ULL << t.m) - 1);
        BinaryRepeatParams madd{1, 1, 1, 8, 8, 8};
#pragma unroll 4
        for (uint32_t kk = 0; kk < static_cast<uint32_t>(t.k); ++kk) {
            MulAddDst<float, float, false>(sum, av[kk * lanes], coefficients[kk * lanes], MASK_PLACEHOLDER, 1, madd);
            PipeBarrier<PIPE_V>();
        }
        ResetMask();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    static constexpr uint32_t WIDTH = 32;
    float alpha;
    float beta;
    uint32_t lanes;
    TQue<QuePosition::VECIN, 1> aq;
    TQue<QuePosition::VECIN, 1> bq;
    TQue<QuePosition::VECIN, 1> old;
    TQue<QuePosition::VECOUT, 1> output;
    TBuf<TPosition::VECCALC> broadcastBuffer;
    TBuf<TPosition::VECCALC> coefficientBuffer;
    LocalTensor<float> broadcast;
    LocalTensor<float> coefficients;
};

__global__ __aicore__ void GemmBatchedExSmallNN(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SmallNN op;
    op.Init(pipe, t);
    const uint64_t tasks = static_cast<uint64_t>(t.batchCount) * t.n;
    for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum())
        op.Column(a, b, c, t, task);
    ResetMask();
}

template <uint32_t SIZE>
__aicore__ inline void ComputeSmallSquare(
    const LocalTensor<float>& av, const LocalTensor<float>& bv, const LocalTensor<float>& sum,
    const LocalTensor<float>& products, const LocalTensor<float>& broadcast, const LocalTensor<float>& magnitudes,
    const LocalTensor<float>& reduced)
{
    Abs(magnitudes, av, SIZE * SIZE + SIZE);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduced, magnitudes, reduced[32], SIZE * SIZE + SIZE);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if (reduced.GetValue(0) < 0x1p50f) {
        Brcb(broadcast, bv, SIZE / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        Mul(products, av, broadcast, SIZE, SIZE, {1, 1, 0, SIZE / 8, SIZE / 8, 1});
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t rows = SIZE / 2; rows > 1; rows /= 2) {
            Add(products, products, products[rows * SIZE], rows * SIZE);
            PipeBarrier<PIPE_V>();
        }
        Add(sum, products, products[SIZE], SIZE);
    } else {
        for (uint32_t row = 0; row < SIZE; ++row) {
            float value = 0.0f;
            for (uint32_t kk = 0; kk < SIZE; ++kk)
                value = FusedFloat(av.GetValue(kk * SIZE + row), bv.GetValue(kk), value);
            sum.SetValue(row, value);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
}

// A short square NN problem fits entirely in UB. Ordinary finite inputs use
// a tree reduction; exceptional magnitudes keep BLAS's ordered FMA sequence.
template <uint32_t SIZE>
__global__ __aicore__ void GemmBatchedExSmallSquare(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> inputs;
    TBuf<TPosition::VECOUT> result;
    TBuf<TPosition::VECCALC> productBuffer;
    TBuf<TPosition::VECCALC> broadcastBuffer;
    TBuf<TPosition::VECCALC> magnitudeBuffer;
    TBuf<TPosition::VECCALC> reduceBuffer;
    pipe.InitBuffer(inputs, (SIZE * SIZE + SIZE) * sizeof(float));
    pipe.InitBuffer(result, SIZE * sizeof(float));
    pipe.InitBuffer(productBuffer, SIZE * SIZE * sizeof(float));
    pipe.InitBuffer(broadcastBuffer, SIZE * 8 * sizeof(float));
    pipe.InitBuffer(magnitudeBuffer, (SIZE * SIZE + SIZE) * sizeof(float));
    pipe.InitBuffer(reduceBuffer, 256);
    auto av = inputs.Get<float>();
    auto bv = av[SIZE * SIZE];
    auto sum = result.Get<float>();
    auto products = productBuffer.Get<float>();
    auto broadcast = broadcastBuffer.Get<float>();
    auto magnitudes = magnitudeBuffer.Get<float>();
    auto reduced = reduceBuffer.Get<float>();
    for (uint32_t task = GetBlockIdx(); task < static_cast<uint32_t>(t.batchCount) * SIZE; task += GetBlockNum()) {
        const uint32_t batch = task / SIZE;
        const uint32_t j = task % SIZE;
        GlobalTensor<float> ag;
        GlobalTensor<float> bg;
        GlobalTensor<float> cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]) + j * SIZE);
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(c)[batch]) + j * SIZE);
        DataCopy(av, ag, SIZE * SIZE);
        DataCopy(bv, bg, SIZE);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ComputeSmallSquare<SIZE>(av, bv, sum, products, broadcast, magnitudes, reduced);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(cg, sum, SIZE);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    ResetMask();
}

// Follow the sequential fp32 BLAS reduction for small outputs. Output DMA owns
// disjoint row ranges; scalar GM stores would share cache lines between cores.
class ExactReduction {
public:
    __aicore__ inline void Init(TPipe& pipe, const GemmBatchedExArch22Tiling& t)
    {
        alpha = Alpha(t);
        beta = Beta(t);
        pipe.InitBuffer(input, 1, ROWS * REDUCTION * sizeof(float));
        pipe.InitBuffer(old, 1, ROWS * sizeof(float));
        pipe.InitBuffer(output, 1, ROWS * sizeof(float));
        pipe.InitBuffer(gathered, ROWS * sizeof(float));
        pipe.InitBuffer(indices, ROWS * sizeof(uint32_t));
        av = gathered.Get<float>();
        offsets = indices.Get<uint32_t>();
        // Transposed A stores one contiguous reduction per row in UB.
        if (t.transA && t.k < 512) {
            for (uint32_t i = 0; i < ROWS; ++i)
                offsets.SetValue(i, i * REDUCTION * sizeof(float));
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void Task(GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmBatchedExArch22Tiling& t, uint64_t task)
    {
        const uint32_t tiles = (t.m + ROWS - 1) / ROWS;
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        const uint32_t batch = task / (static_cast<uint64_t>(t.n) * tiles);
        j = task / tiles % t.n;
        row = task % tiles * ROWS;
        count = min(ROWS, static_cast<uint32_t>(t.m) - row);
        lanes = (count + 7) / 8 * 8;
        ap = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
        bp = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
        GlobalTensor<float> dst;
        dst.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(c)[batch]) +
            static_cast<uint64_t>(j) * t.ldc + row);
        auto sum = output.AllocTensor<float>();
        auto previous = old.AllocTensor<float>();
        DataCopyExtParams rowCopy{1, count * 4, 0, 0, 0};
        if (beta != 0.0f) {
            DataCopyPad(previous, dst, rowCopy, pad);
            old.EnQue(previous);
            previous = old.DeQue<float>();
        }
        if (t.transA && t.k >= 512 && alpha != 0.0f)
            ComputeScalar(t, sum, previous);
        else
            ComputeVector(t, sum, previous);
        old.FreeTensor(previous);
        output.EnQue(sum);
        sum = output.DeQue<float>();
        DataCopyPad(dst, sum, rowCopy);
        output.FreeTensor(sum);
    }

private:
    __aicore__ inline void ComputeScalar(
        const GemmBatchedExArch22Tiling& t, const LocalTensor<float>& sum, const LocalTensor<float>& previous)
    {
        if (beta != 0.0f) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        for (uint32_t i = 0; i < count; ++i) {
            float value = 0.0f;
            for (uint32_t kk = 0; kk < static_cast<uint32_t>(t.k); ++kk) {
                float left = ap[static_cast<uint64_t>(row + i) * t.lda + kk];
                float right =
                    bp[t.transB ? static_cast<uint64_t>(kk) * t.ldb + j : static_cast<uint64_t>(j) * t.ldb + kk];
                value = FusedFloat(left, right, value);
            }
            volatile float product = alpha * value;
            sum.SetValue(
                i, beta == 0.0f ? static_cast<float>(product) :
                                  FusedFloat(beta, previous.GetValue(i), static_cast<float>(product)));
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void ComputeVector(
        const GemmBatchedExArch22Tiling& t, const LocalTensor<float>& sum, const LocalTensor<float>& previous)
    {
        if ((!t.transA || alpha == 0.0f) && beta != 0.0f)
            Muls(sum, previous, beta, count);
        else
            Duplicate(sum, 0.0f, count);
        PipeBarrier<PIPE_V>();
        if (alpha != 0.0f) {
            for (uint32_t k0 = 0; k0 < static_cast<uint32_t>(t.k); k0 += REDUCTION) {
                ReduceBlock(t, sum, k0);
            }
            if (t.transA) {
                Muls(sum, sum, alpha, count);
                PipeBarrier<PIPE_V>();
                if (beta != 0.0f) {
                    Axpy(sum, previous, beta, count);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    __aicore__ inline void ReduceBlock(const GemmBatchedExArch22Tiling& t, const LocalTensor<float>& sum, uint32_t k0)
    {
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        const uint32_t kc = min(REDUCTION, static_cast<uint32_t>(t.k) - k0);
        GlobalTensor<float> src;
        src.SetGlobalBuffer(
            ap + (t.transA ? static_cast<uint64_t>(row) * t.lda + k0 : static_cast<uint64_t>(k0) * t.lda + row));
        auto values = input.AllocTensor<float>();
        DataCopyExtParams read{
            static_cast<uint16_t>(t.transA ? count : kc), (t.transA ? kc : count) * 4,
            static_cast<uint32_t>((t.lda - (t.transA ? kc : count)) * 4),
            static_cast<uint32_t>(t.transA ? (REDUCTION - (kc + 7) / 8 * 8) / 8 : 0), 0};
        DataCopyPad(values, src, read, pad);
        input.EnQue(values);
        values = input.DeQue<float>();
        ApplyBlock(t, sum, values, k0, kc);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        input.FreeTensor(values);
    }

    __aicore__ inline void ApplyBlock(
        const GemmBatchedExArch22Tiling& t, const LocalTensor<float>& sum, const LocalTensor<float>& values,
        uint32_t k0, uint32_t kc)
    {
        for (uint32_t kk = 0; kk < kc; ++kk) {
            const uint64_t reduction = k0 + kk;
            const float bv = bp[t.transB ? reduction * t.ldb + j : static_cast<uint64_t>(j) * t.ldb + reduction];
            if (t.transA) {
                Gather(av, values, offsets, kk * sizeof(float), count);
                PipeBarrier<PIPE_V>();
                Axpy(sum, av, bv, count);
            } else {
                volatile float scaled = alpha * bv;
                Axpy(sum, values[kk * lanes], static_cast<float>(scaled), count);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    static constexpr uint32_t ROWS = 64;
    static constexpr uint32_t REDUCTION = 128;
    float alpha;
    float beta;
    uint32_t row;
    uint32_t j;
    uint32_t count;
    uint32_t lanes;
    __gm__ float* ap;
    __gm__ float* bp;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECIN, 1> old;
    TQue<QuePosition::VECOUT, 1> output;
    TBuf<TPosition::VECCALC> gathered;
    TBuf<TPosition::VECCALC> indices;
    LocalTensor<float> av;
    LocalTensor<uint32_t> offsets;
};

__global__ __aicore__ void GemmBatchedExExact(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ExactReduction op;
    op.Init(pipe, t);
    const uint32_t tiles = (t.m + 63U) / 64;
    const uint64_t tasks = static_cast<uint64_t>(t.batchCount) * t.n * tiles;
    for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum())
        op.Task(a, b, c, t, task);
    ResetMask();
}

__global__ __aicore__ void GemmBatchedExScale(GM_ADDR c, GM_ADDR workspace, GemmBatchedExArch22Tiling t, bool scaleOnly)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Epilogue op;
    op.Init(c, workspace, t, scaleOnly);
    op.Process();
    ResetMask();
}
// Compact per-core ping-pong output hides the AIV scatter behind the next Cube tile.
// Flag 7: all-AIV Pack barrier. Flags 8/9: Cube output ready. Flags 10/11: AIV slot free.
// Flags 12/13 mark the first four completed N tiles in the short TT stream.
// Matmul uses flags 0..3. Tail AIVs still pair every ready/free event.
using AlignedOutput = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulShapeParams FUSED_SHAPE{128, 512, 128, 128, 128, 32};
constexpr MatmulFuncParams FUSED_FUNC{false, false, false, false, 2, IterateOrder::ORDER_N, ScheduleType::INNER_PRODUCT,
                                      true,  true};
template <bool STRIP, bool TB>
__aicore__ constexpr MatmulConfig MakeMacroConfig()
{
    constexpr MatmulShapeParams strip{128, 4096, 128, 128, TB ? 128 : 256, TB ? 64 : 32};
    auto cfg = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(STRIP ? strip : FUSED_SHAPE, FUSED_FUNC, BIAS);
    cfg.enUnitFlag = !(STRIP && TB);
    return cfg;
}
template <bool STRIP, bool TB>
constexpr MatmulConfig FUSED_CONFIG = MakeMacroConfig<STRIP, TB>();
template <bool TB, bool STRIP>
using HybridLeft = matmul::MatmulType<STRIP ? TPosition::A1 : TPosition::GM, CubeFormat::NZ, float, TB>;
template <bool TA, bool TB, bool STRIP = false>
__aicore__ constexpr MatmulApiStaticTiling MakeMacroTiling()
{
    auto t = matmul::GetMatmulApiTiling<HybridLeft<TB, STRIP>, Input<TA, true>, AlignedOutput, Matrix>(
        FUSED_CONFIG<STRIP, TB>);
    if constexpr (STRIP && TA && TB)
        t.dbL0C = 2;
    t.stepM = 1;
    t.stepN = 1;
    t.stepKa = STRIP && TB ? 2 : 4;
    t.stepKb = STRIP && TB ? 2 : 4;
    t.depthA1 = STRIP && TB ? 4 : 8;
    t.depthB1 = STRIP && TB ? 4 : 8;
    t.shareL1Size = (128 + (STRIP && !TB ? 256 : 128)) * 128 * 4 * 2;
    return t;
}
template <bool TA, bool TB, bool STRIP = false>
constexpr MatmulApiStaticTiling FUSED_TILING = MakeMacroTiling<TA, TB, STRIP>();
template <bool TA, bool TB, bool STRIP = false>
using FusedMM =
    matmul::MatmulImpl<HybridLeft<TB, STRIP>, Input<TA, true>, AlignedOutput, Matrix, FUSED_TILING<TA, TB, STRIP>>;

struct RingCoordinates {
    uint32_t batch, row, col;
};

__aicore__ inline RingCoordinates DecodeRingTask(
    uint64_t task, uint32_t rows, uint32_t columns, uint32_t height, uint32_t width)
{
    if (rows == 0 || columns == 0)
        return {};
    return {
        static_cast<uint32_t>(task / (static_cast<uint64_t>(rows) * columns)),
        static_cast<uint32_t>(task / columns % rows * height), static_cast<uint32_t>(task % columns * width)};
}

using OutputQueue = TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1>;
__aicore__ inline void StoreRingTask(
    OutputQueue& input, GM_ADDR c, __gm__ float* scratch, const GemmBatchedExArch22Tiling& t, uint64_t task,
    uint64_t round, uint32_t rows, uint32_t columns, uint32_t height, uint32_t WIDTH, uint32_t ELEMENTS)
{
    const auto coord = DecodeRingTask(task, rows, columns, height, WIDTH);
    uint32_t batch = coord.batch;
    uint32_t rowBase = coord.row;
    uint32_t subRow = GetSubBlockIdx() * (height / 2);
    uint32_t row = rowBase + subRow;
    uint32_t col = coord.col;
    uint32_t width = min(static_cast<uint32_t>(WIDTH), t.m - col);
    uint32_t realHeight = min(height, t.n - rowBase);
    if (subRow < realHeight) {
        uint16_t outputRows = min(height / 2, realHeight - subRow);
        GlobalTensor<float> src;
        GlobalTensor<float> dst;
        src.SetGlobalBuffer(scratch + (round % 2) * ELEMENTS + subRow * WIDTH);
        src.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
        auto local = input.AllocTensor<float>();
        DataCopy(local, src, outputRows * WIDTH);
        input.EnQue(local);
        local = input.DeQue<float>();
        dst.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(c)[batch]) +
            static_cast<uint64_t>(row) * t.ldc + col);
        DataCopyExtParams write{outputRows, width * 4U, (WIDTH - (width + 7) / 8 * 8) / 8, (t.ldc - width) * 4U, 0};
        DataCopyPad(dst, local, write);
        input.FreeTensor(local);
    }
}

__aicore__ inline void StoreRingPiece(
    OutputQueue& input, GM_ADDR c, __gm__ float* scratch, const GemmBatchedExArch22Tiling& t, uint64_t task,
    uint64_t round, uint32_t rows, uint32_t height, uint32_t width, uint32_t elements, uint32_t first)
{
    const auto coord = DecodeRingTask(task, rows, 1, height, width);
    uint32_t sub = GetSubBlockIdx() * (height / 2);
    uint32_t valid = min(height, t.n - coord.row);
    if (sub < valid) {
        uint16_t count = min(height / 2, valid - sub);
        uint32_t columns = min(512U, t.m - first);
        uint32_t aligned = (columns + 7) / 8 * 8;
        GlobalTensor<float> src;
        GlobalTensor<float> dst;
        src.SetGlobalBuffer(scratch + (round % 2) * elements + sub * width + first);
        src.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
        dst.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(c)[coord.batch]) +
            static_cast<uint64_t>(coord.row + sub) * t.ldc + first);
        auto local = input.AllocTensor<float>();
        DataCopyParams read{count, static_cast<uint16_t>(aligned / 8), static_cast<uint16_t>((width - aligned) / 8), 0};
        DataCopy(local, src, read);
        input.EnQue(local);
        local = input.DeQue<float>();
        DataCopyExtParams write{count, columns * 4U, 0, (t.ldc - columns) * 4U, 0};
        DataCopyPad(dst, local, write);
        input.FreeTensor(local);
    }
}

template <bool TB>
__aicore__ inline void LoadRingStrip(
    const LocalTensor<float>& left, GM_ADDR b, const GemmBatchedExArch22Tiling& t, uint32_t batch, uint32_t row,
    uint32_t height)
{
    GlobalTensor<float> original;
    original.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]) +
        (TB ? row : static_cast<uint64_t>(row) * t.originalLdb));
    uint32_t validRows = min(height, t.n - row);
    Nd2NzParams copy{
        1,
        static_cast<uint16_t>(TB ? t.originalK : validRows),
        static_cast<uint16_t>(TB ? validRows : t.originalK),
        0,
        static_cast<uint16_t>(t.originalLdb),
        static_cast<uint16_t>(TB ? t.k : (height + 15) / 16 * 16),
        1,
        0};
    DataCopy(left, original, copy);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID6);
}

template <bool STRIP, bool STREAM, typename Matmul>
__aicore__ inline void FinishRingProduct(Matmul& mm, const GlobalTensor<float>& z, uint16_t slot)
{
    if constexpr (STREAM) {
        uint32_t piece = 0;
        while (mm.Iterate()) {
            mm.GetTensorC(z);
            if (++piece == 4)
                CrossCoreSetFlag<2, PIPE_FIX>(12 + slot);
        }
    } else {
        mm.IterateAll(z);
    }
    CrossCoreSetFlag<2, PIPE_FIX>(8 + slot);
    if constexpr (STRIP)
        mm.End();
}

template <bool TA, bool TB, bool STRIP, bool STREAM>
__aicore__ inline void ComputeRingTask(
    FusedMM<TA, TB, STRIP>& mm, TBuf<TPosition::A1>& leftBuffer, const GemmBatchedExArch22Tiling& t, GM_ADDR b,
    GM_ADDR workspace, __gm__ float* scratch, uint64_t task, uint64_t round, uint32_t rows, uint32_t columns,
    uint32_t height, uint32_t WIDTH, uint32_t ELEMENTS)
{
    uint16_t slot = round % 2;
    const auto coord = DecodeRingTask(task, rows, columns, height, WIDTH);
    uint32_t batch = coord.batch;
    uint32_t row = coord.row;
    uint32_t col = coord.col;
    auto ap = reinterpret_cast<__gm__ float*>(workspace) + static_cast<uint64_t>(batch) * t.packAElements;
    auto bp = reinterpret_cast<__gm__ float*>(workspace) + static_cast<uint64_t>(t.batchCount) * t.packAElements +
              static_cast<uint64_t>(batch) * t.packBElements;
    uint64_t aRows = TA ? (t.m + 15LL) / 16 * 16 : t.k;
    uint64_t bRows = TB ? t.k : (t.n + 15LL) / 16 * 16;
    GlobalTensor<float> x;
    GlobalTensor<float> y;
    GlobalTensor<float> z;
    x.SetGlobalBuffer(bp + (TB ? row * bRows : row * 8));
    y.SetGlobalBuffer(ap + (TA ? col * 8 : col * aRows));
    z.SetGlobalBuffer(scratch + slot * ELEMENTS);
    z.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
    if constexpr (STRIP)
        LoadRingStrip<TB>(leftBuffer.Get<float>(), b, t, batch, row, height);
    CrossCoreWaitFlag<2>(10 + slot);
    mm.SetOrgShape(
        STRIP ? (height + 15) / 16 * 16 : (TB ? t.ldb : t.n), TA ? t.m : t.lda, STRIP ? t.k : (TB ? t.k : t.ldb),
        TA ? t.lda : t.k, WIDTH);
    if constexpr (STRIP)
        mm.SetTensorA(leftBuffer.Get<float>(), TB);
    else
        mm.SetTensorA(x, TB);
    mm.SetTensorB(y, TA);
    // TT strips retain padded L1 shapes while storing only their logical output.
    constexpr bool LOGICAL_SHAPE = STRIP && TA && TB;
    mm.SetSingleShape(
        LOGICAL_SHAPE ? min(height, t.n - row) : (min(height, t.n - row) + 15) / 16 * 16,
        LOGICAL_SHAPE ? min(static_cast<uint32_t>(WIDTH), t.m - col) :
                        (min(static_cast<uint32_t>(WIDTH), t.m - col) + 15) / 16 * 16,
        LOGICAL_SHAPE ? (t.originalK + 7) / 8 * 8 : t.k);
    FinishRingProduct<STRIP, STREAM>(mm, z, slot);
}

template <bool TA, bool TB, bool STRIP, bool STREAM = false>
__aicore__ inline void ComputeRing(
    const GemmBatchedExArch22Tiling& t, GM_ADDR b, GM_ADDR workspace, __gm__ float* scratch, uint32_t block,
    uint64_t rounds, uint32_t rows, uint32_t columns, uint32_t height, uint32_t WIDTH, uint32_t ELEMENTS)
{
    TPipe pipe;
    FusedMM<TA, TB, STRIP> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const AscendC::tiling::TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    TBuf<TPosition::A1> leftBuffer;
    if constexpr (STRIP)
        pipe.InitBuffer(leftBuffer, ((height + 15) / 16 * 16) * t.k * sizeof(float));
    for (uint64_t round = 0, task = block; round < rounds; ++round, task += GetBlockNum()) {
        ComputeRingTask<TA, TB, STRIP, STREAM>(
            mm, leftBuffer, t, b, workspace, scratch, task, round, rows, columns, height, WIDTH, ELEMENTS);
    }
    mm.End();
    mm.SetHF32(false);
}

__aicore__ inline void PackTransposeNormalTask(
    TQue<QuePosition::VECIN, 1>& input, TQue<QuePosition::VECOUT, 1>& output, const LocalTensor<float>& padded,
    const LocalTensor<int32_t>& index, GM_ADDR a, GM_ADDR workspace, const GemmBatchedExArch22Tiling& t, uint32_t task,
    uint32_t groups, uint32_t reduction, uint32_t matrixRows)
{
    if (groups == 0) {
        return;
    }
    constexpr uint32_t ROWS = 48;
    uint32_t batch = task / groups;
    uint32_t row = task % groups * ROWS;
    uint16_t valid = min(ROWS, t.m - row);
    uint16_t stored = min(ROWS, matrixRows - row);
    GlobalTensor<float> src;
    GlobalTensor<float> dst;
    src.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]) +
        static_cast<uint64_t>(row) * t.originalK);
    dst.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace) + static_cast<uint64_t>(batch) * t.packAElements + row * 8U);
    auto data = input.AllocTensor<float>();
    DataCopyExtParams read{1, static_cast<uint32_t>(valid) * t.originalK * 4U, 0, 0, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
    DataCopyPad(data, src, read, noPad);
    Duplicate(padded, 0.0f, ROWS * reduction);
    input.EnQue(data);
    data = input.DeQue<float>();
    PipeBarrier<PIPE_V>();
    Gather(
        padded, data, index.ReinterpretCast<uint32_t>(), 0U, uint64_t(64), static_cast<uint8_t>(valid), reduction / 8);
    PipeBarrier<PIPE_V>();
    Gather(
        padded[64], data, index.ReinterpretCast<uint32_t>(), 64U * sizeof(float),
        static_cast<uint64_t>(t.originalK - 64), static_cast<uint8_t>(valid), reduction / 8);
    input.FreeTensor(data);
    PipeBarrier<PIPE_V>();
    auto packed = output.AllocTensor<float>();
    CopyRepeatParams repeat{ROWS, 1, 1, static_cast<uint16_t>(reduction / 8)};
    for (uint32_t col = 0; col < reduction; col += 64) {
        Copy(packed[col * ROWS], padded[col], static_cast<uint64_t>(min(64U, reduction - col)), ROWS, repeat);
    }
    output.EnQue(packed);
    packed = output.DeQue<float>();
    DataCopyExtParams write{
        static_cast<uint16_t>(reduction / 8), stored * 8U * 4U, ROWS - stored, (matrixRows - stored) * 8U * 4U, 0};
    DataCopyPad(dst, packed, write);
    output.FreeTensor(packed);
}

// Gather contiguous short-K TN rows after one bulk read, then store the Cube NZ layout.
__aicore__ inline void PackContiguousTransposeNormal(
    TPipe& pipe, GM_ADDR a, GM_ADDR workspace, const GemmBatchedExArch22Tiling& t)
{
    constexpr uint32_t ROWS = 48;
    uint32_t reduction = (t.originalK + 7U) / 8 * 8;
    uint32_t matrixRows = (t.m + 15U) / 16 * 16;
    uint32_t groups = (matrixRows + ROWS - 1U) / ROWS;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECOUT, 1> output;
    TBuf<TPosition::VECCALC> padBuffer;
    TBuf<TPosition::VECCALC> indexBuffer;
    TBuf<TPosition::VECCALC> rowBuffer;
    pipe.InitBuffer(input, 1, ROWS * 128 * sizeof(float));
    pipe.InitBuffer(output, 1, ROWS * 128 * sizeof(float));
    pipe.InitBuffer(padBuffer, ROWS * 128 * sizeof(float));
    pipe.InitBuffer(indexBuffer, ROWS * 64 * sizeof(int32_t));
    pipe.InitBuffer(rowBuffer, ROWS * 64 * sizeof(int32_t));
    auto padded = padBuffer.Get<float>();
    auto index = indexBuffer.Get<int32_t>();
    auto rowIndex = rowBuffer.Get<int32_t>();
    CreateVecIndex(index, int32_t(0), ROWS * 64);
    PipeBarrier<PIPE_V>();
    ShiftRight(rowIndex, index, 6, ROWS * 64);
    PipeBarrier<PIPE_V>();
    Muls(rowIndex, rowIndex, static_cast<int32_t>(t.originalK - 64), ROWS * 64);
    PipeBarrier<PIPE_V>();
    Add(index, index, rowIndex, ROWS * 64);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, 2, ROWS * 64);
    PipeBarrier<PIPE_V>();
    for (uint32_t task = GetBlockIdx(); task < static_cast<uint64_t>(groups) * t.batchCount; task += t.vectorBlocks) {
        PackTransposeNormalTask(input, output, padded, index, a, workspace, t, task, groups, reduction, matrixRows);
    }
}

template <bool HYBRID = false, bool STREAM = false>
__aicore__ inline void StoreRing(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const GemmBatchedExArch22Tiling& t, __gm__ float* scratch,
    uint32_t block, uint64_t rounds, uint32_t rows, uint32_t columns, uint32_t height, uint32_t WIDTH,
    uint32_t ELEMENTS)
{
    Pack pack;
    pack.Init(a, b, workspace, t);
    if constexpr (HYBRID) {
        if (t.transA && !t.transB && t.originalLda == t.originalK) {
            pack.Pipe().Reset();
            PackContiguousTransposeNormal(pack.Pipe(), a, workspace, t);
        } else {
            pack.Process(true);
        }
    } else {
        pack.Process();
    }
    CrossCoreSetFlag<0, PIPE_MTE3>(7);
    CrossCoreWaitFlag<0>(7);
    // Release the packing buffers before allocating the output conversion buffers.
    pack.Pipe().Reset();
    auto& pipe = pack.Pipe();
    OutputQueue input;
    pipe.InitBuffer(input, STREAM && height <= 96 ? 2 : 1, height / 2 * (STREAM ? 512 : WIDTH) * sizeof(float));
    if (rounds > 0)
        CrossCoreSetFlag<2, PIPE_MTE3>(10);
    if (rounds > 1)
        CrossCoreSetFlag<2, PIPE_MTE3>(11);
    for (uint64_t round = 0, task = block; round < rounds; ++round, task += GetBlockNum()) {
        uint16_t slot = round % 2;
        if constexpr (STREAM) {
            CrossCoreWaitFlag<2>(12 + slot);
            StoreRingPiece(input, c, scratch, t, task, round, rows, height, WIDTH, ELEMENTS, 0);
            CrossCoreWaitFlag<2>(8 + slot);
            StoreRingPiece(input, c, scratch, t, task, round, rows, height, WIDTH, ELEMENTS, 512);
        } else {
            CrossCoreWaitFlag<2>(8 + slot);
            StoreRingTask(input, c, scratch, t, task, round, rows, columns, height, WIDTH, ELEMENTS);
        }
        if (round + 2 < rounds)
            CrossCoreSetFlag<2, PIPE_MTE3>(10 + slot);
    }
    ResetMask();
}

template <bool TA, bool TB, bool STRIP = false, bool STREAM = false>
__global__ __aicore__ __schedmode__(1) void GemmBatchedExMacroRing(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSyncBaseAddr(t.syncAddress);
    uint32_t WIDTH = STRIP ? t.productLd : 512;
    uint32_t height = t.tileM;
    uint32_t ELEMENTS = STRIP ? ((height + 15) / 16 * 16) * WIDTH : 128 * WIDTH;
    uint32_t block = GetBlockIdx();
    if ASCEND_IS_AIV {
        block /= 2;
    }
    uint32_t columns = (t.m + WIDTH - 1U) / WIDTH;
    uint32_t rows = (t.n + height - 1U) / height;
    uint64_t rounds = (t.tasks + GetBlockNum() - 1 - block) / GetBlockNum();
    auto scratch =
        reinterpret_cast<__gm__ float*>(workspace) + t.productOffset + static_cast<uint64_t>(block) * 2 * ELEMENTS;
    if ASCEND_IS_AIC {
        ComputeRing<TA, TB, STRIP, STREAM>(
            t, b, workspace, scratch, block, rounds, rows, columns, height, WIDTH, ELEMENTS);
    }
    if ASCEND_IS_AIV {
        StoreRing<STRIP, STREAM>(a, b, c, workspace, t, scratch, block, rounds, rows, columns, height, WIDTH, ELEMENTS);
    }
    FinishCrossCore();
}

// Read original A directly into the two Cube L1 slots.
__aicore__ inline void LoadInputRight(
    const LocalTensor<float>& right, GM_ADDR workspace, const GemmBatchedExArch22Tiling& t, uint32_t col,
    uint32_t height, uint32_t reduction)
{
    auto event = static_cast<event_t>(4 + (col / 128) % 2);
    WaitFlag<HardEvent::MTE1_MTE2>(event);
    GlobalTensor<float> source;
    uint32_t stride = t.originalK > 48 ? 64 : (t.originalK + 7U) / 8 * 8;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + col * stride);
    uint16_t valid = min(128U, static_cast<uint32_t>(t.m) - col);
    Nd2NzParams read{
        1, valid, static_cast<uint16_t>(stride), 0, static_cast<uint16_t>(stride), static_cast<uint16_t>(height), 1, 0};
    DataCopy(right, source, read);
    SetFlag<HardEvent::MTE2_MTE1>(event);
}

class InputStream {
public:
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR b, const GemmBatchedExArch22Tiling& t, uint32_t row)
    {
        height = (t.tileM + 15U) / 16 * 16;
        reduction = (t.originalK + 63U) / 64 * 64;
        pipe.InitBuffer(left, height * t.k * sizeof(float));
        pipe.InitBuffer(right, 2 * 128 * reduction * sizeof(float));
        pipe.InitBuffer(l0a, height * reduction * sizeof(float));
        pipe.InitBuffer(l0b, 2 * 128 * reduction * sizeof(float));
        pipe.InitBuffer(l0c, 2 * height * 128 * sizeof(float));
        GlobalTensor<float> original;
        original.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[0]) + row);
        valid = min(static_cast<uint32_t>(t.tileM), static_cast<uint32_t>(t.n) - row);
        Nd2NzParams load{
            1,
            static_cast<uint16_t>(t.originalK),
            static_cast<uint16_t>(valid),
            0,
            static_cast<uint16_t>(t.originalLdb),
            static_cast<uint16_t>(t.k),
            1,
            0};
        DataCopy(left.Get<float>(), original, load);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID6);
        LoadData3DParamsV2<float> transpose;
        transpose.l1H = 1;
        transpose.l1W = t.k;
        transpose.channelSize = height;
        transpose.mExtension = reduction;
        transpose.kExtension = height;
        transpose.enTranspose = true;
        LoadData(l0a.Get<float>(), left.Get<float>(), transpose);
    }

    __aicore__ inline void Process(GM_ADDR workspace, __gm__ float* scratch, const GemmBatchedExArch22Tiling& t)
    {
        SetHF32Mode(false);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto event = static_cast<event_t>(4 + slot);
            SetFlag<HardEvent::MTE1_MTE2>(event);
            SetFlag<HardEvent::M_MTE1>(event);
            SetFlag<HardEvent::FIX_M>(event);
        }
        CrossCoreWaitFlag<2>(10);
        LoadInputRight(right.Get<float>(), workspace, t, 0, min(128U, (t.m + 15U) / 16 * 16), reduction);
        for (uint32_t col = 0; col < t.m; col += 128) {
            ComputeColumn(workspace, scratch, t, col);
        }
        CrossCoreSetFlag<2, PIPE_FIX>(8);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto event = static_cast<event_t>(4 + slot);
            WaitFlag<HardEvent::MTE1_MTE2>(event);
            WaitFlag<HardEvent::M_MTE1>(event);
            WaitFlag<HardEvent::FIX_M>(event);
        }
        SetHF32Mode(false);
    }

private:
    __aicore__ inline void ComputeColumn(
        GM_ADDR workspace, __gm__ float* scratch, const GemmBatchedExArch22Tiling& t, uint32_t col)
    {
        uint32_t slot = col / 128 % 2;
        auto event = static_cast<event_t>(4 + slot);
        uint16_t columns = (min(128U, static_cast<uint32_t>(t.m) - col) + 15U) / 16 * 16;
        auto l1 = right.Get<float>()[slot * 128 * reduction];
        auto matrixB = l0b.Get<float>()[slot * 128 * reduction];
        auto matrixC = l0c.Get<float>()[slot * height * 128];
        WaitFlag<HardEvent::MTE2_MTE1>(event);
        WaitFlag<HardEvent::M_MTE1>(event);
        LoadData2DParams copy{0, static_cast<uint8_t>(columns * reduction / 128), 1, 0, 0, false, 0};
        LoadData(matrixB, l1, copy);
        SetFlag<HardEvent::MTE1_MTE2>(event);
        SetFlag<HardEvent::MTE1_M>(event);
        if (col + 128 < t.m) {
            uint32_t next = col + 128;
            uint32_t nextColumns = (min(128U, static_cast<uint32_t>(t.m) - next) + 15U) / 16 * 16;
            LoadInputRight(
                right.Get<float>()[((col / 128 + 1) % 2) * 128 * reduction], workspace, t, next, nextColumns,
                reduction);
        }
        WaitFlag<HardEvent::MTE1_M>(event);
        WaitFlag<HardEvent::FIX_M>(event);
        MmadParams mad;
        mad.m = height;
        mad.n = columns;
        mad.k = t.originalK;
        mad.kDirectionAlign = true;
        Mmad(matrixC, l0a.Get<float>(), matrixB, mad);
        SetFlag<HardEvent::M_MTE1>(event);
        SetFlag<HardEvent::M_FIX>(event);
        WaitFlag<HardEvent::M_FIX>(event);
        StoreColumn(scratch, matrixC, t, col, columns, event);
        if (col == 384)
            CrossCoreSetFlag<2, PIPE_FIX>(12);
    }

    __aicore__ inline void StoreColumn(
        __gm__ float* scratch, const LocalTensor<float>& matrixC, const GemmBatchedExArch22Tiling& t, uint32_t col,
        uint16_t columns, event_t event)
    {
        GlobalTensor<float> out;
        out.SetGlobalBuffer(scratch + col);
        out.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
        FixpipeParamsV220 store{
            columns, static_cast<uint16_t>(valid), static_cast<uint16_t>(height), static_cast<uint32_t>(t.productLd),
            false};
        Fixpipe(out, matrixC, store);
        SetFlag<HardEvent::FIX_M>(event);
    }

    uint32_t height;
    uint32_t reduction;
    uint32_t valid;
    TBuf<TPosition::A1> left;
    TBuf<TPosition::A1> right;
    TBuf<TPosition::A2> l0a;
    TBuf<TPosition::B2> l0b;
    TBuf<TPosition::CO1> l0c;
};

__aicore__ inline void ComputeInputStream(
    GM_ADDR b, GM_ADDR workspace, __gm__ float* scratch, const GemmBatchedExArch22Tiling& t, uint32_t block)
{
    uint32_t row = block * t.tileM;
    if (row < t.n) {
        TPipe pipe;
        InputStream op;
        op.Init(pipe, b, t, row);
        op.Process(workspace, scratch, t);
    } else {
        CrossCoreWaitFlag<2>(10);
        CrossCoreSetFlag<2, PIPE_FIX>(12);
        CrossCoreSetFlag<2, PIPE_FIX>(8);
    }
}

// Discard padded columns while concatenating rows inside UB.
__aicore__ inline void StoreCompactMask(
    TPipe& pipe, GM_ADDR c, __gm__ float* scratch, const GemmBatchedExArch22Tiling& t, uint32_t block)
{
    constexpr uint32_t ROWS = 12;
    TQue<QuePosition::VECIN, 2> input;
    TQue<QuePosition::VECOUT, 2> output;
    pipe.InitBuffer(input, 2, ROWS * t.productLd * sizeof(float));
    pipe.InitBuffer(output, 2, ROWS * ((t.m + 7U) / 8 * 8) * sizeof(float));
    CrossCoreWaitFlag<2>(12);
    CrossCoreWaitFlag<2>(8);
    uint32_t sub = GetSubBlockIdx() * (t.tileM / 2);
    uint32_t row = block * t.tileM + sub;
    if (row >= t.n)
        return;
    uint32_t rows = min(static_cast<uint32_t>(t.tileM / 2), static_cast<uint32_t>(t.n) - row);
    GlobalTensor<float> src;
    GlobalTensor<float> dst;
    src.SetGlobalBuffer(scratch + sub * t.productLd);
    src.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
    dst.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(c)[0]) + static_cast<uint64_t>(row) * t.ldc);
    auto initial = input.AllocTensor<float>();
    DataCopy(initial, src, min(ROWS, rows) * t.productLd);
    input.EnQue(initial);
    for (uint32_t first = 0; first < rows; first += ROWS) {
        uint32_t count = min(ROWS, rows - first);
        if (first + ROWS < rows) {
            auto next = input.AllocTensor<float>();
            DataCopy(next, src[(first + ROWS) * t.productLd], min(ROWS, rows - first - ROWS) * t.productLd);
            input.EnQue(next);
        }
        auto data = input.DeQue<float>();
        auto compact = output.AllocTensor<float>();
        uint64_t selected = 0;
        GatherMask(
            compact, data, static_cast<uint8_t>(7), true, static_cast<uint32_t>(t.m),
            {1, static_cast<uint16_t>(count), static_cast<uint16_t>(t.productLd / 8), 0}, selected);
        input.FreeTensor(data);
        output.EnQue(compact);
        compact = output.DeQue<float>();
        DataCopyExtParams write{1, count * t.m * 4U, 0, 0, 0};
        DataCopyPad(dst[first * t.ldc], compact, write);
        output.FreeTensor(compact);
    }
}

// Read compact source rows in one DMA and insert K padding in UB.
__aicore__ inline void PackContiguousInput(
    TPipe& pipe, const GlobalTensor<float>& src, const GlobalTensor<float>& dst, uint32_t count, uint32_t reduction)
{
    uint32_t elements = count * 64;
    TBuf<TPosition::VECCALC> sourceBuffer;
    TBuf<TPosition::VECCALC> outputBuffer;
    TBuf<TPosition::VECCALC> indexBuffer;
    pipe.InitBuffer(sourceBuffer, elements * 4);
    pipe.InitBuffer(outputBuffer, elements * 4);
    pipe.InitBuffer(indexBuffer, elements * 4);
    auto source = sourceBuffer.Get<float>();
    auto output = outputBuffer.Get<float>();
    auto index = indexBuffer.Get<int32_t>();
    DataCopyExtParams read{1, count * reduction * 4U, 0, 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, 0, 0};
    DataCopyPad(source, src, read, pad);
    CreateVecIndex(index, int32_t(0), 64);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, 2, 64);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 1; row < count; ++row) {
        Adds(index[row * 64], index, static_cast<int32_t>(row * reduction * 4), 64);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    PipeBarrier<PIPE_V>();
    Gather(
        output, source, index.ReinterpretCast<uint32_t>(), 0U, static_cast<uint64_t>(reduction),
        static_cast<uint8_t>(count), 8);
    if (reduction < 64) {
        PipeBarrier<PIPE_V>();
        uint64_t mask[2] = {UINT64_MAX << reduction, 0};
        Duplicate(output, 0.0f, mask, static_cast<uint8_t>(count), 1, 8);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    DataCopy(dst, output, elements);
}

__aicore__ inline void PackInputStream(TPipe& pipe, GM_ADDR a, GM_ADDR workspace, const GemmBatchedExArch22Tiling& t)
{
    uint32_t stride = t.originalK > 48 ? 64 : (t.originalK + 7U) / 8 * 8;
    uint32_t rowsPerVector = (t.m + t.vectorBlocks - 1U) / t.vectorBlocks;
    uint32_t first = GetBlockIdx() * rowsPerVector;
    TBuf<TPosition::VECCALC> packed;
    pipe.InitBuffer(packed, rowsPerVector * stride * sizeof(float));
    if (first < t.m) {
        uint32_t count = min(rowsPerVector, static_cast<uint32_t>(t.m) - first);
        GlobalTensor<float> src;
        GlobalTensor<float> dst;
        src.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(a)[0]) +
            static_cast<uint64_t>(first) * t.originalLda);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + first * stride);
        if (t.originalLda == t.originalK && stride == 64) {
            PackContiguousInput(pipe, src, dst, count, t.originalK);
        } else {
            auto data = packed.Get<float>();
            DataCopyExtParams read{
                static_cast<uint16_t>(count), static_cast<uint32_t>(t.originalK) * 4U,
                static_cast<uint32_t>(t.originalLda - t.originalK) * 4U, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(min(8U, stride - t.originalK)), 0};
            DataCopyPad(data, src, read, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(dst, data, count * stride);
        }
    }
}

__aicore__ inline void StoreInputStream(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, __gm__ float* scratch, const GemmBatchedExArch22Tiling& t,
    uint32_t block)
{
    TPipe pipe;
    PackInputStream(pipe, a, workspace, t);
    CrossCoreSetFlag<0, PIPE_MTE3>(7);
    CrossCoreWaitFlag<0>(7);
    CrossCoreSetFlag<2, PIPE_MTE3>(10);
    // Packing uses raw TBuf storage; complete DMA before reusing the UB pool.
    PipeBarrier<PIPE_ALL>();
    pipe.Reset();
    if (t.ldc == t.m) {
        StoreCompactMask(pipe, c, scratch, t, block);
        ResetMask();
        return;
    }
    OutputQueue input;
    pipe.InitBuffer(input, 2, t.tileM / 2 * 512 * sizeof(float));
    uint32_t rows = (t.n + t.tileM - 1U) / t.tileM;
    uint32_t elements = ((t.tileM + 15U) / 16 * 16) * t.productLd;
    CrossCoreWaitFlag<2>(12);
    StoreRingPiece(input, c, scratch, t, block, 0, rows, t.tileM, t.productLd, elements, 0);
    CrossCoreWaitFlag<2>(8);
    StoreRingPiece(input, c, scratch, t, block, 0, rows, t.tileM, t.productLd, elements, 512);
    ResetMask();
}

__global__ __aicore__ __schedmode__(1) void GemmBatchedExInputStream(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GemmBatchedExArch22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSyncBaseAddr(t.syncAddress);
    uint32_t block = GetBlockIdx();
    if ASCEND_IS_AIV {
        block /= 2;
    }
    auto scratch = reinterpret_cast<__gm__ float*>(workspace) + t.productOffset +
                   static_cast<uint64_t>(block) * ((t.tileM + 15U) / 16 * 16) * t.productLd;
    if ASCEND_IS_AIC {
        ComputeInputStream(b, workspace, scratch, t, block);
    }
    if ASCEND_IS_AIV {
        StoreInputStream(a, b, c, workspace, scratch, t, block);
    }
    FinishCrossCore();
}

} // namespace

// Return the status of this launch rather than an unrelated thread-local error.
template <typename Arguments>
int32_t LaunchKernel(const void* symbol, uint32_t blocks, void* stream, Arguments& args)
{
    static_assert(offsetof(Arguments, tiling) == sizeof(args.tensors), "kernel argument ABI mismatch");
    aclrtFuncHandle function = nullptr;
    const aclError status = aclrtGetFuncBySymbol(symbol, &function);
    if (status != ACL_SUCCESS)
        return status;
    return aclrtLaunchKernelWithHostArgs(function, blocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}

template <uint32_t COUNT>
struct GemmLaunchArguments {
    GM_ADDR tensors[COUNT];
    GemmBatchedExArch22Tiling tiling;
};

struct ScaleLaunchArguments {
    GM_ADDR tensors[2];
    GemmBatchedExArch22Tiling tiling;
    bool scaleOnly;
};

template <bool NZ>
int32_t LaunchCube(
    uint32_t blocks, void* stream, GM_ADDR aa, GM_ADDR bb, GM_ADDR cc, GM_ADDR ww, GemmBatchedExArch22Tiling t)
{
    GemmLaunchArguments<4> args{{aa, bb, cc, ww}, t};
    if (t.transA && t.transB)
        return LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExCube<true, true, NZ>), blocks, stream, args);
    else if (t.transA)
        return LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExCube<true, false, NZ>), blocks, stream, args);
    else if (t.transB)
        return LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExCube<false, true, NZ>), blocks, stream, args);
    else
        return LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExCube<false, false, NZ>), blocks, stream, args);
}

template <bool STRIP>
int32_t LaunchRing(
    uint32_t blocks, void* stream, GM_ADDR aa, GM_ADDR bb, GM_ADDR cc, GM_ADDR ww, GemmBatchedExArch22Tiling t)
{
    GemmLaunchArguments<4> args{{aa, bb, cc, ww}, t};
    if constexpr (STRIP) {
        if (t.transA && t.transB && t.batchCount == 1 && t.m > 512 && t.m <= 1024 && t.n >= 512 && t.originalK <= 64) {
            return LaunchKernel(
                reinterpret_cast<const void*>(GemmBatchedExMacroRing<true, true, true, true>), blocks, stream, args);
        }
    }
    if (t.transA && t.transB)
        return LaunchKernel(
            reinterpret_cast<const void*>(GemmBatchedExMacroRing<true, true, STRIP>), blocks, stream, args);
    else if (t.transA)
        return LaunchKernel(
            reinterpret_cast<const void*>(GemmBatchedExMacroRing<true, false, STRIP>), blocks, stream, args);
    else if (t.transB)
        return LaunchKernel(
            reinterpret_cast<const void*>(GemmBatchedExMacroRing<false, true, STRIP>), blocks, stream, args);
    else
        return LaunchKernel(
            reinterpret_cast<const void*>(GemmBatchedExMacroRing<false, false, STRIP>), blocks, stream, args);
}

bool IsSmallSquare(const GemmBatchedExArch22Tiling& t)
{
    return !t.transA && !t.transB && !t.alphaDevice && !t.betaDevice && t.alpha == 1.0f && t.beta == 0.0f &&
           t.m == t.n && t.m == t.k && t.lda == t.m && t.ldb == t.k && t.ldc == t.m && (t.m == 16 || t.m == 32);
}

int32_t LaunchExact(void* stream, GM_ADDR aa, GM_ADDR bb, GM_ADDR cc, GemmBatchedExArch22Tiling t)
{
    GemmLaunchArguments<3> args{{aa, bb, cc}, t};
    if (IsSmallSquare(t)) {
        if (t.m == 16)
            return LaunchKernel(
                reinterpret_cast<const void*>(GemmBatchedExSmallSquare<16>), t.vectorBlocks, stream, args);
        else
            return LaunchKernel(
                reinterpret_cast<const void*>(GemmBatchedExSmallSquare<32>), t.vectorBlocks, stream, args);
    } else if (!t.transA && !t.transB && t.m <= 32 && t.n <= 32 && t.k <= 32)
        return LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExSmallNN), t.vectorBlocks, stream, args);
    else
        return LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExExact), t.vectorBlocks, stream, args);
}

void BalanceRowStrips(uint32_t blocks, GemmBatchedExArch22Tiling& t)
{
    // Balance the row strips without increasing the existing aligned scratch allocation.
    if (blocks > 0 && t.route == GemmBatchedExRoute::RowStrips && t.transA && t.transB && t.batchCount == 1 &&
        t.m >= 512 && t.n >= 512) {
        uint32_t height = ((t.n + blocks - 1U) / blocks + 1U) / 2 * 2;
        if (height > 128)
            height = 128;
        if (height < t.tileM) {
            t.tileM = height;
            t.tasks = (t.n + height - 1U) / height;
        }
    }
}

bool UseInputStream(uint32_t blocks, const GemmBatchedExArch22Tiling& t)
{
    return t.route == GemmBatchedExRoute::RowStrips && t.transA && t.transB && t.batchCount == 1 && t.m > 512 &&
           t.m <= 1024 && t.n >= 512 && t.originalK <= 64 && t.tasks <= blocks && t.tileM <= 96;
}

int32_t LaunchRegular(
    uint32_t blocks, void* stream, GM_ADDR aa, GM_ADDR bb, GM_ADDR cc, GM_ADDR ww, GemmBatchedExArch22Tiling t,
    bool scaleOnly)
{
    if (t.packed) {
        GemmLaunchArguments<3> args{{aa, bb, ww}, t};
        const aclError status =
            LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExPack), t.vectorBlocks, stream, args);
        if (status != ACL_SUCCESS)
            return status;
    }
    if (!scaleOnly) {
        const aclError status = t.packed ? LaunchCube<true>(blocks, stream, aa, bb, cc, ww, t) :
                                           LaunchCube<false>(blocks, stream, aa, bb, cc, ww, t);
        if (status != ACL_SUCCESS)
            return status;
    }
    if (scaleOnly || t.temporary) {
        ScaleLaunchArguments args{{cc, ww}, t, scaleOnly};
        return LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExScale), t.vectorBlocks, stream, args);
    }
    return ACL_SUCCESS;
}

int32_t LaunchReference(void* stream, GM_ADDR aa, GM_ADDR bb, GM_ADDR cc, GemmBatchedExArch22Tiling t)
{
    if (t.packed) {
        t.k = t.originalK;
        t.lda = t.originalLda;
        t.ldb = t.originalLdb;
    }
    if (t.policy != 0) {
        GemmLaunchArguments<2> args{{aa, bb}, t};
        const aclError status =
            LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExStatistics), t.vectorBlocks, stream, args);
        if (status != ACL_SUCCESS)
            return status;
    }
    const uint32_t tile = t.transA ? REFERENCE_CAPACITY / 8 : REFERENCE_CAPACITY;
    const uint64_t tasks = (static_cast<uint64_t>(t.m) + tile - 1) / tile * t.n * t.batchCount;
    // Bound each ordered launch without splitting an output element's K sequence.
    constexpr uint64_t MAX_REFERENCE_STEPS = 65536;
    const uint64_t orderedTasks =
        t.k > 0 ? std::max<uint64_t>(1, MAX_REFERENCE_STEPS / static_cast<uint64_t>(t.k)) : MAX_REFERENCE_STEPS;
    const uint64_t step = t.policy != 0 ? tasks : orderedTasks;
    for (t.referenceFirst = 0; t.referenceFirst < tasks; t.referenceFirst += step) {
        t.referenceLast = std::min(tasks, t.referenceFirst + step);
        GemmLaunchArguments<3> args{{aa, bb, cc}, t};
        const aclError status =
            LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExReference), t.vectorBlocks, stream, args);
        if (status != ACL_SUCCESS)
            return status;
    }
    return ACL_SUCCESS;
}

int32_t GemmBatchedExArch22Launch(
    uint32_t blocks, void* stream, const void* a, const void* b, const void* c, void* workspace,
    GemmBatchedExArch22Tiling t, bool scaleOnly)
{
    auto aa = const_cast<uint8_t*>(static_cast<const uint8_t*>(a));
    auto bb = const_cast<uint8_t*>(static_cast<const uint8_t*>(b));
    auto cc = const_cast<uint8_t*>(static_cast<const uint8_t*>(c));
    auto ww = static_cast<uint8_t*>(workspace);
    if (blocks == 0)
        return ACL_ERROR_INVALID_PARAM;
    if (t.policy != 0 || t.route == GemmBatchedExRoute::ReferenceOrder) {
        const int32_t status = LaunchReference(stream, aa, bb, cc, t);
        if (status != ACL_SUCCESS || t.route == GemmBatchedExRoute::ReferenceOrder)
            return status;
    }
    if (t.route == GemmBatchedExRoute::ExactSmall) {
        return LaunchExact(stream, aa, bb, cc, t);
    }
    if (t.route != GemmBatchedExRoute::Regular) {
        void* syncAddress = nullptr;
        const aclError status = aclrtGetHardwareSyncAddr(&syncAddress);
        if (status != ACL_SUCCESS)
            return status;
        t.syncAddress = reinterpret_cast<uint64_t>(syncAddress);
    }
    BalanceRowStrips(blocks, t);
    if (UseInputStream(blocks, t)) {
        GemmLaunchArguments<4> args{{aa, bb, cc, ww}, t};
        return LaunchKernel(reinterpret_cast<const void*>(GemmBatchedExInputStream), blocks, stream, args);
    }
    if (t.route != GemmBatchedExRoute::Regular) {
        if (t.route == GemmBatchedExRoute::RowStrips)
            return LaunchRing<true>(blocks, stream, aa, bb, cc, ww, t);
        return LaunchRing<false>(blocks, stream, aa, bb, cc, ww, t);
    }
    return LaunchRegular(blocks, stream, aa, bb, cc, ww, t, scaleOnly);
}
