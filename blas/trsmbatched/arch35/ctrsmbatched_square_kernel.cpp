/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Reuse private device templates while keeping kernel UB allocation isolated.
#define CTRSM_BATCHED_SHARED_ONLY
#include "ctrsmbatched_kernel.cpp"

// Pipelined square NON_UNIT solve for identity alpha.
namespace CtrsmBatchedSquare {
using namespace AscendC;
using namespace AscendC::Reg;

__simd_callee__ inline void Check(
    RegTensor<float>& value, RegTensor<float>& flags, RegTensor<float>& one, MaskReg& active, MaskReg& all)
{
    RegTensor<float> absolute;
    MaskReg good, bad;
    Abs(absolute, value, all);
    Compares<float, CMPMODE::LE>(good, absolute, 1.0e15f, all);
    Not(bad, good, all);
    And(bad, bad, active, all);
    Select(flags, one, flags, bad);
}

template <uint32_t K, uint32_t W, uint32_t INPUT_THREADS = 0, bool PAD_OUTPUT = false>
__aicore__ inline void Store(
    __ubuf__ float* bb, __ubuf__ float* rawB, __ubuf__ float* flag, uint32_t groups, bool upper,
    __ubuf__ uint32_t* inputFlags = nullptr)
{
    constexpr uint32_t STRIDE = 2 * W + 8;
    __VEC_SCOPE__
    {
        auto all = CreateMask<float>();
        uint32_t oneCount = 1;
        auto oneMask = UpdateMask<float>(oneCount);
        RegTensor<uint32_t> lane, base, index;
        RegTensor<float> ar, ai, flags, one, reduced;
        Duplicate(flags, 0.0f);
        Duplicate(one, 1.0f);
        if constexpr (INPUT_THREADS > 0) {
            RegTensor<uint32_t> input;
            MaskReg unsafe;
            for (uint16_t chunk = 0; chunk < uint16_t(INPUT_THREADS / 64); ++chunk) {
                DataCopy(input, inputFlags + chunk * 64);
                Compares<uint32_t, CMPMODE::NE>(unsafe, input, 0U, all);
                Select(flags, one, flags, unsafe);
            }
        }
        for (uint16_t chunk = 0; chunk < uint16_t(W / 64); ++chunk) {
            Arange(reinterpret_cast<RegTensor<int32_t>&>(lane), int32_t(chunk * 64));
            Muls(base, lane, 2 * K + (PAD_OUTPUT ? 8 : 0), all);
            uint32_t active = groups * K > chunk * 64 ? groups * K - chunk * 64 : 0;
            auto activeMask = UpdateMask<float>(active);
            for (uint16_t r = 0; r < uint16_t(K); ++r) {
                const uint32_t row = upper ? K - 1 - r : r;
                DataCopy(ar, bb + r * STRIDE + chunk * 64);
                DataCopy(ai, bb + r * STRIDE + W + chunk * 64);
                Check(ar, flags, one, activeMask, all);
                Check(ai, flags, one, activeMask, all);
                Adds(index, base, 2 * row, all);
                Scatter(rawB, ar, index, activeMask);
                Adds(index, index, 1U, all);
                Scatter(rawB, ai, index, activeMask);
            }
        }
        Reduce<ReduceType::MAX>(reduced, flags, all);
        DataCopy(flag, reduced, oneMask);
    }
}

template <uint32_t K, uint32_t G>
__simt_vf__ __aicore__ LAUNCH_BOUND(128) inline void Fallback(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint32_t base, uint32_t groups)
{
    t.side = ACLBLAS_SIDE_LEFT;
    t.trans = ACLBLAS_OP_N;
    t.diag = ACLBLAS_NON_UNIT;
    t.m = K;
    t.n = K;
    t.alpha.real = 1.0f;
    t.alpha.imag = 0.0f;
    for (uint32_t q = threadIdx.x; q < groups * K; q += 128) {
        const uint32_t batch = base + q / K;
        auto ap = reinterpret_cast<const __gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
        auto bp = reinterpret_cast<__gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
        CtrsmBatched::SolveOrderedRhs(ap, bp, q % K, t);
    }
}
} // namespace CtrsmBatchedSquare

template <uint32_t K, uint32_t W>
__aicore__ inline void PrefetchSquare(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint32_t base, __ubuf__ float* rawB)
{
    constexpr uint32_t G = W / K;
    auto rawA = rawB + 2 * G * K * K;
    auto pointers = reinterpret_cast<__ubuf__ uint64_t*>(rawA + 2 * G * K * K);
    const uint32_t groups = t.batchCount - base < G ? t.batchCount - base : G;
    // Fetch the contiguous pointer lists together instead of serial GM scalar
    // reads for every small matrix. Reuse B pointers again during the store.
    if constexpr (K == 8) {
        asc_copy_gm2ub_align(
            reinterpret_cast<__ubuf__ uint8_t*>(pointers),
            reinterpret_cast<__gm__ uint8_t*>(a) + base * sizeof(uint64_t), 1, groups * sizeof(uint64_t), 0, 0, false,
            0, 0, 0);
        asc_copy_gm2ub_align(
            reinterpret_cast<__ubuf__ uint8_t*>(pointers + G),
            reinterpret_cast<__gm__ uint8_t*>(b) + base * sizeof(uint64_t), 1, groups * sizeof(uint64_t), 0, 0, false,
            0, 0, 0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t group = 0; group < groups; ++group) {
        const uint64_t aAddress = K == 8 ? pointers[group] : reinterpret_cast<__gm__ uint64_t*>(a)[base + group];
        const uint64_t bAddress = K == 8 ? pointers[G + group] : reinterpret_cast<__gm__ uint64_t*>(b)[base + group];
        auto ap = reinterpret_cast<__gm__ float*>(aAddress);
        auto bp = reinterpret_cast<__gm__ float*>(bAddress);
        if (t.lda == K) {
            asc_copy_gm2ub_align(rawA + group * 2 * K * K, ap, 1, 2 * K * K * sizeof(float), 0, 0, false, 0, 0, 0);
        } else {
            asc_copy_gm2ub_align(
                rawA + group * 2 * K * K, ap, K, 2 * K * sizeof(float), 0, 0, false, 0,
                uint64_t(t.lda) * 2 * sizeof(float), 2 * K * sizeof(float));
        }
        if (t.ldb == K) {
            asc_copy_gm2ub_align(rawB + group * 2 * K * K, bp, 1, 2 * K * K * sizeof(float), 0, 0, false, 0, 0, 0);
        } else {
            asc_copy_gm2ub_align(
                rawB + group * 2 * K * K, bp, K, 2 * K * sizeof(float), 0, 0, false, 0,
                uint64_t(t.ldb) * 2 * sizeof(float), 2 * K * sizeof(float));
        }
    }
}

template <uint32_t K, uint32_t W, bool PAD_OUTPUT>
__aicore__ inline void StoreSquareResult(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint32_t base, uint32_t groups, __ubuf__ float* flag,
    __ubuf__ float* rawB, __ubuf__ float* paddedOut)
{
    using namespace AscendC;
    constexpr uint32_t G = W / K;
    if (reinterpret_cast<volatile __ubuf__ float*>(flag)[0] != 0) {
        asc_vf_call<CtrsmBatchedSquare::Fallback<K, G>>(dim3{128, 1, 1}, a, b, t, base, groups);
        PipeBarrier<PIPE_ALL>();
    } else {
        auto pointers = reinterpret_cast<__ubuf__ uint64_t*>(rawB + 4 * G * K * K);
        for (uint32_t group = 0; group < groups; ++group) {
            const uint64_t address = K == 8 ? pointers[G + group] : reinterpret_cast<__gm__ uint64_t*>(b)[base + group];
            auto bp = reinterpret_cast<__gm__ float*>(address);
            asc_copy_ub2gm_align(
                bp, (PAD_OUTPUT ? paddedOut : rawB) + group * K * (2 * K + (PAD_OUTPUT ? 8 : 0)), K,
                2 * K * sizeof(float), 0, uint64_t(t.ldb) * 2 * sizeof(float),
                (2 * K + (PAD_OUTPUT ? 8 : 0)) * sizeof(float));
        }
        PipeBarrier<PIPE_ALL>();
    }
}

template <uint32_t K, uint32_t W, bool PIPELINED>
__aicore__ inline void ComputeSquareTile(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint32_t base, uint32_t stepBase, uint32_t iteration,
    __ubuf__ float* aa, __ubuf__ float* bb, __ubuf__ uint32_t* indices, __ubuf__ uint32_t* badBase)
{
    using namespace AscendC;
    constexpr uint32_t G = W / K, T = 512, REGION = T + 8 + 4 * G * K * K + 4 * G;
    // The next batch reads a separate region while the current batch is computed.
    if constexpr (PIPELINED) {
        const uint32_t next = base + stepBase;
        if (next < t.batchCount)
            PrefetchSquare<K, W>(
                a, b, t, next, reinterpret_cast<__ubuf__ float*>(badBase + ((iteration + 1) % 2) * REGION + T + 8));
    }
    if constexpr (K == 8)
        CtrsmBatchedSmallVector::ComputeFixed8<K, G, W, 9>(aa, bb, indices);
    else
        CtrsmBatchedSmallVector::Compute<K, G, W, 9>(aa, bb, indices, K, 0, W);
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t K, uint32_t W, bool PIPELINED = false, bool PAD_OUTPUT = false>
__global__ __aicore__ void ctrsm_batched_square_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    // Separate kernel-local storage avoids sharing compiler-generated UB symbols between entry points.
    // Padded output columns spread strided SIMD stores across UB banks.
    constexpr uint32_t G = W / K, AS = G * K * (K + 1), BS = K * (2 * W + 8), T = 512,
                       REGION = T + 8 + 4 * G * K * K + 4 * G;
    __ubuf__ float storage[49152];
    static_assert(AS + BS + W + (PIPELINED ? 2 : 1) * REGION + (PAD_OUTPUT ? W * (2 * K + 8) : 0) <= 49152);
    auto aa = storage, bb = aa + AS;
    auto indices = reinterpret_cast<__ubuf__ uint32_t*>(bb + BS), badBase = indices + W;
    auto paddedOut = reinterpret_cast<__ubuf__ float*>(badBase + (PIPELINED ? 2 : 1) * REGION);
    const bool upper = t.uplo == ACLBLAS_UPPER;
    const uint32_t firstBase = GetBlockIdx() * G;
    const uint32_t stepBase = GetBlockNum() * G;
    uint32_t iteration = 0;
    if constexpr (PIPELINED) {
        if (firstBase < t.batchCount)
            PrefetchSquare<K, W>(a, b, t, firstBase, reinterpret_cast<__ubuf__ float*>(badBase + T + 8));
        PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t base = firstBase; base < t.batchCount; base += stepBase, ++iteration) {
        const uint32_t groups = t.batchCount - base < G ? t.batchCount - base : G;
        auto bad = badBase + (PIPELINED ? (iteration % 2) * REGION : 0);
        auto rawB = reinterpret_cast<__ubuf__ float*>(bad + T + 8);
        auto flag = reinterpret_cast<__ubuf__ float*>(bad + T);
        if constexpr (!PIPELINED) {
            PrefetchSquare<K, W>(a, b, t, base, rawB);
            PipeBarrier<PIPE_ALL>();
        }
        asc_vf_call<CtrsmBatchedSmallVector::Transfer<K, G, W, false, false, K, K, 1, T, true>>(
            dim3{T, 1, 1}, a, b, t, base, aa, bb, indices, bad, K);
        PipeBarrier<PIPE_ALL>();
        ComputeSquareTile<K, W, PIPELINED>(a, b, t, base, stepBase, iteration, aa, bb, indices, badBase);
        CtrsmBatchedSquare::Store<K, W, T, PAD_OUTPUT>(bb, PAD_OUTPUT ? paddedOut : rawB, flag, groups, upper, bad);
        PipeBarrier<PIPE_ALL>();
        StoreSquareResult<K, W, PAD_OUTPUT>(a, b, t, base, groups, flag, rawB, paddedOut);
    }
}
void ctrsm_batched_square_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream)
{
    if (t.m == 8 && t.batchCount <= 256)
        ctrsm_batched_square_kernel<8, 64, false, false><<<blocks, nullptr, stream>>>(a, b, t);
    else if (t.m == 8)
        ctrsm_batched_square_kernel<8, 256, false, true><<<blocks, nullptr, stream>>>(a, b, t);
    else if (t.m == 16)
        ctrsm_batched_square_kernel<16, 128, true, true><<<blocks, nullptr, stream>>>(a, b, t);
    else
        ctrsm_batched_square_kernel<32, 128, false, true><<<blocks, nullptr, stream>>>(a, b, t);
}
