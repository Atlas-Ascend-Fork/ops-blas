/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// This source defines the ctrsm batched kernel entry points and launchers
// for the small/ordered families; the shared device templates live in
// ctrsmbatched_kernel_shared.h.

#include "ctrsmbatched_kernel_shared.h"

// Grouped inverse of small diagonal panels.
namespace CtrsmBatchedGroupedInverse {
using namespace AscendC;
constexpr uint32_t K = 32, GROUPS = 4, WIDTH = 128, STRIDE = 264, OUT_STRIDE = 72;
constexpr uint32_t OUT_SIZE = 64 * OUT_STRIDE;

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Load(
    GM_ADDR workspace, uint32_t kp, uint32_t k, uint32_t batches, uint64_t base, __ubuf__ float* aa, __ubuf__ float* bb,
    __ubuf__ uint32_t* indices)
{
    const uint32_t panels = (k + K - 1) / K;
    const auto source = reinterpret_cast<__gm__ float*>(workspace);
    if (threadIdx.x < WIDTH)
        indices[threadIdx.x] = threadIdx.x / K * 2;
    for (uint32_t q = threadIdx.x; q < GROUPS * K; q += 256) {
        const uint32_t group = q / K, r = q % K;
        const uint64_t job = base + group;
        const uint32_t start = job % panels * K, size = k - start < K ? k - start : K;
        aclblasComplex value{1, 0};
        if (job < static_cast<uint64_t>(batches) * panels && r < size) {
            const uint64_t off = job / panels * 4ULL * kp * kp + 2ULL * (start + r) * (2 * kp) + 2 * (start + r);
            value = {-source[off], -source[off + 2 * kp]};
            const float inverse = __fdividef(1.0f, value.real * value.real + value.imag * value.imag);
            value = {value.real * inverse, -value.imag * inverse};
        }
        const uint32_t dst = (r * (r + 1) + 2 * r) * GROUPS + 2 * group;
        aa[dst] = value.real;
        aa[dst + 1] = value.imag;
    }
    for (uint32_t q = threadIdx.x; q < GROUPS * K * K; q += 256) {
        const uint32_t group = q / (K * K), r = q / K % K, c = q % K;
        const uint64_t job = base + group;
        const uint32_t start = job % panels * K;
        const uint32_t size = k - start < K ? k - start : K;
        const bool valid = job < static_cast<uint64_t>(batches) * panels;
        if (r > c) {
            aclblasComplex value{0, 0};
            if (valid && r < size && c < size) {
                const uint64_t off = job / panels * 4ULL * kp * kp + 2ULL * (start + r) * (2 * kp) + 2 * (start + c);
                value = {-source[off], -source[off + 2 * kp]};
            }
            const uint32_t dst = (r * (r + 1) + 2 * c) * GROUPS + 2 * group;
            aa[dst] = value.real;
            aa[dst + 1] = value.imag;
        }
        bb[r * STRIDE + group * K + c] = valid && r < size && r == c ? 1.0f : 0.0f;
        bb[r * STRIDE + WIDTH + group * K + c] = 0.0f;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Pack(__ubuf__ float* bb, __ubuf__ float* out)
{
    for (uint32_t q = threadIdx.x; q < GROUPS * K * K; q += 256) {
        const uint32_t group = q / (K * K), r = q / K % K, c = q % K;
        const float real = bb[r * STRIDE + group * K + c];
        const float imag = bb[r * STRIDE + WIDTH + group * K + c];
        const uint32_t offset = group * OUT_SIZE + 2 * r * OUT_STRIDE + 2 * c;
        out[offset] = real;
        out[offset + 1] = -imag;
        out[offset + OUT_STRIDE] = imag;
        out[offset + OUT_STRIDE + 1] = real;
    }
}

__aicore__ inline void Run(GM_ADDR workspace, uint32_t kp, uint32_t k, uint32_t batches, __ubuf__ float* storage)
{
    auto aa = storage;
    auto bb = aa + GROUPS * K * (K + 1);
    auto indices = reinterpret_cast<__ubuf__ uint32_t*>(bb + K * STRIDE);
    auto out = reinterpret_cast<__ubuf__ float*>(indices + WIDTH);
    static_assert(GROUPS * K * (K + 1) + K * STRIDE + WIDTH + GROUPS * OUT_SIZE <= 53248);
    const uint32_t panels = (k + K - 1) / K;
    const uint64_t jobs = static_cast<uint64_t>(batches) * panels;
    for (uint64_t base = static_cast<uint64_t>(GetBlockIdx()) * GROUPS; base < jobs;
         base += static_cast<uint64_t>(GetBlockNum()) * GROUPS) {
        asc_vf_call<Load>(dim3{256, 1, 1}, workspace, kp, k, batches, base, aa, bb, indices);
        PipeBarrier<PIPE_ALL>();
        CtrsmBatchedSmallVector::Compute<K, GROUPS, WIDTH, 2>(aa, bb, indices, K);
        PipeBarrier<PIPE_ALL>();
        asc_vf_call<Pack>(dim3{256, 1, 1}, bb, out);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t group = 0; group < GROUPS && base + group < jobs; ++group) {
            const uint64_t job = base + group, batch = job / panels;
            const uint32_t start = job % panels * K, size = k - start < K ? k - start : K;
            auto dst = reinterpret_cast<__gm__ float*>(workspace) + batch * 4ULL * kp * kp + 2ULL * start * (2 * kp) +
                       2 * start;
            asc_copy_ub2gm_align(
                dst, out + group * OUT_SIZE, 2 * size, 2 * size * sizeof(float), 0, 2ULL * kp * sizeof(float),
                OUT_STRIDE * sizeof(float));
        }
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace CtrsmBatchedGroupedInverse

// Kernel entry points and small-vector dispatch.
namespace {
constexpr uint32_t THREADS = 128;
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void CtrsmBatchedSimt(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint32_t block, uint32_t blocks)
{
    const uint64_t rhsCount = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    const uint64_t work = rhsCount * t.batchCount;
    const bool zero = t.alpha.real == 0.0f && t.alpha.imag == 0.0f;
    for (uint64_t job = static_cast<uint64_t>(block) * THREADS + threadIdx.x; job < work;
         job += static_cast<uint64_t>(blocks) * THREADS) {
        const uint64_t batch = job / rhsCount;
        const auto bp = reinterpret_cast<__gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
        const __gm__ aclblasComplex* ap = nullptr;
        if (!zero) {
            ap = reinterpret_cast<const __gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
        }
        CtrsmBatched::SolveRhs(ap, bp, static_cast<uint32_t>(job % rhsCount), t);
    }
}
} // namespace

extern "C" __global__ __aicore__ void ctrsm_batched_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<CtrsmBatchedSimt>(dim3{THREADS, 1, 1}, a, b, t, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}

void ctrsm_batched_vector_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream);

void ctrsm_batched_unit_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream);

template <uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH = 128, uint32_t TRANSFER_THREADS = 256>
__global__ __aicore__ void ctrsm_batched_small_vector_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float storage[CtrsmBatchedSmallVector::STORAGE_FLOATS];
    CtrsmBatchedSmallVector::Run<MAX_K, GROUPS, WIDTH, 0, 0, TRANSFER_THREADS>(a, b, t, storage);
}

template <uint32_t K, uint32_t GROUPS, uint32_t WIDTH, bool SQUARE, uint32_t TRANSFER_THREADS = 256>
__global__ __aicore__ void ctrsm_batched_fixed_vector_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float storage[CtrsmBatchedSmallVector::STORAGE_FLOATS];
    CtrsmBatchedSmallVector::Run<K, GROUPS, WIDTH, K, SQUARE ? K : 0, TRANSFER_THREADS>(a, b, t, storage);
}

// Only instantiate group sizes whose tiles fit in UB. The shared launch plan
// chooses the same width/group pair used by Host to calculate block count.
template <uint32_t MAX_K, uint32_t WIDTH, uint32_t GROUPS>
void LaunchGrouped(
    uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t groups, uint32_t blocks, void* stream)
{
    if constexpr (GROUPS > 1) {
        if (groups < GROUPS) {
            LaunchGrouped<MAX_K, WIDTH, GROUPS / 2>(a, b, t, groups, blocks, stream);
            return;
        }
    }
    ctrsm_batched_small_vector_kernel<MAX_K, GROUPS, WIDTH><<<blocks, nullptr, stream>>>(a, b, t);
}

void LaunchFixedSquare(
    uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, const CtrsmBatchedVectorPlan& plan, uint32_t blocks,
    void* stream)
{
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    if (k == 8 && plan.width == 64) {
        ctrsm_batched_fixed_vector_kernel<8, 8, 64, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 8) {
        ctrsm_batched_fixed_vector_kernel<8, 32, 256, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 16 && plan.width == 64) {
        ctrsm_batched_fixed_vector_kernel<16, 4, 64, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 16) {
        ctrsm_batched_fixed_vector_kernel<16, 8, 128, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 32 && plan.width == 64) {
        ctrsm_batched_fixed_vector_kernel<32, 2, 64, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 32) {
        ctrsm_batched_fixed_vector_kernel<32, 4, 128, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    ctrsm_batched_fixed_vector_kernel<64, 1, 64, true><<<blocks, nullptr, stream>>>(a, b, t);
}

void LaunchSmallVector(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream)
{
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const auto plan = GetCtrsmBatchedVectorPlan(t);
    if (plan.fixedSquare) {
        LaunchFixedSquare(a, b, t, plan, blocks, stream);
        return;
    }
    if (k <= 8 && plan.width == 64) {
        LaunchGrouped<8, 64, 8>(a, b, t, plan.groups, blocks, stream);
    } else if (k <= 8) {
        LaunchGrouped<8, 256, 32>(a, b, t, plan.groups, blocks, stream);
    } else if (k <= 16) {
        LaunchGrouped<16, 128, 16>(a, b, t, plan.groups, blocks, stream);
    } else if (k <= 32) {
        LaunchGrouped<32, 128, 16>(a, b, t, plan.groups, blocks, stream);
    } else if (k <= 64) {
        LaunchGrouped<64, 128, 4>(a, b, t, plan.groups, blocks, stream);
    } else {
        LaunchGrouped<128, 128, 1>(a, b, t, plan.groups, blocks, stream);
    }
}

void ctrsm_batched_kernel_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream)
{
    if (t.diag == ACLBLAS_UNIT) {
        const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
        if ((t.alpha.real != 0 || t.alpha.imag != 0) && k > 1 && k <= 2048)
            ctrsm_batched_unit_do(a, b, t, blocks, stream);
        else
            ctrsm_batched_kernel<<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    if (t.alpha.real == 0 && t.alpha.imag == 0)
        ctrsm_batched_kernel<<<blocks, nullptr, stream>>>(a, b, t);
    else if (k < 128)
        LaunchSmallVector(a, b, t, blocks, stream);
    else
        ctrsm_batched_vector_do(a, b, t, blocks, stream);
}

extern "C" __global__ __aicore__ void ctrsm_cube_inverse_grouped(
    GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float storage[CtrsmBatchedSmallVector::STORAGE_FLOATS];
    CtrsmBatchedGroupedInverse::Run(workspace, kp, t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n, t.batchCount, storage);
}

void ctrsm_batched_grouped_inverse_do(
    uint8_t* workspace, const CtrsmBatchedTilingData& t, uint32_t kp, uint32_t blocks, void* stream)
{
    ctrsm_cube_inverse_grouped<<<blocks, nullptr, stream>>>(workspace, t, kp);
}
