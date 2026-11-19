/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "csyr2_kernel.h"
#include "csyr2_atomic_common.h"
#include "csyr2_vector_common.h"
using namespace AscendC;

static uint32_t AtomicBoundary(uint32_t part, uint32_t parts, uint32_t n)
{
    if (parts == 0U) {
        return 0U;
    }
    if (part >= parts) {
        return n;
    }
    const uint64_t weight = n <= 1152U ? 0U : 1024U;
    const uint64_t extra = n <= 1152U ? 0U : 768U;
    const uint64_t target = (uint64_t(n) * (n + 1U) / 2U + weight * n + extra * (n / 4U)) * part / parts;
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2U;
        const uint64_t work =
            uint64_t(mid) * (2ULL * n - mid + 1U) / 2U + weight * mid + extra * (mid < n / 4U ? mid : n / 4U);
        if (work < target) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Compute the rank-two increment in UB, then perform A += delta in GM.
// Each matrix element has one owner; no cross-core floating-point reduction.
template <uint32_t Batch>
__global__ __aicore__ __vector__ void csyr2_lower_atomic_manual(
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t, uint32_t cut1, uint32_t cut2)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csyr2Atomic::Run<false, Batch>(xGm, yGm, aGm, t, cut1, cut2);
}

// Empirical cost model for the existing Batch=4 lower-column ownership.
// Estimate each path in microseconds as the maximum per-core service cost:
//   T = max_core(launch + a*vectorRepeats + b*computeGroups
//                      + c*publications + d*storeKiB + e*startupKiB).
// Pairing shares one mask and three V barriers across two columns, but computes
// an extra 16-complex prefix for the second column and publishes every two
// columns instead of every four. These work counts drive the decision; no
// target-size equality or case ID is used to select the faster path.

// Calibrated on NPU7 / CANN 9.1.0-beta.3: 18 training sizes and five held-out
// sizes in the non-contiguous 1152 < n <= 2048 family; n=1213 excluded from both.
// Keep the existing short-size policy. The 192 KiB UB bound is a hard resource
// constraint, separate from the fitted performance estimate.

// Coefficients are effective fitted costs, NOT individual instruction latencies.
// Features are correlated and the additive model does not explicitly simulate
// V/MTE overlap; it is more useful for path differences than absolute latency.
// Counts are source-derived estimates, not compiler-emitted facts or a proof of
// synchronization/access legality. Revalidate on other devices/toolchains or
// after changing ownership, batching, arithmetic, or synchronization.
// The 0.57 us selection margin below comes from held-out prediction error plus
// repeat variation; it is neither a confidence bound nor acceptance headroom.
static bool PreferFullPair(uint32_t n, uint32_t cores, uint32_t cut1, uint32_t cut2)
{
    if (12ULL * ((8ULL * n + 255ULL) & ~255ULL) > 192ULL * 1024ULL) {
        return false;
    }
    double manualMax = 0.0;
    double pairMax = 0.0;
    for (uint32_t core = 0; core < cores; ++core) {
        const uint32_t group = core / 16U;
        const uint32_t lane = core % 16U;
        const uint32_t groups = cores / 16U;
        const uint32_t start = group == 0U ? 0U : group == 1U ? cut1 : cut2;
        const uint32_t stop = group + 1U == groups ? n : group == 0U ? cut1 : cut2;
        const uint32_t first = ((start + 15U) & ~15U) + lane;
        const uint32_t limit = (stop + 15U) & ~15U;
        const uint32_t last = limit < n ? limit : n;
        if (first >= last) {
            continue;
        }
        const uint32_t count = (last - first + 15U) / 16U;
        const uint32_t rows = n - first;
        const uint32_t pairs = count / 2U;
        const uint32_t odd = count % 2U;
        const uint32_t r0 = (rows + 31U) / 32U;
        const uint32_t r1 = rows > 16U ? (rows - 16U + 31U) / 32U : 0U;
        const uint32_t staircase = pairs == 0U ? 0U : pairs * (pairs - 1U);
        const uint32_t manualRepeats = 4U * (pairs * (r0 + r1) - staircase + odd * (r0 - pairs));
        const uint32_t pairRepeats = 4U * (2U * pairs * r0 - staircase + odd * (r0 - pairs));
        const double storeKiB = 8.0 * (double(count) * rows - 8.0 * count * (count - 1U)) / 1024.0;
        const double startupKiB = 16.0 * rows / 1024.0;
        const double common = 4.0300792994 + 0.00632654449 * storeKiB + 0.07285235332 * startupKiB;
        const double manual =
            common + 0.0001544727243 * manualRepeats + 0.1053052489 * count + 0.1222919099 * ((count + 3U) / 4U);
        const double paired =
            common + 0.0001544727243 * pairRepeats + (0.1053052489 + 0.1222919099) * ((count + 1U) / 2U);
        if (manual > manualMax) {
            manualMax = manual;
        }
        if (paired > pairMax) {
            pairMax = paired;
        }
    }
    // Held-out maximum saving-prediction error (0.2814 us) plus maximum
    // half round-to-round saving difference (0.2809 us), rounded upward.
    return manualMax - pairMax > 0.57;
}

void csyr2_lower_columns_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    // The contiguous route consumes A directly with UB-resident delta plus
    // SetAtomicAdd write-back; it pays off for longer columns with aligned
    // row pitch. Keep residue-lane atomic stores for short or unaligned matrices.
    if (t.n >= 1024U && t.n <= 2048U && t.n % 8U == 0U) {
        csyr2_lower_contiguous_do(blocks, stream, x, y, a, t);
        return;
    }
    // Caller must guarantee 64 <= n <= 3072 and at least 16 blocks.
    // Four resident vectors plus two output batches: <= 192 KiB.
    const uint32_t cap = t.n <= 1152U ? 32U : 48U;
    const uint32_t cores = ((blocks > cap ? cap : blocks) / 16U) * 16U;
    const uint32_t cut1 = AtomicBoundary(1U, cores / 16U, t.n);
    const uint32_t cut2 = AtomicBoundary(2U, cores / 16U, t.n);
    // Experimental range dispatch: start each atomic write after two of the
    // four columns are ready, allowing MTE3 to overlap the second half's VECTOR work.
    if (t.n <= 1152U || PreferFullPair(t.n, cores, cut1, cut2)) {
        csyr2_lower_halfwrite_do(cores, stream, x, y, a, t, cut1, cut2);
        return;
    }
    if (t.n <= 2048U) {
        csyr2_lower_atomic_manual<4U><<<cores, nullptr, stream>>>(x, y, a, t, cut1, cut2);
    } else {
        csyr2_lower_atomic_manual<2U><<<cores, nullptr, stream>>>(x, y, a, t, cut1, cut2);
    }
}

// Full batches have two known-valid pairs; tail batches retain the guarded path.
template <uint32_t Begin>
__aicore__ inline void FullPair(
    LocalTensor<float> av, Csyr2Vector::Vectors& v, GlobalTensor<float> a, uint32_t pitch, uint32_t col, uint32_t first,
    const Csyr2TilingData& t, uint32_t buffer)
{
    const uint32_t c = col + 16U * Begin;
    const uint32_t off = 2U * (c - first);
    auto d0 = av[pitch * Begin];
    auto d1 = av[pitch * (Begin + 1U)];
    v.PairIncrement(d0, d1, off, t.n - c);
    const event_t ready = Begin == 0U ? (buffer ? EVENT_ID1 : EVENT_ID0) : (buffer ? EVENT_ID5 : EVENT_ID4);
    set_flag(PIPE_V, PIPE_MTE3, ready);
    wait_flag(PIPE_V, PIPE_MTE3, ready);
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        a[2ULL * (uint64_t(c) * t.lda + c)], d0, 0, 1, 8U * (t.n - c), 0, 0, 0, 0);
    const uint32_t c1 = c + 16U;
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        a[2ULL * (uint64_t(c1) * t.lda + c1)], d1[32U], 0, 1, 8U * (t.n - c1), 0, 0, 0, 0);
}

template <uint32_t Batch>
__aicore__ inline void PublishPartialPairs(
    LocalTensor<float> av, Csyr2Vector::Vectors& v, uint32_t pitch, uint32_t col, uint32_t first, uint32_t last,
    const Csyr2TilingData& t, uint32_t buffer)
{
    constexpr uint32_t Half = Batch == 4U ? 2U : Batch;
    for (uint32_t begin = 0; begin < Batch; begin += Half) {
        if (col + 16U * begin >= last) {
            break;
        }
        if (Half == 2U && col + 16U * (begin + 1U) < last) {
            const uint32_t c = col + 16U * begin;
            const uint32_t off = 2U * (c - first);
            v.PairIncrement(av[pitch * begin], av[pitch * (begin + 1U)], off, t.n - c);
        } else {
            for (uint32_t j = begin; j < begin + Half && col + 16U * j < last; ++j) {
                const uint32_t c = col + 16U * j;
                const uint32_t offset = 2U * (c - first);
                v.Increment(av[pitch * j], offset, offset, t.n - c);
            }
        }
        const event_t ready = begin == 0U ? (buffer ? EVENT_ID1 : EVENT_ID0) : (buffer ? EVENT_ID5 : EVENT_ID4);
        set_flag(PIPE_V, PIPE_MTE3, ready);
        wait_flag(PIPE_V, PIPE_MTE3, ready);
        for (uint32_t j = begin; j < begin + Half && col + 16U * j < last; ++j) {
            const uint32_t c = col + 16U * j;
            ub_to_gm_align<ArchType::ASCEND_V220, float>(
                v.a[2ULL * (uint64_t(c) * t.lda + c)], av[pitch * j + ((j & 1U) ? 32U : 0U)], 0, 1, 8U * (t.n - c), 0,
                0, 0, 0);
        }
    }
}

// Compute the rank-two increment in UB, then perform A += delta in GM.
// Each matrix element has one owner; no cross-core floating-point reduction.
template <uint32_t Batch>
__global__ __aicore__ __vector__ void csyr2_lower_full_pair(
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t, uint32_t cut1, uint32_t cut2)
{
    static_assert(Batch == 4U, "Two pairs per full batch");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const Csyr2Vector::ColumnRange range = Csyr2Vector::AtomicRange(t.n, cut1, cut2);
    const uint32_t first = range.first;
    const uint32_t last = range.last;
    if (first >= last) {
        return;
    }
    TPipe pipe;
    const uint32_t rows0 = t.n - first;
    // Both columns compute from the earlier row origin. The second output
    // discards a 16-complex prefix; no input read needs an extra UB guard.
    const uint32_t bytes = (8U * rows0 + 255U) & ~255U;
    Csyr2Vector::AtomicBuffers storage;
    storage.Init(pipe, xGm, yGm, aGm, bytes, Batch);
    auto index = storage.ab0.Get<int32_t>();
    storage.v.Load(first, rows0);
    // (i + 1) - 2*(i is odd) == (i xor 1), then convert to byte offsets.
    Csyr2Vector::InitRotationIndex(index, 2U * rows0);
    PipeBarrier<PIPE_ALL>();
    Csyr2Vector::RotateVectors(
        storage.v.xr, storage.v.yr, storage.v.xv, storage.v.yv, index.ReinterpretCast<uint32_t>(), 2U * rows0);
    PipeBarrier<PIPE_V>();
    const uint32_t pitch = bytes / sizeof(float);
    bool pending[2] = {false, false};
    uint32_t buffer = 0;
    SetAtomicAdd<float>();
    SetMaskCount();
    for (uint32_t col = first; col < last; col += 16U * Batch) {
        auto av = Csyr2Atomic::AcquireOutputBuffer(storage, buffer, pending);
        // Publish the first half while VECTOR computes the second half.
        // The MTE3->V release below still protects the entire output buffer.
        if (col + 48U < last) {
            FullPair<0U>(av, storage.v, storage.v.a, pitch, col, first, t, buffer);
            FullPair<2U>(av, storage.v, storage.v.a, pitch, col, first, t, buffer);
        } else {
            PublishPartialPairs<Batch>(av, storage.v, pitch, col, first, last, t, buffer);
        }
        set_flag(PIPE_MTE3, PIPE_V, buffer ? EVENT_ID3 : EVENT_ID2);
        pending[buffer] = true;
        buffer ^= 1U;
    }
    Csyr2Vector::Finish<false, true>(pending);
}

void csyr2_lower_halfwrite_do(
    uint32_t cores, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t, uint32_t cut1,
    uint32_t cut2)
{
    csyr2_lower_full_pair<4U><<<cores, nullptr, stream>>>(x, y, a, t, cut1, cut2);
}
