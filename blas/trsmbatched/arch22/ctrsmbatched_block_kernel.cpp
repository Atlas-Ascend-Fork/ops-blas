/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define ASCENDC_CUBE_ONLY
#include "ctrsmbatched_kernel.h"
#include "ctrsmbatched_kernel_common.h"
#include "trsv/arch22/ctrsv_generic_kernel.h"
using namespace AscendC;
__aicore__ inline void CtrsmPublishDiagonal(
    GlobalTensor<float> workspace, LocalTensor<float> out, int task, int panels, int order, bool diagonalOnly,
    bool clearTail = true)
{
    if (panels == 0) {
        return;
    }
    const int sub = task % panels, half = sub % 2, large = (order + 63) / 64;
    const uint64_t dest = (uint64_t(task / panels) * large + sub / 2) * 16384;
    CtrsmWrite(workspace[dest + half * 64 * 128 + half * 64], out, 64, 256, 0, 256);
    if (!diagonalOnly) {
        CtrsmFence<HardEvent::MTE3_V>();
        Duplicate(out, 0.f, 4096);
        CtrsmFence<HardEvent::V_MTE3>();
        CtrsmWrite(workspace[dest + half * 64 * 128 + (1 - half) * 64], out, 64, 256, 0, 256);
        if (sub + 1 == panels && half == 0 && clearTail) {
            DataCopy(workspace[dest + 8192], out, 4096);
            DataCopy(workspace[dest + 12288], out, 4096);
        }
        CtrsmFence<HardEvent::MTE3_V>();
    }
}
namespace CtrsmBatchedGroupedInverse {
template <bool Upper, bool Swap = false, bool Conj = false, bool Unit = false>
class PlanarBatch : private CtrsmPlanar<32, 8, 2080, 0, 0, 0, 4224> {
    bool diagonalOnly = false;
    GlobalTensor<uint64_t> ap;
    GlobalTensor<uint32_t> indices;
    int panels, lda, order;
    __aicore__ inline void LoadIndices(int offset, int count)
    {
        CtrsmFence<HardEvent::V_MTE2>();
        DataCopy(idx.ReinterpretCast<uint32_t>(), indices[offset], count);
        CtrsmFence<HardEvent::MTE2_V>();
    }
    __aicore__ inline void PadDiagonal(int first, int count)
    {
        if (order % K) {
            for (int b = 0; b < count; b++) {
                const int valid = (first + b) % panels == panels - 1 ? order % K : K;
                if (valid >= K)
                    continue;
                for (int row = valid; row < K; row += 8) {
                    const int rows = K - row < 8 ? K - row : 8;
                    uint64_t mask = 0;
                    for (int j = 0; j < rows; j++)
                        mask |= uint64_t(1) << (j * G + b);
                    AscendC::SetVectorMask<float>(0, mask);
                    Duplicate<float, false>(den[row * G], 1.f, mask, 1, 1, 8);
                }
            }
            ResetMask();
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadDiagonal(int first, int count)
    {
        if constexpr (Unit) {
            Duplicate(dr, 1.f, D);
            Duplicate(di, 0.f, D);
        } else {
            LoadIndices(0, D);
            PipeBarrier<PIPE_V>();
            Gather(dr, raw, idx.ReinterpretCast<uint32_t>(), 0, D);
            Gather(di, raw, idx.ReinterpretCast<uint32_t>(), 4, D);
            Mul(den, dr, dr, D);
            Mul(tmp, di, di, D);
            Add(den, den, tmp, D);
            PipeBarrier<PIPE_V>();
            PadDiagonal(first, count);
            Div(dr, dr, den, D);
            Div(di, di, den, D);
            if constexpr (Conj)
                Muls(di, di, -1.f, D);
        }
    }
    __aicore__ inline void LoadA(int first, int count)
    {
        if constexpr (!Unit) {
            if (order % K) {
                for (int b = 0; b < count; b++) {
                    if ((first + b) % panels == panels - 1)
                        Duplicate(raw[b * A], 0.f, A);
                }
            }
        }
        CtrsmFence<HardEvent::V_MTE2>();
        for (int b = 0; b < count; b++) {
            const int task = first + b, base = task % panels * K,
                      valid = Unit ? K : (order - base < K ? order - base : K);
            auto ag = CtrsmGlobal<float>((__gm__ float*)ap.GetValue(task / panels) + uint64_t(base) * (lda + 1) * 2);
            CtrsmRead(raw[b * A], ag, valid, valid * 8, (lda - valid) * 8, (K - (valid + 3) / 4 * 4) / 4);
        }
        CtrsmFence<HardEvent::MTE2_V>();
        LoadDiagonal(first, count);
        PipeBarrier<PIPE_V>();
        LoadIndices(D, T);
        Gather(cr, raw, idx.ReinterpretCast<uint32_t>(), 0, T);
        Gather(ci, raw, idx.ReinterpretCast<uint32_t>(), 4, T);
        if constexpr (Conj)
            Muls(ci, ci, -1.f, T);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void NormalizeA()
    {
        if constexpr (Unit) {
            Muls(cr, cr, -1.f, T);
            PipeBarrier<PIPE_V>();
            return;
        }
        LoadIndices(D + T, T);
        Normalize(idx.ReinterpretCast<uint32_t>());
    }
    __aicore__ inline void Publish(GM_ADDR workspace, int first, int count)
    {
        constexpr int E = 2 * K, N = E * E;
        LoadIndices(D + 2 * T + 8, N);
        auto wg = CtrsmGlobal<float>((__gm__ float*)workspace);
        if (diagonalOnly) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        }
        for (int b = 0; b < count; b++) {
            const auto event = (b % 2) ? EVENT_ID3 : EVENT_ID2;
            if (diagonalOnly)
                WaitFlag<HardEvent::MTE3_V>(event);
            auto out = raw[(diagonalOnly ? b % 2 : 0) * N];
            Gather(
                out, LocalTensor<float>{TPosition::VECCALC, A * G * 4, 2 * X}, idx.ReinterpretCast<uint32_t>(), b * 4,
                N);
            PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<float>(0, 0xAAAAAAAAAAAAAAAAULL);
            Muls<float, float, false>(
                out, out, -1.f, uint64_t(0xAAAAAAAAAAAAAAAAULL), K, {1, 1, uint8_t(E / 4), uint8_t(E / 4)});
            ResetMask();
            CtrsmFence<HardEvent::V_MTE3>();
            CtrsmPublishDiagonal(wg, out, first + b, panels, order, diagonalOnly, !Unit || order % 32 == 0);
            if (diagonalOnly)
                SetFlag<HardEvent::MTE3_V>(event);
        }
        if (diagonalOnly) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        }
    }

public:
    __aicore__ inline void Run(
        GM_ADDR a, GM_ADDR workspace, int batches, int n, int aLd, int halfOnly = 0, int firstBatch = 0,
        int lastBatch = -1, int worker = -1, int workers = 0)
    {
        indices.SetGlobalBuffer((__gm__ uint32_t*)&CtrsmBatchedPlanarSquare::Indices<K, G, Upper, true, Swap>::value);
        diagonalOnly = halfOnly != 0;
        order = n;
        ap.SetGlobalBuffer((__gm__ uint64_t*)a);
        panels = Unit ? order / K : (order + K - 1) / K;
        lda = aLd;
        const int tasks = (lastBatch < 0 ? batches : lastBatch) * panels;
        if (worker < 0)
            worker = GetBlockIdx();
        if (!workers)
            workers = GetBlockNum();
        for (int first = firstBatch * panels + worker * G; first < tasks; first += workers * G) {
            const int count = tasks - first < G ? tasks - first : G;
            LoadA(first, count);
            NormalizeA();
            Identity();
            RowSweep<Upper, true>();
            Publish(workspace, first, count);
        }
        ResetMask();
        CtrsmFence<HardEvent::V_S>();
    }
};
} // namespace CtrsmBatchedGroupedInverse
namespace CtrsmBatchedInverseCommon { // Layout 0 publishes a whole inverse; 1/2 publish 32/64-panel workspaces.
template <int P, int Layout>
class DirectRhs {
    static_assert(P <= 64, "large orders use blocked inverses");
    static constexpr int L = P, W = 2 * P, AS = 2 * P * P, BS = P * W, CS = P * 8;
    LocalTensor<float> a{TPosition::VECCALC, 0, AS};
    // A and the inverse coexist. Index scratch only needs one vector row.
    LocalTensor<float> x{TPosition::VECCALC, AS * 4, BS};
    LocalTensor<int32_t> ix{TPosition::VECCALC, (AS + BS) * 4, W}, ia{TPosition::VECCALC, (AS + BS + W) * 4, CS};
    LocalTensor<float> cr{TPosition::VECCALC, (AS + BS + W + CS) * 4, CS},
        ci{TPosition::VECCALC, (AS + BS + W + CS * 2) * 4, CS}, dr{TPosition::VECCALC, (AS + BS + W + CS * 3) * 4, CS},
        di{TPosition::VECCALC, (AS + BS + W + CS * 4) * 4, CS}, den{TPosition::VECCALC, (AS + BS + W + CS * 5) * 4, CS};
    LocalTensor<int32_t> id{TPosition::VECCALC, (AS + BS + W + CS * 6) * 4, CS};
    LocalTensor<float> tmp{TPosition::VECCALC, (AS + BS + W + CS * 7) * 4, W * 3};
    LocalTensor<int32_t> sw{TPosition::VECCALC, (AS + BS + W + CS * 7 + W * 3) * 4, W};
    LocalTensor<float> sign{TPosition::VECCALC, (AS + BS + W + CS * 7 + W * 4) * 4, 8};
    static_assert((AS + BS + W + CS * 7 + W * 4 + 8) * 4 <= 192 * 1024, "inverse scratch exceeds UB");
    static constexpr int PAIR = P <= 32 ? 608 : 1344, TRI = PAIR * 8, END = AS + BS + W + CS * 7 + W * 4 + 8;
    LocalTensor<int32_t> compactIndex{TPosition::VECCALC, END * 4, P <= 48 ? PAIR : 0};
    LocalTensor<float> allCr{TPosition::VECCALC, (END + PAIR) * 4, P <= 48 ? TRI : 0},
        allCi{TPosition::VECCALC, (END + PAIR + TRI) * 4, P <= 48 ? TRI : 0};
    static_assert(P > 48 || (END + 3 * PAIR + 2 * TRI) * 4 <= 192 * 1024, "cached inverse coefficients UB");
    static constexpr int MIDPAIR = P == 64 ? 2240 : 0;
    LocalTensor<int32_t> midIndex{TPosition::VECCALC, END * 4, MIDPAIR};
    LocalTensor<float> midR{TPosition::VECCALC, (END + MIDPAIR) * 4, MIDPAIR},
        midI{TPosition::VECCALC, (END + 2 * MIDPAIR) * 4, MIDPAIR};
    static_assert(P != 64 || (END + 3 * MIDPAIR) * 4 <= 192 * 1024, "mid cached coefficients UB");
    GlobalTensor<uint64_t> ap;
    GlobalTensor<float> ag;
    int k, lda, pitch;
    bool swap, reverse, conj, unit;
    static constexpr int SHORTPAIR = 1736, SHORTTRI = SHORTPAIR * 8, SHORTBASE = AS + BS,
                         SHORTEND = SHORTBASE + 2 * SHORTTRI;
    LocalTensor<float> shortR{TPosition::VECCALC, SHORTBASE * 4, P == 64 ? SHORTTRI : 0},
        shortI{TPosition::VECCALC, (SHORTBASE + SHORTTRI) * 4, P == 64 ? SHORTTRI : 0};
    LocalTensor<int32_t> shortIndex{TPosition::VECCALC, SHORTEND * 4, P == 64 ? SHORTPAIR : 0};
    LocalTensor<float> shortSign{TPosition::VECCALC, (SHORTEND + SHORTPAIR) * 4, 8};
    static_assert(P != 64 || (SHORTEND + SHORTPAIR + 8 + W * 4) * 4 <= 192 * 1024, "order56 full coefficients UB");
    static constexpr int COMPACT_BASE = END + PAIR + 2 * TRI;
    LocalTensor<int32_t> diagIndex{TPosition::VECCALC, (COMPACT_BASE + PAIR * 2) * 4, PAIR};
    __aicore__ inline void BuildIndices()
    {
        for (int q = 0; q < 8; q++) {
            const int row = reverse ? k - 1 : 0;
            ia.SetValue(q, row * (swap ? pitch : 2) * 4);
            sw.SetValue(q, (q ^ 1) * 4);
            sign.SetValue(q, ((q & 1) ? -1.f : 1.f) * (conj ? -1.f : 1.f));
        }
        CtrsmFence<HardEvent::S_V>();
        CtrsmRepeatIndices(ia, 8, k * 8, (reverse ? -1 : 1) * (swap ? pitch : 2) * 4);
        CtrsmRepeatIndices(sw, 8, W, 32);
        Adds(id, ia, int32_t((reverse ? k - 1 : 0) * (swap ? 2 : pitch) * 4), 8);
        CtrsmRepeatIndices(id, 8, k * 8, (reverse ? -1 : 1) * (pitch + 2) * 4);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PrepareDiagonal()
    {
        Gather(dr, a, id.ReinterpretCast<uint32_t>(), 0, k * 8);
        Gather(di, a, id.ReinterpretCast<uint32_t>(), 4, k * 8);
        Mul(cr, dr, dr, k * 8);
        Mul(ci, di, di, k * 8);
        Add(den, cr, ci, k * 8);
        Div(dr, dr, den, k * 8);
        Div(di, di, den, k * 8);
        Mul(di, di, sign, 64, uint8_t((k + 7) / 8), {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DivideRow(int i)
    {
        const int active = ((2 * i + 2 + 7) / 8) * 8;
        Gather(tmp, x[i * W], sw.ReinterpretCast<uint32_t>(), 0, active);
        for (int q = 0; q < active; q += 64) {
            const int lanes = active - q < 64 ? active - q : 64;
            Mul(tmp[W + q], x[i * W + q], dr[i * 8], lanes, 1, {1, 1, 0, 8, 8, 0});
            MulAddDst<float, float>(tmp[W + q], tmp[q], di[i * 8], lanes, 1, {1, 1, 0, 8, 8, 0});
        }
        Muls(x[i * W], tmp[W], 1.f, active);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CacheIndices(LocalTensor<int32_t> destination)
    {
        CtrsmTriangleIndices<true>(destination, ix, k, (swap ? pitch : 2) * 4, (swap ? 2 : pitch) * 4, reverse, true);
    }
    __aicore__ inline void CacheCoefficients()
    {
        const int count = CtrsmTriangleSize(k);
        const bool shortCache = P == 64 && unit && k <= 56;
        const uint32_t stride = P <= 48 ? PAIR : shortCache ? SHORTPAIR : MIDPAIR;
        auto index = P <= 48 ? compactIndex : shortCache ? shortIndex : midIndex;
        const uint32_t base = P <= 48 ? COMPACT_BASE : shortCache ? SHORTBASE : END + MIDPAIR;
        LocalTensor<float> compact{TPosition::VECCALC, base * 4, 2 * stride};
        Gather(compact, a, index.ReinterpretCast<uint32_t>(), 0, count);
        Gather(compact[stride], a, index.ReinterpretCast<uint32_t>(), 4, count);
        PipeBarrier<PIPE_V>();
        if (P == 64 && !shortCache) {
            Muls(compact, compact, -1.f, count);
            PipeBarrier<PIPE_V>();
            return;
        }
        if (shortCache) {
            LocalTensor<float> scratch{TPosition::VECCALC, 0, 2 * stride};
            DataCopy(scratch, compact, count);
            DataCopy(scratch[stride], compact[stride], count);
            PipeBarrier<PIPE_V>();
            compact = scratch;
        }
        if (Layout != 0 && !unit) {
            LocalTensor<float> rr{TPosition::VECCALC, (COMPACT_BASE + PAIR * 3) * 4, PAIR},
                ii{TPosition::VECCALC, (COMPACT_BASE + PAIR * 4) * 4, PAIR},
                tr{TPosition::VECCALC, (COMPACT_BASE + PAIR * 5) * 4, PAIR};
            Gather(rr, dr, diagIndex.ReinterpretCast<uint32_t>(), 0, count);
            Gather(ii, di, diagIndex.ReinterpretCast<uint32_t>(), 0, count);
            if (conj)
                Muls(ii, ii, -1.f, count);
            PipeBarrier<PIPE_V>();
            Mul(tr, compact, rr, count);
            MulAddDst<float, float>(tr, compact[stride], ii, count);
            Mul(compact[stride], compact[stride], rr, count);
            Muls(ii, ii, -1.f, count);
            MulAddDst<float, float>(compact[stride], compact, ii, count);
            Muls(compact, tr, 1.f, count);
            PipeBarrier<PIPE_V>();
        }
        auto real = P <= 48 ? allCr : shortR, imag = P <= 48 ? allCi : shortI, signs = P <= 48 ? sign : shortSign;
        Brcb(real, compact, count / 8, {1, 8});
        Brcb(imag, compact[stride], count / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        Muls(real, real, -1.f, count * 8);
        Mul(imag, imag, signs, 64, count / 8, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Solve()
    {
        if (!unit)
            PrepareDiagonal();
        if (Layout != 0 && !unit)
            CtrsmTriangleIndices<true>(diagIndex, tmp.ReinterpretCast<int32_t>(), k, 32, 0, false, false);
        CacheCoefficients();
        if (Layout != 0 && !unit)
            NormalizeIdentity();
        int cachedAt = 0;
        for (int i = 0; i < k; i++) {
            if constexpr (Layout == 0 || P > 48) {
                if (!unit)
                    DivideRow(i);
            }
            const int rows = k - i - 1;
            if (!rows)
                break;
            const int pivot = reverse ? k - 1 - i : i;
            LocalTensor<float> coefR = cr, coefI = ci;
            if constexpr (P <= 48) {
                coefR = allCr[cachedAt];
                coefI = allCi[cachedAt];
                cachedAt += (rows + 7) / 8 * 64;
            } else if constexpr (P == 64) {
                if (unit && k <= 56) {
                    coefR = shortR[cachedAt];
                    coefI = shortI[cachedAt];
                    cachedAt += (rows + 7) / 8 * 64;
                } else {
                    const int n = (rows + 7) / 8 * 8;
                    Brcb(cr, midR[cachedAt], n / 8, {1, 8});
                    Brcb(ci, midI[cachedAt], n / 8, {1, 8});
                    PipeBarrier<PIPE_V>();
                    Mul(ci, ci, sign, 64, uint8_t((k + 7) / 8), {1, 1, 0, 8, 8, 0});
                    cachedAt += n;
                }
            }
            const int active = ((2 * i + 2 + 7) / 8) * 8;
            Gather(tmp, x[i * W], sw.ReinterpretCast<uint32_t>(), 0, active);
            for (int q = 0; q < active; q += 64) {
                const int lanes = active - q < 64 ? active - q : 64;
                MulAddDst<float, float>(x[(i + 1) * W + q], x[i * W + q], coefR, lanes, rows, {1, 1, 0, W / 8, 0, 1});
                MulAddDst<float, float>(x[(i + 1) * W + q], tmp[q], coefI, lanes, rows, {1, 1, 0, W / 8, 0, 1});
            }
            PipeBarrier<PIPE_V>();
        }
    }
    template <bool Split>
    __aicore__ inline void PadPublishIndices(LocalTensor<int32_t> idx)
    {
        constexpr int C = Split ? P : W, Q = Split ? 1 : 2;
        for (int q = 8; q < C; q += 8) {
            const int n = k * Q - q < 8 ? k * Q - q : 8;
            Duplicate(idx[q], int32_t(0), 8);
            if constexpr (!Split)
                Duplicate(idx[C + q], int32_t(0), 8);
            if (n > 0) {
                Adds(idx[q], idx, int32_t((reverse ? -q : q) * (Split ? 8 : 4)), n);
                if constexpr (!Split)
                    Adds(idx[C + q], idx[C], int32_t((reverse ? -q : q) * 4), n);
            }
        }
    }
    template <bool Split>
    __aicore__ inline void BuildPublishIndices(LocalTensor<int32_t> idx, int size)
    {
        constexpr int C = Split ? P : W, Q = Split ? 1 : 2;
        CtrsmFence<HardEvent::V_S>();
        for (int q = 0; q < 8; q++) {
            const int c = q / Q < k ? (reverse ? k - 1 - q / Q : q / Q) * 2 + q % Q : 0;
            idx.SetValue(q, c * 4);
            if constexpr (!Split)
                idx.SetValue(C + q, (c ^ 1) * 4);
        }
        CtrsmFence<HardEvent::S_V>();
        if (k == P) {
            CtrsmRepeatIndices(idx, 8, C, (reverse ? -1 : 1) * (Split ? 64 : 32));
            if constexpr (!Split)
                CtrsmRepeatIndices(idx[C], 8, C, (reverse ? -1 : 1) * 32);
        } else {
            PadPublishIndices<Split>(idx);
        }
        CtrsmRepeatIndices(idx, Q * C, size, (reverse ? -1 : 1) * W * 4);
        PipeBarrier<PIPE_V>();
    }
    template <bool Split>
    __aicore__ inline void MaskPublishedRows(LocalTensor<float> out, LocalTensor<float> im, int valid)
    {
        constexpr int C = Split ? P : W, Q = Split ? 1 : 2;
        for (int q = 0; q < C; q += 64) {
            const int lanes = C - q < 64 ? C - q : 64, used = k * Q - q;
            const uint64_t active = lanes < 64 ? (1ULL << lanes) - 1 : ~0ULL;
            if constexpr (!Split) {
                const uint64_t mask = active & 0xAAAAAAAAAAAAAAAAULL;
                AscendC::SetVectorMask<float>(0, mask);
                Muls<float, float, false>(
                    out[q], out[q], -1.f, mask, uint8_t(valid), {1, 1, uint8_t(W / 4), uint8_t(W / 4)});
            }
            const uint64_t tail = (used <= 0 ? ~0ULL : used >= 64 ? 0ULL : ~0ULL << used) & active;
            AscendC::SetVectorMask<float>(0, tail);
            Duplicate<float, false>(out[q], 0.f, tail, uint8_t(valid * Q), 1, uint8_t(C / 8));
            if constexpr (Split)
                Duplicate<float, false>(im[q], 0.f, tail, uint8_t(valid), 1, uint8_t(C / 8));
        }
        ResetMask();
    }
    template <bool Split>
    __aicore__ inline void PublishTiled(GM_ADDR workspace, int batch, int panels = 0, int order = 0)
    {
        constexpr bool Bulk = !Split && P <= 48;
        constexpr int C = Split ? P : W, ROWS = Bulk ? P * 2 : (Split ? P * 2 / 3 / 8 * 8 : P / 2), SZ = ROWS * C,
                      Q = Split ? 1 : 2, BASE = Bulk ? END + PAIR : 0;
        auto wg = CtrsmGlobal<float>((__gm__ float*)workspace + uint64_t(batch) * W * W);
        LocalTensor<float> out{TPosition::VECCALC, BASE * 4, SZ},
            im{TPosition::VECCALC, (BASE + SZ) * 4, Split ? SZ : 0};
        LocalTensor<int32_t> idx{TPosition::VECCALC, (BASE + (Split ? 2 * SZ : SZ)) * 4, SZ};
        static_assert(
            Bulk ? (BASE + 2 * SZ) * 4 <= 192 * 1024 : (Split ? 3 * SZ : 2 * SZ) <= AS, "inverse publication scratch");
        BuildPublishIndices<Split>(idx, SZ);
        for (int first = 0; first < P; first += ROWS / Q) {
            const int valid = k - first < ROWS / Q ? k - first : ROWS / Q;
            Duplicate(out, 0.f, SZ);
            if constexpr (Split)
                Duplicate(im, 0.f, SZ);
            if (valid > 0) {
                const int source = (reverse ? k - 1 - first : first) * W * 4;
                BulkPublishGather(out, x, idx, source, valid * Q * C);
                if constexpr (Split)
                    BulkPublishGather(im, x, idx, source + 4, valid * C);
                MaskPublishedRows<Split>(out, im, valid);
            }
            CtrsmFence<HardEvent::V_MTE3>();
            if constexpr (Layout < 2) {
                const int stored = (P - first < ROWS / Q ? P - first : ROWS / Q) * Q * C;
                DataCopy(wg[first * Q * C], out, stored);
                if constexpr (Split)
                    DataCopy(wg[P * P + first * P], im, stored);
            } else {
                wg.SetGlobalBuffer((__gm__ float*)workspace);
                CtrsmPublishDiagonal(wg, out, batch, panels, order, false);
            }
            CtrsmFence<HardEvent::MTE3_V>();
        }
    }
    __aicore__ inline void BulkPublishGather(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<int32_t>& idx, int base,
        int count)
    { // Apply the signed row movement in the index itself before Gather.
        Adds(idx, idx, int32_t(base), count);
        PipeBarrier<PIPE_V>();
        CtrsmGather(dst, src, idx, count);
        Adds(idx, idx, int32_t(-base), count);
        PipeBarrier<PIPE_V>();
    }

public:
    __aicore__ inline void ConfigureIndices()
    {
        BuildIndices();
        CacheIndices(P <= 48 ? compactIndex : unit && k <= 56 ? shortIndex : midIndex);
        if constexpr (P == 64) {
            if (unit && k <= 56) {
                DataCopy(shortSign, sign, 8);
                sw = LocalTensor<int32_t>{TPosition::VECCALC, (SHORTEND + SHORTPAIR + 8) * 4, W};
                tmp = LocalTensor<float>{TPosition::VECCALC, (SHORTEND + SHORTPAIR + 8 + W) * 4, W * 3};
                CtrsmSwapIndices(sw, W);
            }
        }
    }
    __aicore__ inline void Run(
        GM_ADDR aPtr, GM_ADDR workspace, int batches, int order, int aLd, int side, int upper, int trans, int diag,
        bool tailsOnly = false)
    {
        lda = aLd;
        swap = (trans != 0) != (side != 0);
        conj = trans == 2;
        unit = diag != 0;
        ap.SetGlobalBuffer((__gm__ uint64_t*)aPtr, batches);
        const int blocks = Layout == 0 ? 1 : (order + P - 1) / P;
        int cachedK = -1;
        reverse = (upper != 0) != swap;
        for (int work = GetBlockIdx(); work < (tailsOnly ? batches : batches * blocks); work += GetBlockNum()) {
            const int task = tailsOnly ? work * blocks + blocks - 1 : work, batch = task / blocks,
                      base = task % blocks * P;
            k = Layout == 0 ? order : (order - base < P ? order - base : P);
            pitch = (k * 2 + 7) / 8 * 8;
            if (k != cachedK) {
                ConfigureIndices();
                cachedK = k;
            }
            ag.SetGlobalBuffer((__gm__ float*)ap.GetValue(batch) + uint64_t(base) * (lda + 1) * 2);
            Duplicate(a, 0.f, AS);
            Duplicate(x, 0.f, BS);
            CtrsmFence<HardEvent::V_S>();
            for (int row = 0; row < k; row++)
                x.SetValue(row * W + 2 * row, 1.f);
            CtrsmFence<HardEvent::S_MTE2>();
            CtrsmRead(a, ag, k, k * 8, (lda - k) * 8);
            CtrsmFence<HardEvent::MTE2_V>();
            Solve();
            if constexpr (Layout == 0) {
                if (side)
                    PublishTiled<true>(workspace, batch);
                else
                    PublishTiled<false>(workspace, batch);
            } else
                PublishTiled<false>(workspace, task, blocks, order);
            CtrsmFence<HardEvent::V_MTE2>();
        }
        ResetMask();
        CtrsmFence<HardEvent::V_S>();
    }

private:
    __aicore__ inline void NormalizeIdentity()
    {
        LocalTensor<float> swapped{TPosition::VECCALC, COMPACT_BASE * 4, BS};
        LocalTensor<int32_t> indices{TPosition::VECCALC, (COMPACT_BASE + BS) * 4, BS};
        CtrsmSwapIndices(indices, W);
        CtrsmRepeatIndices(indices, W, BS, W * 4);
        PipeBarrier<PIPE_V>();
        Gather(swapped, x, indices.ReinterpretCast<uint32_t>(), 0, BS);
        PipeBarrier<PIPE_V>();
        Mul(x, x, dr, 64, k, {1, 1, 0, W / 8, W / 8, 1});
        MulAddDst<float, float>(x, swapped, di, 64, k, {1, 1, 0, W / 8, W / 8, 1});
        PipeBarrier<PIPE_V>();
    }
};
template <int NR, int LC = 128, int KC = 128, int P = 64, int StorageP = P, int CACHEK = 0, bool Prefetch = false>
class BlockCubeSolver {
    static constexpr int D = 2 * P, SD = 2 * StorageP, AS = NR * (CACHEK ? CACHEK : LC),
                         BS = D * (Prefetch ? CACHEK : LC), CS = NR * D;
    static constexpr int H = CACHEK - D;
    static_assert(!Prefetch || (AS + 2 * BS) * 4 <= 512 * 1024, "prefetch L1");
    LocalTensor<float> al1{TPosition::A1, 0, AS}, bc{TPosition::B1, AS * 4, Prefetch ? 2 * BS : BS},
        al0{TPosition::A2, 0, NR* KC}, bl0{TPosition::B2, 0, D* KC}, cl0{TPosition::CO1, 0, CS};

    LocalTensor<float> bl1;
    GlobalTensor<uint64_t> bp;
    GlobalTensor<float> wg, bg;
    GM_ADDR workspace;
    int batches, k, rhs, ldb, blocks, kp, dim, rp, tiles, batch, tile, count, M, stride, step, panel, start, valid, N,
        K, historyStart, history;
    bool right, reverse;
    uint64_t invSize, coeffSize;
    Nd2NzParams np;
    LoadData2DParams lb;
    decltype(CtrsmCube::Load3d(1, 1, 1, 1)) la;
    decltype(CtrsmCube::Mmad(1, 1, 1)) mm;
    decltype(CtrsmCube::Fix(1, 1, 1, 1)) fp;
    __aicore__ inline void PrefetchPanel(int next)
    {
        if (next >= blocks)
            return;
        const int panel = reverse ? blocks - 1 - next : next, start = panel * P, valid = k - start < P ? k - start : P,
                  N = (valid * 2 + 15) / 16 * 16;
        const int hs = reverse ? start + valid : 0, history = reverse ? k - start - valid : start, slot = next % 2;
        if (next >= 2)
            WaitFlag<HardEvent::MTE1_MTE2>(slot + 2);
        auto np = CtrsmCube::Nd(N, history * 2, dim, N);
        if (history)
            DataCopy(bc[slot * BS], wg[invSize + uint64_t(batch) * dim * dim + uint64_t(start * 2) * dim + hs * 2], np);
        np.dValue = D;
        np.srcDValue = SD;
        DataCopy(
            bc[slot * BS + H * D],
            wg[(uint64_t(batch) * ((k + StorageP - 1) / StorageP) + panel / (StorageP / P)) * SD * SD +
               (panel % (StorageP / P)) * D * (SD + 1)],
            np);
        SetFlag<HardEvent::MTE2_MTE1>(slot + 2);
    }
    __aicore__ inline void AccumulateHistory(int off, int kk)
    {
        for (int sub = 0; sub < kk; sub += KC) {
            const int ka = ((kk - sub < KC ? kk - sub : KC) + 7) / 8 * 8;
            la.kStartPt = (CACHEK ? historyStart * 2 + off : 0) + sub;
            la.kExtension = ka;
            LoadData(al0, al1, la);
            lb.repeatTimes = ka * N / 128;
            LoadData(bl0, bl1[sub * N], lb);
            CtrsmFence<HardEvent::MTE1_M>();
            mm.k = ka;
            mm.cmatrixInitVal = off == 0 && sub == 0;
            Mmad(cl0, al0, bl0, mm);
            CtrsmFence<HardEvent::M_MTE1>();
        }
    }
    __aicore__ inline void UpdateHistory()
    {
        for (int off = 0; off < history * 2; off += Prefetch ? CACHEK : LC) {
            const int kk = Prefetch ? history * 2 : (history * 2 - off < LC ? history * 2 - off : LC);
            np = CtrsmCube::Nd(count, kk, stride, NR);
            if constexpr (CACHEK == 0)
                DataCopy(al1, bg[historyStart * 2 + off], np);
            np = CtrsmCube::Nd(N, kk, dim, N);
            if constexpr (!Prefetch) {
                DataCopy(
                    bl1, wg[invSize + uint64_t(batch) * dim * dim + uint64_t(start * 2) * dim + historyStart * 2 + off],
                    np);
                CtrsmFence<HardEvent::MTE2_MTE1>();
            }
            AccumulateHistory(off, kk);
            if constexpr (!Prefetch)
                CtrsmFence<HardEvent::MTE1_MTE2>();
        }
        CtrsmFence<HardEvent::M_FIX>();
        SetAtomicAdd<float>();
        Fixpipe(bg[start * 2], cl0, fp);
        CtrsmFence<HardEvent::FIX_MTE2>();
        DisableDmaAtomic();
        CtrsmFence<HardEvent::FIX_M>();
    }
    __aicore__ inline void SolveDiagonal()
    {
        np = CtrsmCube::Nd(count, valid * 2, stride, NR);
        if constexpr (CACHEK == 0)
            DataCopy(al1, bg[start * 2], np);
        else if (history)
            DataCopy(al1[start * 2 * NR], bg[start * 2], np);
        np = CtrsmCube::Nd(N, D, SD, N);
        const uint64_t diagonal =
            (uint64_t(batch) * ((k + StorageP - 1) / StorageP) + panel / (StorageP / P)) * SD * SD +
            (panel % (StorageP / P)) * D * (SD + 1);
        if constexpr (!Prefetch)
            DataCopy(bl1, wg[diagonal], np);
        if (!Prefetch || history)
            CtrsmFence<HardEvent::MTE2_MTE1>();
        for (int sub = 0; sub < K; sub += KC) {
            const int ka = K - sub < KC ? K - sub : KC;
            la.kStartPt = (CACHEK ? start * 2 : 0) + sub;
            la.kExtension = ka;
            LoadData(al0, al1, la);
            lb.repeatTimes = ka * N / 128;
            LoadData(bl0, bl1[(Prefetch ? H * D : 0) + sub * N], lb);
            CtrsmFence<HardEvent::MTE1_M>();
            if constexpr (Prefetch)
                SetFlag<HardEvent::MTE1_MTE2>(step % 2 + 2);
            mm.k = ka;
            mm.cmatrixInitVal = sub == 0;
            Mmad(cl0, al0, bl0, mm);
            if constexpr (StorageP != 32 && !Prefetch)
                CtrsmFence<HardEvent::M_MTE1>();
        }
        CtrsmFence<HardEvent::M_FIX>();
        Fixpipe(bg[start * 2], cl0, fp);
        CtrsmFence<HardEvent::FIX_MTE2>();
        CtrsmFence<HardEvent::M_MTE1>();
        if constexpr (!Prefetch)
            CtrsmFence<HardEvent::MTE1_MTE2>();
        CtrsmFence<HardEvent::FIX_M>();
    }
    __aicore__ inline void SolvePanel(int current)
    {
        step = current;
        if constexpr (Prefetch) {
            const int loads = step == 0 ? 2 : 1;
            for (int q = 0; q < loads; q++)
                PrefetchPanel(step == 0 ? q : step + 1);
            WaitFlag<HardEvent::MTE2_MTE1>(step % 2 + 2);
        }
        bl1 = bc[Prefetch ? step % 2 * BS : 0];
        panel = reverse ? blocks - 1 - step : step, start = panel * P, valid = k - start < P ? k - start : P,
        N = (valid * 2 + 15) / 16 * 16, K = (valid * 2 + 7) / 8 * 8;
        historyStart = reverse ? start + valid : 0, history = reverse ? k - start - valid : start;
        np = Nd2NzParams();
        np.ndNum = 1;
        np.dstNzNStride = 1;
        la = CtrsmCube::Load3d(NR, CACHEK ? CACHEK : LC, KC, M, false, 0, 0);

        lb.srcStride = 1;
        mm = CtrsmCube::Mmad(StorageP == 32 ? count : M, N, KC, true);
        fp = CtrsmCube::Fix(count, valid * 2, M, stride);
        if (history)
            UpdateHistory();
        SolveDiagonal();
        if constexpr (CACHEK != 0) {
            if (step + 1 < blocks) {
                np = CtrsmCube::Nd(count, valid * 2, stride, NR);
                DataCopy(al1[start * 2 * NR], bg[start * 2], np);
                CtrsmFence<HardEvent::MTE2_MTE1>();
            }
        }
    }
    __aicore__ inline void SolveTile(int task)
    {
        batch = task / tiles, tile = task % tiles, count = rhs - tile * NR < NR ? rhs - tile * NR : NR,
        M = (count + 15) / 16 * 16;
        stride = right ? dim : ldb * 2;
        if (right)
            bg.SetGlobalBuffer(
                (__gm__ float*)workspace + invSize + coeffSize + uint64_t(batch) * rp * dim +
                uint64_t(tile) * NR * dim);
        else
            bg.SetGlobalBuffer((__gm__ float*)bp.GetValue(batch) + uint64_t(tile) * NR * ldb * 2);
        if constexpr (CACHEK != 0) {
            auto cached = CtrsmCube::Nd(count, k * 2, stride, NR);
            DataCopy(al1, bg, cached);
            CtrsmFence<HardEvent::MTE2_MTE1>();
        }

        for (int current = 0; current < blocks; current++)
            SolvePanel(current);
        if constexpr (Prefetch)
            for (int slot = 0; slot < 2; slot++)
                if (blocks > slot)
                    WaitFlag<HardEvent::MTE1_MTE2>(slot + 2);
    }

public:
    __aicore__ inline void Run(
        GM_ADDR bPtr, GM_ADDR workspacePtr, int batchCount, int order, int columns, int leading, int side, int upper,
        int trans, int firstBatch, int lastBatch, int worker, int workers)
    {
        batches = batchCount;
        k = order;
        rhs = columns;
        ldb = leading;
        workspace = workspacePtr;
        blocks = (k + P - 1) / P, kp = (k + StorageP - 1) / StorageP * StorageP, dim = kp * 2,
        rp = (rhs + 63) / 64 * 64, tiles = (rhs + NR - 1) / NR;
        right = side != 0, reverse = (upper != 0) != ((trans != 0) != right);
        invSize = uint64_t(batches) * ((k + StorageP - 1) / StorageP) * SD * SD,
        coeffSize = uint64_t(batches) * dim * dim;
        bp = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)bPtr);
        wg = CtrsmGlobal<float>((__gm__ float*)workspacePtr);

        InitSocState();
        SetHF32Mode(false);
        if (worker < 0)
            worker = GetBlockIdx();
        if (!workers)
            workers = GetBlockNum();
        const int end = (lastBatch < 0 ? batches : lastBatch) * tiles;
        for (int task = firstBatch * tiles + worker; task < end; task += workers)
            SolveTile(task);
        DisableDmaAtomic();
        CtrsmFence<HardEvent::FIX_S>();
    }
};
template <int NR, int LC = 128, int KC = 128, int P = 64, int StorageP = P, int CACHEK = 0, bool Prefetch = false>
__aicore__ inline void BlockCubeSweep(
    GM_ADDR bPtr, GM_ADDR workspace, int batches, int k, int rhs, int ldb, int side, int upper, int trans,
    int firstBatch = 0, int lastBatch = -1, int worker = -1, int workers = 0)
{
    BlockCubeSolver<NR, LC, KC, P, StorageP, CACHEK, Prefetch> solver;
    solver.Run(bPtr, workspace, batches, k, rhs, ldb, side, upper, trans, firstBatch, lastBatch, worker, workers);
}
template <int NR, int LC = 128, int KC = 128, int P = 64, int StorageP = P, int CACHEK = 0, bool Prefetch = false>
__global__ __aicore__ void block_cube_sweep(
    GM_ADDR b, GM_ADDR w, int batches, int k, int rhs, int ldb, int side, int upper, int trans)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    BlockCubeSweep<NR, LC, KC, P, StorageP, CACHEK, Prefetch>(b, w, batches, k, rhs, ldb, side, upper, trans);
}
template <int RR, bool Pack>
__aicore__ inline void TransposeBlockRhs(GM_ADDR bPtr, GM_ADDR workspace, int batches, int k, int rhs, int ldb)
{
    constexpr int P = 64, D = 128, ROWS = Pack ? P : RR, COLS = Pack ? RR : P;
    constexpr int PITCH = COLS * 2 + 8, RAW = ROWS * PITCH, S = P * RR * 2, IX = 8192;
    LocalTensor<float> raw{TPosition::VECCALC, 0, RAW}, out{TPosition::VECCALC, RAW * 4, S};
    LocalTensor<int32_t> idx{TPosition::VECCALC, (RAW + S) * 4, IX};
    static_assert((RAW + S + IX) * 4 <= 192 * 1024, "block RHS transpose UB");
    const int blocks = (k + P - 1) / P, dim = blocks * D, rp = (rhs + 63) / 64 * 64, rt = (rhs + RR - 1) / RR;
    const uint64_t base = uint64_t(batches) * (blocks * D * D + dim * dim);
    auto bp = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)bPtr);
    auto wg = CtrsmGlobal<float>((__gm__ float*)workspace);
    CtrsmIndexTile<2>(idx, ROWS * 2, 4, PITCH * 4);
    CtrsmRepeatIndices(idx, ROWS * 2, IX, 8);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for (int task = GetBlockIdx(); task < batches * blocks * rt; task += GetBlockNum()) {
        const int batch = task / (blocks * rt), kt = task / rt % blocks, rr = task % rt,
                  rows = k - kt * P < P ? k - kt * P : P, cols = rhs - rr * RR < RR ? rhs - rr * RR : RR;
        const uint64_t at = base + uint64_t(batch) * rp * dim + uint64_t(rr * RR) * dim + kt * D;
        auto bg = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(batch) + uint64_t(kt * P) * ldb * 2 + rr * RR * 2);
        if (Pack && (rows < P || cols < RR))
            Duplicate(raw, 0.f, RAW);
        CtrsmFence<HardEvent::V_MTE2>();
        if constexpr (Pack)
            CtrsmRead(raw, bg, rows, cols * 8, (ldb - cols) * 8, (PITCH - (cols * 2 + 7) / 8 * 8) / 8);
        else
            CtrsmRead(raw, wg[at], RR, D * 4, (dim - D) * 4, (PITCH - D) / 8);
        CtrsmFence<HardEvent::MTE2_V>();
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        Gather(out, raw, idx.ReinterpretCast<uint32_t>(), 0, IX);
        if constexpr (RR == 128)
            Gather(out[IX], raw, idx.ReinterpretCast<uint32_t>(), IX / (ROWS * 2) * 8, IX);
        CtrsmFence<HardEvent::V_MTE3>();
        if constexpr (Pack) {
            const int nr = rp - rr * RR < RR ? rp - rr * RR : RR;
            CtrsmWrite(wg[at], out, nr, D * 4, 0, (dim - D) * 4);
        } else
            CtrsmWrite(bg, out, rows, cols * 8, (RR * 2 - (cols * 2 + 7) / 8 * 8) / 8, (ldb - cols) * 8);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
}
template <int RR>
__global__ __aicore__ void restore_block_rhs(GM_ADDR bPtr, GM_ADDR workspace, int batches, int k, int rhs, int ldb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TransposeBlockRhs<RR, false>(bPtr, workspace, batches, k, rhs, ldb);
    ResetMask();
    CtrsmFence<HardEvent::V_S>();
}
} // namespace CtrsmBatchedInverseCommon
namespace CtrsmBatchedInverseCube {
template <int P, int Layout = 0>
__global__ __aicore__ void prepare_triangular_inverse(
    GM_ADDR a, GM_ADDR w, int batches, int k, int lda, int side, int upper, int trans, int diag)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrsmBatchedInverseCommon::DirectRhs<P, Layout> kernel;
    kernel.Run(a, w, batches, k, lda, side, upper, trans, diag);
}
template <int P>
constexpr int CubeRhs = P <= 32 ? 256 : (P <= 48 ? 160 : 128);
template <int P>
class LeftCubePipeline {
    static constexpr int D = 2 * P, NR = P == 48 ? 128 : CubeRhs<P>, SZ = NR * D;
    static_assert(NR * D * 4 <= 64 * 1024, "pipelined L0A");
    LocalTensor<float> al1{TPosition::A1, 0, 2 * SZ}, bl1{TPosition::B1, 2 * SZ * 4, D* D}, al0{TPosition::A2, 0, SZ},
        bl0{TPosition::B2, 0, D* D}, cl0{TPosition::CO1, 0, SZ * 2};
    static_assert(SZ * 8 <= 128 * 1024, "double output L0C capacity");

    GlobalTensor<uint64_t> bp;
    GlobalTensor<float> wg;
    int k, rhs, ldb, tiles, K, N, loadedBatch = -1;
    __aicore__ inline void LoadRhs(int task, int slot)
    {
        auto src = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(task / tiles) + uint64_t(task % tiles) * NR * ldb * 2);
        auto np = CtrsmCube::Nd(rhs - task % tiles * NR < NR ? rhs - task % tiles * NR : NR, k * 2, ldb * 2, NR);
        DataCopy(al1[slot * SZ], src, np);
        SetFlag<HardEvent::MTE2_MTE1>(slot);
    }
    __aicore__ inline void ApplyTile(int task, int start, int end)
    {
        const int slot = (task - start) & 1, batch = task / tiles, tile = task % tiles,
                  count = rhs - tile * NR < NR ? rhs - tile * NR : NR;
        WaitFlag<HardEvent::MTE2_MTE1>(slot);
        const bool fresh = batch != loadedBatch;
        if (fresh) {
            auto np = CtrsmCube::Nd(N, D, D, N);
            DataCopy(bl1, wg[uint64_t(batch) * D * D], np);
            CtrsmFence<HardEvent::MTE2_MTE1>(2);
        }
        if (task + 1 < end) {
            LoadRhs(task + 1, 1 - slot);
        }
        if (task > start)
            WaitFlag<HardEvent::M_MTE1>(0);
        auto la = CtrsmCube::Load3d(NR, D, K, (count + 15) / 16 * 16, false, 0, 0);
        LoadData(al0, al1[slot * SZ], la);
        if (fresh) {
            LoadData2DParams lb;
            lb.repeatTimes = K * N / 128;
            lb.srcStride = 1;
            LoadData(bl0, bl1, lb);
        } // Finish L1 reads before the next iteration can refill either input slot.
        CtrsmFence<HardEvent::MTE1_MTE2>();
        CtrsmFence<HardEvent::MTE1_M>();
        if (task >= start + 2)
            WaitFlag<HardEvent::FIX_M>(slot);
        auto mm = CtrsmCube::Mmad((count + 15) / 16 * 16, N, K, true);
        Mmad(cl0[slot * SZ], al0, bl0, mm);
        SetFlag<HardEvent::M_MTE1>(0);
        CtrsmFence<HardEvent::M_FIX>();
        auto dst = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(batch) + uint64_t(tile) * NR * ldb * 2);
        auto fp = CtrsmCube::Fix(count, k * 2, (count + 15) / 16 * 16, ldb * 2);
        Fixpipe(dst, cl0[slot * SZ], fp);
        SetFlag<HardEvent::FIX_M>(slot);
        loadedBatch = batch;
    }

public:
    __aicore__ inline void Run(GM_ADDR bPtr, GM_ADDR workspace, int batches, int order, int columns, int leading)
    {
        k = order;
        rhs = columns;
        ldb = leading;
        bp = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)bPtr, batches);
        wg = CtrsmGlobal<float>((__gm__ float*)workspace);
        tiles = (rhs + NR - 1) / NR, K = (k * 2 + 7) / 8 * 8, N = (k * 2 + 15) / 16 * 16;
        const int group = (batches * tiles + GetBlockNum() - 1) / GetBlockNum(), start = GetBlockIdx() * group,
                  end = start + group < batches * tiles ? start + group : batches * tiles;
        InitSocState();
        SetHF32Mode(false);

        if (start < end)
            LoadRhs(start, 0);
        for (int task = start; task < end; task++)
            ApplyTile(task, start, end);

        if (start < end) {
            WaitFlag<HardEvent::FIX_M>(0);
            if (end - start > 1)
                WaitFlag<HardEvent::FIX_M>(1);
            WaitFlag<HardEvent::M_MTE1>(0);
        }
        CtrsmFence<HardEvent::FIX_S>();
    }
};
template <int P>
__global__ __aicore__ void apply_inverse_pipelined_left(
    GM_ADDR bPtr, GM_ADDR workspace, int batches, int k, int rhs, int ldb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    LeftCubePipeline<P> pipeline;
    pipeline.Run(bPtr, workspace, batches, k, rhs, ldb);
}
template <int P, bool Packed>
__aicore__ inline void LoadRightInputs(
    LocalTensor<float> al1, LocalTensor<float> bl1, GlobalTensor<float> bg, GlobalTensor<float> wg, int k, int count,
    int ldb, int batch, int task, uint64_t base)
{
    constexpr int D = 2 * P, N = 2 * CubeRhs<P>, SZ = P * N, M = Packed ? P : D;
    const int K = (k + 7) / 8 * 8;
    auto np = CtrsmCube::Nd(k, count * 2, ldb * 2, P);
    // ND2NZ pads D only. The aligned reduction rows must also be finite:
    // an uninitialized NaN times a zero inverse coefficient remains NaN.
    if (K != k) {
        const int columns = (count * 2 + 7) / 8;
        Fill(bl1[k * 8], {uint16_t(columns), uint16_t(K - k), uint16_t(P - K + k), 0.f});
    }
    DataCopy(bl1, bg, np);
    if constexpr (Packed) {
        np.srcDValue = N;
        if (K != k) {
            const int columns = (count * 2 + 7) / 8;
            Fill(bl1[SZ + k * 8], {uint16_t(columns), uint16_t(K - k), uint16_t(P - K + k), 0.f});
        }
        DataCopy(bl1[SZ], wg[base + uint64_t(task) * 2 * SZ], np);
    }
    np = CtrsmCube::Nd(M, P, P, M);
    DataCopy(al1, wg[uint64_t(batch) * D * D], np);
    if constexpr (Packed)
        DataCopy(al1[P * P], wg[uint64_t(batch) * D * D + P * P], np);
    CtrsmFence<HardEvent::MTE2_MTE1>();
}
template <int P, bool Packed>
__aicore__ inline void ComputeRightTile(
    LocalTensor<float> al1, LocalTensor<float> bl1, LocalTensor<float> al0, LocalTensor<float> bl0,
    LocalTensor<float> cl0, int K, int count)
{
    constexpr int D = 2 * P, N = 2 * CubeRhs<P>, SZ = P * N, KC = 32, M = Packed ? P : D;
    for (int kp = 0; kp < K; kp += KC) {
        const int kk = K - kp < KC ? K - kp : KC;
        for (int part = 0; part < (Packed ? 2 : 1); part++) {
            auto lb = CtrsmCube::Load3d(P, N, (count * 2 + 15) / 16 * 16, (kk + 15) / 16 * 16, true, 0, kp);
            LoadData(bl0, bl1[part * SZ], lb);
            LoadData2DParams la;
            la.repeatTimes = kk / 8;
            la.srcStride = M / 16;
            for (int row = 0; row < M / 16; row++)
                LoadData(al0[row * 16 * kk], al1[part * P * P + kp * M + row * 128], la);
            CtrsmFence<HardEvent::MTE1_M>();
            auto mm = CtrsmCube::Mmad(M, count * 2, kk, kp == 0 && part == 0);
            Mmad(cl0, al0, bl0, mm);
            CtrsmFence<HardEvent::M_MTE1>();
        }
    }
}
template <int P, bool Packed>
__aicore__ inline void ApplyInverseRightGeneric(GM_ADDR bPtr, GM_ADDR workspace, int batches, int k, int rhs, int ldb)
{
    constexpr int D = 2 * P, NR = CubeRhs<P>, N = 2 * NR, SZ = P * N, KC = 32, M = Packed ? P : D;
    LocalTensor<float> al1(TPosition::A1, 0, 2 * P * P), bl1(TPosition::B1, 2 * P * P * 4, Packed ? 2 * SZ : SZ);
    LocalTensor<float> al0(TPosition::A2, 0, M * KC), bl0(TPosition::B2, 0, KC * N),
        cl0(TPosition::CO1, 0, Packed ? SZ : 2 * SZ);
    static_assert(SZ * 4 <= 128 * 1024, "complex L0C capacity");
    auto bp = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)bPtr, batches);
    auto wg = CtrsmGlobal<float>((__gm__ float*)workspace);
    const int tiles = (rhs + NR - 1) / NR, K = (k + 7) / 8 * 8;
    const uint64_t base = uint64_t(batches) * D * D;
    InitSocState();
    SetHF32Mode(false);
    int issued = 0;
    for (int task = GetBlockIdx(); task < batches * tiles; task += GetBlockNum(), issued++) {
        if constexpr (!Packed)
            if (issued >= 8)
                CrossCoreWaitFlag<2>(1);
        const int batch = task / tiles, tile = task % tiles, count = rhs - tile * NR < NR ? rhs - tile * NR : NR;
        auto bg = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(batch) + uint64_t(tile) * NR * 2);
        LoadRightInputs<P, Packed>(al1, bl1, bg, wg, k, count, ldb, batch, task, base);
        ComputeRightTile<P, Packed>(al1, bl1, al0, bl0, cl0, K, count);
        CtrsmFence<HardEvent::M_FIX>();
        auto fp = CtrsmCube::Fix(Packed ? k : D, count * 2, M, Packed ? ldb * 2 : N);
        if constexpr (Packed)
            Fixpipe(bg, cl0, fp);
        else
            Fixpipe(wg[base + uint64_t(task) * 2 * SZ], cl0, fp);
        if constexpr (!Packed)
            CrossCoreSetFlag<2, PIPE_FIX>(0);
        CtrsmFence<HardEvent::FIX_M>();
        CtrsmFence<HardEvent::MTE1_MTE2>();
    }
    if constexpr (!Packed)
        for (int done = 0; done < issued && done < 8; done++)
            CrossCoreWaitFlag<2>(1);
}
template <int P, bool Merge>
__aicore__ inline void TransformRightTile(
    LocalTensor<float> raw, LocalTensor<float> rotated, LocalTensor<int32_t> ix, GlobalTensor<float> bg,
    GlobalTensor<float> wg, uint64_t offset, int rows, int count, int ldb)
{
    constexpr int N = 2 * CubeRhs<P>, SZ = P * N;
    CtrsmFence<HardEvent::V_MTE2>();
    if constexpr (Merge)
        DataCopy(raw, wg[offset + SZ], rows * N);
    else
        CtrsmRead(raw, bg, rows, count * 8, (ldb - count) * 8, (N - (count * 2 + 7) / 8 * 8) / 8);
    CtrsmFence<HardEvent::MTE2_V>();
    CtrsmGather(rotated, raw, ix, rows * N);
    if constexpr (Merge) {
        CtrsmFence<HardEvent::V_MTE2>();
        DataCopy(raw, wg[offset], rows * N);
    }
    AscendC::SetVectorMask<float>(0, 0x5555555555555555ULL);
    for (int at = 0; at < rows * N; at += 8192) {
        const int count = rows * N - at < 8192 ? rows * N - at : 8192;
        Muls<float, float, false>(
            rotated[at], rotated[at], -1.f, 0x5555555555555555ULL, uint8_t((count + 63) / 64), {1, 1, 8, 8});
    }
    ResetMask();
    if constexpr (Merge) {
        CtrsmFence<HardEvent::MTE2_V>();
        Add(rotated, raw, rotated, rows * N);
    }
    CtrsmFence<HardEvent::V_MTE3>();
    if constexpr (Merge)
        CtrsmWrite(bg, rotated, rows, count * 8, (N - (count * 2 + 7) / 8 * 8) / 8, (ldb - count) * 8);
    else
        DataCopy(wg[offset], rotated, rows * N);
    CtrsmFence<HardEvent::MTE3_V>();
}
template <int P, bool Merge>
__aicore__ inline void TransformRightRhs(GM_ADDR bPtr, GM_ADDR workspace, int batches, int k, int rhs, int ldb)
{
    constexpr int NR = CubeRhs<P>, N = NR * 2, SZ = P * N, ROWS = Merge ? P / 2 : P, Z = ROWS * N;
    LocalTensor<float> raw(TPosition::VECCALC, 0, Z), rotated(TPosition::VECCALC, Z * 4, Z);
    LocalTensor<int32_t> ix(TPosition::VECCALC, Z * 8, Z);
    static_assert(Z * 12 <= 192 * 1024, "right transform UB");
    const int first = Merge ? GetSubBlockIdx() * ROWS : 0, rows = k - first < ROWS ? k - first : ROWS,
              tiles = (rhs + NR - 1) / NR;
    const bool reserve = !Merge && batches < GetBlockNum() / 2;
    const int worker = Merge ? GetBlockIdx() / 2 : GetBlockIdx() - (reserve ? batches : 0),
              workers = GetBlockNum() - (reserve ? batches : 0);
    if (!Merge && worker < 0)
        return;
    auto bp = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)bPtr);
    auto wg = CtrsmGlobal<float>((__gm__ float*)workspace);
    CtrsmSwapIndices(ix, Z);
    for (int task = worker; task < batches * tiles; task += workers) {
        if constexpr (Merge)
            CrossCoreWaitFlag<2>(0);
        if (rows > 0) {
            const int tile = task % tiles, count = rhs - tile * NR < NR ? rhs - tile * NR : NR;
            const uint64_t offset = uint64_t(batches) * 4 * P * P + uint64_t(task) * 2 * SZ + first * N;
            auto bg = CtrsmGlobal<float>(
                (__gm__ float*)bp.GetValue(task / tiles) + uint64_t(first) * ldb * 2 + tile * NR * 2);
            TransformRightTile<P, Merge>(raw, rotated, ix, bg, wg, offset, rows, count, ldb);
        }
        if constexpr (Merge)
            CrossCoreSetFlag<2, PIPE_MTE3>(1);
    }
}
template <int P>
__global__ __aicore__ void apply_inverse_right(GM_ADDR b, GM_ADDR w, int batches, int k, int rhs, int ldb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        ApplyInverseRightGeneric<P, false>(b, w, batches, k, rhs, ldb);
    }
    if ASCEND_IS_AIV {
        TransformRightRhs<P, true>(b, w, batches, k, rhs, ldb);
        ResetMask();
    }
    if ASCEND_IS_AIV {
        CtrsmFence<HardEvent::V_S>();
    } else {
        CtrsmFence<HardEvent::FIX_S>();
    }
}
template <int P>
__global__ __aicore__ void prepare_right_packed(
    GM_ADDR a, GM_ADDR b, GM_ADDR w, int batches, int k, int rhs, int lda, int ldb, int upper, int trans, int diag)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrsmBatchedInverseCommon::DirectRhs<P, 0> inverse;
    inverse.Run(a, w, batches, k, lda, 1, upper, trans, diag);
    TransformRightRhs<P, false>(b, w, batches, k, rhs, ldb);
    CtrsmFence<HardEvent::V_S>();
}
template <int P>
__global__ __aicore__ void apply_inverse_right_packed(GM_ADDR b, GM_ADDR w, int batches, int k, int rhs, int ldb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    ApplyInverseRightGeneric<P, true>(b, w, batches, k, rhs, ldb);
    CtrsmFence<HardEvent::FIX_S>();
}
template <int P>
void LaunchInverseCube(
    void* a, void* b, void* w, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans, int diag,
    int cores, void* stream)
{
    if (side && batches <= cores) {
        prepare_right_packed<P><<<cores * 2, nullptr, stream>>>(
            (GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, lda, ldb, upper, trans, diag);
        apply_inverse_right_packed<P><<<cores, nullptr, stream>>>((GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, ldb);
    } else {
        prepare_triangular_inverse<P>
            <<<cores * 2, nullptr, stream>>>((GM_ADDR)a, (GM_ADDR)w, batches, k, lda, side, upper, trans, diag);
        if (side)
            apply_inverse_right<P><<<cores, nullptr, stream>>>((GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, ldb);
        else
            apply_inverse_pipelined_left<P><<<cores, nullptr, stream>>>((GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, ldb);
    }
}
} // namespace CtrsmBatchedInverseCube
struct CtrsmBlockTasks {
    int blocks, per, full, total, batch = -1, row = 0, base = 0;
    bool reverse, mixed;
    __aicore__ CtrsmBlockTasks(int n, int batches, bool upper, bool includeDiagonal, bool appendDiagonal)
        : blocks(n),
          per(n * (n + (includeDiagonal ? 1 : -1)) / 2),
          full(batches * per),
          total(full + (appendDiagonal && !includeDiagonal ? batches * n : 0)),
          reverse(upper),
          mixed(includeDiagonal)
    {}
    __aicore__ inline void Locate(int task, int& b, int& out, int& in)
    {
        if (task >= full) {
            b = (task - full) / blocks;
            out = in = (task - full) % blocks;
            return;
        }
        b = task / per;
        const int local = task % per;
        if (b != batch) {
            batch = b;
            row = mixed || reverse ? 0 : 1;
            base = 0;
        }
        int count = reverse ? blocks - row - (mixed ? 0 : 1) : row + (mixed ? 1 : 0);
        while (local >= base + count) {
            base += count;
            row++;
            count = reverse ? blocks - row - (mixed ? 0 : 1) : row + (mixed ? 1 : 0);
        }
        out = row;
        in = (reverse ? row + (mixed ? 0 : 1) : 0) + local - base;
    }
};
namespace CtrsmBatchedInverseCommon {
template <int RR, bool Prefetch = false>
class BlockPacker {
    static constexpr int P = 64, D = 128, RPITCH = D + 8, SZ = D * D, RAW = P * RPITCH;
    static constexpr int INPUTS = Prefetch ? 2 : 1, OUTPUTS = Prefetch ? 2 : 1, OUT = Prefetch ? 8192 : SZ,
                         IX = Prefetch ? 8192 : SZ, OUTBASE = INPUTS * RAW, IDXBASE = OUTBASE + OUTPUTS * OUT,
                         DIAGBASE = IDXBASE + IX + (Prefetch ? 0 : 32),
                         SIGNBASE = Prefetch ? DIAGBASE + 4096 : IDXBASE + IX;
    LocalTensor<float> raw{TPosition::VECCALC, 0, INPUTS* RAW}, out{TPosition::VECCALC, OUTBASE * 4, OUTPUTS* OUT},
        signs{TPosition::VECCALC, SIGNBASE * 4, 16};
    LocalTensor<int32_t> idx{TPosition::VECCALC, IDXBASE * 4, IX}, diagIdx{TPosition::VECCALC, DIAGBASE * 4, 4096};
    static_assert((DIAGBASE + 4096 + (Prefetch ? 16 : 0)) * 4 <= 192 * 1024, "block pack UB");

    bool right, swap, reverse, conj, trim;
    int batches, k, rhs, lda, ldb, blocks, kp, dim, rp, outputs = 0;
    int sb[INPUTS], so[INPUTS], si[INPUTS], srn[INPUTS], scn[INPUTS];
    uint64_t invSize, coeffSize;
    GlobalTensor<uint64_t> ap, bp;
    GlobalTensor<float> wg;
    __aicore__ inline void BuildIndices()
    {
        for (int q = 0; q < 8; q++) {
            idx.SetValue(q, ((q / 2) * (swap ? 2 : RPITCH) + q % 2) * 4);
            idx.SetValue(D + q, ((q / 2) * (swap ? 2 : RPITCH) + (q % 2 ^ 1)) * 4);
            signs.SetValue(q, q % 2 ? (conj ? -1.f : 1.f) : -1.f);
            signs.SetValue(8 + q, q % 2 ? -1.f : (conj ? 1.f : -1.f));
        }
        CtrsmFence<HardEvent::S_V>();
        for (int q = 8; q < D; q *= 2) {
            Adds(idx[q], idx, int32_t(q / 2 * (swap ? 2 : RPITCH) * 4), q);
            Adds(idx[D + q], idx[D], int32_t(q / 2 * (swap ? 2 : RPITCH) * 4), q);
        }
        for (int rows = 2; rows < IX / D; rows *= 2)
            Adds(idx[rows * D], idx, int32_t(rows / 2 * (swap ? RPITCH : 2) * 4), rows * D);
        for (int row = 0; row < 32; row++) {
            Adds(diagIdx[row * 128], idx[row * 256], int32_t(0), 64);
            Adds(diagIdx[row * 128 + 64], idx[row * 256 + D], int32_t(0), 64);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadBlock(int ft, int slot, CtrsmBlockTasks& tasks)
    {
        int batch, ot, it;
        tasks.Locate(ft, batch, ot, it);
        const bool diag = ot == it;
        const int os = ot * P + (diag && !reverse ? 32 : 0), is = it * P + (diag && reverse ? 32 : 0),
                  cap = diag ? 32 : P;
        const int nr = os < k ? (k - os < cap ? k - os : cap) : 0, nc = is < k ? (k - is < cap ? k - is : cap) : 0;
        sb[slot] = batch;
        so[slot] = os;
        si[slot] = is;
        srn[slot] = nr;
        scn[slot] = nc;
        WaitFlag<HardEvent::V_MTE2>(slot);
        auto rb = raw[slot * RAW];
        const int rows = swap ? nr : nc, cols = swap ? nc : nr;
        if (rows < cap || cols < cap) {
            Duplicate(rb, 0.f, RAW);
            CtrsmFence<HardEvent::V_MTE2>(slot);
        }
        if (rows && cols) {
            auto ag = CtrsmGlobal<float>(
                (__gm__ float*)ap.GetValue(batch) + uint64_t(swap ? os : is) * lda * 2 + (swap ? is : os) * 2);
            CtrsmRead(rb, ag, rows, cols * 8, (lda - cols) * 8, (RPITCH - (cols * 2 + 7) / 8 * 8) / 8);
        }
        SetFlag<HardEvent::MTE2_V>(slot);
    }
    __aicore__ inline int ActiveRows(int valid, int nr, bool diag)
    {
        if constexpr (Prefetch)
            return (valid + 7) / 8 * 8;
        return diag ? 32 : trim ? (nr + 7) / 8 * 8 : P;
    }
    __aicore__ inline void WriteOutput(
        LocalTensor<float> ob, int slot, int nr, int nc, int cap, int width, int half, int active)
    {
        const uint64_t at =
            invSize + uint64_t(sb[slot]) * dim * dim + uint64_t((so[slot] + half * 32) * 2) * dim + si[slot] * 2;
        const int rows = Prefetch ? active : (trim ? (nr + 7) / 8 * 8 : cap),
                  cols = Prefetch || trim ? (nc + 3) / 4 * 4 : cap;
        CtrsmWrite(wg[at], ob, rows * 2, cols * 8, (width - cols * 2) / 8, (dim - cols * 2) * 4);
    }
    __aicore__ inline void PublishHalf(int slot, int nr, int nc, int half)
    {
        auto rb = raw[slot * RAW];
        const bool diag = so[slot] / P == si[slot] / P;
        const int cap = diag ? 32 : P, width = cap * 2;
        const int outSlot = outputs % OUTPUTS, event = outSlot + 2;
        if (outputs >= OUTPUTS)
            WaitFlag<HardEvent::MTE3_V>(event);
        auto ob = out[outSlot * OUT];
        const int valid = Prefetch ? (nr - half * 32 < 32 ? nr - half * 32 : 32) : nr,
                  active = ActiveRows(valid, nr, diag);
        auto index = diag ? diagIdx : idx;
        CtrsmGather(ob, rb, index, active * 2 * width, half * 32 * (swap ? RPITCH : 2) * 4);
        PipeBarrier<PIPE_V>();
        for (int q = 0; q < width; q += 64) {
            Mul(ob[q], ob[q], signs, 64, active, {1, 1, 0, uint8_t(width / 4), uint8_t(width / 4), 0});
            Mul(ob[width + q], ob[width + q], signs[8], 64, active,
                {1, 1, 0, uint8_t(width / 4), uint8_t(width / 4), 0});
        }
        CtrsmFence<HardEvent::V_MTE3>();
        WriteOutput(ob, slot, nr, nc, cap, width, half, active);
        SetFlag<HardEvent::MTE3_V>(event);
        outputs++;
    }
    __aicore__ inline void PublishBlock(int slot)
    {
        const int nr = srn[slot], nc = scn[slot];
        const bool diag = so[slot] / P == si[slot] / P;
        if (!nr || !nc)
            return;
        for (int half = 0; half < (Prefetch && !diag ? 2 : 1); half++) {
            if (half * 32 >= nr)
                break;
            PublishHalf(slot, nr, nc, half);
        }
    }

public:
    __aicore__ inline void Run(
        GM_ADDR aPtr, GM_ADDR bPtr, GM_ADDR workspace, int batchCount, int order, int columns, int aLd, int bLd,
        int side, int upper, int trans, int firstBatch, int lastBatch, int worker, int workers)
    {
        batches = batchCount;
        k = order;
        rhs = columns;
        lda = aLd;
        ldb = bLd;
        right = side != 0;
        swap = (trans != 0) != right, reverse = (upper != 0) != swap, conj = trans == 2;
        blocks = (k + P - 1) / P, kp = blocks * P, dim = kp * 2, rp = (rhs + 63) / 64 * 64;
        invSize = uint64_t(batches) * blocks * D * D, coeffSize = uint64_t(batches) * dim * dim;
        ap = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)aPtr);
        bp = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)bPtr);
        wg = CtrsmGlobal<float>((__gm__ float*)workspace);

        BuildIndices();
        trim = rhs <= 128 && batches > GetBlockNum() * 2;
        CtrsmBlockTasks tasks(blocks, batches, reverse, rhs <= 64 && (Prefetch || trim), true);
        if (worker < 0)
            worker = GetBlockIdx();
        if (!workers)
            workers = GetBlockNum();
        const int begin = firstBatch * tasks.per, end = lastBatch < 0 ? tasks.total : lastBatch * tasks.per;
        for (int slot = 0; slot < INPUTS; slot++)
            SetFlag<HardEvent::V_MTE2>(slot);
        for (int iter = 0, task = begin + worker; task < end; iter++, task += workers) {
            const int loads = iter == 0 ? INPUTS : 1;
            for (int q = 0; q < loads; q++) {
                const int future = iter == 0 ? q : iter + INPUTS - 1, ft = begin + worker + future * workers;
                if (ft < end)
                    LoadBlock(ft, future % INPUTS, tasks);
            }
            const int slot = iter % INPUTS;
            WaitFlag<HardEvent::MTE2_V>(slot);
            PublishBlock(slot);
            SetFlag<HardEvent::V_MTE2>(slot);
        }
        for (int slot = 0; slot < INPUTS; slot++)
            WaitFlag<HardEvent::V_MTE2>(slot);
        for (int slot = 0; slot < OUTPUTS; slot++)
            if (outputs > slot)
                WaitFlag<HardEvent::MTE3_V>(slot + 2);
        if (right)
            TransposeBlockRhs<RR, true>(bPtr, workspace, batches, k, rhs, ldb);
        ResetMask();
        CtrsmFence<HardEvent::V_S>();
    }
};
template <int RR, bool Prefetch = false>
__aicore__ inline void PackBlockSystem(
    GM_ADDR aPtr, GM_ADDR bPtr, GM_ADDR workspace, int batches, int k, int rhs, int lda, int ldb, int side, int upper,
    int trans, int firstBatch = 0, int lastBatch = -1, int worker = -1, int workers = 0)
{
    BlockPacker<RR, Prefetch> packer;
    packer.Run(
        aPtr, bPtr, workspace, batches, k, rhs, lda, ldb, side, upper, trans, firstBatch, lastBatch, worker, workers);
}
template <int RR, bool Prefetch = false>
__global__ __aicore__ void pack_block_system(
    GM_ADDR a, GM_ADDR b, GM_ADDR w, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    PackBlockSystem<RR, Prefetch>(a, b, w, batches, k, rhs, lda, ldb, side, upper, trans);
}
} // namespace CtrsmBatchedInverseCommon
namespace CtrsmBatchedBlockInverse {
template <bool Upper, bool Swap, bool Conj, bool Unit>
__global__ __aicore__ void prepare_grouped_block_inverse(
    GM_ADDR a, GM_ADDR w, int batches, int k, int lda, int side, int upper, int trans, int diag, int halfOnly)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrsmBatchedGroupedInverse::PlanarBatch<Upper, Swap, Conj, Unit> kernel;
    kernel.Run(a, w, batches, k, lda, halfOnly);
    if constexpr (Unit) {
        if (k % 32) {
            CtrsmBatchedInverseCommon::DirectRhs<32, 2> tail;
            tail.Run(a, w, batches, k, lda, side, upper, trans, diag, true);
        }
    }
}
__global__ __aicore__ void combine_inverse_cube(GM_ADDR w, int batches, int k, int rhs, int side, int upper, int trans)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    constexpr int H = 64, S = H * H, D = 128;
    LocalTensor<float> al1{TPosition::A1, 0, S}, bl1{TPosition::B1, S * 4, S}, al0{TPosition::A2, 0, S},
        bl0{TPosition::B2, 0, S}, cl0{TPosition::CO1, 0, S};
    const int blocks = (k + 63) / 64, small = (k + 31) / 32, dim = blocks * D, rp = (rhs + 63) / 64 * 64;
    const uint64_t inv = uint64_t(batches) * blocks * D * D, coef = uint64_t(batches) * dim * dim,
                   rb = side ? uint64_t(batches) * rp * dim : 0, temp = inv + coef + rb;
    const bool reverse = (upper != 0) != ((trans != 0) != (side != 0));
    auto wg = CtrsmGlobal<float>((__gm__ float*)w);
    InitSocState();
    SetHF32Mode(false);
    for (int task = GetBlockIdx(); task < batches * blocks; task += GetBlockNum()) {
        const int batch = task / blocks, panel = task % blocks;
        if (panel * 64 + 32 >= k)
            continue;
        const int outHalf = reverse ? 0 : 1, inHalf = 1 - outHalf;
        const uint64_t coff =
            inv + uint64_t(batch) * dim * dim + uint64_t(panel * D + outHalf * H) * dim + panel * D + inHalf * H;
        const uint64_t iIn = uint64_t(task) * D * D + inHalf * H * (D + 1),
                       iOut = uint64_t(task) * D * D + outHalf * H * (D + 1);
        for (int step = 0; step < 2; step++) {
            auto np = CtrsmCube::Nd(H, H, step ? D : dim, H);
            DataCopy(al1, wg[step ? iOut : coff], np);
            np.srcDValue = step ? H : D;
            DataCopy(bl1, wg[step ? temp + uint64_t(task) * S : iIn], np);
            CtrsmFence<HardEvent::MTE2_MTE1>();
            auto lp = CtrsmCube::Load3d(H, H, H, H, false, 0, 0);
            LoadData(al0, al1, lp);
            lp.enTranspose = true;
            LoadData(bl0, bl1, lp);
            CtrsmFence<HardEvent::MTE1_M>();
            auto mm = CtrsmCube::Mmad(H, H, H, true);
            Mmad(cl0, al0, bl0, mm);
            CtrsmFence<HardEvent::M_FIX>();
            auto fp = CtrsmCube::Fix(H, H, H, step ? D : H);
            const uint64_t dest =
                step ? uint64_t(task) * D * D + outHalf * H * D + inHalf * H : temp + uint64_t(task) * S;
            Fixpipe(wg[dest], cl0, fp);
            CtrsmFence<HardEvent::FIX_MTE2>();
            CtrsmFence<HardEvent::FIX_M>();
            CtrsmFence<HardEvent::M_MTE1>();
            CtrsmFence<HardEvent::MTE1_MTE2>();
        }
    }
    CtrsmFence<HardEvent::FIX_S>();
}
template <bool Upper, bool Swap, bool Conj, bool Unit>
__global__ __aicore__ void fused_grouped_sweep(
    GM_ADDR a, GM_ADDR b, GM_ADDR w, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans,
    int diag, int cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const int chunk = (batches + cores - 1) / cores, rounds = (chunk + 7) / 8;
    int core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= 2;
    }
    const int limit = (core + 1) * chunk < batches ? (core + 1) * chunk : batches;
    if ASCEND_IS_AIV {
        CtrsmBatchedGroupedInverse::PlanarBatch<Upper, Swap, Conj, Unit> inverse;
        for (int r = 0; r < rounds; r++) {
            if (r >= 8)
                CrossCoreWaitFlag<2>(1);
            const int first = core * chunk + r * 8, last = first + 8 < limit ? first + 8 : limit;
            inverse.Run(a, w, batches, k, lda, 1, first, last, GetSubBlockIdx(), 2);
            CtrsmBatchedInverseCommon::PackBlockSystem<64, true>(
                a, b, w, batches, k, rhs, lda, ldb, side, upper, trans, first, last, GetSubBlockIdx(), 2);
            CrossCoreSetFlag<2, PIPE_MTE3>(0);
        }
        for (int r = 0; r < rounds && r < 8; r++)
            CrossCoreWaitFlag<2>(1);
    } else {
        for (int r = 0; r < rounds; r++) {
            CrossCoreWaitFlag<2>(0);
            const int first = core * chunk + r * 8, last = first + 8 < limit ? first + 8 : limit;
            CtrsmBatchedInverseCommon::BlockCubeSweep<64, 640, 256, 32, 64, 640, true>(
                b, w, batches, k, rhs, ldb, side, upper, trans, first, last, 0, 1);
            CrossCoreSetFlag<2, PIPE_FIX>(1);
        }
    }
    ResetMask();
    if ASCEND_IS_AIV {
        CtrsmFence<HardEvent::V_S>();
    } else {
        CtrsmFence<HardEvent::FIX_S>();
    }
}
template <int Mode = 15>
void LaunchGeneralGroupedInverseDispatch(
    void* a, void* w, int batches, int k, int lda, int side, int upper, int trans, int diag, int cores, void* stream,
    int halfOnly)
{
    const bool swap = (trans != 0) != (side != 0), reverse = (upper != 0) != swap;
    const int mode = int(reverse) * 8 + int(swap) * 4 + int(trans == 2) * 2 + int(diag != 0);
    if (mode == Mode)
        prepare_grouped_block_inverse<bool(Mode & 8), bool(Mode & 4), bool(Mode & 2), bool(Mode & 1)>
            <<<cores * 2, nullptr, stream>>>(
                (GM_ADDR)a, (GM_ADDR)w, batches, k, lda, side, upper, trans, diag, halfOnly);
    else if constexpr (Mode > 0)
        LaunchGeneralGroupedInverseDispatch<Mode - 1>(
            a, w, batches, k, lda, side, upper, trans, diag, cores, stream, halfOnly);
}
} // namespace CtrsmBatchedBlockInverse
namespace CtrsmBatchedVectorFallback {
__aicore__ inline void Scale(GM_ADDR vector, int k, int inc, float re, float im)
{
    if (re == 1.f && im == 0.f)
        return;
    constexpr int P = 64;
    LocalTensor<float> values{TPosition::VECCALC, 0, P * 8};
    auto bg = CtrsmGlobal<float>((__gm__ float*)vector);
    for (int first = 0; first < k; first += P) {
        const int count = k - first < P ? k - first : P, stride = inc == 1 ? 2 : 8;
        CtrsmFence<HardEvent::V_MTE2>();
        if (inc == 1)
            CtrsmRead(values, bg[uint64_t(first) * 2], 1, count * 8);
        else
            CtrsmRead(values, bg[uint64_t(first) * inc * 2], count, 8, (uint64_t(inc) - 1) * 8);
        CtrsmFence<HardEvent::MTE2_S>();
        for (int q = 0; q < count; q++) {
            const auto x =
                ctrsv_netlib::Product({re, im}, {values.GetValue(q * stride), values.GetValue(q * stride + 1)});
            values.SetValue(q * stride, x.real);
            values.SetValue(q * stride + 1, x.imag);
        }
        CtrsmFence<HardEvent::S_MTE3>();
        if (inc == 1)
            CtrsmWrite(bg[uint64_t(first) * 2], values, 1, count * 8);
        else
            CtrsmWrite(bg[uint64_t(first) * inc * 2], values, count, 8, 0, (uint64_t(inc) - 1) * 8);
        CtrsmFence<HardEvent::MTE3_V>();
    }
}
template <int Mode = 7>
__aicore__ inline void Solve(GM_ADDR a, GM_ADDR b, const CtrsvTilingData& td, int mode)
{
    if (mode == Mode) {
        ctrsv_generic::Kernel<bool(Mode & 4), bool(Mode & 2), bool(Mode & 1), true> kernel;
        kernel.Init(a, b, td);
        kernel.Process();
    } else if constexpr (Mode > 0)
        Solve<Mode - 1>(a, b, td, mode);
}
__global__ __aicore__ void tiled_solve(
    GM_ADDR a, GM_ADDR b, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans, int diag,
    float re, float im)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (rhs == 0) {
        return;
    }
    auto ap = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)a);
    auto bp = CtrsmGlobal<uint64_t>((__gm__ uint64_t*)b);
    const bool right = side != 0, transpose = (trans != 0) != right, scaleAfter = right && trans != 0;
    const int mode = int(transpose) * 4 + int(trans == 2) * 2 + int(diag != 0);
    const CtrsvTilingData td{k, lda, right ? ldb : 1, upper ? ACLBLAS_UPPER : ACLBLAS_LOWER, trans, diag};
    for (uint64_t task = GetBlockIdx(); task < uint64_t(batches) * rhs; task += GetBlockNum()) {
        GM_ADDR matrix = (GM_ADDR)ap.GetValue(task / rhs);
        GM_ADDR vector = (GM_ADDR)bp.GetValue(task / rhs) + uint64_t(task % rhs) * (right ? 1 : ldb) * 8;
        if (!scaleAfter)
            Scale(vector, k, td.incx, re, im);
        Solve(matrix, vector, td, mode);
        if (scaleAfter)
            Scale(vector, k, td.incx, re, im);
    }
    CtrsmFence<HardEvent::V_S>();
}
} // namespace CtrsmBatchedVectorFallback

void LaunchCtrsmBatchedInverseCube(
    void* a, void* b, void* w, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans, int diag,
    int cores, void* stream)
{
    using namespace CtrsmBatchedInverseCube;
    if (k <= 32)
        LaunchInverseCube<32>(a, b, w, batches, k, rhs, lda, ldb, side, upper, trans, diag, cores, stream);
    else if (k <= 48)
        LaunchInverseCube<48>(a, b, w, batches, k, rhs, lda, ldb, side, upper, trans, diag, cores, stream);
    else
        LaunchInverseCube<64>(a, b, w, batches, k, rhs, lda, ldb, side, upper, trans, diag, cores, stream);
}
template <bool Prefetch>
void LaunchBlockPack(
    void* a, void* b, void* w, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans, int cores,
    void* stream)
{
    if (side && rhs > 64 && (rhs <= 128 || (k <= 256 && rhs <= 320 && batches > cores * 4)))
        CtrsmBatchedInverseCommon::pack_block_system<128, Prefetch><<<cores * 2, nullptr, stream>>>(
            (GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, lda, ldb, side, upper, trans);
    else
        CtrsmBatchedInverseCommon::pack_block_system<64, Prefetch><<<cores * 2, nullptr, stream>>>(
            (GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, lda, ldb, side, upper, trans);
}
template <int NR, int LC = 128, int KC = 128, int P = 64, int StorageP = P, int Cache = 0, bool Prefetch = false>
void LaunchSweep(
    void* b, void* w, int batches, int k, int rhs, int ldb, int side, int upper, int trans, int cores, void* stream)
{
    CtrsmBatchedInverseCommon::block_cube_sweep<NR, LC, KC, P, StorageP, Cache, Prefetch>
        <<<cores, nullptr, stream>>>((GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, ldb, side, upper, trans);
}
void LaunchPrefetchedSweep(
    void* b, void* w, int batches, int k, int rhs, int ldb, int side, int upper, int trans, int cores, void* stream)
{
    if (!side && rhs > 128)
        LaunchSweep<256, 320, 64, 32, 64, 320, true>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else if (k <= 320 && rhs <= 64)
        LaunchSweep<64, 640, 256, 32, 64, 640, true>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else if (k <= 256 && rhs > 64)
        LaunchSweep<128, 512, 128, 32, 64, 512, true>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else if (k <= 768 && rhs <= 64)
        LaunchSweep<64, 512, 256, 32, 64, 1536>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else if (k <= 384 && rhs > 64)
        LaunchSweep<128, 512, 128, 32, 64, 768>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else if (rhs > 64)
        LaunchSweep<128, 512, 128, 32, 64>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else
        LaunchSweep<64, 1024, 256, 32, 64>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
}
void LaunchOrdinarySweep(
    void* b, void* w, int batches, int k, int rhs, int ldb, int side, int upper, int trans, int cores, void* stream)
{
    CtrsmBatchedBlockInverse::combine_inverse_cube<<<cores, nullptr, stream>>>(
        (GM_ADDR)w, batches, k, rhs, side, upper, trans);
    if (!side && k <= 256 && rhs > 128 && rhs <= 256 && batches > cores * 4)
        LaunchSweep<256, 256, 64>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else if (k <= 1024 && rhs > 32 && rhs <= 256 && batches > cores * 4)
        LaunchSweep<128, 512>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else if (rhs > 32)
        LaunchSweep<128>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else
        LaunchSweep<32>(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
}
bool CanFuseGroupedSweep(int side, int upper, int trans, int diag, int k, int rhs, int batches, int cores)
{
    return !side && upper && !trans && !diag && k % 32 == 0 && k <= 320 && rhs <= 64 && batches >= cores;
}
void LaunchCtrsmBatchedBlockInverse(
    void* a, void* b, void* w, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans, int diag,
    int cores, void* stream)
{
    using namespace CtrsmBatchedBlockInverse;
    // Fuse complete upper non-unit panels with narrow left RHS into one mixed launch.
    if (CanFuseGroupedSweep(side, upper, trans, diag, k, rhs, batches, cores)) {
        fused_grouped_sweep<true, false, false, false><<<cores, nullptr, stream>>>(
            (GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, lda, ldb, side, upper, trans, diag, cores);
        return;
    }
    // Cache the full RHS only when it and both coefficient slots fit in L1.
    const bool prefetched =
        batches > cores * 4 && (rhs <= 128 || (k <= (side ? 256 : 160) && rhs <= (side ? 320 : 256)));
    if (batches > cores * 4)
        LaunchGeneralGroupedInverseDispatch(a, w, batches, k, lda, side, upper, trans, diag, cores, stream, prefetched);
    else
        CtrsmBatchedInverseCube::prepare_triangular_inverse<32, 2>
            <<<cores * 2, nullptr, stream>>>((GM_ADDR)a, (GM_ADDR)w, batches, k, lda, side, upper, trans, diag);
    if (prefetched)
        LaunchBlockPack<true>(a, b, w, batches, k, rhs, lda, ldb, side, upper, trans, cores, stream);
    else
        LaunchBlockPack<false>(a, b, w, batches, k, rhs, lda, ldb, side, upper, trans, cores, stream);
    if (prefetched)
        LaunchPrefetchedSweep(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    else
        LaunchOrdinarySweep(b, w, batches, k, rhs, ldb, side, upper, trans, cores, stream);
    if (side && rhs > 64 && (rhs <= 128 || (k <= 256 && rhs <= 320 && batches > cores * 4)))
        CtrsmBatchedInverseCommon::restore_block_rhs<128>
            <<<cores * 2, nullptr, stream>>>((GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, ldb);
    else if (side)
        CtrsmBatchedInverseCommon::restore_block_rhs<64>
            <<<cores * 2, nullptr, stream>>>((GM_ADDR)b, (GM_ADDR)w, batches, k, rhs, ldb);
}
void LaunchCtrsmBatchedVectorFallback(
    void* a, void* b, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans, int diag, float re,
    float im, int cores, void* stream)
{
    CtrsmBatchedVectorFallback::tiled_solve<<<cores, nullptr, stream>>>(
        (GM_ADDR)a, (GM_ADDR)b, batches, k, rhs, lda, ldb, side, upper, trans, diag, re, im);
}
