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
__global__ __aicore__ void ctrsmbatched_zero(GM_ADDR b, int m, int n, int ldb, int batches)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto pointers = CtrsmGlobal<uint64_t>(reinterpret_cast<__gm__ uint64_t*>(b), batches);
    LocalTensor<float> zero(TPosition::VECCALC, 0, 8192);
    Duplicate(zero, 0.0f, 8192);
    CtrsmFence<HardEvent::V_MTE3>();
    for (uint64_t task = GetBlockIdx(); task < uint64_t(batches) * n; task += GetBlockNum()) {
        auto output =
            CtrsmGlobal<float>(reinterpret_cast<__gm__ float*>(pointers.GetValue(task / n)) + (task % n) * ldb * 2);
        for (uint64_t offset = 0; offset < uint64_t(m) * 2; offset += 8192) {
            const uint32_t count = uint64_t(m) * 2 - offset < 8192 ? uint64_t(m) * 2 - offset : 8192;
            CtrsmWrite(output[offset], zero, 1, count * 4);
        }
    }
    CtrsmFence<HardEvent::MTE3_S>();
}
class CtrsmSquareBoundary {
    static constexpr int K = 32, PLANE = 2048;
    static constexpr int32_t kStrictMagnitudeBits = 0x42800000; // IEEE-754 bits of 64.0f.
    LocalTensor<float> rawA{TPosition::VECCALC, 98336, 8192}, rawB{TPosition::VECCALC, 0, 8192},
        cr{TPosition::VECCALC, 32768, PLANE}, ci{TPosition::VECCALC, 40960, PLANE}, tmp{TPosition::VECCALC, 49152, 256};
    __aicore__ inline void AccumulateMagnitude(LocalTensor<float> src, int n)
    {
        Abs(ci, src, n); // Nonnegative IEEE magnitude bits preserve order, including NaN markers.
        Max(cr.template ReinterpretCast<int32_t>(), cr.template ReinterpretCast<int32_t>(),
            ci.template ReinterpretCast<int32_t>(), n);
    }
    __aicore__ inline bool NeedsStrict(int count)
    {
        Duplicate(cr, 0.f, PLANE);
        for (int q = 0; q < count * K * K * 2; q += PLANE) {
            const int n = count * K * K * 2 - q < PLANE ? count * K * K * 2 - q : PLANE;
            AccumulateMagnitude(rawA[q], n);
            AccumulateMagnitude(rawB[q], n);
        }
        Cast(cr, cr.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, PLANE);
        WholeReduceMax(tmp, cr, 64U, uint8_t(PLANE / 64), 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(tmp[128], tmp, uint64_t(PLANE / 64), 1U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        CtrsmFence<HardEvent::V_S>();
        const float maximum = tmp.GetValue(128);
        CtrsmFence<HardEvent::S_V>();
        // Cast above converts nonnegative magnitude bits to float values, preserving NaN marker ordering.
        // Compare in that encoded domain, rather than against the original magnitude 64.0f.
        return maximum > static_cast<float>(kStrictMagnitudeBits);
    }

public:
    __aicore__ inline bool Run(GlobalTensor<uint64_t> ap, GlobalTensor<uint64_t> bp, int first, int count, bool upper)
    {
        CtrsmFence<HardEvent::V_MTE2>();
        for (int b = 0; b < count; b++) {
            auto a = CtrsmGlobal<float>((__gm__ float*)ap.GetValue(first + b));
            auto x = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(first + b));
            DataCopy(rawA[b * PLANE], a, PLANE);
            DataCopy(rawB[b * PLANE], x, PLANE);
        }
        CtrsmFence<HardEvent::MTE2_V>();
        if (!NeedsStrict(count))
            return false;
        const CtrsvTilingData td{K, K, 1, upper ? ACLBLAS_UPPER : ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT};
        for (int task = 0; task < count * K; task++) {
            ctrsv_generic::Kernel<false, false, false, true> kernel;
            kernel.Init(
                (GM_ADDR)ap.GetValue(first + task / K), (GM_ADDR)bp.GetValue(first + task / K) + (task % K) * K * 8,
                td);
            kernel.Process();
        }
        return true;
    }
};
namespace CtrsmBatchedPlanarSquare {
template <int Order, bool Upper, int Group>
class PlanarSmallBatch
    : private CtrsmPlanar<Order, Group, 2 * Order * Order + 8, 0, 0, 0, (2 * Order * Order + 8) * Group> {
    using Base = CtrsmPlanar<Order, Group, 2 * Order * Order + 8, 0, 0, 0, (2 * Order * Order + 8) * Group>;
    using Base::A, Base::BASE, Base::ci, Base::cr, Base::D, Base::den, Base::di, Base::dr, Base::G, Base::idx, Base::IX,
        Base::K, Base::L, Base::Normalize, Base::raw, Base::Reciprocal, Base::S, Base::T, Base::tmp, Base::X, Base::xi,
        Base::xr;
    static constexpr int BBASE = (BASE + IX + 3 * D + 2 * L + 7) / 8 * 8;
    LocalTensor<float> rawB{TPosition::VECCALC, BBASE * 4, A* G};
    static_assert((BBASE + A * G) * 4 <= 192 * 1024, "fused input UB");
    static constexpr int TABBASE = BBASE + A * G + 4 * G, TABLE = D + 2 * T + X + (K == 8 ? A * G : 2 * K * K);
    LocalTensor<uint32_t> table{TPosition::VECCALC, TABBASE * 4, TABLE};
    static_assert((TABBASE + TABLE) * 4 <= 192 * 1024, "constantindex UB");
    GlobalTensor<uint64_t> ap, bp;
    __aicore__ inline void LoadA(int first, int count)
    {
        CtrsmFence<HardEvent::V_MTE2>();
        for (int b = 0; b < count; b++) {
            auto a = CtrsmGlobal<float>((__gm__ float*)ap.GetValue(first + b));
            DataCopy(raw[b * A], a, 2 * K * K);
        }
        CtrsmFence<HardEvent::MTE2_V>();
        Gather(dr, raw, table, 0, D);
        Gather(di, raw, table, 4, D);
        Reciprocal();
        PipeBarrier<PIPE_V>();
        Gather(cr, raw, table[D], 0, T);
        Gather(ci, raw, table[D], 4, T);
        for (int b = 0; b < count; b++) {
            auto g = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(first + b));
            DataCopy(rawB[b * A], g, 2 * K * K);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadB(int first, int count)
    {
        CtrsmFence<HardEvent::MTE2_V>();
        Gather(xr, rawB, table[D + 2 * T], 0, K * S);
        Gather(xi, rawB, table[D + 2 * T], 4, K * S);
        PipeBarrier<PIPE_V>();
        Muls(den, di, -1.f, D);
        PipeBarrier<PIPE_V>();
        for (int q = 0; q < L; q += 64) {
            const int h = q / (8 * K) * 8;
            auto work = raw;
            Mul(work, xr[q], dr[h], 64, K, {1, 1, 0, 8, S / 8, G / 8});
            MulAddDst<float, float>(work, xi[q], di[h], 64, K, {1, 1, 0, 8, S / 8, G / 8});
            Mul(xi[q], xi[q], dr[h], 64, K, {1, 1, 0, S / 8, S / 8, G / 8});
            MulAddDst<float, float>(xi[q], xr[q], den[h], 64, K, {1, 1, 0, S / 8, S / 8, G / 8});
            DataCopy(xr[q], work, {K, 8, 0, uint16_t(S / 8 - 8)});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Store(int first, int count)
    {
        if constexpr (K == 8)
            Gather(raw, LocalTensor<float>{TPosition::VECCALC, A * G * 4, 2 * X}, table[D + 2 * T + X], 0, A * G);
        else
            for (int b = 0; b < count; b++)
                Gather(
                    raw[b * A], LocalTensor<float>{TPosition::VECCALC, A * G * 4, 2 * X}, table[D + 2 * T + X],
                    ((b / 8) * K * 8 + b % 8) * 4, 2 * K * K);
        CtrsmFence<HardEvent::V_MTE3>();
        for (int b = 0; b < count; b++) {
            auto g = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(first + b));
            DataCopy(g, raw[b * A], 2 * K * K);
        }
        CtrsmFence<HardEvent::MTE3_V>();
    }

public:
    __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, int n)
    {
        if (GetBlockIdx() * G >= n)
            return;
        ap.SetGlobalBuffer((__gm__ uint64_t*)a);
        bp.SetGlobalBuffer((__gm__ uint64_t*)b);
        auto t = CtrsmGlobal<uint32_t>((__gm__ uint32_t*)&Indices<K, G, Upper>::value);
        DataCopy(table, t, TABLE);
        CtrsmFence<HardEvent::MTE2_V>();
        for (int first = GetBlockIdx() * G; first < n; first += GetBlockNum() * G) {
            int count = n - first < G ? n - first : G;
            LoadA(first, count);
            PipeBarrier<PIPE_V>();
            Normalize(table[D + T]);
            LoadB(first, count);
            Base::template RowSweep<Upper, false>();
            Store(first, count);
        }
        ResetMask();
        CtrsmFence<HardEvent::V_S>();
    }
};
template <int K, bool U, int G = K == 8 ? 32 : 16>
__global__ __aicore__ void planar_square(GM_ADDR a, GM_ADDR b, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    PlanarSmallBatch<K, U, G> k;
    k.Run(a, b, n);
}
} // namespace CtrsmBatchedPlanarSquare
namespace CtrsmBatchedSmallMixed {
struct MixedIndexTable {
    uint32_t diagonal[128], coefficient[960], normalize[960], publish[1024], offdiag[1024];
};
constexpr MixedIndexTable MakeMixedIndices(bool upper)
{
    MixedIndexTable x{};
    for (int row = 0; row < 16; row++)
        for (int b = 0; b < 8; b++)
            x.diagonal[row * 8 + b] = ((b / 2) * 2048 + (b % 2) * 1056 + row * 66) * 4;
    int at = 0;
    for (int i = 0; i < 15; i++) {
        int p = upper ? 15 - i : i, start = upper ? 0 : p + 1, rows = 15 - i;
        for (int row = 0; row < rows; row++)
            for (int b = 0; b < 8; b++) {
                x.coefficient[at] = ((b / 2) * 2048 + (b % 2) * 1056 + (p * 32 + start + row) * 2) * 4;
                x.normalize[at] = ((start + row) * 8 + b) * 4;
                at++;
            }
    }
    for (int row = 0; row < 16; row++)
        for (int q = 0; q < 64; q++) {
            int j = q % 32;
            x.publish[row * 64 + q] = ((j / 2) * 136 + row * 8 + ((j % 2) ^ (q / 32)) * 2240) * 4;
            x.offdiag[row * 64 + q] = ((upper ? 1024 : 32) + row * 2 + (j / 2) * 64 + ((j % 2) ^ (q / 32))) * 4;
        }
    return x;
}
static __gm__ MixedIndexTable mixedUpper __attribute__((aligned(32))) = MakeMixedIndices(true);
static __gm__ MixedIndexTable mixedLower __attribute__((aligned(32))) = MakeMixedIndices(false);
template <bool Upper, bool Guard>
class PlanarBatch : private CtrsmPlanar<16, 8, 544, 64, 160, 224, 2176> {
    LocalTensor<float> out{TPosition::VECCALC, 65536, 4096};
    LocalTensor<uint32_t> table{TPosition::VECCALC, 81920, 4096};
    LocalTensor<uint64_t> pointers{TPosition::VECCALC, 98304, 4};
    LocalTensor<float> fullA{TPosition::VECCALC, 98336, 8192};
    GlobalTensor<uint64_t> ap, bp;
    int totalBatches;
    __aicore__ inline void LoadA(int first, int count)
    {
        CtrsmFence<HardEvent::V_MTE2>();
        CtrsmRead(pointers, ap[first], 1, count / 2 * 8);
        CtrsmFence<HardEvent::MTE2_S>();
        for (int b = 0; b < count / 2; b++) {
            auto ag = CtrsmGlobal<float>((__gm__ float*)pointers.GetValue(b));
            DataCopy(fullA[b * 2048], ag, 2048);
        }
        CtrsmFence<HardEvent::MTE2_V>();
        Gather(dr, fullA, table, 0, D);
        Gather(di, fullA, table, 4, D);
        Reciprocal();
        PipeBarrier<PIPE_V>();
        Gather(cr, fullA, table[128], 0, T);
        Gather(ci, fullA, table[128], 4, T);
        PipeBarrier<PIPE_V>();
    }
    template <int Part>
    __aicore__ inline void Publish(GM_ADDR workspace, int first, int count)
    {
        constexpr bool Offdiag = Part == 2;
        constexpr int type = Offdiag ? 0 : Part;
        auto wg = CtrsmGlobal<float>((__gm__ float*)workspace);
        auto wh = CtrsmGlobal<half>((__gm__ half*)workspace);
#pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            if (pair * 4 >= count)
                continue;
#pragma unroll
            for (int g = 0; g < 2; g++) {
                const int lane = pair * 4 + g * 2 + type;
                if (lane >= count)
                    continue;
                auto tile = Offdiag ? xr : raw;
                if constexpr (Offdiag) {
                    Gather(tile, fullA, table[3072], uint32_t((pair * 2 + g) * 2048 * 4), 1024);
                } else
                    Gather(
                        tile, LocalTensor<float>{TPosition::VECCALC, A * G * 4, 2 * X + 64}, table[2048], lane * 4,
                        1024);
                PipeBarrier<PIPE_V>();
                constexpr uint64_t sign = Offdiag ? ~0xAAAAAAAAULL : 0xAAAAAAAAULL;
                AscendC::SetVectorMask<float>(0, sign);
                Muls<float, float, false>(tile, tile, -1.f, sign, 16, {1, 1, 8, 8});
                ResetMask();
                DataCopy(out[g * 32 * 65], tile, {32, 4, 0, 4});
                PipeBarrier<PIPE_V>();
            }
            if constexpr (Offdiag)
                Cast(raw.ReinterpretCast<half>(), out, RoundMode::CAST_RINT, 4096);
            CtrsmFence<HardEvent::V_MTE3>();
            const uint64_t dest = (uint64_t(first / 4 + pair) * 3 + (Offdiag ? 2 : type)) * 4096;
            if constexpr (Offdiag)
                DataCopy(wh[dest * 2], raw.ReinterpretCast<half>(), 4096);
            else
                DataCopy(wg[dest], out, 4096);
            CtrsmFence<HardEvent::MTE3_V>();
        }
    }

public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, int n, int cores)
    {
        ap.SetGlobalBuffer((__gm__ uint64_t*)a);
        bp.SetGlobalBuffer((__gm__ uint64_t*)b);
        totalBatches = n;
        auto gt = CtrsmGlobal<uint32_t>((__gm__ uint32_t*)(Upper ? &mixedUpper : &mixedLower));
        DataCopy(table, gt, 4096);
    }
    __aicore__ inline void Prepare(GM_ADDR w, int firstBatch, int batches)
    {
        const int first = firstBatch * 2, count = (batches - firstBatch) * 2 < 8 ? (batches - firstBatch) * 2 : 8;
        if (count > 0) {
            if constexpr (Guard) {
                CtrsmSquareBoundary boundary;
                const bool strict = boundary.Run(ap, bp, firstBatch, count / 2, Upper);
                LocalTensor<int32_t> flags{TPosition::VECCALC, 65536, 8};
                auto output = CtrsmGlobal<int32_t>((__gm__ int32_t*)w + uint64_t((totalBatches + 1) / 2) * 3 * 4096);
                Duplicate(flags, int32_t(strict), 8);
                CtrsmFence<HardEvent::V_MTE3>();
                CtrsmWrite(output[firstBatch / 2], flags, 1, ((count + 3) / 4) * 4);
                CtrsmFence<HardEvent::MTE3_V>();
                if (strict) {
                    CrossCoreSetFlag<2, PIPE_MTE3>(0);
                    return;
                }
            }
            LoadA(firstBatch, count);
            Normalize(table[1088]);
            Identity();
            Muls(raw, ci, -1.f, T);
            PipeBarrier<PIPE_V>();
            ColumnSweep<Upper>();
            ResetMask();
            Duplicate(out, 0.f, 4096);
            PipeBarrier<PIPE_V>();
            Publish<Upper ? 1 : 0>(w, first, count);
            CrossCoreSetFlag<2, PIPE_MTE3>(0);
            Publish<Upper ? 0 : 1>(w, first, count);
            Publish<2>(w, first, count);
        } else
            CrossCoreSetFlag<2, PIPE_MTE3>(0);
    }
};
template <bool Guard>
class PairedCube {
    static constexpr int D = 64, S = 4096, W = 4;
    LocalTensor<float> al1{TPosition::A1, 0, W * 2048}, bl1{TPosition::B1, W * 8192, W * 3 * S},
        al0{TPosition::A2, 0, 2 * S}, bl0{TPosition::B2, 0, 2 * S}, cl0{TPosition::CO1, 0, 2 * S};
    LocalTensor<half> rhsHalf{TPosition::A1, 229376, 4 * 2048}, coefHalf{TPosition::B1, 245760, 4 * 4096};
    GlobalTensor<uint64_t> bp;
    GlobalTensor<float> wg;
    GlobalTensor<half> wh;
    GlobalTensor<int32_t> strictFlags;

    uint64_t address[8];
    bool contiguous[4], skipped[4];
    __aicore__ inline bool SkipGroup(int g)
    {
        if constexpr (Guard)
            return skipped[g];
        return false;
    }
    __aicore__ inline void LoadAddresses(int firstGroup, int batches, int groups)
    {
        for (int g = 0; g < groups; g++) {
            if constexpr (Guard)
                skipped[g] = strictFlags.GetValue(firstGroup + g) != 0;
            address[g * 2] = bp.GetValue((firstGroup + g) * 2);
            address[g * 2 + 1] = 0;
            if ((firstGroup + g) * 2 + 1 < batches)
                address[g * 2 + 1] = bp.GetValue((firstGroup + g) * 2 + 1);
            contiguous[g] = address[g * 2 + 1] && address[g * 2 + 1] == address[g * 2] + 2048 * sizeof(float);
            if (!address[g * 2 + 1])
                InitConstValue(al1[g * 2048], InitConstValueParams<float>{1, 2048 / 8, 0, 0.f});
        }
    }
    __aicore__ inline void LoadStageRhs(int input, int groups)
    {
        auto np = CtrsmCube::Nd(32, 32, 64, 32);
        for (int g = 0; g < groups; g++) {
            if (SkipGroup(g))
                continue;
            np.ndNum = contiguous[g] ? 2 : 1;
            np.srcNdMatrixStride = contiguous[g] ? 2048 : 0;
            np.dstNzMatrixStride = contiguous[g] ? 1024 : 0;
            for (int h = 0; h < (contiguous[g] ? 1 : 2); h++)
                if (address[g * 2 + h]) {
                    auto bg = CtrsmGlobal<float>((__gm__ float*)address[g * 2 + h]);
                    DataCopy(al1[g * 2048 + h * 1024], bg[input * 32], np);
                }
        }
    }
    __aicore__ inline void LoadOperands(int stage, int g, int bank, int coef)
    {
        if (stage == 1) {
            auto lh = CtrsmCube::Load3d<half>(32, 64, 64, 32, false, 0, 0);
            LoadData2DParams lbh;
            lbh.srcStride = 1;
            lbh.repeatTimes = S / 256;
            LoadData(al0.ReinterpretCast<half>()[bank * S * 2], rhsHalf[g * 2048], lh);
            LoadData(bl0.ReinterpretCast<half>()[bank * S * 2], coefHalf[g * S], lbh);
        } else {
            auto la = CtrsmCube::Load3d(32, D, D, 32, false, 0, 0);
            LoadData2DParams lb;
            lb.srcStride = 1;
            lb.repeatTimes = S / 128;
            LoadData(al0[bank * S], al1[g * 2048], la);
            LoadData(bl0[bank * S], bl1[(g * 3 + coef) * S], lb);
        }
    }
    __aicore__ inline void StorePair(int stage, int g, int bank, int output, bool complete)
    {
        auto fp = CtrsmCube::Fix(32, 32, 32, 64);
        fp.unitFlag = complete ? (stage == 0 ? 2 : 3) : 0;
        fp.ndNum = contiguous[g] ? 2 : 1;
        fp.srcNdStride = contiguous[g] ? 4 : 0;
        fp.dstNdStride = contiguous[g] ? 2048 : 0;
        for (int h = 0; h < (contiguous[g] ? 1 : 2); h++)
            if (address[g * 2 + h]) {
                auto bg = CtrsmGlobal<float>((__gm__ float*)address[g * 2 + h]);
                Fixpipe(bg[output * 32], cl0[bank * S + h * 1024], fp);
            }
        if (stage == 0) {
            auto feedback = CtrsmCube::Fix(32, 64, 32, 32);
            feedback.quantPre = QuantMode_t::F322F16;
            feedback.unitFlag = complete ? 3 : 0;
            Fixpipe<half, float, CFG_NZ>(rhsHalf[g * 2048], cl0[bank * S], feedback);
        }
    }
    __aicore__ inline void ComputePair(int stage, int g, int coef, int output, int issued)
    {
        const int bank = issued % 2;
        if (issued >= 2)
            WaitFlag<HardEvent::M_MTE1>(bank);
        LoadOperands(stage, g, bank, coef);
        CtrsmFence<HardEvent::MTE1_M>(bank);
        const bool complete = address[g * 2 + 1] != 0;
        // An unpaired tail must drain prior L0C reads before disabling unitFlag.
        if (!complete)
            CtrsmFence<HardEvent::FIX_M>();
        auto mm = CtrsmCube::Mmad(32, D, D, true);
        mm.unitFlag = complete ? 3 : 0;
        if (stage == 1)
            Mmad(
                cl0[bank * S], al0.ReinterpretCast<half>()[bank * S * 2], bl0.ReinterpretCast<half>()[bank * S * 2],
                mm);
        else
            Mmad(cl0[bank * S], al0[bank * S], bl0[bank * S], mm);
        SetFlag<HardEvent::M_MTE1>(bank);
        if (!complete)
            CtrsmFence<HardEvent::M_FIX>(bank);
        StorePair(stage, g, bank, output, complete);
        if (!complete)
            CtrsmFence<HardEvent::FIX_M>();
    }
    __aicore__ inline void ApplyStage(int stage, int firstGroup, int groups, int first, int& issued)
    {
        const int second = 1 - first, input = stage < 2 ? first : second, output = stage == 0 ? first : second,
                  coef = stage == 1 ? 2 : (stage == 0 ? first : second);
        if (stage == 1) {
            CrossCoreWaitFlag<2>(2);
            auto diag = CtrsmCube::Nd(D, D, D, D, groups, 3 * S, 3 * S);
            DataCopy(bl1[second * S], wg[(uint64_t(firstGroup) * 3 + second) * S], diag);
            auto hp = CtrsmCube::Nd(64, 64, 64, 64, groups, 3 * S * 2, S);
            DataCopy(coefHalf, wh[(uint64_t(firstGroup) * 3 + 2) * S * 2], hp);
            CtrsmFence<HardEvent::MTE2_MTE1>();
        }
        if (stage == 2) {
            CtrsmFence<HardEvent::MTE1_MTE2>();
            CtrsmFence<HardEvent::FIX_MTE2>();
        }
        if (stage != 1)
            LoadStageRhs(input, groups);
        if (stage == 1) {
            CtrsmFence<HardEvent::FIX_MTE1>();
            SetAtomicAdd<float>();
        } else
            CtrsmFence<HardEvent::MTE2_MTE1>();
        for (int g = 0; g < groups; g++) {
            if (SkipGroup(g))
                continue;
            ComputePair(stage, g, coef, output, issued++);
        }
        DisableDmaAtomic();
    }

public:
    __aicore__ inline void Init(GM_ADDR b, GM_ADDR w, int batches, int cores)
    {
        bp.SetGlobalBuffer((__gm__ uint64_t*)b);
        wg.SetGlobalBuffer((__gm__ float*)w);
        wh.SetGlobalBuffer((__gm__ half*)w);
        strictFlags.SetGlobalBuffer((__gm__ int32_t*)w + uint64_t((batches + 1) / 2) * 3 * S);
        InitSocState();
        SetHF32Mode(true);
        SetMMRowMajor();
    }
    __aicore__ inline void ApplyWave(int firstGroup, int batches, int upper)
    {
        const int groups = (batches - firstGroup * 2 + 1) / 2 < W ? (batches - firstGroup * 2 + 1) / 2 : W;
        if (groups <= 0) {
            CrossCoreWaitFlag<2>(2);
            return;
        }
        auto np = CtrsmCube::Nd(D, D, D, D, groups, 3 * S, 3 * S);
        const int first = upper ? 1 : 0;
        DataCopy(bl1[first * S], wg[(uint64_t(firstGroup) * 3 + first) * S], np);
        LoadAddresses(firstGroup, batches, groups);
        int issued = 0;
#pragma unroll
        for (int stage = 0; stage < 3; stage++)
            ApplyStage(stage, firstGroup, groups, first, issued);
        for (int bank = 0; bank < 2 && bank < issued; bank++)
            WaitFlag<HardEvent::M_MTE1>(bank);
        CtrsmFence<HardEvent::MTE1_MTE2>();
        CtrsmFence<HardEvent::FIX_MTE2>();
    }
};
template <bool Upper, bool Guard = false>
__global__ __aicore__ void paired_mixed(GM_ADDR a, GM_ADDR b, GM_ADDR w, int n, int cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const int pairsPerCore = ((n + 1) / 2 + cores - 1) / cores, rounds = (pairsPerCore + 3) / 4;
    if ASCEND_IS_AIV {
        PlanarBatch<Upper, Guard> v;
        v.Init(a, b, n, cores);
        const int core = GetBlockIdx() / 2, sub = GetSubBlockIdx();
        for (int r = 0; r < rounds; r++) {
            const int limit = (core + 1) * pairsPerCore * 2 < n ? (core + 1) * pairsPerCore * 2 : n;
            // Bound both ready counters before preparing the next wave.
            if (r >= 8)
                CrossCoreWaitFlag<2>(1);
            v.Prepare(w, core * pairsPerCore * 2 + r * 8 + sub * 4, limit);
            CrossCoreSetFlag<2, PIPE_MTE3>(2);
        }
        for (int r = 0; r < rounds && r < 8; r++)
            CrossCoreWaitFlag<2>(1);
        ResetMask();
        CtrsmFence<HardEvent::V_S>();
    } else {
        PairedCube<Guard> c;
        c.Init(b, w, n, cores);
        const int core = GetBlockIdx();
        for (int r = 0; r < rounds; r++) {
            CrossCoreWaitFlag<2>(0);
            const int limit = (core + 1) * pairsPerCore * 2 < n ? (core + 1) * pairsPerCore * 2 : n;
            c.ApplyWave(core * pairsPerCore + r * 4, limit, Upper);
            CrossCoreSetFlag<2, PIPE_FIX>(1);
        }
        DisableDmaAtomic();
        CtrsmFence<HardEvent::FIX_S>();
    }
}
} // namespace CtrsmBatchedSmallMixed
namespace CtrsmBatchedSmallInverse {
constexpr int MB = 48;       // padded matrix order
constexpr int NR = 208;      // padded RHS count
constexpr int WLD = 64;      // W row stride (matches one 64-lane vector)
constexpr int WN = MB * WLD; // 3072 floats per batch W
constexpr int BN = MB * NR;  // 9984 floats per batch B/X
constexpr int AP = 1152;
constexpr int CP = 23 * 192;
constexpr int CHUNKS = 3;
template <int Order, bool Pad>
struct RhsIndices {
    int32_t input[2][4 * MB], output[2][4 * MB];
    static __gm__ RhsIndices value;
};
template <int Order, bool Pad>
constexpr RhsIndices<Order, Pad> MakeRhsIndices()
{
    RhsIndices<Order, Pad> x{};
    constexpr int pitch = Pad ? (Order * 2 + 7) / 8 * 8 : Order * 2;
    for (int reverse = 0; reverse < 2; reverse++)
        for (int col = 0; col < 4; col++)
            for (int q = 0; q < MB; q++) {
                const int at = q < Order * 2 ? (reverse ? (Order - 1 - q / 2) * 2 + q % 2 : q) : 0;
                x.input[reverse][col * MB + q] = (col * pitch + at) * 4;
                if (q < pitch)
                    x.output[reverse][col * pitch + q] = (col * MB + at) * 4;
            }
    return x;
}
template <int Order, bool Pad>
__gm__ RhsIndices<Order, Pad> RhsIndices<Order, Pad>::value __attribute__((aligned(32))) = MakeRhsIndices<Order, Pad>();
template <int ORDER>
class Vector {
    LocalTensor<float> a0{TPosition::VECCALC, 0, AP}, a1{TPosition::VECCALC, AP * 4, AP},
        rr{TPosition::VECCALC, 9216, 24 * 64}, ii{TPosition::VECCALC, 15392, 24 * 64},
        idbuf{TPosition::VECCALC, 21536, 24 * 64};
    LocalTensor<int32_t> weightIdx{TPosition::VECCALC, 27680, WN}, ix{TPosition::VECCALC, 172064, CP},
        inputIdx{TPosition::VECCALC, 52256, BN}, outIdx{TPosition::VECCALC, 0, BN};
    LocalTensor<float> inputRaw{TPosition::VECCALC, 92192, BN}, inputPad{TPosition::VECCALC, 132128, BN};
    LocalTensor<float> cr{TPosition::VECCALC, 92192, CP}, ci{TPosition::VECCALC, 109856, CP},
        ni{TPosition::VECCALC, 127520, CP}, weightOut{TPosition::VECCALC, 39968, WN};
    GlobalTensor<float> wg, packedB, gout;
    GlobalTensor<uint64_t> aptrs, bptrs;
    int lda, ldb, rhs, matrixMode, aPitch, bPitch;
    template <bool Pack>
    __aicore__ inline void TransferRhs(int bt, int count, int tile, int tiles)
    {
        auto user = CtrsmGlobal<float>((__gm__ float*)bptrs.GetValue(bt) + uint64_t(tile) * NR * ldb * 2);
        const uint64_t offset = (uint64_t(bt) * tiles + tile) * BN;
        if constexpr (Pack) {
            CtrsmReadMatrix(inputRaw, user, count, ORDER, ldb, 0, true);
        } else {
            CtrsmFence<HardEvent::V_MTE2>();
            DataCopy(inputRaw, gout[offset], count * 48);
        }
        CtrsmFence<HardEvent::MTE2_V>();
        Gather(
            inputPad, inputRaw, (Pack ? inputIdx : outIdx).ReinterpretCast<uint32_t>(), 0,
            uint32_t(count * (Pack ? 48 : bPitch)));
        CtrsmFence<HardEvent::V_MTE3>();
        if constexpr (Pack)
            DataCopy(packedB[offset], inputPad, count * 48);
        else if (ldb == ORDER)
            CtrsmWrite(user, inputPad, 1, count * ORDER * 8);
        else
            CtrsmWrite(user, inputPad, count, ORDER * 8, 0, (ldb - ORDER) * 8);
        CtrsmFence<HardEvent::MTE3_V>();
    }
    template <bool Pack>
    __aicore__ inline void BuildRhsIndices(int count)
    {
        auto source = CtrsmGlobal<int32_t>(
            ldb == ORDER ? (__gm__ int32_t*)&RhsIndices<ORDER, false>::value :
                           (__gm__ int32_t*)&RhsIndices<ORDER, true>::value);
        CtrsmFence<HardEvent::V_MTE2>();
        const int seed = 4 * (Pack ? MB : bPitch), total = (count * (Pack ? MB : bPitch) + 7) / 8 * 8;
        auto dst = Pack ? inputIdx : outIdx;
        DataCopy(dst, source[(Pack ? 0 : 8 * MB) + (matrixMode >= 3 ? 4 * MB : 0)], total < seed ? total : seed);
        CtrsmFence<HardEvent::MTE2_V>();
        CtrsmRepeatIndices(dst, seed, total, (Pack ? bPitch : MB) * 16);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void BuildInverseIndices()
    {
        Duplicate(idbuf, 0.0f, 24 * 64);
        CtrsmFence<HardEvent::V_S>();
        for (int q = 0; q < 8; ++q) {
            weightIdx.SetValue(q, (q / 2 + (q % 2) * (24 * 64 + 8)) * 4);
            weightIdx.SetValue(64 + q, (q / 2 + ((q % 2) ^ 1) * (24 * 64 + 8)) * 4);
        }
        for (int q = 48; q < 64; ++q) {
            weightIdx.SetValue(q, 43 * 4);
            weightIdx.SetValue(64 + q, 43 * 4);
        }
        const int origin = matrixMode >= 3 ? (ORDER - 1) * (aPitch + 2) * 4 : 0, direction = matrixMode >= 3 ? -1 : 1;
        for (int q = 0; q < 8; ++q)
            ix.SetValue(q, origin + (q < 4 ? 0 : AP * 4));
        CtrsmFence<HardEvent::S_V>();
        for (int q = 8; q < 48; q += 8) {
            Adds(weightIdx[q], weightIdx, int32_t(q / 8 * 32), 8);
            Adds(weightIdx[64 + q], weightIdx[64], int32_t(q / 8 * 32), 8);
        }
        CtrsmRepeatIndices(weightIdx, 128, WN, 256);
        for (int row = 1; row < 24; ++row)
            Adds(
                ix[row * 8], ix,
                int32_t(direction * (row >= ORDER ? ORDER - 1 : row) * (matrixMode % 3 == 1 ? 2 : aPitch) * 4), 8);
        if (matrixMode >= 3) {
            for (int k = 1; k < 23; ++k) {
                const int column = k < ORDER ? k : ORDER - 1;
                Adds(ix[k * 192], ix, int32_t(-column * (matrixMode % 3 == 1 ? aPitch * 4 : 8)), 192);
            }
        } else {
            CtrsmRepeatIndices(ix, 192, CP, direction * (matrixMode % 3 == 1 ? aPitch * 4 : 8));
        }
        CtrsmFence<HardEvent::V_S>();
        for (int row = 0; row < ORDER; ++row) {
            idbuf.SetValue(row * 64 + (row / 4) * 8 + row % 4, 1.0f);
            idbuf.SetValue(row * 64 + (row / 4) * 8 + row % 4 + 4, 1.0f);
        }
        CtrsmFence<HardEvent::S_V>();
    }
    __aicore__ inline void LoadMatrixPair(int bt, int bt2)
    {
        GlobalTensor<float> ag;
        CtrsmFence<HardEvent::V_MTE2>();
        for (int slot = 0; slot < 2; slot++) {
            ag.SetGlobalBuffer((__gm__ float*)aptrs.GetValue(slot ? bt2 : bt));
            auto dst = slot ? a1 : a0;
            CtrsmReadMatrix(dst, ag, ORDER, ORDER, lda, 0, true);
        }
        CtrsmFence<HardEvent::MTE2_V>();
    }
    __aicore__ inline void ComputeInverse()
    {
        auto uix = ix.ReinterpretCast<uint32_t>();
        BinaryRepeatParams outer(1, 1, 0, 8, 0, 1);
        Gather(cr, a0, uix, 0, uint32_t(CP));
        Gather(ci, a0, uix, 4, uint32_t(CP));
        if (matrixMode % 3 == 2) {
            Muls(ci, ci, -1.0f, CP);
        }
        Muls(cr, cr, -1.0f, CP);
        Muls(ni, ci, -1.0f, CP);
        Muls(rr, idbuf, 1.0f, 24 * 64);
        Duplicate(ii, 0.0f, 24 * 64);
        AscendC::SetVectorMask<float>(0, ~0ULL);
#pragma unroll
        for (int k = 0; k < ORDER - 1; ++k) {
            const int row = k * 64, dst = (k + 1) * 64, coef = k * 192 + (k + 1) * 8;
            MulAddDst<float, float, false>(rr[dst], rr[row], cr[coef], ~0ULL, ORDER - 1 - k, outer);
            MulAddDst<float, float, false>(rr[dst], ii[row], ci[coef], ~0ULL, ORDER - 1 - k, outer);
            MulAddDst<float, float, false>(ii[dst], ii[row], cr[coef], ~0ULL, ORDER - 1 - k, outer);
            MulAddDst<float, float, false>(ii[dst], rr[row], ni[coef], ~0ULL, ORDER - 1 - k, outer);
        }
    }
    __aicore__ inline void PublishInverse(int bt, int bt2)
    {
        for (int slot = 0; slot < 2; slot++) {
            Gather(weightOut, rr, weightIdx.ReinterpretCast<uint32_t>(), slot * 16, uint32_t(WN));
            AscendC::SetVectorMask<float>(0, 0x0000AAAAAAAAAAAAULL);
            Muls<float, float, false>(weightOut, weightOut, -1.0f, 0x0000AAAAAAAAAAAAULL, 24, {1, 1, 16, 16});
            CtrsmFence<HardEvent::V_MTE3>();
            DataCopy(wg[(slot ? bt2 : bt) * WN], weightOut, WN);
            CtrsmFence<HardEvent::MTE3_V>();
        }
        ResetMask();
    }
    __aicore__ inline void PreparePair(int bt, int bt2)
    {
        LoadMatrixPair(bt, bt2);
        const int tiles = (rhs + NR - 1) / NR;
        for (int tile = 0; tile < tiles; ++tile) {
            const int remaining = rhs - tile * NR, count = remaining < NR ? remaining : NR;
            TransferRhs<true>(bt, count, tile, tiles);
            if (bt2 != bt) {
                CtrsmFence<HardEvent::V_MTE2>();
                TransferRhs<true>(bt2, count, tile, tiles);
            }
            CtrsmFence<HardEvent::V_MTE2>();
        }
        ComputeInverse();
        PublishInverse(bt, bt2);
    }
    __aicore__ inline void PrepareChunks(int n, int blk, int sub, int blocks)
    {
        for (int c = 0; c < CHUNKS; ++c) {
            const int pairs = (n + 1) / 2, p0 = c * pairs / CHUNKS, p1 = (c + 1) * pairs / CHUNKS,
                      pmid = p0 + (p1 - p0) / 2, plo = sub ? pmid : p0, phi = sub ? p1 : pmid;
            for (int p = plo; p < phi; ++p) {
                const int s = 2 * p, s2 = s + 1 < n ? s + 1 : s;
                PreparePair(blk + s * blocks, blk + s2 * blocks);
            }
            CrossCoreSetFlag<2, PIPE_MTE3>(static_cast<uint16_t>(c));
        }
    }
    __aicore__ inline void WriteBackChunks(int n, int blk, int sub, int blocks)
    {
        BuildRhsIndices<false>(rhs < NR ? rhs : NR);
        for (int c = 0; c < CHUNKS; ++c) {
            CrossCoreWaitFlag<2, PIPE_MTE2>(static_cast<uint16_t>(8 + c));
            int pairs = (n + 1) / 2;
            int s0 = 2 * (c * pairs / CHUNKS), s1 = 2 * ((c + 1) * pairs / CHUNKS);
            if (s1 > n)
                s1 = n;
            int mid = s0 + (s1 - s0) / 2;
            int lo = sub ? mid : s0, hi = sub ? s1 : mid;
            const int tiles = (rhs + NR - 1) / NR;
            for (int s = lo; s < hi; ++s) {
                for (int tile = 0; tile < tiles; ++tile) {
                    const int remaining = rhs - tile * NR;
                    TransferRhs<false>(blk + s * blocks, remaining < NR ? remaining : NR, tile, tiles);
                }
            }
        }
    }

public:
    __aicore__ inline void Run(
        GM_ADDR aGm, GM_ADDR wGm, GM_ADDR bGm, GM_ADDR oGm, GM_ADDR pGm, int batches, int columns, int aLd, int bLd,
        int mode)
    {
        lda = aLd;
        ldb = bLd;
        rhs = columns;
        matrixMode = mode;
        gout.SetGlobalBuffer((__gm__ float*)oGm);
        const int blocks = GetBlockNum(), gid = GetBlockIdx(), blk = gid / 2, sub = gid % 2;
        if (blocks == 0) {
            return;
        }
        const int n = blk < batches ? (batches - 1 - blk) / blocks + 1 : 0;
        wg.SetGlobalBuffer((__gm__ float*)wGm, uint64_t(batches) * WN);
        packedB.SetGlobalBuffer((__gm__ float*)pGm, uint64_t(batches) * ((rhs + NR - 1) / NR) * BN);
        aptrs.SetGlobalBuffer((__gm__ uint64_t*)aGm, uint64_t(batches));
        bptrs.SetGlobalBuffer((__gm__ uint64_t*)bGm, uint64_t(batches));
        aPitch = lda == ORDER ? ORDER * 2 : ((ORDER * 2 + 7) / 8) * 8;
        bPitch = ldb == ORDER ? ORDER * 2 : ((ORDER * 2 + 7) / 8) * 8;
        BuildRhsIndices<true>(rhs < NR ? rhs : NR);
        BuildInverseIndices();
        PrepareChunks(n, blk, sub, blocks);
        WriteBackChunks(n, blk, sub, blocks);
    }
};
__aicore__ inline void LoadCubeBatch(GM_ADDR wGm, GM_ADDR pGm, int bt, int slot, int rhs, uint64_t panel)
{
    auto wg = CtrsmGlobal<float>((__gm__ float*)wGm + bt * WN, WN);
    auto bg = CtrsmGlobal<float>((__gm__ float*)pGm + panel * BN, BN);
    LocalTensor<float> rawL1(TPosition::A1, 0, 2 * BN), wL1(TPosition::B1, 2 * BN * 4, 2 * MB * MB);
    DataCopy(rawL1[slot * BN], bg, CtrsmCube::Nd(rhs, MB, MB, NR));
    DataCopy(wL1[slot * MB * MB], wg, CtrsmCube::Nd(MB, MB, WLD, MB));
}
__aicore__ inline void ProcessCubeTiles(
    GM_ADDR wGm, GM_ADDR oGm, GM_ADDR pGm, int blk, int blocks, int lo, int hi, int rhs)
{
    LocalTensor<float> rawL1(TPosition::A1, 0, 2 * BN), wL1(TPosition::B1, 2 * BN * 4, 2 * MB * MB);
    LocalTensor<float> a2(TPosition::A2, 0, BN), b2(TPosition::B2, 0, MB * MB), c1(TPosition::CO1, 0, BN);
    const int tiles = (rhs + NR - 1) / NR;
    if (lo < hi) {
        LoadCubeBatch(wGm, pGm, blk + lo * blocks, 0, rhs < NR ? rhs : NR, uint64_t(blk + lo * blocks) * tiles);
    }
    for (int task = lo * tiles; task < hi * tiles; ++task) {
        const int bt = blk + (task / tiles) * blocks, tile = task % tiles, cur = (task - lo * tiles) & 1;
        CtrsmFence<HardEvent::MTE2_MTE1>();
        LoadData2DParams lb;
        lb.repeatTimes = 18;
        lb.srcStride = 1;
        LoadData(a2, rawL1[cur * BN], CtrsmCube::Load3d(NR, MB, MB, NR));
        LoadData(b2, wL1[cur * MB * MB], lb);
        CtrsmFence<HardEvent::MTE1_MTE2>();
        if (task + 1 < hi * tiles) {
            const int nextBatch = blk + ((task + 1) / tiles) * blocks, nextTile = (task + 1) % tiles,
                      remaining = rhs - nextTile * NR;
            LoadCubeBatch(
                wGm, pGm, nextBatch, cur ^ 1, remaining < NR ? remaining : NR, uint64_t(nextBatch) * tiles + nextTile);
        }
        auto og = CtrsmGlobal<float>((__gm__ float*)oGm + (uint64_t(bt) * tiles + tile) * BN, BN);
        auto mm = CtrsmCube::Mmad(NR, MB, MB, true);
        auto fx = CtrsmCube::Fix(rhs - tile * NR < NR ? rhs - tile * NR : NR, MB, NR, MB);
        CtrsmFence<HardEvent::MTE1_M>();
        Mmad(c1, a2, b2, mm);
        CtrsmFence<HardEvent::M_FIX>();
        Fixpipe(og, c1, fx);
        CtrsmFence<HardEvent::FIX_M>();
        CtrsmFence<HardEvent::M_MTE1>();
    }
}
__aicore__ inline void RunCube(GM_ADDR wGm, GM_ADDR oGm, GM_ADDR pGm, int batches, int rhs)
{
    const int blocks = GetBlockNum(), blk = GetBlockIdx();
    if (blocks == 0) {
        return;
    }
    const int n = blk < batches ? (batches - 1 - blk) / blocks + 1 : 0;
    InitSocState();
    SetHF32Mode(false);
    for (int c = 0; c < CHUNKS; ++c) {
        CrossCoreWaitFlag<2, PIPE_MTE2>(static_cast<uint16_t>(c));
        int pairs = (n + 1) / 2;
        int lo = 2 * (c * pairs / CHUNKS), hi = 2 * ((c + 1) * pairs / CHUNKS);
        if (hi > n)
            hi = n;
        ProcessCubeTiles(wGm, oGm, pGm, blk, blocks, lo, hi, rhs);
        CrossCoreSetFlag<2, PIPE_FIX>(static_cast<uint16_t>(8 + c));
    }
}
template <int ORDER>
__global__ __aicore__ void direct_aos(
    GM_ADDR aGm, GM_ADDR wGm, GM_ADDR bGm, int batches, int rhs, int lda, int ldb, int matrixMode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR oGm = wGm + uint64_t(batches) * WN * sizeof(float);
    GM_ADDR pGm = oGm + uint64_t(batches) * ((rhs + NR - 1) / NR) * BN * sizeof(float);
    if ASCEND_IS_AIV {
        Vector<ORDER> vector;
        vector.Run(aGm, wGm, bGm, oGm, pGm, batches, rhs, lda, ldb, matrixMode);
    }
    if ASCEND_IS_AIC {
        RunCube(wGm, oGm, pGm, batches, rhs);
    }
    if ASCEND_IS_AIV {
        ResetMask();
    }
    if ASCEND_IS_AIV {
        CtrsmFence<HardEvent::V_S>();
    } else {
        CtrsmFence<HardEvent::FIX_S>();
    }
}
template <int ORDER>
void DispatchOrder(
    void* a, void* b, void* w, int batches, int order, int rhs, int lda, int ldb, int cores, void* stream,
    int matrixMode)
{
    if (order == ORDER) {
        direct_aos<ORDER>
            <<<cores, nullptr, stream>>>((GM_ADDR)a, (GM_ADDR)w, (GM_ADDR)b, batches, rhs, lda, ldb, matrixMode);
    } else if constexpr (ORDER > 1) {
        DispatchOrder<ORDER - 1>(a, b, w, batches, order, rhs, lda, ldb, cores, stream, matrixMode);
    }
}
} // namespace CtrsmBatchedSmallInverse
namespace CtrsmBatchedDirectRhs {
template <int R, bool Reverse, bool Swap, bool Conjugate>
class FiveBatch {
    static constexpr int P = 44, G = 4, CH = 8 / G, W = 2 * R, L = G * W + 8, A = 2 * P * P, X = P * L,
                         T = P * (P - 1) / 2 * 8, C = (A * G + T > 3 * X ? A * G + T : 3 * X);
    LocalTensor<float> rawA{TPosition::VECCALC, 2 * T * 4, A* G};
    LocalTensor<int32_t> seed{TPosition::VECCALC, 0, P * 8}, ia{TPosition::VECCALC, (2 * T + A * G) * 4, T};
    LocalTensor<float> cr{TPosition::VECCALC, 0, T}, ci{TPosition::VECCALC, T * 4, T}, rawB{TPosition::VECCALC, 0, X},
        x{TPosition::VECCALC, X * 4, X};
    LocalTensor<int32_t> ix{TPosition::VECCALC, 2 * X * 4, X}, sw{TPosition::VECCALC, 2 * X * 4, L};
    LocalTensor<float> tmp{TPosition::VECCALC, (2 * X + L) * 4, L};
    static_assert((C + 2 * T) * 4 <= 192 * 1024, "grouped complex UB");
    static constexpr int TW = 64, TAILPAIR = 1104, TAILTRI = TAILPAIR * 8, TRBASE = 2 * T + A,
                         TXBASE = TRBASE + 2 * TAILTRI, GXBASE = TXBASE + P * TW;
    LocalTensor<float> tx{TPosition::VECCALC, TXBASE * 4, P* TW}, tr{TPosition::VECCALC, TRBASE * 4, TAILTRI},
        ti{TPosition::VECCALC, (TRBASE + TAILTRI) * 4, TAILTRI};
    LocalTensor<float> tswap{TPosition::VECCALC, (2 * T + L * 2) * 4, TW};
    LocalTensor<int32_t> tailSwap{TPosition::VECCALC, (2 * T + L * 2 + TW) * 4, TW};
    static_assert((GXBASE + X) * 4 <= 192 * 1024, "five batch complete live data");
    GlobalTensor<uint64_t> ap, bp;
    int k, rhs, lda, ldb, pitch;
    static constexpr bool reverse = Reverse, swap = Swap, conj = Conjugate;
    __aicore__ inline void CoefficientIndices()
    {
        CtrsmFence<HardEvent::V_S>();
        for (int q = 0; q < 8; q++)
            seed.SetValue(q, (q / CH) * A * 4);
        CtrsmFence<HardEvent::S_V>();
        CtrsmRepeatIndices(seed, 8, k * 8, (swap ? pitch : 2) * 4);
        PipeBarrier<PIPE_V>();
        int at = 0;
        for (int i = 0; i < k - 1; i++) {
            Adds(
                ia[at], seed[(reverse ? 0 : i + 1) * 8], int32_t((reverse ? k - 1 - i : i) * (swap ? 2 : pitch) * 4),
                (k - i - 1) * 8);
            at += (k - i - 1) * 8;
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void InputIndices()
    {
        CtrsmFence<HardEvent::V_S>();
        for (int q = 0; q < 8; q++)
            ix.SetValue(q, ((q / CH) * P * W + q % CH) * 4);
        CtrsmFence<HardEvent::S_V>();
        for (int q = 8; q < L; q += 8)
            Adds(ix[q], ix, int32_t(q / 8 * CH * 4), 8);
        CtrsmRepeatIndices(ix, L, k * L, (ldb == rhs ? rhs * 2 : W) * 4);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void OutputIndices()
    {
        CtrsmFence<HardEvent::V_S>();
        const int step = ldb == rhs ? rhs * 2 : W, total = (k * step + 7) / 8 * 8;
        CtrsmIndexTile<1>(ix, total, 0, 1);
        LocalTensor<float> quot = rawB;
        Cast(quot, ix, RoundMode::CAST_NONE, total);
        Muls(quot, quot, 1.f / step, total);
        Adds(quot, quot, 0.0001f, total);
        Cast(rawB.ReinterpretCast<int32_t>(), quot, RoundMode::CAST_FLOOR, total);
        Muls(rawB.ReinterpretCast<int32_t>(), rawB.ReinterpretCast<int32_t>(), int32_t(L - step * 4), total);
        CtrsmIndexTile<CH>(ix, total, 1, 8);
        Add(ix, ix, rawB.ReinterpretCast<int32_t>(), total);
        Muls(ix, ix, int32_t(4), total);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Load(int first, int count)
    {
        CtrsmFence<HardEvent::V_MTE2>();
        for (int b = 0; b < G; b++) {
            auto ga = CtrsmGlobal<float>((__gm__ float*)ap.GetValue(first + (b < count ? b : 0)));
            CtrsmReadMatrix(rawA[b * A], ga, k, k, lda, 0, false);
        }
        CoefficientIndices();
        CtrsmFence<HardEvent::MTE2_V>();
        const int total = k * (k - 1) / 2 * 8;
        CtrsmGather(cr, rawA, ia, total);
        CtrsmGather(ci, rawA, ia, total, 4);
        Muls(cr, cr, -1.f, total);
        AscendC::SetVectorMask<float>(0, conj ? 0x5555555555555555ULL : 0xAAAAAAAAAAAAAAAAULL);
        Muls<float, float, false>(ci, ci, -1.f, 64, (total + 63) / 64, {1, 1, 8, 8});
        ResetMask();
        PipeBarrier<PIPE_V>();
        rawB = LocalTensor<float>{TPosition::VECCALC, 2 * T * 4, X};
        x = LocalTensor<float>{TPosition::VECCALC, GXBASE * 4, X};
        ix = LocalTensor<int32_t>{TPosition::VECCALC, (2 * T + X) * 4, X};
        CtrsmFence<HardEvent::V_MTE2>();
        for (int b = 0; b < G; b++) {
            auto gb = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(first + (b < count ? b : 0)));
            CtrsmReadMatrix(rawB[b * P * W], gb, k, rhs, ldb, W, false);
        }
        InputIndices();
        CtrsmFence<HardEvent::MTE2_V>();
        CtrsmGather(x, rawB, ix, k * L);
        PipeBarrier<PIPE_V>();
        sw = LocalTensor<int32_t>{TPosition::VECCALC, 2 * T * 4, L};
        tmp = LocalTensor<float>{TPosition::VECCALC, (2 * T + L) * 4, L};
    }
    __aicore__ inline void TailLoad(int batch)
    {
        LocalTensor<float> ta{TPosition::VECCALC, TRBASE * 4, A};
        LocalTensor<int32_t> tidx{TPosition::VECCALC, 2 * T * 4, TAILPAIR},
            tseed{TPosition::VECCALC, (2 * T + TAILPAIR) * 4, P};
        LocalTensor<float> gathered{TPosition::VECCALC, (2 * T + TAILPAIR + ((P + 7) / 8 * 8)) * 4, TAILPAIR * 2};
        static_assert(3 * TAILPAIR + ((P + 7) / 8 * 8) <= A, "tail gather scratch below coefficient output");
        auto ga = CtrsmGlobal<float>((__gm__ float*)ap.GetValue(batch));
        auto gb = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(batch));
        CtrsmFence<HardEvent::V_MTE2>();
        CtrsmReadMatrix(ta, ga, k, k, lda, 0, false);
        CtrsmRead(tx, gb, k, rhs * 8, (ldb - rhs) * 8, (TW - (rhs * 2 + 7) / 8 * 8) / 8);
        Duplicate(tidx, int32_t(0), TAILPAIR);
        CtrsmTriangleIndices<false>(tidx, tseed, k, (swap ? pitch : 2) * 4, (swap ? 2 : pitch) * 4, reverse, false);
        const int at = ((k - 1) / 8 + 1) * ((k - 1) / 8 * 32 + (k - 1) % 8 * 8);
        CtrsmFence<HardEvent::MTE2_V>();
        Gather(gathered, ta, tidx.ReinterpretCast<uint32_t>(), 0, at);
        Gather(gathered[TAILPAIR], ta, tidx.ReinterpretCast<uint32_t>(), 4, at);
        PipeBarrier<PIPE_V>();
        Muls(gathered, gathered, -1.f, at);
        Brcb(tr, gathered, at / 8, {1, 8});
        Brcb(ti, gathered[TAILPAIR], at / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        LocalTensor<float> sg{TPosition::VECCALC, (2 * T + L * 2 + 2 * TW) * 4, 8};
        CtrsmFence<HardEvent::V_S>();
        for (int q = 0; q < 8; q++)
            sg.SetValue(q, (q % 2 ? -1.f : 1.f) * (conj ? -1.f : 1.f));
        CtrsmFence<HardEvent::S_V>();
        Mul(ti, ti, sg, 64, at / 8, {1, 1, 0, 8, 8, 0});
        CtrsmSwapIndices(tailSwap, TW);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Solve(bool tail)
    {
        AscendC::SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        int at = 0, tailAt = 0;
        const BinaryRepeatParams bc(1, 1, 0, L / 8, 0, 1), tc(1, 1, 0, TW / 8, 0, 1);
        for (int i = 0; i < k - 1; i++) {
            const int row = reverse ? k - 1 - i : i, rows = k - i - 1, target = reverse ? 0 : row + 1;
            Gather(tmp, x[row * L], sw.ReinterpretCast<uint32_t>(), 0, L);
#pragma unroll
            for (int q = 0; q < G * W; q += 64) {
                const int lanes = G * W - q < 64 ? G * W - q : 64;
                MulAddDst<float, float, false>(x[target * L + q], x[row * L + q], cr[at], lanes, rows, bc);
                MulAddDst<float, float, false>(x[target * L + q], tmp[q], ci[at], lanes, rows, bc);
            }
            if (tail) {
                Gather(tswap, tx[row * TW], tailSwap.ReinterpretCast<uint32_t>(), 0, TW);
                MulAddDst<float, float, false>(tx[target * TW], tx[row * TW], tr[tailAt], TW, rows, tc);
                MulAddDst<float, float, false>(tx[target * TW], tswap, ti[tailAt], TW, rows, tc);
                tailAt += (rows + 7) / 8 * 64;
            }
            PipeBarrier<PIPE_V>();
            at += rows * 8;
        }
        ResetMask();
    }
    __aicore__ inline void Store(int first, int count)
    {
        OutputIndices();
        for (int b = 0; b < count; b++)
            Gather(
                rawB[b * P * W], x, ix.ReinterpretCast<uint32_t>(), uint32_t(b * CH * 4),
                (k * (ldb == rhs ? rhs * 2 : W) + 7) / 8 * 8);
        CtrsmFence<HardEvent::V_MTE3>();
        for (int b = 0; b < count; b++) {
            auto gb = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(first + b));
            if (ldb == rhs)
                CtrsmWrite(gb, rawB[b * P * W], 1, k * rhs * 8);
            else
                CtrsmWrite(gb, rawB[b * P * W], k, rhs * 8, (W - ((rhs * 2 + 7) / 8) * 8) / 8, (ldb - rhs) * 8);
        }
        CtrsmFence<HardEvent::MTE3_V>();
        CtrsmFence<HardEvent::V_MTE2>();
    }

public:
    __aicore__ inline void Run(
        GM_ADDR aa, GM_ADDR bb, int batches, int order, int nrhs, int aLd, int bLd, int upper, int trans)
    {
        k = order;
        rhs = nrhs;
        lda = aLd;
        ldb = bLd;
        pitch = lda == k ? 2 * k : (2 * k + 7) / 8 * 8;
        ap.SetGlobalBuffer((__gm__ uint64_t*)aa);
        bp.SetGlobalBuffer((__gm__ uint64_t*)bb);
        for (int first = GetBlockIdx() * 5; first < batches; first += GetBlockNum() * 5) {
            const int count = batches - first < 5 ? batches - first : 5;
            rawA = LocalTensor<float>{TPosition::VECCALC, 2 * T * 4, A * G};
            seed = LocalTensor<int32_t>{TPosition::VECCALC, 0, P * 8};
            ia = LocalTensor<int32_t>{TPosition::VECCALC, (2 * T + A * G) * 4, T};
            cr = LocalTensor<float>{TPosition::VECCALC, 0, T};
            ci = LocalTensor<float>{TPosition::VECCALC, T * 4, T};
            Load(first, count < 4 ? count : 4);
            if (count == 5)
                TailLoad(first + 4);
            CtrsmSwapIndices(sw, L);
            Solve(count == 5);
            Store(first, count < 4 ? count : 4);
            if (count == 5) {
                auto gb = CtrsmGlobal<float>((__gm__ float*)bp.GetValue(first + 4));
                CtrsmFence<HardEvent::V_MTE3>();
                CtrsmWrite(gb, tx, k, rhs * 8, (TW - (rhs * 2 + 7) / 8 * 8) / 8, (ldb - rhs) * 8);
                CtrsmFence<HardEvent::MTE3_V>();
            }
        }
        ResetMask();
        CtrsmFence<HardEvent::V_S>();
    }
};
template <int R, bool Reverse, bool Swap, bool Conjugate>
__global__ __aicore__ void five_batch_rhs(
    GM_ADDR a, GM_ADDR b, int batches, int k, int rhs, int lda, int ldb, int upper, int trans)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FiveBatch<R, Reverse, Swap, Conjugate> kernel;
    kernel.Run(a, b, batches, k, rhs, lda, ldb, upper, trans);
}
template <int R, int Mode = 5>
void LaunchFiveOriented(
    void* a, void* b, int batches, int k, int rhs, int lda, int ldb, int upper, int trans, int cores, void* stream)
{
    const int mode = (trans == 0 ? 0 : trans == 2 ? 4 : 2) + upper;
    if (mode == Mode)
        five_batch_rhs < R, bool(Mode & 1) != bool(Mode < 2),
            Mode<2, Mode >= 4>
            <<<cores, nullptr, stream>>>((GM_ADDR)a, (GM_ADDR)b, batches, k, rhs, lda, ldb, upper, trans);
    else if constexpr (Mode > 0)
        LaunchFiveOriented<R, Mode - 1>(a, b, batches, k, rhs, lda, ldb, upper, trans, cores, stream);
}
} // namespace CtrsmBatchedDirectRhs
void LaunchCtrsmBatchedZero(void* b, int m, int n, int ldb, int batches, int cores, void* stream)
{
    ctrsmbatched_zero<<<cores, nullptr, stream>>>((GM_ADDR)b, m, n, ldb, batches);
}
template <int Mode = 7>
void LaunchPlanarSquare(void* a, void* b, int order, int n, int upper, int cores, void* stream)
{
    const int mode = (order == 16 ? 4 : 0) + (n > cores * 8 ? 2 : 0) + upper;
    constexpr int K = Mode & 4 ? 16 : 8, G = Mode & 2 ? (K == 8 ? 32 : 16) : 8;
    if (mode == Mode)
        CtrsmBatchedPlanarSquare::planar_square<K, bool(Mode & 1), G>
            <<<cores, nullptr, stream>>>((GM_ADDR)a, (GM_ADDR)b, n);
    else if constexpr (Mode > 0)
        LaunchPlanarSquare<Mode - 1>(a, b, order, n, upper, cores, stream);
}
void LaunchCtrsmBatchedPlanarSquare(void* a, void* b, int order, int n, int upper, int cores, void* stream)
{
    LaunchPlanarSquare(a, b, order, n, upper, cores, stream);
}
void LaunchCtrsmBatchedHalfDiagonalSmall(void* a, void* b, void* w, int n, int upper, int cores, void* s)
{
    using namespace CtrsmBatchedSmallMixed;
    if (n <= cores * 4) {
        if (upper)
            paired_mixed<true, true><<<cores, nullptr, s>>>((GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)w, n, cores);
        else
            paired_mixed<false, true><<<cores, nullptr, s>>>((GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)w, n, cores);
    } else if (upper)
        paired_mixed<true><<<cores, nullptr, s>>>((GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)w, n, cores);
    else
        paired_mixed<false><<<cores, nullptr, s>>>((GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)w, n, cores);
}
void LaunchCtrsmBatchedSmallInverse(
    void* a, void* b, void* w, int batches, int order, int rhs, int lda, int ldb, int cores, void* stream,
    int matrixMode)
{
    using namespace CtrsmBatchedSmallInverse;
    DispatchOrder<24>(a, b, w, batches, order, rhs, lda, ldb, cores, stream, matrixMode);
}
void LaunchCtrsmBatchedDirectRhs(
    void* a, void* b, int batches, int k, int rhs, int lda, int ldb, int side, int upper, int trans, int diag,
    int cores, void* stream)
{
    using namespace CtrsmBatchedDirectRhs;
    if (rhs <= 16)
        LaunchFiveOriented<16>(a, b, batches, k, rhs, lda, ldb, upper, trans, cores, stream);
    else
        LaunchFiveOriented<24>(a, b, batches, k, rhs, lda, ldb, upper, trans, cores, stream);
}
