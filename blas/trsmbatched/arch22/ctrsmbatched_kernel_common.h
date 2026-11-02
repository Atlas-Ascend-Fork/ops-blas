/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>
#include "kernel_operator.h"
template <AscendC::HardEvent E>
__aicore__ inline void CtrsmFence(uint8_t event = 0)
{
    AscendC::SetFlag<E>(event);
    AscendC::WaitFlag<E>(event);
} // Copy strides retain the CANN units: GM bytes and UB 32-byte blocks.
template <typename T>
__aicore__ inline void CtrsmRead(
    const AscendC::LocalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, int rows, int bytes, int srcStride = 0,
    int dstStride = 0, bool pad = false)
{
    AscendC::DataCopyPad(
        dst, src, {uint16_t(rows), uint32_t(bytes), uint32_t(srcStride), uint32_t(dstStride), 0}, {pad, 0, 0, 0});
}
template <typename T>
__aicore__ inline void CtrsmWrite(
    const AscendC::GlobalTensor<T>& dst, const AscendC::LocalTensor<T>& src, int rows, int bytes, int srcStride = 0,
    int dstStride = 0)
{
    AscendC::DataCopyPad(dst, src, {uint16_t(rows), uint32_t(bytes), uint32_t(srcStride), uint32_t(dstStride), 0});
}
__aicore__ inline void CtrsmReadMatrix(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, int rows, int cols, int ld, int pitch = 0,
    bool pad = false)
{
    if (ld == cols)
        CtrsmRead(dst, src, 1, rows * cols * 8, 0, 0, pad);
    else
        CtrsmRead(dst, src, rows, cols * 8, (ld - cols) * 8, pitch ? (pitch - (cols * 2 + 7) / 8 * 8) / 8 : 0, pad);
}
__aicore__ inline void CtrsmGather(
    const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<int32_t>& ix, int count, uint32_t base = 0)
{ // The installed AscendC::Gather lowers the repeat field to eight bits.
    for (int offset = 0; offset < count; offset += 8192) {
        const int length = count - offset < 8192 ? count - offset : 8192;
        AscendC::Gather(dst[offset], src, ix[offset].ReinterpretCast<uint32_t>(), base, length);
    }
}
__aicore__ inline void CtrsmRepeatIndices(AscendC::LocalTensor<int32_t> index, int seed, int count, int step)
{
    for (int q = seed, delta = step; q < count; q *= 2, delta *= 2)
        AscendC::Adds(index[q], index, int32_t(delta), q < count - q ? q : count - q);
}
template <int Lanes>
__aicore__ inline void CtrsmIndexTile(
    AscendC::LocalTensor<int32_t> index, int count, int inner, int outer, int origin = 0)
{
    static_assert(Lanes > 0 && 8 % Lanes == 0, "index tile lanes");
    CtrsmFence<AscendC::HardEvent::V_S>();
    for (int q = 0; q < 8; q++)
        index.SetValue(q, origin + (q / Lanes) * outer + (q % Lanes) * inner);
    CtrsmFence<AscendC::HardEvent::S_V>();
    CtrsmRepeatIndices(index, 8, count, 8 / Lanes * outer);
    AscendC::PipeBarrier<PIPE_V>();
}
__aicore__ inline void CtrsmSwapIndices(AscendC::LocalTensor<int32_t> index, int count)
{
    CtrsmIndexTile<2>(index, count, -4, 8, 4);
}
__aicore__ inline int CtrsmTriangleSize(int k) { return ((k - 1) / 8 + 1) * ((k - 1) / 8 * 32 + (k - 1) % 8 * 8); }
template <bool Clear>
__aicore__ inline void CtrsmTriangleIndices(
    AscendC::LocalTensor<int32_t> dst, AscendC::LocalTensor<int32_t> seed, int k, int rowStep, int colStep,
    bool reverse, bool ordered)
{
    const int origin = reverse && ordered ? (k - 1) * rowStep : 0, step = reverse && ordered ? -rowStep : rowStep;
    CtrsmFence<AscendC::HardEvent::V_S>();
    for (int q = 0; q < 8; q++)
        seed.SetValue(q, origin + q * step);
    CtrsmFence<AscendC::HardEvent::S_V>();
    CtrsmRepeatIndices(seed, 8, (k + 7) / 8 * 8, 8 * step);
    AscendC::PipeBarrier<PIPE_V>();
    if (Clear && k > 1)
        AscendC::Duplicate(dst, int32_t(0), CtrsmTriangleSize(k));
    int at = 0;
    for (int i = 0; i < k - 1; i++) {
        const int rows = k - i - 1, n = (rows + 7) / 8 * 8;
        AscendC::Adds(
            dst[at], seed, int32_t((ordered || !reverse ? i + 1 : 0) * step + (reverse ? k - 1 - i : i) * colStep),
            rows);
        at += n;
    }
    AscendC::PipeBarrier<PIPE_V>();
}
template <typename T>
__aicore__ inline AscendC::GlobalTensor<T> CtrsmGlobal(__gm__ T* address, uint64_t count = 0)
{
    AscendC::GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(address, count);
    return tensor;
}
namespace CtrsmCube {
__aicore__ inline AscendC::Nd2NzParams Nd(
    int rows, int cols, int srcPitch, int dstPitch, int matrices = 1, int srcMatrix = 0, int dstMatrix = 0)
{
    return AscendC::Nd2NzParams(matrices, rows, cols, srcMatrix, srcPitch, dstPitch, 1, dstMatrix);
}
template <typename T = float>
__aicore__ inline AscendC::LoadData3DParamsV2<T> Load3d(
    int width, int channels, int k, int m, bool transpose = false, int kStart = 0, int mStart = 0)
{
    AscendC::LoadData3DParamsV2<T> x;
    x.l1H = 1;
    x.l1W = width;
    x.channelSize = channels;
    x.kExtension = k;
    x.mExtension = m;
    x.kStartPt = kStart;
    x.mStartPt = mStart;
    x.enTranspose = transpose;
    return x;
}
__aicore__ inline AscendC::MmadParams Mmad(int m, int n, int k, bool first = true)
{
    return AscendC::MmadParams(m, n, k, 0, false, first);
}
__aicore__ inline AscendC::FixpipeParamsV220 Fix(int m, int n, int srcPitch, int dstPitch)
{
    return AscendC::FixpipeParamsV220(n, m, srcPitch, dstPitch, false);
}
} // namespace CtrsmCube
template <int K_, int G_, int A_, int XPad, int RPad, int IPad, int IX_>
class CtrsmPlanar {
protected:
    static constexpr int K = K_, G = G_, A = A_, L = G * K, S = L + 8, X = K * S, T = K * (K - 1) / 2 * G, D = K * G,
                         IX = IX_;
    AscendC::LocalTensor<float> raw{AscendC::TPosition::VECCALC, 0, A* G}, xr{AscendC::TPosition::VECCALC, A* G * 4, X},
        xi{AscendC::TPosition::VECCALC, (A * G + X + XPad) * 4, X};
    AscendC::LocalTensor<float> cr{AscendC::TPosition::VECCALC, (A * G + 2 * X + RPad) * 4, T},
        ci{AscendC::TPosition::VECCALC, (A * G + 2 * X + T + IPad) * 4, T};
    static constexpr int BASE = A * G + 2 * X + 2 * T + IPad;
    AscendC::LocalTensor<int32_t> idx{AscendC::TPosition::VECCALC, BASE * 4, IX};
    AscendC::LocalTensor<float> dr{AscendC::TPosition::VECCALC, (BASE + IX) * 4, D},
        di{AscendC::TPosition::VECCALC, (BASE + IX + D) * 4, D};
    AscendC::LocalTensor<float> den{AscendC::TPosition::VECCALC, (BASE + IX + 2 * D) * 4, D},
        tmp{AscendC::TPosition::VECCALC, (BASE + IX + 3 * D) * 4, 2 * L};
    static_assert((BASE + IX + 3 * D + 2 * L) * 4 <= 192 * 1024, "planar UB");
    __aicore__ inline void Reciprocal()
    {
        AscendC::Mul(den, dr, dr, D);
        AscendC::Mul(tmp, di, di, D);
        AscendC::Add(den, den, tmp, D);
        AscendC::Div(dr, dr, den, D);
        AscendC::Div(di, di, den, D);
    }
    __aicore__ inline void Normalize(const AscendC::LocalTensor<uint32_t>& index)
    {
        auto rr = raw, ii = raw[T], tr = raw[2 * T], ti = raw[3 * T];
        AscendC::Gather(rr, dr, index, 0, T);
        AscendC::Gather(ii, di, index, 0, T);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tr, cr, rr, T);
        AscendC::MulAddDst<float, float>(tr, ci, ii, T);
        AscendC::Mul(ti, ci, rr, T);
        AscendC::Muls(ii, ii, -1.f, T);
        AscendC::MulAddDst<float, float>(ti, cr, ii, T);
        AscendC::Muls(cr, tr, -1.f, T);
        AscendC::Muls(ci, ti, 1.f, T);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Identity()
    {
        AscendC::Duplicate(xr, 0.f, X);
        AscendC::Duplicate(xi, 0.f, X);
        AscendC::Muls(den, di, -1.f, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(xr, dr, {K, 1, 0, uint16_t(S / 8)});
        AscendC::DataCopy(xi, den, {K, 1, 0, uint16_t(S / 8)});
        AscendC::PipeBarrier<PIPE_V>();
    }
    template <bool Upper, bool Inverse>
    __aicore__ inline void RowSweep()
    {
        AscendC::Muls(raw, ci, -1.f, T);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(0, ~0ULL);
        int at = 0;
        for (int i = 0; i < K - 1; i++) {
            const int p = Upper ? K - 1 - i : i, start = Upper ? 0 : p + 1, rows = K - 1 - i;
#pragma unroll
            for (int q = Inverse && Upper ? (p / 8) * 64 : 0; q < (Inverse && !Upper ? (p / 8 + 1) * 64 : L); q += 64) {
                const int coef = at + (Inverse ? 0 : q / (8 * K) * 8);
                const AscendC::BinaryRepeatParams rp(1, 1, 0, S / 8, 0, G / 8);
                AscendC::MulAddDst<float, float, false>(xr[start * S + q], xr[p * S + q], cr[coef], 64, rows, rp);
                AscendC::MulAddDst<float, float, false>(xr[start * S + q], xi[p * S + q], ci[coef], 64, rows, rp);
                AscendC::MulAddDst<float, float, false>(xi[start * S + q], xi[p * S + q], cr[coef], 64, rows, rp);
                AscendC::MulAddDst<float, float, false>(xi[start * S + q], xr[p * S + q], raw[coef], 64, rows, rp);
            }
            AscendC::PipeBarrier<PIPE_V>();
            at += rows * G;
        }
        AscendC::ResetMask();
    }
    template <bool Upper, int I = 0, int Q = 0>
    __aicore__ inline void ColumnSweep()
    {
        constexpr int p = Upper ? K - 1 - I : I, start = Upper ? 0 : p + 1, rows = K - 1 - I,
                      at = I * (2 * K - I - 1) / 2 * G, lanes = rows * G - Q < 64 ? rows * G - Q : 64,
                      count = Upper ? K - p : p + 1, col = Upper ? p : 0, src = col * S + p * G,
                      dst = col * S + start * G + Q;
        AscendC::SetVectorMask<float>(0, lanes == 64 ? ~0ULL : ((1ULL << lanes) - 1));
        AscendC::MulAddDst<float, float, false>(xr[dst], cr[at + Q], xr[src], 64, count, {1, 1, 0, S / 8, 0, S / 8});
        AscendC::MulAddDst<float, float, false>(xr[dst], ci[at + Q], xi[src], 64, count, {1, 1, 0, S / 8, 0, S / 8});
        AscendC::MulAddDst<float, float, false>(xi[dst], cr[at + Q], xi[src], 64, count, {1, 1, 0, S / 8, 0, S / 8});
        AscendC::MulAddDst<float, float, false>(xi[dst], raw[at + Q], xr[src], 64, count, {1, 1, 0, S / 8, 0, S / 8});
        if constexpr (Q + 64 < rows * G) {
            ColumnSweep<Upper, I, Q + 64>();
        } else {
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (I + 1 < K - 1)
                ColumnSweep<Upper, I + 1>();
        }
    }
};
namespace CtrsmBatchedPlanarSquare {
template <int K, int G, bool Inverse = false, bool Swap = false>
struct IndexTable {
    static constexpr int A = K * K * 2 + (Inverse ? 32 : 8), S = K * G + 8, D = K * G, T = K * (K - 1) / 2 * G,
                         X = K * S, O = Inverse ? 4 * K * K : (K == 8 ? A * G : 2 * K * K);
    uint32_t diagonal[D], coefficient[T], normalize[T], rhs[Inverse ? 8 : X], output[O];
};
template <int K, int G, bool Inverse = false, bool Swap = false>
constexpr IndexTable<K, G, Inverse, Swap> MakeIndexTable(bool upper)
{
    IndexTable<K, G, Inverse, Swap> x{};
    constexpr int A = IndexTable<K, G, Inverse, Swap>::A, S = IndexTable<K, G, Inverse, Swap>::S,
                  X = IndexTable<K, G, Inverse, Swap>::X;
    for (int r = 0; r < K; r++)
        for (int b = 0; b < G; b++)
            x.diagonal[r * G + b] = (b * A + r * (2 * K + 2)) * 4;
    int at = 0;
    for (int i = 0; i < K - 1; i++) {
        const int p = upper ? K - 1 - i : i, start = upper ? 0 : p + 1, rows = K - 1 - i;
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < G; b++) {
                x.coefficient[at] = (b * A + (Swap ? (start + r) * K + p : p * K + start + r) * 2) * 4;
                x.normalize[at] = ((start + r) * G + b) * 4;
                at++;
            }
    }
    if constexpr (Inverse) {
        for (int r = 0; r < 2 * K; r++)
            for (int c = 0; c < 2 * K; c++)
                x.output[r * 2 * K + c] = ((r / 2) * S + (c / 2) * G + ((r ^ c) % 2) * X) * 4;
    } else {
        for (int r = 0; r < K; r++)
            for (int j = 0; j < K * G; j++)
                x.rhs[r * S + j] = ((j / (K * 8) * 8 + j % 8) * A + 2 * (j / 8 % K * K + r)) * 4;
        for (int b = 0; b < (K == 8 ? G : 1); b++)
            for (int j = 0; j < 2 * K * K; j++)
                x.output[b * A + j] = (j / 2 % K * S + (b / 8) * K * 8 + j / (2 * K) * 8 + b % 8 + j % 2 * X) * 4;
    }
    return x;
}
template <int K, int G, bool Upper, bool Inverse = false, bool Swap = false>
struct Indices {
    static __gm__ IndexTable<K, G, Inverse, Swap> value;
};
template <int K, int G, bool Upper, bool Inverse, bool Swap>
__gm__ IndexTable<K, G, Inverse, Swap> Indices<K, G, Upper, Inverse, Swap>::value __attribute__((aligned(32))) =
    MakeIndexTable<K, G, Inverse, Swap>(Upper);
} // namespace CtrsmBatchedPlanarSquare
