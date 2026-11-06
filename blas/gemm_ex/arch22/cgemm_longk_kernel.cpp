/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "gemm_ex_device_common.h"
#include "cgemm_ex_kernel.h"
#include "lib/matmul_intf.h"
#include "gemm_ex_tiling.h"

using namespace AscendC;
struct Pad305Args {
    uint32_t rows;
    uint32_t kp;
    uint32_t np;
};
__aicore__ inline void MakeLongKPaddingIndices(
    LocalTensor<uint32_t> idx, LocalTensor<uint32_t> aux, LocalTensor<float> src)
{
    constexpr uint32_t CAP = 8192;
    CreateVecIndex(idx.ReinterpretCast<int32_t>(), 0, CAP);
    PipeBarrier<PIPE_V>();
    ShiftRight(aux, idx, 2U, CAP);
    ShiftRight(src.ReinterpretCast<uint32_t>(), idx, 1U, CAP);
    PipeBarrier<PIPE_V>();
    ShiftLeft(aux, aux, 1U, CAP);
    ShiftLeft(src.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), 1U, CAP);
    PipeBarrier<PIPE_V>();
    Sub(idx.ReinterpretCast<int32_t>(), idx.ReinterpretCast<int32_t>(), src.ReinterpretCast<int32_t>(), CAP);
    PipeBarrier<PIPE_V>();
    Add(idx.ReinterpretCast<int32_t>(), idx.ReinterpretCast<int32_t>(), aux.ReinterpretCast<int32_t>(), CAP);
    PipeBarrier<PIPE_V>();
    ShiftLeft(idx, idx, 2U, CAP);
    PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void PadLongKLeft(GlobalTensor<float> a, GlobalTensor<float> aa, LocalTensor<float> dst, Pad305Args t)
{
    constexpr uint32_t CAP = 8192;
    const uint32_t tiles = (t.rows + CAP - 1) / CAP;
    for (uint64_t task = GetBlockIdx(); task < uint64_t(4 * t.kp) * tiles; task += GetBlockNum()) {
        uint32_t col = task / tiles;
        uint32_t row = (task % tiles) * CAP;
        uint32_t len = min(CAP, t.rows - row);
        if (col % 4 < 2) {
            uint32_t oldcol = (col / 4) * 2 + col % 4;
            DataCopy(dst, a[uint64_t(oldcol) * t.rows + row], len);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        } else {
            Duplicate(dst, 0.0f, len);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        DataCopy(aa[uint64_t(col) * t.rows + row], dst, len);
        PipeBarrier<PIPE_ALL>();
    }
}

__global__ __aicore__ void cgemm_pad305_kernel(GM_ADDR olda, GM_ADDR oldb, GM_ADDR newa, GM_ADDR newb, Pad305Args t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    constexpr uint32_t CAP = 8192;
    LocalTensor<float> src(TPosition::VECCALC, 0, CAP);
    LocalTensor<float> dst(TPosition::VECCALC, CAP * 4, CAP);
    LocalTensor<uint32_t> idx(TPosition::VECCALC, CAP * 8, CAP);
    LocalTensor<uint32_t> aux(TPosition::VECCALC, CAP * 12, CAP);
    GlobalTensor<float> a;
    GlobalTensor<float> b;
    GlobalTensor<float> aa;
    GlobalTensor<float> bb;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(olda));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(oldb));
    aa.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(newa));
    bb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(newb));
    PadLongKLeft(a, aa, dst, t);
    MakeLongKPaddingIndices(idx, aux, src);
    uint64_t total = uint64_t(4 * t.kp) * t.np;
    for (uint64_t start = uint64_t(GetBlockIdx()) * CAP; start < total; start += uint64_t(GetBlockNum()) * CAP) {
        uint32_t len = min(uint64_t(CAP), total - start);
        DataCopy(src, b[start / 2], len / 2);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Gather(dst, src, idx, 0U, len);
        PipeBarrier<PIPE_V>();
        uint64_t mask[2] = {0xccccccccccccccccULL, 0};
        Duplicate(dst, 0.0f, mask, uint8_t(len / 64), 1, 8);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(bb[start], dst, len);
        PipeBarrier<PIPE_ALL>();
    }
}
void cgemm_pad305_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR aa, GM_ADDR bb, uint32_t rows, uint32_t kp,
    uint32_t np)
{
    cgemm_pad305_kernel<<<blocks, nullptr, stream>>>(a, b, aa, bb, Pad305Args{rows, kp, np});
}

using namespace AscendC;
using namespace matmul;
namespace c_long301 {
using F = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulFuncParams func{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                true,  true};
constexpr MatmulBiasParams bias{false};
template <int TILE>
struct Config {
    static constexpr MatmulShapeParams shape{TILE, TILE, 32768, 128, 128, 128};
    static constexpr MatmulConfig config = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shape, func, bias);
    static constexpr MatmulApiStaticTiling tiling = GetMatmulApiTiling<F, F, F, F>(config);
    using MM = MatmulImpl<F, F, F, F, tiling>;
};
template <int TILE>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    uint32_t nt = (t.n + TILE - 1) / TILE;
    uint32_t id = GetBlockIdx();
    uint32_t nb = (id % nt) * TILE;
    uint32_t mb = (id / nt) * TILE;
    uint32_t rows = min(uint32_t(TILE), uint32_t(t.n) - nb);
    uint32_t cols = min(uint32_t(TILE), uint32_t(t.m) - mb);
    GlobalTensor<float> L;
    GlobalTensor<float> R;
    GlobalTensor<float> C;
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b) + uint64_t(nb) * t.k);
    R.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a) + mb);
    C.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c) + uint64_t(nb) * t.m + mb);
    TPipe pipe;
    typename Config<TILE>::MM mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetOrgShape(rows, t.m, t.k, t.k, t.m);
    mm.SetSingleShape(rows, cols, t.k);
    mm.SetTensorA(L, false);
    mm.SetTensorB(R, false);
    mm.IterateAll(C, 0);
    mm.End();
}
} // namespace c_long301
__global__ __aicore__ __cube__ void cgemm_long301_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetHF32Mode(HF32Mode::DISABLE);
    if (uint64_t(t.m) * t.n < 4194304) {
        c_long301::Run<128>(a, b, c, t);
    } else {
        c_long301::Run<256>(a, b, c, t);
    }
}
void cgemm_long301_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmExTiling& t)
{
    cgemm_long301_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
}

__aicore__ inline void SumEightPartials(LocalTensor<float> local, uint32_t cap)
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t part = 1; part < 8; ++part) {
        Add(local, local, local[part * cap], cap);
        PipeBarrier<PIPE_V>();
    }
}

struct CgemmSplit104Args {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t groups;
    uint32_t reciprocal;
};
template <bool RIGHT_TRANS>
__global__ __aicore__ __cube__ void cgemm_ex_split104_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outGm, CgemmSplit104Args t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using namespace Catlass;
    SetHF32Mode(HF32Mode::DISABLE);
    constexpr uint32_t TM = RIGHT_TRANS ? 128 : 176;
    constexpr uint32_t TN = 128;
    constexpr uint32_t TK = 128;
    constexpr uint32_t SPLITS = 8;
    using LA = std::conditional_t<RIGHT_TRANS, layout::ColumnMajor, layout::RowMajor>;
    using LB = layout::RowMajor;
    using Block = Gemm::Block::BlockMmad<
        Gemm::MmadAtlasA2Pingpong<true>, GemmShape<TM, TN, TK>, GemmShape<TM, TN, 32>, Gemm::GemmType<float, LA>,
        Gemm::GemmType<float, LB>, Gemm::GemmType<float, layout::RowMajor>>;
    Arch::Resource<Arch::AtlasA2> resource;
    Block block(resource);
    auto left = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(bGm));
    auto right = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(aGm));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outGm));
    const LA la(t.n, t.k, t.ldb);
    const LB lb(t.k, t.m, t.lda);
    const layout::RowMajor lc(t.n, t.m, t.m);
    const uint32_t chunks = (t.k + 31) / 32;
    for (uint32_t job = GetBlockIdx(); job < t.groups * SPLITS; job += GetBlockNum()) {
        const uint32_t part =
            t.groups == 1 ? job : static_cast<uint32_t>((static_cast<uint64_t>(job) * t.reciprocal) >> 32);
        const uint32_t row = job - part * t.groups;
        const uint32_t mb = row * TM;
        const uint32_t first = (chunks * part / SPLITS) * 32;
        const uint32_t last = min(t.k, (chunks * (part + 1) / SPLITS) * 32);
        block(
            left[la.GetOffset(MatrixCoord(mb, first))], la, right[lb.GetOffset(MatrixCoord(first, 0))], lb,
            dest[static_cast<uint64_t>(part) * t.n * t.m + static_cast<uint64_t>(mb) * t.m], lc,
            GemmCoord(min(TM, t.n - mb), t.m, last - first));
    }
    PipeBarrier<PIPE_ALL>();
}
extern "C" __global__ __aicore__ void cgemm_ex_split104_sum(GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t first = GetBlockIdx() * 8;
    if (first >= n) {
        return;
    }
    const uint32_t count = min(8U, n - first);
    const uint32_t cap = count * m;
    LocalTensor<float> local(TPosition::VECCALC, 0, 8192);
    auto input = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(inputGm));
    auto output = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outputGm));
    for (uint32_t part = 0; part < 8; ++part) {
        DataCopy(local[part * cap], input[static_cast<uint64_t>(part) * n * m + static_cast<uint64_t>(first) * m], cap);
    }
    SumEightPartials(local, cap);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[static_cast<uint64_t>(first) * m], local, cap);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
CgemmSplit104Args LaunchComplexSplit(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* partial, GemmExTiling t)
{
    const uint32_t tile = t.transB ? 128U : 176U;
    const uint32_t groups = (static_cast<uint32_t>(t.n) + tile - 1) / tile;
    CgemmSplit104Args q{
        static_cast<uint32_t>(t.m),
        static_cast<uint32_t>(t.n),
        static_cast<uint32_t>(t.k),
        static_cast<uint32_t>(t.lda),
        static_cast<uint32_t>(t.ldb),
        groups,
        groups == 1 ? 0U : static_cast<uint32_t>(((1ULL << 32) + groups - 1) / groups)};
    const uint32_t use = std::min(blocks, groups * 8);
    if (t.transB) {
        cgemm_ex_split104_kernel<true><<<use, nullptr, stream>>>(a, b, partial, q);
    } else {
        cgemm_ex_split104_kernel<false><<<use, nullptr, stream>>>(a, b, partial, q);
    }
    return q;
}
void cgemm_ex_split104_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* partial, uint8_t* product, GemmExTiling t)
{
    const auto q = LaunchComplexSplit(blocks, stream, a, b, partial, t);
    cgemm_ex_split104_sum<<<(q.n + 7) / 8, nullptr, stream>>>(partial, product, q.m, q.n);
}

struct ComplexReductionOutput {
    CgemmExLayout t;
    uint32_t first;
    uint32_t count;
    uint32_t rows;
    uint32_t sourceWidth;
    uint32_t dma;
    uint32_t cap;
    LocalTensor<float> local;
    LocalTensor<float> out;
    LocalTensor<uint32_t> indices;
    LocalTensor<uint32_t> scratch;
    GlobalTensor<float> output;
    __aicore__ inline void Planar()
    {
        auto i = scratch;
        auto half = scratch[256];
        CreateVecIndex(i.ReinterpretCast<int32_t>(), 0, sourceWidth);
        PipeBarrier<PIPE_V>();
        ShiftRight(half, i, 1U, sourceWidth);
        ShiftLeft(i, i, 31U, sourceWidth);
        PipeBarrier<PIPE_V>();
        ShiftRight(i, i, 31U, sourceWidth);
        PipeBarrier<PIPE_V>();
        Muls(i.ReinterpretCast<int32_t>(), i.ReinterpretCast<int32_t>(), static_cast<int32_t>(rows), sourceWidth);
        PipeBarrier<PIPE_V>();
        Add(i.ReinterpretCast<int32_t>(), i.ReinterpretCast<int32_t>(), half.ReinterpretCast<int32_t>(), sourceWidth);
        PipeBarrier<PIPE_V>();
        ShiftLeft(i, i, 2U, sourceWidth);
        PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < count; ++col) {
            Adds(
                indices[col * sourceWidth].ReinterpretCast<int32_t>(), i.ReinterpretCast<int32_t>(),
                static_cast<int32_t>(col * sourceWidth * sizeof(float)), sourceWidth);
            Muls(local[(2 * col + 1) * rows], local[(2 * col + 1) * rows], -1.0f, rows);
        }
        PipeBarrier<PIPE_V>();
        Gather(out, local, indices, 0U, cap);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(
            output[static_cast<uint64_t>(first) * 2 * t.ldc], out,
            {static_cast<uint16_t>(count), static_cast<uint32_t>(t.m) * 8U, (sourceWidth - dma) / 8,
             static_cast<uint32_t>(t.ldc - t.m) * 8U, 0});
    }
    __aicore__ inline void Interleaved()
    {
        const uint32_t width = 2 * static_cast<uint32_t>(t.m);
        const uint32_t elements = count * width;
        if (t.ldc == t.m) {
            if ASCEND_IS_AIV {
                SetMaskCount();
                set_vector_mask(0, width);
                auto* p = reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
                vreducev2(p, p, p + 1024, static_cast<uint16_t>(count), 1, 7, static_cast<uint16_t>(rows / 8), 0);
                SetMaskNorm();
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(output[static_cast<uint64_t>(first) * width], local, {1, elements * 4U, 0, 0, 0});
        } else {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(
                output[static_cast<uint64_t>(first) * 2 * t.ldc], local,
                {static_cast<uint16_t>(count), width * 4U, (rows - dma) / 8, static_cast<uint32_t>(t.ldc - t.m) * 8U,
                 0});
        }
    }
};

extern "C" __global__ __aicore__ void cgemm_ex_reduce_finish107(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t rows, uint32_t columns, CgemmExLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetMaskNorm();
    const uint32_t group = t.mode == 3 ? 8U : 32U;
    const uint32_t first = GetBlockIdx() * group;
    if (first >= t.n) {
        return;
    }
    const uint32_t count = min(group, static_cast<uint32_t>(t.n) - first);
    const uint32_t physical = t.mode == 3 ? 2U : 1U;
    const uint32_t cap = count * physical * rows;
    LocalTensor<float> local(TPosition::VECCALC, 0, 16384);
    LocalTensor<float> out(TPosition::VECCALC, 16384 * 4, 2048);
    LocalTensor<uint32_t> indices(TPosition::VECCALC, 18432 * 4, 2048);
    LocalTensor<uint32_t> scratch(TPosition::VECCALC, 20480 * 4, 512);
    auto input = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(inputGm));
    auto output = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outputGm));
    DataCopy(
        local, input[static_cast<uint64_t>(first) * physical * rows],
        DataCopyParams{8, static_cast<uint16_t>(cap / 8), static_cast<uint16_t>((columns * rows - cap) / 8), 0});
    SumEightPartials(local, cap);
    const uint32_t sourceWidth = physical * rows;
    const uint32_t dma = (2 * static_cast<uint32_t>(t.m) + 7) / 8 * 8;
    ComplexReductionOutput finish{t, first, count, rows, sourceWidth, dma, cap, local, out, indices, scratch, output};
    if (t.mode == 3) {
        finish.Planar();
    } else {
        finish.Interleaved();
    }

    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
void cgemm_ex_split107_finish_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* partial, uint8_t* output, GemmExTiling t,
    CgemmExLayout finish)
{
    const auto q = LaunchComplexSplit(blocks, stream, a, b, partial, t);
    cgemm_ex_reduce_finish107<<<
        (finish.n + (finish.mode == 3 ? 7 : 31)) / (finish.mode == 3 ? 8 : 32), nullptr, stream>>>(
        partial, output, q.m, q.n, finish);
}

struct CgemmNative110Args {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t columns;
    uint32_t grid;
    uint32_t reciprocal;
    uint32_t ldc;
};
template <class LA, class LB, uint32_t TM, uint32_t TN, uint32_t TK>
__global__ __aicore__ __cube__ void cgemm_ex_native110_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CgemmNative110Args t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using namespace Catlass;
    SetHF32Mode(HF32Mode::DISABLE);
    using Block = Gemm::Block::BlockMmad<
        Gemm::MmadAtlasA2Pingpong<true>, GemmShape<TM, TN, TK>, GemmShape<TM, TN, 32>, Gemm::GemmType<float, LA>,
        Gemm::GemmType<float, LB>, Gemm::GemmType<float, layout::RowMajor>>;
    Arch::Resource<Arch::AtlasA2> resource;
    Block block(resource);
    auto left = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(b));
    auto right = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(a));
    auto out = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(c));
    const LA la(t.n, t.k, t.ldb);
    const LB lb(t.k, t.m, t.lda);
    const layout::RowMajor lc(t.n, t.m, t.ldc);
    for (uint32_t job = GetBlockIdx(); job < t.grid; job += GetBlockNum()) {
        const uint32_t row =
            t.columns == 1 ? job : static_cast<uint32_t>((static_cast<uint64_t>(job) * t.reciprocal) >> 32);
        const uint32_t col = job - row * t.columns;
        const uint32_t mb = row * TM;
        const uint32_t nb = col * TN;
        block(
            left[la.GetOffset(MatrixCoord(mb, 0))], la, right[lb.GetOffset(MatrixCoord(0, nb))], lb,
            out[lc.GetOffset(MatrixCoord(mb, nb))], lc, GemmCoord(min(TM, t.n - mb), min(TN, t.m - nb), t.k));
    }
    PipeBarrier<PIPE_ALL>();
}
template <uint32_t TM, uint32_t TN, uint32_t TK>
void LaunchComplexNative110(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, GemmExTiling t)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + TN - 1) / TN;
    const uint32_t grid = ((static_cast<uint32_t>(t.n) + TM - 1) / TM) * columns;
    CgemmNative110Args q{
        static_cast<uint32_t>(t.m),
        static_cast<uint32_t>(t.n),
        static_cast<uint32_t>(t.k),
        static_cast<uint32_t>(t.lda),
        static_cast<uint32_t>(t.ldb),
        columns,
        grid,
        columns == 1 ? 0U : static_cast<uint32_t>(((1ULL << 32) + columns - 1) / columns),
        static_cast<uint32_t>(t.ldc)};
    using R = Catlass::layout::RowMajor;
    using C = Catlass::layout::ColumnMajor;
    const uint32_t use = std::min(blocks, grid);
    if (t.transB) {
        if (t.transA) {
            cgemm_ex_native110_kernel<C, C, TM, TN, TK><<<use, nullptr, stream>>>(a, b, c, q);
        } else {
            cgemm_ex_native110_kernel<C, R, TM, TN, TK><<<use, nullptr, stream>>>(a, b, c, q);
        }
    } else {
        if (t.transA) {
            cgemm_ex_native110_kernel<R, C, TM, TN, TK><<<use, nullptr, stream>>>(a, b, c, q);
        } else {
            cgemm_ex_native110_kernel<R, R, TM, TN, TK><<<use, nullptr, stream>>>(a, b, c, q);
        }
    }
}
void cgemm_ex_native110_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, GemmExTiling t)
{
    if (t.n <= 32) {
        LaunchComplexNative110<32, 80, 256>(blocks, stream, a, b, c, t);
    } else if (t.n <= 128 && t.m >= 1536) {
        LaunchComplexNative110<32, 208, 256>(blocks, stream, a, b, c, t);
    } else if (t.n <= 128) {
        LaunchComplexNative110<32, 128, 256>(blocks, stream, a, b, c, t);
    } else if (t.m <= 256 && t.n >= 512) {
        LaunchComplexNative110<96, 128, 256>(blocks, stream, a, b, c, t);
    } else if (t.m > 512 && t.m <= 640 && t.n > 128 && t.n <= 512) {
        LaunchComplexNative110<80, 128, 256>(blocks, stream, a, b, c, t);
    } else {
        LaunchComplexNative110<64, 160, 256>(blocks, stream, a, b, c, t);
    }
}

void cgemm_ex_raw114_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, GemmExTiling t)
{
    LaunchComplexNative110<64, 64, 512>(blocks, stream, a, b, c, t);
}

void cgemm_ex_cross_compute_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, GemmExTiling t)
{
    LaunchComplexNative110<64, 208, 128>(blocks, stream, a, b, c, t);
}
