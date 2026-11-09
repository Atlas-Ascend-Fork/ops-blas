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
#include "simt_api/asc_simt.h"
#include "cgemm3m_strided_batched_tiling_data.h"
namespace {
constexpr uint32_t THREADS = 256;
struct Complex {
    float real, imag;
};

__simt_callee__ __aicore__ inline Complex ReadOperand(
    __gm__ const Complex* input, int64_t base, uint32_t row, uint32_t col, uint32_t ld, uint32_t trans)
{
    int64_t offset = base + (trans == 0 ? int64_t(col) * ld + row : int64_t(row) * ld + col);
    Complex v{input[offset].real, input[offset].imag};
    if (trans == 2) {
        v.imag = -v.imag;
    }
    return v;
}
__simt_callee__ __aicore__ inline Complex Product(
    __gm__ const Complex* a, __gm__ const Complex* b, const Cgemm3mSbTiling& t, uint32_t row, uint32_t col,
    uint32_t batch)
{
    float p = 0, q = 0, r = 0;
    for (uint32_t depth = 0; depth < t.k; ++depth) {
        Complex av = ReadOperand(a, int64_t(batch) * t.strideA, row, depth, t.lda, t.transa);
        Complex bv = ReadOperand(b, int64_t(batch) * t.strideB, depth, col, t.ldb, t.transb);
        p += av.real * bv.real;
        q += av.imag * bv.imag;
        r += (av.real + av.imag) * (bv.real + bv.imag);
    }
    return {p - q, (r - p) - q};
}
__simt_callee__ __aicore__ inline void StoreResult(
    __gm__ Complex* c, int64_t offset, Complex v, const Cgemm3mSbTiling& t, bool scaleOnly)
{
    Complex out = scaleOnly ? Complex{0, 0} : Complex{t.ar * v.real - t.ai * v.imag, t.ar * v.imag + t.ai * v.real};
    if (t.br != 0 || t.bi != 0) {
        Complex old{c[offset].real, c[offset].imag};
        out.real += t.br * old.real - t.bi * old.imag;
        out.imag += t.br * old.imag + t.bi * old.real;
    }
    c[offset].real = out.real;
    c[offset].imag = out.imag;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void Direct(
    __gm__ const Complex* a, __gm__ const Complex* b, __gm__ Complex* c, Cgemm3mSbTiling t, uint32_t block,
    uint32_t blocks)
{
    const uint64_t matrix = uint64_t(t.m) * t.n;
    const uint64_t count = matrix * (t.strideC == 0 ? 1 : t.batch);
    const bool scaleOnly = t.k == 0 || (t.ar == 0 && t.ai == 0);
    for (uint64_t i = uint64_t(block) * THREADS + threadIdx.x; i < count; i += uint64_t(blocks) * THREADS) {
        const uint32_t row = i % t.m;
        const uint32_t col = (i / t.m) % t.n;
        const uint32_t first = t.strideC == 0 ? 0 : i / matrix;
        const uint32_t last = t.strideC == 0 ? t.batch : first + 1;
        for (uint32_t batch = first; batch < last; ++batch) {
            Complex v{0, 0};
            if (!scaleOnly) {
                v = Product(a, b, t, row, col, batch);
            }
            StoreResult(c, int64_t(batch) * t.strideC + int64_t(col) * t.ldc + row, v, t, scaleOnly);
        }
    }
}
} // namespace
extern "C" __global__ __aicore__ void cgemm3m_sb_direct(GM_ADDR a, GM_ADDR b, GM_ADDR c, Cgemm3mSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<Direct>(
        dim3{THREADS, 1, 1}, reinterpret_cast<__gm__ const Complex*>(a), reinterpret_cast<__gm__ const Complex*>(b),
        reinterpret_cast<__gm__ Complex*>(c), t, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}
void cgemm3m_sb_direct_do(
    uint32_t blocks, void* stream, const void* a, const void* b, void* c, const Cgemm3mSbTiling& t)
{
    cgemm3m_sb_direct<<<blocks, nullptr, stream>>>(
        reinterpret_cast<uint8_t*>(const_cast<void*>(a)), reinterpret_cast<uint8_t*>(const_cast<void*>(b)),
        reinterpret_cast<uint8_t*>(c), t);
}

namespace cgemm3m_panel {
using namespace AscendC;
constexpr uint32_t ROWS = CGEMM3M_PANEL_ROWS;
constexpr uint32_t COLS = CGEMM3M_PANEL_COLS;
constexpr uint32_t DEPTH = 128;
constexpr uint32_t COMPLEX = 2;
constexpr uint32_t PLANES = 3;
constexpr uint32_t A_FLOATS = COMPLEX * ROWS * DEPTH;
constexpr uint32_t B_FLOATS = COMPLEX * COLS * DEPTH;
constexpr uint32_t SUM_FLOATS = PLANES * ROWS * COLS;

template <uint32_t ROW, typename Register, typename Mask>
__simd_callee__ inline void AccumulateRow(
    Register& p, Register& q, Register& r, Register& br, Register& bi, Register& bs, __ubuf__ float* a, uint32_t k,
    uint32_t aRowStride, uint32_t aDepthStride, Mask& mask)
{
    MicroAPI::RegTensor<float> ar, ai, as;
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(ar, a + ROW * aRowStride + k * aDepthStride);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(ai, a + ROW * aRowStride + k * aDepthStride + 1);
    MicroAPI::Add(as, ar, ai, mask);
    MicroAPI::MulAddDst(p, ar, br, mask);
    MicroAPI::MulAddDst(q, ai, bi, mask);
    MicroAPI::MulAddDst(r, as, bs, mask);
}
template <bool ConjugateB>
__simd_vf__ inline void Accumulate(
    __ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* sums, uint32_t depth, uint32_t aRowStride,
    uint32_t aDepthStride)
{
    auto mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> p0, q0, r0, p1, q1, r1, p2, q2, r2, p3, q3, r3;
    MicroAPI::LoadAlign(p0, sums + 0 * COLS);
    MicroAPI::LoadAlign(q0, sums + 1 * COLS);
    MicroAPI::LoadAlign(r0, sums + 2 * COLS);
    MicroAPI::LoadAlign(p1, sums + 3 * COLS);
    MicroAPI::LoadAlign(q1, sums + 4 * COLS);
    MicroAPI::LoadAlign(r1, sums + 5 * COLS);
    MicroAPI::LoadAlign(p2, sums + 6 * COLS);
    MicroAPI::LoadAlign(q2, sums + 7 * COLS);
    MicroAPI::LoadAlign(r2, sums + 8 * COLS);
    MicroAPI::LoadAlign(p3, sums + 9 * COLS);
    MicroAPI::LoadAlign(q3, sums + 10 * COLS);
    MicroAPI::LoadAlign(r3, sums + 11 * COLS);
#pragma clang loop unroll(disable)
    for (uint16_t k = 0; k < depth; ++k) {
        MicroAPI::RegTensor<float> br, bi, bs;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(br, bi, b + k * COMPLEX * COLS);
        if constexpr (ConjugateB)
            MicroAPI::Muls(bi, bi, -1.0f, mask);
        MicroAPI::Add(bs, br, bi, mask);
        AccumulateRow<0>(p0, q0, r0, br, bi, bs, a, k, aRowStride, aDepthStride, mask);
        AccumulateRow<1>(p1, q1, r1, br, bi, bs, a, k, aRowStride, aDepthStride, mask);
        AccumulateRow<2>(p2, q2, r2, br, bi, bs, a, k, aRowStride, aDepthStride, mask);
        AccumulateRow<3>(p3, q3, r3, br, bi, bs, a, k, aRowStride, aDepthStride, mask);
    }
    MicroAPI::StoreAlign(sums + 0 * COLS, p0, mask);
    MicroAPI::StoreAlign(sums + 1 * COLS, q0, mask);
    MicroAPI::StoreAlign(sums + 2 * COLS, r0, mask);
    MicroAPI::StoreAlign(sums + 3 * COLS, p1, mask);
    MicroAPI::StoreAlign(sums + 4 * COLS, q1, mask);
    MicroAPI::StoreAlign(sums + 5 * COLS, r1, mask);
    MicroAPI::StoreAlign(sums + 6 * COLS, p2, mask);
    MicroAPI::StoreAlign(sums + 7 * COLS, q2, mask);
    MicroAPI::StoreAlign(sums + 8 * COLS, r2, mask);
    MicroAPI::StoreAlign(sums + 9 * COLS, p3, mask);
    MicroAPI::StoreAlign(sums + 10 * COLS, q3, mask);
    MicroAPI::StoreAlign(sums + 11 * COLS, r3, mask);
}

__simd_vf__ inline void Combine(__ubuf__ float* sums, __ubuf__ float* output, float signA)
{
    auto mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> index;
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t>&>(index), int32_t(0));
    MicroAPI::Muls(index, index, COMPLEX * ROWS, mask);
#pragma unroll
    for (uint32_t row = 0; row < ROWS; ++row) {
        MicroAPI::RegTensor<float> p, q, r, real, imag;
        MicroAPI::LoadAlign(p, sums + (PLANES * row) * COLS);
        MicroAPI::LoadAlign(q, sums + (PLANES * row + 1) * COLS);
        MicroAPI::LoadAlign(r, sums + (PLANES * row + 2) * COLS);
        MicroAPI::Sub(real, p, q, mask);
        MicroAPI::Sub(imag, r, p, mask);
        MicroAPI::Sub(imag, imag, q, mask);
        MicroAPI::Muls(imag, imag, signA, mask);
        MicroAPI::Scatter(output + COMPLEX * row, real, index, mask);
        MicroAPI::Scatter(output + COMPLEX * row + 1, imag, index, mask);
    }
}

class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const Cgemm3mSbTiling& t)
    {
        t_ = t;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        c_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
        pipe_.InitBuffer(input_, 2, (A_FLOATS + B_FLOATS) * sizeof(float));
        pipe_.InitBuffer(output_, 1, COMPLEX * ROWS * COLS * sizeof(float));
        pipe_.InitBuffer(sums_, SUM_FLOATS * sizeof(float));
    }
    __aicore__ inline void Run()
    {
        const uint32_t rowTiles = (t_.m + ROWS - 1) / ROWS;
        const uint32_t colTiles = (t_.n + COLS - 1) / COLS;
        const uint64_t tiles = uint64_t(rowTiles) * colTiles;
        for (uint64_t i = GetBlockIdx(); i < tiles * t_.batch; i += GetBlockNum()) {
            batch_ = i / tiles;
            row_ = i % rowTiles * ROWS;
            col_ = i / rowTiles % colTiles * COLS;
            rows_ = (ROWS < t_.m - row_ ? ROWS : t_.m - row_);
            cols_ = (COLS < t_.n - col_ ? COLS : t_.n - col_);
            Duplicate(sums_.Get<float>(), 0.0f, SUM_FLOATS);
            Read(0);
            for (uint32_t k = 0; k < t_.k; k += DEPTH) {
                if (k + DEPTH < t_.k) {
                    Read(k + DEPTH);
                }
                Compute(k);
            }
            Write();
        }
    }

private:
    __aicore__ inline void Read(uint32_t k)
    {
        auto input = input_.AllocTensor<float>();
        const uint32_t depth = (DEPTH < t_.k - k ? DEPTH : t_.k - k);
        const bool trans = t_.transa != 0;
        const uint32_t count = trans ? rows_ : depth;
        const uint32_t width = trans ? depth : rows_;
        const uint32_t pitch = trans ? DEPTH : ROWS;
        const int64_t offset = trans ? int64_t(row_) * t_.lda + k : int64_t(k) * t_.lda + row_;
        DataCopyExtParams acopy{
            uint16_t(count), uint32_t(width * COMPLEX * sizeof(float)),
            uint32_t((t_.lda - width) * COMPLEX * sizeof(float)),
            uint32_t((pitch * COMPLEX * sizeof(float) - ((width * COMPLEX * sizeof(float) + 31) / 32) * 32) / 32), 0};
        DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
        DataCopyPad(input, a_[COMPLEX * (int64_t(batch_) * t_.strideA + offset)], acopy, pad);
        DataCopyExtParams bCopyParams{
            uint16_t(depth), uint32_t(cols_ * COMPLEX * sizeof(float)),
            uint32_t((t_.ldb - cols_) * COMPLEX * sizeof(float)),
            uint32_t((COLS * COMPLEX * sizeof(float) - ((cols_ * COMPLEX * sizeof(float) + 31) / 32) * 32) / 32), 0};
        DataCopyPad(
            input[A_FLOATS], b_[COMPLEX * (int64_t(batch_) * t_.strideB + int64_t(k) * t_.ldb + col_)], bCopyParams,
            pad);
        input_.EnQue(input);
    }
    __aicore__ inline void Compute(uint32_t k)
    {
        auto input = input_.DeQue<float>();
        auto sums = sums_.Get<float>();
        // Factor conjugate(A) out of the three products and apply it once at writeback.
        // Only opposite input conjugations require an imaginary sign inside the K loop.
        if ((t_.transa == 2) != (t_.transb == 2))
            AccumulateInput<true>(input, sums, k);
        else
            AccumulateInput<false>(input, sums, k);
        input_.FreeTensor(input);
    }
    template <bool ConjugateB>
    __aicore__ inline void AccumulateInput(LocalTensor<float> input, LocalTensor<float> sums, uint32_t k)
    {
        asc_vf_call<Accumulate<ConjugateB>>(
            reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(input[A_FLOATS].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(sums.GetPhyAddr()), (DEPTH < t_.k - k ? DEPTH : t_.k - k),
            t_.transa == 0 ? COMPLEX : COMPLEX * DEPTH, t_.transa == 0 ? COMPLEX * ROWS : COMPLEX);
    }
    __aicore__ inline void Write()
    {
        auto sums = sums_.Get<float>();
        auto output = output_.AllocTensor<float>();
        asc_vf_call<Combine>(
            reinterpret_cast<__ubuf__ float*>(sums.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), t_.transa == 2 ? -1.0f : 1.0f);
        output_.EnQue(output);
        output = output_.DeQue<float>();
        DataCopyExtParams copy{
            uint16_t(cols_), uint32_t(rows_ * COMPLEX * sizeof(float)),
            uint32_t((ROWS * COMPLEX * sizeof(float) - ((rows_ * COMPLEX * sizeof(float) + 31) / 32) * 32) / 32),
            uint32_t((t_.ldc - rows_) * COMPLEX * sizeof(float)), 0};
        DataCopyPad(c_[COMPLEX * (int64_t(batch_) * t_.strideC + int64_t(col_) * t_.ldc + row_)], output, copy);
        output_.FreeTensor(output);
    }
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> input_;
    TQue<TPosition::VECOUT, 1> output_;
    TBuf<TPosition::VECCALC> sums_;
    GlobalTensor<float> a_, b_, c_;
    Cgemm3mSbTiling t_{};
    uint32_t batch_, row_, col_, rows_, cols_;
};
} // namespace cgemm3m_panel

extern "C" __global__ __aicore__ void cgemm3m_sb_panel(GM_ADDR a, GM_ADDR b, GM_ADDR c, Cgemm3mSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cgemm3m_panel::Kernel kernel;
    kernel.Init(a, b, c, t);
    kernel.Run();
}
void cgemm3m_sb_panel_do(uint32_t blocks, void* stream, const void* a, const void* b, void* c, const Cgemm3mSbTiling& t)
{
    cgemm3m_sb_panel<<<blocks, nullptr, stream>>>(
        reinterpret_cast<uint8_t*>(const_cast<void*>(a)), reinterpret_cast<uint8_t*>(const_cast<void*>(b)),
        reinterpret_cast<uint8_t*>(c), t);
}
