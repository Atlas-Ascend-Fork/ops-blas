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
#include "chbmv_kernel.h"

// Compensated FP32 arithmetic requires separate multiply/add rounding.
#pragma clang fp contract(off)

#include "chbmv_extended.h"
#include "chbmv_vector.h"
using namespace AscendC;

namespace {
struct Complex {
    float real, imag;
};
__aicore__ inline Complex AddComplex(Complex a, Complex b) { return {a.real + b.real, a.imag + b.imag}; }
__aicore__ inline Complex MulComplex(Complex a, Complex b)
{
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
__aicore__ inline Complex LoadComplex(const GlobalTensor<float>& p, int64_t i)
{
    return {p.GetValue(2 * i), p.GetValue(2 * i + 1)};
}

class ChbmvKernel {
public:
    __aicore__ inline explicit ChbmvKernel(const ChbmvTilingData& tiling) : t(tiling) {}

    __aicore__ inline void Run(GM_ADDR a, GM_ADDR x, GM_ADDR y)
    {
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        TPipe pipe;
        TBuf<TPosition::VECCALC> outBuffer;
        pipe.InitBuffer(outBuffer, CHBMV_OUTPUT_BYTES);
        auto out = outBuffer.Get<float>();
        const auto sToMte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
        const auto mte3ToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
        const uint32_t start = GetBlockIdx() * t.blockRows;
        const uint32_t end = start + t.blockRows < t.n ? start + t.blockRows : t.n;
        for (uint32_t r = start; r < end; ++r) {
            const auto result = ComputeRow(r);
            const int64_t yi = t.yStart + static_cast<int64_t>(r) * t.incy;
            out.SetValue(0, result.real);
            out.SetValue(1, result.imag);
            SetFlag<HardEvent::S_MTE3>(sToMte3);
            WaitFlag<HardEvent::S_MTE3>(sToMte3);
            DataCopyPad(yGm[2 * yi], out, DataCopyExtParams{1, 8, 0, 0, 0});
            SetFlag<HardEvent::MTE3_S>(mte3ToS);
            WaitFlag<HardEvent::MTE3_S>(mte3ToS);
        }
    }

private:
    __aicore__ inline Complex ComputeRow(uint32_t r)
    {
        Complex result{0, 0};
        const int64_t yi = t.yStart + static_cast<int64_t>(r) * t.incy;
        if (t.betaReal != 0 || t.betaImag != 0) {
            result = LoadComplex(yGm, yi);
            if (t.betaReal != 1 || t.betaImag != 0)
                result = MulComplex({t.betaReal, t.betaImag}, result);
        }
        if (t.alphaReal != 0 || t.alphaImag != 0) {
            const Complex alpha{t.alphaReal, t.alphaImag};
            Complex mirror{0, 0};
            const uint32_t first = r > t.band ? r - t.band : 0;
            const uint32_t last = r + (t.band < t.n - 1 - r ? t.band : t.n - 1 - r) + 1;
            for (uint32_t c = first; c < r; ++c) {
                auto term = Term(r, c, t.upper != 0);
                if (t.upper)
                    mirror = AddComplex(mirror, term);
                else
                    result = AddComplex(result, term);
            }
            const auto ax = MulComplex(alpha, LoadComplex(xGm, t.xStart + static_cast<int64_t>(r) * t.incx));
            // Do not even read the diagonal imaginary component.
            const float diag = aGm.GetValue(2 * (static_cast<int64_t>(r) * t.lda + (t.upper ? t.k : 0)));
            result = AddComplex(result, {ax.real * diag, ax.imag * diag});
            if (t.upper)
                result = AddComplex(result, MulComplex(alpha, mirror));
            for (uint32_t c = r + 1; c < last; ++c) {
                auto term = Term(r, c, t.upper == 0);
                if (t.upper)
                    result = AddComplex(result, term);
                else
                    mirror = AddComplex(mirror, term);
            }
            if (!t.upper)
                result = AddComplex(result, MulComplex(alpha, mirror));
        }
        if (!chbmv_extended::Finite(result.real) || !chbmv_extended::Finite(result.imag)) {
            chbmv_extended::Result extended{0, 0};
            chbmv_extended::Row(aGm, xGm, yGm, t, r, extended);
            result = {extended.real, extended.imag};
        }
        return result;
    }

    __aicore__ inline Complex Term(uint32_t row, uint32_t col, bool mirror)
    {
        const int64_t r = row;
        const int64_t c = col;
        const int64_t index =
            mirror ? r * t.lda + (t.upper ? t.k + c - r : c - r) : c * t.lda + (t.upper ? t.k + r - c : r - c);
        auto av = LoadComplex(aGm, index);
        const auto xv = LoadComplex(xGm, t.xStart + c * t.incx);
        if (mirror) {
            av.imag = -av.imag;
            return MulComplex(av, xv);
        }
        return MulComplex(MulComplex({t.alphaReal, t.alphaImag}, xv), av);
    }
    const ChbmvTilingData& t;
    GlobalTensor<float> aGm, xGm, yGm;
};
} // namespace

extern "C" __global__ __aicore__ void chbmv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, ChbmvTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (t.vectorMode) {
        ChbmvVector kernel(t);
        kernel.Run(a, x, y);
    } else {
        ChbmvKernel kernel(t);
        kernel.Run(a, x, y);
    }
}
aclError chbmv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, ChbmvTilingData t, aclrtStream stream)
{
    // Report this launch's status directly. Thread-wide last-error state can
    // also contain unrelated runtime calls, including tool polling timeouts.
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(chbmv_kernel), &function);
    if (lookup != ACL_SUCCESS)
        return lookup;
    struct Arguments {
        uint8_t* a;
        uint8_t* x;
        uint8_t* y;
        ChbmvTilingData tiling;
    } arguments{a, x, y, t};
    static_assert(sizeof(Arguments) == 3 * sizeof(uint8_t*) + sizeof(ChbmvTilingData));
    return aclrtLaunchKernelWithHostArgs(
        function, t.numBlocks, stream, nullptr, &arguments, sizeof(arguments), nullptr, 0);
}
