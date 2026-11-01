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
#include "cann_ops_blas_common.h"
#include "cher2_tiling_data.h"

#pragma clang fp contract(off)

namespace {
__simt_callee__ inline uint64_t Offset(uint32_t i, uint32_t n, int64_t inc)
{
    return inc > 0 ? static_cast<uint64_t>(i) * inc : static_cast<uint64_t>(n - 1U - i) * (-inc);
}

__simt_callee__ inline float2 Multiply(float2 a, float2 b)
{
    // SIMT's later lowering can otherwise contract/reassociate an inlined
    // complex expression despite the frontend FP pragma. Round each BLAS
    // FP32 intermediate, including cancellation with large alpha.
    volatile float rr = a.x * b.x;
    volatile float ii = a.y * b.y;
    volatile float ri = a.x * b.y;
    volatile float ir = a.y * b.x;
    volatile float real = rr - ii;
    volatile float imag = ri + ir;
    return {real, imag};
}

template <bool PAIRED>
__simt_callee__ inline float2 LoadComplex(__gm__ const float* values, uint64_t index)
{
    if constexpr (PAIRED) {
        return reinterpret_cast<__gm__ const float2*>(values)[index];
    } else {
        return {values[2ULL * index], values[2ULL * index + 1U]};
    }
}

template <bool UPPER, bool PAIRED>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Cher2(
    uint32_t n, uint32_t lda, float ar, float ai, int64_t incx, int64_t incy, __gm__ const float* x,
    __gm__ const float* y, __gm__ float* a)
{
    for (uint32_t col = blockIdx.x; col < n; col += gridDim.x) {
        const float2 xc = LoadComplex<PAIRED>(x, Offset(col, n, incx));
        const float2 yc = LoadComplex<PAIRED>(y, Offset(col, n, incy));
        const bool nonzero = xc.x != 0.0f || xc.y != 0.0f || yc.x != 0.0f || yc.y != 0.0f;
        const float2 t1 = Multiply(float2{ar, ai}, float2{yc.x, -yc.y});
        float2 t2 = Multiply(float2{ar, ai}, xc);
        t2.y = -t2.y;
        const uint32_t begin = UPPER ? 0U : col;
        const uint32_t end = UPPER ? col + 1U : n;
        for (uint32_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
            const uint64_t index = 2ULL * (static_cast<uint64_t>(col) * lda + row);
            if (row == col) {
                if (nonzero) {
                    const float2 p = Multiply(xc, t1);
                    const float2 q = Multiply(yc, t2);
                    volatile float sum = p.x + q.x;
                    a[index] = a[index] + sum;
                }
                // Never load the old imaginary diagonal, including NaN/Inf.
                a[index + 1] = 0.0f;
            } else if (nonzero) {
                const float2 p = Multiply(LoadComplex<PAIRED>(x, Offset(row, n, incx)), t1);
                const float2 q = Multiply(LoadComplex<PAIRED>(y, Offset(row, n, incy)), t2);
                volatile float real = a[index] + p.x;
                volatile float imag = a[index + 1] + p.y;
                a[index] = real + q.x;
                a[index + 1] = imag + q.y;
            }
        }
    }
}
template <bool PAIRED>
__aicore__ inline void Dispatch(GM_ADDR x, GM_ADDR y, GM_ADDR a, const Cher2TilingData& t)
{
    auto* xp = reinterpret_cast<__gm__ const float*>(x);
    auto* yp = reinterpret_cast<__gm__ const float*>(y);
    auto* ap = reinterpret_cast<__gm__ float*>(a);
    if (t.uplo == ACLBLAS_UPPER) {
        asc_vf_call<Cher2<true, PAIRED>>(
            dim3{t.numThreads, 1, 1}, t.n, t.lda, t.alphaReal, t.alphaImag, t.incx, t.incy, xp, yp, ap);
    } else {
        asc_vf_call<Cher2<false, PAIRED>>(
            dim3{t.numThreads, 1, 1}, t.n, t.lda, t.alphaReal, t.alphaImag, t.incx, t.incy, xp, yp, ap);
    }
}
} // namespace

__global__ __aicore__ void cher2_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR a, const Cher2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // The public complex ABI requires only four-byte alignment. Pair loads
    // are reserved for eight-byte-aligned x/y; A always uses scalar accesses.
    const bool paired = ((reinterpret_cast<uint64_t>(x) | reinterpret_cast<uint64_t>(y)) & 7U) == 0;
    if (paired) {
        Dispatch<true>(x, y, a, t);
    } else {
        Dispatch<false>(x, y, a, t);
    }
}

void cher2_kernel_do(uint8_t* x, uint8_t* y, uint8_t* a, const Cher2TilingData& t, uint32_t blocks, void* stream)
{
    cher2_kernel<<<blocks, nullptr, stream>>>(x, y, a, t);
}
