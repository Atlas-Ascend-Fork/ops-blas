/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "csyr2_kernel.h"
#include "csyr2_precise_scalar.h"
using namespace AscendC;
namespace {
constexpr uint32_t kRowTile = 1088U;
__aicore__ inline uint64_t PhysicalIndex(uint32_t i, uint32_t n, int64_t inc)
{
    uint64_t s = inc >= 0 ? static_cast<uint64_t>(inc) : static_cast<uint64_t>(-inc);
    return inc >= 0 ? static_cast<uint64_t>(i) * s : static_cast<uint64_t>(n - 1U - i) * s;
}
__aicore__ inline uint64_t CeilRunSum(uint64_t m, uint64_t g)
{
    if (!m || !g) {
        return 0;
    }
    uint64_t q = m / g;
    uint64_t r = m - q * g;
    return g * q * (q + 1ULL) / 2ULL + r * (q + 1ULL);
}
__aicore__ inline uint64_t WorkPrefix(uint32_t c, uint32_t n, uint32_t u)
{
    uint64_t cols = c;
    uint64_t elems = u == ACLBLAS_UPPER ? cols * (cols + 1ULL) / 2ULL : cols * (2ULL * n - cols + 1ULL) / 2ULL;
    uint64_t transfers =
        u == ACLBLAS_UPPER ? CeilRunSum(cols, kRowTile) : CeilRunSum(n, kRowTile) - CeilRunSum(n - cols, kRowTile);
    return 2048ULL * cols + elems + 4096ULL * transfers;
}
__aicore__ inline uint32_t Boundary(uint32_t part, uint32_t parts, uint32_t n, uint32_t u)
{
    if (!part || !parts) {
        return 0;
    }
    if (part >= parts) {
        return n;
    }
    uint64_t total = WorkPrefix(n, n, u);
    uint64_t target = (total / parts) * part + (total % parts) * part / parts;
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        uint32_t m = lo + (hi - lo) / 2;
        if (WorkPrefix(m, n, u) < target) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    return lo;
}
} // namespace
extern "C" __global__ __aicore__ __vector__ void csyr2_kernel(GM_ADDR X, GM_ADDR Y, GM_ADDR A, const Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> x;
    GlobalTensor<float> y;
    GlobalTensor<float> a;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(X));
    y.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Y));
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    uint32_t b = GetBlockIdx();
    uint32_t nb = GetBlockNum();
    uint32_t lo = Boundary(b, nb, t.n, t.uplo);
    uint32_t hi = Boundary(b + 1U, nb, t.n, t.uplo);
    // Scalar GM stores share cache lines. Keep a whole group of eight
    // complex columns on one core even when lda is not aligned.
    lo = lo < t.n ? ((lo + 7U) / 8U) * 8U : t.n;
    hi = hi < t.n ? ((hi + 7U) / 8U) * 8U : t.n;
    lo = lo < t.n ? lo : t.n;
    hi = hi < t.n ? hi : t.n;
    if (lo >= hi) {
        return;
    }
    for (uint32_t c = lo; c < hi; c++) {
        uint64_t xcIdx = PhysicalIndex(c, t.n, t.incx);
        uint64_t ycIdx = PhysicalIndex(c, t.n, t.incy);
        float xcR = x.GetValue(2 * xcIdx);
        float xcI = x.GetValue(2 * xcIdx + 1);
        float ycR = y.GetValue(2 * ycIdx);
        float ycI = y.GetValue(2 * ycIdx + 1);
        for (uint32_t r = 0; r < t.n; r++) {
            if ((t.uplo == ACLBLAS_UPPER && r > c) || (t.uplo != ACLBLAS_UPPER && r < c)) {
                continue;
            }
            uint64_t xi = PhysicalIndex(r, t.n, t.incx);
            uint64_t yi = PhysicalIndex(r, t.n, t.incy);
            float xr = x.GetValue(2 * xi);
            float xx = x.GetValue(2 * xi + 1);
            float yr = y.GetValue(2 * yi);
            float yy = y.GetValue(2 * yi + 1);
            const uint64_t z = (static_cast<uint64_t>(c) * t.lda + r) * 2ULL;
            float ar = a.GetValue(z);
            float ai = a.GetValue(z + 1U);
            Csyr2Precise::Update(ar, ai, t.ar, t.ai, xcR, xcI, ycR, ycI, xr, xx, yr, yy);
            a.SetValue(z, ar);
            a.SetValue(z + 1U, ai);
        }
    }
}
void csyr2_kernel_do(uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t, uint32_t n, void* s)
{
    csyr2_kernel<<<n, nullptr, s>>>(x, y, a, t);
}
