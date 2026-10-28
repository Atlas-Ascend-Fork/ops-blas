/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file csyr2_kernel.cpp
 * \brief Vector-only kernel。Init + 逐列 1D CopyIn/Compute/CopyOut。
 *        选定 fallback：complex64 拆成 float32 实/虚平面，Mul/Sub/Add。
 *        有复数原语可优先调用。不硬 include 仓内未公开头。
 */

#include "csyr2_kernel.h"
extern "C" uint32_t GetAivCoreNum(void);

#if defined(CSYR2_ASCENDC)
#include "kernel_operator.h"
#include "acl/acl.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#if !defined(CSYR2_ASCENDC)
/* 以下为无 CANN 时的 CPU 模拟实现。device 趟不编：std::vector / 堆分配不能进核。 */
namespace {

inline int32_t MinI32(int32_t a, int32_t b) { return (a < b) ? a : b; }

inline void Cmul(float ar, float ai, float br, float bi, float* zr, float* zi)
{
    /* z = a * b，普通复数乘：zr = ar*br - ai*bi，zi = ar*bi + ai*br */
    *zr = ar * br - ai * bi;
    *zi = ar * bi + ai * br;
}

inline void CopyComplex(aclblasComplex* dst, const aclblasComplex* src, int32_t len)
{
    /* CPU simulation preserves the logical copy order. */
    if (len <= 0) {
        return;
    }
    (void)Csyr2NeedDataCopyPad(len);
    for (int32_t i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
}

/* A 交织 staging 与实/虚平面共用同一块：float[2*cur] 先交织，再 in-place 拆成 [re | im]。
 * scratch 来自 tmp 平面（tmpCost 与 A 不别名）。 */
inline void UnpackInPlace(float* stage, int32_t cur, float* scratchIm)
{
    for (int32_t k = 0; k < cur; ++k) {
        scratchIm[k] = stage[2 * k + 1];
    }
    for (int32_t k = 0; k < cur; ++k) {
        stage[k] = stage[2 * k];
    }
    for (int32_t k = 0; k < cur; ++k) {
        stage[cur + k] = scratchIm[k];
    }
}

inline void PackInPlace(float* stage, int32_t cur, float* scratchRe, float* scratchIm)
{
    for (int32_t k = 0; k < cur; ++k) {
        scratchRe[k] = stage[k];
        scratchIm[k] = stage[cur + k];
    }
    for (int32_t k = 0; k < cur; ++k) {
        stage[2 * k] = scratchRe[k];
        stage[2 * k + 1] = scratchIm[k];
    }
}

inline void GatherXY(
    const aclblasComplex* x, const aclblasComplex* y, const Csyr2TilingData& td, int32_t i0, int32_t len, float* xRe,
    float* xIm, float* yRe, float* yIm, uint32_t tilingKey)
{
    /* 非 incx==incy==1：按 kx0 + idx(k)*incx gather 到连续 UB 平面。
     * |inc|==1 且为负：可连续搬入后反向逻辑下标访问。 */
    const bool unitPos =
        (tilingKey == CSYR2_TILING_KEY_UPPER_UNIT || tilingKey == CSYR2_TILING_KEY_LOWER_UNIT ||
         tilingKey == CSYR2_TILING_KEY_UPPER_WINDOWED || tilingKey == CSYR2_TILING_KEY_LOWER_WINDOWED);
    if (unitPos) {
        for (int32_t k = 0; k < len; ++k) {
            const int32_t i = i0 + k;
            xRe[k] = x[i].real;
            xIm[k] = x[i].imag;
            yRe[k] = y[i].real;
            yIm[k] = y[i].imag;
        }
        return;
    }
    for (int32_t k = 0; k < len; ++k) {
        const int32_t i = i0 + k;
        const int64_t xIdx = Csyr2VectorOffset(td.n, td.incx, i);
        const int64_t yIdx = Csyr2VectorOffset(td.n, td.incy, i);
        xRe[k] = x[xIdx].real;
        xIm[k] = x[xIdx].imag;
        yRe[k] = y[yIdx].real;
        yIm[k] = y[yIdx].imag;
    }
}

inline void PlaneMulAdd(
    float* aRe, float* aIm, const float* xRe, const float* xIm, const float* yRe, const float* yIm, float t1r,
    float t1i, float t2r, float t2i, float* zr1, float* zi1, float* zr2, float* zi2, int32_t len)
{
    /* fallback：zr = xr*tr - xi*ti；zi = xr*ti + xi*tr；再 Add 到 A 两平面。对角同一公式。 */
    for (int32_t k = 0; k < len; ++k) {
        zr1[k] = xRe[k] * t1r - xIm[k] * t1i; /* Mul / Sub */
        zi1[k] = xRe[k] * t1i + xIm[k] * t1r; /* Mul / Add */
        zr2[k] = yRe[k] * t2r - yIm[k] * t2i;
        zi2[k] = yRe[k] * t2i + yIm[k] * t2r;
        aRe[k] = aRe[k] + zr1[k] + zr2[k]; /* Add */
        aIm[k] = aIm[k] + zi1[k] + zi2[k];
    }
}

class Csyr2Kernel {
public:
    void Init(
        uint32_t blockIdx, uint32_t tilingKey, const Csyr2TilingData* tiling, const aclblasComplex* x,
        const aclblasComplex* y, aclblasComplex* A)
    {
        skip_ = true;
        if (tiling == nullptr || x == nullptr || y == nullptr || A == nullptr) {
            return;
        }
        td_ = *tiling;
        tilingKey_ = tilingKey;
        if (blockIdx >= td_.usedCoreNum) {
            return;
        }
        x_ = x;
        y_ = y;
        a_ = A;
        jBegin_ = static_cast<int32_t>(blockIdx);
        jEnd_ = td_.n;
        if (jBegin_ >= jEnd_) {
            return;
        }

        tileRow_ = (td_.tileRow > 0) ? static_cast<int32_t>(td_.tileRow) : 1;
        const bool upper = (td_.uplo == ACLBLAS_UPPER);
        unionLen_ = upper ? jEnd_ : (td_.n - jBegin_);
        if (unionLen_ < 0) {
            unionLen_ = 0;
        }
        xyReuse_ = (td_.xyReuse != 0);
        const int32_t xyCap = xyReuse_ ? (unionLen_ > 0 ? unionLen_ : 1) : tileRow_;
        constexpr int32_t bufNum = 2;

        /* UB：A double buffer = BUFFER_NUM * tileRow * 8，交织与平面共用同一块；tmpCost 与 A 不别名。 */
        for (int32_t b = 0; b < bufNum; ++b) {
            aStage_[static_cast<size_t>(b)].assign(static_cast<size_t>(tileRow_) * 2U, 0.0f);
        }
        xRe_.assign(static_cast<size_t>(xyCap), 0.0f);
        xIm_.assign(static_cast<size_t>(xyCap), 0.0f);
        yRe_.assign(static_cast<size_t>(xyCap), 0.0f);
        yIm_.assign(static_cast<size_t>(xyCap), 0.0f);
        winXRe_.assign(static_cast<size_t>(tileRow_), 0.0f);
        winXIm_.assign(static_cast<size_t>(tileRow_), 0.0f);
        winYRe_.assign(static_cast<size_t>(tileRow_), 0.0f);
        winYIm_.assign(static_cast<size_t>(tileRow_), 0.0f);
        zr1_.assign(static_cast<size_t>(tileRow_), 0.0f);
        zi1_.assign(static_cast<size_t>(tileRow_), 0.0f);
        zr2_.assign(static_cast<size_t>(tileRow_), 0.0f);
        zi2_.assign(static_cast<size_t>(tileRow_), 0.0f);
        skip_ = false;
    }

    void Process()
    {
        if (skip_) {
            return;
        }
        const bool upper = (td_.uplo == ACLBLAS_UPPER);
        if (xyReuse_ && unionLen_ > 0) {
            const int32_t i0 = upper ? 0 : jBegin_;
            GatherXY(x_, y_, td_, i0, unionLen_, xRe_.data(), xIm_.data(), yRe_.data(), yIm_.data(), tilingKey_);
        }

        int32_t ping = 0;
        const int32_t jStep = (td_.usedCoreNum > 0) ? static_cast<int32_t>(td_.usedCoreNum) : 1;
        for (int32_t j = jBegin_; j < jEnd_; j += jStep) {
            const int32_t len = upper ? (j + 1) : (td_.n - j);
            const int64_t gmBase = static_cast<int64_t>(j) * td_.lda + (upper ? 0 : j);
            const int32_t iBase = upper ? 0 : j;

            float xjr = 0.0f;
            float xji = 0.0f;
            float yjr = 0.0f;
            float yji = 0.0f;
            LoadLogicXY(j, &xjr, &xji, &yjr, &yji);
            float t1r = 0.0f;
            float t1i = 0.0f;
            float t2r = 0.0f;
            float t2i = 0.0f;
            Cmul(td_.alphaReal, td_.alphaImag, yjr, yji, &t1r, &t1i); /* temp1 = alpha * y[j] */
            Cmul(td_.alphaReal, td_.alphaImag, xjr, xji, &t2r, &t2i); /* temp2 = alpha * x[j] */

            for (int32_t off = 0; off < len; off += tileRow_) {
                const int32_t cur = MinI32(tileRow_, len - off);
                const int32_t buf = ping;
                ping ^= 1; /* BUFFER_NUM=2 double buffer */

                CopyIn(gmBase + off, cur, buf);
                const float* xr = nullptr;
                const float* xi = nullptr;
                const float* yr = nullptr;
                const float* yi = nullptr;
                BindXYWindow(iBase + off, cur, &xr, &xi, &yr, &yi);
                float* aPlanes = aStage_[static_cast<size_t>(buf)].data();
                PlaneMulAdd(
                    aPlanes, aPlanes + cur, xr, xi, yr, yi, t1r, t1i, t2r, t2i, zr1_.data(), zi1_.data(), zr2_.data(),
                    zi2_.data(), cur);
                CopyOut(gmBase + off, cur, buf);
            }
        }
    }

private:
    void LoadLogicXY(int32_t i, float* xr, float* xi, float* yr, float* yi) const
    {
        const bool upper = (td_.uplo == ACLBLAS_UPPER);
        if (xyReuse_) {
            const int32_t idx = upper ? i : (i - jBegin_);
            *xr = xRe_[static_cast<size_t>(idx)];
            *xi = xIm_[static_cast<size_t>(idx)];
            *yr = yRe_[static_cast<size_t>(idx)];
            *yi = yIm_[static_cast<size_t>(idx)];
            return;
        }
        const int64_t xIdx = Csyr2VectorOffset(td_.n, td_.incx, i);
        const int64_t yIdx = Csyr2VectorOffset(td_.n, td_.incy, i);
        *xr = x_[xIdx].real;
        *xi = x_[xIdx].imag;
        *yr = y_[yIdx].real;
        *yi = y_[yIdx].imag;
    }

    void BindXYWindow(int32_t i0, int32_t cur, const float** xr, const float** xi, const float** yr, const float** yi)
    {
        const bool upper = (td_.uplo == ACLBLAS_UPPER);
        if (xyReuse_) {
            const int32_t uoff = upper ? i0 : (i0 - jBegin_);
            *xr = xRe_.data() + uoff;
            *xi = xIm_.data() + uoff;
            *yr = yRe_.data() + uoff;
            *yi = yIm_.data() + uoff;
            return;
        }
        GatherXY(x_, y_, td_, i0, cur, winXRe_.data(), winXIm_.data(), winYRe_.data(), winYIm_.data(), tilingKey_);
        *xr = winXRe_.data();
        *xi = winXIm_.data();
        *yr = winYRe_.data();
        *yi = winYIm_.data();
    }

    void CopyIn(int64_t gmOff, int32_t cur, int32_t buf)
    {
        float* stage = aStage_[static_cast<size_t>(buf)].data();
        CopyComplex(reinterpret_cast<aclblasComplex*>(stage), a_ + gmOff, cur);
        UnpackInPlace(stage, cur, zr1_.data());
    }

    void CopyOut(int64_t gmOff, int32_t cur, int32_t buf)
    {
        float* stage = aStage_[static_cast<size_t>(buf)].data();
        PackInPlace(stage, cur, zr1_.data(), zi1_.data());
        CopyComplex(a_ + gmOff, reinterpret_cast<aclblasComplex*>(stage), cur);
    }

    Csyr2TilingData td_{};
    uint32_t tilingKey_{0};
    const aclblasComplex* x_{nullptr};
    const aclblasComplex* y_{nullptr};
    aclblasComplex* a_{nullptr};
    int32_t jBegin_{0};
    int32_t jEnd_{0};
    int32_t tileRow_{1};
    int32_t unionLen_{0};
    bool xyReuse_{false};
    bool skip_{true};
    std::vector<float> aStage_[2]; /* tileRow*2 floats = tileRow*8B；re/im 别名此块 */
    std::vector<float> xRe_;
    std::vector<float> xIm_;
    std::vector<float> yRe_;
    std::vector<float> yIm_;
    std::vector<float> winXRe_;
    std::vector<float> winXIm_;
    std::vector<float> winYRe_;
    std::vector<float> winYIm_;
    std::vector<float> zr1_;
    std::vector<float> zi1_;
    std::vector<float> zr2_;
    std::vector<float> zi2_;
};

} // namespace

void Csyr2KernelRun(
    uint32_t blockIdx, uint32_t tilingKey, const Csyr2TilingData* tiling, const aclblasComplex* x,
    const aclblasComplex* y, aclblasComplex* A)
{
    Csyr2Kernel op;
    op.Init(blockIdx, tilingKey, tiling, x, y, A);
    op.Process();
}

#endif /* !CSYR2_ASCENDC */

#if defined(CSYR2_ASCENDC)

/* x / y / A 一律按 float 绑定，下标 2k 实部、2k+1 虚部。
 * 与仓内 cgerc / chemm / cherk 一致：aclblasComplex 是聚合体，
 * DataCopyPadExtParams<T> 有 `T paddingValue = 0;`，用它做元素类型实例化不出来。
 * 实虚部拆分用 GatherMask（pattern 1 取偶、2 取奇），合并用一次 Gather 加常驻偏移表。 */

#include "csyr2_device.h"

// Keep the host tiling layout and reserved UB allocation at 64 bytes.
// Generic kernels capture only compute metadata as explicit scalar arguments.
static_assert(sizeof(Csyr2TilingData) == 16 * 4, "Csyr2TilingData host layout must remain 64 bytes");

// Small matrices use scalar arithmetic after a single vector staging step.
__aicore__ inline void Csyr2ComputeTinyRows(
    AscendC::LocalTensor<float> xv, AscendC::LocalTensor<float> yv, AscendC::LocalTensor<float> out, int first, int end,
    float cr, float ci, float dr, float di)
{
    for (int row = first; row < end; ++row) {
        const float xr = xv.GetValue(row * 2);
        const float xi = xv.GetValue(row * 2 + 1);
        const float yr = yv.GetValue(row * 2);
        const float yi = yv.GetValue(row * 2 + 1);
        float re = xr * cr;
        re -= xi * ci;
        re += yr * dr;
        re -= yi * di;
        float im = xi * cr;
        im += xr * ci;
        im += yi * dr;
        im += yr * di;
        out.SetValue((row - first) * 2, re);
        out.SetValue((row - first) * 2 + 1, im);
    }
}

extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_tiny(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int n, int lda, int triangle, float ar, float ai)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 192);
    auto xv = buffer.Get<float>();
    auto yv = xv[16];
    auto out = xv[32];
    AscendC::GlobalTensor<float> xg, yg, ag;
    xg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    AscendC::DataCopyExtParams read{1, static_cast<uint32_t>(n * 8), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pad(false, 0, 0, 0.0f);
    AscendC::DataCopyPad(xv, xg, read, pad);
    AscendC::DataCopyPad(yv, yg, read, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    bool pending = false;
    AscendC::SetAtomicAdd<float>();
    for (int col = AscendC::GetBlockIdx(); col < n; col += AscendC::GetBlockNum()) {
        if (pending)
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        const float xjr = xv.GetValue(col * 2), xji = xv.GetValue(col * 2 + 1);
        const float yjr = yv.GetValue(col * 2), yji = yv.GetValue(col * 2 + 1);
        const volatile float ay0 = ar * yjr, ay1 = ai * yji, ay2 = ar * yji, ay3 = ai * yjr;
        const volatile float ax0 = ar * xjr, ax1 = ai * xji, ax2 = ar * xji, ax3 = ai * xjr;
        const float cr = ay0 - ay1, ci = ay2 + ay3, dr = ax0 - ax1, di = ax2 + ax3;
        const int first = triangle == ACLBLAS_UPPER ? 0 : col;
        const int end = triangle == ACLBLAS_UPPER ? col + 1 : n;
        Csyr2ComputeTinyRows(xv, yv, out, first, end, cr, ci, dr, di);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyExtParams write{1, static_cast<uint32_t>((end - first) * 8), 0, 0, 0};
        AscendC::DataCopyPad(ag[(static_cast<int64_t>(col) * lda + first) * 2], out, write);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        pending = true;
    }
    if (pending)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetAtomicNone();
}

/* Atomic paths must drain all writes before restoring non-atomic mode. */
template <typename Kernel>
__aicore__ inline __attribute__((always_inline)) void Csyr2RunAtomic(Kernel& op)
{
    AscendC::SetAtomicAdd<float>();
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetAtomicNone();
}

extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_upper_full_unit(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, uint32_t tileRow, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_UPPER_UNIT, true> op;
    op.InitDirect(x, y, A, n, lda, tileRow, alphaReal, alphaImag);
    Csyr2RunAtomic(op);
}

extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_upper_full_window(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_UPPER_WINDOWED, true> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    Csyr2RunAtomic(op);
}

extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_0(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_UPPER_UNIT> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    Csyr2RunAtomic(op);
}

extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_lower_one(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, uint32_t tileRow, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_LOWER_UNIT, 2, true> op;
    op.InitDirect(x, y, A, n, lda, tileRow, alphaReal, alphaImag);
    Csyr2RunAtomic(op);
}

// Align the generic LOWER entry without changing its optimization level.
extern "C" __global__ __aicore__ __vector__ __attribute__((aligned(4096))) void csyr2_kernel_1(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_LOWER_UNIT> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    Csyr2RunAtomic(op);
}

extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_2(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_UPPER_STRIDED> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    Csyr2RunAtomic(op);
}

extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_3(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_LOWER_STRIDED> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    op.Process();
}

extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_4(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_UPPER_WINDOWED> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    Csyr2RunAtomic(op);
}

// Keep the windowed LOWER entry aligned independently of preceding kernels.
extern "C" __global__ __aicore__ __vector__ __attribute__((aligned(4096))) void csyr2_kernel_5(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_LOWER_WINDOWED> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    Csyr2RunAtomic(op);
}

// Keep the original entries ahead of this specialized full-window body.
extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_lower_full_window_2(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_LOWER_WINDOWED> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    AscendC::SetAtomicAdd<float>();
    op.ProcessFullLowerWindow<2>();
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetAtomicNone();
}

// Keep the original entries ahead of this specialized full-window body.
extern "C" __global__ __aicore__ __vector__ void csyr2_kernel_lower_full_window_4(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    Csyr2AscendKernel<CSYR2_TILING_KEY_LOWER_WINDOWED> op;
    op.Init(x, y, A, n, lda, incx, incy, usedCoreNum, tileRow, xyReuse, alphaReal, alphaImag);
    AscendC::SetAtomicAdd<float>();
    op.ProcessFullLowerWindow<4>();
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetAtomicNone();
}

/* host 侧一律用 uint8_t*：device 趟里 __gm__ 是实的，void* 与 __gm__ uint8_t* 之间不能转换。
 * 到 <<<>>> 处由编译器隐式转成 __gm__，与 ssyr2_kernel_do 的写法一致。 */
#if defined(__CCE_KT_TEST__) || defined(ASCENDC_CPU_DEBUG)
static void LaunchCsyr2ByKeyCpu(
    uint32_t tilingKey, uint32_t usedCoreNum, uint8_t* xGm, uint8_t* yGm, uint8_t* aGm, const Csyr2TilingData& tiling)
{
    if (tilingKey == CSYR2_TILING_KEY_UPPER_UNIT) {
        ICPU_RUN_KF(
            csyr2_kernel_0, usedCoreNum, xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy,
            tiling.usedCoreNum, tiling.tileRow, tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else if (tilingKey == CSYR2_TILING_KEY_LOWER_UNIT) {
        ICPU_RUN_KF(
            csyr2_kernel_1, usedCoreNum, xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy,
            tiling.usedCoreNum, tiling.tileRow, tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else if (tilingKey == CSYR2_TILING_KEY_UPPER_STRIDED) {
        ICPU_RUN_KF(
            csyr2_kernel_2, usedCoreNum, xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy,
            tiling.usedCoreNum, tiling.tileRow, tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else if (tilingKey == CSYR2_TILING_KEY_LOWER_STRIDED) {
        ICPU_RUN_KF(
            csyr2_kernel_3, usedCoreNum, xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy,
            tiling.usedCoreNum, tiling.tileRow, tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else if (tilingKey == CSYR2_TILING_KEY_UPPER_WINDOWED) {
        ICPU_RUN_KF(
            csyr2_kernel_4, usedCoreNum, xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy,
            tiling.usedCoreNum, tiling.tileRow, tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else {
        ICPU_RUN_KF(
            csyr2_kernel_5, usedCoreNum, xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy,
            tiling.usedCoreNum, tiling.tileRow, tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    }
}
#else
static void LaunchCsyr2ByKeyNpu(
    uint32_t tilingKey, uint32_t usedCoreNum, void* stream, uint8_t* xGm, uint8_t* yGm, uint8_t* aGm,
    const Csyr2TilingData& tiling)
{
    if (tilingKey == CSYR2_TILING_KEY_UPPER_UNIT) {
        csyr2_kernel_0<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.usedCoreNum, tiling.tileRow,
            tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else if (tilingKey == CSYR2_TILING_KEY_LOWER_UNIT) {
        csyr2_kernel_1<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.usedCoreNum, tiling.tileRow,
            tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else if (tilingKey == CSYR2_TILING_KEY_UPPER_STRIDED) {
        csyr2_kernel_2<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.usedCoreNum, tiling.tileRow,
            tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else if (tilingKey == CSYR2_TILING_KEY_LOWER_STRIDED) {
        csyr2_kernel_3<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.usedCoreNum, tiling.tileRow,
            tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else if (tilingKey == CSYR2_TILING_KEY_UPPER_WINDOWED) {
        csyr2_kernel_4<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.usedCoreNum, tiling.tileRow,
            tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else {
        csyr2_kernel_5<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.usedCoreNum, tiling.tileRow,
            tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    }
}
#endif

static void LaunchCsyr2ByKey(
    uint32_t tilingKey, uint32_t usedCoreNum, void* stream, uint8_t* xGm, uint8_t* yGm, uint8_t* aGm,
    const Csyr2TilingData& tiling)
{
#if defined(__CCE_KT_TEST__) || defined(ASCENDC_CPU_DEBUG)
    LaunchCsyr2ByKeyCpu(tilingKey, usedCoreNum, xGm, yGm, aGm, tiling);
#else
    LaunchCsyr2ByKeyNpu(tilingKey, usedCoreNum, stream, xGm, yGm, aGm, tiling);
#endif
}

#endif /* CSYR2_ASCENDC */

#if defined(CSYR2_ASCENDC) && !defined(__CCE_KT_TEST__) && !defined(ASCENDC_CPU_DEBUG)
static bool TryLaunchCsyr2Tiny(
    void* stream, const aclblasComplex* x, const aclblasComplex* y, aclblasComplex* A, const Csyr2TilingData* tiling)
{
    if (tiling->n > 0 && tiling->n <= 8 && tiling->incx == 1 && tiling->incy == 1 && tiling->alphaReal == 1.0f &&
        tiling->alphaImag == 0.0f) {
        const uint32_t physicalCores = GetAivCoreNum();
        const uint32_t blocks = std::min(static_cast<uint32_t>(tiling->n), physicalCores);
        csyr2_kernel_tiny<<<blocks, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(A), tiling->n,
            tiling->lda, tiling->uplo, tiling->alphaReal, tiling->alphaImag);
        return true;
    }
    return false;
}

static bool TryLaunchCsyr2UpperFull(
    void* stream, uint32_t tilingKey, uint32_t usedCoreNum, const aclblasComplex* x, const aclblasComplex* y,
    aclblasComplex* A, const Csyr2TilingData* tiling)
{
    if ((tilingKey == CSYR2_TILING_KEY_UPPER_UNIT || tilingKey == CSYR2_TILING_KEY_UPPER_WINDOWED) &&
        tiling->n >= 640 && tiling->n <= 1408 && tiling->incx == 1 && tiling->incy == 1 && tiling->xyReuse != 0 &&
        tiling->tileRow >= static_cast<uint32_t>(tiling->n)) {
        csyr2_kernel_upper_full_unit<<<usedCoreNum, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(A), tiling->n,
            tiling->lda, tiling->tileRow, tiling->alphaReal, tiling->alphaImag);
        return true;
    }
    return false;
}

static bool TryLaunchCsyr2LowerOne(
    void* stream, uint32_t tilingKey, uint32_t usedCoreNum, const aclblasComplex* x, const aclblasComplex* y,
    aclblasComplex* A, const Csyr2TilingData* tiling)
{
    if (tilingKey == CSYR2_TILING_KEY_LOWER_UNIT && tiling->n >= 640 && tiling->n <= 1408 && tiling->incx == 1 &&
        tiling->incy == 1 && tiling->xyReuse != 0 && tiling->tileRow >= static_cast<uint32_t>(tiling->n) &&
        tiling->alphaReal == 1.0f && tiling->alphaImag == 0.0f) {
        csyr2_kernel_lower_one<<<usedCoreNum, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(A), tiling->n,
            tiling->lda, tiling->tileRow, tiling->alphaReal, tiling->alphaImag);
        return true;
    }
    return false;
}

#endif

#if defined(CSYR2_ASCENDC)
static int32_t Csyr2LowerFullWindowBatch(const Csyr2TilingData& tiling)
{
    if (tiling.n <= 0 || tiling.tileRow > 0x7fffffffU ||
        tiling.tileRow < static_cast<uint32_t>(tiling.n)) {
        return 1;
    }
    // Match the window path's InitVectorViews clamp and vector-width rounding.
    const uint64_t alignedN = (static_cast<uint64_t>(tiling.n) + 3u) & ~3ULL;
    const uint64_t rows = std::min(static_cast<uint64_t>(tiling.tileRow), alignedN);
    const uint64_t ilvBytes = ((rows * 2u + 63u) / 64u) * 64u * sizeof(float);
    const uint64_t planeBytes = ((rows + 63u) / 64u) * 64u * sizeof(float);
    // A single buffer larger than UB cannot satisfy the shared batch formula.
    // Check before narrowing so large legal inputs retain the generic fallback.
    if (rows > 0x7fffffffULL || ilvBytes > 192u * 1024u || planeBytes > 192u * 1024u) {
        return 1;
    }
    return Csyr2SelectWindowBatchCount(
        tiling.n, static_cast<int32_t>(rows), static_cast<uint32_t>(ilvBytes),
        static_cast<uint32_t>(planeBytes), false);
}

static bool TryLaunchCsyr2LowerFullWindow(
    void* stream, uint32_t tilingKey, uint32_t usedCoreNum, uint8_t* xGm, uint8_t* yGm, uint8_t* aGm,
    const Csyr2TilingData& tiling)
{
    if (tilingKey != CSYR2_TILING_KEY_LOWER_WINDOWED) {
        return false;
    }
    const int32_t batch = Csyr2LowerFullWindowBatch(tiling);
    if (batch < 2) {
        return false;
    }
#if defined(__CCE_KT_TEST__) || defined(ASCENDC_CPU_DEBUG)
    if (batch == 4) {
        ICPU_RUN_KF(
            csyr2_kernel_lower_full_window_4, usedCoreNum, xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx,
            tiling.incy, tiling.usedCoreNum, tiling.tileRow, tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else {
        ICPU_RUN_KF(
            csyr2_kernel_lower_full_window_2, usedCoreNum, xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx,
            tiling.incy, tiling.usedCoreNum, tiling.tileRow, tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    }
#else
    if (batch == 4) {
        csyr2_kernel_lower_full_window_4<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.usedCoreNum, tiling.tileRow,
            tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    } else {
        csyr2_kernel_lower_full_window_2<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.usedCoreNum, tiling.tileRow,
            tiling.xyReuse, tiling.alphaReal, tiling.alphaImag);
    }
#endif
    return true;
}

static void LaunchCsyr2WithTiling(
    void* stream, uint32_t tilingKey, uint32_t usedCoreNum, uint8_t* xGm, uint8_t* yGm, uint8_t* aGm,
    const Csyr2TilingData* tiling)
{
    if (TryLaunchCsyr2LowerFullWindow(stream, tilingKey, usedCoreNum, xGm, yGm, aGm, *tiling)) {
        return;
    }
#if !defined(__CCE_KT_TEST__) && !defined(ASCENDC_CPU_DEBUG)
    const bool fullUpper = tiling->uplo == ACLBLAS_UPPER && tiling->incx == 1 && tiling->incy == 1 &&
                           tiling->n >= 640 && tiling->n <= 1728 && tiling->tileRow >= static_cast<uint32_t>(tiling->n);
    if (fullUpper && tilingKey == CSYR2_TILING_KEY_UPPER_WINDOWED) {
        csyr2_kernel_upper_full_window<<<usedCoreNum, nullptr, stream>>>(
            xGm, yGm, aGm, tiling->n, tiling->lda, tiling->incx, tiling->incy, tiling->usedCoreNum,
            tiling->tileRow, tiling->xyReuse, tiling->alphaReal, tiling->alphaImag);
    } else
#endif
    {
        LaunchCsyr2ByKey(tilingKey, usedCoreNum, stream, xGm, yGm, aGm, *tiling);
    }
}
#endif

extern "C" aclblasStatus_t LaunchCsyr2Kernel(
    void* stream, uint32_t tilingKey, uint32_t usedCoreNum, const aclblasComplex* x, const aclblasComplex* y,
    aclblasComplex* A, const Csyr2TilingData* tiling)
{
    /* 合入 ops-blas 时与同族 host 直调对齐：kernel<<<usedCoreNum, nullptr, stream>>>(...)。
     * SetBlockDim(usedCoreNum) 按 AIV。Host 不在此同步 stream。 */
    if (tiling == nullptr || usedCoreNum == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
#if defined(CSYR2_ASCENDC)
#if !defined(__CCE_KT_TEST__) && !defined(ASCENDC_CPU_DEBUG)
    if (TryLaunchCsyr2Tiny(stream, x, y, A, tiling)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
#endif
#if !defined(__CCE_KT_TEST__) && !defined(ASCENDC_CPU_DEBUG)
    if (TryLaunchCsyr2UpperFull(stream, tilingKey, usedCoreNum, x, y, A, tiling)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (TryLaunchCsyr2LowerOne(stream, tilingKey, usedCoreNum, x, y, A, tiling)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
#endif
    uint8_t* xGm = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y));
    uint8_t* aGm = reinterpret_cast<uint8_t*>(A);
    LaunchCsyr2WithTiling(stream, tilingKey, usedCoreNum, xGm, yGm, aGm, tiling);
    return ACLBLAS_STATUS_SUCCESS;
#else
    (void)stream;
    for (uint32_t c = 0; c < usedCoreNum; ++c) {
        Csyr2KernelRun(c, tilingKey, tiling, x, y, A);
    }
    return ACLBLAS_STATUS_SUCCESS;
#endif
}
