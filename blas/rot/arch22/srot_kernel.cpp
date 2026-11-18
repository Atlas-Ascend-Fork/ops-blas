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
#include "srot_kernel.h"

using namespace AscendC;

namespace {
constexpr uint32_t SLOT = 8;
constexpr uint32_t VECTOR_LANES = 64;

template <bool STRIDED, bool SERIAL = false>
class SrotKernel {
public:
    __aicore__ inline void Init(
        __gm__ float* x, __gm__ float* y, const __gm__ float* c, const __gm__ float* s, SrotTilingData tiling)
    {
        t_ = tiling;
        offset_ = GetBlockIdx() * t_.blockLength;
        count_ = offset_ < t_.totalN ? t_.totalN - offset_ : 0;
        count_ = count_ < t_.blockLength ? count_ : t_.blockLength;
        x_.SetGlobalBuffer(x);
        y_.SetGlobalBuffer(y);
        c_ = t_.cIsDevice ? *c : t_.c;
        s_ = t_.sIsDevice ? *s : t_.s;
        const uint32_t bytes = t_.tileLength * (STRIDED ? SLOT : 1U) * sizeof(float);
        pipe_.InitBuffer(inX_, 2, bytes);
        pipe_.InitBuffer(inY_, 2, bytes);
        pipe_.InitBuffer(outX_, 2, bytes);
        pipe_.InitBuffer(outY_, 2, bytes);
        pipe_.InitBuffer(work_, bytes);
        if constexpr (STRIDED) {
            pipe_.InitBuffer(indices_, bytes);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t done = 0; done < count_; done += t_.tileLength) {
            const uint32_t left = count_ - done;
            const uint32_t n = left < t_.tileLength ? left : t_.tileLength;
            CopyIn(offset_ + done, n);
            Compute(n);
            CopyOut(offset_ + done, n);
        }
    }

private:
    __aicore__ inline uint64_t PhysicalStart(uint32_t logical, uint32_t n, int32_t inc)
    {
        return inc > 0 ? static_cast<uint64_t>(logical) * inc :
                         static_cast<uint64_t>(t_.totalN - logical - n) * (-static_cast<int64_t>(inc));
    }

    __aicore__ inline void Load(
        LocalTensor<float> dst, GlobalTensor<float> src, uint32_t logical, uint32_t n, int32_t inc)
    {
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        if constexpr (!STRIDED) {
            DataCopyPad(dst, src[logical], DataCopyExtParams{1, n * 4U, 0, 0, 0}, pad);
        } else {
            const uint64_t stride = inc > 0 ? inc : -static_cast<int64_t>(inc);
            const uint64_t start = PhysicalStart(logical, n, inc);
            const uint64_t gap = (stride - 1) * sizeof(float);
            if (gap <= UINT32_MAX) {
                DataCopyPad(
                    dst, src[start], DataCopyExtParams{static_cast<uint16_t>(n), 4, static_cast<uint32_t>(gap), 0, 0},
                    pad);
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    DataCopyPad(dst[i * SLOT], src[start + i * stride], DataCopyExtParams{1, 4, 0, 0, 0}, pad);
                }
            }
        }
    }

    __aicore__ inline void Store(
        GlobalTensor<float> dst, LocalTensor<float> src, uint32_t logical, uint32_t n, int32_t inc)
    {
        if constexpr (!STRIDED) {
            DataCopyPad(dst[logical], src, DataCopyExtParams{1, n * 4U, 0, 0, 0});
        } else {
            const uint64_t stride = inc > 0 ? inc : -static_cast<int64_t>(inc);
            const uint64_t start = PhysicalStart(logical, n, inc);
            const uint64_t gap = (stride - 1) * sizeof(float);
            if (gap <= UINT32_MAX) {
                DataCopyPad(
                    dst[start], src, DataCopyExtParams{static_cast<uint16_t>(n), 4, 0, static_cast<uint32_t>(gap), 0});
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    DataCopyPad(dst[start + i * stride], src[i * SLOT], DataCopyExtParams{1, 4, 0, 0, 0});
                }
            }
        }
    }

    __aicore__ inline void CopyIn(uint32_t logical, uint32_t n)
    {
        auto x = inX_.AllocTensor<float>();
        auto y = inY_.AllocTensor<float>();
        Load(x, x_, logical, n, t_.incx);
        Load(y, y_, logical, n, t_.incy);
        inX_.EnQue(x);
        inY_.EnQue(y);
    }

    __aicore__ inline void PrepareIndices(uint32_t n)
    {
        if constexpr (STRIDED) {
            if ((t_.incx < 0 || t_.incy < 0) && indexCount_ != n) {
                auto event = pipe_.FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(event);
                WaitFlag<HardEvent::V_S>(event);
                auto idx = indices_.Get<uint32_t>();
                for (uint32_t i = 0; i < n * SLOT; ++i) {
                    idx.SetValue(i, (n - 1 - i / SLOT) * SLOT * sizeof(float));
                }
                event = pipe_.FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(event);
                WaitFlag<HardEvent::S_V>(event);
                indexCount_ = n;
            }
        }
    }

    __aicore__ inline LocalTensor<float> ReverseIfNeeded(LocalTensor<float> src, int32_t inc, uint32_t lanes)
    {
        if constexpr (STRIDED) {
            if (inc < 0) {
                auto tmp = work_.Get<float>();
                Gather(tmp, src, indices_.Get<uint32_t>(), 0U, lanes);
                PipeBarrier<PIPE_V>();
                return tmp;
            }
        }
        return src;
    }

    __aicore__ inline void ReverseOutput(LocalTensor<float> dst, int32_t inc, uint32_t lanes)
    {
        if constexpr (STRIDED) {
            if (inc < 0) {
                auto reversed = ReverseIfNeeded(dst, inc, lanes);
                Copy(
                    dst, reversed, VECTOR_LANES, (lanes + VECTOR_LANES - 1) / VECTOR_LANES,
                    CopyRepeatParams{1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void Compute(uint32_t n)
    {
        auto x = inX_.DeQue<float>();
        auto y = inY_.DeQue<float>();
        auto newX = outX_.AllocTensor<float>();
        auto newY = outY_.AllocTensor<float>();
        const uint32_t lanes = n * (STRIDED ? SLOT : 1U);
        PrepareIndices(n);
        auto oldX = ReverseIfNeeded(x, t_.incx, lanes);
        Muls(newX, oldX, c_, lanes);
        Muls(newY, oldX, s_, lanes);
        PipeBarrier<PIPE_V>();
        auto oldY = ReverseIfNeeded(y, t_.incy, lanes);
        // Both products of old x are complete before reusing its input tile.
        Muls(x, oldY, c_, lanes);
        PipeBarrier<PIPE_V>();
        Muls(y, oldY, s_, lanes);
        PipeBarrier<PIPE_V>();
        Add(newX, newX, y, lanes);
        Sub(newY, x, newY, lanes);
        PipeBarrier<PIPE_V>();
        ReverseOutput(newX, t_.incx, lanes);
        ReverseOutput(newY, t_.incy, lanes);
        outX_.EnQue(newX);
        outY_.EnQue(newY);
        inX_.FreeTensor(x);
        inY_.FreeTensor(y);
    }

    __aicore__ inline void CopyOut(uint32_t logical, uint32_t n)
    {
        auto x = outX_.DeQue<float>();
        auto y = outY_.DeQue<float>();
        Store(x_, x, logical, n, t_.incx);
        Store(y_, y, logical, n, t_.incy);
        outX_.FreeTensor(x);
        outY_.FreeTensor(y);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inX_, inY_;
    TQue<TPosition::VECOUT, 2> outX_, outY_;
    TBuf<TPosition::VECCALC> work_, indices_;
    GlobalTensor<float> x_, y_;
    SrotTilingData t_;
    uint32_t offset_ = 0, count_ = 0, indexCount_ = 0;
    float c_ = 0, s_ = 0;
};

// A zero stride reuses the previous iteration's result, so the whole vector is
// processed by one core in Netlib order. Tiles only batch GM transfers; the
// recurrence itself remains serial and uses separately rounded FP32 products.
template <>
class SrotKernel<true, true> {
public:
    __aicore__ inline void Init(
        __gm__ float* x, __gm__ float* y, const __gm__ float* c, const __gm__ float* s, SrotTilingData tiling)
    {
        t_ = tiling;
        x_.SetGlobalBuffer(x);
        y_.SetGlobalBuffer(y);
        c_ = t_.cIsDevice ? *c : t_.c;
        s_ = t_.sIsDevice ? *s : t_.s;
        const uint32_t bytes = t_.tileLength * SLOT * sizeof(float);
        pipe_.InitBuffer(inputX_, bytes);
        pipe_.InitBuffer(inputY_, bytes);
        pipe_.InitBuffer(products_, 4U * SLOT * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto x = inputX_.Get<float>();
        auto y = inputY_.Get<float>();
        for (uint32_t logical = 0; logical < t_.totalN; logical += t_.tileLength) {
            const uint32_t left = t_.totalN - logical;
            const uint32_t count = left < t_.tileLength ? left : t_.tileLength;
            Load(x, x_, logical, count, t_.incx);
            Load(y, y_, logical, count, t_.incy);
            Sync<HardEvent::MTE2_V>();
            Compute(x, y, count);
            Sync<HardEvent::V_MTE3>();
            Store(x_, x, logical, count, t_.incx);
            Store(y_, y, logical, count, t_.incy);
            // The next tile must read the newly written repeated element.
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

private:
    template <HardEvent EVENT>
    __aicore__ inline void Sync()
    {
        const auto event = pipe_.FetchEventID(EVENT);
        SetFlag<EVENT>(event);
        WaitFlag<EVENT>(event);
    }

    __aicore__ inline uint64_t PhysicalStart(uint32_t logical, uint32_t count, int32_t inc)
    {
        return inc >= 0 ? static_cast<uint64_t>(logical) * inc :
                          static_cast<uint64_t>(t_.totalN - logical - count) * (-static_cast<int64_t>(inc));
    }

    __aicore__ inline void Load(
        LocalTensor<float> dst, GlobalTensor<float> src, uint32_t logical, uint32_t count, int32_t inc)
    {
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        if (inc == 0) {
            DataCopyPad(dst, src, DataCopyExtParams{1, sizeof(float), 0, 0, 0}, pad);
            return;
        }
        const uint64_t stride = inc > 0 ? inc : -static_cast<int64_t>(inc);
        const uint64_t start = PhysicalStart(logical, count, inc);
        const uint64_t gap = (stride - 1) * sizeof(float);
        if (gap <= UINT32_MAX) {
            DataCopyPad(
                dst, src[start],
                DataCopyExtParams{static_cast<uint16_t>(count), sizeof(float), static_cast<uint32_t>(gap), 0, 0}, pad);
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                DataCopyPad(dst[i * SLOT], src[start + i * stride], DataCopyExtParams{1, sizeof(float), 0, 0, 0}, pad);
            }
        }
    }

    __aicore__ inline void Store(
        GlobalTensor<float> dst, LocalTensor<float> src, uint32_t logical, uint32_t count, int32_t inc)
    {
        if (inc == 0) {
            DataCopyPad(dst, src, DataCopyExtParams{1, sizeof(float), 0, 0, 0});
            return;
        }
        const uint64_t stride = inc > 0 ? inc : -static_cast<int64_t>(inc);
        const uint64_t start = PhysicalStart(logical, count, inc);
        const uint64_t gap = (stride - 1) * sizeof(float);
        if (gap <= UINT32_MAX) {
            DataCopyPad(
                dst[start], src,
                DataCopyExtParams{static_cast<uint16_t>(count), sizeof(float), 0, static_cast<uint32_t>(gap), 0});
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                DataCopyPad(dst[start + i * stride], src[i * SLOT], DataCopyExtParams{1, sizeof(float), 0, 0, 0});
            }
        }
    }

    __aicore__ inline uint32_t LocalOffset(uint32_t logical, uint32_t count, int32_t inc)
    {
        if (inc == 0) {
            return 0;
        }
        return (inc > 0 ? logical : count - 1 - logical) * SLOT;
    }

    __aicore__ inline void Compute(LocalTensor<float> x, LocalTensor<float> y, uint32_t count)
    {
        auto products = products_.Get<float>();
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t ix = LocalOffset(i, count, t_.incx);
            const uint32_t iy = LocalOffset(i, count, t_.incy);
            Muls(products, x[ix], c_, 1);
            Muls(products[SLOT], y[iy], c_, 1);
            Muls(products[2U * SLOT], x[ix], s_, 1);
            Muls(products[3U * SLOT], y[iy], s_, 1);
            PipeBarrier<PIPE_V>();
            Add(x[ix], products, products[3U * SLOT], 1);
            Sub(y[iy], products[SLOT], products[2U * SLOT], 1);
            PipeBarrier<PIPE_V>();
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputX_;
    TBuf<TPosition::VECCALC> inputY_;
    TBuf<TPosition::VECCALC> products_;
    GlobalTensor<float> x_;
    GlobalTensor<float> y_;
    SrotTilingData t_{};
    float c_ = 0.0f;
    float s_ = 0.0f;
};
} // namespace

extern "C" __global__ __aicore__ void srot_contiguous_kernel(
    __gm__ float* x, __gm__ float* y, const __gm__ float* c, const __gm__ float* s, SrotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SrotKernel<false> op;
    op.Init(x, y, c, s, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void srot_strided_kernel(
    __gm__ float* x, __gm__ float* y, const __gm__ float* c, const __gm__ float* s, SrotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SrotKernel<true> op;
    op.Init(x, y, c, s, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void srot_serial_kernel(
    __gm__ float* x, __gm__ float* y, const __gm__ float* c, const __gm__ float* s, SrotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SrotKernel<true, true> op;
    op.Init(x, y, c, s, tiling);
    op.Process();
}

void srot_kernel_do(
    float* x, float* y, const float* c, const float* s, uint32_t blocks, SrotTilingData tiling, void* stream)
{
    if (tiling.incx == 0 || tiling.incy == 0) {
        srot_serial_kernel<<<1, nullptr, stream>>>(x, y, c, s, tiling);
    } else if (tiling.incx == 1 && tiling.incy == 1) {
        srot_contiguous_kernel<<<blocks, nullptr, stream>>>(x, y, c, s, tiling);
    } else {
        srot_strided_kernel<<<blocks, nullptr, stream>>>(x, y, c, s, tiling);
    }
}
