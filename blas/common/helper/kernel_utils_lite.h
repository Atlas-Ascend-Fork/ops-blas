/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

/* ========== Tiling utilities ========== */

#define CONST_2 2

#ifndef __force_inline__
#define __force_inline__ inline __attribute__((always_inline))
#endif

#include <limits>
#include <type_traits>

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#else
#include "kernel_macros.h"
#endif

template <uint32_t ALIGN, typename T = uint32_t>
inline __aicore__ T RoundUp(const T val)
{
    static_assert(ALIGN != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T align = ALIGN;
    if (val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline __aicore__ T RoundUpPow2(T v)
{
    static_assert(std::is_unsigned<T>::value, "RoundUpPow2 requires unsigned type");
    if (v == 0) {
        return 1;
    }
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    if (sizeof(T) >= 4) {
        v |= v >> 16;
    }
    if (sizeof(T) >= 8) {
        v |= v >> 32;
    }
    v++;
    return v;
}

template <uint32_t DIVISOR, typename T = uint32_t>
inline __aicore__ T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T divisor = DIVISOR;
    if (dividend + divisor - 1 < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T>
inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (divisor == 0 || dividend + divisor - 1 < dividend) {
        return T_MAX<T>;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline T Min(const T lhs, const T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint32_t BlockSize()
{
    return 32 / sizeof(Dtype);
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint32_t MatrixSize()
{
    return 512 / sizeof(Dtype);
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t BlockSizeRoundUp(uint64_t num)
{
    return (num + BlockSize<Dtype>() - 1) / BlockSize<Dtype>() * BlockSize<Dtype>();
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t NumBlocksRoundUp(uint64_t num)
{
    return (num + BlockSize<Dtype>() - 1) / BlockSize<Dtype>();
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t MatrixSizeRoundUp(uint64_t num)
{
    return (num + MatrixSize<Dtype>() - 1) / MatrixSize<Dtype>() * MatrixSize<Dtype>();
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t NumMatrixsRoundUp(uint64_t num)
{
    return (num + MatrixSize<Dtype>() - 1) / MatrixSize<Dtype>();
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t L0HalfSize()
{
    return 32 * 1024 / sizeof(Dtype);
}

/* ========== Pipe / sync utilities ========== */

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<PIPE_##pipe>()
