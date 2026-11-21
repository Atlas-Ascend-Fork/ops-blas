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
#include "acl/acl.h"
#include "srotg_tiling_data.h"

using namespace AscendC;

namespace {

constexpr uint32_t SROTG_DATA_COUNT = 1;
constexpr float SROTG_ZERO = 0.0f;
constexpr float SROTG_ONE = 1.0f;
constexpr float SROTG_SAFMIN = 1.1754943508222875e-38f;
constexpr float SROTG_SAFMAX = 1.7014118346046923e+38f;

class SrotgKernel {
public:
    __aicore__ inline SrotgKernel() = default;
    __aicore__ inline void Init(const SrotgTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline float Abs(float value) const;
    __aicore__ inline float Max(float lhs, float rhs) const;
    __aicore__ inline float Min(float lhs, float rhs) const;
    __aicore__ inline float Sign(float value) const;
    __aicore__ inline float DivideOr(float numerator, float denominator, float fallback) const;
    __aicore__ inline void WriteResult(float r, float z, float c, float s);

    GlobalTensor<float> aGM;
    GlobalTensor<float> bGM;
    GlobalTensor<float> cGM;
    GlobalTensor<float> sGM;
};

__aicore__ inline void SrotgKernel::Init(const SrotgTilingData& tiling)
{
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.a), SROTG_DATA_COUNT);
    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.b), SROTG_DATA_COUNT);
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.c), SROTG_DATA_COUNT);
    sGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.s), SROTG_DATA_COUNT);
}

__aicore__ inline float SrotgKernel::Abs(float value) const { return value < SROTG_ZERO ? -value : value; }

__aicore__ inline float SrotgKernel::Max(float lhs, float rhs) const { return lhs > rhs ? lhs : rhs; }

__aicore__ inline float SrotgKernel::Min(float lhs, float rhs) const { return lhs < rhs ? lhs : rhs; }

__aicore__ inline float SrotgKernel::Sign(float value) const { return value < SROTG_ZERO ? -SROTG_ONE : SROTG_ONE; }

__aicore__ inline float SrotgKernel::DivideOr(float numerator, float denominator, float fallback) const
{
    return denominator == 0.0f ? fallback : numerator / denominator;
}

__aicore__ inline void SrotgKernel::WriteResult(float r, float z, float c, float s)
{
    aGM.SetValue(0, r);
    bGM.SetValue(0, z);
    cGM.SetValue(0, c);
    sGM.SetValue(0, s);
}

__aicore__ inline void SrotgKernel::Process()
{
    const float aValue = aGM.GetValue(0);
    const float bValue = bGM.GetValue(0);
    const float absA = Abs(aValue);
    const float absB = Abs(bValue);

    if (absA == SROTG_ZERO && absB == SROTG_ZERO) {
        WriteResult(SROTG_ZERO, SROTG_ZERO, SROTG_ONE, SROTG_ZERO);
        return;
    }

    const float scale = Min(SROTG_SAFMAX, Max(SROTG_SAFMIN, Max(absA, absB)));
    const float scaledA = aValue / scale;
    const float scaledB = bValue / scale;
    const float sigma = absA > absB ? Sign(aValue) : Sign(bValue);
    const float normSq = scaledA * scaledA + scaledB * scaledB;
    const float r = sigma * (scale * sqrt(normSq));

    if (r == SROTG_ZERO) {
        WriteResult(r, SROTG_ZERO, SROTG_ONE, SROTG_ZERO);
        return;
    }

    const float c = aValue / r;
    const float s = bValue / r;
    const float z = absA > absB ? s : DivideOr(SROTG_ONE, c, SROTG_ONE);
    WriteResult(r, z, c, s);
}

} // namespace

extern "C" __global__ __aicore__ void srotg_kernel(const SrotgTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SrotgKernel op;
    op.Init(tiling);
    op.Process();
}

aclError srotg_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* s, const SrotgTilingData& tiling, uint32_t numBlocks, void* stream)
{
    // issue #1152: report the launch status instead of discarding it, mirroring the
    // chbmv/ssbmv pattern (aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs).
    (void)a;
    (void)b;
    (void)c;
    (void)s;
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(srotg_kernel), &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        SrotgTilingData tiling;
    } args{tiling};
    static_assert(sizeof(Arguments) == sizeof(SrotgTilingData));
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}
