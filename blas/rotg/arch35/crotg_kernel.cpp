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
#include <cmath>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "crotg_common.h"
#include "crotg_kernel.h"

using namespace AscendC;

constexpr uint32_t CROTG_SIMT_THREAD_NUM = SIMT_MIN_THREAD_NUM;

// SIMT compute function — only thread 0 performs the complex Givens rotation parameter generation.
// The algorithm body is defined by the CROTG_ALGO macro in crotg_common.h (shared with host).
__simt_vf__ __aicore__ LAUNCH_BOUND(CROTG_SIMT_THREAD_NUM)
inline void crotg_simt_compute(__gm__ float* aGm, __gm__ float* bGm,
                                __gm__ float* cGm, __gm__ float* sGm)
{
    if (threadIdx.x != 0) {
        return;
    }

    // complex64 stored as two consecutive floats: [real, imag]
    float ar = aGm[0];
    float ai = aGm[1];
    float br = bGm[0];
    float bi = bGm[1];

    float cr, sr, si, rr, ri;
    CROTG_ALGO(ar, ai, br, bi, &cr, &sr, &si, &rr, &ri);

    // Write back: a overwritten with r, c is real, s is complex, b not overwritten
    aGm[0] = rr;
    aGm[1] = ri;
    cGm[0] = cr;
    sGm[0] = sr;
    sGm[1] = si;
}

extern "C" __global__ __aicore__ void crotg_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<crotg_simt_compute>(
        dim3{CROTG_SIMT_THREAD_NUM, 1, 1},
        reinterpret_cast<__gm__ float*>(a),
        reinterpret_cast<__gm__ float*>(b),
        reinterpret_cast<__gm__ float*>(c),
        reinterpret_cast<__gm__ float*>(s));
}

void crotg_kernel_do(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* s, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    crotg_kernel<<<1, nullptr, aclStream>>>(a, b, c, s);
}
