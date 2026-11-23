/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: SIMT arch35 Kernel 实现
// 标准三层结构：
//   Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop）
//   Layer 2: __global__ 调度器（读 tiling → asc_vf_call 分发）
//   Layer 3: _kernel_do 启动器（<<<>>> 语法）
//
// 与 SIMD membase 的关键区别：
//   - 无 TPipe/TQue/DataCopyPad，直接通过 __gm__ 指针访问 GM
//   - 线程级并行：threadIdx.x / blockDim.x / blockIdx.x / gridDim.x
//   - 同步：asc_syncthreads()（线程间屏障）
//   - 无类封装，使用自由函数

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "cann_ops_blas_common.h"
#include "{{op}}_kernel.h"

// TEMPLATE: 计算函数
// - __simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) 装饰器（必需）
// - 可选模板参数做编译期分支（如 <bool UPLO_IS_UPPER>、<bool TRANS_IS_N>）
// - 参数为标量 + __gm__ 指针（不用 GlobalTensor）
// - grid-stride loop: for (row = blockIdx.x*blockDim.x+threadIdx.x; row<n; row += gridDim.x*blockDim.x)
template </* bool COMPILE_TIME_FLAG */>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void
{
    {
        Op
    }
}
SimtCompute(
    /* 标量参数: uint32_t n, float alpha, float beta, ... */
    /* GM 指针: __gm__ const float* inputGm, __gm__ float* outputGm, ... */)
{
    // TEMPLATE: grid-stride loop — 每个线程处理一个或多个输出元素
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < /* totalWork */; row += gridDim.x * blockDim.x) {
        // TEMPLATE: 计算当前线程负责的输出元素
        // float acc = 0.0f;
        // for (uint32_t col = 0; col < n; ++col) {
        //     acc += inputGm[idx] * xGm[col * incx];
        // }
        // outputGm[row * incy] = alpha * acc + beta * outputGm[row * incy];
    }
}

// TEMPLATE: 如有需要，可添加 __simt_callee__ 辅助函数
// __simt_callee__ inline float HelperFunc(/* ... */) { ... }

// TEMPLATE: __global__ 调度器
// - KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
// - tiling 通过 by value 方式从 host 传入（运行时 launch 参数自动拷贝）
// - 按运行时条件选择不同模板特化的 asc_vf_call
// - **强制**使用 `extern "C"` 禁止 C++ name mangling，确保 kernel 链接安全（reviewer HIGH 检视）
extern "C" __global__ __aicore__ void
{
    {
        op
    }
}
_kernel(
    /* GM_ADDR 各参数, */
    GM_ADDR workSpace, const {
        {
            Op
        }
    } TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TEMPLATE: 按运行时参数分发到不同模板特化
    // 示例 A — 单一路径（无编译期分支）：
    //   asc_vf_call<{{Op}}SimtCompute>(dim3{tiling.nthreads, 1, 1}, /* 参数 */);
    //
    // 示例 B — 按 uplo 分发：
    //   if (tiling.uplo == ACLBLAS_UPPER) {
    //       asc_vf_call<{{Op}}SimtCompute<true>>(dim3{tiling.nthreads, 1, 1}, ...);
    //   } else {
    //       asc_vf_call<{{Op}}SimtCompute<false>>(dim3{tiling.nthreads, 1, 1}, ...);
    //   }
    //
    // 示例 C — 按 trans 分发：
    //   if (tiling.trans == ACLBLAS_OP_N) {
    //       asc_vf_call<{{Op}}SimtComputeN>(dim3{tiling.nthreads, 1, 1}, ...);
    //   } else {
    //       asc_vf_call<{{Op}}SimtComputeT>(dim3{tiling.nthreads, 1, 1}, ...);
    //   }
}

// TEMPLATE: Kernel 启动器（host 侧调用）
// Tiling 通过 const 引用从 host 传入，kernel launch 时自动拷贝至 kernel 函数参数（by value）
void
{
    {
        op
    }
}
_kernel_do(
    /* GM_ADDR 各参数, */
    GM_ADDR workSpace, uint32_t numBlocks,
    const {
        {
            Op
        }
    } TilingData& tiling,
    void* stream)
{
    {
        {
            op
        }
    }
    _kernel<<<numBlocks, nullptr, stream>>>(/* ..., */ workSpace, tiling);
}
