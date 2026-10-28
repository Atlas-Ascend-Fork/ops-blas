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
 * \file csyr2_kernel.h
 * \brief Vector kernel：Init / 逐列 CopyIn-Compute-CopyOut。
 *        CSYR2_ASCENDC 下 LaunchCsyr2Kernel 按 tilingKey 启动 csyr2_kernel_{0,1,2,3}。
 */
#ifndef CSYR2_KERNEL_H
#define CSYR2_KERNEL_H

/* blas/ 下源码由 CMake 以 LANGUAGE ASC 编译，默认走 Ascend C。
 * 无 CANN 的 g++ 自测传 -DCSYR2_HOST_SIM 切到 CPU 模拟实现。 */
#if !defined(CSYR2_HOST_SIM) && !defined(CSYR2_ASCENDC)
#define CSYR2_ASCENDC
#endif

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "csyr2_tiling.h"

/* Ascend C 路径下 GM_ADDR 来自 kernel_operator.h，重复定义会告警。 */
#if !defined(CSYR2_ASCENDC)
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*! 单核入口：blockIdx >= usedCoreNum 则直接返回。HOST_SIM 与 Ascend C 共用此语义。 */
void Csyr2KernelRun(
    uint32_t blockIdx,
    uint32_t tilingKey,
    const Csyr2TilingData* tiling,
    const aclblasComplex* x,
    const aclblasComplex* y,
    aclblasComplex* A);

/*!
 * 按 tilingKey 启动。Host 不调用 stream 同步。
 * CSYR2_ASCENDC：tiling 的计算字段作为标量启动参数按值传递，无共享 GM 或 host staging。
 */
aclblasStatus_t LaunchCsyr2Kernel(
    void* stream,
    uint32_t tilingKey,
    uint32_t usedCoreNum,
    const aclblasComplex* x,
    const aclblasComplex* y,
    aclblasComplex* A,
    const Csyr2TilingData* tiling);

#ifdef __cplusplus
}
#endif

#endif /* CSYR2_KERNEL_H */
