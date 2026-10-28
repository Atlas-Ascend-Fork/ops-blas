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
 * \file aclblas_csyr2_host.h
 * \brief Host 包装：GetAivCoreNum / GetUbBytes / LaunchCsyr2Kernel。
 *        CSYR2_ASCENDC 下 include tiling/platform/platform_ascendc.h，
 *        调用 PlatformAscendC(...).GetCoreNumAiv() 与 GetCoreMemSize(UB)。
 */
#ifndef ACLBLAS_CSYR2_HOST_H
#define ACLBLAS_CSYR2_HOST_H

#include <cstdint>
#include "cann_ops_blas.h"
#include "csyr2_tiling.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! PlatformAscendC(...).GetCoreNumAiv()。源码无写死核数常量。 */
uint32_t GetAivCoreNum(void);

/*! GetCoreMemSize(CoreMemType::UB)。 */
uint64_t GetUbBytes(void);

/*! Host 按实际代入的 overhead；验算可取 8KB 量级。 */
uint32_t GetCsyr2OverheadBytes(void);

/*!
 * 从 handle 取 aclrtStream（aclblasSetStream 已绑定）。
 * 返回 void*，避免本头依赖 acl.h。
 */
void* GetCsyr2Stream(aclblasHandle_t handle);

/*!
 * 按 tilingKey 启动 kernel。SetBlockDim(usedCoreNum) 按 AIV。
 * Host 不调用 stream 同步。
 * tiling 的计算字段作为标量启动参数按值传递，无共享 GM 或 host staging。
 * malloc 失败返回 ALLOC_FAILED，memcpy 失败返回 INTERNAL_ERROR。
 */
aclblasStatus_t LaunchCsyr2Kernel(
    void* stream,
    uint32_t tilingKey,
    uint32_t usedCoreNum,
    const aclblasComplex* x,
    const aclblasComplex* y,
    aclblasComplex* A,
    const Csyr2TilingData* tiling);

/*! 仅 HOST_SIM：注入模拟 AIV 数。不是设备核数，也不是设计验算数 40。 */
void Csyr2HostSimSetAivNum(uint32_t aivNum);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
aclblasStatus_t Csyr2FillTiling(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, int incx, int incy, int lda,
    Csyr2TilingData* tiling);
#endif

#endif /* ACLBLAS_CSYR2_HOST_H */
