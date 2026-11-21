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
 * \file cdgmm_kernel.h
 * \brief Declaration of the kernel launcher for aclblasCdgmm (arch22).
 *        Shared by host.cpp and kernel.cpp.
 *
 *        Row-major storage.  Only LEFT mode is implemented (C[i,j] = x[i] * A[i,j]).
 */

#pragma once

#include <cstdint>
#include "cdgmm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

/*!
 * \brief Kernel launcher: synchronously launches the Cdgmm kernel.
 *
 * \param A         GM address of matrix A (row-major complex, m x n, stride lda)
 * \param x         GM address of vector x (interleaved complex)
 * \param C         GM address of output matrix C (row-major complex, m x n, stride ldc)
 * \param aug       GM address of gather offset table (pre-computed by Host)
 * \param workSpace GM address of workspace (currently unused, reserved)
 * \param tilingGm  GM address of tiling data
 * \param numBlocks block count for the <<<>>> launch
 * \param stream    aclrtStream handle
 */
void cdgmm_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR C,
                     GM_ADDR aug, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void* stream);
