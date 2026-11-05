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
 * \file ssbmv_kernel.h
 * \brief Kernel entry of aclblasSsbmv for arch22.
 */

#pragma once

#include <cstdint>

#include "ssbmv_tiling_data.h"

/*!
 * \brief Launches the Ssbmv AIV kernel.
 *
 * \param a         device pointer of the banded matrix, column major, lda x n
 * \param x         device pointer of the input vector
 * \param y         device pointer of the in/out vector
 * \param workSpace device workspace; unused by this kernel (kept for signature parity)
 * \param numBlocks number of AIV cores to launch
 * \param tiling    tiling data, passed by value into the launch parameter buffer
 * \param stream    runtime stream the handle carries
 */
void ssbmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks, const SsbmvTilingData& tiling,
    void* stream);
