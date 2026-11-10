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

#include <cstdint>
#include "common/helper/gemm_strided_tiling_arch35.h"

void gemm_strided_batched_ex_kernel_do(
    uint32_t numBlocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c,
    const GemmStridedBatchedExTilingData& tilingData, GemmStridedBatchedExDtypeCase dtypeCase);

void gemm_strided_batched_ex_epilogue_do(
    uint32_t numBlocks, void* stream, uint8_t* partial, uint8_t* c, const GemmStridedBatchedExTilingData& tilingData,
    GemmStridedBatchedExDtypeCase dtypeCase, bool earlyExit);

void gemm_strided_batched_ex_complex_do(
    uint32_t numBlocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c,
    const GemmStridedBatchedExTilingData& tilingData);
