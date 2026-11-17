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
 * \file matmul_mxfp4_host.h
 * \brief Host-side MXFP4 matmul tiling API and dtype-dispatch launch entry (arch35).
 */

#pragma once

#include <acl/acl.h>
#include <cstdint>

struct QuantMatmulTilingData;

void matmul_mxfp4_get_tiling(
    uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, uint32_t numBlocks,
    QuantMatmulTilingData& tilingData);

// MXFP4 launch entry: routes to the FP32 or BF16 output kernel based on dtypeD.
void matmul_mxfp4_do(
    uint8_t* dA, uint8_t* dB, uint8_t* dScaleA, uint8_t* dScaleB, uint8_t* dC, const QuantMatmulTilingData& tiling,
    aclDataType dtypeD, bool transA, bool transB, void* stream);
