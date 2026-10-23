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
 * \file epilogue_alpha_beta_host.h
 * \brief Host-side tiling and launch API for the alpha*D_raw + beta*C epilogue (arch35).
 */

#pragma once

#include <cstdint>

#include <acl/acl.h>

#include "epilogue_alpha_beta_tiling_data.h"

// Epilogue: D = alpha * D_raw + beta * C
void epilogue_alpha_beta_get_tiling(uint32_t m, uint32_t n, uint32_t numBlocks, EpilogueAlphaBetaTilingData& tilingData);
void epilogue_alpha_beta_do(
    uint8_t* dRaw, uint8_t* c, uint8_t* d, uint32_t m, uint32_t n, uint32_t ldc, uint32_t ldd, uint32_t lddRaw,
    float alpha, float beta, aclDataType dtypeC, aclDataType dtypeDRaw, aclDataType dtypeD, void* stream,
    const float* alphaDevice = nullptr, const float* betaDevice = nullptr);
