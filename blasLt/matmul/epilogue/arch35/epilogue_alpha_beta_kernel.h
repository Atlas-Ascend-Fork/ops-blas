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
 * \file epilogue_alpha_beta_kernel.h
 * \brief Host-side launch entry for the alpha*D_raw + beta*C epilogue device kernel (arch35).
 */

#pragma once

#include <cstdint>

#include "epilogue_alpha_beta_tiling_data.h"

void epilogue_alpha_beta_kernel_do(
    uint8_t* dRaw, uint8_t* c, uint8_t* d, const EpilogueAlphaBetaTilingData& tiling, void* stream,
    const float* alphaDevice, const float* betaDevice);
