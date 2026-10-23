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
 * \file epilogue_alpha_beta_tiling_data.h
 * \brief Host-to-device tiling for alpha*D_raw + beta*C epilogue.
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
struct alignas(8) EpilogueAlphaBetaTilingData {
    uint32_t m{0};
    uint32_t n{0};
    uint32_t ldc{0};
    uint32_t ldd{0};
    uint32_t lddRaw{0};
    float alpha{1.0f};
    float beta{0.0f};
    uint32_t usedCoreNum{0};
    uint32_t numThreads{0};
    uint8_t dtypeDRaw{0}; // 0: FP32, 1: BF16
    uint8_t dtypeC{0};    // 0: FP32, 1: BF16
    uint8_t dtypeD{0};    // 0: FP32, 1: BF16
    uint8_t useC{0};
    uint8_t deviceScalarMode{0};
};
#pragma pack(pop)
