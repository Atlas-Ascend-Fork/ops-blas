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

struct CsyrkxTilingData {
    uint32_t n, k, lda, ldb, ldc, productLd, parts, riskBlocks;
    uint64_t partStride;
    float ar, ai, br, bi;
    uint8_t upper, trans, scaleOnly, betaZero;
};

// Each producer owns a full 32-byte slot, avoiding sub-block write sharing.
constexpr uint32_t CSYRKX_RISK_SLOT_WORDS = 8;
// Every split block writes its entire slot (no initialization required).
// A/B components with abs bits > 0x49800000 (2^20, including Inf/NaN) are risky.
void csyrkx_split_do(
    uint32_t blocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* ar, uint8_t* ai, uint8_t* br,
    uint8_t* bi, uint8_t* riskFlags, const CsyrkxTilingData& t);
void csyrkx_epilogue_do(
    uint32_t blocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* p0, uint8_t* p1, uint8_t* p2,
    uint8_t* p3, const uint8_t* riskFlags, uint8_t* c, const CsyrkxTilingData& t);
void csyrkx_direct_do(
    uint32_t blocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* c, const CsyrkxTilingData& t);
void csyrkx_ordered_do(
    uint32_t blocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* c, const CsyrkxTilingData& t);
