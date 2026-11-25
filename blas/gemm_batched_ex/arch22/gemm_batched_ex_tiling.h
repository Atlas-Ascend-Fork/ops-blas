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

enum class GemmBatchedExRoute : uint32_t { Regular, MacroRing, RowStrips, ExactSmall, ReferenceOrder };

struct GemmBatchedExArch22Tiling {
    int32_t m, n, k, lda, ldb, ldc, batchCount;
    int32_t transA, transB, tileM, tileN;
    int32_t packed, originalK, originalLda, originalLdb;
    uint32_t vectorBlocks;
    GemmBatchedExRoute route;
    uint64_t packAElements, packBElements, productOffset;
    int32_t productLd;
    int32_t alphaDevice, betaDevice, temporary;
    float alpha, beta;
    uint64_t alphaAddress, betaAddress, tasks;
    uint64_t policy;
    uint64_t syncAddress;
    uint64_t referenceFirst;
    uint64_t referenceLast;
};

int32_t GemmBatchedExArch22Launch(
    uint32_t blocks, void* stream, const void* a, const void* b, const void* c, void* workspace,
    GemmBatchedExArch22Tiling tiling, bool scaleOnly);
