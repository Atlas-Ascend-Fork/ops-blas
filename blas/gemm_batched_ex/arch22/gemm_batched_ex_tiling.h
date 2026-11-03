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
#include <cstddef>

struct GemmBatchedEx22Tiling {
    int32_t m, n, k, lda, ldb, ldc;
    int32_t transA, transB, outputLd;
    uint32_t firstBatch, batches;
    float alpha, beta;
    bool temporaryOutput;
    uint32_t splitK = 0, parts = 0, taskCount = 0;
    uint64_t firstTask = 0;
    uint64_t policy = 0;
    uint32_t policyCores = 0;
    uint32_t taskRotation = 0; // Reuses the original four bytes of tail padding.
};

static_assert(sizeof(GemmBatchedEx22Tiling) == 96);
static_assert(offsetof(GemmBatchedEx22Tiling, firstTask) == 72);
static_assert(offsetof(GemmBatchedEx22Tiling, policy) == 80);
static_assert(offsetof(GemmBatchedEx22Tiling, policyCores) == 88);
static_assert(offsetof(GemmBatchedEx22Tiling, taskRotation) == 92);

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void GemmBatchedEx22Cube(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR temporary, GemmBatchedEx22Tiling tiling);
void GemmBatchedEx22Vector(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR temporary, GemmBatchedEx22Tiling tiling,
    bool combine);

void GemmBatchedEx22Policy(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GemmBatchedEx22Tiling tiling, bool finalize);
void GemmBatchedEx22Reference(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmBatchedEx22Tiling tiling);

void GemmBatchedEx22PolicyReference(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmBatchedEx22Tiling tiling);

void GemmBatchedEx22PolicyFused(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmBatchedEx22Tiling tiling);

void GemmBatchedEx22Overlap(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmBatchedEx22Tiling tiling);
