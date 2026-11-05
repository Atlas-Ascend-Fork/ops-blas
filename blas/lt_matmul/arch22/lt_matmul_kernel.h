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

#include "lt_matmul_tiling.h"

void LtMatmul22Cube(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* product, LtMatmul22Tiling t);
void LtMatmul22CubeTN64x192(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* product, LtMatmul22Tiling t);
void LtMatmul22Mix(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, LtMatmul22Tiling t);
void LtMatmul22Epilogue(uint32_t blocks, void* stream, uint8_t* product, uint8_t* c, uint8_t* d, LtMatmul22Tiling t);
void LtMatmul22Tail(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, LtMatmul22Tiling t);
void LtMatmul22Scale(uint32_t blocks, void* stream, uint8_t* c, uint8_t* d, LtMatmul22Tiling t);
void LtMatmul22Ordered(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, LtMatmul22Tiling t);
void LtMatmul22PackNZ(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* aNz, uint8_t* bNz, LtMatmul22Tiling t);
void LtMatmul22CubeNZ(uint32_t blocks, void* stream, uint8_t* aNz, uint8_t* bNz, uint8_t* product, LtMatmul22Tiling t);

void LtMatmul22SplitCube(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, LtMatmul22Tiling, uint32_t);
void LtMatmul22SplitReduce(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, LtMatmul22Tiling, uint32_t);

void LtMatmul22PackOneNZ(uint32_t, void*, uint8_t*, uint8_t*, LtMatmul22Tiling, uint32_t);
void LtMatmul22MixedNZCube(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, LtMatmul22Tiling, uint32_t);

void LtMatmul22RectShort(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, LtMatmul22Tiling);

void LtMatmul22FusedShort(
    uint32_t, uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, LtMatmul22Tiling, uint64_t);

void LtMatmul22FusedScale(uint32_t, void*, uint8_t*, uint8_t*, LtMatmul22Tiling);

void LtMatmul22PackNZ64(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* aNz, uint8_t* bNz, LtMatmul22Tiling t);

void LtMatmul22PackNZ80(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* aNz, uint8_t* bNz, LtMatmul22Tiling t);

void LtMatmul22Rows160(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* product, LtMatmul22Tiling t);

void LtMatmul22ShortKColumns(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* output, LtMatmul22Tiling tiling);
