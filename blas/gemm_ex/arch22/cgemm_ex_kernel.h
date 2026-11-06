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
struct CgemmExLayout {
    int32_t m;
    int32_t n;
    int32_t ldc;
    float ar;
    float ai;
    float br;
    float bi;
    uint32_t mode;
    int32_t packedLd;
};
void cgemm_ex_split_do(
    uint32_t blocks, void* stream, uint8_t* input, uint8_t* real, uint8_t* imag, uint64_t count, float sign);
void cgemm_ex_combine_do(
    uint32_t blocks, void* stream, uint8_t* real, uint8_t* imag, uint8_t* output, CgemmExLayout layout);

struct CgemmExDirectLayout {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t ta;
    int32_t tb;
    float ar;
    float ai;
    float br;
    float bi;
};
void cgemm_ex_direct_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, CgemmExDirectLayout t);

struct CgemmExpandedLayout {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t mp;
    int32_t np;
    int32_t kp;
    int32_t lda;
    int32_t ldb;
    int32_t ta;
    int32_t tb;
    int32_t canonicalB;
};
void cgemm_ex_expand_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* ea, uint8_t* eb, CgemmExpandedLayout t);

struct GemmExTiling;
void cgemm_ex_split104_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, GemmExTiling);
void cgemm_ex_split107_finish_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, GemmExTiling, CgemmExLayout);
void cgemm_ex_native110_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, GemmExTiling);

void cgemm_ex_rawb114_do(uint32_t, void*, uint8_t*, uint8_t*, CgemmExpandedLayout);
void cgemm_ex_raw114_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, GemmExTiling);

void cgemm_ex_cross115_do(
    uint32_t, uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, CgemmExpandedLayout, CgemmExLayout);

void cgemm_ex_cross_compute_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, GemmExTiling t);
