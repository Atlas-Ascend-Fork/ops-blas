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
 * \file ctrmv_fill.h
 * \brief ctrmv-specific fill helpers for ascend950 tests
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "fill.h"

// trmv is a multiply-add operator: the generic RANDOM_ALTER (±(i+1), up to ±2048
// for large n) and RANDOM_EXTREME (FLT_MAX / denormal values) generators produce
// results whose dynamic range makes the ecosystem per-element limit
// max(1e-2, 32*ULP) unreachable for any implementation - accumulation-order
// differences alone exceed it. The bounded variants below (RANDOM_ALTERB /
// RANDOM_EXTREMEB) keep the alternating-sign / extreme-magnitude coverage while
// staying judgeable.
// These helpers live next to the test (instead of extending the shared
// test/frame/fill.h) so that the P_ALTER / P_EXTREME semantics relied upon by
// every other operator test stay untouched.

inline std::vector<float> CtrmvFillArray(int64_t size, const std::string& fillStr, uint32_t seed)
{
    if (fillStr == "NULLPTR" || size <= 0) {
        return {};
    }
    if (fillStr == "RANDOM_ALTERB") {
        std::vector<float> data(static_cast<size_t>(size));
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (i % 2 == 0) ? static_cast<float>((i % 8) + 1) : -static_cast<float>((i % 8) + 1);
        }
        return data;
    }
    if (fillStr == "RANDOM_EXTREMEB") {
        static const float kVals[] = {100.0f, 0.0f, -100.0f, 1e-3f, -1e-3f, 50.0f, -50.0f};
        std::vector<float> data(static_cast<size_t>(size));
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = kVals[i % 7];
        }
        return data;
    }
    return makeBlasArray(size, fillStr, seed);
}

inline std::vector<float> CtrmvFillStrided(int count, int inc, const std::string& fillStr, uint32_t seed)
{
    if (fillStr != "RANDOM_ALTERB" && fillStr != "RANDOM_EXTREMEB") {
        return makeBlasStrided(count, inc, fillStr, seed);
    }
    if (count <= 0) {
        return {};
    }
    const int absInc = std::abs(inc);
    const size_t size = static_cast<size_t>((count - 1) * absInc + 1);
    std::vector<float> data(size, 0.0f);
    const std::vector<float> vals = CtrmvFillArray(count, fillStr, seed);
    for (int i = 0; i < count; i++) {
        const int idx = (inc > 0) ? (i * inc) : ((count - 1 - i) * absInc);
        data[static_cast<size_t>(idx)] = vals[static_cast<size_t>(i)];
    }
    return data;
}

// Overwrite every matrix cell that ctrmv must not read for the given uplo/diag
// with NaN. A kernel that touches a poisoned cell propagates NaN into x and
// fails verification, so the CSV RANDOM_NAN_OFFTRI cases prove the contract
// "only the selected triangle is accessed; diag=UNIT never reads the diagonal".
inline void CtrmvPoisonUnreadTri(std::vector<aclblasComplex>& a, int n, int lda, bool upper, bool unitDiag)
{
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            const bool read = upper ? (unitDiag ? row < col : row <= col) : (unitDiag ? row > col : row >= col);
            if (read) {
                continue;
            }
            a[static_cast<size_t>(col) * lda + row] = aclblasComplex{NAN, NAN};
        }
    }
}

// Complex64 lda x n column-major matrix; real part uses `seed`, imag part `seed + 1000`
// (same convention as fill.h::makeBlasComplexMatrix).
inline std::vector<aclblasComplex> CtrmvMakeMatrix(int m, int n, int lda, const std::string& fillStr, uint32_t seed)
{
    if (fillStr == "NULLPTR" || m <= 0 || n <= 0 || lda <= 0) {
        return {};
    }
    const size_t storage = static_cast<size_t>(lda) * static_cast<size_t>(n);
    const std::vector<float> realPart = CtrmvFillArray(static_cast<int64_t>(storage), fillStr, seed);
    const std::vector<float> imagPart = CtrmvFillArray(static_cast<int64_t>(storage), fillStr, seed + 1000U);
    std::vector<aclblasComplex> data(storage);
    for (size_t i = 0; i < storage; i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

// Complex64 strided vector: logical element i is stored at i*inc (inc > 0) or
// (count-1-i)*|inc| (inc < 0).
inline std::vector<aclblasComplex> CtrmvMakeStrided(int count, int inc, const std::string& fillStr, uint32_t seed)
{
    if (fillStr == "NULLPTR" || count <= 0) {
        return {};
    }
    const int absInc = std::abs(inc);
    const size_t size = static_cast<size_t>((count - 1) * absInc + 1);
    const std::vector<float> realPart = CtrmvFillStrided(count, inc, fillStr, seed);
    const std::vector<float> imagPart = CtrmvFillStrided(count, inc, fillStr, seed + 1000U);
    std::vector<aclblasComplex> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}
