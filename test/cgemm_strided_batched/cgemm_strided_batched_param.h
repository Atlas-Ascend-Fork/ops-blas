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
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include "csv_loader.h"
#include "fill.h"
static std::string ReadOr(const csv_map& map, const std::string& key, const std::string& fallback)
{
    std::string value = ReadMap(map, key, "");
    return value.empty() ? fallback : value;
}

static aclblasOperation_t ReadTrans(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return parseOpTrans(value);
}

struct CgemmSbParam : BlasTestParamBase {
    aclblasOperation_t transA, transB;
    int m, n, k, lda, ldb, ldc, batchCount;
    int64_t strideA, strideB, strideC;
    size_t aOffset, bOffset, cOffset;
    aclblasComplex alpha, beta;
    BlasFillMode aFill, bFill, cFill;
    bool alphaNull, betaNull, aNull, bNull, cNull;

    explicit CgemmSbParam(const csv_map& map) : BlasTestParamBase(map)
    {
        transA = ReadTrans(ReadMap(map, "transA", "N"));
        transB = ReadTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        batchCount = parseInt(ReadMap(map, "batchCount", "1"));
        alpha = {parseFloat(ReadMap(map, "alpha_real", "1")), parseFloat(ReadMap(map, "alpha_imag", "0"))};
        beta = {parseFloat(ReadMap(map, "beta_real", "0")), parseFloat(ReadMap(map, "beta_imag", "0"))};
        const std::string distribution = ReadMap(map, "scalar_distribution", "");
        if (!distribution.empty()) {
            std::mt19937 rng(randomSeed + 173U);
            std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
            std::normal_distribution<float> normal(0.0f, 1.0f / 3.0f);
            auto draw = [&]() { return distribution == "normal" ? normal(rng) : uniform(rng); };
            alpha = {draw(), draw()};
            beta = {draw(), draw()};
        }
        lda = parseInt(ReadOr(map, "lda", std::to_string(std::max(1, transA == ACLBLAS_OP_N ? m : k))));
        ldb = parseInt(ReadOr(map, "ldb", std::to_string(std::max(1, transB == ACLBLAS_OP_N ? k : n))));
        ldc = parseInt(ReadOr(map, "ldc", std::to_string(std::max(1, m))));
        const int64_t compactA = static_cast<int64_t>(lda) * (transA == ACLBLAS_OP_N ? k : m);
        const int64_t compactB = static_cast<int64_t>(ldb) * (transB == ACLBLAS_OP_N ? n : k);
        const int64_t compactC = static_cast<int64_t>(ldc) * n;
        strideA = parseInt64(ReadOr(map, "strideA", std::to_string(compactA)));
        strideB = parseInt64(ReadOr(map, "strideB", std::to_string(compactB)));
        strideC = parseInt64(ReadOr(map, "strideC", std::to_string(compactC)));
        aOffset = static_cast<size_t>(parseInt(ReadOr(map, "a_offset", "0")));
        bOffset = static_cast<size_t>(parseInt(ReadOr(map, "b_offset", "0")));
        cOffset = static_cast<size_t>(parseInt(ReadOr(map, "c_offset", "0")));
        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_1_1"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_NORM_1_1"));
        cFill = parseFill(ReadMap(map, "c_fill", "RANDOM_NORM_1_1"));
        alphaNull = ReadMap(map, "alpha_null", "") == "NULLPTR";
        betaNull = ReadMap(map, "beta_null", "") == "NULLPTR";
        aNull = ReadMap(map, "a_null", "") == "NULLPTR" || aFill.method == BlasFillMode::M_NULLPTR;
        bNull = ReadMap(map, "b_null", "") == "NULLPTR" || bFill.method == BlasFillMode::M_NULLPTR;
        cNull = ReadMap(map, "c_null", "") == "NULLPTR" || cFill.method == BlasFillMode::M_NULLPTR;
    }
};

static size_t Span(int64_t stride, int64_t matrix, int batches)
{
    if (matrix <= 0 || batches <= 0 || stride < 0) {
        return 0;
    }
    return static_cast<size_t>(stride * (batches - 1) + matrix);
}

static std::vector<float> MakeData(size_t complexCount, const BlasFillMode& fill, uint32_t seed)
{
    if (complexCount == 0 || fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    auto values = makeBlasArray(complexCount * 2, fill, seed);
    if (fill.method == BlasFillMode::M_RANDOM && fill.pattern == BlasFillMode::P_NORM) {
        std::mt19937 rng(seed + 977U);
        const float scale = std::max(std::abs(fill.val1), std::abs(fill.val2));
        std::normal_distribution<float> normal(0.0f, scale / 3.0f);
        // Equal uniform/normal complex samples; odd tails alternate by seed.
        for (size_t i = 0; i < complexCount; ++i) {
            if ((i + seed) % 2U) {
                values[2U * i] = normal(rng);
                values[2U * i + 1U] = normal(rng);
            }
        }
    }
    return values;
}
