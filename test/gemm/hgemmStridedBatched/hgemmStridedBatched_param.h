/*
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
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "csv_loader.h"
#include "dtype_cast.h"

namespace hgemm_test {
using Half = aclblasHalf;
// Fixture settings and task precision policy; keep seed/data generation reproducible.
constexpr int INVALID_OPERATION_VALUE = 999;
constexpr int MAX_FIXTURE_OFFSET = 31;
constexpr size_t MAX_FIXTURE_WORKSPACE_BYTES = 1024 * 1024;
constexpr uint32_t DEFAULT_RANDOM_SEED = 20261028;
constexpr uint32_t B_SEED_OFFSET = 100;
constexpr uint32_t C_SEED_OFFSET = 200;
constexpr size_t GUARD_ELEMENTS = 16;
constexpr Half GUARD_PATTERN = 0x3555;
constexpr int HALF_MANTISSA_BITS = 10;
constexpr int HALF_MIN_NORMAL_EXPONENT = -14;
constexpr int HALF_MIN_SUBNORMAL_EXPONENT = HALF_MIN_NORMAL_EXPONENT - HALF_MANTISSA_BITS;
constexpr float MIN_ABSOLUTE_ERROR_LIMIT = 0.1f;
constexpr int MAX_ULP_MULTIPLIER = 32;
constexpr double REQUIRED_MATCHED_RATIO = 0.99;
constexpr float INPUT_MIN = -5.0f;
constexpr float INPUT_MAX = 5.0f;
constexpr float NORMAL_SIGMA_MIN = 0.1f;
constexpr float NORMAL_SIGMA_MAX = 2.0f;
constexpr size_t ALTERNATING_MAGNITUDES = 5;
constexpr size_t EXTREME_VALUE_CYCLE = 3;

inline float F(Half x) { return blas_common::HalfToFloat(x); }
inline Half H(float x) { return blas_common::FloatToHalf(x); }

inline aclblasOperation_t ParseOperation(const std::string& v)
{
    return v == "ACLBLAS_OP_N" ? ACLBLAS_OP_N :
           v == "ACLBLAS_OP_T" ? ACLBLAS_OP_T :
           v == "ACLBLAS_OP_C" ? ACLBLAS_OP_C :
                                 static_cast<aclblasOperation_t>(INVALID_OPERATION_VALUE);
}

inline int ExpectedStatus(const std::string& value)
{
    if (value == "ACLBLAS_STATUS_SUCCESS")
        return ACLBLAS_STATUS_SUCCESS;
    if (value == "ACLBLAS_STATUS_ALLOC_FAILED")
        return ACLBLAS_STATUS_ALLOC_FAILED;
    if (value == "ACLBLAS_STATUS_INVALID_VALUE")
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (value == "ACLBLAS_STATUS_INVALID_ENUM")
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (value == "ACLBLAS_STATUS_HANDLE_IS_NULLPTR")
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    throw std::runtime_error("Unknown expected result: " + value);
}

struct HgemmStridedBatchedParam {
    std::string caseName, description, af, bf, cf;
    aclblasOperation_t ta, tb;
    int m, n, k, lda, ldb, ldc, batches, expected, cOffset, workspaceOffset;
    size_t workspaceBytes;
    int64_t sa, sb, sc;
    Half alpha, beta;
    bool alphaNull, betaNull;
    uint32_t seed;
    explicit HgemmStridedBatchedParam(const csv_map& r)
    {
        auto s = [&](const char* k, std::string fallback = "") { return ReadMap(r, std::string(k), fallback); };
        auto i = [&](const char* k, int64_t fallback) -> int64_t {
            auto v = s(k);
            return v.empty() ? fallback : std::stoll(v);
        };
        caseName = s("case_name");
        description = s("description");
        cOffset = i("c_offset", 0);
        workspaceOffset = i("workspace_offset", 0);
        workspaceBytes = i("workspace_bytes", 0);
        if (cOffset < 0 || cOffset > MAX_FIXTURE_OFFSET)
            throw std::runtime_error("Invalid C fixture offset");
        if (workspaceOffset < 0 || workspaceOffset > MAX_FIXTURE_OFFSET || workspaceBytes > MAX_FIXTURE_WORKSPACE_BYTES)
            throw std::runtime_error("Invalid workspace fixture");
        ta = ParseOperation(s("transA"));
        tb = ParseOperation(s("transB"));
        m = i("m", 0);
        n = i("n", 0);
        k = i("k", 0);
        batches = i("batchCount", 0);
        lda = i("lda", std::max(1, ta == ACLBLAS_OP_N ? m : k));
        ldb = i("ldb", std::max(1, tb == ACLBLAS_OP_N ? k : n));
        ldc = i("ldc", std::max(1, m));
        sa = i("strideA", static_cast<int64_t>(lda) * std::max(0, ta == ACLBLAS_OP_N ? k : m));
        sb = i("strideB", static_cast<int64_t>(ldb) * std::max(0, tb == ACLBLAS_OP_N ? n : k));
        sc = i("strideC", static_cast<int64_t>(ldc) * std::max(0, n));
        alphaNull = s("alpha") == "null";
        betaNull = s("beta") == "null";
        alpha = H(alphaNull ? 0 : std::stof(s("alpha", "1")));
        beta = H(betaNull ? 0 : std::stof(s("beta", "0")));
        af = s("a_fill");
        bf = s("b_fill");
        cf = s("c_fill");
        seed = i("random_seed", DEFAULT_RANDOM_SEED);
        expected = ExpectedStatus(s("expect_result", "ACLBLAS_STATUS_SUCCESS"));
    }
};

inline size_t Span(int rows, int cols, int ld, int64_t stride, int batches)
{
    if (rows <= 0 || cols <= 0 || batches <= 0 || ld < rows || stride < 0)
        return 0;
    return static_cast<size_t>(batches - 1) * stride + static_cast<size_t>(cols) * ld;
}

inline std::vector<Half> Input(size_t count, const std::string& fill, uint32_t seed)
{
    if (fill == "NULLPTR")
        return {};
    std::vector<Half> v(count);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(INPUT_MIN, INPUT_MAX), mean(INPUT_MIN, INPUT_MAX),
        sigma(NORMAL_SIGMA_MIN, NORMAL_SIGMA_MAX);
    const float mu = mean(rng), sd = sigma(rng);
    std::normal_distribution<float> normal(mu, sd);
    for (size_t j = 0; j < count; ++j) {
        float x;
        if (fill == "VALUE_NORM_0")
            x = 0;
        else if (fill == "VALUE_NORM_1")
            x = 1;
        else if (fill == "VALUE_NORM_INF")
            x = std::numeric_limits<float>::infinity();
        else if (fill == "VALUE_NORM_NAN")
            x = std::numeric_limits<float>::quiet_NaN();
        else if (fill == "RANDOM_ALTER")
            x = (j % 2 ? 1.0f : -1.0f) * (1 + (j % ALTERNATING_MAGNITUDES));
        else if (fill == "RANDOM_EXTREME")
            x = j % EXTREME_VALUE_CYCLE == 0 ? std::numeric_limits<float>::max() :
                j % EXTREME_VALUE_CYCLE == 1 ? -std::numeric_limits<float>::max() :
                                               0;
        else if (fill == "NORMAL")
            x = normal(rng);
        else if (fill == "RANDOM_NORM_5_5" || fill == "UNIFORM")
            x = uniform(rng);
        else
            throw std::runtime_error("Unknown fill: " + fill);
        v[j] = H(x);
    }
    return v;
}

} // namespace hgemm_test
