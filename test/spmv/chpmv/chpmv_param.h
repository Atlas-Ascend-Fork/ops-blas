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
#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "csv_loader.h"

namespace chpmv_test {
using Complex = aclblasComplex;
static_assert(sizeof(Complex) == 2 * sizeof(float), "complex64 layout");

inline size_t Offset(int i, int n, int inc)
{
    return inc > 0 ? static_cast<size_t>(i) * inc : static_cast<size_t>(n - 1 - i) * (-int64_t(inc));
}
inline size_t Count(int n, int inc)
{
    return n > 0 ? 1 + static_cast<size_t>(n - 1) * (inc < 0 ? -int64_t(inc) : int64_t(inc)) : 0;
}
inline size_t PackedCount(int n) { return n > 0 ? static_cast<size_t>(n) * (n + int64_t(1)) / 2 : 0; }
inline size_t Diagonal(int i, int n, aclblasFillMode_t uplo)
{
    return uplo == ACLBLAS_UPPER ? static_cast<size_t>(i) * (i + int64_t(1)) / 2 + i :
                                   static_cast<size_t>(i) * (2 * int64_t(n) - i + 1) / 2;
}

struct Param : public BlasTestParamBase {
    int n, incx, incy;
    aclblasFillMode_t uplo;
    Complex alpha{}, beta{};
    bool nullAlpha, nullBeta;
    std::string apFill, xFill, yFill;
    bool poisonDiagonal = false;
    int numericMode = 0;
    size_t prefix = 16;
    explicit Param(const csv_map& m) : BlasTestParamBase(m)
    {
        n = std::stoi(m.at("n"));
        incx = std::stoi(m.at("incx"));
        incy = std::stoi(m.at("incy"));
        const auto& u = m.at("uplo");
        uplo = u == "ACLBLAS_UPPER" ? ACLBLAS_UPPER :
               u == "ACLBLAS_LOWER" ? ACLBLAS_LOWER :
                                      static_cast<aclblasFillMode_t>(std::stoi(u));
        nullAlpha = m.at("alpha_real") == "null" || m.at("alpha_imag") == "null";
        nullBeta = m.at("beta_real") == "null" || m.at("beta_imag") == "null";
        if (!nullAlpha)
            alpha = {std::stof(m.at("alpha_real")), std::stof(m.at("alpha_imag"))};
        if (!nullBeta)
            beta = {std::stof(m.at("beta_real")), std::stof(m.at("beta_imag"))};
        apFill = m.at("ap_fill");
        xFill = m.at("x_fill");
        yFill = m.at("y_fill");
        const auto& status = m.at("expect_result");
        if (status == "ACLBLAS_STATUS_SUCCESS")
            expectResult = ACLBLAS_STATUS_SUCCESS;
        else if (status == "ACLBLAS_STATUS_INVALID_VALUE")
            expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        else if (status == "ACLBLAS_STATUS_INVALID_ENUM")
            expectResult = ACLBLAS_STATUS_INVALID_ENUM;
        else
            throw std::runtime_error("unsupported expected status: " + status);
    }
};

inline void AppendNormalCases(std::vector<Param>& cases)
{
    const size_t originalCount = cases.size();
    for (size_t i = 0; i < originalCount; ++i) {
        Param p = cases[i];
        if (p.n <= 0 || p.expectResult != ACLBLAS_STATUS_SUCCESS)
            continue;
        bool changed = false;
        for (auto* fill : {&p.apFill, &p.xFill, &p.yFill}) {
            if (*fill == "RANDOM_NORM_5_5") {
                *fill = "GAUSS_NORM_5_2";
                changed = true;
            }
        }
        if (changed) {
            p.caseName = "NORMAL_" + p.caseName;
            cases.push_back(p);
        }
    }
}

inline Param DistributionCase(const Param& original, int distribution, int lower, size_t i, int n)
{
    Param p = original;
    p.caseName = "SUP_DIST_" + std::to_string(distribution) + "_" + std::to_string(lower) + "_" + std::to_string(i);
    p.n = n;
    if (i >= 17)
        p.prefix = 17;
    p.uplo = lower ? ACLBLAS_LOWER : ACLBLAS_UPPER;
    p.incx = (i % 2 ? -1 : 1) * (1 + int(i % 3));
    p.incy = (i % 2 ? 1 : -1) * (1 + int((i + 1) % 3));
    p.randomSeed = 910000 + distribution * 1000 + lower * 100 + i;
    p.apFill = p.xFill = p.yFill = distribution ? "GAUSS_NORM_5_2" : "RANDOM_NORM_5_5";
    std::mt19937 rng(p.randomSeed);
    std::uniform_real_distribution<float> uniform(-5, 5), sigma(0.1f, 2.0f);
    if (distribution) {
        const float mu = uniform(rng), sd = sigma(rng);
        std::normal_distribution<float> normal(mu, sd);
        p.alpha = {normal(rng), normal(rng)};
        p.beta = {normal(rng), normal(rng)};
    } else {
        p.alpha = {uniform(rng), uniform(rng)};
        p.beta = {uniform(rng), uniform(rng)};
    }
    return p;
}

inline void AppendDistributionCases(std::vector<Param>& cases)
{
    const std::array<int, 21> sizes{1,   2,   3,    17,   31,   32,   33, 65, 127, 128, 129,
                                    257, 513, 1025, 8191, 8192, 8193, 15, 16, 511, 512};
    for (int distribution = 0; distribution < 2; ++distribution) {
        for (int lower = 0; lower < 2; ++lower) {
            for (size_t i = 0; i < sizes.size(); ++i) {
                cases.push_back(DistributionCase(cases.front(), distribution, lower, i, sizes[i]));
            }
        }
    }
}

inline void AppendBoundaryCases(std::vector<Param>& cases)
{
    for (int lower = 0; lower < 2; ++lower) {
        for (int n : {1, 2, 3, 31, 32, 33, 127, 129, 257}) {
            Param p = cases.front();
            p.caseName = "SUP_DIAG_" + std::to_string(lower) + "_" + std::to_string(n);
            p.n = n;
            p.uplo = lower ? ACLBLAS_LOWER : ACLBLAS_UPPER;
            p.poisonDiagonal = true;
            p.prefix = 17;
            p.incx = -2;
            p.incy = 3;
            cases.push_back(p);
        }
    }
    for (int inc : {INT_MIN, INT_MAX}) {
        Param p = cases.front();
        p.caseName = inc == INT_MIN ? "SUP_INT_MIN_N1" : "SUP_INT_MAX_N1";
        p.n = 1;
        p.incx = p.incy = inc;
        cases.push_back(p);
    }
}

inline void AppendNumericalCases(std::vector<Param>& cases)
{
    for (int mode : {1, 2}) {
        Param p = cases.front();
        p.caseName = "SUP_NUMERICAL_" + std::to_string(mode);
        p.n = 1;
        p.numericMode = mode;
        p.alpha = mode == 1 ? Complex{std::numeric_limits<float>::max(), 0} : Complex{0.5f, 0};
        p.apFill = p.xFill = "VALUE_NORM_0";
        p.poisonDiagonal = true;
        cases.push_back(p);
        p.n = 65;
        p.caseName += "_N65";
        cases.push_back(p);
    }
    for (int mode : {3, 4, 5, 6}) {
        Param p = cases.front();
        p.caseName = "SUP_NUMERICAL_" + std::to_string(mode);
        p.n = 65;
        p.numericMode = mode;
        p.uplo = mode % 2 ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        p.alpha = {std::numeric_limits<float>::max(), 0};
        p.apFill = p.xFill = "VALUE_NORM_0";
        p.poisonDiagonal = true;
        if (mode >= 5)
            p.yFill = "VALUE_NORM_0";
        cases.push_back(p);
        if (mode >= 5) {
            for (int n : {512, 513}) {
                p.n = n;
                p.caseName = "SUP_NUMERICAL_" + std::to_string(mode) + "_N" + std::to_string(n);
                cases.push_back(p);
            }
        }
    }
}

inline std::vector<Param> Cases(const std::string& path)
{
    auto cases = GetCasesFromCsv<Param>(path);
    if (cases.size() != 1200)
        throw std::runtime_error("original CSV must contain 1200 cases");
    AppendNormalCases(cases);
    AppendDistributionCases(cases);
    AppendBoundaryCases(cases);
    AppendNumericalCases(cases);
    std::set<std::string> names;
    for (const auto& p : cases) {
        if (!names.insert(p.caseName).second)
            throw std::runtime_error("duplicate case: " + p.caseName);
    }
    return cases;
}

} // namespace chpmv_test
