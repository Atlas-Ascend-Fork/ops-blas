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
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <random>
#include <tuple>
#include <vector>
#include "securec.h"
#include "ctrmm_param.h"

namespace {
inline std::vector<std::tuple<aclblasSideMode_t, aclblasFillMode_t, aclblasOperation_t, aclblasDiagType_t>>
AllBlasModes()
{
    std::vector<std::tuple<aclblasSideMode_t, aclblasFillMode_t, aclblasOperation_t, aclblasDiagType_t>> modes;
    for (auto side : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                for (auto diag : {ACLBLAS_NON_UNIT, ACLBLAS_UNIT}) {
                    modes.emplace_back(side, uplo, trans, diag);
                }
            }
        }
    }
    return modes;
}

inline std::vector<std::tuple<aclblasSideMode_t, aclblasFillMode_t, aclblasOperation_t>> SideUploTransModes()
{
    std::vector<std::tuple<aclblasSideMode_t, aclblasFillMode_t, aclblasOperation_t>> modes;
    for (auto side : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                modes.emplace_back(side, uplo, trans);
            }
        }
    }
    return modes;
}

inline std::vector<std::tuple<aclblasFillMode_t, aclblasOperation_t, aclblasDiagType_t>> UploTransDiagModes()
{
    std::vector<std::tuple<aclblasFillMode_t, aclblasOperation_t, aclblasDiagType_t>> modes;
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            for (auto diag : {ACLBLAS_NON_UNIT, ACLBLAS_UNIT}) {
                modes.emplace_back(uplo, trans, diag);
            }
        }
    }
    return modes;
}

inline std::vector<std::tuple<aclblasSideMode_t, aclblasOperation_t>> SideTransModes()
{
    std::vector<std::tuple<aclblasSideMode_t, aclblasOperation_t>> modes;
    for (auto side : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            modes.emplace_back(side, trans);
        }
    }
    return modes;
}

inline CtrmmParam MakeShortCancellationParam(const std::string& name, int q, int r, const char* side, const char* trans)
{
    const bool right = std::string(side) == "RIGHT";
    return CtrmmParam(
        csv_map{
            {"case_name", name},
            {"m", std::to_string(right ? r : q)},
            {"n", std::to_string(right ? q : r)},
            {"side", side},
            {"uplo", "UPPER"},
            {"trans", trans},
            {"diag", "NON_UNIT"},
            {"alpha_real", "8"},
            {"alpha_imag", "0"}});
}

inline void FillShortCancellationInputs(std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    for (auto& z : a) {
        z = {31.3f, 31.7f};
    }
    for (auto& z : b) {
        z = {31.7f, 31.3f};
    }
}

inline float ExponentInputValue(uint32_t i)
{
    if (i % 17U == 0) {
        return 0.0f;
    }
    if (i % 19U == 0) {
        return 32.0f;
    }
    const uint32_t exponent = 97U + i % 35U; // Unbiased -30..4.
    const uint32_t bits = ((i & 1U) << 31U) | (exponent << 23U) | ((i * 2654435761U) & 0x7fffffU);
    float result;
    if (memcpy_s(&result, sizeof(result), &bits, sizeof(bits)) != EOK) {
        throw std::runtime_error("Cannot construct the FP32 test input");
    }
    return result;
}

inline void MakeGdnKeys(
    int q, float betaScale, std::vector<double>& keys, std::vector<double>& beta, std::vector<double>& decay,
    std::mt19937& rng, std::normal_distribution<float>& normal, std::uniform_real_distribution<float>& gate)
{
    constexpr int d = 64;
    for (int i = 0; i < q; ++i) {
        double norm = 0;
        for (int k = 0; k < d; ++k) {
            auto& x = keys[i * d + k];
            x = normal(rng);
            norm += x * x;
        }
        for (int k = 0; k < d; ++k) {
            keys[i * d + k] /= std::sqrt(norm);
        }
        beta[i] = betaScale * gate(rng);
        decay[i] = 0.8 + 0.2 * gate(rng);
    }
}

inline void MakeGdnInverse(
    int q, const std::vector<double>& keys, const std::vector<double>& beta, const std::vector<double>& decay,
    std::vector<double>& lower, std::vector<double>& inverse)
{
    constexpr int d = 64;
    for (int i = 0; i < q; ++i) {
        inverse[i * q + i] = 1;
        for (int j = 0; j < i; ++j) {
            double dot = 0;
            double factor = 1;
            for (int k = 0; k < d; ++k) {
                dot += keys[i * d + k] * keys[j * d + k];
            }
            for (int k = j + 1; k <= i; ++k) {
                factor *= decay[k];
            }
            lower[i * q + j] = beta[i] * factor * dot;
        }
        for (int j = 0; j < i; ++j) {
            double x = 0;
            for (int k = j; k < i; ++k) {
                x -= lower[i * q + k] * inverse[k * q + j];
            }
            inverse[i * q + j] = x;
        }
    }
}

inline void FillGdnInputs(
    const CtrmmParam& p, int q, int r, float betaScale, std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    constexpr int d = 64;
    std::mt19937 rng(7291 + q + r);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> gate(0.0f, 1.0f);
    std::vector<double> keys(q * d);
    std::vector<double> beta(q);
    std::vector<double> decay(q);
    std::vector<double> lower(q * q);
    std::vector<double> inverse(q * q);
    MakeGdnKeys(q, betaScale, keys, beta, decay, rng, normal, gate);
    MakeGdnInverse(q, keys, beta, decay, lower, inverse);
    for (int j = 0; j < q; ++j) {
        for (int i = 0; i < q; ++i) {
            a[j * p.lda + i] = {static_cast<float>(inverse[i * q + j]), 0};
        }
    }
    for (int j = 0; j < r; ++j) {
        for (int i = 0; i < q; ++i) {
            b[j * p.ldb + i] = {static_cast<float>(beta[i]) * std::clamp(normal(rng), -8.0f, 8.0f), 0};
        }
    }
}

inline void FillSsdInputs(
    const CtrmmParam& p, int q, int r, std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    std::mt19937 rng(15851 + q + r);
    std::normal_distribution<float> normal(0, 1);
    std::uniform_real_distribution<double> gate(0.8, 1.0);
    std::vector<double> bv(q * 32);
    std::vector<double> cv(q * 32);
    std::vector<double> decay(q);
    for (auto& x : bv) {
        x = normal(rng) / std::sqrt(32.0);
    }
    for (auto& x : cv) {
        x = normal(rng) / std::sqrt(32.0);
    }
    for (auto& x : decay) {
        x = gate(rng);
    }
    for (int j = 0; j < q; ++j) {
        for (int i = j; i < q; ++i) {
            double dot = 0;
            double factor = 1;
            for (int k = 0; k < 32; ++k) {
                dot += cv[i * 32 + k] * bv[j * 32 + k];
            }
            for (int k = j + 1; k <= i; ++k) {
                factor *= decay[k];
            }
            a[j * p.lda + i] = {static_cast<float>(dot * factor), 0};
        }
    }
    for (auto& x : b) {
        x = {std::clamp(normal(rng), -8.0f, 8.0f), 0};
    }
}

inline void FillSmallNormInputs(std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    for (auto& z : a) {
        z.real *= 0x1p-16f;
        z.imag *= 0x1p-16f;
    }
    for (auto& z : b) {
        z.real *= 0x1p-16f;
        z.imag *= 0x1p-16f;
    }
}

inline void FillRoutingBoundaryInputs(std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b, float magnitude)
{
    for (auto& z : a) {
        z.real *= magnitude / 5.0f;
        z.imag *= magnitude / 5.0f;
    }
    for (auto& z : b) {
        z.real *= magnitude / 5.0f;
        z.imag *= magnitude / 5.0f;
    }
    a[0] = {magnitude, -magnitude};
    b[0] = {-magnitude, magnitude};
}
inline void ZeroStridedInputs(std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    for (size_t i = 0; i < a.size(); i += 7) {
        a[i] = {0, 0};
    }
    for (size_t i = 0; i < b.size(); i += 5) {
        b[i] = {0, 0};
    }
}
inline void FillBatchedFmaBoundary(std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b, int edge)
{
    float low = edge < 2 ? 0x1p-16f : 0x1p-24f;
    float high = edge < 2 ? 256.0f : 65536.0f;
    if (edge < 0 || edge == 2) {
        low = std::nextafter(low, 0.0f);
    }
    if (edge == 1 || edge == 4) {
        high = std::nextafter(high, std::numeric_limits<float>::infinity());
    }
    size_t i = 0;
    for (auto* data : {&a, &b}) {
        for (auto& z : *data) {
            const float v = i % 3 == 0 ? low : i % 3 == 1 ? high : 0.0f;
            z = {v, i % 2 ? -v : v};
            ++i;
        }
    }
}
} // namespace
