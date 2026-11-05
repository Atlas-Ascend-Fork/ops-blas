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
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cblas.h>
#include "lt_matmul_copy.h"
#include "lt_matmul_param.h"

namespace LtMatmul22Test {

#ifdef LT_MATMUL_NETLIB_LIBRARY
inline auto ReferenceSgemm() -> decltype(&cblas_sgemm)
{
    static auto function = []() -> decltype(&cblas_sgemm) {
        int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
        flags |= RTLD_DEEPBIND;
#endif
        void* library = dlopen(LT_MATMUL_NETLIB_LIBRARY, flags);
        if (library == nullptr)
            throw std::runtime_error(std::string("Netlib dlopen failed: ") + dlerror());
        auto version = reinterpret_cast<const char* (*)()>(dlsym(library, "LtMatmulNetlibVersion"));
        auto sgemm = reinterpret_cast<decltype(&cblas_sgemm)>(dlsym(library, "cblas_sgemm"));
        if (version == nullptr || std::string(version()) != "3.10.0" || sgemm == nullptr) {
            throw std::runtime_error("Netlib 3.10.0 provenance/symbol check failed");
        }
        Dl_info location{};
        if (dladdr(reinterpret_cast<void*>(sgemm), &location) == 0 || location.dli_fname == nullptr) {
            throw std::runtime_error("Unable to identify cblas_sgemm provider");
        }
        std::cout << "LT_MATMUL_NETLIB version=3.10.0 library=" << location.dli_fname
                  << " sha256=" << LT_MATMUL_NETLIB_SHA256 << std::endl;
        return sgemm;
    }();
    return function;
}
#else
inline auto ReferenceSgemm() -> decltype(&cblas_sgemm)
{
    static auto function = []() -> decltype(&cblas_sgemm) {
        auto sgemm = &cblas_sgemm;
        Dl_info location{};
        if (dladdr(reinterpret_cast<void*>(sgemm), &location) == 0 || location.dli_fname == nullptr) {
            throw std::runtime_error("Unable to identify the repository CBLAS provider");
        }
        std::cout << "LT_MATMUL_REFERENCE provider=repository-cblas library=" << location.dli_fname << std::endl;
        return sgemm;
    }();
    return function;
}
#endif

inline std::vector<float> Golden(const Param& p, const std::array<std::vector<float>, 4>& input)
{
    auto output = input[3];
    if (p.m == 0 || p.n == 0)
        return output;
    const auto& c = p.inplace ? input[3] : input[2];
    const int ldc = p.inplace ? p.StorageLd(3) : p.StorageLd(2);
    const int ldd = p.StorageLd(3);
    if (p.alpha == 0.0f || p.k == 0) {
        for (int i = 0; i < p.m; ++i) {
            for (int j = 0; j < p.n; ++j) {
                const size_t dIndex = static_cast<size_t>(i) * ldd + j;
                if (p.beta == 0.0f)
                    output[dIndex] = 0.0f;
                else if (p.beta == 1.0f)
                    CopyBytes(
                        &output[dIndex], sizeof(output[dIndex]), &c[static_cast<size_t>(i) * ldc + j], sizeof(float));
                else
                    output[dIndex] = p.beta * c[static_cast<size_t>(i) * ldc + j];
            }
        }
        return output;
    }
    if (p.beta != 0.0f) {
        for (int i = 0; i < p.m; ++i) {
            std::copy_n(c.data() + static_cast<size_t>(i) * ldc, p.n, output.data() + static_cast<size_t>(i) * ldd);
        }
    }
    ReferenceSgemm()(
        CblasRowMajor, p.transA == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans,
        p.transB == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans, p.m, p.n, p.k, p.alpha, input[0].data(), p.StorageLd(0),
        input[1].data(), p.StorageLd(1), p.beta, output.data(), ldd);
    return output;
}

inline uint32_t FloatBits(float value)
{
    uint32_t bits = 0;
    CopyBytes(&bits, sizeof(bits), &value, sizeof(bits));
    return bits;
}

struct Metrics {
    size_t finite = 0, matched = 0, nonfinite = 0, nonfiniteMatched = 0, exactMismatches = 0;
    double matchedRatio = 1.0, maxAbs = 0.0, maxAbsLimit = 0.01;
    size_t worstIndex = 0;
    bool passed = true;
};

inline bool EquivalentNonfinite(float actual, float golden)
{
    if (std::isfinite(actual) || std::isfinite(golden))
        return false;
    if (std::isnan(actual) || std::isnan(golden))
        return true;
    return std::signbit(actual) == std::signbit(golden);
}

inline double FloatUlp(float value)
{
    const float magnitude = std::abs(value);
    if (magnitude < std::numeric_limits<float>::min())
        return std::numeric_limits<float>::denorm_min();
    int exponent = 0;
    std::frexp(static_cast<double>(magnitude), &exponent);
    return std::ldexp(1.0, exponent - 24);
}

inline Metrics Compare(const Param& p, const std::vector<float>& actual, const std::vector<float>& golden)
{
    Metrics result;
    const bool exact = p.alpha == 0.0f || p.k == 0 || p.m == 0 || p.n == 0;
    for (int i = 0; i < p.m; ++i) {
        for (int j = 0; j < p.n; ++j) {
            const size_t index = static_cast<size_t>(i) * p.StorageLd(3) + j;
            const float a = actual[index], g = golden[index];
            if (exact && FloatBits(a) != FloatBits(g))
                ++result.exactMismatches;
            if (!std::isfinite(a) || !std::isfinite(g)) {
                ++result.nonfinite;
                if (EquivalentNonfinite(a, g))
                    ++result.nonfiniteMatched;
                continue;
            }
            ++result.finite;
            const double error = std::abs(static_cast<double>(a) - static_cast<double>(g));
            constexpr double tolerance = 0x1p-13;
            if (error <= tolerance + tolerance * std::abs(static_cast<double>(g)))
                ++result.matched;
            if (error > result.maxAbs) {
                result.maxAbs = error;
                result.worstIndex = index;
                result.maxAbsLimit = std::max(0.01, 32.0 * FloatUlp(g));
            }
        }
    }
    result.matchedRatio = result.finite == 0 ? 1.0 : static_cast<double>(result.matched) / result.finite;
    result.passed = result.matchedRatio >= 0.99 && result.maxAbs <= result.maxAbsLimit &&
                    result.nonfinite == result.nonfiniteMatched && result.exactMismatches == 0;
    return result;
}

} // namespace LtMatmul22Test
