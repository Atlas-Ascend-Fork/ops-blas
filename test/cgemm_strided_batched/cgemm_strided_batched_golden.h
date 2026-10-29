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
#include <future>
#include <iomanip>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "cblas_compat.h"
#include "cgemm_strided_batched_param.h"
#include <cblas.h>
#include <dlfcn.h>
#include <link.h>
#include <iostream>
#include <stdexcept>
// Isolate CBLAS and its Fortran calls from process-wide BLAS interposition.
// dlsym resolves implementations rather than the executable's PLT stubs.
struct NetlibReference {
    void* library = nullptr;
    decltype(&cblas_cgemm) gemm = nullptr;
    NetlibReference()
    {
        library = dlmopen(LM_ID_NEWLM, CGEMM_SB_NETLIB_LIBRARY, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            throw std::runtime_error(dlerror());
        }
        gemm = reinterpret_cast<decltype(gemm)>(dlsym(library, "cblas_cgemm"));
        void* fortran = dlsym(library, "cgemm_");
        Dl_info cblas{}, fblas{};
        if (!gemm || !fortran || dlsym(library, "openblas_get_config") ||
            !dladdr(reinterpret_cast<void*>(gemm), &cblas) || !dladdr(fortran, &fblas) ||
            cblas.dli_fbase != fblas.dli_fbase) {
            dlclose(library);
            library = nullptr;
            throw std::runtime_error("Configured reference must provide Netlib CBLAS and Fortran CGEMM");
        }
        std::cout << "REFERENCE_BLAS config=Netlib BLAS"
                  << " cgemm_library=" << cblas.dli_fname << " fortran_library=" << fblas.dli_fname
                  << " configured_sha256=" << CGEMM_SB_NETLIB_SHA256 << '\n';
    }
    ~NetlibReference()
    {
        if (library) {
            dlclose(library);
        }
    }
    NetlibReference(const NetlibReference&) = delete;
    NetlibReference& operator=(const NetlibReference&) = delete;
};
inline const NetlibReference& GetNetlibReference()
{
    static const NetlibReference ref;
    return ref;
}

static bool EqualComponent(float actual, float golden, double& error)
{
    if (std::isnan(golden)) {
        if (std::isnan(actual)) {
            return true;
        }
        error = std::numeric_limits<double>::infinity();
        return false;
    }
    if (std::isinf(golden)) {
        if (std::isinf(actual) && std::signbit(actual) == std::signbit(golden)) {
            return true;
        }
        error = std::numeric_limits<double>::infinity();
        return false;
    }
    if (!std::isfinite(actual)) {
        error = std::numeric_limits<double>::infinity();
        return false;
    }
    error = std::abs(static_cast<double>(actual) - golden);
    return error <= (1.0 / 8192.0) * (1.0 + std::abs(static_cast<double>(golden)));
}

static void GenerateGolden(
    const CgemmSbParam& p, const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& golden)
{
    auto goldenBatch = [&](int batch) {
        auto* out = golden.data() + 2 * static_cast<size_t>(batch * p.strideC);
        if (p.alpha.real == 0.0f && p.alpha.imag == 0.0f) {
            for (int col = 0; col < p.n; ++col) {
                for (int row = 0; row < p.m; ++row) {
                    size_t i = 2 * (static_cast<size_t>(col) * p.ldc + row);
                    const float r = out[i], im = out[i + 1];
                    out[i] = p.beta.real == 0.0f && p.beta.imag == 0.0f ? 0.0f : p.beta.real * r - p.beta.imag * im;
                    out[i + 1] = p.beta.real == 0.0f && p.beta.imag == 0.0f ? 0.0f : p.beta.real * im + p.beta.imag * r;
                }
            }
        } else {
            GetNetlibReference().gemm(
                CblasColMajor, ToCblasOp(p.transA), ToCblasOp(p.transB), p.m, p.n, p.k, &p.alpha,
                a.data() + 2 * static_cast<size_t>(batch * p.strideA), p.lda,
                b.data() + 2 * static_cast<size_t>(batch * p.strideB), p.ldb, &p.beta, out, p.ldc);
        }
    };
    // Each independent batch calls the same Netlib golden routine.
    const int workers = p.strideC >= static_cast<int64_t>(p.ldc) * p.n ? std::min(16, p.batchCount) : 1;
    if (workers == 1) {
        for (int batch = 0; batch < p.batchCount; ++batch) {
            goldenBatch(batch);
        }
    } else {
        std::vector<std::future<void>> pending;
        for (int worker = 0; worker < workers; ++worker) {
            pending.emplace_back(std::async(std::launch::async, [&, worker] {
                for (int batch = worker; batch < p.batchCount; batch += workers) {
                    goldenBatch(batch);
                }
            }));
        }
        for (auto& task : pending) {
            task.get();
        }
    }
}

static void CheckOutput(
    const CgemmSbParam& p, const std::vector<float>& c, const std::vector<float>& golden, size_t cCount)
{
    size_t matched = 0, total = 0;
    double maxError = 0.0;
    for (int batch = 0; batch < p.batchCount; ++batch) {
        for (int col = 0; col < p.n; ++col) {
            for (int row = 0; row < p.m; ++row) {
                for (int component = 0; component < 2; ++component) {
                    const size_t index =
                        2 * (static_cast<size_t>(batch * p.strideC) + static_cast<size_t>(col) * p.ldc + row) +
                        component;
                    double error = 0.0;
                    const bool equal = EqualComponent(c[index], golden[index], error);
                    matched += equal ? 1 : 0;
                    maxError = std::max(maxError, error);
                    ++total;
                }
            }
        }
    }
    EXPECT_GE(static_cast<double>(matched) / total, 0.99) << p.caseName;
    EXPECT_LE(maxError, 1e-2) << p.caseName;
    if (p.caseName.rfind("TC_RF_", 0) == 0) {
        std::vector<uint8_t> logical(cCount, 0);
        for (int batch = 0; batch < p.batchCount; ++batch) {
            for (int col = 0; col < p.n; ++col) {
                for (int row = 0; row < p.m; ++row) {
                    logical[batch * p.strideC + col * p.ldc + row] = 1;
                }
            }
        }
        for (size_t i = 0; i < cCount; ++i) {
            if (!logical[i]) {
                ASSERT_EQ(c[2 * i], golden[2 * i]) << "C padding real at " << i;
                ASSERT_EQ(c[2 * i + 1], golden[2 * i + 1]) << "C padding imag at " << i;
            }
        }
    }
    std::cout << std::setprecision(10) << "ACCURACY_METRIC " << p.caseName
              << " matched_ratio=" << static_cast<double>(matched) / total << " max_abs_error=" << maxError
              << " total_components=" << total << '\n';
}
