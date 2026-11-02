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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <complex>
#include <limits>
#include <utility>
#include <functional>
#include <random>
#include <sstream>
#include "blas_test.h"
#include "ctrmm_param.h"
#include "ctrmm_golden.h"
#include "ctrmm_npu_wrapper.h"
#include "aclblas_handle_internal.h"

class CtrmmArch22Test : public BlasTest<CtrmmParam> {
protected:
    using Matrix = std::vector<aclblasComplex>;
    using InputTransform = std::function<void(Matrix&, Matrix&)>;
    void Run(
        const CtrmmParam& p, bool inPlace = false, size_t offset = 0, const InputTransform& transform = {},
        bool exact = false)
    {
        if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m == 0 || p.n == 0) {
            ASSERT_NO_FATAL_FAILURE(CheckQuickReturn(p));
            return;
        }
        const int q = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
        auto a = makeBlasComplexMatrix(q, q, p.lda, p.aFill, p.randomSeed);
        auto b = makeBlasComplexMatrix(p.m, p.n, p.ldb, p.bFill, p.randomSeed + 1U);
        if (transform) {
            transform(a, b);
        }
        PoisonUnusedA(p, a, q);
        std::vector<aclblasComplex> c(static_cast<size_t>(p.ldc) * p.n, {713.25f, -917.5f});
        if (inPlace) {
            b.resize(static_cast<size_t>(std::max(p.ldb, p.ldc)) * p.n, {713.25f, -917.5f});
            c = b;
        }
        auto golden = c;
        const bool memoryOnly = std::getenv("CTRMM_MEMORY_ONLY") != nullptr;
        const bool perfOnly = std::getenv("CTRMM_PERF_ONLY") != nullptr || memoryOnly;
        if (!perfOnly) {
            CtrmmGolden(p, a, b, golden);
        }
        Matrix actual;
        ASSERT_NO_FATAL_FAILURE(Execute(p, a, b, c, actual, inPlace, offset, memoryOnly, perfOnly));
        if (perfOnly) {
            return;
        }
        ASSERT_NO_FATAL_FAILURE(CompareResults(p, a, b, actual, golden, exact));
    }

private:
    struct AccuracyStats {
        double maximum[2] = {0, 0};
        double maximumUlp[2] = {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min()};
        float maximumGolden[2] = {0, 0};
        size_t fixedCapFailures[2] = {0, 0};
        size_t matched[2] = {0, 0};
        size_t specialMismatch = 0;
    };
    void DiagnoseAccuracy(
        const CtrmmParam& p, const Matrix& a, const Matrix& b, int i, int j, int component, const float* av,
        const float* gv)
    {
        const int q = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
        std::complex<double> sum{};
        for (int k = 0; k < q; ++k) {
            int row = p.side == ACLBLAS_SIDE_LEFT ? i : k;
            int col = p.side == ACLBLAS_SIDE_LEFT ? k : j;
            if (p.trans != ACLBLAS_OP_N) {
                std::swap(row, col);
            }
            if (p.uplo == ACLBLAS_UPPER ? row > col : row < col) {
                continue;
            }
            auto avalue = a[static_cast<size_t>(col) * p.lda + row];
            std::complex<double> z = p.diag == ACLBLAS_UNIT && row == col ?
                                         std::complex<double>{1, 0} :
                                         std::complex<double>{avalue.real, avalue.imag};
            if (p.trans == ACLBLAS_OP_C) {
                z = std::conj(z);
            }
            auto x =
                b[p.side == ACLBLAS_SIDE_LEFT ? static_cast<size_t>(j) * p.ldb + k :
                                                static_cast<size_t>(k) * p.ldb + i];
            sum += z * std::complex<double>{x.real, x.imag};
        }
        sum *= std::complex<double>{p.alphaReal, p.alphaImag};
        std::cout << std::setprecision(12) << "CTRMM_DIAG " << p.caseName << " row=" << i << " col=" << j
                  << " component=" << component << " actual=" << av[component] << " netlib=" << gv[component]
                  << " fp64=" << (component == 0 ? sum.real() : sum.imag()) << '\n';
    }

    void RecordSpecial(
        const CtrmmParam& p, int i, int j, int component, const float* av, const float* gv, AccuracyStats& stats)
    {
        bool same = (std::isnan(gv[component]) && std::isnan(av[component])) ||
                    (std::isinf(gv[component]) && av[component] == gv[component]);
        stats.matched[component] += same;
        stats.specialMismatch += !same;
        if (!same && stats.specialMismatch <= 32U) {
            // Use a separate stream so diagnostics do not change
            // the formatting of the aggregate accuracy metrics.
            std::ostringstream detail;
            detail << std::setprecision(std::numeric_limits<float>::max_digits10) << "CTRMM_SPECIAL_MISMATCH "
                   << p.caseName << " row=" << i << " col=" << j << " component=" << (component == 0 ? "real" : "imag")
                   << " actual=" << av[component] << " golden=" << gv[component];
            std::cout << detail.str() << '\n';
        }
    }

    double RecordFinite(int component, const float* av, const float* gv, AccuracyStats& stats)
    {
        const double error = std::abs(static_cast<double>(av[component]) - gv[component]);
        const float magnitude = std::abs(gv[component]);
        // Pin the referenced opbase standard: use the golden value
        // at the maximum-error point, narrowed to the output dtype.
        // At tied maxima choose the smaller ULP (conservative).
        const double ulp =
            magnitude == std::numeric_limits<float>::max() ?
                static_cast<double>(magnitude) - std::nextafter(magnitude, 0.0f) :
                static_cast<double>(std::nextafter(magnitude, std::numeric_limits<float>::infinity())) - magnitude;
        if (error > stats.maximum[component] ||
            (error == stats.maximum[component] && ulp < stats.maximumUlp[component])) {
            stats.maximum[component] = error;
            stats.maximumUlp[component] = ulp;
            stats.maximumGolden[component] = gv[component];
        }
        stats.fixedCapFailures[component] += error > 0.01;
        stats.matched[component] += error <= (1.0 + std::abs(static_cast<double>(gv[component]))) / 8192.0;
        return error;
    }

    void CheckAccuracyStats(const CtrmmParam& p, const AccuracyStats& stats, bool exact)
    {
        const double total = static_cast<double>(p.m) * p.n;
        std::cout << "CTRMM_ACCURACY " << p.caseName << " matched_real=" << stats.matched[0] / total
                  << " matched_imag=" << stats.matched[1] / total << " max_real=" << stats.maximum[0]
                  << " max_imag=" << stats.maximum[1] << " special_mismatch=" << stats.specialMismatch << '\n';
        EXPECT_EQ(stats.specialMismatch, 0U);
        for (int cidx = 0; cidx < 2; ++cidx) {
            EXPECT_GE(stats.matched[cidx] / total, 0.99) << "component " << cidx;
            const double taskLimit = std::max(1e-2, 32.0 * stats.maximumUlp[cidx]);
            std::cout << "CTRMM_TASK_LIMIT " << p.caseName << " component=" << cidx
                      << " max_golden=" << std::setprecision(12) << stats.maximumGolden[cidx]
                      << " max_error=" << stats.maximum[cidx] << " ulp=" << stats.maximumUlp[cidx]
                      << " task_limit=" << taskLimit << " fixed_cap_failures=" << stats.fixedCapFailures[cidx] << '\n';
            EXPECT_LE(stats.maximum[cidx], taskLimit) << "component " << cidx;
            if (exact && std::getenv("CTRMM_CHECK_EXACT")) {
                EXPECT_EQ(stats.maximum[cidx], 0.0) << "component " << cidx;
            }
        }
    }

    void CompareComponents(
        const CtrmmParam& p, const Matrix& a, const Matrix& b, int i, int j, const float* av, const float* gv,
        AccuracyStats& stats)
    {
        for (int component = 0; component < 2; ++component) {
            if (!std::isfinite(gv[component]) || !std::isfinite(av[component])) {
                RecordSpecial(p, i, j, component, av, gv, stats);
            } else {
                const double error = RecordFinite(component, av, gv, stats);
                if (error > 0.01 && std::getenv("CTRMM_DIAG_ACCURACY")) {
                    DiagnoseAccuracy(p, a, b, i, j, component, av, gv);
                }
            }
        }
    }

    void CompareResults(
        const CtrmmParam& p, const Matrix& a, const Matrix& b, const Matrix& actual, const Matrix& golden, bool exact)
    {
        AccuracyStats stats;
        for (int j = 0; j < p.n; ++j) {
            for (int i = 0; i < p.ldc; ++i) {
                const size_t idx = static_cast<size_t>(j) * p.ldc + i;
                if (i >= p.m) {
                    ASSERT_EQ(std::memcmp(&actual[idx], &golden[idx], sizeof(aclblasComplex)), 0)
                        << "output padding row=" << i << " col=" << j;
                    continue;
                }
                const float av[2] = {actual[idx].real, actual[idx].imag};
                const float gv[2] = {golden[idx].real, golden[idx].imag};
                CompareComponents(p, a, b, i, j, av, gv, stats);
            }
        }
        CheckAccuracyStats(p, stats, exact);
    }

    void ReportTiming(
        const CtrmmParam& p, bool eventTiming, int repeats, const CtrmmEvent* starts, const CtrmmEvent* stops)
    {
        if (eventTiming) {
            if (repeats == 0 || repeats < 0) {
                ADD_FAILURE() << "Timing repeat count must be positive";
                return;
            }
            double sum = 0;
            std::cout << "CTRMM_PERF " << p.caseName << " samples_us=" << std::setprecision(9);
            for (int i = 0; i < repeats; ++i) {
                float elapsed = 0;
                CtrmmDeviceBuffer::Check(aclrtEventElapsedTime(&elapsed, starts[i].Get(), stops[i].Get()));
                const double us = elapsed * 1000.0;
                sum += us;
                std::cout << (i ? "," : "") << us;
            }
            std::cout << " mean_us=" << sum / repeats << '\n';
        }
    }

    void ReportMemory(
        const CtrmmParam& p, size_t matrixElements, size_t offset, size_t workspaceBefore, size_t freeBefore,
        size_t totalMemory, bool memoryOnly)
    {
        if (memoryOnly) {
            size_t freeAfter = 0;
            CtrmmDeviceBuffer::Check(aclrtGetMemInfo(ACL_HBM_MEM, &freeAfter, &totalMemory));
            std::cout << "CTRMM_MEMORY " << p.caseName << " matrix_bytes=" << matrixElements * sizeof(aclblasComplex)
                      << " guard_bytes=" << 3U * (16U + offset) * sizeof(aclblasComplex)
                      << " workspace_before=" << workspaceBefore
                      << " workspace_after=" << GetEffectiveWorkspaceSize(handle_)
                      << " device_free_before=" << freeBefore << " device_free_after=" << freeAfter
                      << " device_total=" << totalMemory << '\n';
        }
    }

    void ExecuteInvocations(
        const CtrmmParam& p, bool inPlace, bool perfOnly, bool memoryOnly,
        const std::function<aclblasStatus_t()>& invoke, const std::function<void()>& reportMemory)
    {
        int repeats = 1;
        if (perfOnly && !memoryOnly) {
            ASSERT_FALSE(inPlace);
            for (int i = 0; i < 5; ++i) {
                ASSERT_EQ(invoke(), ACLBLAS_STATUS_SUCCESS);
            }
            CtrmmDeviceBuffer::Check(aclrtSynchronizeStream(stream_));
            repeats = 11;
        }
        const bool eventTiming = perfOnly && !memoryOnly && std::getenv("CTRMM_EVENT_TIMING");
        std::unique_ptr<CtrmmEvent[]> starts(eventTiming ? new CtrmmEvent[11] : nullptr);
        std::unique_ptr<CtrmmEvent[]> stops(eventTiming ? new CtrmmEvent[11] : nullptr);
        for (int i = 0; i < repeats; ++i) {
            if (eventTiming) {
                CtrmmDeviceBuffer::Check(aclrtRecordEvent(starts[i].Get(), stream_));
            }
            ASSERT_EQ(invoke(), p.expectResult);
            if (eventTiming) {
                CtrmmDeviceBuffer::Check(aclrtRecordEvent(stops[i].Get(), stream_));
            }
        }
        CtrmmDeviceBuffer::Check(aclrtSynchronizeStream(stream_));
        reportMemory();
        ReportTiming(p, eventTiming, repeats, starts.get(), stops.get());
    }

    void Execute(
        const CtrmmParam& p, const Matrix& a, const Matrix& b, const Matrix& c, Matrix& actual, bool inPlace,
        size_t offset, bool memoryOnly, bool perfOnly)
    {
        size_t freeBefore = 0;
        size_t totalMemory = 0;
        if (memoryOnly) {
            CtrmmDeviceBuffer::Check(aclrtGetMemInfo(ACL_HBM_MEM, &freeBefore, &totalMemory));
        }
        const size_t workspaceBefore = GetEffectiveWorkspaceSize(handle_);
        CtrmmDeviceBuffer da(a, offset);
        CtrmmDeviceBuffer db(b, offset);
        CtrmmDeviceBuffer dc(c, offset);
        auto* out = inPlace ? db.Data() : dc.Data();
        const aclblasComplex alpha = p.GetAlpha();
        auto invoke = [&]() {
            return aclblasCtrmm(
                handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &alpha, p.nullA ? nullptr : da.Data(), p.lda,
                p.nullB ? nullptr : db.Data(), p.ldb, p.nullC ? nullptr : out, p.ldc);
        };
        auto reportMemory = [&]() {
            ReportMemory(
                p, a.size() + b.size() + c.size(), offset, workspaceBefore, freeBefore, totalMemory, memoryOnly);
        };
        ASSERT_NO_FATAL_FAILURE(ExecuteInvocations(p, inPlace, perfOnly, memoryOnly, invoke, reportMemory));
        actual = inPlace ? db.Read() : dc.Read();
    }

    void PoisonUnusedA(const CtrmmParam& p, Matrix& a, int q)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (int j = 0; j < q; ++j) {
            for (int i = 0; i < p.lda; ++i) {
                if (i >= q || (p.uplo == ACLBLAS_UPPER ? i > j : i < j) || (p.diag == ACLBLAS_UNIT && i == j)) {
                    a[static_cast<size_t>(j) * p.lda + i] = {nan, nan};
                }
            }
        }
    }

    void CheckQuickReturn(const CtrmmParam& p)
    {
        // All these cases must return before dereferencing matrix data.
        aclblasComplex dummy{};
        const aclblasComplex alpha = p.GetAlpha();
        auto status = aclblasCtrmm(
            p.description == "null_handle" ? nullptr : handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n,
            p.nullAlpha ? nullptr : &alpha, p.nullA ? nullptr : &dummy, p.lda, p.nullB ? nullptr : &dummy, p.ldb,
            p.nullC ? nullptr : &dummy, p.ldc);
        ASSERT_EQ(status, p.expectResult);
    }
};
