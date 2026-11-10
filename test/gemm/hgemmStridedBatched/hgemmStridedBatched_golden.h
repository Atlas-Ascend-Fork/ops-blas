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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <gtest/gtest.h>
#include <cblas.h>
#include "hgemmStridedBatched_param.h"
#include "reference_parallel.h"
#include "verify.h"

namespace hgemm_test {
inline float Ulp(float x) { return static_cast<float>(getUlpsAt(x, HALF_MANTISSA_BITS, HALF_MIN_NORMAL_EXPONENT)); }

struct BatchMetrics {
    uint64_t finite, matched, special, specialMatched;
    float maxError, anchor;
};

struct HgemmStridedBatchedGolden {
    const HgemmStridedBatchedParam& p;
    const std::vector<Half>& a;
    const std::vector<Half>& b;
    const std::vector<Half>& initial;
    const std::vector<Half>& actual;

    std::vector<float> Golden(int batch) const
    {
        const auto cta = p.ta == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans;
        const auto ctb = p.tb == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans;
        std::vector<float> golden(static_cast<size_t>(p.m) * p.n);
        for (int col = 0; col < p.n; ++col)
            for (int row = 0; row < p.m; ++row)
                golden[static_cast<size_t>(col) * p.m + row] =
                    F(p.beta) == 0 ? 0 : F(initial[batch * p.sc + col * p.ldc + row]);
        if (p.k == 0 || F(p.alpha) == 0) {
            for (auto& v : golden)
                v = F(p.beta) == 0 ? 0 : F(p.beta) * v;
        } else {
            std::vector<float> fa(static_cast<size_t>(p.lda) * (p.ta == ACLBLAS_OP_N ? p.k : p.m)),
                fb(static_cast<size_t>(p.ldb) * (p.tb == ACLBLAS_OP_N ? p.n : p.k));
            for (size_t j = 0; j < fa.size(); ++j)
                fa[j] = F(a[batch * p.sa + j]);
            for (size_t j = 0; j < fb.size(); ++j)
                fb[j] = F(b[batch * p.sb + j]);
            hgemm_reference::Gemm(
                cta, ctb, p.m, p.n, p.k, F(p.alpha), fa, p.lda, fb, p.ldb, F(p.beta), golden,
                std::filesystem::path(HGEMM_REFERENCE_WORKER_PATH));
        }
        return golden;
    }

    BatchMetrics Measure(int batch, const std::vector<float>& golden) const
    {
        uint64_t finite = 0, matched = 0, special = 0, specialMatched = 0;
        float maxError = 0, anchor = 0;
        for (int col = 0; col < p.n; ++col)
            for (int row = 0; row < p.m; ++row) {
                const float ref = F(H(golden[static_cast<size_t>(col) * p.m + row]));
                const float got = F(actual[batch * p.sc + col * p.ldc + row]);
                if (std::isfinite(ref) && std::isfinite(got)) {
                    ++finite;
                    const float err = std::abs(ref - got);
                    if (err <= std::ldexp(1.0f, -HALF_MANTISSA_BITS) * (1 + std::abs(ref)))
                        ++matched;
                    if (err > maxError) {
                        maxError = err;
                        anchor = ref;
                    }
                } else {
                    ++special;
                    if ((std::isnan(ref) && !std::isfinite(got)) || (std::isnan(got) && !std::isfinite(ref)) ||
                        (std::isinf(ref) && ref == got))
                        ++specialMatched;
                }
            }
        return {finite, matched, special, specialMatched, maxError, anchor};
    }

    bool VerifyFinite(int batch, const std::vector<float>& golden, size_t finite, float hardLimit) const
    {
        if (finite == 0)
            return true;
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT16, golden.data(), golden.size());
        // Task section 3.2 requires 2^-10 instead of the framework default 2^-9.
        cfg.mixedAtol = cfg.mixedRtol = std::ldexp(1.0, -HALF_MANTISSA_BITS);
        cfg.mixedRequiredMatchedRatio = REQUIRED_MATCHED_RATIO;
        // The task applies the hard limit at the maximum-error anchor. Compare()
        // enforces that limit; this floor avoids adding a different element policy.
        cfg.mixedMaxAbsErrorLimit = hardLimit;
        return Verifier::verifyEach(
            [&](auto consume) {
                for (int col = 0; col < p.n; ++col)
                    for (int row = 0; row < p.m; ++row) {
                        const float ref = F(H(golden[static_cast<size_t>(col) * p.m + row]));
                        const float got = F(actual[batch * p.sc + col * p.ldc + row]);
                        if (std::isfinite(ref) && std::isfinite(got))
                            consume(got, ref);
                    }
            },
            finite, cfg, p.caseName + "/batch" + std::to_string(batch));
    }

    void Compare() const
    {
        std::ofstream report;
        if (const char* path = std::getenv("HGEMM_RESULTS"))
            report.open(path, std::ios::app);
        uint64_t caseFinite = 0, caseMatched = 0, caseSpecial = 0, caseSpecialMatched = 0;
        float caseMax = 0;
        bool casePassed = true;
        for (int batch = 0; batch < p.batches; ++batch) {
            auto golden = Golden(batch);
            const auto [finite, matched, special, specialMatched, maxError, anchor] = Measure(batch, golden);
            const float hardLimit = std::max(MIN_ABSOLUTE_ERROR_LIMIT, MAX_ULP_MULTIPLIER * Ulp(anchor));
            const bool pass =
                VerifyFinite(batch, golden, finite, hardLimit) && maxError <= hardLimit && special == specialMatched;
            casePassed = casePassed && pass;
            caseFinite += finite;
            caseMatched += matched;
            caseSpecial += special;
            caseSpecialMatched += specialMatched;
            caseMax = std::max(caseMax, maxError);
            if (report)
                report << std::setprecision(std::numeric_limits<float>::max_digits10) << "{\"case\":\"" << p.caseName
                       << "\",\"batch\":" << batch << ",\"finite\":" << finite << ",\"matched\":" << matched
                       << ",\"nonfinite\":" << special << ",\"nonfinite_matched\":" << specialMatched
                       << ",\"max_abs\":" << maxError << ",\"anchor\":" << anchor << ",\"limit\":" << hardLimit
                       << ",\"passed\":" << (pass ? "true" : "false") << "}\n";
            EXPECT_TRUE(pass) << "batch=" << batch << " finite=" << finite << " matched=" << matched
                              << " max_abs=" << maxError << " limit=" << hardLimit << " nonfinite=" << special
                              << " matched=" << specialMatched;
            if (!pass)
                break; // a failing batch fails the case, retaining its diagnostics
        }
        ::testing::Test::RecordProperty("finite", std::to_string(caseFinite));
        ::testing::Test::RecordProperty("matched", std::to_string(caseMatched));
        ::testing::Test::RecordProperty("nonfinite", std::to_string(caseSpecial));
        ::testing::Test::RecordProperty("nonfinite_matched", std::to_string(caseSpecialMatched));
        ::testing::Test::RecordProperty("max_abs", std::to_string(caseMax));
        EXPECT_TRUE(casePassed);
    }
};
} // namespace hgemm_test
