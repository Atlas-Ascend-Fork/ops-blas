/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <string>
#include <vector>
#include "verify.h"

template <typename Param>
static inline void VerifyUploTriangle(const Param& p, const float* cPtr, const float* goldPtr, size_t cSize)
{
    if (cSize == 0) {
        return;
    }

    std::vector<float> npuUplo;
    std::vector<float> goldenUplo;
    std::vector<float> npuNonUplo;
    std::vector<float> goldenNonUplo;
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                npuUplo.push_back(cPtr[idx]);
                goldenUplo.push_back(goldPtr[idx]);
            } else {
                npuNonUplo.push_back(cPtr[idx]);
                goldenNonUplo.push_back(goldPtr[idx]);
            }
        }
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenUplo.data(), goldenUplo.size());
    EXPECT_TRUE(Verifier::verifyVector(npuUplo.data(), goldenUplo.data(), npuUplo.size(), 1, cfg, p.caseName));

    // Non-uplo triangle carries the "value unchanged" contract (NaN/Inf must
    // stay in place and must not spread): compare exactly, matching the csyrk
    // and cherk non-uplo checks (issue #750 SRK-06).
    if (!npuNonUplo.empty()) {
        VerifyConfig cfgNonUplo;
        cfgNonUplo.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(
            npuNonUplo.data(), goldenNonUplo.data(), npuNonUplo.size(), 1, cfgNonUplo,
            std::string(p.caseName) + "_nonuplo"));
    }
}

// Collect uplo / non-uplo real/imag element pairs from the NPU output and the
// golden buffer; the gold side is parameterized by GoldT — GoldT=float is the
// float chain (goldPtr is aclblasComplex*), GoldT=double is the double golden
// dual chain (goldPtr is aclblasDoubleComplex*, the non-uplo segment takes the
// float old value via static_cast<float>). UPPER ⇒ i <= j, LOWER ⇒ i >= j;
// `goldPtr` doubles as the "input old" reference for the non-uplo triangle
// because the golden restores that region to the input-old values.
template <typename Param, typename GoldT, typename GoldPtrT>
inline void CollectUploNonUploElementsT(
    const Param& p, const aclblasComplex* cPtr, const GoldPtrT* goldPtr, std::vector<float>& npuUploRe,
    std::vector<float>& npuUploIm, std::vector<GoldT>& goldUploRe, std::vector<GoldT>& goldUploIm,
    std::vector<float>& npuNonRe, std::vector<float>& npuNonIm, std::vector<float>& oldNonRe,
    std::vector<float>& oldNonIm)
{
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                npuUploRe.push_back(cPtr[idx].real);
                npuUploIm.push_back(cPtr[idx].imag);
                goldUploRe.push_back(goldPtr[idx].real);
                goldUploIm.push_back(goldPtr[idx].imag);
            } else {
                npuNonRe.push_back(cPtr[idx].real);
                npuNonIm.push_back(cPtr[idx].imag);
                oldNonRe.push_back(static_cast<float>(goldPtr[idx].real));
                oldNonIm.push_back(static_cast<float>(goldPtr[idx].imag));
            }
        }
    }
}
