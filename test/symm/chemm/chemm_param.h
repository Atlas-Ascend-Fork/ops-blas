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
#include <stdexcept>
#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct ChemmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int64_t m = 0;
    int64_t n = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    bool nullAlpha = false;
    BlasFillMode aFill;
    int64_t lda = 0;
    BlasFillMode bFill;
    int64_t ldb = 0;
    float betaReal = 0.0f;
    float betaImag = 0.0f;
    bool nullBeta = false;
    BlasFillMode cFill;
    int64_t ldc = 0;

    std::string inputDistribution;
    std::array<float, 3> meanReal{}, meanImag{}, sigmaReal{}, sigmaImag{};

    ChemmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        LoadDistributionParameters(map);
        side = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        m = static_cast<int64_t>(parseInt64(ReadMap(map, "m", "0")));
        n = static_cast<int64_t>(parseInt64(ReadMap(map, "n", "0")));

        std::string alphaRealStr = ReadMap(map, "alpha_real", "1.0");
        std::string alphaImagStr = ReadMap(map, "alpha_imag", "0.0");
        nullAlpha = (alphaRealStr == "null" || alphaRealStr == "nullptr");
        alphaReal = nullAlpha ? 0.0f : parseFloat(alphaRealStr, 1.0f);
        alphaImag = nullAlpha ? 0.0f : parseFloat(alphaImagStr, 0.0f);

        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_1E3_1E3"));
        lda = static_cast<int64_t>(parseInt64(ReadMap(map, "lda", "0")));

        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_NORM_1E3_1E3"));
        ldb = static_cast<int64_t>(parseInt64(ReadMap(map, "ldb", "0")));

        std::string betaRealStr = ReadMap(map, "beta_real", "0.5");
        std::string betaImagStr = ReadMap(map, "beta_imag", "0.0");
        nullBeta = (betaRealStr == "null" || betaRealStr == "nullptr");
        betaReal = nullBeta ? 0.0f : parseFloat(betaRealStr, 0.5f);
        betaImag = nullBeta ? 0.0f : parseFloat(betaImagStr, 0.0f);

        cFill = parseFill(ReadMap(map, "c_fill", "RANDOM_NORM_1E3_1E3"));
        ldc = static_cast<int64_t>(parseInt64(ReadMap(map, "ldc", "0")));
        if (ReadMap(map, "lda", "").empty())
            lda = side == ACLBLAS_SIDE_LEFT ? m : n;
        if (ReadMap(map, "ldb", "").empty())
            ldb = m;
        if (ReadMap(map, "ldc", "").empty())
            ldc = m;
    }

private:
    void LoadDistributionParameters(const csv_map& map)
    {
        inputDistribution = ReadMap(map, "input_distribution", "");
        if (!inputDistribution.empty() && inputDistribution != "uniform" && inputDistribution != "normal")
            throw std::invalid_argument("Unknown Chemm input distribution");
        for (size_t i = 0; i < 3; ++i) {
            const std::string name(1, "abc"[i]);
            meanReal[i] = parseFloat(ReadMap(map, name + "_mean_real", "0"));
            meanImag[i] = parseFloat(ReadMap(map, name + "_mean_imag", "0"));
            sigmaReal[i] = parseFloat(ReadMap(map, name + "_sigma_real", "1"));
            sigmaImag[i] = parseFloat(ReadMap(map, name + "_sigma_imag", "1"));
            if (!inputDistribution.empty() && !DistributionInBounds(i))
                throw std::invalid_argument("Chemm distribution parameters outside task bounds");
        }
    }

    bool DistributionInBounds(size_t index) const
    {
        return meanReal[index] >= -5 && meanReal[index] <= 5 && meanImag[index] >= -5 && meanImag[index] <= 5 &&
               sigmaReal[index] >= .1f && sigmaReal[index] <= 2 && sigmaImag[index] >= .1f && sigmaImag[index] <= 2;
    }
};
