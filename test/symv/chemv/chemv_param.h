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

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cann_ops_blas.h"
#include "csv_loader.h"

struct ChemvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    int lda = 1;
    int incx = 1;
    int incy = 1;
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    bool nullAlpha = false;
    bool nullBeta = false;
    BlasFillMode aFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode xFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode yFill = parseFill("RANDOM_NORM_5_5");
    double gpuBaselineMs = 0.0;

    explicit ChemvParam(const csv_map& row) : BlasTestParamBase(row)
    {
        for (const char* key :
             {"case_name", "uplo", "n", "lda", "incx", "incy", "a_fill", "x_fill", "y_fill", "alpha_real", "alpha_imag",
              "beta_real", "beta_imag", "expect_result"}) {
            if (ReadMap(row, key).empty()) {
                throw std::runtime_error(std::string("missing Chemv CSV column: ") + key);
            }
        }
        uplo = parseFillMode(ReadMap(row, "uplo"));
        n = ParseIntColumn(row, "n");
        lda = ParseIntColumn(row, "lda");
        incx = ParseIntColumn(row, "incx");
        incy = ParseIntColumn(row, "incy");
        const std::string ar = ReadMap(row, "alpha_real");
        const std::string br = ReadMap(row, "beta_real");
        nullAlpha = (ar == "null" || ar == "NULLPTR");
        nullBeta = (br == "null" || br == "NULLPTR");
        alpha = {nullAlpha ? 0.0f : ParseFloatColumn(row, "alpha_real"), ParseFloatColumn(row, "alpha_imag")};
        beta = {nullBeta ? 0.0f : ParseFloatColumn(row, "beta_real"), ParseFloatColumn(row, "beta_imag")};
        aFill = parseFill(ReadMap(row, "a_fill"));
        xFill = parseFill(ReadMap(row, "x_fill"));
        yFill = parseFill(ReadMap(row, "y_fill"));
    }

    bool IsPerformanceCase() const { return caseName.rfind("TC_PF_", 0) == 0; }

    bool IsQuickOrNegativeCase() const
    {
        const bool qr = !nullAlpha && !nullBeta && alpha.real == 0.0f && alpha.imag == 0.0f && beta.real == 1.0f &&
                        beta.imag == 0.0f;
        return expectResult != ACLBLAS_STATUS_SUCCESS || n <= 0 || qr;
    }
};
