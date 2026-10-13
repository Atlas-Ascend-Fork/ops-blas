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

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// Pivot mode: whether PivotArray is allocated or nullptr
enum class GetrfPivotMode {
    PIVOT,   // PivotArray is allocated (use partial pivoting)
    NO_PIVOT // PivotArray is nullptr (no pivoting)
};

inline GetrfPivotMode parsePivotMode(const std::string& s)
{
    if (s == "nullptr" || s == "NULLPTR" || s == "NO_PIVOT" || s == "null")
        return GetrfPivotMode::NO_PIVOT;
    return GetrfPivotMode::PIVOT;
}

// Matrix type for input data generation
enum class GetrfMatrixType {
    RANDOM_NONSINGULAR,     // Random complex values with a strengthened diagonal
    SINGULAR_ZERO_COL,      // First column all zeros
    SINGULAR_DEPENDENT_ROW, // One row = 2x another row
    IDENTITY,               // Identity matrix
    DIAGONALLY_DOMINANT,    // Strong real diagonal, complex off-diagonal values
    ILL_CONDITIONED,        // Hilbert matrix
    MIXED,                  // Mix of nonsingular and singular across batches
    NULLPTR_AARRAY,         // Error test: pass nullptr as Aarray
    NULLPTR_INFOARRAY,      // Error test: pass nullptr as infoArray
    NULLPTR_HANDLE          // Error test: pass nullptr as handle
};

inline GetrfMatrixType parseMatrixType(const std::string& s)
{
    if (s == "SINGULAR_ZERO_COL")
        return GetrfMatrixType::SINGULAR_ZERO_COL;
    if (s == "SINGULAR_DEPENDENT_ROW")
        return GetrfMatrixType::SINGULAR_DEPENDENT_ROW;
    if (s == "IDENTITY")
        return GetrfMatrixType::IDENTITY;
    if (s == "DIAGONALLY_DOMINANT")
        return GetrfMatrixType::DIAGONALLY_DOMINANT;
    if (s == "ILL_CONDITIONED")
        return GetrfMatrixType::ILL_CONDITIONED;
    if (s == "MIXED")
        return GetrfMatrixType::MIXED;
    if (s == "NULLPTR_AARRAY" || s == "nullptr_Aarray")
        return GetrfMatrixType::NULLPTR_AARRAY;
    if (s == "NULLPTR_INFOARRAY")
        return GetrfMatrixType::NULLPTR_INFOARRAY;
    if (s == "NULLPTR_HANDLE")
        return GetrfMatrixType::NULLPTR_HANDLE;
    return GetrfMatrixType::RANDOM_NONSINGULAR;
}

struct CgetrfBatchedParam : public BlasTestParamBase {
    int n = 0;
    int lda = 0;
    int batchSize = 0;
    GetrfPivotMode pivotMode = GetrfPivotMode::PIVOT;
    GetrfMatrixType matrixType = GetrfMatrixType::RANDOM_NONSINGULAR;

    CgetrfBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        const int defaultLda = std::max(1, n);
        lda = parseInt(ReadMap(map, "lda", std::to_string(defaultLda)), defaultLda);
        batchSize = parseInt(ReadMap(map, "batch_size", "1"));
        pivotMode = parsePivotMode(ReadMap(map, "pivot_mode", "PIVOT"));
        matrixType = parseMatrixType(ReadMap(map, "matrix_type", "RANDOM_NONSINGULAR"));
    }
};
