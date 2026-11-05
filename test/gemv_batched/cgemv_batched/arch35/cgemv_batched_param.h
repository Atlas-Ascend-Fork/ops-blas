/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cgemv_batched_param.h
 * \brief CSV parameter struct for cgemv_batched tests (arch35).
 *        Column order matches task §2.3 signature (handle removed):
 *        case_name,description,trans,m,n,alpha_real,alpha_imag,aarray_fill,lda,
 *        xarray_fill,incx,beta_real,beta_imag,yarray_fill,incy,batchCount,
 *        expect_result,mere_threshold,mare_multiplier,random_seed
 */

#pragma once

#include <string>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct CgemvBatchedParam : public BlasTestParamBase {
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    BlasFillMode aFill = parseFill("RANDOM");
    int lda = 0;
    BlasFillMode xFill = parseFill("RANDOM");
    int incx = 1;
    float betaReal = 0.0f;
    float betaImag = 0.0f;
    BlasFillMode yFill = parseFill("RANDOM");
    int incy = 1;
    int batchCount = 1;

    // Null-pointer flags parsed from fill tokens.
    bool alphaIsNull = false;
    bool betaIsNull = false;
    bool aIsNull = false;
    bool xIsNull = false;
    bool yIsNull = false;

    CgemvBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        alphaReal = parseFloat(ReadMap(map, "alpha_real", "1.0"));
        alphaImag = parseFloat(ReadMap(map, "alpha_imag", "0.0"));
        std::string aStr = ReadMap(map, "aarray_fill", "RANDOM");
        aFill = parseFill(aStr);
        aIsNull = (aStr == "NULLPTR" || aFill.method == BlasFillMode::M_NULLPTR);
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
        std::string xStr = ReadMap(map, "xarray_fill", "RANDOM");
        xFill = parseFill(xStr);
        xIsNull = (xStr == "NULLPTR" || xFill.method == BlasFillMode::M_NULLPTR);
        incx = parseInt(ReadMap(map, "incx", "1"));
        betaReal = parseFloat(ReadMap(map, "beta_real", "0.0"));
        betaImag = parseFloat(ReadMap(map, "beta_imag", "0.0"));
        std::string yStr = ReadMap(map, "yarray_fill", "RANDOM");
        yFill = parseFill(yStr);
        yIsNull = (yStr == "NULLPTR" || yFill.method == BlasFillMode::M_NULLPTR);
        incy = parseInt(ReadMap(map, "incy", "1"));
        batchCount = parseInt(ReadMap(map, "batchCount", "1"));

        // Scalar null pointers are encoded as "null" in the CSV.
        std::string aRaw = ReadMap(map, "alpha_real", "1.0");
        std::string bRaw = ReadMap(map, "beta_real", "0.0");
        alphaIsNull = alphaIsNull || (aRaw == "null");
        betaIsNull = betaIsNull || (bRaw == "null");
    }
};
