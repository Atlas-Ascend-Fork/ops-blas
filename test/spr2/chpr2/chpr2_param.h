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

#include <cstdint>
#include <stdexcept>
#include <string>
#include "csv_loader.h"

using Complex = aclblasComplex;
struct Case {
    std::string name, xFill, yFill, apFill;
    int n = 0, incx = 1, incy = 1;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    Complex alpha{1.0f, 0.0f};
    bool alphaNull = false, poisonDiagonal = false;
    uint32_t seed = 0;
    aclblasStatus_t expected = ACLBLAS_STATUS_SUCCESS;
    double mereThreshold = 1.0 / 8192.0, mareMultiplier = 10.0;
    Case() = default;
    explicit Case(const csv_map& row)
    {
        name = row.at("case_name");
        const auto& fillMode = row.at("uplo");
        uplo = fillMode == "ACLBLAS_UPPER" ? ACLBLAS_UPPER :
               fillMode == "ACLBLAS_LOWER" ? ACLBLAS_LOWER :
                                             static_cast<aclblasFillMode_t>(std::stoi(fillMode));
        n = std::stoi(row.at("n"));
        alphaNull = row.at("alpha_real") == "null";
        alpha = {alphaNull ? 0.0f : std::stof(row.at("alpha_real")), std::stof(row.at("alpha_imag"))};
        xFill = row.at("x_fill");
        incx = std::stoi(row.at("incx"));
        yFill = row.at("y_fill");
        incy = std::stoi(row.at("incy"));
        apFill = row.at("ap_fill");
        const auto& status = row.at("expect_result");
        if (status == "ACLBLAS_STATUS_SUCCESS")
            expected = ACLBLAS_STATUS_SUCCESS;
        else if (status == "ACLBLAS_STATUS_INVALID_VALUE")
            expected = ACLBLAS_STATUS_INVALID_VALUE;
        else if (status == "ACLBLAS_STATUS_INVALID_ENUM")
            expected = ACLBLAS_STATUS_INVALID_ENUM;
        else
            throw std::runtime_error("Unsupported expected status: " + status);
        mereThreshold = std::stod(row.at("mere_threshold"));
        mareMultiplier = std::stod(row.at("mare_multiplier"));
        seed = static_cast<uint32_t>(std::stoul(row.at("random_seed")));
    }
};
