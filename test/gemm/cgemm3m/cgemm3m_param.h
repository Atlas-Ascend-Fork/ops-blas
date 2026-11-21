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

#include "../cgemm/cgemm_param.h"

inline csv_map Cgemm3mColumns(const csv_map& input)
{
    csv_map result = input;
    result["transA"] = ReadMap(input, "transa", "ACLBLAS_OP_N");
    result["transB"] = ReadMap(input, "transb", "ACLBLAS_OP_N");
    // Give the base parser valid placeholders; the derived constructor restores 999.
    for (const auto* key : {"transA", "transB"}) {
        if (result[key] == "999") {
            result[key] = "ACLBLAS_OP_N";
        }
    }
    result["alpha_null"] = ReadMap(input, "alpha_real", "") == "null" ? "true" : "false";
    result["beta_null"] = ReadMap(input, "beta_real", "") == "null" ? "true" : "false";
    // Placeholder values are never passed to the API when the corresponding pointer is null.
    if (result["alpha_null"] == "true") {
        result["alpha_real"] = "1";
        result["alpha_imag"] = "0";
    }
    if (result["beta_null"] == "true") {
        result["beta_real"] = "0";
        result["beta_imag"] = "0";
    }
    result["a_null"] = ReadMap(input, "a_fill", "") == "NULLPTR" ? "true" : "false";
    result["b_null"] = ReadMap(input, "b_fill", "") == "NULLPTR" ? "true" : "false";
    result["c_null"] = ReadMap(input, "c_fill", "") == "NULLPTR" ? "true" : "false";
    return result;
}

struct Cgemm3mParam : public CgemmParam {
    explicit Cgemm3mParam(const csv_map& input) : CgemmParam(Cgemm3mColumns(input))
    {
        if (ReadMap(input, "transa", "") == "999") {
            transA = static_cast<aclblasOperation_t>(999);
        }
        if (ReadMap(input, "transb", "") == "999") {
            transB = static_cast<aclblasOperation_t>(999);
        }
    }
};
