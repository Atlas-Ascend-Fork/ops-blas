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
#include <ostream>
#include <string>

#include "cann_ops_blas.h"
#include "csv_loader.h"

namespace cgemm_strided_batched_test {

using Complex = aclblasComplex;

struct CgemmStridedBatchedParam : public BlasTestParamBase {
    csv_map fields;

    explicit CgemmStridedBatchedParam(const csv_map& row) : BlasTestParamBase(row), fields(row) {}

    std::string S(const char* key) const { return ReadMap(fields, key); }
    int I(const char* key) const { return parseInt(S(key)); }
    int64_t L(const char* key) const { return parseInt64(S(key)); }
    float F(const char* key) const { return parseFloat(S(key)); }
    bool Flag(const char* key) const
    {
        const std::string value = S(key);
        return value == "NULLPTR" || value == "true" || value == "1";
    }
};

using Case = CgemmStridedBatchedParam;

inline void PrintTo(const CgemmStridedBatchedParam& param, std::ostream* out) { *out << param.caseName; }

inline int ParseTrans(const std::string& value)
{
    if (value == "N") {
        return ACLBLAS_OP_N;
    }
    if (value == "T") {
        return ACLBLAS_OP_T;
    }
    if (value == "C") {
        return ACLBLAS_OP_C;
    }
    return parseInt(value, -1);
}

} // namespace cgemm_strided_batched_test
