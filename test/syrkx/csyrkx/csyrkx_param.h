/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Independent Csyrkx acceptance tests; see repository LICENSE. */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "csv_loader.h"

namespace csyrkx_test {
struct Param {
    std::string caseName = "supplemental", description;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 7, k = 5, lda = 11, ldb = 13, ldc = 12;
    aclblasComplex alpha{0.5f, -0.75f}, beta{0.25f, 0.5f};
    bool alphaNull = false, betaNull = false;
    std::string aFill = "RANDOM_NORM_5_5", bFill = "RANDOM_NORM_5_5", cFill = "RANDOM_NORM_5_5";
    aclblasStatus_t expectResult = ACLBLAS_STATUS_SUCCESS;
    uint32_t randomSeed = 20261028U;
    // CSV legacy MERE/MARE are metadata, never substitute for fixed tolerances.
    float mereThreshold = 0.0f, mareMultiplier = 0.0f;

    Param() = default;
    void SetUplo(const std::string& u)
    {
        if (u == "ACLBLAS_UPPER")
            uplo = ACLBLAS_UPPER;
        else if (u == "ACLBLAS_LOWER")
            uplo = ACLBLAS_LOWER;
        else if (u == "999")
            uplo = static_cast<aclblasFillMode_t>(999);
        else
            throw std::runtime_error("unsupported uplo token: " + u);
    }

    explicit Param(const csv_map& row)
    {
        auto required = [&row](const char* key) -> std::string {
            auto it = row.find(key);
            if (it == row.end() || it->second.empty())
                throw std::runtime_error(std::string("missing ") + key);
            return it->second;
        };
        caseName = required("case_name");
        description = required("description");
        const auto u = required("uplo");
        const auto t = required("trans");
        SetUplo(u);
        if (t == "ACLBLAS_OP_N")
            trans = ACLBLAS_OP_N;
        else if (t == "ACLBLAS_OP_T")
            trans = ACLBLAS_OP_T;
        else if (t == "ACLBLAS_OP_C")
            trans = ACLBLAS_OP_C;
        else if (t == "999")
            trans = static_cast<aclblasOperation_t>(999);
        else
            throw std::runtime_error("unsupported trans token: " + t);
        n = ParseIntColumn(row, "n");
        k = ParseIntColumn(row, "k");
        lda = ParseIntColumn(row, "lda");
        ldb = ParseIntColumn(row, "ldb");
        ldc = ParseIntColumn(row, "ldc");
        alphaNull = required("alpha_real") == "null" || required("alpha_imag") == "null";
        betaNull = required("beta_real") == "null" || required("beta_imag") == "null";
        if (!alphaNull)
            alpha = {ParseFloatColumn(row, "alpha_real"), ParseFloatColumn(row, "alpha_imag")};
        if (!betaNull)
            beta = {ParseFloatColumn(row, "beta_real"), ParseFloatColumn(row, "beta_imag")};
        aFill = required("a_fill");
        bFill = required("b_fill");
        cFill = required("c_fill");
        const auto status = required("expect_result");
        if (status == "ACLBLAS_STATUS_SUCCESS")
            expectResult = ACLBLAS_STATUS_SUCCESS;
        else if (status == "ACLBLAS_STATUS_INVALID_VALUE")
            expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        else
            throw std::runtime_error("unsupported expected status: " + status);
        randomSeed = static_cast<uint32_t>(std::stoul(required("random_seed")));
        mereThreshold = ParseFloatColumn(row, "mere_threshold");
        mareMultiplier = ParseFloatColumn(row, "mare_multiplier");
    }
    bool IsPerformance() const { return caseName.rfind("TC_PF", 0) == 0; }
    bool Selected(int row, int col) const { return uplo == ACLBLAS_UPPER ? row <= col : row >= col; }
};

static_assert(sizeof(aclblasComplex) == 2 * sizeof(float), "CBLAS complex64 ABI required");
inline float Component(const aclblasComplex& z, unsigned component)
{
    float result;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&z) + component * sizeof(float);
    std::copy_n(bytes, sizeof(float), reinterpret_cast<unsigned char*>(&result));
    return result;
}
inline bool Zero(const aclblasComplex& z) { return Component(z, 0) == 0.0f && Component(z, 1) == 0.0f; }
inline bool One(const aclblasComplex& z) { return Component(z, 0) == 1.0f && Component(z, 1) == 0.0f; }

inline std::string CsvPath(const char* source)
{
    if (const char* path = std::getenv("CSYRKX_TEST_CSV"))
        return path;
    char executable[4096];
    const auto len = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (len > 0) {
        executable[len] = '\0';
        std::string path(executable);
        path = path.substr(0, path.find_last_of('/') + 1) + "csyrkx_test.csv";
        if (std::ifstream(path).good())
            return path;
    }
    return ReplaceFileExtension2Csv(source);
}

inline std::vector<Param> LoadCases(const char* source)
{
    auto result = GetCasesFromCsv<Param>(CsvPath(source));
    std::set<std::string> names;
    size_t performance = 0;
    for (const auto& p : result) {
        if (!names.insert(p.caseName).second)
            throw std::runtime_error("duplicate case: " + p.caseName);
        performance += p.IsPerformance() ? 1U : 0U;
    }
    // Fail closed: never silently drop malformed rows or accept a filtered CSV.
    if (result.size() != 1200 || performance != 200)
        throw std::runtime_error("Expected unchanged CSV: 1000 correctness + 200 performance rows");
    return result;
}
} // namespace csyrkx_test
