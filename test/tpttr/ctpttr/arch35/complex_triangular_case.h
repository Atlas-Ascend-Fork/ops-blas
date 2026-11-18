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

#include <cmath>
#include <cstdint>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cann_ops_blas.h"

namespace complex_triangular_test {

using CaseRow = std::unordered_map<std::string, std::string>;

inline std::string Require(const CaseRow& row, const char* key)
{
    const auto found = row.find(key);
    if (found == row.end() || found->second.empty()) {
        throw std::runtime_error(std::string("missing or empty column: ") + key);
    }
    return found->second;
}

inline int Integer(const CaseRow& row, const char* key)
{
    const std::string text = Require(row, key);
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
        throw std::runtime_error(std::string("invalid integer column: ") + key);
    }
    return value;
}

inline void RequireZero(const CaseRow& row, const char* key)
{
    const std::string text = Require(row, key);
    size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value) || value != 0.0) {
        throw std::runtime_error(std::string("bit-exact threshold must be zero: ") + key);
    }
}

struct TriangularCase {
    std::string caseName;
    std::string description;
    aclblasFillMode_t uplo;
    int n;
    int lda;
    std::string apFill;
    std::string aFill;
    aclblasStatus_t expected;
    uint32_t seed;

    explicit TriangularCase(const CaseRow& row)
    {
        caseName = Require(row, "case_name");
        description = Require(row, "description");
        const std::string value = Require(row, "uplo");
        if (value == "ACLBLAS_UPPER" || value == "UPPER") {
            uplo = ACLBLAS_UPPER;
        } else if (value == "ACLBLAS_LOWER" || value == "LOWER") {
            uplo = ACLBLAS_LOWER;
        } else if (value == "INVALID") {
            uplo = static_cast<aclblasFillMode_t>(255);
        } else {
            uplo = static_cast<aclblasFillMode_t>(Integer(row, "uplo"));
        }
        n = Integer(row, "n");
        lda = Integer(row, "lda");
        apFill = Require(row, "ap");
        aFill = Require(row, "a");
        const std::string status = Require(row, "expect_result");
        if (status == "ACLBLAS_STATUS_SUCCESS") {
            expected = ACLBLAS_STATUS_SUCCESS;
        } else if (status == "ACLBLAS_STATUS_INVALID_VALUE") {
            expected = ACLBLAS_STATUS_INVALID_VALUE;
        } else {
            throw std::runtime_error("unsupported expected status: " + status);
        }
        const int seedValue = Integer(row, "random_seed");
        if (seedValue <= 0) {
            throw std::runtime_error("random_seed must be positive");
        }
        seed = static_cast<uint32_t>(seedValue);
        RequireZero(row, "mere_threshold");
        RequireZero(row, "mare_multiplier");
    }

    bool Performance() const { return caseName.rfind("TC_PF_", 0) == 0; }
};

inline std::vector<std::string> SplitCsv(const std::string& line)
{
    std::vector<std::string> fields;
    std::string value;
    bool quoted = false;
    for (size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (quoted && index + 1U < line.size() && line[index + 1U] == '"') {
                value += '"';
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            fields.push_back(value);
            value.clear();
        } else if (character != '\r') {
            value += character;
        }
    }
    if (quoted) {
        throw std::runtime_error("unterminated quoted CSV field");
    }
    fields.push_back(value);
    return fields;
}

inline std::vector<TriangularCase> LoadCases(const std::string& path)
{
    std::ifstream input(path);
    std::string line;
    if (!input || !std::getline(input, line)) {
        throw std::runtime_error("cannot read CSV: " + path);
    }
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        line.erase(0, 3);
    }
    const auto headers = SplitCsv(line);
    if (std::set<std::string>(headers.begin(), headers.end()).size() != headers.size()) {
        throw std::runtime_error("duplicate CSV columns");
    }
    std::vector<TriangularCase> cases;
    std::set<std::string> names;
    while (std::getline(input, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }
        const auto values = SplitCsv(line);
        if (values.size() != headers.size()) {
            throw std::runtime_error("CSV field count mismatch");
        }
        CaseRow row;
        for (size_t index = 0; index < values.size(); ++index) {
            row.emplace(headers[index], values[index]);
        }
        cases.emplace_back(row);
        if (!names.insert(cases.back().caseName).second) {
            throw std::runtime_error("duplicate case: " + cases.back().caseName);
        }
    }
    if (cases.empty()) {
        throw std::runtime_error("CSV contains no test cases");
    }
    return cases;
}

inline TriangularCase ManualCase(int n, int lda, aclblasFillMode_t uplo)
{
    const CaseRow row = {
        {"case_name", "Manual"},
        {"description", "manual triangular storage test"},
        {"uplo", uplo == ACLBLAS_UPPER ? "ACLBLAS_UPPER" : "ACLBLAS_LOWER"},
        {"n", std::to_string(n)},
        {"lda", std::to_string(lda)},
        {"ap", "INDEX"},
        {"a", "INDEX"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"random_seed", "137"},
        {"mere_threshold", "0.0"},
        {"mare_multiplier", "0.0"}};
    return TriangularCase(row);
}

} // namespace complex_triangular_test
