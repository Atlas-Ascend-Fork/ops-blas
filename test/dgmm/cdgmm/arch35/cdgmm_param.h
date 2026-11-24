/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cann_ops_blas.h"

namespace cdgmm_arch35_test {

using CaseRow = std::unordered_map<std::string, std::string>;

inline std::string RequireColumn(const CaseRow& row, const char* key)
{
    const auto found = row.find(key);
    if (found == row.end() || found->second.empty()) {
        throw std::runtime_error(std::string("missing or empty CSV column: ") + key);
    }
    return found->second;
}

inline int ReadInteger(const CaseRow& row, const char* key)
{
    const std::string value = RequireColumn(row, key);
    size_t used = 0;
    const int result = std::stoi(value, &used);
    if (used != value.size()) {
        throw std::runtime_error(std::string("invalid integer column: ") + key);
    }
    return result;
}

inline double ReadReal(const CaseRow& row, const char* key)
{
    const std::string value = RequireColumn(row, key);
    size_t used = 0;
    const double result = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(result) || result <= 0.0) {
        throw std::runtime_error(std::string("invalid positive real column: ") + key);
    }
    return result;
}

struct CdgmmArch35Param {
    std::string caseName;
    std::string description;
    aclblasSideMode_t mode;
    int m;
    int n;
    int incx;
    int lda;
    int ldc;
    std::string xFill;
    std::string aFill;
    int nullx;
    int nullA;
    int nullC;
    aclblasStatus_t expected;
    double mereThreshold;
    double mareMultiplier;
    uint32_t randomSeed;

    explicit CdgmmArch35Param(const CaseRow& row)
    {
        caseName = RequireColumn(row, "case_name");
        description = RequireColumn(row, "description");
        const std::string modeText = RequireColumn(row, "mode");
        if (modeText == "LEFT" || modeText == "ACLBLAS_SIDE_LEFT") {
            mode = ACLBLAS_SIDE_LEFT;
        } else if (modeText == "RIGHT" || modeText == "ACLBLAS_SIDE_RIGHT") {
            mode = ACLBLAS_SIDE_RIGHT;
        } else {
            mode = static_cast<aclblasSideMode_t>(ReadInteger(row, "mode"));
        }
        m = ReadInteger(row, "m");
        n = ReadInteger(row, "n");
        incx = ReadInteger(row, "incx");
        lda = ReadInteger(row, "lda");
        ldc = ReadInteger(row, "ldc");
        xFill = RequireColumn(row, "x_fill");
        aFill = RequireColumn(row, "a_fill");
        nullx = ReadInteger(row, "nullx");
        nullA = ReadInteger(row, "nullA");
        nullC = ReadInteger(row, "nullC");
        if ((nullx != 0 && nullx != 1) || (nullA != 0 && nullA != 1) || (nullC != 0 && nullC != 1)) {
            throw std::runtime_error("null flags must be 0 or 1");
        }
        const std::string status = RequireColumn(row, "expect_result");
        if (status == "ACLBLAS_STATUS_SUCCESS") {
            expected = ACLBLAS_STATUS_SUCCESS;
        } else if (status == "ACLBLAS_STATUS_INVALID_VALUE") {
            expected = ACLBLAS_STATUS_INVALID_VALUE;
        } else if (status == "ACLBLAS_STATUS_HANDLE_IS_NULLPTR") {
            expected = ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
        } else {
            throw std::runtime_error("unknown expected status: " + status);
        }
        mereThreshold = ReadReal(row, "mere_threshold");
        mareMultiplier = ReadReal(row, "mare_multiplier");
        const int seed = ReadInteger(row, "random_seed");
        if (seed <= 0) {
            throw std::runtime_error("random_seed must be positive");
        }
        randomSeed = static_cast<uint32_t>(seed);
    }

    bool IsPerformance() const { return caseName.rfind("TC_PF_", 0) == 0; }
};

inline std::vector<std::string> SplitCsv(const std::string& line)
{
    std::vector<std::string> values;
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
            values.push_back(value);
            value.clear();
        } else if (character != '\r') {
            value += character;
        }
    }
    if (quoted) {
        throw std::runtime_error("unterminated CSV quoted field");
    }
    values.push_back(value);
    return values;
}

inline std::vector<CdgmmArch35Param> LoadCdgmmCases(const std::string& fileName)
{
    std::ifstream input(fileName);
    if (!input) {
        throw std::runtime_error("cannot open CSV: " + fileName);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty CSV: " + fileName);
    }
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        line.erase(0, 3);
    }
    const auto keys = SplitCsv(line);
    if (std::set<std::string>(keys.begin(), keys.end()).size() != keys.size()) {
        throw std::runtime_error("duplicate CSV header");
    }
    std::vector<CdgmmArch35Param> cases;
    std::set<std::string> names;
    while (std::getline(input, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }
        const auto values = SplitCsv(line);
        if (values.size() != keys.size()) {
            throw std::runtime_error("CSV field count mismatch");
        }
        CaseRow row;
        for (size_t index = 0; index < keys.size(); ++index) {
            row.emplace(keys[index], values[index]);
        }
        cases.emplace_back(row);
        if (!names.insert(cases.back().caseName).second) {
            throw std::runtime_error("duplicate case_name: " + cases.back().caseName);
        }
    }
    if (cases.empty()) {
        throw std::runtime_error("CSV contains no cases");
    }
    return cases;
}

} // namespace cdgmm_arch35_test
