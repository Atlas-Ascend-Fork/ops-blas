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

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "arch35/chemv_param.h"

inline const std::vector<std::string>& ChemvCsvColumns()
{
    static const std::vector<std::string> columns = {
        "case_name",      "description",     "uplo",       "n",      "alpha_real",
        "alpha_imag",     "a_fill",          "lda",        "x_fill", "incx",
        "beta_real",      "beta_imag",       "y_fill",     "incy",   "expect_result",
        "mere_threshold", "mare_multiplier", "random_seed"};
    return columns;
}

inline void ValidateChemvCsvValues(const csv_map& row)
{
    for (const auto* key : {"n", "lda", "incx", "incy", "random_seed"}) {
        size_t consumed = 0;
        std::stoi(row.at(key), &consumed);
        if (consumed != row.at(key).size()) {
            throw std::runtime_error(std::string("Invalid integer in ") + key);
        }
    }
    constexpr const char* scalarKeys[] = {"alpha_real", "alpha_imag",     "beta_real",
                                          "beta_imag",  "mere_threshold", "mare_multiplier"};
    for (const auto* key : scalarKeys) {
        const auto& value = row.at(key);
        const bool nullable = std::string(key) == "alpha_real" || std::string(key) == "beta_real";
        if (nullable && (value == "null" || value == "NULLPTR")) {
            continue;
        }
        size_t consumed = 0;
        std::stof(value, &consumed);
        if (consumed != value.size()) {
            throw std::runtime_error(std::string("Invalid floating-point value in ") + key);
        }
    }
}

inline ChemvParam ParseChemvCsvRow(const std::string& line, size_t lineNumber)
{
    const auto values = CsvSplitLine(line);
    const auto& columns = ChemvCsvColumns();
    if (values.size() != columns.size() || std::count(line.begin(), line.end(), '"') % 2 != 0) {
        throw std::runtime_error("Chemv CSV malformed fields at line " + std::to_string(lineNumber));
    }
    csv_map row;
    for (size_t i = 0; i < columns.size(); ++i) {
        if (values[i].empty()) {
            throw std::runtime_error("Chemv CSV empty " + columns[i] + " at line " + std::to_string(lineNumber));
        }
        row.emplace(columns[i], values[i]);
    }
    try {
        ValidateChemvCsvValues(row);
        return ChemvParam(row);
    } catch (const std::exception& error) {
        throw std::runtime_error("Chemv CSV line " + std::to_string(lineNumber) + ": " + error.what());
    }
}

inline void ValidateChemvCaseNames(const std::vector<ChemvParam>& cases)
{
    size_t performance = 0;
    std::unordered_set<std::string> names;
    for (const auto& item : cases) {
        const auto& name = item.caseName;
        const bool validName = !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char value) {
            return std::isalnum(value) || value == '_';
        });
        if (!validName || !names.insert(name).second) {
            throw std::runtime_error("Chemv CSV invalid or duplicate case_name: " + name);
        }
        performance += name.rfind("TC_PF_", 0) == 0;
    }
    if (cases.size() != 1200U || performance != 200U) {
        throw std::runtime_error(
            "Chemv CSV requires 1200 cases (1000 accuracy + 200 TC_PF_); found " + std::to_string(cases.size()) +
            " total, " + std::to_string(performance) + " TC_PF_");
    }
}

inline std::vector<ChemvParam> GetChemvCasesFromCsv(const std::string& path)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open Chemv CSV: " + path);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Chemv CSV has no header: " + path);
    }
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        line.erase(0, 3);
    }
    if (CsvSplitLine(line) != ChemvCsvColumns()) {
        throw std::runtime_error("Chemv CSV header does not match the required schema: " + path);
    }
    std::vector<ChemvParam> cases;
    size_t lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        cases.emplace_back(ParseChemvCsvRow(line, lineNumber));
    }
    if (input.bad()) {
        throw std::runtime_error("Cannot read complete Chemv CSV: " + path);
    }
    ValidateChemvCaseNames(cases);
    return cases;
}
