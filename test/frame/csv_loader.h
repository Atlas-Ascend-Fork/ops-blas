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

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "fill.h"
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ── csv_map: each CSV row is parsed into a string→string map ──
using csv_map = std::unordered_map<std::string, std::string>;

// ── ReadMap: safe lookup with default ──
template <typename Map>
inline typename Map::mapped_type ReadMap(
    const Map& m, const typename Map::key_type& key,
    const typename Map::mapped_type& defaultValue = typename Map::mapped_type{})
{
    auto it = m.find(key);
    return it != m.end() ? it->second : defaultValue;
}

// ── CSV line splitting ──
inline std::vector<std::string> CsvSplitLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            // trim
            size_t s = field.find_first_not_of(" \t\r\n");
            size_t e = field.find_last_not_of(" \t\r\n");
            fields.push_back((s == std::string::npos) ? "" : field.substr(s, e - s + 1));
            field.clear();
        } else {
            field += c;
        }
    }
    size_t s = field.find_first_not_of(" \t\r\n");
    size_t e = field.find_last_not_of(" \t\r\n");
    fields.push_back((s == std::string::npos) ? "" : field.substr(s, e - s + 1));
    return fields;
}

// ── GetCasesFromCsv<T>: parse CSV, construct one T(csv_map) per data row ──
template <typename T>
inline std::vector<T> GetCasesFromCsv(const std::string& csvPath)
{
    std::ifstream ifs(csvPath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + csvPath);
    }

    std::string headerLine;
    bool hasHeader = false;
    while (std::getline(ifs, headerLine)) {
        if (headerLine.compare(0, 3, "\xEF\xBB\xBF") == 0) {
            headerLine.erase(0, 3);
        }
        size_t first = headerLine.find_first_not_of(" \t\r\n");
        if (first != std::string::npos && headerLine[first] != '#') {
            hasHeader = true;
            break;
        }
    }
    if (!hasHeader) {
        throw std::runtime_error("CSV file has no header: " + csvPath);
    }
    auto keys = CsvSplitLine(headerLine);

    std::vector<T> cases;
    std::string line;
    while (std::getline(ifs, line)) {
        // skip empty and comment lines
        auto trimmed = line;
        size_t ns = trimmed.find_first_not_of(" \t\r\n");
        if (ns == std::string::npos)
            continue;
        if (trimmed[ns] == '#')
            continue;

        auto values = CsvSplitLine(line);
        if (values.size() < keys.size())
            continue;

        csv_map row;
        for (size_t i = 0; i < keys.size(); i++) {
            if (!values[i].empty()) {
                row[keys[i]] = values[i];
            }
        }
        cases.emplace_back(row);
    }
    return cases;
}

// ── ReplaceFileExtension2Csv: derive CSV path from __FILE__ ──
inline std::string ReplaceFileExtension2Csv(const std::string& filePath)
{
    std::string p = filePath;
    size_t dot = p.rfind('.');
    if (dot != std::string::npos)
        p = p.substr(0, dot);
    return p + ".csv";
}

// ── Safe numeric parsers ──
inline int parseInt(const std::string& s, int def = 0)
{
    try {
        return std::stoi(s);
    } catch (...) {
        return def;
    }
}
inline int64_t parseInt64(const std::string& s, int64_t def = 0)
{
    try {
        return std::stoll(s);
    } catch (...) {
        return def;
    }
}
inline float parseFloat(const std::string& s, float def = 0.0f)
{
    try {
        return std::stof(s);
    } catch (...) {
        return def;
    }
}
inline double parseDouble(const std::string& s, double def = 0.0)
{
    try {
        return std::stod(s);
    } catch (...) {
        return def;
    }
}
inline uint32_t parseUint(const std::string& s, uint32_t def = 0)
{
    try {
        unsigned long val = std::stoul(s);
        if (val > UINT32_MAX)
            return def;
        return static_cast<uint32_t>(val);
    } catch (...) {
        return def;
    }
}

// ── Generic enum parser: map lookup → numeric fallback → default ──
template <typename EnumT>
inline EnumT parseEnum(const std::string& s, const std::unordered_map<std::string, EnumT>& table, EnumT defaultVal)
{
    if (s.empty())
        return defaultVal;
    auto it = table.find(s);
    if (it != table.end())
        return it->second;
    try {
        return static_cast<EnumT>(std::stoi(s));
    } catch (...) {
        return defaultVal;
    }
}

// aclblasStatus_t parser (must be defined before BlasTestParamBase)
inline aclblasStatus_t parseStatus(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasStatus_t> t = {
        {"ACLBLAS_STATUS_SUCCESS", ACLBLAS_STATUS_SUCCESS},
        {"SUCCESS", ACLBLAS_STATUS_SUCCESS},
        {"ACLBLAS_STATUS_NOT_INITIALIZED", ACLBLAS_STATUS_NOT_INITIALIZED},
        {"NOT_INITIALIZED", ACLBLAS_STATUS_NOT_INITIALIZED},
        {"ACLBLAS_STATUS_ALLOC_FAILED", ACLBLAS_STATUS_ALLOC_FAILED},
        {"ALLOC_FAILED", ACLBLAS_STATUS_ALLOC_FAILED},
        {"ACLBLAS_STATUS_INVALID_VALUE", ACLBLAS_STATUS_INVALID_VALUE},
        {"INVALID_VALUE", ACLBLAS_STATUS_INVALID_VALUE},
        {"ACLBLAS_STATUS_MAPPING_ERROR", ACLBLAS_STATUS_MAPPING_ERROR},
        {"MAPPING_ERROR", ACLBLAS_STATUS_MAPPING_ERROR},
        {"ACLBLAS_STATUS_EXECUTION_FAILED", ACLBLAS_STATUS_EXECUTION_FAILED},
        {"EXECUTION_FAILED", ACLBLAS_STATUS_EXECUTION_FAILED},
        {"ACLBLAS_STATUS_INTERNAL_ERROR", ACLBLAS_STATUS_INTERNAL_ERROR},
        {"INTERNAL_ERROR", ACLBLAS_STATUS_INTERNAL_ERROR},
        {"ACLBLAS_STATUS_NOT_SUPPORTED", ACLBLAS_STATUS_NOT_SUPPORTED},
        {"NOT_SUPPORTED", ACLBLAS_STATUS_NOT_SUPPORTED},
        {"ACLBLAS_STATUS_ARCH_MISMATCH", ACLBLAS_STATUS_ARCH_MISMATCH},
        {"ARCH_MISMATCH", ACLBLAS_STATUS_ARCH_MISMATCH},
        {"ACLBLAS_STATUS_HANDLE_IS_NULLPTR", ACLBLAS_STATUS_HANDLE_IS_NULLPTR},
        {"HANDLE_IS_NULLPTR", ACLBLAS_STATUS_HANDLE_IS_NULLPTR},
        {"ACLBLAS_STATUS_INVALID_ENUM", ACLBLAS_STATUS_INVALID_ENUM},
        {"INVALID_ENUM", ACLBLAS_STATUS_INVALID_ENUM},
        {"ACLBLAS_STATUS_UNKNOWN", ACLBLAS_STATUS_UNKNOWN},
        {"UNKNOWN", ACLBLAS_STATUS_UNKNOWN}};
    return parseEnum(s, t, ACLBLAS_STATUS_INTERNAL_ERROR);
}

// ── BlasTestParamBase: common fields loaded from every CSV row ──
struct BlasTestParamBase {
    std::string caseName;
    std::string description;
    aclblasStatus_t expectResult = ACLBLAS_STATUS_SUCCESS;
    double mereThreshold = 0.0;
    double mareMultiplier = 0.0;
    uint32_t randomSeed = 0;

    BlasTestParamBase(const csv_map& m)
    {
        caseName = ReadMap(m, "case_name");
        description = ReadMap(m, "description");
        expectResult = parseStatus(ReadMap(m, "expect_result", "SUCCESS"));
        mereThreshold = parseDouble(ReadMap(m, "mere_threshold", "0"));
        mareMultiplier = parseDouble(ReadMap(m, "mare_multiplier", "0"));
        randomSeed = parseUint(ReadMap(m, "random_seed", "0"));
    }
};

// ── operator<< for GTest parameter name printing ──
inline std::ostream& operator<<(std::ostream& os, const BlasTestParamBase& p) { return os << p.caseName; }

// ── PrintCaseInfoString: GTest name generator ──
template <typename T>
inline std::string PrintCaseInfoString(const ::testing::TestParamInfo<T>& info)
{
    return info.param.caseName;
}

// ── Common BLAS enum parsers (values match cann_ops_blas_common.h) ──

// aclblasFillMode_t  (121=UPPER, 122=LOWER)
inline aclblasFillMode_t parseFillMode(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasFillMode_t> t = {
        {"ACLBLAS_UPPER", ACLBLAS_UPPER},
        {"UPPER", ACLBLAS_UPPER},
        {"ACLBLAS_LOWER", ACLBLAS_LOWER},
        {"LOWER", ACLBLAS_LOWER},
        {"INVALID", static_cast<aclblasFillMode_t>(0xFF)}};
    return parseEnum(s, t, static_cast<aclblasFillMode_t>(0xFF));
}

// aclblasDiagType_t  (131=NON_UNIT, 132=UNIT)
inline aclblasDiagType_t parseDiagType(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasDiagType_t> t = {
        {"ACLBLAS_NON_UNIT", ACLBLAS_NON_UNIT},
        {"NON_UNIT", ACLBLAS_NON_UNIT},
        {"ACLBLAS_UNIT", ACLBLAS_UNIT},
        {"UNIT", ACLBLAS_UNIT},
        {"INVALID", static_cast<aclblasDiagType_t>(0xFF)}};
    return parseEnum(s, t, ACLBLAS_NON_UNIT);
}

// aclblasSideMode_t  (141=LEFT, 142=RIGHT)
inline aclblasSideMode_t parseSideMode(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasSideMode_t> t = {
        {"ACLBLAS_SIDE_LEFT", ACLBLAS_SIDE_LEFT},
        {"LEFT", ACLBLAS_SIDE_LEFT},
        {"ACLBLAS_SIDE_RIGHT", ACLBLAS_SIDE_RIGHT},
        {"RIGHT", ACLBLAS_SIDE_RIGHT}};
    return parseEnum(s, t, ACLBLAS_SIDE_LEFT);
}

// aclblasOperation_t  (111=N, 112=T, 113=C)
inline aclblasOperation_t parseOpTrans(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasOperation_t> t = {
        {"ACLBLAS_OP_N", ACLBLAS_OP_N},
        {"N", ACLBLAS_OP_N},
        {"ACLBLAS_OP_T", ACLBLAS_OP_T},
        {"T", ACLBLAS_OP_T},
        {"ACLBLAS_OP_C", ACLBLAS_OP_C},
        {"C", ACLBLAS_OP_C},
        {"ACLBLAS_OP_H", ACLBLAS_OP_C},
        {"H", ACLBLAS_OP_C},
        {"INVALID", static_cast<aclblasOperation_t>(0xFF)}};
    return parseEnum(s, t, ACLBLAS_OP_N);
}

// aclblasComputeType_t  (0..10)
inline aclblasComputeType_t parseComputeType(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasComputeType_t> t = {
        {"ACLBLAS_COMPUTE_16F", ACLBLAS_COMPUTE_16F},
        {"COMPUTE_16F", ACLBLAS_COMPUTE_16F},
        {"ACLBLAS_COMPUTE_16F_PEDANTIC", ACLBLAS_COMPUTE_16F_PEDANTIC},
        {"COMPUTE_16F_PEDANTIC", ACLBLAS_COMPUTE_16F_PEDANTIC},
        {"ACLBLAS_COMPUTE_32F", ACLBLAS_COMPUTE_32F},
        {"COMPUTE_32F", ACLBLAS_COMPUTE_32F},
        {"ACLBLAS_COMPUTE_32F_PEDANTIC", ACLBLAS_COMPUTE_32F_PEDANTIC},
        {"COMPUTE_32F_PEDANTIC", ACLBLAS_COMPUTE_32F_PEDANTIC},
        {"ACLBLAS_COMPUTE_32F_FAST_16F", ACLBLAS_COMPUTE_32F_FAST_16F},
        {"COMPUTE_32F_FAST_16F", ACLBLAS_COMPUTE_32F_FAST_16F},
        {"ACLBLAS_COMPUTE_32F_FAST_16BF", ACLBLAS_COMPUTE_32F_FAST_16BF},
        {"COMPUTE_32F_FAST_16BF", ACLBLAS_COMPUTE_32F_FAST_16BF},
        {"ACLBLAS_COMPUTE_32F_FAST_TF32", ACLBLAS_COMPUTE_32F_FAST_TF32},
        {"COMPUTE_32F_FAST_TF32", ACLBLAS_COMPUTE_32F_FAST_TF32},
        {"ACLBLAS_COMPUTE_64F", ACLBLAS_COMPUTE_64F},
        {"COMPUTE_64F", ACLBLAS_COMPUTE_64F},
        {"ACLBLAS_COMPUTE_64F_PEDANTIC", ACLBLAS_COMPUTE_64F_PEDANTIC},
        {"COMPUTE_64F_PEDANTIC", ACLBLAS_COMPUTE_64F_PEDANTIC},
        {"ACLBLAS_COMPUTE_32I", ACLBLAS_COMPUTE_32I},
        {"COMPUTE_32I", ACLBLAS_COMPUTE_32I},
        {"ACLBLAS_COMPUTE_32I_PEDANTIC", ACLBLAS_COMPUTE_32I_PEDANTIC},
        {"COMPUTE_32I_PEDANTIC", ACLBLAS_COMPUTE_32I_PEDANTIC}};
    return parseEnum(s, t, ACLBLAS_COMPUTE_32F);
}

// ── aclDataType string → int32_t parser ──
//   ACL_FLOAT=0, ACL_FLOAT16=1, ACL_BF16=27, ACL_INT8=8 etc.
//   Stored as int32_t so BlasFillMode / param structs can switch on it.
using T_DtypeLocal = int32_t;

inline T_DtypeLocal parseDataType(const std::string& s)
{
    static const std::unordered_map<std::string, T_DtypeLocal> t = {
        {"ACL_FLOAT16", static_cast<int32_t>(ACL_FLOAT16)}, {"FP16", static_cast<int32_t>(ACL_FLOAT16)},
        {"ACL_FLOAT", static_cast<int32_t>(ACL_FLOAT)},     {"FP32", static_cast<int32_t>(ACL_FLOAT)},
        {"ACL_BF16", static_cast<int32_t>(ACL_BF16)},       {"BF16", static_cast<int32_t>(ACL_BF16)},
        {"ACL_INT8", static_cast<int32_t>(ACL_INT8)},       {"INT8", static_cast<int32_t>(ACL_INT8)},
    };
    return parseEnum<T_DtypeLocal>(s, t, static_cast<int32_t>(ACL_FLOAT));
}

// ── Array parsers: semicolon-separated values for grouped batched operators ──

inline std::vector<std::string> splitBySemicolon(const std::string& s)
{
    std::vector<std::string> parts;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ';'))
        if (!token.empty())
            parts.push_back(token);
    return parts;
}

inline std::vector<int> parseIntArray(const std::string& s)
{
    auto parts = splitBySemicolon(s);
    std::vector<int> result;
    for (const auto& p : parts)
        result.push_back(parseInt(p));
    return result;
}

inline std::vector<float> parseFloatArray(const std::string& s)
{
    auto parts = splitBySemicolon(s);
    std::vector<float> result;
    for (const auto& p : parts)
        result.push_back(parseFloat(p));
    return result;
}

inline std::vector<aclblasOperation_t> parseTransArray(const std::string& s)
{
    auto parts = splitBySemicolon(s);
    std::vector<aclblasOperation_t> result;
    for (const auto& p : parts)
        result.push_back(parseOpTrans(p));
    return result;
}
