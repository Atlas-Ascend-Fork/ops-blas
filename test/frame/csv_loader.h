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
    size_t lineNo = 0;
    size_t rowsWithEmptyCells = 0;
    while (std::getline(ifs, line)) {
        lineNo++;
        // skip empty and comment lines
        auto trimmed = line;
        size_t ns = trimmed.find_first_not_of(" \t\r\n");
        if (ns == std::string::npos)
            continue;
        if (trimmed[ns] == '#')
            continue;

        auto values = CsvSplitLine(line);
        if (values.size() < keys.size()) {
            // issue #607: 残缺行（字段数少于表头）不得静默跳过——用例无声消失导致覆盖丢失，
            // 编辑/合并 CSV 的缺列失误必须在 CI 中可见
            ADD_FAILURE() << "malformed csv row (fields=" << values.size() << ", header=" << keys.size()
                          << "), case dropped";
            continue;
        }
        if (values.size() > keys.size()) {
            // issue #693: 行尾多敲分隔符（字段数多于表头）与缺列同属编辑失误，多余字段
            // 不得被静默丢弃；报告行号与首个多余字段位置，行本身仍按表头前缀解析
            ADD_FAILURE() << "malformed csv row at line " << lineNo << " (fields=" << values.size()
                          << ", header=" << keys.size() << "), extra fields from index " << keys.size();
        }

        csv_map row;
        size_t emptyCells = 0;
        for (size_t i = 0; i < keys.size(); i++) {
            if (values[i].empty()) {
                // issue #693: 显式留空与漏填在下游同样由默认值接管，解析结果保持不变；
                // 此处计数并在收尾输出一次性汇总提示，便于作者核对留空是否符合意图
                emptyCells++;
                continue;
            }
            row[keys[i]] = values[i];
        }
        rowsWithEmptyCells += (emptyCells > 0 ? 1U : 0U);
        try {
            cases.emplace_back(row);
        } catch (const std::exception& e) {
            // issue #643: 参数构造期抛出的异常（如必填列缺失时 stoi("")）不得
            // 终止整个测试二进制——报告行号与原因后跳过该行，其余用例继续执行
            ADD_FAILURE() << "csv row at line " << lineNo << " failed to construct a test case: " << e.what()
                          << ", case dropped";
        }
    }
    if (rowsWithEmptyCells > 0) {
        std::cout << "[csv_loader] " << csvPath << ": " << rowsWithEmptyCells
                  << " data row(s) contain empty cells, treated as absent (defaults apply)\n";
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

// issue #643/#654: required numeric column parse failure with column name
inline int ParseIntColumn(const csv_map& row, const char* key, const std::string& def = std::string())
{
    const std::string raw = ReadMap(row, key, def);
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("column '") + key + "' is missing, empty or non-numeric: '" + raw + "'");
    }
}

inline float ParseFloatColumn(const csv_map& row, const char* key, const std::string& def = std::string())
{
    const std::string raw = ReadMap(row, key, def);
    try {
        return std::stof(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("column '") + key + "' is missing, empty or non-numeric: '" + raw + "'");
    }
}

// ── Safe numeric parsers（issue #660：解析失败输出可见告警后回退默认值，
//    不再静默吞掉——CSV 拼写错误/脏数据至少留下可检索的信号）──
inline void ReportParseFallback(const char* parser, const std::string& s, double def)
{
    std::cout << "[csv_loader] " << parser << " failed for '" << s << "', using default " << def << std::endl;
}

inline int parseInt(const std::string& s, int def = 0)
{
    try {
        return std::stoi(s);
    } catch (...) {
        ReportParseFallback("parseInt", s, def);
        return def;
    }
}
inline int64_t parseInt64(const std::string& s, int64_t def = 0)
{
    try {
        return std::stoll(s);
    } catch (...) {
        ReportParseFallback("parseInt64", s, def);
        return def;
    }
}
inline float parseFloat(const std::string& s, float def = 0.0f)
{
    try {
        return std::stof(s);
    } catch (...) {
        ReportParseFallback("parseFloat", s, def);
        return def;
    }
}
inline double parseDouble(const std::string& s, double def = 0.0)
{
    try {
        return std::stod(s);
    } catch (...) {
        ReportParseFallback("parseDouble", s, def);
        return def;
    }
}
inline uint32_t parseUint(const std::string& s, uint32_t def = 0)
{
    try {
        unsigned long val = std::stoul(s);
        if (val > UINT32_MAX) {
            ReportParseFallback("parseUint", s, def);
            return def;
        }
        return static_cast<uint32_t>(val);
    } catch (...) {
        ReportParseFallback("parseUint", s, def);
        return def;
    }
}

// ── Generic enum parser: map lookup → validated numeric fallback → default ──
template <typename EnumT>
inline EnumT parseEnum(const std::string& s, const std::unordered_map<std::string, EnumT>& table, EnumT defaultVal)
{
    if (s.empty())
        return defaultVal;
    auto it = table.find(s);
    if (it != table.end())
        return it->second;
    try {
        // Out-of-range numeric enums are a deliberate negative-injection channel
        // used by many existing CSV suites (e.g. trans=999 expecting
        // INVALID_ENUM), so the value is passed through as before (!723 review).
        // The notice only makes the fallback visible (issue #644).
        const int v = std::stoi(s);
        bool legal = false;
        for (const auto& kv : table) {
            if (static_cast<int>(kv.second) == v) {
                legal = true;
                break;
            }
        }
        if (!legal) {
            std::cout << "[csv_loader] numeric enum fallback '" << s << "' is not in the legal value set of "
                      << "this enum (negative-injection row?)\n";
        }
        return static_cast<EnumT>(v);
    } catch (...) {
        // issue #660: 非数值且未命中符号表的字符串（拼写错误）输出可见告警
        std::cout << "[csv_loader] unknown enum string '" << s << "', using default" << std::endl;
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
