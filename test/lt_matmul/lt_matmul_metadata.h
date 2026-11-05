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
#include <cstring>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <locale>
#include <sstream>
#include <utility>

#include "lt_matmul_copy.h"
#include "lt_matmul_param.h"

namespace LtMatmul22Test {

inline std::string JsonString(const std::string& value)
{
    static constexpr char HEX[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : value) {
        switch (c) {
            case '"':
                result += "\\\"";
                break;
            case '\\':
                result += "\\\\";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                if (c < 0x20U) {
                    result += "\\u00";
                    result += HEX[c >> 4U];
                    result += HEX[c & 15U];
                } else {
                    result += static_cast<char>(c);
                }
        }
    }
    return result + '"';
}

inline std::string NamedParameter(int value, std::initializer_list<std::pair<int, const char*>> names)
{
    for (const auto& item : names)
        if (value == item.first)
            return item.second;
    return std::to_string(value);
}

template <typename T>
inline std::string ScalarParameter(T value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
    return out.str();
}

template <typename Field>
inline void AddRuntimeParameters(const Param& p, const Field& field)
{
    field(
        "layout_order",
        NamedParameter(
            p.order, {{ACLBLASLT_ORDER_ROW, "ACLBLASLT_ORDER_ROW"}, {ACLBLASLT_ORDER_COL, "ACLBLASLT_ORDER_COL"}}));
    field("layout_batchCount", std::to_string(p.batch));
    field("layout_stride", std::to_string(p.stride));
    field("desc_nullptr", p.nullDescriptor);
    field("algo_mode", p.algoMode);
    field("workspaceSizeInBytes", p.workspaceBytes);
    field("preference_budget_bytes", std::to_string(p.preferenceBytes));
    field(
        "expect_result", NamedParameter(
                             p.expectResult, {{ACLBLAS_STATUS_SUCCESS, "ACLBLAS_STATUS_SUCCESS"},
                                              {ACLBLAS_STATUS_NOT_INITIALIZED, "ACLBLAS_STATUS_NOT_INITIALIZED"},
                                              {ACLBLAS_STATUS_ALLOC_FAILED, "ACLBLAS_STATUS_ALLOC_FAILED"},
                                              {ACLBLAS_STATUS_INVALID_VALUE, "ACLBLAS_STATUS_INVALID_VALUE"},
                                              {ACLBLAS_STATUS_MAPPING_ERROR, "ACLBLAS_STATUS_MAPPING_ERROR"},
                                              {ACLBLAS_STATUS_EXECUTION_FAILED, "ACLBLAS_STATUS_EXECUTION_FAILED"},
                                              {ACLBLAS_STATUS_INTERNAL_ERROR, "ACLBLAS_STATUS_INTERNAL_ERROR"},
                                              {ACLBLAS_STATUS_NOT_SUPPORTED, "ACLBLAS_STATUS_NOT_SUPPORTED"},
                                              {ACLBLAS_STATUS_ARCH_MISMATCH, "ACLBLAS_STATUS_ARCH_MISMATCH"},
                                              {ACLBLAS_STATUS_HANDLE_IS_NULLPTR, "ACLBLAS_STATUS_HANDLE_IS_NULLPTR"},
                                              {ACLBLAS_STATUS_INVALID_ENUM, "ACLBLAS_STATUS_INVALID_ENUM"},
                                              {ACLBLAS_STATUS_UNKNOWN, "ACLBLAS_STATUS_UNKNOWN"}}));
    field("random_seed", std::to_string(p.randomSeed));
    field("mere_threshold", ScalarParameter(p.mereThreshold));
    field("mare_multiplier", ScalarParameter(p.mareMultiplier));
    field("inplace", p.inplace ? "true" : "false");
    field("misalign_workspace", p.misalignWorkspace ? "true" : "false");
    field("forge_workspace_capacity", p.forgeWorkspaceCapacity ? "true" : "false");
    field("invalid_stream", p.invalidStream ? "true" : "false");
    field("record_input_fingerprints", p.recordInputFingerprints ? "true" : "false");
    field("distribution_metadata_json", p.distributionMetadata);
}

inline std::string OperationParameter(int value)
{
    return NamedParameter(
        value, {{ACLBLAS_OP_N, "ACLBLAS_OP_N"}, {ACLBLAS_OP_T, "ACLBLAS_OP_T"}, {ACLBLAS_OP_C, "ACLBLAS_OP_C"}});
}

inline std::string DtypeParameter(aclDataType value)
{
    return NamedParameter(
        value, {{ACL_FLOAT, "ACL_FLOAT"}, {ACL_DOUBLE, "ACL_DOUBLE"}, {ACL_COMPLEX64, "ACL_COMPLEX64"}});
}

inline std::string ParameterJson(const Param& p)
{
    std::ostringstream out;
    out << '{';
    bool first = true;
    const auto field = [&](const char* key, const std::string& value) {
        if (!first)
            out << ',';
        first = false;
        out << JsonString(key) << ':' << JsonString(value);
    };
    field("case_name", p.caseName);
    field("description", p.description);
    field("transA", OperationParameter(p.transA));
    field("transB", OperationParameter(p.transB));
    field("computeType", NamedParameter(p.computeType, {{ACLBLAS_COMPUTE_32F, "ACLBLAS_COMPUTE_32F"}}));
    field("scaleType", DtypeParameter(p.scaleType));
    field(
        "pointerMode", NamedParameter(
                           p.pointerMode, {{ACLBLASLT_POINTER_MODE_HOST, "ACLBLASLT_POINTER_MODE_HOST"},
                                           {ACLBLASLT_POINTER_MODE_DEVICE, "ACLBLASLT_POINTER_MODE_DEVICE"}}));
    field("m", std::to_string(p.m));
    field("n", std::to_string(p.n));
    field("k", std::to_string(p.k));
    field("alpha", p.alphaNull ? "null" : ScalarParameter(p.alpha));
    field("beta", p.betaNull ? "null" : ScalarParameter(p.beta));
    uint32_t alphaBits = 0, betaBits = 0;
    CopyBytes(&alphaBits, sizeof(alphaBits), &p.alpha, sizeof(alphaBits));
    CopyBytes(&betaBits, sizeof(betaBits), &p.beta, sizeof(betaBits));
    field("alpha_bits_uint32", std::to_string(alphaBits));
    field("beta_bits_uint32", std::to_string(betaBits));
    constexpr const char* DTYPE_NAMES[] = {"dtypeA", "dtypeB", "dtypeC", "dtypeD"};
    constexpr const char* LD_NAMES[] = {"lda", "ldb", "ldc", "ldd"};
    constexpr const char* FILL_NAMES[] = {"a_fill", "b_fill", "c_fill", "d_fill"};
    for (size_t i = 0; i < 4; ++i) {
        field(DTYPE_NAMES[i], DtypeParameter(p.dtype[i]));
        field(LD_NAMES[i], std::to_string(p.ld[i]));
        field(FILL_NAMES[i], p.fill[i]);
    }
    AddRuntimeParameters(p, field);
    out << '}';
    return out.str();
}

inline std::string CurrentGtestName()
{
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    return test == nullptr ? "" : std::string(test->test_suite_name()) + '.' + test->name();
}

inline void RecordParameterMetadata(const Param& p, const char* observationKind)
{
    const char* path = std::getenv("LT_MATMUL_PARAMETERS");
    if (path == nullptr)
        return;
    std::ofstream output(path, std::ios::app);
    if (!output)
        throw std::runtime_error("Cannot append actual parameter metadata");
    output << "{\"schema_version\":\"ltmatmul-case-parameters-v1\",\"case_id\":" << JsonString(p.caseName)
           << ",\"gtest_name\":" << JsonString(CurrentGtestName())
           << ",\"observation_kind\":" << JsonString(observationKind)
           << ",\"parameter_scope\":\"Param object after factory mutations; runtime pointer overrides remain in GTest "
              "assertions\""
           << ",\"parameters\":" << ParameterJson(p) << "}\n";
    if (!output)
        throw std::runtime_error("Cannot write actual parameter metadata");
}

} // namespace LtMatmul22Test
