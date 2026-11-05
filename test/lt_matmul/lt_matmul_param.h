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
#include <array>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cann_ops_blasLt.h"
#include "csv_loader.h"

namespace LtMatmul22Test {

inline std::string DecodeMetadata(const std::string& encoded)
{
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string decoded;
    uint32_t value = 0;
    int bits = -8;
    for (char character : encoded) {
        if (character == '=')
            break;
        const auto digit = alphabet.find(character);
        if (digit == std::string::npos)
            throw std::runtime_error("Invalid supplemental metadata base64");
        value = (value << 6U) | static_cast<uint32_t>(digit);
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<char>((value >> bits) & 255U));
            bits -= 8;
        }
    }
    return decoded;
}

inline aclDataType Dtype(const std::string& value)
{
    if (value == "ACL_FLOAT")
        return ACL_FLOAT;
    if (value == "ACL_DOUBLE")
        return ACL_DOUBLE;
    if (value == "ACL_COMPLEX64")
        return ACL_COMPLEX64;
    return static_cast<aclDataType>(std::stoi(value));
}

struct Param : BlasTestParamBase {
    int m, n, k;
    int transA, transB, computeType, pointerMode;
    aclDataType scaleType;
    std::array<aclDataType, 4> dtype;
    std::array<int64_t, 4> ld;
    std::array<std::string, 4> fill;
    float alpha, beta;
    bool alphaNull, betaNull, inplace;
    int order, batch;
    int64_t stride;
    std::string nullDescriptor, algoMode, workspaceBytes;
    size_t preferenceBytes = 128U * 1024U * 1024U;
    bool misalignWorkspace = false;
    bool forgeWorkspaceCapacity = false;
    bool invalidStream = false;
    bool recordInputFingerprints = false;
    std::string distributionMetadata;

    explicit Param(const csv_map& row) : BlasTestParamBase(row)
    {
        auto read = [&row](const char* name, const char* fallback) { return ReadMap(row, name, fallback); };
        m = std::stoi(read("m", "16"));
        n = std::stoi(read("n", "16"));
        k = std::stoi(read("k", "16"));
        transA = static_cast<int>(parseOpTrans(read("transA", "ACLBLAS_OP_N")));
        transB = static_cast<int>(parseOpTrans(read("transB", "ACLBLAS_OP_N")));
        const auto compute = read("computeType", "ACLBLAS_COMPUTE_32F");
        computeType = compute == "ACLBLAS_COMPUTE_32F" ? ACLBLAS_COMPUTE_32F : std::stoi(compute);
        scaleType = Dtype(read("scaleType", "ACL_FLOAT"));
        const auto mode = read("pointerMode", "ACLBLASLT_POINTER_MODE_HOST");
        pointerMode = mode == "ACLBLASLT_POINTER_MODE_HOST"   ? ACLBLASLT_POINTER_MODE_HOST :
                      mode == "ACLBLASLT_POINTER_MODE_DEVICE" ? ACLBLASLT_POINTER_MODE_DEVICE :
                                                                std::stoi(mode);
        dtype = {
            Dtype(read("dtypeA", "ACL_FLOAT")), Dtype(read("dtypeB", "ACL_FLOAT")), Dtype(read("dtypeC", "ACL_FLOAT")),
            Dtype(read("dtypeD", "ACL_FLOAT"))};
        ld = {
            std::stoll(read("lda", "0")), std::stoll(read("ldb", "0")), std::stoll(read("ldc", "0")),
            std::stoll(read("ldd", "0"))};
        fill = {
            read("a_fill", "RANDOM_NORM_5_5"), read("b_fill", "RANDOM_NORM_5_5"), read("c_fill", "RANDOM_NORM_5_5"),
            read("d_fill", "VALUE_NORM_0")};
        const auto a = read("alpha", "1"), b = read("beta", "0");
        alphaNull = a == "null";
        betaNull = b == "null";
        alpha = alphaNull ? 0.0f : std::stof(a);
        beta = betaNull ? 0.0f : std::stof(b);
        inplace = fill[2] == "INPLACE";
        const auto layout = read("layout_order", "ACLBLASLT_ORDER_ROW");
        order = layout == "ACLBLASLT_ORDER_ROW" ? ACLBLASLT_ORDER_ROW :
                layout == "ACLBLASLT_ORDER_COL" ? ACLBLASLT_ORDER_COL :
                                                  std::stoi(layout);
        batch = std::stoi(read("layout_batchCount", "1"));
        stride = std::stoll(read("layout_stride", "0"));
        nullDescriptor = read("desc_nullptr", "none");
        algoMode = read("algo_mode", "heuristic");
        workspaceBytes = read("workspaceSizeInBytes", "auto");
        recordInputFingerprints = read("record_input_fingerprints", "0") == "1";
        distributionMetadata = DecodeMetadata(read("distribution_metadata_b64", ""));
        if (caseName.empty())
            throw std::runtime_error("CSV row lacks case_name");
        if (algoMode != "heuristic" && algoMode != "nullptr" && algoMode != "explicit_h2" && algoMode != "explicit_h5")
            throw std::runtime_error("Unknown algo_mode");
    }

    std::array<int, 4> Rows() const { return {transA == ACLBLAS_OP_N ? m : k, transB == ACLBLAS_OP_N ? k : n, m, m}; }
    std::array<int, 4> Cols() const { return {transA == ACLBLAS_OP_N ? k : m, transB == ACLBLAS_OP_N ? n : k, n, n}; }
    int StorageLd(size_t i) const { return static_cast<int>(std::max<int64_t>(std::max(1, Cols()[i]), ld[i])); }
};

inline std::vector<Param> TaskCases(const char* source)
{
    const char* custom = std::getenv("LT_MATMUL_CSV");
    auto cases = GetCasesFromCsv<Param>(custom == nullptr ? ReplaceFileExtension2Csv(source) : custom);
    if (cases.size() != 1200U)
        throw std::runtime_error("Expected exactly 1200 task CSV cases");
    const auto performanceCount =
        std::count_if(cases.begin(), cases.end(), [](const Param& p) { return p.caseName.rfind("TC_PF_", 0) == 0; });
    if (performanceCount != 200)
        throw std::runtime_error("Expected exactly 200 task performance cases");
    std::set<std::string> names;
    for (const auto& item : cases) {
        if (!names.insert(item.caseName).second)
            throw std::runtime_error("Duplicate CSV case: " + item.caseName);
    }
    return cases;
}

inline std::vector<Param> GapCases(const char* source)
{
    const char* custom = std::getenv("LT_MATMUL_GAP_CSV");
    const std::string path(source);
    const auto separator = path.find_last_of('/');
    const std::string defaultPath = path.substr(0, separator + 1) + "blasLtMatmul_gap_cases.csv";
    auto cases = GetCasesFromCsv<Param>(custom == nullptr ? defaultPath : custom);
    if (cases.size() != 36)
        throw std::runtime_error("Expected exactly 36 independent coverage-gap cases");
    std::set<std::string> names;
    for (const auto& item : cases) {
        if (!names.insert(item.caseName).second)
            throw std::runtime_error("Duplicate coverage-gap case ID");
    }
    return cases;
}

inline Param Supplemental(const std::string& name, int m = 65, int n = 33, int k = 513)
{
    return Param(
        {{"case_name", name},
         {"description", "supplemental_zero_workspace_inplace"},
         {"m", std::to_string(m)},
         {"n", std::to_string(n)},
         {"k", std::to_string(k)},
         {"alpha", "0.5"},
         {"beta", "-0.5"},
         {"c_fill", "INPLACE"},
         {"d_fill", "RANDOM_NORM_5_5"},
         {"algo_mode", "nullptr"},
         {"workspaceSizeInBytes", "0"},
         {"random_seed", "20261028"}});
}

inline std::vector<Param> SupplementalCases()
{
    std::vector<Param> cases;
    for (int n : {1, 3, 15, 16, 17, 33, 129}) {
        for (int ta : {ACLBLAS_OP_N, ACLBLAS_OP_T}) {
            for (int tb : {ACLBLAS_OP_N, ACLBLAS_OP_T}) {
                auto p = Supplemental(
                    "H0_n" + std::to_string(n) + "_" + std::to_string(ta) + std::to_string(tb), 65, n, 513);
                p.transA = ta;
                p.transB = tb;
                p.ld = {p.Cols()[0] + 3, p.Cols()[1] + 5, n + 7, n + 7};
                cases.push_back(p);
            }
        }
    }
    for (bool inplace : {false, true}) {
        for (float beta : {0.0f, 1.0f, -0.5f}) {
            auto p =
                Supplemental("AlphaZero_" + std::to_string(inplace) + "_" + std::to_string(cases.size()), 7, 17, 513);
            p.alpha = 0.0f;
            p.beta = beta;
            p.fill[0] = p.fill[1] = "NULLPTR";
            if (!inplace) {
                p.inplace = false;
                p.fill[2] = "RANDOM_NORM_5_5";
                p.ld[2] = 25;
                p.ld[3] = 29;
            }
            cases.push_back(p);
        }
    }
    auto padding = Supplemental("H1_independent_C_D_pitch", 67, 33, 513);
    padding.algoMode = "heuristic";
    padding.workspaceBytes = "auto";
    padding.inplace = false;
    padding.fill[2] = "RANDOM_NORM_5_5";
    padding.ld = {521, 40, 45, 49};
    cases.push_back(padding);
    const int paddedWidths[] = {17, 31, 33, 65, 129};
    const int paddedRows[] = {1, 17, 65};
    for (size_t ni = 0; ni < 5; ++ni) {
        for (int combination = 0; combination < 4; ++combination) {
            const int n = paddedWidths[ni];
            const int m = paddedRows[(ni + combination) % 3];
            const int k = combination % 2 == 0 ? 17 : 513;
            const std::string trans = std::string(combination & 1 ? "T" : "N") + (combination & 2 ? "T" : "N");
            auto p = Supplemental(
                "H1Padded_n" + std::to_string(n) + "_" + trans + "_m" + std::to_string(m) + "_k" + std::to_string(k), m,
                n, k);
            p.description = "H1_padded_product_precise_output_guards";
            p.transA = combination & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
            p.transB = combination & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
            p.algoMode = "heuristic";
            p.workspaceBytes = "auto";
            p.alpha = combination < 2 ? 1.0f : 0.5f;
            p.beta = combination < 2 ? 0.0f : -0.5f;
            p.inplace = combination == 3;
            p.fill[2] = p.inplace ? "INPLACE" : "RANDOM_NORM_5_5";
            p.ld = {p.Cols()[0] + 5, p.Cols()[1] + 3, n + 7, n + 11};
            if (p.inplace)
                p.ld[2] = p.ld[3];
            cases.push_back(p);
        }
    }
    auto exact = Supplemental("H1CapacityExact_n17", 17, 17, 17);
    exact.algoMode = "heuristic";
    exact.workspaceBytes = std::to_string(4U * 17U * 32U);
    cases.push_back(exact);
    auto oneShort = exact;
    oneShort.caseName = "H1CapacityMinusOneForged_n17";
    oneShort.workspaceBytes = std::to_string(4U * 17U * 32U - 1U);
    oneShort.forgeWorkspaceCapacity = true;
    oneShort.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
    cases.push_back(oneShort);
    auto misaligned = Supplemental("WorkspaceMisaligned", 64, 64, 64);
    misaligned.algoMode = "heuristic";
    misaligned.workspaceBytes = "auto";
    misaligned.misalignWorkspace = true;
    misaligned.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
    cases.push_back(misaligned);
    auto forged = misaligned;
    forged.caseName = "WorkspaceForgedCapacity";
    forged.misalignWorkspace = false;
    forged.forgeWorkspaceCapacity = true;
    forged.workspaceBytes = "1";
    cases.push_back(forged);
    auto deviceMode = Supplemental("RejectDevicePointerMode", 16, 16, 16);
    deviceMode.pointerMode = ACLBLASLT_POINTER_MODE_DEVICE;
    deviceMode.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
    cases.push_back(deviceMode);
    auto budget = Supplemental("HeuristicZeroBudget", 65, 129, 513);
    budget.algoMode = "heuristic";
    budget.workspaceBytes = "auto";
    budget.preferenceBytes = 0;
    cases.push_back(budget);
    for (int n : {3, 17, 33}) {
        auto p = Supplemental("Gaussian_n" + std::to_string(n), 17, n, 65);
        p.fill[0] = p.fill[1] = p.fill[3] = "GAUSS_5_2";
        cases.push_back(p);
    }
    return cases;
}

} // namespace LtMatmul22Test
