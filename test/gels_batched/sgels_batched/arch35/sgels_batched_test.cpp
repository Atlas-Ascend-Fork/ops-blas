/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sgels_batched_param.h"
#include "sgels_batched_golden.h"
#include "sgels_batched_npu_wrapper.h"

class GelsBatchedArch35Test : public BlasTest<GelsBatchedParam> {};

TEST_F(GelsBatchedArch35Test, NullHandle)
{
    aclblasStatus_t ret =
        aclblasSgelsBatched_npu_error(nullptr, ACLBLAS_OP_N, 8, 4, 1, nullptr, 8, nullptr, 8, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    GelsBatched, GelsBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<GelsBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GelsBatchedParam>);

static std::vector<float> generateMatrix(int rows, int cols, int ld, BlasDataFill fill, uint32_t seed)
{
    const size_t size = static_cast<size_t>(ld) * cols;
    std::vector<float> data(size, 0.0f);

    if (fill == BlasDataFill::ZEROS)
        return data;

    if (fill == BlasDataFill::ONES) {
        std::fill(data.begin(), data.end(), 1.0f);
        return data;
    }

    std::mt19937 rng(seed ? seed : 42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++)
            data[i + j * ld] = dist(rng);
    return data;
}

static std::vector<float> generateIdentity(int m, int n, int ld)
{
    const size_t size = static_cast<size_t>(ld) * n;
    std::vector<float> data(size, 0.0f);
    int minmn = std::min(m, n);
    for (int i = 0; i < minmn; i++)
        data[i + i * ld] = 1.0f;
    return data;
}

static void runErrorPath(const GelsBatchedParam& p, aclblasHandle_t handle)
{
    const bool isHandleNull = (p.description.find("handle_null") != std::string::npos);
    const bool isCarrayNull = (p.description.find("carray_null") != std::string::npos);
    const bool isDevinfoNull = (p.description.find("devinfo_null") != std::string::npos);

    aclblasHandle_t h = isHandleNull ? nullptr : handle;
    int devInfo = 0;
    int* devInfoPtr = isDevinfoNull ? nullptr : &devInfo;

    float* dummyPtrs[1] = {nullptr};
    float* const* aPtr = (isCarrayNull || isDevinfoNull) ? dummyPtrs : nullptr;
    float* const* cPtr = isCarrayNull ? nullptr : (isDevinfoNull ? dummyPtrs : nullptr);

    aclblasStatus_t ret =
        aclblasSgelsBatched_npu_error(h, p.trans, p.m, p.n, p.nrhs, aPtr, p.lda, cPtr, p.ldc, devInfoPtr, p.batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

static void applyKnownSolution(
    int m, int n, int nrhs, int lda, int ldc, uint32_t seed, std::vector<float>& Abatch, std::vector<float>& Cbatch)
{
    std::mt19937 rng(seed + 2000);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> X(static_cast<size_t>(n) * nrhs);
    for (int j = 0; j < nrhs; j++)
        for (int i = 0; i < n; i++)
            X[i + j * n] = dist(rng);
    for (int j = 0; j < nrhs; j++)
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += Abatch[i + k * lda] * X[k + j * n];
            Cbatch[i + j * ldc] = sum;
        }
}

static void generateBatchData(
    const GelsBatchedParam& p, int bs, int cInputRows, std::vector<std::vector<float>>& Ahost,
    std::vector<std::vector<float>>& Chost)
{
    const bool isIdentity = (p.description.find("identity") != std::string::npos);
    const bool isRankDeficient = (p.description.find("rank_deficient") != std::string::npos);
    const bool isKnownSolution = (p.description.find("known_solution") != std::string::npos);

    for (int b = 0; b < bs; b++) {
        uint32_t seed = p.randomSeed ? (p.randomSeed * 31 + b + 1) : (42u * 31 + b + 1);

        Ahost[b] = isIdentity ? generateIdentity(p.m, p.n, p.lda) : generateMatrix(p.m, p.n, p.lda, p.aFill, seed);
        Chost[b] = generateMatrix(cInputRows, p.nrhs, p.ldc, p.cFill, seed + 1000);

        if (isRankDeficient && p.n > 1) {
            for (int i = 0; i < p.m; i++)
                Ahost[b][i + 1 * p.lda] = 0.0f;
        }

        if (isKnownSolution && p.m > 0 && p.n > 0 && p.nrhs > 0) {
            applyKnownSolution(p.m, p.n, p.nrhs, p.lda, p.ldc, seed, Ahost[b], Chost[b]);
        }
    }
}

static int computeGolden(
    aclblasHandle_t handle, const GelsBatchedParam& p, int bs, int cInputRows, int cGoldenCols,
    const std::vector<std::vector<float>>& Ahost, const std::vector<std::vector<float>>& Chost,
    std::vector<std::vector<float>>& Cgolden)
{
    std::vector<std::vector<float>> Agolden(bs);
    Cgolden.resize(bs);
    for (int b = 0; b < bs; b++) {
        Agolden[b] = Ahost[b];
        Cgolden[b].resize(static_cast<size_t>(p.ldc) * cGoldenCols, 0.0f);
        for (int j = 0; j < p.nrhs; j++)
            for (int i = 0; i < cInputRows; i++)
                Cgolden[b][i + j * p.ldc] = Chost[b][i + j * p.ldc];
    }

    std::vector<float*> Aptrs(bs), Cptrs(bs);
    for (int b = 0; b < bs; b++) {
        Aptrs[b] = Agolden[b].data();
        Cptrs[b] = Cgolden[b].data();
    }

    int goldenDevInfo = 0;
    aclblasSgelsBatched_cpu(
        handle, p.trans, p.m, p.n, cGoldenCols, Aptrs.data(), p.lda, Cptrs.data(), p.ldc, &goldenDevInfo, p.batchSize);
    return goldenDevInfo;
}

static void verifyResults(
    const GelsBatchedParam& p, int bs, int solRows, int cGoldenCols, const std::vector<std::vector<float>>& Cout,
    const std::vector<std::vector<float>>& Cgolden)
{
    const size_t solElements = static_cast<size_t>(solRows) * cGoldenCols * bs;
    if (solElements == 0)
        return;

    std::vector<float> npuSol(solElements);
    std::vector<float> goldenSol(solElements);

    size_t offset = 0;
    for (int b = 0; b < bs; b++)
        for (int j = 0; j < cGoldenCols; j++)
            for (int i = 0; i < solRows; i++) {
                npuSol[offset] = Cout[b][i + j * p.ldc];
                goldenSol[offset] = Cgolden[b][i + j * p.ldc];
                offset++;
            }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenSol.data(), goldenSol.size());
    EXPECT_TRUE(Verifier::verifyVector(npuSol.data(), goldenSol.data(), solElements, 1, cfg, p.caseName));
}

TEST_P(GelsBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        runErrorPath(p, GelsBatchedArch35Test::handle_);
        return;
    }

    const int bs = std::max(1, p.batchSize);
    const int solRows = (p.trans == ACLBLAS_OP_N) ? p.n : p.m;
    const int cInputRows = std::max(p.m, p.n);

    std::vector<std::vector<float>> Ahost(bs), Chost(bs);
    generateBatchData(p, bs, cInputRows, Ahost, Chost);

    std::vector<std::vector<float>> Aout, Cout;
    int hostDevInfo = 0;
    aclblasStatus_t ret = aclblasSgelsBatched_npu(
        GelsBatchedArch35Test::handle_, p.trans, p.m, p.n, p.nrhs, Ahost, Chost, p.lda, p.ldc, hostDevInfo, p.batchSize,
        Aout, Cout);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.m == 0 || p.n == 0 || p.nrhs == 0 || p.batchSize == 0)
        return;

    const int cGoldenCols = std::max(p.nrhs, solRows);
    std::vector<std::vector<float>> Cgolden;
    int goldenDevInfo =
        computeGolden(GelsBatchedArch35Test::handle_, p, bs, cInputRows, p.nrhs, Ahost, Chost, Cgolden);

    const bool isRankDeficient = (p.description.find("rank_deficient") != std::string::npos);
    if (isRankDeficient) {
        EXPECT_GT(hostDevInfo, 0) << "[" << p.caseName << "] Expected NPU devInfo > 0 for rank deficient matrix";
        EXPECT_GT(goldenDevInfo, 0) << "[" << p.caseName << "] Expected golden devInfo > 0 for rank deficient matrix";
        return;
    }

    verifyResults(p, bs, solRows, p.nrhs, Cout, Cgolden);
}
