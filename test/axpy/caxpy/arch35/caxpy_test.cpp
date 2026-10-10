/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <vector>

#include "blas_test.h"
#include "caxpy_golden.h"
#include "caxpy_npu_wrapper.h"
#include "caxpy_param.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"

class CaxpyArch35Test : public BlasTest<CaxpyParam> {};

TEST_F(CaxpyArch35Test, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    EXPECT_EQ(aclblasCaxpy_npu(nullptr, 1, &alpha, nullptr, 1, nullptr, 1), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CaxpyArch35Test, ZeroLengthDoesNotAccessVectors)
{
    EXPECT_EQ(aclblasCaxpy_npu(CaxpyArch35Test::handle_, 0, nullptr, nullptr, 0, nullptr, 0), ACLBLAS_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    Caxpy, CaxpyArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CaxpyParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CaxpyParam>);

TEST_P(CaxpyArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    int64_t xSpan = p.n > 0 ? std::abs(static_cast<int64_t>(p.incx)) * (p.n - 1) + 1 : 1;
    int64_t ySpan = p.n > 0 ? std::abs(static_cast<int64_t>(p.incy)) * (p.n - 1) + 1 : 1;
    std::vector<aclblasComplex> x =
        makeBlasComplexMatrix(static_cast<int>(xSpan), 1, static_cast<int>(xSpan), p.x, p.randomSeed);
    std::vector<aclblasComplex> y =
        makeBlasComplexMatrix(static_cast<int>(ySpan), 1, static_cast<int>(ySpan), p.y, p.randomSeed + 1U);
    aclblasComplex alpha{p.alphaReal, p.alphaImag};

    const aclblasComplex* xPointer = x.empty() ? nullptr : x.data();
    aclblasComplex* yPointer = y.empty() ? nullptr : y.data();
    aclblasStatus_t status = aclblasCaxpy_npu(
        CaxpyArch35Test::handle_, p.n, &alpha, xPointer, p.incx, yPointer, p.incy);
    EXPECT_EQ(status, p.expectResult);
    if (status != ACLBLAS_STATUS_SUCCESS || p.n <= 0) {
        return;
    }

    std::vector<aclblasComplex> goldenX = makeBlasComplexMatrix(
        static_cast<int>(xSpan), 1, static_cast<int>(xSpan), p.x, p.randomSeed);
    std::vector<aclblasComplex> goldenY = makeBlasComplexMatrix(
        static_cast<int>(ySpan), 1, static_cast<int>(ySpan), p.y, p.randomSeed + 1U);
    ASSERT_EQ(
        aclblasCaxpy_cpu(CaxpyArch35Test::handle_, p.n, &alpha, goldenX.data(), p.incx, goldenY.data(), p.incy),
        ACLBLAS_STATUS_SUCCESS);

    std::vector<float> outputReal(static_cast<size_t>(p.n));
    std::vector<float> outputImag(static_cast<size_t>(p.n));
    std::vector<float> goldenReal(static_cast<size_t>(p.n));
    std::vector<float> goldenImag(static_cast<size_t>(p.n));
    for (int i = 0; i < p.n; ++i) {
        int64_t offset = caxpyCpuOffset(p.n, p.incy, i);
        outputReal[static_cast<size_t>(i)] = y[static_cast<size_t>(offset)].real;
        outputImag[static_cast<size_t>(i)] = y[static_cast<size_t>(offset)].imag;
        goldenReal[static_cast<size_t>(i)] = goldenY[static_cast<size_t>(offset)].real;
        goldenImag[static_cast<size_t>(i)] = goldenY[static_cast<size_t>(offset)].imag;
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, goldenReal.data(), goldenReal.size());
    EXPECT_TRUE(
        Verifier::verifyVector(
            outputReal.data(), goldenReal.data(), goldenReal.size(), 1, config, p.caseName + "_real"));
    EXPECT_TRUE(
        Verifier::verifyVector(
            outputImag.data(), goldenImag.data(), goldenImag.size(), 1, config, p.caseName + "_imag"));
}
