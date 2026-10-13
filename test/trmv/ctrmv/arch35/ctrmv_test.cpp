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
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ctrmv_fill.h"
#include "ctrmv_param.h"
#include "ctrmv_golden.h"
#include "ctrmv_npu_wrapper.h"

// Perf cases (TC_PF_*): warmup + >50 effective samples, average single-call
// latency on device pointers (task §3.3: "须先 warmup 再有效采样 >50 次取平均").
static void RunPerfCase(
    const CtrmvParam& p, aclblasHandle_t handle, aclrtStream stream, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& x)
{
    const int absIncx = std::abs(p.incx);
    const size_t xBytes = static_cast<size_t>((p.n - 1) * absIncx + 1) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(p.lda) * p.n * sizeof(aclblasComplex);

    void* dA = nullptr;
    void* dX = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dA, aBytes, a.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dX, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    auto* dAC = static_cast<const aclblasComplex*>(dA);
    auto* dXC = static_cast<aclblasComplex*>(dX);
    for (int i = 0; i < 10; i++) {
        aclblasCtrmv(handle, p.uplo, p.trans, p.diag, p.n, dAC, p.lda, dXC, p.incx);
    }
    aclrtSynchronizeStream(stream);

    const int samples = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; i++) {
        aclblasCtrmv(handle, p.uplo, p.trans, p.diag, p.n, dAC, p.lda, dXC, p.incx);
    }
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avgUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / samples;
    printf(
        "[%s] avg_single_us=%.2f n=%d uplo=%s trans=%s diag=%s\n", p.caseName.c_str(), avgUs, p.n,
        p.uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER",
        p.trans == ACLBLAS_OP_N ? "N" : (p.trans == ACLBLAS_OP_T ? "T" : "C"),
        p.diag == ACLBLAS_UNIT ? "UNIT" : "NON_UNIT");

    aclrtFree(dA);
    aclrtFree(dX);
}

// Compare device result against the cblas_ctrmv golden in logical order,
// real/imag verified separately per FLOAT32 ecosystem standard (task §3.2).
static void VerifyAccuracy(
    const CtrmvParam& p, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& golden)
{
    const int absIncx = std::abs(p.incx);
    std::vector<aclblasComplex> outLogical(p.n);
    std::vector<aclblasComplex> goldLogical(p.n);
    for (int i = 0; i < p.n; i++) {
        const int64_t phys = (p.incx > 0) ? static_cast<int64_t>(i) * absIncx :
                                            static_cast<int64_t>(p.n - 1 - static_cast<int>(i)) * absIncx;
        outLogical[i] = x[phys];
        goldLogical[i] = golden[phys];
    }
    const float* outF = reinterpret_cast<const float*>(outLogical.data());
    const float* goldF = reinterpret_cast<const float*>(goldLogical.data());
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldF, static_cast<size_t>(p.n) * 2);
    EXPECT_TRUE(Verifier::verifyVector(outF, goldF, static_cast<size_t>(p.n), 2, cfg, p.caseName + "_real"));
    EXPECT_TRUE(Verifier::verifyVector(outF + 1, goldF + 1, static_cast<size_t>(p.n), 2, cfg, p.caseName + "_imag"));
}

class CtrmvTest : public BlasTest<CtrmvParam> {};

INSTANTIATE_TEST_SUITE_P(
    Ctrmv, CtrmvTest, ::testing::ValuesIn(GetCasesFromCsv<CtrmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtrmvParam>);

TEST_P(CtrmvTest, CsvDriven)
{
    const auto& p = GetParam();

    const int absIncx = std::abs(p.incx);
    const size_t xBytes = (p.n > 0) ? static_cast<size_t>((p.n - 1) * absIncx + 1) * sizeof(aclblasComplex) : 0;
    const size_t aBytes = (p.n > 0) ? static_cast<size_t>(p.lda) * p.n * sizeof(aclblasComplex) : 0;

    // Build A and x on host. NULLPTR fill produces an empty vector whose data()
    // is null, exercising the API's null-pointer validation paths.
    // RANDOM_NAN_OFFTRI fills the readable triangle normally and poisons every
    // cell ctrmv must not read (opposite triangle; the diagonal too when
    // diag=UNIT) with NaN - reading a poisoned cell would fail the case.
    std::vector<aclblasComplex> a;
    if (p.aFill == "RANDOM_NAN_OFFTRI") {
        a = CtrmvMakeMatrix(p.n, p.n, p.lda, "RANDOM_NORM_5_5", p.randomSeed);
        CtrmvPoisonUnreadTri(a, p.n, p.lda, p.uplo == ACLBLAS_UPPER, p.diag == ACLBLAS_UNIT);
    } else {
        a = CtrmvMakeMatrix(p.n, p.n, p.lda, p.aFill, p.randomSeed);
    }
    std::vector<aclblasComplex> x = CtrmvMakeStrided(p.n, p.incx, p.xFill, p.randomSeed + 1);
    std::vector<aclblasComplex> golden = x;

    aclblasStatus_t ret =
        aclblasCtrmv_npu(CtrmvTest::handle_, p.uplo, p.trans, p.diag, p.n, a.data(), p.lda, x.data(), p.incx);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // n == 0 is a valid no-op (task spec §2.4); there is no output to compare.
    if (p.n == 0) {
        return;
    }

    if (p.caseName.rfind("TC_PF_", 0) == 0) {
        RunPerfCase(p, CtrmvTest::handle_, CtrmvTest::stream_, a, x);
        return;
    }

    aclblasStatus_t cpuRet =
        aclblasCtrmv_cpu(CtrmvTest::handle_, p.uplo, p.trans, p.diag, p.n, a.data(), p.lda, golden.data(), p.incx);
    ASSERT_EQ(cpuRet, ACLBLAS_STATUS_SUCCESS);

    VerifyAccuracy(p, x, golden);
}
