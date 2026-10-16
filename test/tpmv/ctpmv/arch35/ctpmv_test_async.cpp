/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <vector>

#include "blas_test.h"
#include "ctpmv_param.h"
#include "ctpmv_golden.h"

using CtpmvArch35Test = BlasTest<CtpmvParam>;

namespace {
struct CtpmvAsyncCase {
    int n;
    int incx;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    size_t workspaceBytes;
};

const CtpmvAsyncCase ASYNC_CASES[] = {
    {4, 1, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 64},
    {8, -2, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_UNIT, 128},
    {17, 1, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 272},
    {32, -1, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 512},
    {33, 2, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 8712},
    {128, 1, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 33792},
    {129, -2, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 0},
    {257, 1, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_UNIT, 0},
    {4, -1, ACLBLAS_LOWER, ACLBLAS_OP_C, ACLBLAS_NON_UNIT, 320},
    {17, 2, ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_UNIT, 0},
    {9, 1, ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 32984},
    {17, 1, ACLBLAS_LOWER, ACLBLAS_OP_C, ACLBLAS_UNIT, 33176},
};

struct CtpmvAsyncBuffers {
    aclblasHandle_t handle = nullptr;
    void* ap = nullptr;
    void* x = nullptr;
    void* seed = nullptr;
    void* workspace = nullptr;

    ~CtpmvAsyncBuffers()
    {
        if (handle != nullptr && aclblasDestroy(handle) != ACLBLAS_STATUS_SUCCESS) {
            return;
        }
        for (void* ptr : {ap, x, seed, workspace}) {
            if (ptr != nullptr) {
                aclrtFree(ptr);
            }
        }
    }
};

void MakeAsyncInputs(const CtpmvAsyncCase& c, std::vector<aclblasComplex>& ap, std::vector<aclblasComplex>& x)
{
    ap.assign(static_cast<size_t>(c.n) * (c.n + 1) / 2, aclblasComplex{0.0f, 0.0f});
    x.assign(1 + static_cast<size_t>(c.n - 1) * std::abs(c.incx), aclblasComplex{37.0f, -37.0f});
    for (int i = 0; i < c.n; ++i) {
        const size_t diagonal = c.uplo == ACLBLAS_UPPER ? static_cast<size_t>(i) * (i + 1) / 2 + i :
                                                          static_cast<size_t>(i) * (2 * c.n - i - 1) / 2 + i;
        ap[diagonal] = aclblasComplex{2.0f, 0.5f};
        const size_t offset = static_cast<size_t>(c.incx > 0 ? i : c.n - 1 - i) * std::abs(c.incx);
        x[offset] = aclblasComplex{static_cast<float>(i % 7 - 3) / 8.0f, static_cast<float>(i % 5 - 2) / 8.0f};
    }
}

void PrepareAsyncBuffers(
    CtpmvAsyncBuffers& buffers, aclrtStream stream, const std::vector<aclblasComplex>& ap,
    const std::vector<aclblasComplex>& x)
{
    ASSERT_EQ(aclblasCreate(&buffers.handle), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSetStream(buffers.handle, stream), ACLBLAS_STATUS_SUCCESS);
    const size_t apBytes = ap.size() * sizeof(aclblasComplex);
    const size_t xBytes = x.size() * sizeof(aclblasComplex);
    ASSERT_EQ(aclrtMalloc(&buffers.ap, apBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.x, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.seed, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(buffers.ap, apBytes, ap.data(), apBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(buffers.seed, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
}

aclblasStatus_t SubmitAsyncCtpmv(const CtpmvAsyncCase& c, CtpmvAsyncBuffers& buffers)
{
    return aclblasCtpmv(
        buffers.handle, c.uplo, c.trans, c.diag, c.n, static_cast<const aclblasComplex*>(buffers.ap),
        static_cast<aclblasComplex*>(buffers.x), c.incx);
}

void CheckAsyncResult(const std::vector<aclblasComplex>& expected, const CtpmvAsyncBuffers& buffers)
{
    std::vector<aclblasComplex> result(expected.size());
    const size_t bytes = result.size() * sizeof(aclblasComplex);
    ASSERT_EQ(aclrtMemcpy(result.data(), bytes, buffers.x, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i].real, expected[i].real) << "index=" << i;
        EXPECT_FLOAT_EQ(result[i].imag, expected[i].imag) << "index=" << i;
    }
}

void RunAsyncCase(const CtpmvAsyncCase& c, aclrtStream stream, bool userWorkspace, bool destroyBeforeRead)
{
    SCOPED_TRACE(::testing::Message() << "n=" << c.n << " incx=" << c.incx << " trans=" << c.trans);
    std::vector<aclblasComplex> ap;
    std::vector<aclblasComplex> expected;
    MakeAsyncInputs(c, ap, expected);
    CtpmvAsyncBuffers buffers;
    ASSERT_NO_FATAL_FAILURE(PrepareAsyncBuffers(buffers, stream, ap, expected));
    if (userWorkspace) {
        ASSERT_EQ(aclrtMalloc(&buffers.workspace, c.workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclblasSetWorkspace(buffers.handle, buffers.workspace, c.workspaceBytes - 1), ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(SubmitAsyncCtpmv(c, buffers), ACLBLAS_STATUS_EXECUTION_FAILED);
        ASSERT_EQ(aclblasSetWorkspace(buffers.handle, buffers.workspace, c.workspaceBytes), ACLBLAS_STATUS_SUCCESS);
    }
    const size_t bytes = expected.size() * sizeof(aclblasComplex);
    ASSERT_EQ(
        aclrtMemcpyAsync(buffers.x, bytes, buffers.seed, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream), ACL_SUCCESS);
    for (int call = 0; call < 2; ++call) {
        ASSERT_EQ(SubmitAsyncCtpmv(c, buffers), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasCtpmv_cpu(buffers.handle, c.uplo, c.trans, c.diag, c.n, ap.data(), expected.data(), c.incx),
            ACLBLAS_STATUS_SUCCESS);
    }
    if (destroyBeforeRead) {
        ASSERT_EQ(aclblasDestroy(buffers.handle), ACLBLAS_STATUS_SUCCESS);
        buffers.handle = nullptr;
    } else {
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    }
    CheckAsyncResult(expected, buffers);
}
} // namespace

TEST_F(CtpmvArch35Test, AsyncBackToBackAllPaths)
{
    for (const auto& c : ASYNC_CASES) {
        ASSERT_NO_FATAL_FAILURE(RunAsyncCase(c, stream_, false, false));
    }
}

TEST_F(CtpmvArch35Test, AsyncUserWorkspaceCapacityAndOwnership)
{
    for (const auto& c : ASYNC_CASES) {
        if (c.workspaceBytes != 0) {
            ASSERT_NO_FATAL_FAILURE(RunAsyncCase(c, stream_, true, true));
        }
    }
}

TEST_F(CtpmvArch35Test, AsyncHandleDestructionCompletesQueuedWork)
{
    for (const auto& c : ASYNC_CASES) {
        ASSERT_NO_FATAL_FAILURE(RunAsyncCase(c, stream_, false, true));
    }
}
