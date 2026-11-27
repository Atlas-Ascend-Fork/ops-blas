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
#include <cstdint>
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ctrttp_golden.h"
#include "ctrttp_npu_wrapper.h"
#include "ctrttp_param.h"

namespace {

// Elements written after AP[n(n+1)/2] before the call and checked afterwards. Any write past
// the packed length is an out-of-bounds store and must be reported, not silently accepted.
constexpr int64_t CTRTTP_TAIL_GUARD = 64;

std::vector<aclblasComplex> MakeSentinel(int64_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count <= 0) {
        return {};
    }
    std::vector<float> real = makeBlasArray(count, fill, seed);
    std::vector<float> imag = makeBlasArray(count, fill, seed + 1000U);
    std::vector<aclblasComplex> data(static_cast<size_t>(count));
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = aclblasComplex{real[i], imag[i]};
    }
    return data;
}

bool TailUnchanged(
    const std::vector<aclblasComplex>& before, const std::vector<aclblasComplex>& after, int64_t tailBegin)
{
    if (before.size() != after.size()) {
        return false;
    }
    for (int64_t i = tailBegin; i < static_cast<int64_t>(after.size()); i++) {
        if (std::memcmp(&before[static_cast<size_t>(i)], &after[static_cast<size_t>(i)], sizeof(aclblasComplex)) != 0) {
            return false;
        }
    }
    return true;
}

} // namespace

// G.NAM.02: the test fixture name must match the file name (ctrttp_test), so no arch suffix here.
class CtrttpTest : public BlasTest<CtrttpParam> { };

TEST_F(CtrttpTest, NullHandle)
{
    aclblasStatus_t ret = aclblasCtrttp_npu(nullptr, ACLBLAS_LOWER, 5, nullptr, 5, nullptr, 0);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// A and AP must not alias (task book 2.4: the behavior is undefined). The contract leaves
// AP's content unconstrained, so this case only pins the call-site status: the library must
// neither reject the call nor crash. Correctness of an aliased call is not claimed.
TEST_F(CtrttpTest, OverlapUndefined)
{
    constexpr int order = 8;
    constexpr size_t bytes = static_cast<size_t>(order) * order * sizeof(aclblasComplex);
    void* buffer = nullptr;
    ASSERT_EQ(aclrtMalloc(&buffer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(buffer, bytes, 0, bytes), ACL_SUCCESS);

    aclblasStatus_t ret = aclblasCtrttp(
        CtrttpTest::handle_, ACLBLAS_LOWER, order, static_cast<const aclblasComplex*>(buffer), order,
        static_cast<aclblasComplex*>(buffer));
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    aclrtFree(buffer);
}

// Task book 2.5 requires deterministic results: the same device, input and tiling must produce
// byte-identical AP on every call. Each block owns whole columns, so repeats are bit-identical.
TEST_F(CtrttpTest, DeterministicRepeat)
{
    constexpr int order = 65;
    constexpr int lda = 67;
    const int64_t packed = static_cast<int64_t>(order) * (order + 1) / 2;
    const std::vector<aclblasComplex> aHost =
        makeBlasComplexMatrix(order, order, lda, parseFill("RANDOM_NORM_5_5"), 424242U);
    const std::vector<aclblasComplex> sentinel = MakeSentinel(packed, parseFill("VALUE_NORM_N999"), 7U);
    for (const aclblasFillMode_t uplo : {ACLBLAS_LOWER, ACLBLAS_UPPER}) {
        std::vector<aclblasComplex> first = sentinel;
        std::vector<aclblasComplex> second = sentinel;
        ASSERT_EQ(
            aclblasCtrttp_npu(handle_, uplo, order, aHost.data(), lda, first.data(), packed),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasCtrttp_npu(handle_, uplo, order, aHost.data(), lda, second.data(), packed),
            ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(std::memcmp(first.data(), second.data(), sizeof(aclblasComplex) * packed), 0)
            << "repeat mismatch, uplo=" << static_cast<int>(uplo);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Ctrttp, CtrttpTest,
    ::testing::ValuesIn(GetCasesFromCsv<CtrttpParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtrttpParam>);

TEST_P(CtrttpTest, CsvDriven)
{
    const auto& p = GetParam();

    const int64_t packed = (p.n > 0) ? static_cast<int64_t>(p.n) * (p.n + 1) / 2 : 0;
    const int64_t apCapacity = packed + CTRTTP_TAIL_GUARD;

    // A is lda x n column-major; the generator fills the whole lda * n storage, so the
    // non-uplo triangle and the lda padding rows hold data the operator must never read.
    std::vector<aclblasComplex> aHost =
        makeBlasComplexMatrix(p.n, p.n, std::max(1, p.lda), p.a, p.randomSeed);
    std::vector<aclblasComplex> apHost = MakeSentinel(apCapacity, p.ap, p.randomSeed + 7U);

    const std::vector<aclblasComplex> apBefore = apHost;
    const aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();
    aclblasComplex* apPtr = apHost.empty() ? nullptr : apHost.data();

    aclblasStatus_t ret = aclblasCtrttp_npu(
        CtrttpTest::handle_, p.uplo, p.n, aPtr, p.lda, apPtr, apCapacity);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult)) << "case " << p.caseName;
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    // AP beyond the packed length must keep its sentinel: n == 0 must not write at all.
    EXPECT_TRUE(TailUnchanged(apBefore, apHost, packed)) << "AP tail beyond packed length was modified";
    if (packed == 0) {
        return;
    }

    std::vector<aclblasComplex> golden(static_cast<size_t>(packed));
    aclblasCtrttp_cpu(CtrttpTest::handle_, p.uplo, p.n, aHost.data(), p.lda, golden.data());
    EXPECT_TRUE(CtrttpVerifyBitExact(apPtr, golden.data(), static_cast<uint64_t>(packed), p.caseName, "AP"));
}
