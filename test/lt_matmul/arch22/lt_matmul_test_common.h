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

#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "lt_matmul_param.h"
#include "blasLtMatmul_npu_wrapper.h"

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

namespace LtMatmul22Test {

inline void CheckInactiveWorkspace(
    DeviceBuffer& buffer, uint8_t* base, size_t offset, size_t capacity, const char* message)
{
    for (const auto region :
         {std::make_pair(size_t(0), offset), std::make_pair(offset + capacity, buffer.Bytes() - offset - capacity)}) {
        std::vector<uint8_t> bytes(region.second);
        if (!bytes.empty()) {
            AclCheck(
                aclrtMemcpy(bytes.data(), bytes.size(), base + region.first, bytes.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                message);
            EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == GUARD_BYTE; }));
        }
    }
}

inline void CheckResult(const Param& p, const Outcome& result, uint32_t expectedAlgorithm = 0U)
{
    Record(p, result);
    EXPECT_EQ(result.status, p.expectResult);
    EXPECT_TRUE(result.metrics.passed);
    if (expectedAlgorithm != 0U)
        EXPECT_EQ(result.algorithm, expectedAlgorithm);
}

inline bool CheckSimpleCase(
    const Param& p, aclblasLtHandle_t handle, aclrtStream stream, uint32_t expectedAlgorithm = 0U)
{
    try {
        Invocation operation(p, handle, stream);
        CheckResult(p, operation.Validate(false), expectedAlgorithm);
        return true;
    } catch (const std::exception& error) {
        ADD_FAILURE() << p.caseName << ": " << error.what();
        return false;
    }
}

inline void SetDirectOutput(Param& p)
{
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.inplace = false;
    p.fill[2] = "NULL";
}

inline void SetPaddedPitches(Param& p)
{
    p.ld = {p.Cols()[0] + 3, p.Cols()[1] + 5, p.n + 7, p.n + 11};
    if (p.inplace)
        p.ld[2] = p.ld[3];
}

inline void ConfigurePackedGuard(Param& p, int combo, bool staged, const char* algorithm)
{
    p.algoMode = algorithm;
    p.workspaceBytes = "auto";
    p.transA = combo & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
    p.transB = combo & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
    p.ld = {p.Cols()[0] + 3, p.Cols()[1] + 5, p.n + 4, p.n + 4};
    p.alpha = staged ? 2.0f : 1.0f;
    p.beta = staged ? -0.5f : 0.0f;
    p.inplace = staged;
    if (!staged)
        p.fill[2] = "NULL";
}

inline void ConfigureUnusedProductScalars(Param& p, int mode, bool nullInputs)
{
    if (mode == 1) {
        p.alpha = 0.0f;
        p.beta = 1.0f;
        if (nullInputs)
            p.fill[0] = p.fill[1] = "NULL";
        p.fill[2] = "BITS_EXACT";
    }
    if (mode == 2) {
        p.alpha = 0.0f;
        p.beta = 0.0f;
        if (nullInputs)
            p.fill[0] = p.fill[1] = "NULL";
        p.fill[2] = "NULL";
    }
    if (mode == 3) {
        p.alpha = std::numeric_limits<float>::infinity();
        p.beta = 0.0f;
    }
}

inline void ConfigureNonfiniteAndExactScalars(Param& p, int mode)
{
    if (mode == 4) {
        p.alpha = 1.0f;
        p.beta = std::numeric_limits<float>::quiet_NaN();
    }
    if (mode == 5) {
        p.alpha = 1.0f;
        p.beta = 0.0f;
        p.fill[0] = p.fill[1] = "BITS_EXACT";
    }
}

inline void CheckNZShiftedGuard(
    const Param& p, aclblasLtHandle_t handle, aclrtStream stream, int combo, size_t mod512, const char* message)
{
    SCOPED_TRACE(p.caseName);
    try {
        Invocation operation(p, handle, stream);
        const size_t capacity = operation.Algorithm().max_workspace_bytes;
        EXPECT_EQ(capacity, ExpectedH2Capacity(p.m, p.n, p.k, combo & 1, combo & 2));
        DeviceBuffer extra;
        extra.Initialize(capacity + 512U);
        auto* base = static_cast<uint8_t*>(extra.Data());
        const size_t offset = (mod512 + 512U - reinterpret_cast<uintptr_t>(base) % 512U) % 512U;
        operation.SetWorkspace(base + offset);
        CheckResult(p, operation.Validate(false), 0x22000002U);
        extra.CheckGuards();
        CheckInactiveWorkspace(extra, base, offset, capacity, message);
    } catch (const std::exception& error) {
        ADD_FAILURE() << error.what();
    }
}

class RuntimeFixture : public ::testing::Test {
public:
    static void SetUpTestSuite()
    {
        const aclError init = aclInit(nullptr);
        ASSERT_TRUE(init == ACL_SUCCESS || init == ACL_ERROR_REPEAT_INITIALIZE) << "aclInit: " << init;
        const char* device = std::getenv("LT_MATMUL_DEVICE");
        device_ = device == nullptr ? TEST_DEVICE_ID : std::stoi(device);
        ASSERT_EQ(aclrtSetDevice(device_), ACL_SUCCESS);
        ASSERT_EQ(aclblasLtCreate(&handle_), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&stream_), ACL_SUCCESS);
        ASSERT_NO_THROW(ReferenceSgemm());
    }

    static void TearDownTestSuite()
    {
        if (stream_ != nullptr) {
            EXPECT_EQ(aclrtDestroyStream(stream_), ACL_SUCCESS);
            stream_ = nullptr;
        }
        if (handle_ != nullptr) {
            EXPECT_EQ(aclblasLtDestroy(handle_), ACLBLAS_STATUS_SUCCESS);
            handle_ = nullptr;
        }
        EXPECT_EQ(aclrtResetDevice(device_), ACL_SUCCESS);
        EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
    }

protected:
    static inline aclblasLtHandle_t handle_ = nullptr;
    static inline aclrtStream stream_ = nullptr;
    static inline int device_ = TEST_DEVICE_ID;
};

} // namespace LtMatmul22Test
