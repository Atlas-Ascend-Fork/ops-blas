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
#include <algorithm>
#include <cstdint>
#include <vector>
#include <random>

#include "verify.h"
#include "csv_loader.h"
#include "fill.h"
#include "blasLtMatmul_param.h"
#include "blasLtMatmul_golden.h"
#include "blasLtMatmul_npu_wrapper.h"

namespace {

struct Fp32MatmulFills {
    BlasFillMode a;
    BlasFillMode b;
    BlasFillMode c;
};

// Plain "RANDOM" expands to [-FLT_MAX, FLT_MAX] and overflows. For large K, use constant
// factors so D ~= 1 and FP32 accumulation stays within the strict FP32 verify threshold.
Fp32MatmulFills makeFp32MatmulFills(int kDim)
{
    Fp32MatmulFills fills{};
    if (kDim >= 128) {
        fills.a.method = BlasFillMode::M_VALUE;
        fills.a.val1 = 1.0f / static_cast<float>(kDim);
        fills.b.method = BlasFillMode::M_VALUE;
        fills.b.val1 = 1.0f;
        fills.c.method = BlasFillMode::M_VALUE;
        fills.c.val1 = 0.0f;
        return fills;
    }

    fills.a.method = BlasFillMode::M_RANDOM;
    const float bound = 1.0f / std::sqrt(static_cast<float>(std::max(kDim, 1)));
    fills.a.val1 = bound;
    fills.a.val2 = bound;
    fills.b = fills.a;
    fills.c = fills.a;
    return fills;
}

struct MatmulLayoutTestData {
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    aclblasLtMatrixLayout_t a = nullptr;
    aclblasLtMatrixLayout_t b = nullptr;
    aclblasLtMatrixLayout_t c = nullptr;
    aclblasLtMatrixLayout_t d = nullptr;

    ~MatmulLayoutTestData()
    {
        if (a != nullptr) {
            aclblasLtMatrixLayoutDestroy(a);
        }
        if (b != nullptr) {
            aclblasLtMatrixLayoutDestroy(b);
        }
        if (c != nullptr) {
            aclblasLtMatrixLayoutDestroy(c);
        }
        if (d != nullptr) {
            aclblasLtMatrixLayoutDestroy(d);
        }
        if (computeDesc != nullptr) {
            aclblasLtMatmulDescDestroy(computeDesc);
        }
    }

    bool Initialize(const uint64_t rows[4], const uint64_t cols[4], const int64_t ld[4])
    {
        if (aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT) != ACLBLAS_STATUS_SUCCESS) {
            return false;
        }
        const int32_t opN = ACLBLAS_OP_N;
        if (aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)) !=
                ACLBLAS_STATUS_SUCCESS ||
            aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)) !=
                ACLBLAS_STATUS_SUCCESS) {
            return false;
        }
        if (aclblasLtMatrixLayoutCreate(&a, ACL_FLOAT, rows[0], cols[0], ld[0]) != ACLBLAS_STATUS_SUCCESS ||
            aclblasLtMatrixLayoutCreate(&b, ACL_FLOAT, rows[1], cols[1], ld[1]) != ACLBLAS_STATUS_SUCCESS ||
            aclblasLtMatrixLayoutCreate(&c, ACL_FLOAT, rows[2], cols[2], ld[2]) != ACLBLAS_STATUS_SUCCESS ||
            aclblasLtMatrixLayoutCreate(&d, ACL_FLOAT, rows[3], cols[3], ld[3]) != ACLBLAS_STATUS_SUCCESS) {
            return false;
        }
        const aclblasLtOrder_t row = ACLBLASLT_ORDER_ROW;
        const bool aSet = aclblasLtMatrixLayoutSetAttribute(a, ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)) ==
                          ACLBLAS_STATUS_SUCCESS;
        const bool bSet = aclblasLtMatrixLayoutSetAttribute(b, ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)) ==
                          ACLBLAS_STATUS_SUCCESS;
        const bool cSet = aclblasLtMatrixLayoutSetAttribute(c, ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)) ==
                          ACLBLAS_STATUS_SUCCESS;
        const bool dSet = aclblasLtMatrixLayoutSetAttribute(d, ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)) ==
                          ACLBLAS_STATUS_SUCCESS;
        return aSet && bSet && cSet && dSet;
    }
};

} // namespace

// ── LtMatmul test fixture using aclblasLtHandle_t ──
class LtMatmulArch35Test : public ::testing::TestWithParam<LtMatmulParam> {
protected:
    static aclblasLtHandle_t ltHandle_;
    static aclrtStream stream_;

    static void SetUpTestSuite()
    {
        aclError initRet = aclInit(nullptr);
        ASSERT_TRUE(initRet == ACL_SUCCESS || initRet == ACL_ERROR_REPEAT_INITIALIZE) << "aclInit failed: " << initRet;
        ASSERT_EQ(aclrtSetDevice(TEST_DEVICE_ID), ACL_SUCCESS);
        ASSERT_EQ(aclblasLtCreate(&ltHandle_), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&stream_), ACL_SUCCESS);
    }

    static void TearDownTestSuite()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (ltHandle_ != nullptr) {
            aclblasLtDestroy(ltHandle_);
            ltHandle_ = nullptr;
        }
        aclrtResetDevice(TEST_DEVICE_ID);
        aclFinalize();
    }
};

aclblasLtHandle_t LtMatmulArch35Test::ltHandle_ = nullptr;
aclrtStream LtMatmulArch35Test::stream_ = nullptr;

namespace {

constexpr int32_t kDeviceMatmulDim = 16;
constexpr size_t kDeviceMatmulElements = static_cast<size_t>(kDeviceMatmulDim) * kDeviceMatmulDim;
constexpr size_t kDeviceMatmulBytes = kDeviceMatmulElements * sizeof(float);

struct DeviceMatmulResources {
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    aclblasLtMatrixLayout_t layouts[4]{};
    void* deviceBuffers[6]{};

    ~DeviceMatmulResources()
    {
        for (void* ptr : deviceBuffers) {
            if (ptr != nullptr) {
                aclrtFree(ptr);
            }
        }
        for (auto layout : layouts) {
            if (layout != nullptr) {
                aclblasLtMatrixLayoutDestroy(layout);
            }
        }
        if (computeDesc != nullptr) {
            aclblasLtMatmulDescDestroy(computeDesc);
        }
    }
};

testing::AssertionResult createDeviceMatmulDescriptor(DeviceMatmulResources& resources)
{
    auto status = aclblasLtMatmulDescCreate(&resources.computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return testing::AssertionFailure() << "descriptor create: " << status;
    }
    const int32_t pointerMode = ACLBLASLT_POINTER_MODE_DEVICE;
    status = aclblasLtMatmulDescSetAttribute(
        resources.computeDesc, ACLBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(pointerMode));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return testing::AssertionFailure() << "pointer mode set: " << status;
    }
    return testing::AssertionSuccess();
}

testing::AssertionResult createDeviceMatmulLayouts(DeviceMatmulResources& resources)
{
    for (auto& layout : resources.layouts) {
        auto status =
            aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, kDeviceMatmulDim, kDeviceMatmulDim, kDeviceMatmulDim);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return testing::AssertionFailure() << "layout create: " << status;
        }
        const aclblasLtOrder_t row = ACLBLASLT_ORDER_ROW;
        status = aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return testing::AssertionFailure() << "layout order set: " << status;
        }
    }
    return testing::AssertionSuccess();
}

testing::AssertionResult allocateDeviceMatmulBuffers(DeviceMatmulResources& resources)
{
    for (size_t i = 0; i < 4; ++i) {
        auto status = aclrtMalloc(&resources.deviceBuffers[i], kDeviceMatmulBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != ACL_SUCCESS) {
            return testing::AssertionFailure() << "matrix allocation " << i << ": " << status;
        }
    }
    for (size_t i = 4; i < 6; ++i) {
        auto status = aclrtMalloc(&resources.deviceBuffers[i], sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != ACL_SUCCESS) {
            return testing::AssertionFailure() << "scalar allocation " << i << ": " << status;
        }
    }
    return testing::AssertionSuccess();
}

testing::AssertionResult copyDeviceMatmulInputs(const DeviceMatmulResources& resources)
{
    const float values[] = {2.0f, 3.0f, 5.0f};
    const float scalars[] = {2.0f, 3.0f};
    std::vector<float> matrix(kDeviceMatmulElements);
    for (size_t i = 0; i < 3; ++i) {
        std::fill(matrix.begin(), matrix.end(), values[i]);
        auto status = aclrtMemcpy(
            resources.deviceBuffers[i], kDeviceMatmulBytes, matrix.data(), kDeviceMatmulBytes,
            ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != ACL_SUCCESS) {
            return testing::AssertionFailure() << "matrix copy " << i << ": " << status;
        }
    }
    for (size_t i = 0; i < 2; ++i) {
        auto status = aclrtMemcpy(
            resources.deviceBuffers[i + 4], sizeof(float), &scalars[i], sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != ACL_SUCCESS) {
            return testing::AssertionFailure() << "scalar copy " << i << ": " << status;
        }
    }
    return testing::AssertionSuccess();
}

testing::AssertionResult runDevicePointerModeMatmul(
    aclblasLtHandle_t handle, aclrtStream stream, const DeviceMatmulResources& resources, std::vector<float>& result)
{
    auto status = aclblasLtMatmul(
        handle, resources.computeDesc, resources.deviceBuffers[4], resources.deviceBuffers[0], resources.layouts[0],
        resources.deviceBuffers[1], resources.layouts[1], resources.deviceBuffers[5], resources.deviceBuffers[2],
        resources.layouts[2], resources.deviceBuffers[3], resources.layouts[3], nullptr, nullptr, 0, stream);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return testing::AssertionFailure() << "matmul: " << status;
    }
    auto syncStatus = aclrtSynchronizeStream(stream);
    if (syncStatus != ACL_SUCCESS) {
        return testing::AssertionFailure() << "stream synchronize: " << syncStatus;
    }
    auto copyStatus = aclrtMemcpy(
        result.data(), kDeviceMatmulBytes, resources.deviceBuffers[3], kDeviceMatmulBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (copyStatus != ACL_SUCCESS) {
        return testing::AssertionFailure() << "result copy: " << copyStatus;
    }
    return testing::AssertionSuccess();
}

} // namespace

TEST_F(LtMatmulArch35Test, DevicePointerModeReadsScalarsOnDevice)
{
    DeviceMatmulResources resources;
    ASSERT_TRUE(createDeviceMatmulDescriptor(resources));
    ASSERT_TRUE(createDeviceMatmulLayouts(resources));
    ASSERT_TRUE(allocateDeviceMatmulBuffers(resources));
    ASSERT_TRUE(copyDeviceMatmulInputs(resources));
    std::vector<float> result(kDeviceMatmulElements, 0.0f);
    ASSERT_TRUE(runDevicePointerModeMatmul(ltHandle_, stream_, resources, result));
    for (float v : result) {
        ASSERT_NEAR(v, 207.0f, 1e-4f);
    }
}

// issue #801: 执行链仅实现 DEFAULT epilogue；BIAS/RELU/GELU 等属性此前被静默忽略按
// DEFAULT 计算产生错误结果。入口与 heuristic 现以 NOT_SUPPORTED 显式拒绝（校验先于
// 设备访问，哑缓冲区不会被读写）。
TEST_F(LtMatmulArch35Test, NonDefaultEpilogueRejected)
{
    aclblasLtMatmulDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&desc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    const int32_t opN = ACLBLAS_OP_N;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(desc, ACLBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(desc, ACLBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)), ACLBLAS_STATUS_SUCCESS);
    const int32_t epilogueBias = ACLBLASLT_EPILOGUE_BIAS;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(desc, ACLBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)),
        ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatrixLayout_t layouts[4] = {nullptr, nullptr, nullptr, nullptr};
    for (auto& layout : layouts) {
        ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 16, 16, 16), ACLBLAS_STATUS_SUCCESS);
        const aclblasLtOrder_t row = ACLBLASLT_ORDER_ROW;
        ASSERT_EQ(
            aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)),
            ACLBLAS_STATUS_SUCCESS);
    }

    constexpr size_t kBytes = 16 * 16 * sizeof(float);
    void* deviceBuffers[4] = {nullptr, nullptr, nullptr, nullptr};
    for (void*& buffer : deviceBuffers) {
        ASSERT_EQ(aclrtMalloc(&buffer, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    EXPECT_EQ(
        aclblasLtMatmul(
            ltHandle_, desc, &alpha, deviceBuffers[0], layouts[0], deviceBuffers[1], layouts[1], &beta,
            deviceBuffers[2], layouts[2], deviceBuffers[3], layouts[3], nullptr, nullptr, 0, stream_),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    aclblasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetHeuristic(
            ltHandle_, desc, layouts[0], layouts[1], layouts[2], layouts[3], nullptr, 1, &heuristic, &returned),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    for (void*& buffer : deviceBuffers) {
        aclrtFree(buffer);
    }
    for (auto& layout : layouts) {
        aclblasLtMatrixLayoutDestroy(layout);
    }
    aclblasLtMatmulDescDestroy(desc);
}

// ── TEST_F: null handle (TC_L0_25) ──
TEST_F(LtMatmulArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasLtMatmul_npu(
        nullptr, nullptr, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 16, 16, 16, ACLBLAS_OP_N, ACLBLAS_OP_N, 16, 16,
        16, 16, 1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "default", false, true, false, false,
        false); // handleNull=true
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

// ── TEST_F: A/B_SCALE_MODE Set/Get roundtrip (issue #639) ──
TEST_F(LtMatmulArch35Test, ScaleModeAttributesRoundTrip)
{
    aclblasLtMatmulDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&desc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);

    // default read-back succeeds and yields 0 instead of NOT_SUPPORTED
    int32_t out = -1;
    size_t sizeWritten = 0;
    EXPECT_EQ(
        aclblasLtMatmulDescGetAttribute(desc, ACLBLASLT_MATMUL_DESC_A_SCALE_MODE, &out, sizeof(out), &sizeWritten),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(out, 0);
    EXPECT_EQ(sizeWritten, sizeof(out));

    const int32_t modeA = 1;
    const int32_t modeB = 2;
    EXPECT_EQ(
        aclblasLtMatmulDescSetAttribute(desc, ACLBLASLT_MATMUL_DESC_A_SCALE_MODE, &modeA, sizeof(modeA)),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasLtMatmulDescSetAttribute(desc, ACLBLASLT_MATMUL_DESC_B_SCALE_MODE, &modeB, sizeof(modeB)),
        ACLBLAS_STATUS_SUCCESS);

    out = -1;
    EXPECT_EQ(
        aclblasLtMatmulDescGetAttribute(desc, ACLBLASLT_MATMUL_DESC_A_SCALE_MODE, &out, sizeof(out), &sizeWritten),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(out, modeA);
    out = -1;
    EXPECT_EQ(
        aclblasLtMatmulDescGetAttribute(desc, ACLBLASLT_MATMUL_DESC_B_SCALE_MODE, &out, sizeof(out), &sizeWritten),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(out, modeB);

    // payload size other than int32_t is rejected
    const int16_t small = 1;
    EXPECT_EQ(
        aclblasLtMatmulDescSetAttribute(desc, ACLBLASLT_MATMUL_DESC_A_SCALE_MODE, &small, sizeof(small)),
        ACLBLAS_STATUS_INVALID_VALUE);

    aclblasLtMatmulDescDestroy(desc);
}

// ── TEST_F: null computeDesc (TC_L0_26) ──
TEST_F(LtMatmulArch35Test, NullComputeDesc)
{
    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasLtMatmul(
        LtMatmulArch35Test::ltHandle_, nullptr, &alpha, nullptr, nullptr, nullptr, nullptr, &beta, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, 0, LtMatmulArch35Test::stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// ── TEST_F: transform desc GetAttribute 探测模式（issue #664）──
TEST_F(LtMatmulArch35Test, TransformDescGetAttributeProbeMode)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);

    // 探测模式：buf=null、sizeInBytes=0 → SUCCESS 且 sizeWritten 回写属性宽度
    size_t sizeWritten = 0;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, nullptr, 0, &sizeWritten),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(sizeWritten, sizeof(int32_t));

    // 常规取值：默认 scaleType 回读
    aclDataType scaleType = ACL_DT_UNDEFINED;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &scaleType, sizeof(scaleType), &sizeWritten),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(scaleType, ACL_FLOAT);

    // Set 后回读一致（确认 Get 拆分未破坏既有语义）：pointer mode 仅接受 HOST
    const int32_t pointerMode = ACLBLASLT_POINTER_MODE_HOST;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &pointerMode, sizeof(pointerMode)),
        ACLBLAS_STATUS_SUCCESS);
    int32_t modeOut = -1;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &modeOut, sizeof(modeOut), &sizeWritten),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeOut, pointerMode);

    // 非 HOST 取值在 Set 阶段被拒（本库标量执行链只实现主机指针模式），且不改写描述符
    const int32_t modeDevice = ACLBLASLT_POINTER_MODE_DEVICE;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &modeDevice, sizeof(modeDevice)),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &modeOut, sizeof(modeOut), &sizeWritten),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeOut, pointerMode);

    // buf 非空但尺寸不足仍拒绝（语义不变）
    int16_t small = 0;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &small, sizeof(small), &sizeWritten),
        ACLBLAS_STATUS_INVALID_VALUE);

    aclblasLtMatrixTransformDescDestroy(desc);
}

// ── TEST_F: 启发式入口拒绝超界维度（issue #665）──
TEST_F(LtMatmulArch35Test, HeuristicRejectsOversizedDimensions)
{
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);

    const uint64_t hugeRows = 0x100000000ULL; // 2^32，超 uint32 表示范围
    aclblasLtMatrixLayout_t aLayout = nullptr;
    aclblasLtMatrixLayout_t bLayout = nullptr;
    aclblasLtMatrixLayout_t dLayout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&aLayout, ACL_FLOAT, hugeRows, 16, hugeRows), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&bLayout, ACL_FLOAT, 16, 16, 16), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&dLayout, ACL_FLOAT, hugeRows, 16, hugeRows), ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatmulHeuristicResult_t heuristicResult[1];
    int returnCount = 0;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetHeuristic(
            LtMatmulArch35Test::ltHandle_, computeDesc, aLayout, bLayout, dLayout, dLayout, nullptr, 1, heuristicResult,
            &returnCount),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(returnCount, 0);

    aclblasLtMatrixLayoutDestroy(aLayout);
    aclblasLtMatrixLayoutDestroy(bLayout);
    aclblasLtMatrixLayoutDestroy(dLayout);
    aclblasLtMatmulDescDestroy(computeDesc);
}

// ── TEST_F: 算法选择链契约矩阵（issue #792）──
TEST_F(LtMatmulArch35Test, AlgoGetIdsContract)
{
    // 正常查询：非空结果、升序、数量由 returnAlgoCount 回传
    std::vector<int> ids(64, -1);
    int numIds = -1;
    ASSERT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            static_cast<int>(ids.size()), ids.data(), &numIds),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_GT(numIds, 0);
    for (int i = 1; i < numIds; i++) {
        ASSERT_GT(ids[i], ids[i - 1]);
    }

    // scaleType 不参与类型组合校验：换 ACL_DOUBLE 结果与 ACL_FLOAT 一致
    std::vector<int> idsAlt(64, -1);
    int numIdsAlt = -1;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_DOUBLE, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            static_cast<int>(idsAlt.size()), idsAlt.data(), &numIdsAlt),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(numIdsAlt, numIds);

    // 失败路径写 returnAlgoCount=0：algoIdsArray 为空、requestedAlgoCount<=0 均拒绝
    int numIdsFail = 7;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            8, nullptr, &numIdsFail),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(numIdsFail, 0);
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            0, ids.data(), &numIdsFail),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(numIdsFail, 0);
}

// ── GetHeuristic/GetIds 契约补全（缺口 G11，评估报告 P2-5）──
// 真值源：blasLt/api/aclblaslt_matmul_algo.cpp —— GetIds 失败路径恒先写 *returnAlgoCount=0；
// handle NULL → NOT_INITIALIZED 的守卫位于 arch 条件编译之前（两 arch 同码，非分叉项）；
// 负 count / 数组 NULL → INVALID_VALUE；arch35 dtype 组合 NOT_SUPPORTED 有两条通道
// （compute 不兼容、Ctype≠Dtype）；成功路径写 min(requested, 枚举总数) 且为全表升序前缀。

// 截断语义：requested < 总数时 returnAlgoCount == requested 且逐元素等于全表升序前缀；
// requested > 总数时封顶为总数（min() 语义）。既有 AlgoGetIdsContract 以大容量请求只钉了
// 正向与升序，真正的截断此前两侧 arch 都未钉住（评估报告 §2.4）。
TEST_F(LtMatmulArch35Test, AlgoGetIdsTruncatesToRequestedPrefix)
{
    // 全量枚举作为截断参照：容量 256 远大于枚举域（3 policy × 2 L1M × 2 L1N × 3 L1K × 3 splitK）
    std::vector<int> full(256, -1);
    int total = 0;
    ASSERT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            static_cast<int>(full.size()), full.data(), &total),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_GT(total, 2);
    ASSERT_LT(total, static_cast<int>(full.size())); // requested > 总数 → 封顶为总数

    auto expectPrefix = [&](int requested) {
        std::vector<int> ids(static_cast<size_t>(requested), -1);
        int got = 0;
        ASSERT_EQ(
            aclblasLtMatmulAlgoGetIds(
                LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                ACL_FLOAT, requested, ids.data(), &got),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(got, requested); // min(requested, 总数) == requested（requested ≤ 总数）
        for (int i = 0; i < got; ++i) {
            EXPECT_EQ(ids[static_cast<size_t>(i)], full[static_cast<size_t>(i)]) << "truncated id[" << i << "]";
        }
    };
    expectPrefix(1);         // 最小截断：仅最小 id
    expectPrefix(total - 1); // 比全量少 1
    expectPrefix(total);     // 恰好容纳（min 边界）
}

// GetIds 负向矩阵：handle NULL → NOT_INITIALIZED（该守卫在 arch 条件编译之前，两 arch 同码，
// 非分叉项）；负 count → INVALID_VALUE（count=0 已由 AlgoGetIdsContract 钉住，此处补负数侧）；
// arch35 dtype 组合 NOT_SUPPORTED 两条通道（compute 不兼容、Ctype≠Dtype；arch22 的
// scaleType≠FLOAT 分支不在本套件构建目标内，按任务口径不钉）。失败路径均写 0 且不改写 ids 缓冲。
TEST_F(LtMatmulArch35Test, AlgoGetIdsRejectsNullHandleAndUnsupportedDtypes)
{
    std::vector<int> ids(8, -1);
    int got = -7;
    // handle NULL → NOT_INITIALIZED + 写 0
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetIds(
            nullptr, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            static_cast<int>(ids.size()), ids.data(), &got),
        ACLBLAS_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(got, 0);
    // 负 count → INVALID_VALUE + 写 0
    got = -7;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            -1, ids.data(), &got),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(got, 0);
    // Ctype ≠ Dtype（FLOAT vs FLOAT16）→ NOT_SUPPORTED + 写 0
    got = -7;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
            static_cast<int>(ids.size()), ids.data(), &got),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(got, 0);
    // compute/dtype 不兼容：COMPUTE_32F + INT8×INT8（INT8 对仅配 32I）→ NOT_SUPPORTED + 写 0
    got = -7;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT,
            static_cast<int>(ids.size()), ids.data(), &got),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(got, 0);
    // ids 缓冲在任何失败路径都不被写入（毒值原封不动）
    for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_EQ(ids[i], -1) << "ids[" << i << "] written on failure path";
    }
}

// 哨兵毒化：失败路径不得回写 *algo，先整体填毒值，调用后校验毒值原封不动，
// 以区分「未写」与「写了 0」（范式同本文件 AlgoCheckFailureLeavesResultUntouched）。
static void PoisonAlgoSentinel(aclblasLtMatmulAlgo_t& algo)
{
    algo.max_workspace_bytes = 0xDEADBEEFu;
    for (auto& byte : algo.data) {
        byte = 0xDEu;
    }
}

// 逐字节校验哨兵毒值原封不动；EXPECT 非致命断言，失败时报出被回写的字节下标。
static void ExpectAlgoSentinelUntouched(const aclblasLtMatmulAlgo_t& algo)
{
    EXPECT_EQ(algo.max_workspace_bytes, 0xDEADBEEFu);
    for (size_t i = 0; i < sizeof(algo.data); ++i) {
        EXPECT_EQ(algo.data[i], 0xDEu) << "algo.data[" << i << "] was written on failure";
    }
}

TEST_F(LtMatmulArch35Test, AlgoInitContract)
{
    std::vector<int> ids(16, -1);
    int numIds = 0;
    ASSERT_EQ(
        aclblasLtMatmulAlgoGetIds(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            static_cast<int>(ids.size()), ids.data(), &numIds),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_GT(numIds, 0);

    // GetIds 返回的 ID 可被 AlgoInit 接受（algoId=0 默认配置亦可）
    aclblasLtMatmulAlgo_t algo{};
    EXPECT_EQ(
        aclblasLtMatmulAlgoInit(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            ids[0], &algo),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_NE(algo.max_workspace_bytes, size_t{0});
    aclblasLtMatmulAlgo_t defaultAlgo{};
    EXPECT_EQ(
        aclblasLtMatmulAlgoInit(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            0, &defaultAlgo),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_NE(defaultAlgo.max_workspace_bytes, size_t{0});

    // 非法 ID 拒绝：负数、无法解码。失败路径不得回写 *algo（哨兵毒化校验，
    // 范式见上方 PoisonAlgoSentinel / ExpectAlgoSentinelUntouched）。
    aclblasLtMatmulAlgo_t badAlgo{};
    PoisonAlgoSentinel(badAlgo);
    EXPECT_EQ(
        aclblasLtMatmulAlgoInit(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            -1, &badAlgo),
        ACLBLAS_STATUS_INVALID_VALUE);
    ExpectAlgoSentinelUntouched(badAlgo);
    EXPECT_EQ(
        aclblasLtMatmulAlgoInit(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            0x7FFFFFFF, &badAlgo),
        ACLBLAS_STATUS_INVALID_VALUE);
    ExpectAlgoSentinelUntouched(badAlgo);
    EXPECT_EQ(
        aclblasLtMatmulAlgoInit(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            0, nullptr),
        ACLBLAS_STATUS_INVALID_VALUE);
}

static aclblasLtMatmulDesc_t CreateFp32HeuristicFixture(
    aclblasLtMatrixLayout_t* aLayout, aclblasLtMatrixLayout_t* bLayout, aclblasLtMatrixLayout_t* dLayout)
{
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    EXPECT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    if (computeDesc != nullptr) {
        EXPECT_EQ(aclblasLtMatrixLayoutCreate(aLayout, ACL_FLOAT, 32, 16, 32), ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(aclblasLtMatrixLayoutCreate(bLayout, ACL_FLOAT, 16, 16, 16), ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(aclblasLtMatrixLayoutCreate(dLayout, ACL_FLOAT, 32, 16, 32), ACLBLAS_STATUS_SUCCESS);
    }
    return computeDesc;
}

static void DestroyFp32HeuristicFixture(
    aclblasLtMatmulDesc_t computeDesc, aclblasLtMatrixLayout_t aLayout, aclblasLtMatrixLayout_t bLayout,
    aclblasLtMatrixLayout_t dLayout)
{
    aclblasLtMatrixLayoutDestroy(aLayout);
    aclblasLtMatrixLayoutDestroy(bLayout);
    aclblasLtMatrixLayoutDestroy(dLayout);
    aclblasLtMatmulDescDestroy(computeDesc);
}

TEST_F(LtMatmulArch35Test, AlgoGetHeuristicContract)
{
    aclblasLtMatrixLayout_t aLayout = nullptr;
    aclblasLtMatrixLayout_t bLayout = nullptr;
    aclblasLtMatrixLayout_t dLayout = nullptr;
    aclblasLtMatmulDesc_t computeDesc = CreateFp32HeuristicFixture(&aLayout, &bLayout, &dLayout);
    ASSERT_NE(computeDesc, nullptr);

    // pref=nullptr 按默认偏好处理，正常返回候选且各项 state 为 SUCCESS
    aclblasLtMatmulHeuristicResult_t results[4];
    int returnCount = 0;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetHeuristic(
            LtMatmulArch35Test::ltHandle_, computeDesc, aLayout, bLayout, dLayout, dLayout, nullptr, 4, results,
            &returnCount),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_GT(returnCount, 0);
    ASSERT_LE(returnCount, 4);
    for (int i = 0; i < returnCount; i++) {
        EXPECT_EQ(results[i].state, ACLBLAS_STATUS_SUCCESS);
        EXPECT_GE(results[i].workspaceSize, size_t{0});
    }

    // requested=0 拒绝且 returnAlgoCount 写 0；returnAlgoCount 为空拒绝
    returnCount = -1;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetHeuristic(
            LtMatmulArch35Test::ltHandle_, computeDesc, aLayout, bLayout, dLayout, dLayout, nullptr, 0, results,
            &returnCount),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(returnCount, 0);
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetHeuristic(
            LtMatmulArch35Test::ltHandle_, computeDesc, aLayout, bLayout, dLayout, dLayout, nullptr, 1, results,
            nullptr),
        ACLBLAS_STATUS_INVALID_VALUE);

    DestroyFp32HeuristicFixture(computeDesc, aLayout, bLayout, dLayout);
}

// 坏用例修复（栈溢出 abort 遮蔽注册序其后全部用例）：原 mismatchResult[1] 单元素栈数组被
// requestedAlgoCount=4 复用为出参，4 个回写踩坏栈金丝雀，测试体退出时 stack smashing abort
// （exit 134），套件不加 filter 无法产生任何可信信号。数组容量对齐最大请求数。
// 同时按实现真实语义修正 tiny 预算断言：FP32 默认候选 splitK=1、epilogue=DEFAULT，内在
// workspace 需求恒 0（CalculateWorkspaceForAscend），FillHeuristicResults 的预算过滤仅在
// workspaceSize > maxWorkspace 时跳过候选，1 字节预算不会过滤任何候选 → SUCCESS
// （基线实测 tinyCount=4），原 NOT_SUPPORTED 期望系误设。
TEST_F(LtMatmulArch35Test, AlgoGetHeuristicTypeMismatchAndTinyWorkspaceBudget)
{
    aclblasLtMatrixLayout_t aLayout = nullptr;
    aclblasLtMatrixLayout_t bLayout = nullptr;
    aclblasLtMatrixLayout_t dLayout = nullptr;
    aclblasLtMatmulDesc_t computeDesc = CreateFp32HeuristicFixture(&aLayout, &bLayout, &dLayout);
    ASSERT_NE(computeDesc, nullptr);

    // 类型组合不兼容：返回 INVALID_VALUE 且 results[0].state 写 INVALID_VALUE
    aclblasLtMatrixLayout_t int8Layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&int8Layout, ACL_INT8, 32, 16, 32), ACLBLAS_STATUS_SUCCESS);
    aclblasLtMatmulHeuristicResult_t results[4] = {}; // 容量覆盖本用例最大 requestedAlgoCount=4
    int mismatchCount = -1;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetHeuristic(
            LtMatmulArch35Test::ltHandle_, computeDesc, int8Layout, bLayout, dLayout, dLayout, nullptr, 1, results,
            &mismatchCount),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(results[0].state, ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(mismatchCount, 0);

    // tiny 预算（1 字节）：内在需求为 0 的候选不被过滤 → SUCCESS 且各候选 workspaceSize=0
    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);
    const size_t tinyWorkspace = 1;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &tinyWorkspace, sizeof(tinyWorkspace)),
        ACLBLAS_STATUS_SUCCESS);
    int tinyCount = -1;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetHeuristic(
            LtMatmulArch35Test::ltHandle_, computeDesc, aLayout, bLayout, dLayout, dLayout, pref, 4, results,
            &tinyCount),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_GT(tinyCount, 0);
    ASSERT_LE(tinyCount, 4);
    for (int i = 0; i < tinyCount; i++) {
        EXPECT_EQ(results[i].state, ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(results[i].workspaceSize, size_t{0});
    }
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatrixLayoutDestroy(int8Layout);
    DestroyFp32HeuristicFixture(computeDesc, aLayout, bLayout, dLayout);
}

// GetHeuristic handle NULL → INVALID_VALUE 且 *returnAlgoCount 先被写 0（ValidateHeuristicArgs
// 把 handle/computeDesc NULL 归入 INVALID_VALUE）。钉 arch35 行为：arch22 分支
// （LtMatmul22Heuristic）对 handle NULL 返回 NOT_INITIALIZED，该分叉是检视报告点名的已披露
// 文档失准项，但本套件仅按 arch35 目标构建，按任务口径 arch22 分支行为一律不钉。
TEST_F(LtMatmulArch35Test, AlgoGetHeuristicRejectsNullHandle)
{
    aclblasLtMatrixLayout_t aLayout = nullptr;
    aclblasLtMatrixLayout_t bLayout = nullptr;
    aclblasLtMatrixLayout_t dLayout = nullptr;
    aclblasLtMatmulDesc_t computeDesc = CreateFp32HeuristicFixture(&aLayout, &bLayout, &dLayout);
    ASSERT_NE(computeDesc, nullptr);

    aclblasLtMatmulHeuristicResult_t results[1] = {};
    int returnCount = -1;
    EXPECT_EQ(
        aclblasLtMatmulAlgoGetHeuristic(
            nullptr, computeDesc, aLayout, bLayout, dLayout, dLayout, nullptr, 1, results, &returnCount),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(returnCount, 0);

    DestroyFp32HeuristicFixture(computeDesc, aLayout, bLayout, dLayout);
}

TEST_F(LtMatmulArch35Test, AlgoCapGetAttributeContract)
{
    aclblasLtMatmulAlgo_t algo{};
    ASSERT_EQ(
        aclblasLtMatmulAlgoInit(
            LtMatmulArch35Test::ltHandle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
            0, &algo),
        ACLBLAS_STATUS_SUCCESS);

    // 实现能力类：SPLITK_SUPPORT 恒 0（sizeWritten 允许为空）
    uint32_t splitk = 9;
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(&algo, ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk, sizeof(splitk), nullptr),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(splitk, 0U);

    // 数组能力类：size=0 探长，sizeWritten 回传所需字节数
    size_t stagesBytes = 0;
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(&algo, ACLBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &stagesBytes),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_GT(stagesBytes, size_t{0});

    // 枚举子集外（8~12、14 及以上含哨兵）→ NOT_SUPPORTED
    uint32_t out = 0;
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(
            &algo, static_cast<aclblasLtMatmulAlgoCapAttributes_t>(8), &out, sizeof(out), nullptr),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(
            &algo, static_cast<aclblasLtMatmulAlgoCapAttributes_t>(ACLBLASLT_ALGO_CAP_MAX), &out, sizeof(out), nullptr),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    // null algo → INVALID_VALUE
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(nullptr, ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT, &out, sizeof(out), nullptr),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// ── CapGetAttribute 契约补全（缺口 G2）──
// 真值源：.cannbot/aclblasLtMatmulAlgoCapGetAttribute/2.2-开发方案设计.md §11.3 能力值表
// （exact match 零容差，SPLITK=0 / REDUCTION_MASK=0 为用户拍板值）与 §11.4 错误码矩阵
// （行 9 为黑盒不可达防御分支，按 2.1 §8.2 E1 正当排除）。

namespace {

// 默认 FP32 algo 胶囊（algoId=0）：Cap 契约组与 AlgoCheck 契约组共用的前置构造。
// 两处此前各持一份同体实现（MakeCapTestAlgo / MakeDefaultFp32Algo），按 CQ.07 收敛为一份。
bool MakeDefaultFp32Algo(aclblasLtHandle_t handle, aclblasLtMatmulAlgo_t* out)
{
    return aclblasLtMatmulAlgoInit(
               handle, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, out) ==
           ACLBLAS_STATUS_SUCCESS;
}

// 标量能力断言：SUCCESS + sizeWritten==4 + 能力值精确一致。
void ExpectCapScalarValue(const aclblasLtMatmulAlgo_t& algo, aclblasLtMatmulAlgoCapAttributes_t cap, uint32_t expected)
{
    uint32_t value = 0xDEADu;
    size_t written = 0;
    ASSERT_EQ(aclblasLtMatmulAlgoCapGetAttribute(&algo, cap, &value, sizeof(value), &written), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(value, expected);
    EXPECT_EQ(written, sizeof(uint32_t));
}

// 数组能力断言：SUCCESS + sizeWritten==16 + 逐元素精确一致（{1,2,3,4} 即升序契约）。
void ExpectCapArrayValue(const aclblasLtMatmulAlgo_t& algo, aclblasLtMatmulAlgoCapAttributes_t cap)
{
    constexpr uint32_t kExpected[4] = {1u, 2u, 3u, 4u};
    uint32_t values[4] = {0};
    size_t written = 0;
    ASSERT_EQ(aclblasLtMatmulAlgoCapGetAttribute(&algo, cap, values, sizeof(values), &written), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(written, sizeof(values));
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        EXPECT_EQ(values[i], kExpected[i]) << "cap array element " << i;
    }
}

// 探长断言（错误码矩阵行 6）：size==0 且 buf 可空 → SUCCESS + sizeWritten==所需字节数。
void ExpectCapProbeBytes(const aclblasLtMatmulAlgo_t& algo, aclblasLtMatmulAlgoCapAttributes_t cap, size_t expected)
{
    size_t written = 0;
    EXPECT_EQ(aclblasLtMatmulAlgoCapGetAttribute(&algo, cap, nullptr, 0, &written), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(written, expected);
}

// 缓冲不足断言（错误码矩阵行 7）：INVALID_VALUE，但 sizeWritten 仍回传所需字节（用户批准设计）。
void ExpectCapShortBuffer(
    const aclblasLtMatmulAlgo_t& algo, aclblasLtMatmulAlgoCapAttributes_t cap, size_t given, size_t expected)
{
    uint32_t out = 0;
    size_t written = 0;
    EXPECT_EQ(aclblasLtMatmulAlgoCapGetAttribute(&algo, cap, &out, given, &written), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(written, expected);
}

} // namespace

// 错误码矩阵缺失的 5 行（2/3/4/6/7）补齐执行——行 1/5/8 已由 AlgoCapGetAttributeContract 钉住。
TEST_F(LtMatmulArch35Test, AlgoCapGetAttributeErrorCodeMatrix)
{
    aclblasLtMatmulAlgo_t algo{};
    ASSERT_TRUE(MakeDefaultFp32Algo(LtMatmulArch35Test::ltHandle_, &algo));

    // 行 2：size==0 且 sizeWritten==NULL → INVALID_VALUE（Cap 族严格守卫，异于 Get 族宽松契约）
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(&algo, ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT, nullptr, 0, nullptr),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 行 3：size!=0 且 capAttribute==NULL → INVALID_VALUE
    size_t written = 0;
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(
            &algo, ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT, nullptr, sizeof(uint32_t), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 行 4：DecodeAlgo 失败（magic 损坏 / 全零未初始化胶囊）→ INVALID_VALUE
    aclblasLtMatmulAlgo_t badMagic = algo;
    badMagic.data[0] = static_cast<uint8_t>(badMagic.data[0] ^ 0xFFu);
    uint32_t out = 0;
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(&badMagic, ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT, &out, sizeof(out), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    const aclblasLtMatmulAlgo_t zeroInit{};
    EXPECT_EQ(
        aclblasLtMatmulAlgoCapGetAttribute(&zeroInit, ACLBLASLT_ALGO_CAP_TILE_IDS, &out, sizeof(out), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 行 6：probe（size==0，buf 可空）→ SUCCESS + sizeWritten==所需字节（标量 4 / 数组 16）
    ExpectCapProbeBytes(algo, ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT, sizeof(uint32_t));
    ExpectCapProbeBytes(algo, ACLBLASLT_ALGO_CAP_TILE_IDS, 4u * sizeof(uint32_t));
    // 行 7：缓冲不足 → INVALID_VALUE，且 sizeWritten 仍回传所需值（用户批准设计）
    ExpectCapShortBuffer(algo, ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT, sizeof(uint16_t), sizeof(uint32_t));
    ExpectCapShortBuffer(algo, ACLBLASLT_ALGO_CAP_TILE_IDS, sizeof(uint32_t), 4u * sizeof(uint32_t));
}

// 9 项拍板能力值逐项钉住（§11.3）——既有 AlgoCapGetAttributeContract 仅钉 SPLITK 一项。
TEST_F(LtMatmulArch35Test, AlgoCapGetAttributeCapabilityValues)
{
    aclblasLtMatmulAlgo_t algo{};
    ASSERT_TRUE(MakeDefaultFp32Algo(LtMatmulArch35Test::ltHandle_, &algo));

    ExpectCapScalarValue(algo, ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT, 0u);
    ExpectCapScalarValue(algo, ACLBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, 0u);
    ExpectCapScalarValue(algo, ACLBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, 0u);
    ExpectCapScalarValue(algo, ACLBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT, 0u);
    ExpectCapScalarValue(algo, ACLBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT, 1u);
    ExpectCapScalarValue(algo, ACLBLASLT_ALGO_CAP_UPLO_SUPPORT, 0u);
    ExpectCapScalarValue(algo, ACLBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, 2u);
    ExpectCapArrayValue(algo, ACLBLASLT_ALGO_CAP_TILE_IDS);
    ExpectCapArrayValue(algo, ACLBLASLT_ALGO_CAP_STAGES_IDS);
}

// ── ALGO_CAP 枚举裸数值钉值（迁移安全，循 PR 682 L2_16~21 先例）──
// 数值契约（2.2 §11.2，cuBLASLt 四源实证权威值）：0~6 + CUSTOM_OPTION_MAX=7 +
// STAGES_IDS=13 + MAX 哨兵=14；7→13 跳号是与权威值一一对应的直接代价。该族已真实
// 发生过一次 5/12 错号（PR 682 修复），符号名引用对数值漂移全盲，裸值钉值是唯一回归网。
TEST_F(LtMatmulArch35Test, AlgoCapGetAttributeMigrationRawValues)
{
    // 编译期编号逐项钉住
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT), 0);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK), 1);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT), 2);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT), 3);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT), 4);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_UPLO_SUPPORT), 5);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_TILE_IDS), 6);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX), 7);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_STAGES_IDS), 13);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_ALGO_CAP_MAX), 14);

    aclblasLtMatmulAlgo_t algo{};
    ASSERT_TRUE(MakeDefaultFp32Algo(LtMatmulArch35Test::ltHandle_, &algo));

    // 行为级裸值可达：raw 7 → CUSTOM_OPTION_MAX 标量 2/4B（committer 指控的静默错配修复点）；
    // raw 13 → STAGES_IDS 数组 {1,2,3,4}/16B（迁移代码按 cuBLASLt 真值传参的正例）
    ExpectCapScalarValue(algo, static_cast<aclblasLtMatmulAlgoCapAttributes_t>(7), 2u);
    ExpectCapArrayValue(algo, static_cast<aclblasLtMatmulAlgoCapAttributes_t>(13));
    // 子集外逐值：8~12（cuBLASLt MATHMODE_IMPL/GAUSSIAN_IMPL/CUSTOM_MEMORY_ORDER/
    // POINTER_MODE_MASK/EPILOGUE_MASK 位）、14（MAX 哨兵，与 cuBLASLt LD_NEGATIVE 数值重合
    // 但本库不提供）、22、999、UINT32_MAX（负数位模式）→ NOT_SUPPORTED
    for (uint32_t raw : {8u, 9u, 10u, 11u, 12u, 14u, 22u, 999u, 0xFFFFFFFFu}) {
        uint32_t out = 0;
        size_t written = 0;
        EXPECT_EQ(
            aclblasLtMatmulAlgoCapGetAttribute(
                &algo, static_cast<aclblasLtMatmulAlgoCapAttributes_t>(raw), &out, sizeof(out), &written),
            ACLBLAS_STATUS_NOT_SUPPORTED)
            << "raw cap " << raw;
    }
}

namespace {

// Preference Get 族「探测」断言：size==0 且 buf 可空 → SUCCESS + sizeWritten==属性宽度。
// written 预置 SIZE_MAX，故第二断言同时证明「确由本次调用回写」而非残留值。
void ExpectPrefProbeBytes(
    aclblasLtMatmulPreference_t pref, aclblasLtMatmulPreferenceAttribute_t attr, size_t expectedBytes)
{
    size_t written = SIZE_MAX;
    EXPECT_EQ(aclblasLtMatmulPreferenceGetAttribute(pref, attr, nullptr, 0, &written), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(written, expectedBytes);
}

// Preference Get 族「拒绝但告知」断言：状态码为 expected，且 sizeWritten 仍先行回传所需宽度
// （未知属性场景传 SIZE_MAX 即断言其不被改写）。
void ExpectPrefRejectedWithWidth(
    aclblasLtMatmulPreference_t pref, aclblasLtMatmulPreferenceAttribute_t attr, void* buf, size_t size,
    aclblasStatus_t expected, size_t expectedWritten)
{
    size_t written = SIZE_MAX;
    EXPECT_EQ(aclblasLtMatmulPreferenceGetAttribute(pref, attr, buf, size, &written), expected);
    EXPECT_EQ(written, expectedWritten);
}

// SEARCH_MODE 回读断言：Get 成功 + 值精确一致 + sizeWritten==4（值域扫描的公共形态）。
void ExpectPrefSearchModeReadback(aclblasLtMatmulPreference_t pref, uint32_t expected)
{
    uint32_t out = 0xDEADu;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &out, sizeof(out), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(out, expected);
    EXPECT_EQ(written, sizeof(uint32_t));
}

} // namespace

TEST_F(LtMatmulArch35Test, MatmulPreferenceGetAttributeContract)
{
    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);

    // Set/Get 回读：SEARCH_MODE（sizeWritten 为空允许）
    const uint32_t searchMode = 2;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &searchMode, sizeof(searchMode)),
        ACLBLAS_STATUS_SUCCESS);
    uint32_t modeOut = 0;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &modeOut, sizeof(modeOut), nullptr),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeOut, searchMode);

    // MAX_WORKSPACE_BYTES：实现为精确宽度匹配、仅接受 8 字节（#786 口径，
    // aclblaslt_matmul_preference.cpp 的 sizeof(size_t)/sizeof(uint64_t) 守卫）。原 4 字节
    // 正向断言源自 api_list 失准文档，ASSERT 中止使本用例后续断言全部僵尸；
    // 现改为 8 字节正向回读（4 字节宽度错配拒绝见 SetAttribute 负向契约用例）。
    const uint64_t workspace = 1024;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)),
        ACLBLAS_STATUS_SUCCESS);
    size_t workspaceOut = 0;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceOut, sizeof(workspaceOut), nullptr),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(workspaceOut, static_cast<size_t>(workspace));

    // 值域校验：SEARCH_MODE>2 拒绝、workspace 超 INT64_MAX 拒绝
    const uint32_t badMode = 3;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &badMode, sizeof(badMode)),
        ACLBLAS_STATUS_INVALID_VALUE);
    const uint64_t hugeWorkspace = 0x8000000000000000ULL;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &hugeWorkspace, sizeof(hugeWorkspace)),
        ACLBLAS_STATUS_INVALID_VALUE);

    // 探测模式：size=0 不写 buf，sizeWritten 回传所需字节数
    ExpectPrefProbeBytes(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, sizeof(uint32_t));

    // 未识别 attr → NOT_SUPPORTED
    uint32_t out = 0;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            pref, static_cast<aclblasLtMatmulPreferenceAttribute_t>(99), &out, sizeof(out), nullptr),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    aclblasLtMatmulPreferenceDestroy(pref);
}

// ── PreferenceGet/Set 负向与探测契约补全（缺口 G4：Set 是偏好族唯一写入口，
// 其拒绝分支此前零执行；Get 族宽松契约与 Cap 族严格守卫的差异此前无钉住）──

TEST_F(LtMatmulArch35Test, MatmulPreferenceSetAttributeRejectsInvalidInputs)
{
    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);

    // NULL 组合：pref NULL / buf NULL → INVALID_VALUE
    const uint32_t mode = 1;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(nullptr, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &mode, sizeof(mode)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, nullptr, sizeof(mode)),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 宽度错配：SEARCH_MODE 仅收 4 字节（8 字节拒绝）、MAX_WS 仅收 8 字节（4 字节拒绝，#786 口径）
    const uint64_t modeWide = 1;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &modeWide, sizeof(modeWide)),
        ACLBLAS_STATUS_INVALID_VALUE);
    const uint32_t wsNarrow = 64;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsNarrow, sizeof(wsNarrow)),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 未知属性（含 MAX 哨兵）→ NOT_SUPPORTED
    uint32_t ignored = 0;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_MAX, &ignored, sizeof(ignored)),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(
            pref, static_cast<aclblasLtMatmulPreferenceAttribute_t>(99), &ignored, sizeof(ignored)),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    // 被拒写入不改写胶囊：SEARCH_MODE 仍为 Create 后默认值 0
    uint32_t modeOut = 0xDEADu;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &modeOut, sizeof(modeOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeOut, 0u);

    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_SUCCESS);
}

// SEARCH_MODE 值域契约（缺口 G7/G4 残项，评估报告 P2-2）：0/1/2 逐一接受并回读一致，
// >2 拒绝且不改写。钉住 header/api_list 已披露的仓内自定义语义——0=启发式、1=穷举、
// 2=快速，其中 1/2 与 cublasLtMatmulSearch_t 分叉（cuBLASLt 侧 1=LIMITED_BY_ALGO_ID、
// 2~7=RESERVED），仅 0 两侧同义可移植；该分叉当前为潜伏状态（searchMode 全链无消费方，
// Set 存 / Get 读回，heuristic 不读取），值域 Set/Get 行为是其唯一可观测面。
// 既有零散断言（契约用例中 2 接受、3 拒绝）之外，此处首次以完整值域扫描钉住披露语义。
TEST_F(LtMatmulArch35Test, MatmulPreferenceSearchModeValueDomain)
{
    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);

    // 域内 0/1/2：Set 逐一接受 + Get 回读一致（含 sizeWritten 宽度断言）
    for (uint32_t mode : {0u, 1u, 2u}) {
        ASSERT_EQ(
            aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &mode, sizeof(mode)),
            ACLBLAS_STATUS_SUCCESS)
            << "mode=" << mode;
        ExpectPrefSearchModeReadback(pref, mode);
    }
    // 域外 3（边界+1）与 UINT32_MAX：拒绝且不改写（回读仍为最后一次合法写入的 2）。
    // 此处刻意保留 sizeWritten=nullptr 的成功路径调用——Get 族「宽出参可空」契约的独立观测点。
    for (uint32_t bad : {3u, 0xFFFFFFFFu}) {
        EXPECT_EQ(
            aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &bad, sizeof(bad)),
            ACLBLAS_STATUS_INVALID_VALUE)
            << "bad mode=" << bad;
        uint32_t out = 0;
        ASSERT_EQ(
            aclblasLtMatmulPreferenceGetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &out, sizeof(out), nullptr),
            ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(out, 2u);
    }

    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatmulArch35Test, MatmulPreferenceGetAttributeProbeAndGuards)
{
    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);

    // 探测族：size==0 → SUCCESS + sizeWritten==属性宽度（SEARCH_MODE=4 / MAX_WS=8）
    ExpectPrefProbeBytes(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, sizeof(uint32_t));
    ExpectPrefProbeBytes(pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, sizeof(uint64_t));
    // (size==0 且 sizeWritten==NULL) → SUCCESS：Get 族宽松契约（异于 Cap 族矩阵行 2 严格守卫）
    EXPECT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, nullptr, 0, nullptr),
        ACLBLAS_STATUS_SUCCESS);
    // 缓冲不足 → INVALID_VALUE，且 sizeWritten 仍回传所需宽度
    uint16_t small = 0;
    ExpectPrefRejectedWithWidth(
        pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &small, sizeof(small), ACLBLAS_STATUS_INVALID_VALUE,
        sizeof(uint64_t));
    // buf NULL + size!=0 → INVALID_VALUE（sizeWritten 已先行回传所需宽度）
    ExpectPrefRejectedWithWidth(
        pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, nullptr, sizeof(uint32_t), ACLBLAS_STATUS_INVALID_VALUE,
        sizeof(uint32_t));
    // pref NULL → INVALID_VALUE
    uint32_t modeOut = 0;
    size_t written = SIZE_MAX;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            nullptr, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &modeOut, sizeof(modeOut), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 未知属性 → NOT_SUPPORTED 且 sizeWritten 不写
    ExpectPrefRejectedWithWidth(
        pref, static_cast<aclblasLtMatmulPreferenceAttribute_t>(99), &modeOut, sizeof(modeOut),
        ACLBLAS_STATUS_NOT_SUPPORTED, SIZE_MAX);

    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_SUCCESS);
}

// ── PREFERENCE 族枚举裸数值钉值（迁移安全）──
// SEARCH_MODE=0、MAX_WORKSPACE_BYTES=1 与 cuBLASLt 对齐；MAX 哨兵=2 与 cuBLASLt
// MATH_MODE_MASK 数值重合，本库对其一律 NOT_SUPPORTED（「响亮拒绝」安全态，由此钉住）。
TEST_F(LtMatmulArch35Test, MatmulPreferenceAttributeEnumValuesPinned)
{
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATMUL_PREF_SEARCH_MODE), 0);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES), 1);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATMUL_PREF_MAX), 2);

    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);
    // 行为级裸值可达：raw 0 → SEARCH_MODE 回环；raw 1 → MAX_WS 探测宽度 8
    const uint32_t mode = 1;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(
            pref, static_cast<aclblasLtMatmulPreferenceAttribute_t>(0), &mode, sizeof(mode)),
        ACLBLAS_STATUS_SUCCESS);
    uint32_t modeOut = 0;
    size_t written = 0;
    EXPECT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            pref, static_cast<aclblasLtMatmulPreferenceAttribute_t>(0), &modeOut, sizeof(modeOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeOut, mode);
    ExpectPrefProbeBytes(pref, static_cast<aclblasLtMatmulPreferenceAttribute_t>(1), sizeof(uint64_t));
    // raw 2（MAX 哨兵）→ Set/Get 双侧 NOT_SUPPORTED
    EXPECT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_MAX, &mode, sizeof(mode)),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(pref, ACLBLASLT_MATMUL_PREF_MAX, &modeOut, sizeof(modeOut), &written),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatmulArch35Test, RejectsMismatchedContractionDimension)
{
    MatmulLayoutTestData layouts;
    const uint64_t rows[] = {2, 2, 2, 2};
    const uint64_t cols[] = {3, 2, 2, 2};
    const int64_t ld[] = {3, 2, 2, 2};
    ASSERT_TRUE(layouts.Initialize(rows, cols, ld));
    float alpha = 1.0f, beta = 0.0f, data[8]{};
    EXPECT_EQ(
        aclblasLtMatmul(
            LtMatmulArch35Test::ltHandle_, layouts.computeDesc, &alpha, data, layouts.a, data, layouts.b, &beta,
            nullptr, layouts.c, data, layouts.d, nullptr, nullptr, 0, LtMatmulArch35Test::stream_),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(LtMatmulArch35Test, RejectsLeadingDimensionShorterThanPhysicalRowWidth)
{
    MatmulLayoutTestData layouts;
    const uint64_t rows[] = {2, 3, 2, 2};
    const uint64_t cols[] = {3, 2, 2, 2};
    const int64_t ld[] = {2, 2, 2, 2};
    ASSERT_TRUE(layouts.Initialize(rows, cols, ld));
    float alpha = 1.0f, beta = 0.0f, data[8]{};
    EXPECT_EQ(
        aclblasLtMatmul(
            LtMatmulArch35Test::ltHandle_, layouts.computeDesc, &alpha, data, layouts.a, data, layouts.b, &beta,
            nullptr, layouts.c, data, layouts.d, nullptr, nullptr, 0, LtMatmulArch35Test::stream_),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(LtMatmulArch35Test, EmptyOutputReturnsSuccessWithoutMatrixPointers)
{
    MatmulLayoutTestData layouts;
    const uint64_t rows[] = {0, 0, 0, 0};
    const uint64_t cols[] = {0, 2, 2, 2};
    const int64_t ld[] = {0, 0, 0, 0};
    ASSERT_TRUE(layouts.Initialize(rows, cols, ld));
    float alpha = 1.0f, beta = 0.0f;
    EXPECT_EQ(
        aclblasLtMatmul(
            LtMatmulArch35Test::ltHandle_, layouts.computeDesc, &alpha, nullptr, layouts.a, nullptr, layouts.b, &beta,
            nullptr, layouts.c, nullptr, layouts.d, nullptr, nullptr, 0, LtMatmulArch35Test::stream_),
        ACLBLAS_STATUS_SUCCESS);
}

// ── TEST_F: alpha=nullptr (TC_L0_27) ──
TEST_F(LtMatmulArch35Test, NullAlpha)
{
    float beta = 0.0f;
    aclblasLtMatmulDesc_t desc;
    aclblasLtMatmulDescCreate(&desc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);
    aclblasStatus_t ret = aclblasLtMatmul(
        LtMatmulArch35Test::ltHandle_, desc, nullptr, nullptr, nullptr, nullptr, nullptr, &beta, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, 0, LtMatmulArch35Test::stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
    aclblasLtMatmulDescDestroy(desc);
}

// ── TEST_F: A=nullptr (TC_L0_28) ──
TEST_F(LtMatmulArch35Test, NullA)
{
    aclblasLtHandle_t handle = LtMatmulArch35Test::ltHandle_;
    aclrtStream stream = LtMatmulArch35Test::stream_;

    aclblasLtMatmulDesc_t computeDesc;
    aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);
    int32_t transN = ACLBLAS_OP_N;
    aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &transN, sizeof(int32_t));
    aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &transN, sizeof(int32_t));
    aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
    aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(uint32_t));

    aclblasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    aclblasLtOrder_t order = ACLBLASLT_ORDER_ROW;
    aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT, 16, 16, 16);
    aclblasLtMatrixLayoutSetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutCreate(&Bdesc, ACL_FLOAT, 16, 16, 16);
    aclblasLtMatrixLayoutSetAttribute(Bdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutCreate(&Cdesc, ACL_FLOAT, 16, 16, 16);
    aclblasLtMatrixLayoutSetAttribute(Cdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutCreate(&Ddesc, ACL_FLOAT, 16, 16, 16);
    aclblasLtMatrixLayoutSetAttribute(Ddesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));

    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasLtMatmul(
        handle, computeDesc, &alpha, nullptr, Adesc, nullptr, Bdesc, &beta, nullptr, Cdesc, nullptr, Ddesc, nullptr,
        nullptr, 0, stream);

    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);

    aclblasLtMatmulDescDestroy(computeDesc);
    aclblasLtMatrixLayoutDestroy(Adesc);
    aclblasLtMatrixLayoutDestroy(Bdesc);
    aclblasLtMatrixLayoutDestroy(Cdesc);
    aclblasLtMatrixLayoutDestroy(Ddesc);
}

TEST_F(LtMatmulArch35Test, DefaultColumnOrderIsRejectedByMatmul)
{
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    aclblasLtMatrixLayout_t layouts[4]{};
    for (auto& layout : layouts) {
        ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 4, 4, 4), ACLBLAS_STATUS_SUCCESS);
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    float data[16]{};
    const aclblasStatus_t ret = aclblasLtMatmul(
        LtMatmulArch35Test::ltHandle_, computeDesc, &alpha, data, layouts[0], data, layouts[1], &beta, nullptr,
        layouts[2], data, layouts[3], nullptr, nullptr, 0, LtMatmulArch35Test::stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_SUPPORTED);
    for (auto layout : layouts) {
        EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
    }
    EXPECT_EQ(aclblasLtMatmulDescDestroy(computeDesc), ACLBLAS_STATUS_SUCCESS);
}

// ── CSV parameterized tests ──
INSTANTIATE_TEST_SUITE_P(
    LtMatmul, LtMatmulArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<LtMatmulParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<LtMatmulParam>);

TEST_P(LtMatmulArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int M = p.M;
    const int N = p.N;
    const int K = p.K;
    const aclDataType dtypeA = p.dtypeA;
    const aclDataType dtypeB = p.dtypeB;
    const aclDataType dtypeC = p.dtypeC;
    const aclDataType dtypeD = p.dtypeD;

    int physRowsA = getPhysicalRowsA(M, K, p.transA);
    int physColsA = getPhysicalColsA(M, K, p.transA);
    int physRowsB = getPhysicalRowsB(K, N, p.transB);
    int physColsB = getPhysicalColsB(K, N, p.transB);

    // ── Step 1: Generate host data ──
    std::vector<float> A_data, B_data, C_data, D_data;
    std::vector<uint16_t> D_bf16;

    const Fp32MatmulFills fp32Fills = makeFp32MatmulFills(K);

    if (dtypeA == ACL_FLOAT) {
        A_data = makeBlasArray(static_cast<int64_t>(physRowsA) * p.lda, fp32Fills.a, p.randomSeed);
    }
    if (dtypeB == ACL_FLOAT) {
        B_data = makeBlasArray(static_cast<int64_t>(physRowsB) * p.ldb, fp32Fills.b, p.randomSeed);
    }
    if (dtypeC == ACL_FLOAT && !p.CIsNull) {
        C_data = makeBlasArray(static_cast<int64_t>(M) * p.ldc, fp32Fills.c, p.randomSeed);
    }
    if (dtypeD == ACL_FLOAT) {
        D_data.resize(static_cast<size_t>(M) * p.ldd, 0.0f);
    } else if (dtypeD == ACL_BF16) {
        D_bf16.resize(static_cast<size_t>(M) * p.ldd, 0);
    }

    // MXFP8/MXFP4 byte data
    std::vector<uint8_t> A_mxfp, B_mxfp;
    std::mt19937 rng(p.randomSeed ? p.randomSeed : 42);
    // Match gen_data.py: positive normal-range MXFP8 (~1..8) and E8M0 scale exponents 127..130 (~1..8)
    std::uniform_int_distribution<int> mxfp8Dist(0x38, 0x4F);
    // Pack two E2M1 nibbles per byte (low nibble first), similar to gen_data.py pack_b4_to_b8.
    std::uniform_int_distribution<int> mxfp4NibbleDist(0x3, 0x8);

    if (dtypeA == ACL_FLOAT8_E4M3FN) {
        size_t aBytes = static_cast<size_t>(physRowsA) * p.lda;
        A_mxfp.resize(aBytes);
        for (auto& b : A_mxfp)
            b = static_cast<uint8_t>(mxfp8Dist(rng));
    } else if (dtypeA == ACL_FLOAT4_E2M1) {
        // lda/ldb are logical element leading dims (same as MXFP8); pack 2 FP4 per byte.
        size_t aPackedBytes = static_cast<size_t>(physRowsA) * ((static_cast<size_t>(p.lda) + 1) / 2);
        A_mxfp.resize(aPackedBytes);
        for (size_t i = 0; i < A_mxfp.size(); ++i) {
            uint8_t lo = static_cast<uint8_t>(mxfp4NibbleDist(rng) & 0xF);
            uint8_t hi = static_cast<uint8_t>(mxfp4NibbleDist(rng) & 0xF);
            A_mxfp[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }

    if (dtypeB == ACL_FLOAT8_E4M3FN) {
        size_t bBytes = static_cast<size_t>(physRowsB) * p.ldb;
        B_mxfp.resize(bBytes);
        for (auto& b : B_mxfp)
            b = static_cast<uint8_t>(mxfp8Dist(rng));
    } else if (dtypeB == ACL_FLOAT4_E2M1) {
        size_t bPackedBytes = static_cast<size_t>(physRowsB) * ((static_cast<size_t>(p.ldb) + 1) / 2);
        B_mxfp.resize(bPackedBytes);
        for (size_t i = 0; i < B_mxfp.size(); ++i) {
            uint8_t lo = static_cast<uint8_t>(mxfp4NibbleDist(rng) & 0xF);
            uint8_t hi = static_cast<uint8_t>(mxfp4NibbleDist(rng) & 0xF);
            B_mxfp[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }

    // Scale Factor data
    std::vector<uint8_t> scaleA_data, scaleB_data;
    std::uniform_int_distribution<int> scale8Dist(127, 130);

    if (isMxfpType(dtypeA)) {
        size_t scaleABytes = mxScaleBufferBytesA(M, K, p.transA);
        scaleA_data.resize(scaleABytes);
        if (p.scaleAFill.method == BlasFillMode::M_VALUE && p.scaleAFill.val1 == 0.0f) {
            for (auto& b : scaleA_data)
                b = 0;
        } else {
            for (auto& b : scaleA_data)
                b = static_cast<uint8_t>(scale8Dist(rng));
        }
    }
    if (isMxfpType(dtypeB)) {
        size_t scaleBBytes = mxScaleBufferBytesB(N, K, p.transB);
        scaleB_data.resize(scaleBBytes);
        if (p.scaleBFill.method == BlasFillMode::M_VALUE && p.scaleBFill.val1 == 0.0f) {
            for (auto& b : scaleB_data)
                b = 0;
        } else {
            for (auto& b : scaleB_data)
                b = static_cast<uint8_t>(scale8Dist(rng));
        }
    }

    // ── Determine data pointers ──
    const void* A_ptr = nullptr;
    const void* B_ptr = nullptr;
    const float* C_ptr = nullptr;
    const void* scaleA_ptr = nullptr;
    const void* scaleB_ptr = nullptr;

    if (dtypeA == ACL_FLOAT && !A_data.empty())
        A_ptr = A_data.data();
    else if (isMxfpType(dtypeA) && !A_mxfp.empty())
        A_ptr = A_mxfp.data();

    if (dtypeB == ACL_FLOAT && !B_data.empty())
        B_ptr = B_data.data();
    else if (isMxfpType(dtypeB) && !B_mxfp.empty())
        B_ptr = B_mxfp.data();

    if (!p.CIsNull && !C_data.empty())
        C_ptr = C_data.data();
    if (isMxfpType(dtypeA) && !scaleA_data.empty())
        scaleA_ptr = scaleA_data.data();
    if (isMxfpType(dtypeB) && !scaleB_data.empty())
        scaleB_ptr = scaleB_data.data();

    void* D_output_ptr = nullptr;
    if (dtypeD == ACL_FLOAT)
        D_output_ptr = D_data.data();
    else if (dtypeD == ACL_BF16)
        D_output_ptr = D_bf16.data();

    // ── Step 2: Execute on NPU ──
    // Pass nullptr ltHandle when handleNull=true (TC_L0_25 CSV parity with TEST_F)
    aclblasLtHandle_t ltHandleForNpu = p.handleNull ? nullptr : LtMatmulArch35Test::ltHandle_;
    aclblasStatus_t ret = aclblasLtMatmul_npu(
        ltHandleForNpu, LtMatmulArch35Test::stream_, dtypeA, dtypeB, dtypeC, dtypeD, M, N, K, p.transA, p.transB, p.lda,
        p.ldb, p.ldc, p.ldd, p.alpha, p.beta, A_ptr, B_ptr, C_ptr, D_output_ptr, scaleA_ptr, scaleB_ptr, p.algoMode,
        p.CIsNull, p.handleNull, p.computeDescNull, p.alphaNull, p.Anull);

    // ── Step 3: Compare error code ──
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // ── Step 4: Compute golden ──
    aclblasLtHandle_t cpuHandle = LtMatmulArch35Test::ltHandle_;
    aclblasLtMatmulDesc_t cpuComputeDesc = nullptr;
    aclblasLtMatrixLayout_t cpuAdesc = nullptr, cpuBdesc = nullptr, cpuCdesc = nullptr, cpuDdesc = nullptr;

    aclblasLtMatmulDescCreate(&cpuComputeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);
    int32_t cpuTransA = static_cast<int32_t>(p.transA);
    int32_t cpuTransB = static_cast<int32_t>(p.transB);
    aclblasLtMatmulDescSetAttribute(cpuComputeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &cpuTransA, sizeof(int32_t));
    aclblasLtMatmulDescSetAttribute(cpuComputeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &cpuTransB, sizeof(int32_t));

    aclblasLtOrder_t order = ACLBLASLT_ORDER_ROW;
    int64_t cpuLdA = (dtypeA == ACL_FLOAT4_E2M1) ? mxfp4PackedLd(p.lda) : p.lda;
    int64_t cpuLdB = (dtypeB == ACL_FLOAT4_E2M1) ? mxfp4PackedLd(p.ldb) : p.ldb;
    aclblasLtMatrixLayoutCreate(&cpuAdesc, dtypeA, physRowsA, physColsA, cpuLdA);
    aclblasLtMatrixLayoutSetAttribute(cpuAdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutCreate(&cpuBdesc, dtypeB, physRowsB, physColsB, cpuLdB);
    aclblasLtMatrixLayoutSetAttribute(cpuBdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutCreate(&cpuCdesc, dtypeC, M, N, p.ldc);
    aclblasLtMatrixLayoutSetAttribute(cpuCdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutCreate(&cpuDdesc, dtypeD, M, N, p.ldd);
    aclblasLtMatrixLayoutSetAttribute(cpuDdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));

    std::vector<float> golden;
    if (dtypeD == ACL_FLOAT) {
        golden.resize(D_data.size(), 0.0f);
    } else if (dtypeD == ACL_BF16) {
        golden.resize(static_cast<size_t>(M) * p.ldd, 0.0f);
    }

    const float* cpuCPtr = (p.CIsNull) ? nullptr : C_ptr;
    std::vector<float> C_fp32_for_golden;

    aclblasStatus_t cpuRet = aclblasLtMatmul_cpu(
        cpuHandle, cpuComputeDesc, &p.alpha, A_ptr, cpuAdesc, B_ptr, cpuBdesc, &p.beta, cpuCPtr, cpuCdesc,
        golden.data(), cpuDdesc, scaleA_ptr, scaleB_ptr);

    EXPECT_EQ(cpuRet, ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatmulDescDestroy(cpuComputeDesc);
    aclblasLtMatrixLayoutDestroy(cpuAdesc);
    aclblasLtMatrixLayoutDestroy(cpuBdesc);
    aclblasLtMatrixLayoutDestroy(cpuCdesc);
    aclblasLtMatrixLayoutDestroy(cpuDdesc);

    // ── Step 5: Verify precision ──
    if (M == 0 || N == 0)
        return;

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;

    if (dtypeA == ACL_FLOAT && dtypeB == ACL_FLOAT && dtypeD == ACL_FLOAT) {
        cfg.mereThreshold = std::pow(2.0, -13);
        cfg.mareMultiplier = 10.0;
    } else if (isMxfpType(dtypeA) || isMxfpType(dtypeB)) {
        cfg.mereThreshold = std::pow(2.0, -7);
        cfg.mareMultiplier = 10.0;
    }

    if (dtypeD == ACL_BF16 && !D_bf16.empty()) {
        std::vector<float> D_fp32(static_cast<size_t>(M) * N);
        for (size_t i = 0; i < static_cast<size_t>(M) * N; i++) {
            D_fp32[i] = bf16ToFP32(D_bf16[i]);
        }
        EXPECT_TRUE(
            Verifier::verifyVector(D_fp32.data(), golden.data(), static_cast<size_t>(M) * N, 1, cfg, p.caseName));
    } else if (dtypeD == ACL_FLOAT && D_output_ptr != nullptr) {
        EXPECT_TRUE(Verifier::verifyVector(
            static_cast<const float*>(D_output_ptr), golden.data(), static_cast<size_t>(M) * N, 1, cfg, p.caseName));
    }
}

// issue #492: DEVICE 指针模式下 alpha/beta 为设备标量地址，Host 不得直接解引用，
// 须经 stream D2H 读回后参与计算（修复前会崩溃或读到错误标量）。
TEST_F(LtMatmulArch35Test, DevicePointerMode)
{
    constexpr int M = 16, N = 16, K = 16;
    constexpr int lda = 16, ldb = 16, ldd = 16;
    constexpr float alphaHost = 2.0f;
    constexpr float betaHost = 0.0f;

    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    int32_t transN = ACLBLAS_OP_N;
    aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &transN, sizeof(int32_t));
    aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &transN, sizeof(int32_t));
    // ACLBLASLT_POINTER_MODE_DEVICE = 1 (device scalar alpha/beta)
    int32_t pointerMode = 1;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(int32_t)),
        ACLBLAS_STATUS_SUCCESS);
    aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
    aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(uint32_t));

    aclblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    aclblasLtOrder_t order = ACLBLASLT_ORDER_ROW;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT, M, K, lda), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Bdesc, ACL_FLOAT, K, N, ldb), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Cdesc, ACL_FLOAT, M, N, ldd), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Ddesc, ACL_FLOAT, M, N, ldd), ACLBLAS_STATUS_SUCCESS);
    aclblasLtMatrixLayoutSetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutSetAttribute(Bdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutSetAttribute(Cdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));
    aclblasLtMatrixLayoutSetAttribute(Ddesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int32_t));

    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(M) * lda, "RANDOM_NORM_5_5", 42);
    std::vector<float> bHost = makeBlasArray(static_cast<int64_t>(K) * ldb, "RANDOM_NORM_5_5", 43);
    const size_t aBytes = aHost.size() * sizeof(float);
    const size_t bBytes = bHost.size() * sizeof(float);
    const size_t dBytes = static_cast<size_t>(M) * ldd * sizeof(float);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dD = nullptr;
    void* dAlpha = nullptr;
    void* dBeta = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dD, dBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dBeta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    // issue #603: H2D 失败时内核会在未初始化内存（含标量）上计算，与同文件 malloc 检查风格一致
    ASSERT_EQ(aclrtMemcpy(dA, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dB, bBytes, bHost.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dAlpha, sizeof(float), &alphaHost, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dBeta, sizeof(float), &betaHost, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    aclblasStatus_t ret = aclblasLtMatmul(
        ltHandle_, computeDesc, dAlpha, dA, Adesc, dB, Bdesc, dBeta, nullptr, Cdesc, dD, Ddesc, nullptr, nullptr, 0,
        stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // aclblasLtMatmul 只把 MMAD 与 epilogue 异步入队到 stream_ 就返回，而 aclrtMemcpy 没有 stream
    // 形参、无法与之建立 happens-before。本用例 rawStaged=false，dRawAddr 就是 dD：MMAD 先写入
    // A·B，epilogue 再原位覆盖成 alpha·(A·B)。不同步就读回会抢到中间态（恰为 golden 的 1/alpha），
    // 与 npu_wrapper.h 及 DevicePointerModeReadsScalarsOnDevice 的做法保持一致。
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    std::vector<float> dHost(static_cast<size_t>(M) * ldd, 0.0f);
    ASSERT_EQ(aclrtMemcpy(dHost.data(), dBytes, dD, dBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    std::vector<float> golden(static_cast<size_t>(M) * ldd, 0.0f);
    aclblasLtMatmul_cpu(
        ltHandle_, computeDesc, &alphaHost, aHost.data(), Adesc, bHost.data(), Bdesc, &betaHost, nullptr, Cdesc,
        golden.data(), Ddesc, nullptr, nullptr);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden.data(), golden.size());
    EXPECT_TRUE(Verifier::verifyVector(dHost.data(), golden.data(), dHost.size(), 1, cfg, "DevicePointerMode"));

    aclrtFree(dBeta);
    aclrtFree(dAlpha);
    aclrtFree(dD);
    aclrtFree(dB);
    aclrtFree(dA);
    aclblasLtMatrixLayoutDestroy(Ddesc);
    aclblasLtMatrixLayoutDestroy(Cdesc);
    aclblasLtMatrixLayoutDestroy(Bdesc);
    aclblasLtMatrixLayoutDestroy(Adesc);
    aclblasLtMatmulDescDestroy(computeDesc);
}

// issue #491: layout 描述符默认 ORDER_COL，而 Matmul kernel 仅按行主序解释数据。
// 未显式设置 ORDER_ROW 的布局必须显式拒绝（NOT_SUPPORTED），不得静默按行主序计算。
TEST_F(LtMatmulArch35Test, ColOrderNotSupported)
{
    constexpr int M = 4, N = 4, K = 4;

    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    int32_t transN = ACLBLAS_OP_N;
    aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &transN, sizeof(int32_t));
    aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &transN, sizeof(int32_t));

    aclblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT, M, K, K), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Bdesc, ACL_FLOAT, K, N, N), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Cdesc, ACL_FLOAT, M, N, N), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Ddesc, ACL_FLOAT, M, N, N), ACLBLAS_STATUS_SUCCESS);
    // 不调用 SetAttribute，保持默认 ACLBLASLT_ORDER_COL

    const size_t aBytes = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bBytes = static_cast<size_t>(K) * N * sizeof(float);
    const size_t dBytes = static_cast<size_t>(M) * N * sizeof(float);
    void* dA = nullptr;
    void* dB = nullptr;
    void* dD = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dD, dBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);

    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasLtMatmul(
        ltHandle_, computeDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, nullptr, Cdesc, dD, Ddesc, nullptr, nullptr, 0,
        stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_SUPPORTED);

    aclrtFree(dD);
    aclrtFree(dB);
    aclrtFree(dA);
    aclblasLtMatrixLayoutDestroy(Ddesc);
    aclblasLtMatrixLayoutDestroy(Cdesc);
    aclblasLtMatrixLayoutDestroy(Bdesc);
    aclblasLtMatrixLayoutDestroy(Adesc);
    aclblasLtMatmulDescDestroy(computeDesc);
}

namespace {

// 构造 4 个 ROW order 的 FP32 layout，供负向用例复用
void CreateRowOrderFloatLayouts(
    aclblasLtMatrixLayout_t (&layouts)[4], uint64_t rows, uint64_t cols, int64_t ld, aclDataType dType = ACL_FLOAT)
{
    const aclblasLtOrder_t row = ACLBLASLT_ORDER_ROW;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layouts[0], ACL_FLOAT, rows, cols, ld), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layouts[1], ACL_FLOAT, rows, cols, ld), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layouts[2], ACL_FLOAT, rows, cols, ld), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layouts[3], dType, rows, cols, ld), ACLBLAS_STATUS_SUCCESS);
    for (auto& layout : layouts) {
        ASSERT_EQ(
            aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)),
            ACLBLAS_STATUS_SUCCESS);
    }
}

void DestroyLayouts(aclblasLtMatrixLayout_t (&layouts)[4])
{
    for (auto& layout : layouts) {
        aclblasLtMatrixLayoutDestroy(layout);
    }
}

} // namespace

// issue #595: OP_C 无共轭实现，此前被静默按普通转置（OP_T 同路径）计算并返回 SUCCESS，
// 现在校验链显式拒绝。
TEST_F(LtMatmulArch35Test, OpCNotSupported)
{
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    const int32_t transC = ACLBLAS_OP_C;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &transC, sizeof(transC)),
        ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatrixLayout_t layouts[4]{};
    CreateRowOrderFloatLayouts(layouts, 4, 4, 4);
    float alpha = 1.0f, beta = 0.0f, data[16]{};
    EXPECT_EQ(
        aclblasLtMatmul(
            LtMatmulArch35Test::ltHandle_, computeDesc, &alpha, data, layouts[0], data, layouts[1], &beta, nullptr,
            layouts[2], data, layouts[3], nullptr, nullptr, 0, LtMatmulArch35Test::stream_),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    DestroyLayouts(layouts);
    aclblasLtMatmulDescDestroy(computeDesc);
}

// issue #594: scaleType 非 ACL_FLOAT 时 HOST 模式按 float 解引用 2 字节标量（主机越界读），
// DEVICE 模式 epilogue 也按 float 读取，校验链显式拒绝。
TEST_F(LtMatmulArch35Test, NonFloatScaleTypeRejected)
{
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT16), ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatrixLayout_t layouts[4]{};
    CreateRowOrderFloatLayouts(layouts, 4, 4, 4);
    const uint16_t alphaFp16 = 0x3C00; // 1.0 in FP16
    const uint16_t betaFp16 = 0;
    EXPECT_EQ(
        aclblasLtMatmul(
            LtMatmulArch35Test::ltHandle_, computeDesc, &alphaFp16, nullptr, layouts[0], nullptr, layouts[1], &betaFp16,
            nullptr, layouts[2], nullptr, layouts[3], nullptr, nullptr, 0, LtMatmulArch35Test::stream_),
        ACLBLAS_STATUS_INVALID_VALUE);

    DestroyLayouts(layouts);
    aclblasLtMatmulDescDestroy(computeDesc);
}

// issue #582: C/D dtype 超出内核支持白名单时不下发内核。HOST 模式 alpha=1/beta=0 无 epilogue，
// D=BF16 与 FP32 内核 4 字节写出宽度不一致，此前会越界写。
TEST_F(LtMatmulArch35Test, UnsupportedOutputDtypeRejected)
{
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    const int32_t opN = ACLBLAS_OP_N;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
        ACLBLAS_STATUS_SUCCESS);

    // FP32 A/B + D=BF16 + 平凡 alpha/beta（无 epilogue）
    {
        aclblasLtMatrixLayout_t layouts[4]{};
        CreateRowOrderFloatLayouts(layouts, 4, 4, 4, ACL_BF16);
        float alpha = 1.0f, beta = 0.0f, data[16]{};
        EXPECT_EQ(
            aclblasLtMatmul(
                LtMatmulArch35Test::ltHandle_, computeDesc, &alpha, data, layouts[0], data, layouts[1], &beta, nullptr,
                layouts[2], data, layouts[3], nullptr, nullptr, 0, LtMatmulArch35Test::stream_),
            ACLBLAS_STATUS_NOT_SUPPORTED);
        DestroyLayouts(layouts);
    }
    // C=INT8：epilogue 仅支持按 FP32/BF16 解释 C
    {
        aclblasLtMatrixLayout_t layouts[4]{};
        CreateRowOrderFloatLayouts(layouts, 4, 4, 4);
        aclblasLtMatrixLayoutDestroy(layouts[2]);
        ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layouts[2], ACL_INT8, 4, 4, 4), ACLBLAS_STATUS_SUCCESS);
        const aclblasLtOrder_t row = ACLBLASLT_ORDER_ROW;
        ASSERT_EQ(
            aclblasLtMatrixLayoutSetAttribute(layouts[2], ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)),
            ACLBLAS_STATUS_SUCCESS);
        float alpha = 2.0f, beta = 0.0f, data[16]{};
        EXPECT_EQ(
            aclblasLtMatmul(
                LtMatmulArch35Test::ltHandle_, computeDesc, &alpha, data, layouts[0], data, layouts[1], &beta, nullptr,
                layouts[2], data, layouts[3], nullptr, nullptr, 0, LtMatmulArch35Test::stream_),
            ACLBLAS_STATUS_NOT_SUPPORTED);
        DestroyLayouts(layouts);
    }
    aclblasLtMatmulDescDestroy(computeDesc);
}

// issue #583: C==D 原位且需要 epilogue 时，workspace 判空/容量校验前移到 MMAD 启动之前，
// workspace 为空时同步返回 INVALID_VALUE，不再对设备地址 0 发起写。
TEST_F(LtMatmulArch35Test, OverlapWorkspaceNullRejected)
{
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    const int32_t opN = ACLBLAS_OP_N;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
        ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatrixLayout_t layouts[4]{};
    CreateRowOrderFloatLayouts(layouts, 4, 4, 4);

    const size_t bytes = 4 * 4 * sizeof(float);
    void* dA = nullptr;
    void* dB = nullptr;
    void* dD = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dB, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dD, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);

    float alpha = 2.0f, beta = 0.0f;
    // C 与 D 为同一缓冲区（原位），workspace 为空
    EXPECT_EQ(
        aclblasLtMatmul(
            LtMatmulArch35Test::ltHandle_, computeDesc, &alpha, dA, layouts[0], dB, layouts[1], &beta, dD, layouts[2],
            dD, layouts[3], nullptr, nullptr, 0, LtMatmulArch35Test::stream_),
        ACLBLAS_STATUS_INVALID_VALUE);

    aclrtFree(dD);
    aclrtFree(dB);
    aclrtFree(dA);
    DestroyLayouts(layouts);
    aclblasLtMatmulDescDestroy(computeDesc);
}

// !691 检视意见修复验证：FP32 计算 + D=BF16 输出 + alpha!=1 + C!=D（非原位）组合——
// 引擎将 FP32 原始乘积 staging 到 workspace 后由 epilogue 按 BF16 转换写出，
// 修复前 FP32 MMAD 内核直接按 4 字节/元素写 2 字节 D 缓冲造成越界写。
TEST_F(LtMatmulArch35Test, Fp32ComputeBf16OutputMixedBuffers)
{
    constexpr int M = 16, N = 16, K = 16;
    constexpr int lda = K, ldb = N, ldd = N;
    constexpr float alphaHost = 2.0f;
    constexpr float betaHost = 0.0f;

    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    const int32_t transN = ACLBLAS_OP_N;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &transN, sizeof(transN)),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &transN, sizeof(transN)),
        ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    const aclblasLtOrder_t row = ACLBLASLT_ORDER_ROW;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT, M, K, lda), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Bdesc, ACL_FLOAT, K, N, ldb), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Cdesc, ACL_FLOAT, M, N, ldd), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&Ddesc, ACL_BF16, M, N, ldd), ACLBLAS_STATUS_SUCCESS);
    for (auto* layout : {Adesc, Bdesc, Cdesc, Ddesc}) {
        ASSERT_EQ(
            aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)),
            ACLBLAS_STATUS_SUCCESS);
    }

    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(M) * lda, "RANDOM_NORM_5_5", 42);
    std::vector<float> bHost = makeBlasArray(static_cast<int64_t>(K) * ldb, "RANDOM_NORM_5_5", 43);
    const size_t aBytes = aHost.size() * sizeof(float);
    const size_t bBytes = bHost.size() * sizeof(float);
    const size_t dBytes = static_cast<size_t>(M) * ldd * sizeof(uint16_t);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dD = nullptr;
    void* dWorkspace = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dD, dBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dWorkspace, 4 * 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dA, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dB, bBytes, bHost.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    std::vector<float> golden;
    golden.reserve(static_cast<size_t>(M) * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                acc += aHost[static_cast<size_t>(i) * lda + kk] * bHost[static_cast<size_t>(kk) * ldb + j];
            }
            golden.push_back(alphaHost * acc);
        }
    }

    float alpha = alphaHost, beta = betaHost;
    // C 与 D 为不同缓冲（C 传 nullptr，beta=0 不读 C）
    aclblasStatus_t ret = aclblasLtMatmul(
        ltHandle_, computeDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, nullptr, Cdesc, dD, Ddesc, nullptr, dWorkspace,
        4 * 1024 * 1024, stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // 同 DevicePointerMode：aclblasLtMatmul 异步入队，aclrtMemcpy 无 stream 形参，读回前必须显式同步。
    // 本用例 rawStaged=true（FP32 计算 / BF16 输出），dD 只由 epilogue 写，抢跑读到的是 dD 的旧内容
    // ——因每轮 aclrtFree/aclrtMalloc 常拿回同一地址、残留上一轮的正确结果，所以多数时候侥幸通过，
    // 只在 allocator 状态被整套件搅乱时暴露。
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    std::vector<uint16_t> dHost(static_cast<size_t>(M) * ldd, 0);
    ASSERT_EQ(aclrtMemcpy(dHost.data(), dBytes, dD, dBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    std::vector<float> dFp32(dHost.size());
    for (size_t idx = 0; idx < dHost.size(); ++idx) {
        dFp32[idx] = bf16ToFP32(dHost[idx]);
    }
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = std::pow(2.0, -7);
    cfg.mareMultiplier = 10.0;
    EXPECT_TRUE(Verifier::verifyVector(dFp32.data(), golden.data(), dFp32.size(), 1, cfg, "Fp32Bf16Mixed"));

    aclrtFree(dWorkspace);
    aclrtFree(dD);
    aclrtFree(dB);
    aclrtFree(dA);
    aclblasLtMatrixLayoutDestroy(Ddesc);
    aclblasLtMatrixLayoutDestroy(Cdesc);
    aclblasLtMatrixLayoutDestroy(Bdesc);
    aclblasLtMatrixLayoutDestroy(Adesc);
    aclblasLtMatmulDescDestroy(computeDesc);
}

// ── aclblasLtMatmulAlgoCheck / aclblasLtMatmulPreferenceInit 契约组 ──
// 覆盖：参数判空优先于 handle 检查、handle 单独为空、algo blob 结构校验、零维 no-op、
// HOST 模式下 FP32×FP32→BF16 的保守判定、行主序约束、共轭转置约束、
// workspaceSize 取 max(内在需求, algo 上限)、algo 原样回显、失败不回写 result，
// 以及 Preference 的 Init/Create 来源防护。

namespace {

// algo blob 里 magic 占 data[0:4]、algoId 占 data[4:8]，均小端。逐字节读写，避免裸 memcpy。
void SetAlgoBlobWord(aclblasLtMatmulAlgo_t& algo, size_t wordOffset, uint32_t value)
{
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        algo.data[wordOffset + i] = static_cast<uint8_t>((value >> (8 * i)) & 0xFFu);
    }
}

uint32_t GetAlgoBlobWord(const aclblasLtMatmulAlgo_t& algo, size_t wordOffset)
{
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        value |= static_cast<uint32_t>(algo.data[wordOffset + i]) << (8 * i);
    }
    return value;
}

// AlgoCheck 契约组公共夹具：16×16 FP32 行主序方阵 workload + 默认 algo 胶囊 + result 出参。
// 该组 12 个用例的「建 workload → 取默认 algo → 以全套有效参调 AlgoCheck」三步完全同构，
// 差异仅在 m/n/k、algo blob 篡改与 desc/layout 属性改写，故按 CQ.07 收敛到一处。
// handle 由调用方传入（夹具位于匿名命名空间，无法访问 fixture 的 protected 静态成员）。
// 两级前置条件保持各自独立断言（CreateWorkload / InitAlgo），失败可定位到具体阶段。
struct AlgoCheckFixture {
    MatmulLayoutTestData layouts;
    aclblasLtMatmulAlgo_t algo{};
    aclblasLtMatmulHeuristicResult_t result{};
    aclblasLtHandle_t handle = nullptr;

    // m/n/k 可覆盖以构造零维等边界
    bool CreateWorkload(uint64_t m = 16, uint64_t n = 16, uint64_t k = 16)
    {
        const uint64_t rows[] = {m, k, m, m};
        const uint64_t cols[] = {k, n, n, n};
        const int64_t ld[] = {
            static_cast<int64_t>(k), static_cast<int64_t>(n), static_cast<int64_t>(n), static_cast<int64_t>(n)};
        return layouts.Initialize(rows, cols, ld);
    }

    bool InitAlgo(aclblasLtHandle_t ltHandle)
    {
        handle = ltHandle;
        return MakeDefaultFp32Algo(ltHandle, &algo);
    }

    // 以夹具当前状态调用 AlgoCheck；out 重载供需要独立零初始化 result 的用例使用。
    aclblasStatus_t Run(aclblasLtMatmulHeuristicResult_t& out)
    {
        return aclblasLtMatmulAlgoCheck(
            handle, layouts.computeDesc, layouts.a, layouts.b, layouts.c, layouts.d, &algo, &out);
    }
    aclblasStatus_t Run() { return Run(result); }
};

} // namespace

TEST_F(LtMatmulArch35Test, AlgoCheckNullArgsPrecedeHandleCheck)
{
    // 参数形状检查优先于 handle 检查：handle 与描述符同时为空时返回 INVALID_VALUE，
    // 而非 aclblasLtMatmul 在同输入下返回的 NOT_INITIALIZED。
    aclblasLtMatmulAlgo_t algo{};
    aclblasLtMatmulHeuristicResult_t result{};
    EXPECT_EQ(
        aclblasLtMatmulAlgoCheck(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &algo, &result),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(LtMatmulArch35Test, AlgoCheckNullHandleAlone)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    fx.handle = nullptr; // 仅 handle 置空，其余 6 参均有效
    EXPECT_EQ(fx.Run(), ACLBLAS_STATUS_NOT_INITIALIZED);
}

TEST_F(LtMatmulArch35Test, AlgoCheckRejectsTamperedMagic)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    SetAlgoBlobWord(fx.algo, 0, GetAlgoBlobWord(fx.algo, 0) ^ 0xFFu); // 破坏 magic
    EXPECT_EQ(fx.Run(), ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(LtMatmulArch35Test, AlgoCheckRejectsUndecodableAlgoId)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    // 低 2 位为 0，落在 splitK 编码域 {1,2,3} 之外 → 不可解码
    SetAlgoBlobWord(fx.algo, 4, 0x00800000u);
    EXPECT_EQ(fx.Run(), ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(LtMatmulArch35Test, AlgoCheckZeroDimNoOpReportsZeroWorkspace)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload(/*m=*/0, /*n=*/16, /*k=*/16));
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    ASSERT_EQ(fx.Run(), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(fx.result.workspaceSize, 0u); // 空问题不报 epilogue 的固定附加项
    EXPECT_EQ(fx.result.state, ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatmulArch35Test, AlgoCheckRejectsColOrder)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    const aclblasLtOrder_t col = ACLBLASLT_ORDER_COL;
    ASSERT_EQ(
        aclblasLtMatrixLayoutSetAttribute(fx.layouts.a, ACLBLASLT_MATRIX_LAYOUT_ORDER, &col, sizeof(col)),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(fx.Run(), ACLBLAS_STATUS_NOT_SUPPORTED);
}

TEST_F(LtMatmulArch35Test, AlgoCheckRejectsConjugateTranspose)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    const int32_t opC = ACLBLAS_OP_C;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(fx.layouts.computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &opC, sizeof(opC)),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(fx.Run(), ACLBLAS_STATUS_NOT_SUPPORTED);
}

TEST_F(LtMatmulArch35Test, AlgoCheckWorkspaceIsMaxOfIntrinsicAndAlgoCap)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));

    // (a) 默认 algo：内在需求为 0（splitK=1、epilogue DEFAULT），报出的是 algo 自带的上限，
    //     使「按 result->workspaceSize 分配后直接调 aclblasLtMatmul」这条自然流程能通过其容量校验。
    aclblasLtMatmulHeuristicResult_t r1{};
    ASSERT_EQ(fx.Run(r1), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(r1.workspaceSize, fx.algo.max_workspace_bytes);
    EXPECT_GT(r1.workspaceSize, 0u);

    // (b) 把上限压到 100 并令 splitK=2 → 内在需求 2*16*16*4 = 2048 > 100，报内在需求。
    fx.algo.max_workspace_bytes = 100;
    const uint32_t splitK = 2;
    ASSERT_EQ(
        aclblasLtMatmulAlgoConfigSetAttribute(&fx.algo, ACLBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK)),
        ACLBLAS_STATUS_SUCCESS);
    aclblasLtMatmulHeuristicResult_t r2{};
    ASSERT_EQ(fx.Run(r2), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(r2.workspaceSize, 2u * 16u * 16u * sizeof(float));
}

TEST_F(LtMatmulArch35Test, AlgoCheckEchoesAlgoUnchanged)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    const aclblasLtMatmulAlgo_t before = fx.algo;
    ASSERT_EQ(fx.Run(), ACLBLAS_STATUS_SUCCESS);
    // 回显必须逐字节相同（含 max_workspace_bytes），调用方可安全复用或再序列化原 algo。
    for (size_t i = 0; i < sizeof(before.data); ++i) {
        ASSERT_EQ(fx.result.algo.data[i], before.data[i]) << "algo.data[" << i << "] differs";
    }
    EXPECT_EQ(fx.result.algo.max_workspace_bytes, before.max_workspace_bytes);
    EXPECT_EQ(fx.result.wavesCount, 1.0f); // 占位值：本接口不做设备查询，不可用于排序
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(fx.result.reserved[i], 0);
    }
}

TEST_F(LtMatmulArch35Test, AlgoCheckFailureLeavesResultUntouched)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    SetAlgoBlobWord(fx.algo, 0, GetAlgoBlobWord(fx.algo, 0) ^ 0xFFu); // 令其在 Stage 2 失败

    aclblasLtMatmulHeuristicResult_t result{};
    result.workspaceSize = 0xDEADBEEFu;
    result.wavesCount = -1.0f;
    result.algo.max_workspace_bytes = 0xDEADBEEFu;
    for (auto& byte : result.algo.data) {
        byte = 0xDEu;
    }
    for (int i = 0; i < 4; ++i) {
        result.reserved[i] = 0x5A5A5A5A;
    }

    EXPECT_EQ(fx.Run(result), ACLBLAS_STATUS_INVALID_VALUE);
    // 失败路径不得回写 result：调用方只能按返回值判定，不能读 result.state
    // （其默认成员初始化值恰为 SUCCESS，失败后读它会误判为成功）。
    EXPECT_EQ(result.workspaceSize, 0xDEADBEEFu);
    EXPECT_EQ(result.wavesCount, -1.0f);
    EXPECT_EQ(result.algo.max_workspace_bytes, 0xDEADBEEFu);
    for (size_t i = 0; i < sizeof(result.algo.data); ++i) {
        EXPECT_EQ(result.algo.data[i], 0xDEu) << "algo.data[" << i << "] was written on failure";
    }
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result.reserved[i], 0x5A5A5A5A);
    }
}

TEST_F(LtMatmulArch35Test, AlgoCheckHostFp32WithBf16OutputNotSupported)
{
    // HOST 指针模式下 alpha/beta 对本接口不可见，无法复现 aclblasLtMatmul 的运行时
    // epilogue 判定，故按「epilogue 可能不运行」的保守判定只允许 D=FP32。
    // 这是方向保守的已知假阴性：alpha!=1 或 beta!=0 时 aclblasLtMatmul 实际可跑。
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    // A/B 为 FP32、C/D 为 BF16：dtype 组合逐位不同，故不用 CreateRowOrderFloatLayouts（其仅 D 可变）
    aclblasLtMatrixLayout_t layouts[4]{};
    const aclDataType layoutTypes[4] = {ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_BF16};
    const aclblasLtOrder_t row = ACLBLASLT_ORDER_ROW;
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layouts[i], layoutTypes[i], 16, 16, 16), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasLtMatrixLayoutSetAttribute(layouts[i], ACLBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)),
            ACLBLAS_STATUS_SUCCESS);
    }
    aclblasLtMatmulAlgo_t algo{};
    ASSERT_TRUE(MakeDefaultFp32Algo(LtMatmulArch35Test::ltHandle_, &algo));
    aclblasLtMatmulHeuristicResult_t result{};
    EXPECT_EQ(
        aclblasLtMatmulAlgoCheck(
            LtMatmulArch35Test::ltHandle_, computeDesc, layouts[0], layouts[1], layouts[2], layouts[3], &algo, &result),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    DestroyLayouts(layouts);
    aclblasLtMatmulDescDestroy(computeDesc);
}

// ── AlgoCheck 契约补全（缺口 G8，评估报告 P2-3）──
// 真值源：blasLt/api/aclblaslt_matmul_algo.cpp —— Stage 0 AlgoCheckArgsAreNonNull 逐参检查
// 7 个指针并在日志指名首个空参；CheckScalarSemantics 按 pointerMode → scaleType → transpose →
// epilogue 顺序校验：scaleType 非 ACL_FLOAT 返回 INVALID_VALUE（本接口只解析 FP32 host 标量），
// epilogue 非 DEFAULT 返回 NOT_SUPPORTED（issue #801 第三入口，与 Matmul 入口/heuristic 同口径）。

// 单参 NULL 矩阵：handle 与其余 6 参均有效，逐一置空 7 个指针参数 → INVALID_VALUE。
// 此前仅钉「全 NULL」组合（AlgoCheckNullArgsPrecedeHandleCheck）与 handle 单独为空
// （AlgoCheckNullHandleAlone），此处补齐逐参路径（Stage 0 校验顺序契约的另一半）。
// 注意 Cdesc 为空同样被拒——AlgoCheck 比 aclblasLtMatmul（beta=0 时允许 C=NULL）更严格。
TEST_F(LtMatmulArch35Test, AlgoCheckRejectsEachSingleNullArgument)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    const MatmulLayoutTestData& lay = fx.layouts;
    const aclblasLtMatmulDesc_t desc = lay.computeDesc;
    auto invoke = [&](aclblasLtMatmulDesc_t descArg, aclblasLtMatrixLayout_t aArg, aclblasLtMatrixLayout_t bArg,
                      aclblasLtMatrixLayout_t cArg, aclblasLtMatrixLayout_t dArg, const aclblasLtMatmulAlgo_t* algoPtr,
                      aclblasLtMatmulHeuristicResult_t* resultPtr) {
        return aclblasLtMatmulAlgoCheck(
            LtMatmulArch35Test::ltHandle_, descArg, aArg, bArg, cArg, dArg, algoPtr, resultPtr);
    };
    // 逐参置空、其余 6 参恒有效；行尾标注被置空的形参名
    EXPECT_EQ(invoke(desc, lay.a, lay.b, lay.c, lay.d, &fx.algo, nullptr), ACLBLAS_STATUS_INVALID_VALUE);   // result
    EXPECT_EQ(invoke(desc, lay.a, lay.b, lay.c, lay.d, nullptr, &fx.result), ACLBLAS_STATUS_INVALID_VALUE); // algo
    EXPECT_EQ(invoke(nullptr, lay.a, lay.b, lay.c, lay.d, &fx.algo, &fx.result), ACLBLAS_STATUS_INVALID_VALUE); // desc
    EXPECT_EQ(invoke(desc, nullptr, lay.b, lay.c, lay.d, &fx.algo, &fx.result), ACLBLAS_STATUS_INVALID_VALUE);  // Adesc
    EXPECT_EQ(invoke(desc, lay.a, nullptr, lay.c, lay.d, &fx.algo, &fx.result), ACLBLAS_STATUS_INVALID_VALUE);  // Bdesc
    EXPECT_EQ(invoke(desc, lay.a, lay.b, nullptr, lay.d, &fx.algo, &fx.result), ACLBLAS_STATUS_INVALID_VALUE);  // Cdesc
    EXPECT_EQ(invoke(desc, lay.a, lay.b, lay.c, nullptr, &fx.algo, &fx.result), ACLBLAS_STATUS_INVALID_VALUE);  // Ddesc
}

// issue #801 第三入口：Matmul 入口与 GetHeuristic 已由 NonDefaultEpilogueRejected 钉住，
// AlgoCheck 的 CheckScalarSemantics 对非 DEFAULT epilogue 同口径 NOT_SUPPORTED——
// 「可校验的必可执行」三处入口一致。该校验先于维度读取与 dtype/order 检查，失败不回写 result。
TEST_F(LtMatmulArch35Test, AlgoCheckRejectsNonDefaultEpilogue)
{
    AlgoCheckFixture fx;
    ASSERT_TRUE(fx.CreateWorkload());
    ASSERT_TRUE(fx.InitAlgo(LtMatmulArch35Test::ltHandle_));
    const int32_t epilogueBias = ACLBLASLT_EPILOGUE_BIAS;
    ASSERT_EQ(
        aclblasLtMatmulDescSetAttribute(
            fx.layouts.computeDesc, ACLBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(fx.Run(), ACLBLAS_STATUS_NOT_SUPPORTED);
}

// CheckScalarSemantics 的 scaleType 守卫（issue #594 同源语义）：Matmul 侧已由
// NonFloatScaleTypeRejected 钉住，AlgoCheck 侧此前无对应。DescCreate 不校验 scaleType
// 入参，FP16 scaleType 的 desc 构造成功、在 AlgoCheck 标量语义检查处拒绝（INVALID_VALUE），
// 该校验先于 transpose/epilogue 与维度/dtype 检查。
TEST_F(LtMatmulArch35Test, AlgoCheckRejectsNonFloatScaleType)
{
    aclblasLtMatmulDesc_t computeDesc = nullptr;
    ASSERT_EQ(aclblasLtMatmulDescCreate(&computeDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT16), ACLBLAS_STATUS_SUCCESS);
    aclblasLtMatrixLayout_t layouts[4]{};
    CreateRowOrderFloatLayouts(layouts, 16, 16, 16);
    aclblasLtMatmulAlgo_t algo{};
    ASSERT_TRUE(MakeDefaultFp32Algo(LtMatmulArch35Test::ltHandle_, &algo));
    aclblasLtMatmulHeuristicResult_t result{};
    EXPECT_EQ(
        aclblasLtMatmulAlgoCheck(
            LtMatmulArch35Test::ltHandle_, computeDesc, layouts[0], layouts[1], layouts[2], layouts[3], &algo, &result),
        ACLBLAS_STATUS_INVALID_VALUE);
    DestroyLayouts(layouts);
    aclblasLtMatmulDescDestroy(computeDesc);
}

TEST_F(LtMatmulArch35Test, PreferenceInitProductRejectedByDestroy)
{
    aclblasLtMatmulPreferenceOpaque_t storage{};
    ASSERT_EQ(aclblasLtMatmulPreferenceInit(&storage), ACLBLAS_STATUS_SUCCESS);
    // Init 产物位于调用方存储，Destroy 必须拒绝而不是 delete 一个非堆指针。
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(&storage), ACLBLAS_STATUS_INVALID_VALUE);
}

// 契约配套用例：公开契约要求调用方存储入口为确定值（零初始化，或 Create/Init 产物胶囊），
// Init 读回首部字节以保留 Create 产物的堆来源位。此处故意预填契约外的**确定**垃圾字节
// （确定值，非 UB），验证 magic 门控对任何非胶囊内容都路由到纯重置路径：heapOwned=0、
// Destroy 照旧拒绝——「残留字节被误标堆来源」须恰好命中 32 位 magic 且 heapOwned 非零，
// 而该输入本身已属违约。
TEST_F(LtMatmulArch35Test, PreferenceInitOnNonCapsuleBytesCarriesNoProvenance)
{
    aclblasLtMatmulPreferenceOpaque_t storage{};
    for (auto& word : storage.data) {
        word = 0xAAAAAAAAAAAAAAAAULL; // 确定的非零垃圾：magic 必不匹配，亦非 Create 产物
    }
    ASSERT_EQ(aclblasLtMatmulPreferenceInit(&storage), ACLBLAS_STATUS_SUCCESS);
    // 纯重置路径：来源位为 0，Destroy 拒绝，绝不 delete 非堆指针。
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(&storage), ACLBLAS_STATUS_INVALID_VALUE);
}

// cuBLASLt 的 Init 语义是「重置一个先前已分配的描述符」，Create→Init（重置复用）→Destroy 属
// 合法模式。此前 Init 无条件以 NSDMI 全新 impl 覆写胶囊，把 heapOwned 从 1 翻回 0，Destroy 的
// 来源防护随后永久拒绝该堆胶囊——既泄漏又给出意外 INVALID_VALUE。现在 Init 保留来源位、只重置属性。
TEST_F(LtMatmulArch35Test, PreferenceInitOnCreateProductKeepsDestroyable)
{
    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);
    ASSERT_NE(pref, nullptr);
    const uint64_t ws = 4096;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)),
        ACLBLAS_STATUS_SUCCESS);

    ASSERT_EQ(aclblasLtMatmulPreferenceInit(pref), ACLBLAS_STATUS_SUCCESS);

    // 属性确已回到默认值（重置生效），而非保留 Set 进去的 4096。
    uint64_t wsOut = 0;
    uint32_t modeOut = 0xDEADu;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsOut, sizeof(wsOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wsOut, static_cast<uint64_t>(32) * 1024 * 1024);
    ASSERT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            pref, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &modeOut, sizeof(modeOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeOut, 0u);

    // 来源位保留 → Destroy 正常释放，Create→Init→Destroy 不再泄漏。
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatmulArch35Test, PreferenceCreateProductSurvivesSetAttributeThenDestroy)
{
    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);
    ASSERT_NE(pref, nullptr);
    const uint64_t ws = 1024;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceSetAttribute(pref, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)),
        ACLBLAS_STATUS_SUCCESS);
    // 来源标记必须能穿过 SetAttribute 的「整体读出—改一项—整体写回」，否则此路径会被误拒。
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatmulArch35Test, PreferenceInitCapsuleIsDeterministic)
{
    // 两次 Init 到预填不同垃圾的存储，产物应逐字节相同（值初始化 + 尾部清零）。
    // 预填垃圾为契约外的确定值输入（确定字节，非 UB）：magic 门控将其路由到纯重置路径，
    // 产物与预填内容无关；来源位不被误授予的断言见
    // PreferenceInitOnNonCapsuleBytesCarriesNoProvenance。
    aclblasLtMatmulPreferenceOpaque_t lhs{}, rhs{};
    for (auto& word : lhs.data) {
        word = 0xAAAAAAAAAAAAAAAAULL;
    }
    for (auto& word : rhs.data) {
        word = 0x5555555555555555ULL;
    }
    ASSERT_EQ(aclblasLtMatmulPreferenceInit(&lhs), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatmulPreferenceInit(&rhs), ACLBLAS_STATUS_SUCCESS);
    for (size_t i = 0; i < sizeof(lhs.data) / sizeof(lhs.data[0]); ++i) {
        EXPECT_EQ(lhs.data[i], rhs.data[i]) << "capsule word " << i << " differs";
    }
}

TEST_F(LtMatmulArch35Test, PreferenceInitDefaultsMatchCreate)
{
    aclblasLtMatmulPreferenceOpaque_t storage{};
    ASSERT_EQ(aclblasLtMatmulPreferenceInit(&storage), ACLBLAS_STATUS_SUCCESS);
    aclblasLtMatmulPreference_t heap = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&heap), ACLBLAS_STATUS_SUCCESS);

    uint32_t modeInit = 0xDEADu, modeCreate = 0xDEADu;
    uint64_t wsInit = 0, wsCreate = 0;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            &storage, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &modeInit, sizeof(modeInit), &written),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            heap, ACLBLASLT_MATMUL_PREF_SEARCH_MODE, &modeCreate, sizeof(modeCreate), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeInit, modeCreate);
    ASSERT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            &storage, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsInit, sizeof(wsInit), &written),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatmulPreferenceGetAttribute(
            heap, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsCreate, sizeof(wsCreate), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wsInit, wsCreate);

    ASSERT_EQ(aclblasLtMatmulPreferenceDestroy(heap), ACLBLAS_STATUS_SUCCESS);
}

// ── Preference 族对称化（缺口 G10，评估报告 P2-6）──
// TransformDesc 族已有 InitRejectsNull / DestroyRejectsCorruptedMagic 三项 counterpart，
// Preference 族此前缺失对应钉住。胶囊布局真值源（aclblaslt_layout_impl.h）：magic 字节 0~3、
// heapOwned 字节 28~31（即 data[3] 高 4 字节，本仓构建目标均为小端）；data[0] ^= 0xFF
// 仅破坏 magic 低字节，heapOwned 不受影响。

// Init(NULL) / Destroy(NULL) → INVALID_VALUE（preference.cpp 的显式判空分支；
// TransformDescInitRejectsNull 已有同族 counterpart，此处补齐 Preference 侧）。
TEST_F(LtMatmulArch35Test, PreferenceInitDestroyRejectNull)
{
    EXPECT_EQ(aclblasLtMatmulPreferenceInit(nullptr), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(nullptr), ACLBLAS_STATUS_INVALID_VALUE);
}

// Preference 损坏胶囊行为——钉实现真相，不照搬 TransformDesc 的拒绝形态（期望值以实现取证
// 为准）。取证：preference.cpp 的 Destroy/Set/Get 均无 magic 守卫（实现注释明示 "No magic
// check here, unlike aclblasLtMatrixTransformDescDestroy"），Destroy 的唯一胶囊级守卫是
// heapOwned 来源位；header/api_list 亦只披露 NULL + 来源防护两条防御。故：
// (a) 损坏 magic 的 Create 产物仍按来源位正常释放（SUCCESS）——与 TransformDescDestroy 的
//     magic 拒绝为有意不对称，钉住该事实，未来增删守卫须有意识地更新本用例与文档；
// (b) 来源位清零（magic 完好）才使 Destroy 拒绝（INVALID_VALUE），复原后正常释放——循
//     TransformDescDestroyRejectsCorruptedMagic 的「损坏→拒绝→复原→释放」防泄漏形态。
TEST_F(LtMatmulArch35Test, PreferenceDestroyGuardsProvenanceNotMagic)
{
    // (a) magic 损坏不触发 Destroy 拒绝（无 magic 守卫），胶囊按 heapOwned=1 正常释放
    aclblasLtMatmulPreference_t magicBroken = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&magicBroken), ACLBLAS_STATUS_SUCCESS);
    auto* brokenCapsule = reinterpret_cast<aclblasLtMatmulPreferenceOpaque_t*>(magicBroken);
    brokenCapsule->data[0] ^= 0xFFu; // 破坏 magic 低字节；heapOwned（字节 28~31）不受影响
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(magicBroken), ACLBLAS_STATUS_SUCCESS);

    // (b) 来源位是 Destroy 的唯一胶囊级守卫：清零即拒绝，复原后正常释放（无泄漏）
    aclblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(aclblasLtMatmulPreferenceCreate(&pref), ACLBLAS_STATUS_SUCCESS);
    auto* capsule = reinterpret_cast<aclblasLtMatmulPreferenceOpaque_t*>(pref);
    const uint64_t savedWord = capsule->data[3];  // 字节 24~31：preferredL0K（低 4）+ heapOwned（高 4）
    capsule->data[3] = savedWord & 0xFFFFFFFFULL; // 清零 heapOwned，magic 与其余字段保持完好
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_INVALID_VALUE);
    capsule->data[3] = savedWord; // 复原来源位，正常释放，避免测试泄漏
    EXPECT_EQ(aclblasLtMatmulPreferenceDestroy(pref), ACLBLAS_STATUS_SUCCESS);
}
