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

// ── TEST_F: null handle (TC_L0_25) ──
TEST_F(LtMatmulArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasLtMatmul_npu(
        nullptr, nullptr, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 16, 16, 16, ACLBLAS_OP_N, ACLBLAS_OP_N, 16, 16,
        16, 16, 1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "default", false, true, false, false,
        false); // handleNull=true
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
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
    aclrtMemcpy(dA, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dB, bBytes, bHost.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dAlpha, sizeof(float), &alphaHost, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dBeta, sizeof(float), &betaHost, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t ret = aclblasLtMatmul(
        ltHandle_, computeDesc, dAlpha, dA, Adesc, dB, Bdesc, dBeta, nullptr, Cdesc, dD, Ddesc, nullptr, nullptr, 0,
        stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<float> dHost(static_cast<size_t>(M) * ldd, 0.0f);
    aclrtMemcpy(dHost.data(), dBytes, dD, dBytes, ACL_MEMCPY_DEVICE_TO_HOST);

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
