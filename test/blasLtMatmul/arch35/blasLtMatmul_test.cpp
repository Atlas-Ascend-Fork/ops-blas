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

} // namespace

// ── LtMatmul test fixture using aclblasLtHandle_t ──
class LtMatmulArch35Test : public ::testing::TestWithParam<LtMatmulParam> {
protected:
    static aclblasLtHandle_t ltHandle_;
    static aclrtStream stream_;

    static void SetUpTestSuite()
    {
        aclError initRet = aclInit(nullptr);
        ASSERT_TRUE(initRet == ACL_SUCCESS || initRet == ACL_ERROR_REPEAT_INITIALIZE)
            << "aclInit failed: " << initRet;
        ASSERT_EQ(aclrtSetDevice(TEST_DEVICE_ID), ACL_SUCCESS);
        ASSERT_EQ(aclblasLtCreate(&ltHandle_), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&stream_), ACL_SUCCESS);
    }

    static void TearDownTestSuite()
    {
        if (stream_ != nullptr) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        if (ltHandle_ != nullptr) { aclblasLtDestroy(ltHandle_); ltHandle_ = nullptr; }
        aclrtResetDevice(TEST_DEVICE_ID);
        aclFinalize();
    }
};

aclblasLtHandle_t LtMatmulArch35Test::ltHandle_ = nullptr;
aclrtStream LtMatmulArch35Test::stream_ = nullptr;

// ── TEST_F: null handle (TC_L0_25) ──
TEST_F(LtMatmulArch35Test, NullHandle) {
    aclblasStatus_t ret = aclblasLtMatmul_npu(
        nullptr, nullptr,
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        16, 16, 16,
        ACLBLAS_OP_N, ACLBLAS_OP_N,
        16, 16, 16, 16,
        1.0f, 0.0f,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, "default", false,
        true, false, false, false);  // handleNull=true
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

// ── TEST_F: null computeDesc (TC_L0_26) ──
TEST_F(LtMatmulArch35Test, NullComputeDesc) {
    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasLtMatmul(
        LtMatmulArch35Test::ltHandle_, nullptr,
        &alpha, nullptr, nullptr, nullptr, nullptr,
        &beta, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, 0, LtMatmulArch35Test::stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// ── TEST_F: alpha=nullptr (TC_L0_27) ──
TEST_F(LtMatmulArch35Test, NullAlpha) {
    float beta = 0.0f;
    aclblasLtMatmulDesc_t desc;
    aclblasLtMatmulDescCreate(&desc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);
    aclblasStatus_t ret = aclblasLtMatmul(
        LtMatmulArch35Test::ltHandle_, desc,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        &beta, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, 0, LtMatmulArch35Test::stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
    aclblasLtMatmulDescDestroy(desc);
}

// ── TEST_F: A=nullptr (TC_L0_28) ──
TEST_F(LtMatmulArch35Test, NullA) {
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
        handle, computeDesc,
        &alpha, nullptr, Adesc, nullptr, Bdesc,
        &beta, nullptr, Cdesc, nullptr, Ddesc,
        nullptr, nullptr, 0, stream);

    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);

    aclblasLtMatmulDescDestroy(computeDesc);
    aclblasLtMatrixLayoutDestroy(Adesc);
    aclblasLtMatrixLayoutDestroy(Bdesc);
    aclblasLtMatrixLayoutDestroy(Cdesc);
    aclblasLtMatrixLayoutDestroy(Ddesc);
}

// ── CSV parameterized tests ──
INSTANTIATE_TEST_SUITE_P(
    LtMatmul, LtMatmulArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<LtMatmulParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<LtMatmulParam>);

TEST_P(LtMatmulArch35Test, CsvDriven) {
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
        for (auto& b : A_mxfp) b = static_cast<uint8_t>(mxfp8Dist(rng));
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
        for (auto& b : B_mxfp) b = static_cast<uint8_t>(mxfp8Dist(rng));
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
            for (auto& b : scaleA_data) b = 0;
        } else {
            for (auto& b : scaleA_data) b = static_cast<uint8_t>(scale8Dist(rng));
        }
    }
    if (isMxfpType(dtypeB)) {
        size_t scaleBBytes = mxScaleBufferBytesB(N, K, p.transB);
        scaleB_data.resize(scaleBBytes);
        if (p.scaleBFill.method == BlasFillMode::M_VALUE && p.scaleBFill.val1 == 0.0f) {
            for (auto& b : scaleB_data) b = 0;
        } else {
            for (auto& b : scaleB_data) b = static_cast<uint8_t>(scale8Dist(rng));
        }
    }

    // ── Determine data pointers ──
    const void* A_ptr = nullptr;
    const void* B_ptr = nullptr;
    const float* C_ptr = nullptr;
    const void* scaleA_ptr = nullptr;
    const void* scaleB_ptr = nullptr;

    if (dtypeA == ACL_FLOAT && !A_data.empty()) A_ptr = A_data.data();
    else if (isMxfpType(dtypeA) && !A_mxfp.empty()) A_ptr = A_mxfp.data();

    if (dtypeB == ACL_FLOAT && !B_data.empty()) B_ptr = B_data.data();
    else if (isMxfpType(dtypeB) && !B_mxfp.empty()) B_ptr = B_mxfp.data();

    if (!p.CIsNull && !C_data.empty()) C_ptr = C_data.data();
    if (isMxfpType(dtypeA) && !scaleA_data.empty()) scaleA_ptr = scaleA_data.data();
    if (isMxfpType(dtypeB) && !scaleB_data.empty()) scaleB_ptr = scaleB_data.data();

    void* D_output_ptr = nullptr;
    if (dtypeD == ACL_FLOAT) D_output_ptr = D_data.data();
    else if (dtypeD == ACL_BF16) D_output_ptr = D_bf16.data();

    // ── Step 2: Execute on NPU ──
    // Pass nullptr ltHandle when handleNull=true (TC_L0_25 CSV parity with TEST_F)
    aclblasLtHandle_t ltHandleForNpu = p.handleNull ? nullptr : LtMatmulArch35Test::ltHandle_;
    aclblasStatus_t ret = aclblasLtMatmul_npu(
        ltHandleForNpu, LtMatmulArch35Test::stream_,
        dtypeA, dtypeB, dtypeC, dtypeD,
        M, N, K,
        p.transA, p.transB,
        p.lda, p.ldb, p.ldc, p.ldd,
        p.alpha, p.beta,
        A_ptr, B_ptr, C_ptr, D_output_ptr,
        scaleA_ptr, scaleB_ptr,
        p.algoMode, p.CIsNull,
        p.handleNull, p.computeDescNull, p.alphaNull, p.Anull);

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
        cpuHandle, cpuComputeDesc,
        &p.alpha, A_ptr, cpuAdesc,
        B_ptr, cpuBdesc,
        &p.beta, cpuCPtr, cpuCdesc,
        golden.data(), cpuDdesc,
        scaleA_ptr, scaleB_ptr);

    EXPECT_EQ(cpuRet, ACLBLAS_STATUS_SUCCESS);

    aclblasLtMatmulDescDestroy(cpuComputeDesc);
    aclblasLtMatrixLayoutDestroy(cpuAdesc);
    aclblasLtMatrixLayoutDestroy(cpuBdesc);
    aclblasLtMatrixLayoutDestroy(cpuCdesc);
    aclblasLtMatrixLayoutDestroy(cpuDdesc);

    // ── Step 5: Verify precision ──
    if (M == 0 || N == 0) return;

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
        EXPECT_TRUE(Verifier::verifyVector(D_fp32.data(), golden.data(),
            static_cast<size_t>(M) * N, 1, cfg, p.caseName));
    } else if (dtypeD == ACL_FLOAT && D_output_ptr != nullptr) {
        EXPECT_TRUE(Verifier::verifyVector(static_cast<const float*>(D_output_ptr), golden.data(),
            static_cast<size_t>(M) * N, 1, cfg, p.caseName));
    }
}

// issue #492: DEVICE 指针模式下 alpha/beta 为设备标量地址，Host 不得直接解引用，
// 须经 stream D2H 读回后参与计算（修复前会崩溃或读到错误标量）。
TEST_F(LtMatmulArch35Test, DevicePointerMode) {
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