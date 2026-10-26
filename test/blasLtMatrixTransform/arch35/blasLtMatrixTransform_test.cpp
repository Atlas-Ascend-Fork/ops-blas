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

#include "verify.h"
#include "csv_loader.h"
#include "blasLtMatrixTransform_param.h"
#include "blasLtMatrixTransform_golden.h"
#include "blasLtMatrixTransform_npu_wrapper.h"

// ── fixture using aclblasLtHandle_t (mirrors blasLtMatmul) ──
class LtMatrixTransformArch35Test : public ::testing::TestWithParam<LtMatrixTransformParam> {
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

aclblasLtHandle_t LtMatrixTransformArch35Test::ltHandle_ = nullptr;
aclrtStream LtMatrixTransformArch35Test::stream_ = nullptr;

// ── build a physical float buffer for a matrix (logical op-applied dims rows×cols, order) ──
// Physical pre-op dims derived inside; fills via fill.h public functions only.
static std::vector<float> buildPhys(
    aclDataType dt, aclblasLtOrder_t order, aclblasOperation_t op, int opRows, int opCols, int ld,
    const std::string& desc, uint32_t seed)
{
    int physRows, physCols;
    ltPhysDims(opRows, opCols, op, physRows, physCols);
    int64_t count = ltTransformPhysCount(order, physRows, physCols, ld);
    if (count <= 0)
        return {};

    const bool wantZeros = desc.find("zero") != std::string::npos;

    if (dt == ACL_INT8) {
        if (wantZeros)
            return std::vector<float>(static_cast<size_t>(count), 0.0f);
        // extreme ±127 probe for saturation-edge cases; smaller range otherwise to avoid
        // unintended INT32->INT8 saturation masking the layout check.
        if (desc.find("extr") != std::string::npos || desc.find("saturat") != std::string::npos ||
            desc.find("127") != std::string::npos)
            return makeBlasIntExtreme(count, -128, 127);
        return makeBlasIntArray(count, -8, 8, seed);
    }
    if (dt == ACL_INT32) {
        if (wantZeros)
            return std::vector<float>(static_cast<size_t>(count), 0.0f);
        return makeBlasIntArray(count, -1000, 1000, seed);
    }
    // FP8 path (E4M3FN / E5M2): quantization-level fill for RINT special-value cases, else a level
    // set keeps quantization noise bounded so layout/permutation bugs stay visible under the
    // single-benchmark threshold.
    if (dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2) {
        if (wantZeros)
            return std::vector<float>(static_cast<size_t>(count), 0.0f);
        return makeBlasFp8Levels(count, dt == ACL_FLOAT8_E5M2, seed);
    }
    // FP4 path (E2M1): always draw from the representable lattice (only 8 magnitudes); level fill
    // for the RINT special-value case, random-lattice otherwise.
    if (dt == ACL_FLOAT4_E2M1) {
        if (wantZeros)
            return std::vector<float>(static_cast<size_t>(count), 0.0f);
        if (desc.find("quant") != std::string::npos || desc.find("level") != std::string::npos)
            return makeBlasFp4Levels(count, seed);
        return makeBlasFp4Random(count, seed);
    }
    // floating-point path
    if (wantZeros)
        return makeBlasArray(count, "VALUE_0", seed);
    if (desc.find("small") != std::string::npos)
        return makeBlasArray(count, "RANDOM_1EN3", seed); // uniform [-1e-3, 1e-3]
    return makeBlasArray(count, "RANDOM_2", seed);        // uniform [-2, 2]
}

// ── TEST_F: null handle (TC_L0_17) ──
TEST_F(LtMatrixTransformArch35Test, NullHandle)
{
    LtMatrixTransformParam p((csv_map{}));
    p.handleNull = true;
    p.rowsA = p.colsA = p.rowsC = p.colsC = 16;
    p.lda = p.ldc = 16;
    std::vector<float> dummyA, dummyB, devNdC;
    aclblasStatus_t ret = aclblasLtMatrixTransform_npu(nullptr, stream_, p, dummyA, dummyB, devNdC);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

// ── I-01 anchor tests: complex-layout permutation correctness, INDEPENDENT of the device. ──
// Each builds a single composite tile of sequential logical values, runs golden's re-layout
// (which uses the test-local perm derivation), and byte-compares the physical buffer against a
// hand-verified expected placement derived directly from the order geometry. These hand-verified
// offsets are the independent "who is right" anchor required by review item I-01 ②: they pin the
// golden's permutation to a third source, so a golden↔device divergence cannot pass as a dual error.
TEST_F(LtMatrixTransformArch35Test, AnchorCol4_4R2_8C_SingleTile)
{
    // single 8-row × 32-col tile; logical ND value at (row,col) = col*8 + row + 1 (col-major seq).
    const int rows = 8, cols = 32;
    std::vector<float> nd(static_cast<size_t>(rows) * cols);
    for (int c = 0; c < cols; c++)
        for (int r = 0; r < rows; r++)
            nd[static_cast<size_t>(c) * rows + r] = static_cast<float>(c * rows + r + 1);

    int ld = static_cast<int>(MT_COL4_4R2_8C_ROWS) * static_cast<int>(MT_COL4_4R2_8C_COLS); // 256
    std::vector<float> phys = ltTransformReLayout(nd, ACLBLASLT_ORDER_COL4_4R2_8C, rows, cols, ld);

    // Hand-verified anchors (offset = ((row/2*8 + col/4)*8) + (row%2*4 + col%4)):
    //   (r0,c0)->0  (r0,c1)->1  (r0,c4)->8  (r1,c0)->4  (r2,c0)->64  (r7,c31)->255
    auto ndVal = [&](int r, int c) { return static_cast<float>(c * rows + r + 1); };
    EXPECT_EQ(phys[0], ndVal(0, 0));
    EXPECT_EQ(phys[1], ndVal(0, 1));
    EXPECT_EQ(phys[8], ndVal(0, 4));
    EXPECT_EQ(phys[4], ndVal(1, 0));
    EXPECT_EQ(phys[64], ndVal(2, 0));
    EXPECT_EQ(phys[255], ndVal(7, 31));
    // round-trip: de-layout(re-layout(nd)) == nd (permutation is a bijection)
    std::vector<float> back = ltTransformDeLayout(phys, ACLBLASLT_ORDER_COL4_4R2_8C, rows, cols, ld);
    ASSERT_EQ(back.size(), nd.size());
    for (size_t i = 0; i < nd.size(); i++)
        EXPECT_EQ(back[i], nd[i]) << "mismatch at " << i;
}

TEST_F(LtMatrixTransformArch35Test, AnchorCol32_2R_4R4_SingleTile)
{
    // single 32-row × 32-col tile; logical ND value at (row,col) = col*32 + row + 1.
    const int rows = 32, cols = 32;
    std::vector<float> nd(static_cast<size_t>(rows) * cols);
    for (int c = 0; c < cols; c++)
        for (int r = 0; r < rows; r++)
            nd[static_cast<size_t>(c) * rows + r] = static_cast<float>(c * rows + r + 1);

    int ld = static_cast<int>(MT_COL32_2R_4R4_ROWS) * static_cast<int>(MT_COL32_2R_4R4_COLS); // 1024
    std::vector<float> phys = ltTransformReLayout(nd, ACLBLASLT_ORDER_COL32_2R_4R4, rows, cols, ld);

    // Hand-verified anchors (offset = rowPerm*32 + col, rowPerm=((row%8)/2*4+row/8)*2+row%2):
    //   (r0,c0)->0  (r1,c0)->32  (r2,c0)->256  (r8,c0)->64  (r16,c0)->128  (r31,c31)->1023
    auto ndVal = [&](int r, int c) { return static_cast<float>(c * rows + r + 1); };
    EXPECT_EQ(phys[0], ndVal(0, 0));
    EXPECT_EQ(phys[32], ndVal(1, 0));
    EXPECT_EQ(phys[256], ndVal(2, 0));
    EXPECT_EQ(phys[64], ndVal(8, 0));
    EXPECT_EQ(phys[128], ndVal(16, 0));
    EXPECT_EQ(phys[1023], ndVal(31, 31));
    std::vector<float> back = ltTransformDeLayout(phys, ACLBLASLT_ORDER_COL32_2R_4R4, rows, cols, ld);
    ASSERT_EQ(back.size(), nd.size());
    for (size_t i = 0; i < nd.size(); i++)
        EXPECT_EQ(back[i], nd[i]) << "mismatch at " << i;
}

// ── FP4 complex-layout anchor tests (TC_L1_110/111). ──
// Independent of the device: build a single composite tile of representable E2M1 magnitudes, run
// golden's re-layout (test-local perm derivation), pack/unpack through the FP4 packed
// codec, and byte-check the perm placement + lossless pack/unpack of the bf16-unpacked domain.
// The permutation index layer is verified EXACT (integer indices); the FP4 value layer is lossless
// here because the inputs are exactly representable E2M1 magnitudes (no quantization on the anchor).
TEST_F(LtMatrixTransformArch35Test, AnchorFp4Col4_4R2_8C_SingleTile)
{
    const int rows = 8, cols = 32;
    // representable E2M1 magnitudes so the pack/unpack round-trip is lossless on the anchor.
    static const float lv[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    std::vector<float> nd(static_cast<size_t>(rows) * cols);
    for (int c = 0; c < cols; c++)
        for (int r = 0; r < rows; r++)
            nd[static_cast<size_t>(c) * rows + r] = lv[(c * rows + r) % 8];

    int ld = static_cast<int>(MT_COL4_4R2_8C_ROWS) * static_cast<int>(MT_COL4_4R2_8C_COLS); // 256
    // re-layout to the complex physical placement (perm_table), pack to fp4x2, unpack, de-layout.
    std::vector<float> phys = ltTransformReLayout(nd, ACLBLASLT_ORDER_COL4_4R2_8C, rows, cols, ld);
    int blocks = ltTransformNumBlocks(ACLBLASLT_ORDER_COL4_4R2_8C, rows, cols);
    std::vector<uint8_t> packed = ltPackFp4(phys, ld, blocks);
    std::vector<float> physBack = ltUnpackFp4(packed, ld, blocks, phys.size());
    // packed nibble adjacency = physical-offset adjacency: byte b holds phys[2b] (low) | phys[2b+1].
    for (size_t b = 0; b < packed.size(); b++) {
        EXPECT_EQ(ltFp4E2m1ToFloat(packed[b] & 0x0Fu), phys[2 * b]) << "low nibble byte " << b;
        if (2 * b + 1 < phys.size()) {
            EXPECT_EQ(ltFp4E2m1ToFloat((packed[b] >> 4) & 0x0Fu), phys[2 * b + 1]) << "high nibble byte " << b;
        }
    }
    // round-trip: de-layout(unpack(pack(re-layout(nd)))) == nd (perm bijection + lossless codec).
    std::vector<float> back = ltTransformDeLayout(physBack, ACLBLASLT_ORDER_COL4_4R2_8C, rows, cols, ld);
    ASSERT_EQ(back.size(), nd.size());
    for (size_t i = 0; i < nd.size(); i++)
        EXPECT_EQ(back[i], nd[i]) << "mismatch at " << i;
}

TEST_F(LtMatrixTransformArch35Test, AnchorFp4Col32_2R_4R4_SingleTile)
{
    const int rows = 32, cols = 32;
    static const float lv[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    std::vector<float> nd(static_cast<size_t>(rows) * cols);
    for (int c = 0; c < cols; c++)
        for (int r = 0; r < rows; r++)
            nd[static_cast<size_t>(c) * rows + r] = lv[(c * rows + r) % 8];

    int ld = static_cast<int>(MT_COL32_2R_4R4_ROWS) * static_cast<int>(MT_COL32_2R_4R4_COLS); // 1024
    std::vector<float> phys = ltTransformReLayout(nd, ACLBLASLT_ORDER_COL32_2R_4R4, rows, cols, ld);
    int blocks = ltTransformNumBlocks(ACLBLASLT_ORDER_COL32_2R_4R4, rows, cols);
    std::vector<uint8_t> packed = ltPackFp4(phys, ld, blocks);
    std::vector<float> physBack = ltUnpackFp4(packed, ld, blocks, phys.size());
    std::vector<float> back = ltTransformDeLayout(physBack, ACLBLASLT_ORDER_COL32_2R_4R4, rows, cols, ld);
    ASSERT_EQ(back.size(), nd.size());
    for (size_t i = 0; i < nd.size(); i++)
        EXPECT_EQ(back[i], nd[i]) << "mismatch at " << i;
}

// ── CSV parameterized tests ──
INSTANTIATE_TEST_SUITE_P(
    LtMatrixTransform, LtMatrixTransformArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<LtMatrixTransformParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<LtMatrixTransformParam>);

TEST_P(LtMatrixTransformArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // ── Step 1: generate physical host data per input order ──
    const bool hasB = !p.BIsNull;
    std::vector<float> physA =
        buildPhys(p.dtypeA, p.orderA, p.transA, p.rowsA, p.colsA, p.lda, p.description, p.randomSeed);
    std::vector<float> physB;
    if (hasB)
        physB = buildPhys(
            p.dtypeB, p.orderB, p.transB, p.rowsB, p.colsB, p.ldb, p.description,
            p.randomSeed ? p.randomSeed + 1u : 7u);

    // ── Step 2: execute on NPU ──
    aclblasLtHandle_t handle = p.handleNull ? nullptr : LtMatrixTransformArch35Test::ltHandle_;
    std::vector<float> devNdC;
    aclblasStatus_t ret =
        aclblasLtMatrixTransform_npu(handle, LtMatrixTransformArch35Test::stream_, p, physA, physB, devNdC);

    // ── Step 3: compare error code on non-success cases ──
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // empty matrix no-op: nothing to verify
    if (p.rowsC == 0 || p.colsC == 0)
        return;

    // ── Step 4: compute golden (logical column-major ND) ──
    // A null beta pointer is interpreted as beta=0: the operator drops the B term and computes
    // C = alpha*op(A). Golden must mirror this, so the B term is excluded when betaNull is set.
    const bool goldenHasB = hasB && !p.betaNull;
    std::vector<float> golden;
    aclblasStatus_t cpuRet = aclblasLtMatrixTransform_cpu(
        LtMatrixTransformArch35Test::ltHandle_, p.dtypeA, p.orderA, p.transA, p.rowsA, p.colsA, p.lda, physA, p.dtypeB,
        p.orderB, p.transB, p.rowsB, p.colsB, p.ldb, physB, goldenHasB, p.dtypeC, p.rowsC, p.colsC, p.scaleType,
        p.alpha, p.beta, golden);
    ASSERT_EQ(cpuRet, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(devNdC.size(), golden.size());

    // ── Step 5: verify precision (by output dtype) ──
    // Threshold by OUTPUT dtype (cross-dtype: output端可表示精度 decides the band, §4.1).
    VerifyConfig cfg;
    if (isIntTransformDtype(p.dtypeC)) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        cfg.mode = PrecisionMode::MERE_MARE;
        cfg.mareMultiplier = 10.0;
        if (p.dtypeC == ACL_FLOAT)
            cfg.mereThreshold = std::pow(2.0, -13);
        else if (p.dtypeC == ACL_FLOAT16)
            cfg.mereThreshold = std::pow(2.0, -10);
        else if (p.dtypeC == ACL_BF16)
            cfg.mereThreshold = std::pow(2.0, -7);
        else if (p.dtypeC == ACL_FLOAT8_E4M3FN)
            cfg.mereThreshold = std::pow(2.0, -3);
        else if (p.dtypeC == ACL_FLOAT8_E5M2)
            cfg.mereThreshold = std::pow(2.0, -2);
        else if (p.dtypeC == ACL_FLOAT4_E2M1)
            cfg.mereThreshold = std::pow(2.0, -1);
    }

    EXPECT_TRUE(Verifier::verifyVector(devNdC.data(), golden.data(), golden.size(), 1, cfg, p.caseName));
}

// issue #585: 内核仅实例化 A/B 同 dtype 组合（MT_DTYPE_LIST），混合 A/B dtype 此前落入
// default 分支静默不启动内核仍返回 SUCCESS，现 host 校验收紧为 NOT_SUPPORTED。
TEST_F(LtMatrixTransformArch35Test, MixedOperandDtypesRejected)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    aclblasLtMatrixLayout_t aLayout = nullptr;
    aclblasLtMatrixLayout_t bLayout = nullptr;
    aclblasLtMatrixLayout_t cLayout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&aLayout, ACL_FLOAT, 4, 4, 4), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&bLayout, ACL_FLOAT16, 4, 4, 4), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&cLayout, ACL_FLOAT, 4, 4, 4), ACLBLAS_STATUS_SUCCESS);

    constexpr size_t kF32Bytes = 4 * 4 * sizeof(float);
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, kF32Bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dB, 4 * 4 * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dC, kF32Bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);

    float alpha = 1.0f;
    float beta = 1.0f;
    const aclblasStatus_t ret = aclblasLtMatrixTransform(
        LtMatrixTransformArch35Test::ltHandle_, desc, &alpha, dA, aLayout, &beta, dB, bLayout, dC, cLayout,
        LtMatrixTransformArch35Test::stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_SUPPORTED);

    aclrtFree(dC);
    aclrtFree(dB);
    aclrtFree(dA);
    aclblasLtMatrixLayoutDestroy(cLayout);
    aclblasLtMatrixLayoutDestroy(bLayout);
    aclblasLtMatrixLayoutDestroy(aLayout);
    aclblasLtMatrixTransformDescDestroy(desc);
}

// issue #586: rows/cols/ld 超过 2^32-1 时 tiling 的 uint32 截断会使内核按错误形状执行并返回
// SUCCESS，入口须显式拒绝（校验在设备访问之前，哑缓冲区不会被读写）。
TEST_F(LtMatrixTransformArch35Test, OversizeDimsRejected)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    constexpr uint64_t kBigDim = 4294967296ULL; // 2^32，超出 uint32 可表示范围
    aclblasLtMatrixLayout_t aLayout = nullptr;
    aclblasLtMatrixLayout_t cLayout = nullptr;
    ASSERT_EQ(
        aclblasLtMatrixLayoutCreate(&aLayout, ACL_FLOAT, kBigDim, 1, static_cast<int64_t>(kBigDim)),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatrixLayoutCreate(&cLayout, ACL_FLOAT, kBigDim, 1, static_cast<int64_t>(kBigDim)),
        ACLBLAS_STATUS_SUCCESS);

    void* dA = nullptr;
    void* dC = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, 16, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dC, 16, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);

    float alpha = 1.0f;
    const aclblasStatus_t ret = aclblasLtMatrixTransform(
        LtMatrixTransformArch35Test::ltHandle_, desc, &alpha, dA, aLayout, nullptr, nullptr, nullptr, dC, cLayout,
        LtMatrixTransformArch35Test::stream_);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);

    aclrtFree(dC);
    aclrtFree(dA);
    aclblasLtMatrixLayoutDestroy(cLayout);
    aclblasLtMatrixLayoutDestroy(aLayout);
    aclblasLtMatrixTransformDescDestroy(desc);
}
