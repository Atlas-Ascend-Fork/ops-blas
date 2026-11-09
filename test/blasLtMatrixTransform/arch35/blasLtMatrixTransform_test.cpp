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
#include <cstring>
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

// cuBLASLt 对齐：标量执行链恒按主机指针解引用 alpha/beta（MatResolveScalars），本库只实现
// HOST 指针模式，故非 HOST 取值在 SetAttribute 阶段就响亮拒绝（NOT_SUPPORTED），而不是记录
// 后由 MatTransformLaunch 的运行期守卫兜底——后者保留，作为绕过 Set 路径构造出的胶囊的纵深
// 防御，但已无法经公开 API 触达。
TEST_F(LtMatrixTransformArch35Test, DevicePointerModeRejectedAtSetAttribute)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);

    const int32_t modeDevice = ACLBLASLT_POINTER_MODE_DEVICE;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &modeDevice, sizeof(modeDevice)),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    // 被拒后描述符不得改写：pointer mode 仍为默认的 HOST。
    int32_t modeOut = -1;
    size_t written = 0;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &modeOut, sizeof(modeOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeOut, static_cast<int32_t>(ACLBLASLT_POINTER_MODE_HOST));

    // 正向路径不受收紧影响：显式写入 HOST 仍被接受并可回读。
    const int32_t modeHost = ACLBLASLT_POINTER_MODE_HOST;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &modeHost, sizeof(modeHost)),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &modeOut, sizeof(modeOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(modeOut, modeHost);

    // 拒绝的是「非 HOST」整个值域，不止 DEVICE 一个取值。
    const int32_t modeBogus = 99;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &modeBogus, sizeof(modeBogus)),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
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

// ── aclblasLtMatrixTransformDescInit 契约组 ──
// 覆盖：Init 产物被 Destroy 拒绝、Create 产物穿过 SetAttribute 后仍可 Destroy、
// 损坏胶囊被 magic 检查拦下（而非 delete 一个非法指针）。

TEST_F(LtMatrixTransformArch35Test, TransformDescInitProductRejectedByDestroy)
{
    aclblasLtMatrixTransformDescOpaque_t storage{};
    ASSERT_EQ(aclblasLtMatrixTransformDescInit(&storage, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    // Init 产物位于调用方存储，Destroy 必须拒绝而不是 delete 一个非堆指针。
    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(&storage), ACLBLAS_STATUS_INVALID_VALUE);
}

// 契约配套用例：公开契约要求调用方存储入口为确定值（零初始化，或 Create/Init 产物胶囊），
// Init 读回首部字节以保留 Create 产物的堆来源位。此处故意预填契约外的**确定**垃圾字节
// （确定值，非 UB），验证 magic 门控对任何非胶囊内容都路由到纯重置路径：heapOwned=0、
// Destroy 照旧拒绝——「残留字节被误标堆来源」须恰好命中 32 位 magic 且 heapOwned 非零，
// 而该输入本身已属违约。
TEST_F(LtMatrixTransformArch35Test, TransformDescInitOnNonCapsuleBytesCarriesNoProvenance)
{
    aclblasLtMatrixTransformDescOpaque_t storage{};
    for (auto& word : storage.data) {
        word = 0xAAAAAAAAAAAAAAAAULL; // 确定的非零垃圾：magic 必不匹配，亦非 Create 产物
    }
    ASSERT_EQ(aclblasLtMatrixTransformDescInit(&storage, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    // 纯重置路径：来源位为 0，Destroy 拒绝，绝不 delete 非堆指针。
    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(&storage), ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(LtMatrixTransformArch35Test, TransformDescInitRejectsNull)
{
    EXPECT_EQ(aclblasLtMatrixTransformDescInit(nullptr, ACL_FLOAT), ACLBLAS_STATUS_INVALID_VALUE);
}

// cuBLASLt 的 Init 语义是「重置一个先前已分配的描述符」，Create→Init（重置复用）→Destroy 属
// 合法模式。此前 Init 无条件以 NSDMI 默认值覆写胶囊，把 heapOwned 从 1 翻回 0，Destroy 的来源
// 防护随后永久拒绝该堆胶囊——既泄漏又给出意外 INVALID_VALUE。现在 Init 保留来源位、只重置属性。
TEST_F(LtMatrixTransformArch35Test, TransformDescInitOnCreateProductKeepsDestroyable)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    ASSERT_NE(desc, nullptr);
    const int32_t transT = ACLBLAS_OP_T;
    ASSERT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transT, sizeof(transT)),
        ACLBLAS_STATUS_SUCCESS);

    // 以另一个 scaleType 重置：scaleType 取入参，其余属性回到默认（transA=OP_N）。
    ASSERT_EQ(aclblasLtMatrixTransformDescInit(desc, ACL_BF16), ACLBLAS_STATUS_SUCCESS);
    int32_t transAOut = -1;
    aclDataType scaleOut = ACL_DT_UNDEFINED;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transAOut, sizeof(transAOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(transAOut, ACLBLAS_OP_N);
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &scaleOut, sizeof(scaleOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(scaleOut, ACL_BF16);

    // 来源位保留 → Destroy 正常释放，Create→Init→Destroy 不再泄漏。
    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatrixTransformArch35Test, TransformDescCreateProductSurvivesSetAttributeThenDestroy)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    ASSERT_NE(desc, nullptr);
    const int32_t transT = ACLBLAS_OP_T;
    ASSERT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transT, sizeof(transT)),
        ACLBLAS_STATUS_SUCCESS);
    // 来源标记必须能穿过 SetAttribute 的「解包—改一项—重打包」，否则此路径会被误拒。
    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
}

// TRANSA/TRANSB 的值域在 Set 阶段校验，白名单与变换引擎运行期的 MatIsValidOp 同域（N/T/C，
// C 按 T 处理）：越界取值返回 INVALID_VALUE 且不改写描述符，消除「成功构造、Transform 阶段
// 才失败」——与 SCALE_TYPE 的收紧口径及 MatrixLayout ORDER 的 Set 期白名单拉齐。
TEST_F(LtMatrixTransformArch35Test, TransformDescRejectsIllegalTransAtSetAttribute)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);

    // 白名单内的三个取值均被接受。
    const int32_t validOps[] = {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C};
    for (int32_t op : validOps) {
        EXPECT_EQ(
            aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)),
            ACLBLAS_STATUS_SUCCESS)
            << "op=" << op;
    }

    // 先写入一个合法值，作为「被拒后不改写」的参照。
    const int32_t transT = ACLBLAS_OP_T;
    ASSERT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transT, sizeof(transT)),
        ACLBLAS_STATUS_SUCCESS);

    // 越界取值：0、负数、以及紧邻白名单边界的 110/114 一律拒绝（TRANSA 与 TRANSB 同口径）。
    const int32_t badOps[] = {0, -1, 99, 110, 114};
    for (int32_t bad : badOps) {
        EXPECT_EQ(
            aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &bad, sizeof(bad)),
            ACLBLAS_STATUS_INVALID_VALUE)
            << "transA bad value=" << bad;
        EXPECT_EQ(
            aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &bad, sizeof(bad)),
            ACLBLAS_STATUS_INVALID_VALUE)
            << "transB bad value=" << bad;
    }

    int32_t transAOut = -1;
    int32_t transBOut = -1;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transAOut, sizeof(transAOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &transBOut, sizeof(transBOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(transAOut, ACLBLAS_OP_T); // 被拒后仍是先前写入的合法值
    EXPECT_EQ(transBOut, ACLBLAS_OP_N); // 从未被成功写入，保持默认

    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
}

// ── TransformDescSet/Get 通用负向补全（缺口 G9，评估报告 P2-4）──
// 真值源：blasLt/api/aclblaslt_transform_desc.cpp —— Set/Get 共用 UnpackTransformDesc
// （判空 + magic 守卫，仅 Set 侧要求 buf 非空）；Set 校验顺序「判空+magic → size → attr」为
// api_list 明文契约（非法 size 搭配未识别 attr 返回 INVALID_VALUE）；SCALE_TYPE 的 Set 期
// 白名单 {FLOAT, BF16, INT32}（issue #788，与引擎 MatScaleTypeMatchesPath 同域）。
// unknown attr → NOT_SUPPORTED 已由 TransformDescAttributeEnumValuesPinned（raw 4/99 双侧）
// 钉住，此处不重复。

// issue #788 修复本体回归网：SCALE_TYPE 在 Set 阶段可覆盖，白名单三值逐一接受并回读一致；
// 越界 ACL_DOUBLE 在 Set 期拒绝（INVALID_VALUE）且不改写描述符，消除「成功构造、
// aclblasLtMatrixTransform 阶段才失败」。
TEST_F(LtMatrixTransformArch35Test, TransformDescScaleTypeSetWhitelist)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);

    const aclDataType allowed[] = {ACL_FLOAT, ACL_BF16, ACL_INT32};
    for (aclDataType scaleType : allowed) {
        const int32_t value = static_cast<int32_t>(scaleType);
        EXPECT_EQ(
            aclblasLtMatrixTransformDescSetAttribute(
                desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &value, sizeof(value)),
            ACLBLAS_STATUS_SUCCESS)
            << "scaleType=" << value;
        int32_t out = -1;
        size_t written = 0;
        ASSERT_EQ(
            aclblasLtMatrixTransformDescGetAttribute(
                desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &out, sizeof(out), &written),
            ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(out, value);
        EXPECT_EQ(written, sizeof(int32_t));
    }

    // 越界拒绝：ACL_DOUBLE → INVALID_VALUE，且不改写（回读仍为最后一次合法写入的 INT32）。
    const int32_t doubled = static_cast<int32_t>(ACL_DOUBLE);
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &doubled, sizeof(doubled)),
        ACLBLAS_STATUS_INVALID_VALUE);
    int32_t out = -1;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &out, sizeof(out), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(out, static_cast<int32_t>(ACL_INT32));

    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
}

// Set 侧通用负向：desc/buf NULL、宽度非法（2B/8B/0B）、损坏 magic 均 → INVALID_VALUE；
// 校验顺序契约——size 校验先于 attr 识别，故非法 size 搭配未知 attr 返回 INVALID_VALUE 而非
// NOT_SUPPORTED（对照 Layout/Preference 族的 attr 先行 → NOT_SUPPORTED，族间不一致为检视
// 报告 C6 已披露观察项，此处钉住 TransformDesc 侧的文档化顺序）。
TEST_F(LtMatrixTransformArch35Test, TransformDescSetAttributeRejectsInvalidInputs)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    const int32_t transT = ACLBLAS_OP_T;

    // desc NULL / buf NULL → INVALID_VALUE（UnpackTransformDesc，Set 侧 requireBuf=true）
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            nullptr, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transT, sizeof(transT)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, nullptr, sizeof(transT)),
        ACLBLAS_STATUS_INVALID_VALUE);
    // size ≠ 4B：2B / 8B / 0B 均拒绝（全属性按 int32 存储，宽度精确匹配）
    const int16_t narrow = static_cast<int16_t>(ACLBLAS_OP_T);
    const int64_t wide = static_cast<int64_t>(ACLBLAS_OP_T);
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &narrow, sizeof(narrow)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &wide, sizeof(wide)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transT, 0),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 校验顺序：非法 size + 未知 attr → INVALID_VALUE（size 先行；若 attr 先行则 NOT_SUPPORTED）
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, static_cast<aclblasLtMatrixTransformDescAttribute_t>(99), &narrow, sizeof(narrow)),
        ACLBLAS_STATUS_INVALID_VALUE);
    // Set 侧损坏 magic → INVALID_VALUE（Destroy 侧已有 counterpart，Set 侧走同一守卫），
    // 随后复原，避免影响后续断言并保证正常释放
    auto* capsule = reinterpret_cast<aclblasLtMatrixTransformDescOpaque_t*>(desc);
    const uint64_t savedWord = capsule->data[0];
    capsule->data[0] ^= 0xFFu; // 破坏 magic 低字节（字节 0~3），其余字段不受影响
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transT, sizeof(transT)),
        ACLBLAS_STATUS_INVALID_VALUE);
    capsule->data[0] = savedWord;
    // 上述全部拒绝均不改写描述符：TRANSA 仍为默认 OP_N
    int32_t transOut = -1;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transOut, sizeof(transOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(transOut, ACLBLAS_OP_N);
    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
}

// Get 侧通用负向：desc NULL、buf NULL + size≠0、损坏 magic 均 → INVALID_VALUE。
// CopyAttributeOut 统一口径：sizeWritten 先行回传所需宽度、再按 buf NULL/不足拒绝；而 magic
// 守卫在 attr 分派与 CopyAttributeOut 之前，命中损坏 magic 时 sizeWritten 不写。
TEST_F(LtMatrixTransformArch35Test, TransformDescGetAttributeRejectsInvalidInputs)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);

    int32_t out = -1;
    size_t written = SIZE_MAX;
    // desc NULL → INVALID_VALUE（Get 侧不要求 buf 非空——探测模式合法，判空只针对 desc）
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            nullptr, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &out, sizeof(out), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    // buf NULL + size≠0 → INVALID_VALUE，且 sizeWritten 已先行回传所需宽度
    written = SIZE_MAX;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, nullptr, sizeof(out), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(written, sizeof(int32_t));
    // Get 侧损坏 magic → INVALID_VALUE（Destroy 侧已有 TransformDescDestroyRejectsCorruptedMagic，
    // Get 侧走同一 UnpackTransformDesc 守卫）；sizeWritten 不写（守卫先于 CopyAttributeOut）
    auto* capsule = reinterpret_cast<aclblasLtMatrixTransformDescOpaque_t*>(desc);
    const uint64_t savedWord = capsule->data[0];
    capsule->data[0] ^= 0xFFu;
    written = SIZE_MAX;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &out, sizeof(out), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(written, SIZE_MAX);
    capsule->data[0] = savedWord;
    // 复原后正常读取（防误伤回归网）：默认 SCALE_TYPE 回读 ACL_FLOAT
    written = 0;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &out, sizeof(out), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(out, static_cast<int32_t>(ACL_FLOAT));
    EXPECT_EQ(written, sizeof(int32_t));
    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatrixTransformArch35Test, TransformDescDestroyRejectsCorruptedMagic)
{
    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    auto* capsule = reinterpret_cast<aclblasLtMatrixTransformDescOpaque_t*>(desc);
    const uint64_t savedWord = capsule->data[0];
    capsule->data[0] ^= 0xFFu; // 破坏 magic；heapOwned 仍为 1，故命中的是 magic 检查
    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_INVALID_VALUE);
    capsule->data[0] = savedWord; // 复原后正常释放，避免测试泄漏
    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatrixTransformArch35Test, TransformDescInitDefaultsMatchCreate)
{
    // transa/transb 不是 Init 的参数，须与刚 Create 出来的描述符逐字节等价（均为 OP_N）。
    aclblasLtMatrixTransformDescOpaque_t fromInit{};
    ASSERT_EQ(aclblasLtMatrixTransformDescInit(&fromInit, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    aclblasLtMatrixTransformDesc_t fromCreate = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&fromCreate, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);

    int32_t transAInit = -1, transACreate = -1, transBInit = -1, transBCreate = -1;
    size_t written = 0;
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            &fromInit, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transAInit, sizeof(transAInit), &written),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            fromCreate, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transACreate, sizeof(transACreate), &written),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            &fromInit, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &transBInit, sizeof(transBInit), &written),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            fromCreate, ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &transBCreate, sizeof(transBCreate), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(transAInit, transACreate);
    EXPECT_EQ(transAInit, ACLBLAS_OP_N);
    EXPECT_EQ(transBInit, transBCreate);
    EXPECT_EQ(transBInit, ACLBLAS_OP_N);

    ASSERT_EQ(aclblasLtMatrixTransformDescDestroy(fromCreate), ACLBLAS_STATUS_SUCCESS);
}

// ── aclblasLtMatrixLayoutSetAttribute / GetAttribute 契约组（缺口 G1：12 接口中唯一
// 零专属用例族，循本文件 TransformDesc 契约组直测风格）──
// 宽度真值源 = aclblaslt_layout_impl.h 字段类型：TYPE/ORDER/BATCH_COUNT 4 字节，
// ROWS/COLS/LD/STRIDED_BATCH_OFFSET 8 字节；Set 精确宽度匹配（LoadLayoutField），
// Get 探测/不足/NULL 语义由 StoreLayoutFieldOut 承载（sizeWritten 先行回传）。

namespace {

// Get 回读断言：SUCCESS + 值精确一致 + sizeWritten==属性宽度。
template <typename T>
void ExpectLayoutAttrEquals(aclblasLtMatrixLayout_t layout, aclblasLtMatrixLayoutAttribute_t attr, T expected)
{
    T out{};
    size_t written = 0;
    ASSERT_EQ(aclblasLtMatrixLayoutGetAttribute(layout, attr, &out, sizeof(out), &written), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(out, expected);
    EXPECT_EQ(written, sizeof(T));
}

// Set→Get 回环断言（Set 写入值此前从未被 Get 验证——「Set 写错字段」类缺陷不可见）。
template <typename T>
void ExpectLayoutSetThenRead(aclblasLtMatrixLayout_t layout, aclblasLtMatrixLayoutAttribute_t attr, T value)
{
    ASSERT_EQ(aclblasLtMatrixLayoutSetAttribute(layout, attr, &value, sizeof(T)), ACLBLAS_STATUS_SUCCESS);
    ExpectLayoutAttrEquals(layout, attr, value);
}

// 探测族断言：sizeInBytes==0 → SUCCESS + sizeWritten==所需宽度（buf 可为空）。
void ExpectLayoutProbeWidth(aclblasLtMatrixLayout_t layout, aclblasLtMatrixLayoutAttribute_t attr, size_t expectedBytes)
{
    size_t written = SIZE_MAX;
    ASSERT_EQ(aclblasLtMatrixLayoutGetAttribute(layout, attr, nullptr, 0, &written), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(written, expectedBytes);
}

} // namespace

TEST_F(LtMatrixTransformArch35Test, MatrixLayoutSetGetRoundTrip)
{
    aclblasLtMatrixLayout_t layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 8, 4, 8), ACLBLAS_STATUS_SUCCESS);

    // 7 属性正向 Set→Get 回环（含宽度断言）
    ExpectLayoutSetThenRead(layout, ACLBLASLT_MATRIX_LAYOUT_TYPE, ACL_BF16);
    ExpectLayoutSetThenRead(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, ACLBLASLT_ORDER_ROW);
    ExpectLayoutSetThenRead(layout, ACLBLASLT_MATRIX_LAYOUT_ROWS, static_cast<uint64_t>(6));
    ExpectLayoutSetThenRead(layout, ACLBLASLT_MATRIX_LAYOUT_COLS, static_cast<uint64_t>(3));
    ExpectLayoutSetThenRead(layout, ACLBLASLT_MATRIX_LAYOUT_LD, static_cast<int64_t>(12));
    ExpectLayoutSetThenRead(layout, ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT, static_cast<int32_t>(2));
    ExpectLayoutSetThenRead(layout, ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, static_cast<int64_t>(7));

    EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatrixTransformArch35Test, MatrixLayoutGetDefaultsAfterCreate)
{
    aclblasLtMatrixLayout_t layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 8, 4, 6), ACLBLAS_STATUS_SUCCESS);

    // Create 后默认值逐项钉住：ORDER=COL、BATCH_COUNT=1、STRIDE=0，其余回显入参
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_TYPE, ACL_FLOAT);
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, ACLBLASLT_ORDER_COL);
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_ROWS, static_cast<uint64_t>(8));
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_COLS, static_cast<uint64_t>(4));
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_LD, static_cast<int64_t>(6));
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT, static_cast<int32_t>(1));
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, static_cast<int64_t>(0));

    EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatrixTransformArch35Test, MatrixLayoutSetAttributeRejectsInvalidInputs)
{
    aclblasLtMatrixLayout_t layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 8, 4, 8), ACLBLAS_STATUS_SUCCESS);

    // NULL 组合：layout NULL / buf NULL → INVALID_VALUE
    const int64_t ld = 10;
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(nullptr, ACLBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_LD, nullptr, sizeof(ld)),
        ACLBLAS_STATUS_INVALID_VALUE);
    // ld<0 Set 期拒绝（issue #783，与 Create 校验对齐；公开 API 路径此前从未被测），且不改写
    const int64_t negativeLd = -1;
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_LD, &negativeLd, sizeof(negativeLd)),
        ACLBLAS_STATUS_INVALID_VALUE);
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_LD, static_cast<int64_t>(8));
    // ORDER 白名单外 Set 期拒绝（issue #783；与 CSV TC_L1_63 同口径），且不改写
    const int32_t badOrder = 99;
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, &badOrder, sizeof(badOrder)),
        ACLBLAS_STATUS_INVALID_VALUE);
    const int32_t negativeOrder = -1;
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, &negativeOrder, sizeof(negativeOrder)),
        ACLBLAS_STATUS_INVALID_VALUE);
    ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, ACLBLASLT_ORDER_COL);

    EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatrixTransformArch35Test, MatrixLayoutSetAttributeWidthAndUnknownAttr)
{
    aclblasLtMatrixLayout_t layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 8, 4, 8), ACLBLAS_STATUS_SUCCESS);

    // 宽度精确匹配矩阵：4 字节属性收 8 字节缓冲、8 字节属性收 4 字节缓冲均拒绝
    const int64_t wide = 1;
    const int32_t narrow = 1;
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_TYPE, &wide, sizeof(wide)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, &wide, sizeof(wide)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &wide, sizeof(wide)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ROWS, &narrow, sizeof(narrow)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_COLS, &narrow, sizeof(narrow)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_LD, &narrow, sizeof(narrow)),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(
            layout, ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &narrow, sizeof(narrow)),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 子集外（cuBLASLt 7=PLANE_OFFSET、8=BATCH_MODE 无仓内对应物）与任意未知值 → NOT_SUPPORTED
    for (int raw : {7, 8, 999}) {
        EXPECT_EQ(
            aclblasLtMatrixLayoutSetAttribute(
                layout, static_cast<aclblasLtMatrixLayoutAttribute_t>(raw), &narrow, sizeof(narrow)),
            ACLBLAS_STATUS_NOT_SUPPORTED)
            << "raw attr " << raw;
    }

    EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatrixTransformArch35Test, MatrixLayoutGetAttributeProbeAndGuards)
{
    aclblasLtMatrixLayout_t layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 8, 4, 8), ACLBLAS_STATUS_SUCCESS);

    // 探测族：7 属性 sizeInBytes==0 → SUCCESS + sizeWritten==属性宽度
    ExpectLayoutProbeWidth(layout, ACLBLASLT_MATRIX_LAYOUT_TYPE, sizeof(aclDataType));
    ExpectLayoutProbeWidth(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, sizeof(aclblasLtOrder_t));
    ExpectLayoutProbeWidth(layout, ACLBLASLT_MATRIX_LAYOUT_ROWS, sizeof(uint64_t));
    ExpectLayoutProbeWidth(layout, ACLBLASLT_MATRIX_LAYOUT_COLS, sizeof(uint64_t));
    ExpectLayoutProbeWidth(layout, ACLBLASLT_MATRIX_LAYOUT_LD, sizeof(int64_t));
    ExpectLayoutProbeWidth(layout, ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT, sizeof(int32_t));
    ExpectLayoutProbeWidth(layout, ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, sizeof(int64_t));
    // (size==0 且 sizeWritten==NULL) → SUCCESS：Get 族宽松契约（异于 Cap 族矩阵行 2 严格守卫）
    EXPECT_EQ(
        aclblasLtMatrixLayoutGetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, nullptr, 0, nullptr),
        ACLBLAS_STATUS_SUCCESS);
    // 缓冲不足 → INVALID_VALUE，且 sizeWritten 仍回传所需宽度
    size_t written = SIZE_MAX;
    uint16_t small = 0;
    EXPECT_EQ(
        aclblasLtMatrixLayoutGetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ROWS, &small, sizeof(small), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(written, sizeof(uint64_t));
    // buf NULL + size!=0 → INVALID_VALUE（sizeWritten 已先行回传所需宽度）
    written = SIZE_MAX;
    int32_t typeOut = 0;
    EXPECT_EQ(
        aclblasLtMatrixLayoutGetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_TYPE, nullptr, sizeof(typeOut), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(written, sizeof(int32_t));
    // layout NULL → INVALID_VALUE
    EXPECT_EQ(
        aclblasLtMatrixLayoutGetAttribute(nullptr, ACLBLASLT_MATRIX_LAYOUT_TYPE, &typeOut, sizeof(typeOut), &written),
        ACLBLAS_STATUS_INVALID_VALUE);
    // 未知属性 → NOT_SUPPORTED 且 sizeWritten 不写
    written = SIZE_MAX;
    EXPECT_EQ(
        aclblasLtMatrixLayoutGetAttribute(
            layout, static_cast<aclblasLtMatrixLayoutAttribute_t>(7), &typeOut, sizeof(typeOut), &written),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(written, SIZE_MAX);

    EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(LtMatrixTransformArch35Test, MatrixLayoutOrderWhitelistAccepted)
{
    aclblasLtMatrixLayout_t layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 8, 4, 8), ACLBLAS_STATUS_SUCCESS);

    // ORDER 白名单 5 个合法值 Set 期全部接受并可回读（此前仅 ROW/COL 经载体覆盖）
    const aclblasLtOrder_t kOrders[] = {
        ACLBLASLT_ORDER_COL, ACLBLASLT_ORDER_ROW, ACLBLASLT_ORDER_COL32, ACLBLASLT_ORDER_COL4_4R2_8C,
        ACLBLASLT_ORDER_COL32_2R_4R4};
    for (aclblasLtOrder_t order : kOrders) {
        EXPECT_EQ(
            aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            ACLBLAS_STATUS_SUCCESS)
            << "order " << static_cast<int>(order);
        ExpectLayoutAttrEquals(layout, ACLBLASLT_MATRIX_LAYOUT_ORDER, order);
    }

    EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
}

// issue #823 回归：属性读写改经 memcpy_s，不再依赖用户缓冲 8 字节对齐（修复前
// uint64 属性在非对齐缓冲上 reinterpret_cast 直读写属 UB）。奇地址缓冲钉住该修复。
TEST_F(LtMatrixTransformArch35Test, MatrixLayoutUnalignedBufferRoundTrip)
{
    aclblasLtMatrixLayout_t layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 8, 4, 8), ACLBLAS_STATUS_SUCCESS);

    std::vector<uint8_t> rawIn(sizeof(int64_t) + 1, 0);
    std::vector<uint8_t> rawOut(sizeof(int64_t) + 1, 0);
    uint8_t* unalignedIn = rawIn.data() + 1; // 奇地址：非 8 字节对齐
    uint8_t* unalignedOut = rawOut.data() + 1;
    const int64_t rows = 5;
    ASSERT_EQ(memcpy_s(unalignedIn, sizeof(rows), &rows, sizeof(rows)), EOK);
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ROWS, unalignedIn, sizeof(rows)),
        ACLBLAS_STATUS_SUCCESS);
    size_t written = 0;
    EXPECT_EQ(
        aclblasLtMatrixLayoutGetAttribute(layout, ACLBLASLT_MATRIX_LAYOUT_ROWS, unalignedOut, sizeof(rows), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(written, sizeof(int64_t));
    int64_t rowsOut = 0;
    ASSERT_EQ(memcpy_s(&rowsOut, sizeof(rowsOut), unalignedOut, sizeof(rowsOut)), EOK);
    EXPECT_EQ(rowsOut, rows);

    EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
}

// ── MATRIX_LAYOUT 族枚举裸数值钉值（迁移安全，补齐 !863 重编号时登记的欠账）──
// 重编号后 0~6 与 cuBLASLt cublasLtMatrixLayoutAttribute_t 子集 1:1（TYPE=0 …
// STRIDED_BATCH_OFFSET=6）。重编号前曾存在 3 条同宽度静默交叉写通道（TYPE↔BATCH_COUNT、
// LD↔ROWS、STRIDE↔LD）且 7/7 全错位；符号名引用对数值漂移全盲，裸值钉值是唯一回归网。
TEST_F(LtMatrixTransformArch35Test, MatrixLayoutAttributeEnumValuesPinned)
{
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_LAYOUT_TYPE), 0);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_LAYOUT_ORDER), 1);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_LAYOUT_ROWS), 2);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_LAYOUT_COLS), 3);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_LAYOUT_LD), 4);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT), 5);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET), 6);

    aclblasLtMatrixLayout_t layout = nullptr;
    ASSERT_EQ(aclblasLtMatrixLayoutCreate(&layout, ACL_FLOAT, 8, 4, 8), ACLBLAS_STATUS_SUCCESS);
    // 行为级裸值可达：raw 1 → ORDER 回环；raw 2 → ROWS（8B）；raw 5 → BATCH_COUNT（4B）
    const int32_t rowOrder = 1; // ACLBLASLT_ORDER_ROW 的数值形态
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(
            layout, static_cast<aclblasLtMatrixLayoutAttribute_t>(1), &rowOrder, sizeof(rowOrder)),
        ACLBLAS_STATUS_SUCCESS);
    int32_t orderOut = -1;
    size_t written = 0;
    EXPECT_EQ(
        aclblasLtMatrixLayoutGetAttribute(
            layout, static_cast<aclblasLtMatrixLayoutAttribute_t>(1), &orderOut, sizeof(orderOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(orderOut, static_cast<int32_t>(ACLBLASLT_ORDER_ROW));
    ExpectLayoutProbeWidth(layout, static_cast<aclblasLtMatrixLayoutAttribute_t>(2), sizeof(uint64_t));
    ExpectLayoutProbeWidth(layout, static_cast<aclblasLtMatrixLayoutAttribute_t>(5), sizeof(int32_t));
    // 子集外（cuBLASLt 7=PLANE_OFFSET、8=BATCH_MODE 无仓内对应物）→ Set/Get 双侧 NOT_SUPPORTED
    EXPECT_EQ(
        aclblasLtMatrixLayoutSetAttribute(
            layout, static_cast<aclblasLtMatrixLayoutAttribute_t>(7), &rowOrder, sizeof(rowOrder)),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        aclblasLtMatrixLayoutGetAttribute(
            layout, static_cast<aclblasLtMatrixLayoutAttribute_t>(8), nullptr, 0, &written),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(aclblasLtMatrixLayoutDestroy(layout), ACLBLAS_STATUS_SUCCESS);
}

// ── TRANSFORM_DESC 族枚举裸数值钉值（迁移安全）──
// SCALE_TYPE=0、POINTER_MODE=1、TRANSA=2、TRANSB=3，与 cuBLASLt
// cublasLtMatrixTransformDescAttributes_t 4/4 对齐；子集外一律 NOT_SUPPORTED。
TEST_F(LtMatrixTransformArch35Test, TransformDescAttributeEnumValuesPinned)
{
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE), 0);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE), 1);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA), 2);
    EXPECT_EQ(static_cast<int>(ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB), 3);

    aclblasLtMatrixTransformDesc_t desc = nullptr;
    ASSERT_EQ(aclblasLtMatrixTransformDescCreate(&desc, ACL_FLOAT), ACLBLAS_STATUS_SUCCESS);
    // 行为级裸值可达：raw 0 → SCALE_TYPE 探测宽度 4；raw 2 → TRANSA 回环（OP_T）
    size_t written = 0;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, static_cast<aclblasLtMatrixTransformDescAttribute_t>(0), nullptr, 0, &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(written, sizeof(int32_t));
    const int32_t transT = ACLBLAS_OP_T;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, static_cast<aclblasLtMatrixTransformDescAttribute_t>(2), &transT, sizeof(transT)),
        ACLBLAS_STATUS_SUCCESS);
    int32_t transOut = -1;
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, static_cast<aclblasLtMatrixTransformDescAttribute_t>(2), &transOut, sizeof(transOut), &written),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(transOut, transT);
    // 子集外 raw 4/99 → Set/Get 双侧 NOT_SUPPORTED
    EXPECT_EQ(
        aclblasLtMatrixTransformDescSetAttribute(
            desc, static_cast<aclblasLtMatrixTransformDescAttribute_t>(4), &transT, sizeof(transT)),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        aclblasLtMatrixTransformDescGetAttribute(
            desc, static_cast<aclblasLtMatrixTransformDescAttribute_t>(99), nullptr, 0, &written),
        ACLBLAS_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(aclblasLtMatrixTransformDescDestroy(desc), ACLBLAS_STATUS_SUCCESS);
}
