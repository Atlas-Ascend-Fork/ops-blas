/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "../../../../blas/gemm_batched/arch22/cgemm_batched_float.h"

extern "C" __complex__ float __mulsc3(float, float, float, float);

inline void CheckCgemmComplexProduct(uint32_t ar, uint32_t ai, uint32_t br, uint32_t bi)
{
    CgbtFloatBits a{};
    CgbtFloatBits b{};
    CgbtFloatBits c{};
    CgbtFloatBits d{};
    a.bits = ar;
    b.bits = ai;
    c.bits = br;
    d.bits = bi;
    const auto actual = CgbtComplexMultiply({a.value, b.value}, {c.value, d.value});
    const auto expected = __mulsc3(a.value, b.value, c.value, d.value);
    const auto same = [](float x, float y) {
        return (std::isnan(x) && std::isnan(y)) || CgbtFloatBits{x}.bits == CgbtFloatBits{y}.bits;
    };
    ASSERT_TRUE(same(actual.real, __real__ expected) && same(actual.imag, __imag__ expected))
        << std::hex << ar << "," << ai << "," << br << "," << bi;
}

constexpr uint32_t CGEMM_COMPLEX_EDGES[] = {0U,          0x80000000U, 1U,          0x80000001U, 0x7fffffU,
                                            0x800000U,   0x3f800000U, 0xbf800000U, 0x7f7fffffU, 0xff7fffffU,
                                            0x7f800000U, 0xff800000U, 0x7fc00000U};

inline void CheckCgemmComplexFactors(uint32_t ar, uint32_t ai)
{
    for (auto br : CGEMM_COMPLEX_EDGES)
        for (auto bi : CGEMM_COMPLEX_EDGES)
            ASSERT_NO_FATAL_FAILURE(CheckCgemmComplexProduct(ar, ai, br, bi));
}

inline void CheckCgemmComplexRecovery()
{
    for (auto ar : CGEMM_COMPLEX_EDGES)
        for (auto ai : CGEMM_COMPLEX_EDGES)
            ASSERT_NO_FATAL_FAILURE(CheckCgemmComplexFactors(ar, ai));
}

TEST_F(CgemmBatchedTest, FusedMultiplyAddReference)
{
    ASSERT_NO_FATAL_FAILURE(CheckCgemmComplexRecovery());
    const float x = std::ldexp(1.0F + std::ldexp(1.0F, -23), 60);
    const float y = std::ldexp(1.0F - std::ldexp(1.0F, -23), 60);
    const float z = -std::ldexp(1.0F, 120);
    EXPECT_EQ(CgbtFma(x, y, z), -std::ldexp(1.0F, 74));
    EXPECT_EQ(CgbtFma(x, y, z), std::fma(x, y, z));
    std::mt19937 random(20261011);
    for (int sample = 0; sample < 1000000; ++sample) {
        CgbtFloatBits x{};
        CgbtFloatBits y{};
        CgbtFloatBits z{};
        CgbtFloatBits actual{};
        CgbtFloatBits expected{};
        x.bits = random();
        y.bits = random();
        z.bits = random();
        actual.value = CgbtFma(x.value, y.value, z.value);
        expected.value = std::fma(x.value, y.value, z.value);
        if (std::isnan(actual.value) && std::isnan(expected.value))
            continue;
        ASSERT_EQ(actual.bits, expected.bits) << "sample=" << sample;
    }
}

// Large alpha and cancelling real/imaginary products must follow the
// sequential complex reference, including its non-finite classifications.
TEST_F(CgemmBatchedTest, LargeAlphaZeroBetaCancellation)
{
    constexpr int extent = 8;
    std::vector<aclblasComplex> a(extent * extent, {1e18F, 1e18F});
    std::vector<aclblasComplex> b(extent * extent, {1e18F, 1e18F});
    for (int col = 0; col < extent; ++col)
        for (int row = 1; row < extent; ++row)
            b[col * extent + row] = {0.0F, 0.0F};
    std::vector<aclblasComplex> actual(extent * extent, {0.0F, 0.0F});
    const aclblasComplex* ap[] = {a.data()};
    const aclblasComplex* bp[] = {b.data()};
    aclblasComplex* cp[] = {actual.data()};
    const aclblasComplex alpha{1000.0F, 0.0F};
    const aclblasComplex beta{0.0F, 0.0F};
    ASSERT_EQ(
        aclblasCgemmBatched_npu(
            handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, extent, extent, extent, &alpha, ap, extent, bp, extent, &beta, cp,
            extent, 1),
        ACLBLAS_STATUS_SUCCESS);
    // N applies alpha to B before accumulation. Each real product is a
    // fused cancellation (finite); imaginary partial sums overflow and cancel.
    float sumReal = 0.0F;
    float sumImag = 0.0F;
    for (int k = 0; k < extent; ++k) {
        const auto v = CgbtComplexMultiply({alpha.real, alpha.imag}, {b[k].real, b[k].imag});
        sumReal += CgbtFma(a[0].real, v.real, -(a[0].imag * v.imag));
        sumImag += CgbtFma(a[0].real, v.imag, a[0].imag * v.real);
    }
    for (const auto& v : actual) {
        EXPECT_TRUE((std::isnan(v.real) && std::isnan(sumReal)) || v.real == sumReal);
        EXPECT_TRUE((std::isnan(v.imag) && std::isnan(sumImag)) || v.imag == sumImag);
    }
}

TEST_F(CgemmBatchedTest, PackedAlphaIntermediateOverflow)
{
    constexpr int size = 16;
    std::vector<aclblasComplex> a(size * size, {1e-20F, 0.0F});
    std::vector<aclblasComplex> b(size * size, {1e18F, 0.0F});
    std::vector<aclblasComplex> c(size * size, {0.0F, 0.0F});
    const aclblasComplex* ap[] = {a.data()};
    const aclblasComplex* bp[] = {b.data()};
    aclblasComplex* cp[] = {c.data()};
    const aclblasComplex alpha{1e30F, 0.0F};
    const aclblasComplex beta{0.0F, 0.0F};
    ASSERT_EQ(
        aclblasCgemmBatched_npu(
            handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, size, size, size, &alpha, ap, size, bp, size, &beta, cp, size, 1),
        ACLBLAS_STATUS_SUCCESS);
    for (const auto& value : c)
        EXPECT_EQ(value.real, std::numeric_limits<float>::infinity());
}

TEST_F(CgemmBatchedTest, LargeLeadingDimension)
{
    struct DeviceBuffers : GemmBatchedDeviceCtx {
        ~DeviceBuffers() { Cleanup(); }
    } device;
    device.Init(1);
    constexpr int ld = (1 << 29) + 2;
    const size_t bytes = (static_cast<size_t>(ld) + 2U) * sizeof(aclblasComplex);
    ASSERT_EQ(aclrtMalloc(&device.dAarray[0], bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    const aclblasComplex first[] = {{1, 0}, {2, 0}};
    const aclblasComplex second[] = {{3, 0}, {4, 0}};
    ASSERT_EQ(aclrtMemcpy(device.dAarray[0], bytes, first, sizeof(first), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    auto column = static_cast<aclblasComplex*>(device.dAarray[0]) + ld;
    ASSERT_EQ(aclrtMemcpy(column, sizeof(second), second, sizeof(second), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    const aclblasComplex identity[] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    aclblasComplex output[4] = {};
    ASSERT_EQ(AllocAndCopyBuffer(&device.dBarray[0], sizeof(identity), identity), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocAndCopyBuffer(&device.dCarray[0], sizeof(output), output), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocAndCopyPtrArray(&device.dAPtrArray, device.dAarray.data(), sizeof(void*)), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocAndCopyPtrArray(&device.dBPtrArray, device.dBarray.data(), sizeof(void*)), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocAndCopyPtrArray(&device.dCPtrArray, device.dCarray.data(), sizeof(void*)), ACLBLAS_STATUS_SUCCESS);
    const aclblasComplex alpha{1, 0};
    const aclblasComplex beta{0, 0};
    ASSERT_EQ(
        aclblasCgemmBatched(
            handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 2, 2, 2, &alpha,
            reinterpret_cast<const aclblasComplex* const*>(device.dAPtrArray), ld,
            reinterpret_cast<const aclblasComplex* const*>(device.dBPtrArray), 2, &beta,
            reinterpret_cast<aclblasComplex* const*>(device.dCPtrArray), 2, 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(output, sizeof(output), device.dCarray[0], sizeof(output), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(output[i].real, static_cast<float>(i + 1));
        EXPECT_EQ(output[i].imag, 0.0F);
    }
}

struct CgemmOffsetCase {
    struct Buffers : GemmBatchedDeviceCtx {
        ~Buffers() { Cleanup(); }
    } device;
    int m, n, k, lda, ldb, ldc, batches;
    bool compactInput;
    aclblasOperation_t ta, tb;
    std::vector<std::vector<aclblasComplex>> a, b, c, golden;
    std::vector<const aclblasComplex*> da, db, ga, gb;
    std::vector<aclblasComplex*> dc, gc;

    CgemmOffsetCase(int rows, bool compact, aclblasOperation_t transA, aclblasOperation_t transB)
        : m(rows), compactInput(compact), ta(transA), tb(transB)
    {
        const bool square = m == 16 || m == 24 || m == 32;
        n = square ? m : Columns(m, compactInput);
        k = square ? m : Reduction(m, compactInput);
        lda = (ta == ACLBLAS_OP_N ? m : k) + (compactInput ? 0 : 3);
        ldb = (tb == ACLBLAS_OP_N ? k : n) + (compactInput ? 0 : 5);
        ldc = square && compactInput ? m : m + 7;
        batches = m == 24 || m == 31 || m == 32 || (m == 16 && compactInput) ? 181 : (m == 16 || compactInput ? 43 : 3);
        device.Init(batches);
        a.resize(batches);
        b.resize(batches);
        c.resize(batches);
        da.resize(batches);
        db.resize(batches);
        ga.resize(batches);
        gb.resize(batches);
        dc.resize(batches);
        gc.resize(batches);
    }

    static int Columns(int m, bool compact)
    {
        if (m == 67)
            return 71;
        if (m == 61)
            return 57;
        if (m == 47)
            return 39;
        if (m == 31)
            return 29;
        return compact ? m - 4 : 23;
    }

    static int Reduction(int m, bool compact)
    {
        if (m == 61)
            return 43;
        if (m == 47)
            return 23;
        if (m == 31)
            return 17;
        return compact ? m - 8 : 13;
    }

    void Allocate()
    {
        const auto sizes = ComputeCgemmBatchedBufferSizes(ta, tb, m, n, k, lda, ldb, ldc);
        for (int q = 0; q < batches; ++q) {
            a[q] = makeComplexArray(
                sizes.aBytes / sizeof(aclblasComplex) + batches + 8, parseFill("RANDOM_NORM_2_2"), 301 + q);
            b[q] = makeComplexArray(
                sizes.bBytes / sizeof(aclblasComplex) + batches + 8, parseFill("RANDOM_NORM_2_2"), 401 + q);
            c[q] = makeComplexArray(
                sizes.cBytes / sizeof(aclblasComplex) + batches + 8, parseFill("RANDOM_NORM_2_2"), 501 + q);
            ASSERT_EQ(
                AllocAndCopyBuffer(&device.dAarray[q], a[q].size() * sizeof(aclblasComplex), a[q].data()),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                AllocAndCopyBuffer(&device.dBarray[q], b[q].size() * sizeof(aclblasComplex), b[q].data()),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                AllocAndCopyBuffer(&device.dCarray[q], c[q].size() * sizeof(aclblasComplex), c[q].data()),
                ACLBLAS_STATUS_SUCCESS);
        }
        golden = c;
    }

    void SetPointers()
    {
        for (int q = 0; q < batches; ++q) {
            const int ai = batches - 1 - q;
            const int bi = (q + 1) % batches;
            const int ci = (q + 2) % batches;
            da[q] = static_cast<const aclblasComplex*>(device.dAarray[ai]) + ai + 1;
            db[q] = static_cast<const aclblasComplex*>(device.dBarray[bi]) + bi + 1;
            dc[q] = static_cast<aclblasComplex*>(device.dCarray[ci]) + ci + 1;
            ga[q] = a[ai].data() + ai + 1;
            gb[q] = b[bi].data() + bi + 1;
            gc[q] = golden[ci].data() + ci + 1;
        }
        const size_t pointerBytes = batches * sizeof(void*);
        ASSERT_EQ(AllocAndCopyPtrArray(&device.dAPtrArray, da.data(), pointerBytes), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(AllocAndCopyPtrArray(&device.dBPtrArray, db.data(), pointerBytes), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(AllocAndCopyPtrArray(&device.dCPtrArray, dc.data(), pointerBytes), ACLBLAS_STATUS_SUCCESS);
    }

    void Run(aclblasHandle_t handle_, aclblasHandle_t limitedHandle, aclrtStream stream_)
    {
        const bool directOutput = compactInput || m == 16 || m == 17 || m == 33;
        const auto testHandle = compactInput ? handle_ : limitedHandle;
        const aclblasComplex alpha = directOutput ? aclblasComplex{1.0f, 0.0f} : aclblasComplex{0.7f, -0.3f};
        const aclblasComplex beta = directOutput ? aclblasComplex{0.0f, 0.0f} : aclblasComplex{-0.2f, 0.4f};
        for (int iteration = 0; iteration < 2; ++iteration) {
            ASSERT_EQ(
                aclblasCgemmBatched(
                    testHandle, ta, tb, m, n, k, &alpha,
                    reinterpret_cast<const aclblasComplex* const*>(device.dAPtrArray), lda,
                    reinterpret_cast<const aclblasComplex* const*>(device.dBPtrArray), ldb, &beta,
                    reinterpret_cast<aclblasComplex* const*>(device.dCPtrArray), ldc, batches),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                aclblasCgemmBatched_cpu(
                    handle_, ta, tb, m, n, k, &alpha, ga.data(), lda, gb.data(), ldb, &beta, gc.data(), ldc, batches),
                ACLBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    }

    void VerifyOutput(int q, const std::vector<aclblasComplex>& actual)
    {
        size_t outsideFailures = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
            const int64_t logical = static_cast<int64_t>(index) - q - 1;
            const bool inMatrix = logical >= 0 && logical < static_cast<int64_t>(ldc) * n && logical % ldc < m;
            if (!inMatrix) {
                outsideFailures += std::memcmp(&actual[index], &c[q][index], sizeof(aclblasComplex)) != 0;
            } else {
                EXPECT_NEAR(actual[index].real, golden[q][index].real, 1e-4) << "q=" << q << " index=" << index;
                EXPECT_NEAR(actual[index].imag, golden[q][index].imag, 1e-4);
            }
        }
        EXPECT_EQ(outsideFailures, 0U) << "prefix/suffix/padding overwritten";
    }

    void VerifyAllocation(int q)
    {
        auto actual = c[q];
        const size_t bytes = actual.size() * sizeof(aclblasComplex);
        ASSERT_EQ(aclrtMemcpy(actual.data(), bytes, device.dCarray[q], bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        VerifyOutput(q, actual);
        for (const auto& input : {std::make_pair(&a[q], device.dAarray[q]), std::make_pair(&b[q], device.dBarray[q])}) {
            auto readback = *input.first;
            const size_t inputBytes = readback.size() * sizeof(aclblasComplex);
            ASSERT_EQ(
                aclrtMemcpy(readback.data(), inputBytes, input.second, inputBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
            EXPECT_EQ(std::memcmp(readback.data(), input.first->data(), inputBytes), 0) << "input modified";
        }
    }
};

TEST_F(CgemmBatchedTest, OffsetPointersAndStreamOrdering)
{
    struct WorkspaceHandle {
        aclblasHandle_t handle = nullptr;
        void* workspace = nullptr;
        ~WorkspaceHandle()
        {
            if (handle)
                aclblasDestroy(handle);
            if (workspace)
                aclrtFree(workspace);
        }
    } limited;
    ASSERT_EQ(aclblasCreate(&limited.handle), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSetStream(limited.handle, stream_), ACLBLAS_STATUS_SUCCESS);
    // 67x71, k=13 needs 109568–111616 bytes per batch with aligned planes.
    // The 250000-byte workspace forces rounds of 2+1 for every transpose pair.
    ASSERT_EQ(aclrtMalloc(&limited.workspace, 250000, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclblasSetWorkspace(limited.handle, limited.workspace, 250000), ACLBLAS_STATUS_SUCCESS);
    const aclblasOperation_t ops[] = {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C};
    const std::pair<int, bool> shapes[] = {
        std::make_pair(7, false),  std::make_pair(16, false), std::make_pair(16, true), std::make_pair(17, false),
        std::make_pair(23, true),  std::make_pair(24, true),  std::make_pair(31, true), std::make_pair(32, true),
        std::make_pair(33, false), std::make_pair(39, true),  std::make_pair(47, true), std::make_pair(55, true),
        std::make_pair(61, true),  std::make_pair(67, false)};
    for (const auto& shape : shapes) {
        const int m = shape.first;
        const bool compactInput = shape.second;
        // Compact rectangular inputs still use independent, offset and
        // permuted pointers. 181 batches exercise three producer rounds
        // with an uneven final group on the 20-core acceptance device.
        for (auto ta : ops) {
            for (auto tb : ops) {
                SCOPED_TRACE(
                    ::testing::Message() << "m=" << m << " compact=" << compactInput << " transA=" << ta
                                         << " transB=" << tb);
                // 0xff bytes are NaNs: unwritten workspace padding must not
                // influence the result or the vector exception status.
                ASSERT_EQ(aclrtMemset(limited.workspace, 250000, 0xff, 250000), ACL_SUCCESS);
                CgemmOffsetCase test(m, compactInput, ta, tb);
                ASSERT_NO_FATAL_FAILURE(test.Allocate());
                ASSERT_NO_FATAL_FAILURE(test.SetPointers());
                ASSERT_NO_FATAL_FAILURE(test.Run(handle_, limited.handle, stream_));
                for (int q = 0; q < test.batches; ++q)
                    ASSERT_NO_FATAL_FAILURE(test.VerifyAllocation(q));
            }
        }
    }
}

struct CgemmConcurrentRun {
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    GemmBatchedDeviceCtx device;
    int m, n, k, batches = 43;
    std::vector<std::vector<aclblasComplex>> a, b, expected;
    std::vector<const aclblasComplex*> hostA, hostB, deviceA, deviceB;
    std::vector<aclblasComplex*> hostC, deviceC;
    explicit CgemmConcurrentRun(int extent) : m(extent), n(extent - 2), k(extent - 8) {}
    ~CgemmConcurrentRun()
    {
        if (stream)
            aclrtSynchronizeStream(stream);
        device.Cleanup();
        if (handle)
            aclblasDestroy(handle);
        if (stream)
            aclrtDestroyStream(stream);
    }
};

inline void AllocateCgemmConcurrentBatch(CgemmConcurrentRun* run, int q)
{
    auto& a = run->a[q];
    auto& b = run->b[q];
    auto& c = run->expected[q];
    a = makeComplexArray(run->k * run->m, parseFill("RANDOM_NORM_2_2"), 710 + q);
    b = makeComplexArray(run->n * run->k, parseFill("RANDOM_NORM_2_2"), 810 + q);
    c = makeComplexArray((run->m + 3) * run->n, parseFill("RANDOM_NORM_2_2"), 910 + q);
    ASSERT_EQ(
        AllocAndCopyBuffer(&run->device.dAarray[q], a.size() * sizeof(aclblasComplex), a.data()),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        AllocAndCopyBuffer(&run->device.dBarray[q], b.size() * sizeof(aclblasComplex), b.data()),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        AllocAndCopyBuffer(&run->device.dCarray[q], c.size() * sizeof(aclblasComplex), c.data()),
        ACLBLAS_STATUS_SUCCESS);
    run->hostA.push_back(a.data());
    run->hostB.push_back(b.data());
    run->hostC.push_back(c.data());
    run->deviceA.push_back(static_cast<const aclblasComplex*>(run->device.dAarray[q]));
    run->deviceB.push_back(static_cast<const aclblasComplex*>(run->device.dBarray[q]));
    run->deviceC.push_back(static_cast<aclblasComplex*>(run->device.dCarray[q]));
}

inline void InitializeCgemmConcurrentRun(CgemmConcurrentRun* run, aclblasHandle_t handle_)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};

    ASSERT_EQ(aclrtCreateStream(&run->stream), ACL_SUCCESS);
    ASSERT_EQ(aclblasCreate(&run->handle), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSetStream(run->handle, run->stream), ACLBLAS_STATUS_SUCCESS);
    run->device.Init(run->batches);
    run->a.resize(run->batches);
    run->b.resize(run->batches);
    run->expected.resize(run->batches);
    for (int q = 0; q < run->batches; ++q) {
        ASSERT_NO_FATAL_FAILURE(AllocateCgemmConcurrentBatch(run, q));
    }
    const size_t bytes = run->batches * sizeof(void*);
    ASSERT_EQ(AllocAndCopyPtrArray(&run->device.dAPtrArray, run->deviceA.data(), bytes), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocAndCopyPtrArray(&run->device.dBPtrArray, run->deviceB.data(), bytes), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocAndCopyPtrArray(&run->device.dCPtrArray, run->deviceC.data(), bytes), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgemmBatched_cpu(
            handle_, ACLBLAS_OP_C, ACLBLAS_OP_C, run->m, run->n, run->k, &alpha, run->hostA.data(), run->k,
            run->hostB.data(), run->n, &beta, run->hostC.data(), run->m + 3, run->batches),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CgemmBatchedTest, ConcurrentStreamsIndependentWorkspaces)
{
    CgemmConcurrentRun first(31);
    CgemmConcurrentRun second(61);
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    for (auto* run : {&first, &second}) {
        ASSERT_NO_FATAL_FAILURE(InitializeCgemmConcurrentRun(run, handle_));
    }
    // Queue both streams repeatedly before waiting on either stream. Each
    // handle owns its workspace; producer/consumer flags must remain local.
    for (int iteration = 0; iteration < 4; ++iteration) {
        for (auto* run : {&first, &second}) {
            ASSERT_EQ(
                aclblasCgemmBatched(
                    run->handle, ACLBLAS_OP_C, ACLBLAS_OP_C, run->m, run->n, run->k, &alpha,
                    reinterpret_cast<const aclblasComplex* const*>(run->device.dAPtrArray), run->k,
                    reinterpret_cast<const aclblasComplex* const*>(run->device.dBPtrArray), run->n, &beta,
                    reinterpret_cast<aclblasComplex* const*>(run->device.dCPtrArray), run->m + 3, run->batches),
                ACLBLAS_STATUS_SUCCESS);
        }
    }
    for (auto* run : {&first, &second}) {
        ASSERT_EQ(aclrtSynchronizeStream(run->stream), ACL_SUCCESS);
        for (int q = 0; q < run->batches; ++q) {
            auto actual = run->expected[q];
            const size_t bytes = actual.size() * sizeof(aclblasComplex);
            ASSERT_EQ(
                aclrtMemcpy(actual.data(), bytes, run->device.dCarray[q], bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
            for (size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i].real, run->expected[q][i].real, 1e-4);
                EXPECT_NEAR(actual[i].imag, run->expected[q][i].imag, 1e-4);
            }
        }
    }
}
