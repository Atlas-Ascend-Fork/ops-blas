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

#include <atomic>
#include <cstring>
#include <thread>

#include "cann_ops_blas.h"
#include "csyr2_test_resources.h"

// NPU regressions: large-offset cases can require more than 8 GiB of free HBM.
namespace {

inline aclError Csyr2CopyToDevice(void** device, const void* host, size_t bytes)
{
    auto status = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        return status;
    }
    return aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

class Csyr2ReviewTest : public BlasTest<int> {
protected:
    void CheckOffsets(int n, int lda, int incx, aclblasFillMode_t uplo)
    {
        Csyr2ReviewBuffers buffers;
        buffers.stream = stream_;
        const int64_t absInc = incx < 0 ? -static_cast<int64_t>(incx) : incx;
        const size_t xBytes = (static_cast<size_t>(n - 1) * absInc + 1) * sizeof(aclblasComplex);
        const size_t yBytes = static_cast<size_t>(n) * sizeof(aclblasComplex);
        const size_t aBytes = static_cast<size_t>(n) * lda * sizeof(aclblasComplex);
        ASSERT_EQ(aclrtMalloc(&buffers.x, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&buffers.y, yBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&buffers.a, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(buffers.a, aBytes, 0, aBytes), ACL_SUCCESS);
        std::vector<aclblasComplex> x(n);
        std::vector<aclblasComplex> y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = {static_cast<float>(i % 7 + 1), 0.0f};
            y[i] = {static_cast<float>(i % 5 + 1), 0.0f};
        }
        if (incx == 1) {
            ASSERT_EQ(aclrtMemcpy(buffers.x, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        } else {
            const int64_t start = incx < 0 ? static_cast<int64_t>(n - 1) * absInc : 0;
            for (int i = 0; i < n; ++i) {
                auto* dst = static_cast<aclblasComplex*>(buffers.x) + start + static_cast<int64_t>(i) * incx;
                ASSERT_EQ(aclrtMemcpy(dst, sizeof(x[i]), &x[i], sizeof(x[i]), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
            }
        }
        ASSERT_EQ(aclrtMemcpy(buffers.y, yBytes, y.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        const aclblasComplex alpha{0.5f, 0.25f};
        const aclblasStatus_t launchStatus = aclblasCsyr2(
            handle_, uplo, n, &alpha, static_cast<aclblasComplex*>(buffers.x), incx,
            static_cast<aclblasComplex*>(buffers.y), 1, static_cast<aclblasComplex*>(buffers.a), lda);
        ASSERT_EQ(launchStatus, ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        // Sample both sides of the diagonal, the 32-bit float-offset boundary,
        // and the last column. Exact integer/binary-fraction data needs no tolerance.
        for (int col : {0, n / 2, n - 1}) {
            for (int row : {0, col, n - 1}) {
                aclblasComplex actual{};
                auto* src = static_cast<aclblasComplex*>(buffers.a) + static_cast<int64_t>(col) * lda + row;
                ASSERT_EQ(
                    aclrtMemcpy(&actual, sizeof(actual), src, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
                const bool selected = uplo == ACLBLAS_UPPER ? row <= col : row >= col;
                const float sum = selected ? x[row].real * y[col].real + y[row].real * x[col].real : 0.0f;
                EXPECT_EQ(actual.real, sum * alpha.real) << row << "," << col;
                EXPECT_EQ(actual.imag, sum * alpha.imag) << row << "," << col;
            }
        }
    }

    void CheckDirectStorageResult(
        int n, aclblasFillMode_t uplo, int lda, size_t guard, const aclblasComplex& initial,
        const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y,
        const std::vector<aclblasComplex>& actual)
    {
        if (lda <= 0) {
            ADD_FAILURE() << "lda must be positive, got " << lda;
            return;
        }
        const size_t matrixElems = static_cast<size_t>(n) * lda;
        // Small integer complex inputs make every expected component exactly representable.
        // Check all selected elements, the other triangle, lda padding, and both GM guards.
        int row = 0;
        int col = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
            aclblasComplex expected = initial;
            const bool inMatrix = index >= guard && index < guard + matrixElems;
            if (inMatrix) {
                if (row < n && (uplo == ACLBLAS_UPPER ? row <= col : row >= col)) {
                    expected.real += x[row].real * y[col].real - x[row].imag * y[col].imag + y[row].real * x[col].real -
                                     y[row].imag * x[col].imag;
                    expected.imag += x[row].real * y[col].imag + x[row].imag * y[col].real + y[row].real * x[col].imag +
                                     y[row].imag * x[col].real;
                }
            }
            ASSERT_EQ(actual[index].real, expected.real) << "index=" << index;
            ASSERT_EQ(actual[index].imag, expected.imag) << "index=" << index;
            if (inMatrix) {
                ++row;
                if (row == lda) {
                    row = 0;
                    ++col;
                }
            }
        }
    }

    void CheckDirectStorage(int n, aclblasFillMode_t uplo)
    {
        SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << static_cast<int>(uplo));
        const int lda = n + 3;
        constexpr size_t guard = 8;
        const size_t matrixElems = static_cast<size_t>(n) * lda;
        const aclblasComplex alpha{1.0f, 0.0f};
        const aclblasComplex initial{17.0f, -23.0f};
        // Exercise the same shapes through the public API. The replacement
        // implementation has no legacy tileRow/xyReuse/tiling-key fields.
        std::vector<aclblasComplex> x(n);
        std::vector<aclblasComplex> y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = {static_cast<float>(i % 7 - 3), static_cast<float>(i % 5 - 2)};
            y[i] = {static_cast<float>(i % 3 + 1), static_cast<float>(i % 11 - 5)};
        }
        std::vector<aclblasComplex> actual(matrixElems + 2 * guard, initial);
        Csyr2ReviewBuffers buffers;
        buffers.stream = stream_;
        const size_t vectorBytes = static_cast<size_t>(n) * sizeof(aclblasComplex);
        const size_t matrixBytes = actual.size() * sizeof(aclblasComplex);
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.x, x.data(), vectorBytes), ACL_SUCCESS);
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.y, y.data(), vectorBytes), ACL_SUCCESS);
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.a, actual.data(), matrixBytes), ACL_SUCCESS);
        const aclblasStatus_t launchStatus = aclblasCsyr2(
            handle_, uplo, n, &alpha, static_cast<aclblasComplex*>(buffers.x), 1,
            static_cast<aclblasComplex*>(buffers.y), 1, static_cast<aclblasComplex*>(buffers.a) + guard, lda);
        ASSERT_EQ(launchStatus, ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(actual.data(), matrixBytes, buffers.a, matrixBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        ASSERT_NO_FATAL_FAILURE(CheckDirectStorageResult(n, uplo, lda, guard, initial, x, y, actual));
    }

    void CheckDeviceAlpha(aclblasFillMode_t uplo)
    {
        constexpr int n = 8;
        constexpr int lda = n + 3;
        constexpr size_t guard = 8;
        const aclblasComplex alpha{0.5f, 0.25f};
        const aclblasComplex initial{17.0f, -23.0f};
        std::vector<aclblasComplex> x(n);
        std::vector<aclblasComplex> y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = {static_cast<float>(i + 1), static_cast<float>(i % 3)};
            y[i] = {static_cast<float>(i % 5), static_cast<float>(i - 2)};
        }
        std::vector<aclblasComplex> actual(n * lda + 2 * guard, initial);
        auto golden = actual;
        aclblasCsyr2_cpu(uplo, n, alpha, x.data(), 1, y.data(), 1, golden.data() + guard, lda);
        const size_t matrixBytes = actual.size() * sizeof(aclblasComplex);
        const size_t vectorBytes = x.size() * sizeof(aclblasComplex);
        Csyr2ReviewBuffers buffers;
        buffers.stream = stream_;
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.x, x.data(), vectorBytes), ACL_SUCCESS);
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.y, y.data(), vectorBytes), ACL_SUCCESS);
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.a, actual.data(), matrixBytes), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&buffers.alpha, sizeof(alpha), ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpyAsync(buffers.alpha, sizeof(alpha), &alpha, sizeof(alpha), ACL_MEMCPY_HOST_TO_DEVICE, stream_),
            ACL_SUCCESS);
        ASSERT_EQ(
            aclblasCsyr2(
                handle_, uplo, n, static_cast<aclblasComplex*>(buffers.alpha), static_cast<aclblasComplex*>(buffers.x),
                1, static_cast<aclblasComplex*>(buffers.y), 1, static_cast<aclblasComplex*>(buffers.a) + guard, lda),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(actual.data(), matrixBytes, buffers.a, matrixBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        EXPECT_EQ(std::memcmp(actual.data(), golden.data(), matrixBytes), 0);
    }

    void CheckLargeLdaColumns(const aclblasComplex* matrix, int lda, const std::vector<aclblasComplex>& x)
    {
        constexpr int n = 5;
        constexpr size_t guard = 8;
        const aclblasComplex initial{17.0f, -23.0f};
        std::vector<aclblasComplex> column(n + guard);
        const size_t columnBytes = column.size() * sizeof(aclblasComplex);
        for (int col = 0; col < n; ++col) {
            ASSERT_EQ(
                aclrtMemcpy(
                    column.data(), columnBytes, matrix + static_cast<uint64_t>(col) * lda, columnBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
            for (size_t row = 0; row < column.size(); ++row) {
                aclblasComplex expected = initial;
                if (row < n && row >= static_cast<size_t>(col)) {
                    expected.real += 2.0f * x[row].real * x[col].real;
                }
                EXPECT_EQ(column[row].real, expected.real) << col << "," << row;
                EXPECT_EQ(column[row].imag, expected.imag) << col << "," << row;
            }
        }
    }

    void CheckLargeLda()
    {
        constexpr int n = 5;
        // At the first four-column LOWER band, 8*(lda-rows+4) wraps to zero.
        constexpr int lda = (1 << 29) + 1;
        constexpr size_t guard = 8;
        const aclblasComplex alpha{1.0f, 0.0f};
        const aclblasComplex initial{17.0f, -23.0f};
        std::vector<aclblasComplex> x(n);
        for (int i = 0; i < n; ++i)
            x[i] = {static_cast<float>(i + 1), 0.0f};
        const size_t matrixElems = static_cast<uint64_t>(n - 1) * lda + n + 2 * guard;
        std::vector<aclblasComplex> column(n + guard, initial);
        const size_t columnBytes = column.size() * sizeof(aclblasComplex);
        Csyr2ReviewBuffers buffers;
        buffers.stream = stream_;
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.x, x.data(), x.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMalloc(&buffers.a, matrixElems * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        auto* matrix = static_cast<aclblasComplex*>(buffers.a) + guard;
        ASSERT_EQ(
            aclrtMemcpy(
                buffers.a, guard * sizeof(initial), column.data(), guard * sizeof(initial), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
        for (int col = 0; col < n; ++col) {
            ASSERT_EQ(
                aclrtMemcpy(
                    matrix + static_cast<uint64_t>(col) * lda, columnBytes, column.data(), columnBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_SUCCESS);
        }
        ASSERT_EQ(
            aclblasCsyr2(
                handle_, ACLBLAS_LOWER, n, &alpha, static_cast<aclblasComplex*>(buffers.x), 1,
                static_cast<aclblasComplex*>(buffers.x), 1, matrix, lda),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_NO_FATAL_FAILURE(CheckLargeLdaColumns(matrix, lda, x));
        ASSERT_EQ(
            aclrtMemcpy(
                column.data(), guard * sizeof(initial), buffers.a, guard * sizeof(initial), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        for (size_t row = 0; row < guard; ++row) {
            EXPECT_EQ(column[row].real, initial.real);
            EXPECT_EQ(column[row].imag, initial.imag);
        }
    }
};

TEST_F(Csyr2ReviewTest, DeviceAlpha)
{
    ASSERT_NO_FATAL_FAILURE(CheckDeviceAlpha(ACLBLAS_UPPER));
    ASSERT_NO_FATAL_FAILURE(CheckDeviceAlpha(ACLBLAS_LOWER));
}

TEST_F(Csyr2ReviewTest, DeviceAlphaZeroNullInputs)
{
    Csyr2ReviewBuffers buffers;
    buffers.stream = stream_;
    ASSERT_EQ(aclrtMalloc(&buffers.alpha, sizeof(aclblasComplex), ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
    // The scalar producer is queued on the same stream as the BLAS call.
    ASSERT_EQ(aclrtMemsetAsync(buffers.alpha, sizeof(aclblasComplex), 0, sizeof(aclblasComplex), stream_), ACL_SUCCESS);
    EXPECT_EQ(
        aclblasCsyr2(
            handle_, ACLBLAS_LOWER, 8, static_cast<aclblasComplex*>(buffers.alpha), nullptr, 1, nullptr, 1, nullptr, 8),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(Csyr2ReviewTest, LargeLdaByteStride) { ASSERT_NO_FATAL_FAILURE(CheckLargeLda()); }

TEST_F(Csyr2ReviewTest, DISABLED_LargeMatrixOffsets)
{
    CheckOffsets(33000, 33000, 1, ACLBLAS_UPPER);
    CheckOffsets(33000, 33000, 1, ACLBLAS_LOWER);
}

TEST_F(Csyr2ReviewTest, LargeVectorOffsets)
{
    CheckOffsets(2, 2, 1073741825, ACLBLAS_UPPER);
    CheckOffsets(2, 2, -1073741825, ACLBLAS_LOWER);
    CheckOffsets(1, 1, (-2147483647 - 1), ACLBLAS_UPPER);
}

TEST_F(Csyr2ReviewTest, DirectStorageBounds)
{
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(1024, ACLBLAS_UPPER)); // Largest oct layout.
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(1025, ACLBLAS_UPPER)); // First quad layout.
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(1408, ACLBLAS_UPPER)); // Largest quad layout.
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(1408, ACLBLAS_LOWER)); // Largest rotated layout.
}

TEST_F(Csyr2ReviewTest, MemorySanitizerPaths)
{
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(8, ACLBLAS_UPPER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(32, ACLBLAS_UPPER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(65, ACLBLAS_UPPER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(255, ACLBLAS_LOWER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(257, ACLBLAS_UPPER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(1024, ACLBLAS_LOWER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(2049, ACLBLAS_LOWER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(2049, ACLBLAS_UPPER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(3073, ACLBLAS_UPPER));
    ASSERT_NO_FATAL_FAILURE(CheckDirectStorage(4097, ACLBLAS_LOWER));
    ASSERT_NO_FATAL_FAILURE(CheckOffsets(65, 69, 1, ACLBLAS_LOWER));
}

struct Csyr2ConcurrentBatch {
    static constexpr int kCalls = 96;
    static constexpr int kPitch = 72;
    static constexpr size_t kMatrixElems = kPitch * kPitch;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    Csyr2ReviewBuffers buffers;

    ~Csyr2ConcurrentBatch()
    {
        if (context != nullptr) {
            aclrtSetCurrentContext(context);
        }
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        buffers.stream = nullptr;
    }

    void Prepare(aclrtContext ctx)
    {
        context = ctx;
        ASSERT_EQ(aclrtSetCurrentContext(context), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
        buffers.stream = stream;
        ASSERT_EQ(aclblasCreate(&handle), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclblasSetStream(handle, stream), ACLBLAS_STATUS_SUCCESS);
        std::vector<aclblasComplex> x(kPitch * 2, {1.0f, 0.0f});
        std::vector<aclblasComplex> y(kPitch, {2.0f, 1.0f});
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.x, x.data(), x.size() * sizeof(x[0])), ACL_SUCCESS);
        ASSERT_EQ(Csyr2CopyToDevice(&buffers.y, y.data(), y.size() * sizeof(y[0])), ACL_SUCCESS);
        const size_t bytes = kCalls * kMatrixElems * sizeof(aclblasComplex);
        ASSERT_EQ(aclrtMalloc(&buffers.a, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(buffers.a, bytes, 0, bytes), ACL_SUCCESS);
    }

    void Enqueue(int worker, std::atomic<int>& failures)
    {
        if (aclrtSetCurrentContext(context) != ACL_SUCCESS) {
            ++failures;
            return;
        }
        for (int call = 0; call < kCalls; ++call) {
            const int n = 17 + call % 48;
            const int lda = n + call % 3;
            const aclblasComplex alpha{
                static_cast<float>(1 + (call + worker) % 4), static_cast<float>((call + worker) % 3)};
            const auto uplo = call % 2 == 0 ? ACLBLAS_UPPER : ACLBLAS_LOWER;
            auto* a = static_cast<aclblasComplex*>(buffers.a) + call * kMatrixElems;
            const aclblasStatus_t launchStatus = aclblasCsyr2(
                handle, uplo, n, &alpha, static_cast<aclblasComplex*>(buffers.x), call % 2 == 0 ? 2 : -2,
                static_cast<aclblasComplex*>(buffers.y), 1, a, lda);
            if (launchStatus != ACLBLAS_STATUS_SUCCESS) {
                ++failures;
                return;
            }
        }
        // Return and destroy thread-local state while queued work may still run.
    }

    void Verify(int worker)
    {
        ASSERT_EQ(aclrtSetCurrentContext(context), ACL_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        std::vector<aclblasComplex> actual(kCalls * kMatrixElems);
        const size_t bytes = actual.size() * sizeof(actual[0]);
        ASSERT_EQ(aclrtMemcpy(actual.data(), bytes, buffers.a, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        for (int call = 0; call < kCalls; ++call) {
            const int n = 17 + call % 48;
            const int lda = n + call % 3;
            const float ar = static_cast<float>(1 + (call + worker) % 4);
            const float ai = static_cast<float>((call + worker) % 3);
            for (size_t index = 0; index < kMatrixElems; ++index) {
                const int row = static_cast<int>(index % lda);
                const int col = static_cast<int>(index / lda);
                const bool selected = row < n && col < n && (call % 2 == 0 ? row <= col : row >= col);
                const auto value = actual[call * kMatrixElems + index];
                ASSERT_EQ(value.real, selected ? 4 * ar - 2 * ai : 0.0f) << worker << "," << call << "," << index;
                ASSERT_EQ(value.imag, selected ? 2 * ar + 4 * ai : 0.0f) << worker << "," << call << "," << index;
            }
        }
    }
};

void Csyr2CheckConcurrentContexts(aclrtContext first, aclrtContext second)
{
    Csyr2ConcurrentBatch batches[4];
    for (int i = 0; i < 4; ++i) {
        batches[i].Prepare(i % 2 == 0 ? first : second);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
    std::atomic<int> failures{0};
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::thread workers[4];
    for (int i = 0; i < 4; ++i) {
        workers[i] = std::thread([&, i] {
            ++ready;
            while (!start.load())
                std::this_thread::yield();
            batches[i].Enqueue(i, failures);
        });
    }
    while (ready.load() != 4)
        std::this_thread::yield();
    start = true;
    for (auto& worker : workers)
        worker.join();
    EXPECT_EQ(failures.load(), 0);
    for (int i = 0; i < 4; ++i)
        batches[i].Verify(i);
}

TEST_F(Csyr2ReviewTest, ConcurrentStreams)
{
    aclrtContext context = nullptr;
    ASSERT_EQ(aclrtGetCurrentContext(&context), ACL_SUCCESS);
    Csyr2CheckConcurrentContexts(context, context);
    ASSERT_EQ(aclrtSetCurrentContext(context), ACL_SUCCESS);
}

TEST_F(Csyr2ReviewTest, DISABLED_ConcurrentDevices)
{
    // Require an explicit second accessible device; never silently pass on one card.
    const char* secondId = std::getenv("CSYR2_REVIEW_SECOND_DEVICE");
    ASSERT_NE(secondId, nullptr);
    char* end = nullptr;
    const long device = std::strtol(secondId, &end, 10);
    ASSERT_NE(end, secondId);
    ASSERT_EQ(*end, '\0');
    ASSERT_GE(device, 0);
    ASSERT_LE(device, 2147483647L);
    ASSERT_NE(device, blas_test_detail::RuntimeDeviceId());
    aclrtContext first = nullptr;
    aclrtContext second = nullptr;
    ASSERT_EQ(aclrtGetCurrentContext(&first), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateContext(&second, static_cast<int32_t>(device)), ACL_SUCCESS);
    Csyr2CheckConcurrentContexts(first, second);
    EXPECT_EQ(aclrtDestroyContext(second), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetCurrentContext(first), ACL_SUCCESS);
}

} // namespace
