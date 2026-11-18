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
#include <array>
#include <memory>
#include "blas_test.h"
#include "device.h"
#include "chbmv_test_data.h"

namespace {
using ChbmvEdgeTest = BlasTest<int>;

std::unique_ptr<DeviceBuffer> Upload(const std::vector<aclblasComplex>& v)
{
    if (v.empty()) {
        return nullptr;
    }
    auto b = std::make_unique<DeviceBuffer>(v.size() * sizeof(aclblasComplex));
    b->copyFromHost(v.data(), v.size() * sizeof(aclblasComplex));
    return b;
}
aclblasComplex* Pointer(const std::unique_ptr<DeviceBuffer>& b)
{
    return b ? static_cast<aclblasComplex*>(b->ptr()) : nullptr;
}

static void CheckCase(aclblasHandle_t handle, aclrtStream stream, const ChbmvParam& p, bool poisonUnused = false)
{
    chbmv_test_data::Data data(p, poisonUnused);
    auto a = Upload(data.a);
    auto x = Upload(data.x);
    auto y = Upload(data.y);
    auto invoke = [&]() {
        return aclblasChbmv(
            handle, p.uplo, p.n, p.k, p.nullAlpha ? nullptr : &p.alpha, Pointer(a), p.lda, Pointer(x), p.incx,
            p.nullBeta ? nullptr : &p.beta, Pointer(y), p.incy);
    };
    ASSERT_EQ(invoke(), p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0) {
        return;
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    auto actual = data.y;
    y->copyToHost(actual.data(), actual.size() * sizeof(aclblasComplex));
    const auto golden = ChbmvReference::Golden(p, data.a, data.x, data.y);
    EXPECT_TRUE(chbmv_test_data::ComponentMatches(p, actual, golden, false));
    EXPECT_TRUE(chbmv_test_data::ComponentMatches(p, actual, golden, true));
    EXPECT_TRUE(chbmv_test_data::GapsUnchanged(p, actual, data.y));
    auto aAfter = data.a;
    auto xAfter = data.x;
    a->copyToHost(aAfter.data(), aAfter.size() * sizeof(aclblasComplex));
    x->copyToHost(xAfter.data(), xAfter.size() * sizeof(aclblasComplex));
    EXPECT_EQ(std::memcmp(aAfter.data(), data.a.data(), data.a.size() * sizeof(aclblasComplex)), 0);
    EXPECT_EQ(std::memcmp(xAfter.data(), data.x.data(), data.x.size() * sizeof(aclblasComplex)), 0);
}

TEST_F(ChbmvEdgeTest, RuntimeErrorIsolation)
{
    int32_t device = -1;
    ASSERT_EQ(aclrtGetDevice(&device), ACL_SUCCESS);
    ASSERT_NE(aclrtSetDevice(-1), ACL_SUCCESS);
    const auto earlier = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    ASSERT_NE(earlier, ACL_SUCCESS);
    std::cout << "INJECTED_RUNTIME_ERROR=" << earlier << '\n';
    int32_t after = -1;
    ASSERT_EQ(aclrtGetDevice(&after), ACL_SUCCESS);
    ASSERT_EQ(after, device);
    ChbmvParam p(csv_map{});
    p.caseName = "runtime_error_isolation";
    p.n = 17;
    p.k = 5;
    p.lda = 9;
    p.incx = p.incy = 1;
    p.alpha = {0.7f, -0.2f};
    p.beta = {-0.3f, 0.5f};
    CheckCase(handle_, stream_, p, true);
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

TEST_F(ChbmvEdgeTest, RandomScalarsAndIgnoredStorage)
{
    uint32_t index = 0;
    for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (const int n : {7, 33, 65, 129, 2559, 2560, 2561}) {
            for (const int k : {0, 1, 4, 5, 8, 31, 32, 33, 64}) {
                ChbmvParam p(csv_map{});
                p.caseName = "supplement_" + std::to_string(index);
                p.n = n;
                p.k = k;
                p.lda = k + 4;
                p.uplo = uplo;
                p.incx = index % 3 == 0 ? -2 : 1;
                p.incy = index % 3 == 1 ? -3 : 2;
                p.randomSeed = 9503100U + index;
                const auto scalars =
                    chbmv_test_data::Fill(2, "RANDOM_NORM_5_5", p.randomSeed + 1000U, (index & 1U) == 0);
                p.alpha = scalars[0];
                p.beta = scalars[1];
                SCOPED_TRACE(p.caseName);
                CheckCase(handle_, stream_, p, true);
                ++index;
            }
        }
    }
}

TEST_F(ChbmvEdgeTest, SmallBoundarySmoke)
{
    ChbmvParam p(csv_map{});
    p.expectResult = ACLBLAS_STATUS_SUCCESS;

    p.nullAlpha = p.nullBeta = false;
    p.aFill = p.xFill = p.yFill = "RANDOM_NORM_5_5";
    p.alpha = {0.7f, -0.2f};
    p.beta = {-0.3f, 0.5f};
    unsigned count = 0;
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
        for (int n : {1, 3, 7, 17, 65})
            for (int k : {0, 1, 4, 9})
                for (int stride : {1, -2}) {
                    p.uplo = uplo;
                    p.n = n;
                    p.k = k;
                    p.lda = k + 3;
                    p.incx = stride;
                    p.incy = stride == 1 ? 1 : -3;
                    p.randomSeed = 220100 + count;
                    p.caseName = "small_" + std::to_string(count++);
                    SCOPED_TRACE(p.caseName);
                    CheckCase(handle_, stream_, p, true);
                }
    std::cout << "SUBCASES SmallBoundarySmoke=" << count << '\n';
}

TEST_F(ChbmvEdgeTest, WideVectorTails)
{
    ChbmvParam p(csv_map{});
    p.expectResult = ACLBLAS_STATUS_SUCCESS;
    p.nullAlpha = p.nullBeta = false;
    p.aFill = p.xFill = p.yFill = "RANDOM_NORM_5_5";
    p.alpha = {0.7f, -0.2f};
    p.beta = {-0.3f, 0.5f};
    unsigned count = 0;
    auto check = [&](aclblasFillMode_t uplo, int n, int k, int stride) {
        p.uplo = uplo;
        p.n = n;
        p.k = k;
        p.lda = k + 4;
        p.incx = stride;
        p.incy = stride == 1 ? -3 : 2;
        p.randomSeed = 220300 + count;
        p.caseName = "wide_tail_" + std::to_string(count++);
        SCOPED_TRACE(p.caseName);
        CheckCase(handle_, stream_, p, true);
    };
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
        for (int n : {257, 513, 1025, 2049})
            for (int k : {33, 127})
                for (int stride : {1, -2})
                    check(uplo, n, k, stride);
    // A single transpose repeat needs zero repeat strides. Cover each
    // possible 1..4 complex-element remainder on both matrix triangles.
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
        for (int n : {129, 513})
            for (int k : {65, 66, 67, 68})
                for (int stride : {1, -2})
                    check(uplo, n, k, stride);
    std::cout << "SUBCASES WideVectorTails=" << count << '\n';
}

TEST_F(ChbmvEdgeTest, ScalarSpecialsAndExtremeStrides)
{
    ChbmvParam p(csv_map{});
    p.expectResult = ACLBLAS_STATUS_SUCCESS;
    p.nullAlpha = p.nullBeta = false;
    p.aFill = p.xFill = p.yFill = "RANDOM_NORM_5_5";
    unsigned count = 0;
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
        for (auto alpha : {aclblasComplex{0, 0}, aclblasComplex{1, 0}, aclblasComplex{0.3f, -0.7f}})
            for (auto beta : {aclblasComplex{0, 0}, aclblasComplex{1, 0}, aclblasComplex{-0.4f, 0.2f}})
                for (const char* fill : {"VALUE_NORM_INF", "VALUE_NORM_NAN", "RANDOM_EXTREME"}) {
                    p.uplo = uplo;
                    p.n = 7;
                    p.k = 2;
                    p.lda = 6;
                    p.incx = -2;
                    p.incy = 3;
                    p.alpha = alpha;
                    p.beta = beta;
                    p.yFill = fill;
                    p.caseName = "special_" + std::to_string(count++);
                    SCOPED_TRACE(p.caseName);
                    CheckCase(handle_, stream_, p, true);
                }
    p.n = 1;
    p.k = 3;
    p.lda = 5;
    p.alpha = {0.7f, 0.3f};
    p.beta = {0.2f, -0.4f};
    p.yFill = "RANDOM_NORM_5_5";
    for (int stride : {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
        p.incx = p.incy = stride;
        p.expectResult = stride == INT32_MIN ? ACLBLAS_STATUS_INVALID_VALUE : ACLBLAS_STATUS_SUCCESS;

        p.caseName = "extreme_stride_" + std::to_string(count++);
        CheckCase(handle_, stream_, p, true);
    }
    p.expectResult = ACLBLAS_STATUS_SUCCESS;

    p.n = 3;
    p.k = 1;
    p.lda = 4;
    p.incx = -1000;
    p.incy = 1000;
    p.caseName = "large_stride";
    CheckCase(handle_, stream_, p, true);
    std::cout << "SUBCASES ScalarSpecialsAndExtremeStrides=" << count + 1 << '\n';
}

TEST_F(ChbmvEdgeTest, InvalidParameters)
{
    DeviceBuffer b(32);
    auto* ptr = static_cast<aclblasComplex*>(b.ptr());
    const aclblasComplex one{1, 0};
    const aclblasComplex zero{0, 0};
    const auto bad = ACLBLAS_STATUS_INVALID_VALUE;
    EXPECT_EQ(aclblasChbmv(handle_, ACLBLAS_UPPER, 1, INT32_MAX, &one, ptr, INT32_MAX, ptr, 1, &one, ptr, 1), bad);
    EXPECT_EQ(aclblasChbmv(handle_, ACLBLAS_UPPER, INT32_MAX, 0, &one, ptr, INT32_MAX, ptr, 1, &one, ptr, 1), bad);
    EXPECT_EQ(aclblasChbmv(handle_, ACLBLAS_UPPER, 1, 0, &zero, nullptr, 1, ptr, 1, &one, ptr, 1), bad);
    EXPECT_EQ(aclblasChbmv(handle_, ACLBLAS_UPPER, 1, 0, &zero, ptr, 1, nullptr, 1, &one, ptr, 1), bad);
    EXPECT_EQ(aclblasChbmv(handle_, ACLBLAS_UPPER, 1, 0, &one, ptr, 1, ptr, 0, &one, ptr, 1), bad);
    EXPECT_EQ(aclblasChbmv(handle_, ACLBLAS_UPPER, 1, 0, &one, ptr, 1, ptr, 1, &one, ptr, 0), bad);
    for (int stride : {INT32_MIN, INT32_MAX / 2 + 1}) {
        EXPECT_EQ(aclblasChbmv(handle_, ACLBLAS_UPPER, 3, 0, &zero, ptr, 1, ptr, stride, &one, ptr, 1), bad);
        EXPECT_EQ(aclblasChbmv(handle_, ACLBLAS_UPPER, 3, 0, &zero, ptr, 1, ptr, 1, &one, ptr, stride), bad);
    }
    // Legal boundary reaches the scalar quick return without accessing dummy storage.
    for (int stride : {INT32_MAX, -INT32_MAX})
        EXPECT_EQ(
            aclblasChbmv(handle_, ACLBLAS_UPPER, 2, 0, &zero, ptr, 1, ptr, stride, &one, ptr, stride),
            ACLBLAS_STATUS_SUCCESS);
}

static void CheckFourByteView(aclblasHandle_t handle, aclrtStream stream, const ChbmvParam& p)
{
    chbmv_test_data::Data data(p, true);
    auto bytes = [](const std::vector<aclblasComplex>& v) {
        std::vector<uint8_t> b(v.size() * 8 + 8, 0x5a);
        std::copy_n(reinterpret_cast<const uint8_t*>(v.data()), v.size() * 8, b.begin() + 4);
        return b;
    };
    auto ab = bytes(data.a);
    auto xb = bytes(data.x);
    auto yb = bytes(data.y);
    DeviceBuffer a(ab.size());
    DeviceBuffer x(xb.size());
    DeviceBuffer y(yb.size());
    a.copyFromHost(ab.data(), ab.size());
    x.copyFromHost(xb.data(), xb.size());
    y.copyFromHost(yb.data(), yb.size());
    ASSERT_EQ(
        aclblasChbmv(
            handle, p.uplo, p.n, p.k, &p.alpha, reinterpret_cast<aclblasComplex*>(a.bytePtr() + 4), p.lda,
            reinterpret_cast<aclblasComplex*>(x.bytePtr() + 4), p.incx, &p.beta,
            reinterpret_cast<aclblasComplex*>(y.bytePtr() + 4), p.incy),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    auto after = yb;
    y.copyToHost(after.data(), after.size());
    auto actual = data.y;
    const auto golden = ChbmvReference::Golden(p, data.a, data.x, data.y);
    std::copy_n(after.begin() + 4, actual.size() * 8, reinterpret_cast<uint8_t*>(actual.data()));
    EXPECT_TRUE(chbmv_test_data::ComponentMatches(p, actual, golden, false));
    EXPECT_TRUE(chbmv_test_data::ComponentMatches(p, actual, golden, true));
    EXPECT_TRUE(chbmv_test_data::GapsUnchanged(p, actual, data.y));
    EXPECT_EQ(std::memcmp(after.data(), yb.data(), 4), 0);
    EXPECT_EQ(std::memcmp(after.data() + after.size() - 4, yb.data() + yb.size() - 4, 4), 0);
    auto aa = ab;
    auto xx = xb;
    a.copyToHost(aa.data(), aa.size());
    x.copyToHost(xx.data(), xx.size());
    EXPECT_EQ(aa, ab);
    EXPECT_EQ(xx, xb);
}

TEST_F(ChbmvEdgeTest, FourByteViews)
{
    ChbmvParam p(csv_map{});
    p.n = 17;
    p.k = 5;
    p.lda = 9;
    p.incx = -2;
    p.incy = -3;
    p.nullAlpha = p.nullBeta = false;
    p.aFill = p.xFill = p.yFill = "RANDOM_NORM_5_5";
    p.alpha = {0.7f, 0.4f};
    p.beta = {-0.2f, 0.6f};
    for (auto shape : {std::array<int, 2>{17, 5}, std::array<int, 2>{513, 16}, std::array<int, 2>{513, 127}})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            p.n = shape[0];
            p.k = shape[1];
            p.lda = p.k + 4;
            p.uplo = uplo;
            p.caseName = "four_byte_view_" + std::to_string(p.n) + "_" + std::to_string(p.k) + "_" +
                         std::to_string(static_cast<int>(uplo));
            ASSERT_NO_FATAL_FAILURE(CheckFourByteView(handle_, stream_, p));
        }
}

TEST_F(ChbmvEdgeTest, FiniteOverflowCancellation)
{
    ChbmvParam p(csv_map{});
    p.alpha = {1, 0};
    p.beta = {0, 0};
    p.incx = -2;
    p.incy = -3;
    unsigned count = 0;
    for (auto shape : {std::array<int, 2>{32, 3}, std::array<int, 2>{513, 16}, std::array<int, 2>{513, 65}})
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER})
            for (bool extremeA : {false, true}) {
                p.n = shape[0];
                p.k = shape[1];
                p.lda = p.k + 4;
                p.uplo = uplo;
                p.aFill = extremeA ? "RANDOM_EXTREME" : "RANDOM_NORM_5_5";
                p.xFill = extremeA ? "RANDOM_NORM_5_5" : "RANDOM_EXTREME";
                p.caseName = "finite_overflow_" + std::to_string(count++);
                CheckCase(handle_, stream_, p, true);
            }
    std::cout << "SUBCASES FiniteOverflowCancellation=" << count << '\n';
}

} // namespace
