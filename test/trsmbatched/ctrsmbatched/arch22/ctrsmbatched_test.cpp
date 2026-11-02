/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <utility>
#include "blas_test.h"
#include "cblas_compat.h"
#include "ctrsmbatched_param.h"
#include "fill.h"

namespace {
struct Geometry {
    int order;
    int rhs;
    int batches;
    int padding;
    uint32_t seed;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_T;
    aclblasDiagType_t diag = ACLBLAS_UNIT;
    bool exceptionalPair = false;
};

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t size) : size_(size)
    {
        status_ = aclrtMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ~DeviceArray()
    {
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    aclError Status() const { return status_; }
    T* Data() const { return data_; }
    aclError Upload(const std::vector<T>& input)
    {
        return aclrtMemcpy(data_, size_ * sizeof(T), input.data(), input.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    aclError Download(std::vector<T>& output) const
    {
        return aclrtMemcpy(
            output.data(), output.size() * sizeof(T), data_, size_ * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    }

private:
    size_t size_;
    T* data_ = nullptr;
    aclError status_;
};

void PrepareHost(const Geometry& g, std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    const int ld = g.order + g.padding;
    const BlasFillMode fill("RANDOM_NORM_P0.1_P0.1");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int batch = 0; batch < g.batches; ++batch) {
        auto matrix = makeBlasComplexMatrix(g.order, g.order, ld, fill, g.seed + batch * 1000U);
        auto panel = makeBlasComplexMatrix(g.order, g.rhs, ld, fill, g.seed + batch * 1000U + 500U);
        for (int col = 0; col < g.order; ++col) {
            for (int row = 0; row < ld; ++row) {
                if (row >= g.order || (g.uplo == ACLBLAS_UPPER ? row > col : row < col) ||
                    (row == col && g.diag == ACLBLAS_UNIT)) {
                    matrix[col * ld + row] = {nan, nan};
                } else if (row == col) {
                    matrix[col * ld + row] = {5.0f + row, 0.1f};
                }
            }
        }
        for (int col = 0; col < g.rhs; ++col) {
            for (int row = g.order; row < ld; ++row) {
                panel[col * ld + row] = {-999.0f, 999.0f};
            }
        }
        if (g.exceptionalPair && batch == (g.batches > 8 ? 8 : 1)) {
            std::fill(panel.begin(), panel.end(), aclblasComplex{0.f, 0.f});
            matrix[g.uplo == ACLBLAS_UPPER ? ld : 1] = {1.e20f, 0.f};
        }
        a.insert(a.end(), matrix.begin(), matrix.end());
        b.insert(b.end(), panel.begin(), panel.end());
    }
}

void ComputeGolden(const Geometry& g, const std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    const int ld = g.order + g.padding;
    const aclblasComplex alpha{1.0f, 0.0f};
    for (int batch = 0; batch < g.batches; ++batch) {
        cblas_ctrsm(
            CblasColMajor, CblasLeft, g.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower,
            g.trans == ACLBLAS_OP_N ? CblasNoTrans : (g.trans == ACLBLAS_OP_C ? CblasConjTrans : CblasTrans),
            g.diag == ACLBLAS_UNIT ? CblasUnit : CblasNonUnit, g.order, g.rhs, &alpha,
            a.data() + size_t(batch) * ld * g.order, ld, b.data() + size_t(batch) * ld * g.rhs, ld);
    }
}

void CompareResult(
    const Geometry& g, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden)
{
    const int ld = g.order + g.padding;
    size_t mismatched = 0;
    double maxAbs = 0.0;
    for (int batch = 0; batch < g.batches; ++batch) {
        for (int col = 0; col < g.rhs; ++col) {
            for (int row = 0; row < ld; ++row) {
                const size_t index = (size_t(batch) * g.rhs + col) * ld + row;
                if (row >= g.order) {
                    EXPECT_EQ(std::memcmp(&actual[index], &golden[index], sizeof(aclblasComplex)), 0);
                    continue;
                }
                for (int part = 0; part < 2; ++part) {
                    const float x = part == 0 ? actual[index].real : actual[index].imag;
                    const float y = part == 0 ? golden[index].real : golden[index].imag;
                    ASSERT_TRUE(std::isfinite(x));
                    ASSERT_TRUE(std::isfinite(y));
                    const double error = std::abs(double(x) - y);
                    maxAbs = std::max(maxAbs, error);
                    mismatched += error > std::ldexp(1.0, -13) * (1.0 + std::abs(double(y)));
                }
            }
        }
    }
    const size_t count = size_t(g.batches) * g.rhs * g.order * 2;
    EXPECT_LE(mismatched, count / 100);
    EXPECT_LE(maxAbs, 0.01);
}

aclblasStatus_t Invoke(
    aclblasHandle_t handle, const Geometry& g, const aclblasComplex* const* a, aclblasComplex* const* b)
{
    const int ld = g.order + g.padding;
    const aclblasComplex alpha{1.f, 0.f};
    return aclblasCtrsmBatched(
        handle, ACLBLAS_SIDE_LEFT, g.uplo, g.trans, g.diag, g.order, g.rhs, &alpha, a, ld, b, ld, g.batches);
}

void RunGeometry(aclblasHandle_t handle, aclrtStream stream, const Geometry& g)
{
    std::vector<aclblasComplex> a, b;
    PrepareHost(g, a, b);
    auto golden = b;
    ComputeGolden(g, a, golden);
    DeviceArray<aclblasComplex> da(a.size()), db(b.size());
    DeviceArray<const aclblasComplex*> ap(g.batches);
    DeviceArray<aclblasComplex*> bp(g.batches);
    ASSERT_EQ(da.Status(), ACL_SUCCESS);
    ASSERT_EQ(db.Status(), ACL_SUCCESS);
    ASSERT_EQ(ap.Status(), ACL_SUCCESS);
    ASSERT_EQ(bp.Status(), ACL_SUCCESS);
    const int ld = g.order + g.padding;
    std::vector<const aclblasComplex*> av(g.batches);
    std::vector<aclblasComplex*> bv(g.batches);
    for (int batch = 0; batch < g.batches; ++batch) {
        av[batch] = da.Data() + size_t(batch) * ld * g.order;
        bv[batch] = db.Data() + size_t(batch) * ld * g.rhs;
    }
    ASSERT_EQ(da.Upload(a), ACL_SUCCESS);
    ASSERT_EQ(db.Upload(b), ACL_SUCCESS);
    ASSERT_EQ(ap.Upload(av), ACL_SUCCESS);
    ASSERT_EQ(bp.Upload(bv), ACL_SUCCESS);
    ASSERT_EQ(Invoke(handle, g, ap.Data(), bp.Data()), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    std::vector<aclblasComplex> actual(b.size()), copiedA(a.size());
    ASSERT_EQ(db.Download(actual), ACL_SUCCESS);
    ASSERT_EQ(da.Download(copiedA), ACL_SUCCESS);
    EXPECT_EQ(std::memcmp(a.data(), copiedA.data(), a.size() * sizeof(aclblasComplex)), 0);
    CompareResult(g, actual, golden);
}

class CtrsmBatchedSmallInverseTest : public BlasTest<Geometry> {};
TEST_P(CtrsmBatchedSmallInverseTest, HardwareSolve) { RunGeometry(handle_, stream_, GetParam()); }

std::vector<Geometry> BoundaryGeometries()
{
    std::vector<Geometry> geometries;
    for (int order = 1; order <= 24; ++order) {
        geometries.push_back({order, 193, 41, order % 2 == 0 ? 3 : 0, 20261018U + uint32_t(order)});
    }
    for (int batch : {1, 2, 3, 19, 20, 21, 39, 40, 209}) {
        geometries.push_back({23, 208, batch, 5, 20261018U + uint32_t(batch)});
    }
    for (const auto& geometry : std::vector<Geometry>{
             {4, 4, 2, 0, 20260007U},
             {8, 8, 2, 0, 20260008U},
             {5, 5, 4, 0, 20260052U},
             {16, 16, 4, 4, 20260142U},
             {24, 24, 2, 0, 20260165U}}) {
        geometries.push_back(geometry);
    }
    for (const auto triangle : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (const auto mode : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            for (int order = 1; order <= 24; ++order) {
                geometries.push_back(
                    {order, order % 2 ? 193 : 17, order % 2 ? 41 : 3, order % 3, 20261019U + uint32_t(order), triangle,
                     mode});
            }
            for (int rhs : {1, 7, 8, 191, 192, 194, 208, 209, 415, 416, 417}) {
                geometries.push_back({23, rhs, 21, 5, 20261019U + uint32_t(rhs), triangle, mode});
            }
        }
    }
    geometries.push_back({1, 1, 1, 0, 20261018U});
    geometries.push_back({24, 1, 1, 3, 20261018U});
    return geometries;
}
INSTANTIATE_TEST_SUITE_P(Boundaries, CtrsmBatchedSmallInverseTest, ::testing::ValuesIn(BoundaryGeometries()));

TEST_F(CtrsmBatchedSmallInverseTest, SharedPlanarSquareMatchesNetlib)
{
    for (const auto triangle : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (const int order : {8, 16})
            for (const int batches : {3, 65, 1025})
                RunGeometry(
                    handle_, stream_, {order, order, batches, 0, 20261021U, triangle, ACLBLAS_OP_N, ACLBLAS_NON_UNIT});
    }
}

TEST_F(CtrsmBatchedSmallInverseTest, SharedIndexedAndFusedPathsMatchNetlib)
{
    for (const auto mode : {ACLBLAS_OP_N, ACLBLAS_OP_C})
        RunGeometry(handle_, stream_, {48, 321, 3, 1, 20261022U, ACLBLAS_LOWER, mode, ACLBLAS_UNIT});
    RunGeometry(handle_, stream_, {64, 64, 64, 0, 20261022U, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT});
}

TEST_F(CtrsmBatchedSmallInverseTest, MediumSquareUsesSharedMixedPath)
{
    for (const auto triangle : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (const int batches : {3, 65, 127}) {
            RunGeometry(handle_, stream_, {32, 32, batches, 0, 20261021U, triangle, ACLBLAS_OP_N, ACLBLAS_NON_UNIT});
            if (batches <= 65)
                RunGeometry(
                    handle_, stream_, {32, 32, batches, 0, 20261021U, triangle, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, true});
        }
    }
}

TEST_F(CtrsmBatchedSmallInverseTest, SharedPrefetchKeepsWholeRhs)
{
    RunGeometry(handle_, stream_, {151, 178, 109, 1, 20261022U, ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_UNIT});
}

TEST_F(CtrsmBatchedSmallInverseTest, SharedBlockedInverseHandlesPartialPanels)
{
    for (const auto triangle : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        RunGeometry(handle_, stream_, {65, 67, 97, 3, 20261021U, triangle, ACLBLAS_OP_C, ACLBLAS_NON_UNIT});
        RunGeometry(handle_, stream_, {96, 129, 3, 1, 20261022U, triangle, ACLBLAS_OP_N, ACLBLAS_UNIT});
    }
}

TEST_F(CtrsmBatchedSmallInverseTest, ReusesWorkspaceAcrossGeometries)
{
    for (const auto& geometry : std::vector<Geometry>{
             {1, 1, 1, 0, 1}, {24, 208, 209, 5, 2}, {7, 17, 3, 2, 3}, {23, 193, 41, 0, 4}, {1, 1, 1, 0, 5}}) {
        RunGeometry(handle_, stream_, geometry);
    }
}
TEST_F(CtrsmBatchedSmallInverseTest, NullHandle)
{
    EXPECT_EQ(Invoke(nullptr, {23, 193, 1, 0, 1}, nullptr, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}
TEST_F(CtrsmBatchedSmallInverseTest, InvalidActivePointers)
{
    EXPECT_EQ(Invoke(handle_, {23, 193, 1, 0, 1}, nullptr, nullptr), ACLBLAS_STATUS_INVALID_VALUE);
}
TEST_F(CtrsmBatchedSmallInverseTest, ZeroDimensionDoesNotAccessPointers)
{
    const aclblasComplex alpha{1.f, 0.f};
    for (const auto dims : {std::pair<int, int>{0, 4}, {4, 0}, {0, 0}}) {
        for (const auto side : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT})
            EXPECT_EQ(
                aclblasCtrsmBatched(
                    handle_, side, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, dims.first, dims.second, &alpha,
                    nullptr, 0, nullptr, 0, 1),
                ACLBLAS_STATUS_SUCCESS);
    }
}
TEST_F(CtrsmBatchedSmallInverseTest, ZeroDimensionRejectsNegativeLeadingDimensions)
{
    const aclblasComplex alpha{1.f, 0.f};
    for (const auto dims : {std::pair<int, int>{0, 4}, {4, 0}, {0, 0}}) {
        for (const auto ld : {std::pair<int, int>{-1, 0}, {0, -1}})
            EXPECT_EQ(
                aclblasCtrsmBatched(
                    handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, dims.first, dims.second,
                    &alpha, nullptr, ld.first, nullptr, ld.second, 1),
                ACLBLAS_STATUS_INVALID_VALUE);
    }
}
TEST_F(CtrsmBatchedSmallInverseTest, DeviceAlphaReadsCurrentValue)
{
    DeviceArray<aclblasComplex> alpha(1), a(1), b(2);
    DeviceArray<const aclblasComplex*> ap(1);
    DeviceArray<aclblasComplex*> bp(1);
    ASSERT_EQ(alpha.Status(), ACL_SUCCESS);
    ASSERT_EQ(a.Status(), ACL_SUCCESS);
    ASSERT_EQ(b.Status(), ACL_SUCCESS);
    ASSERT_EQ(ap.Status(), ACL_SUCCESS);
    ASSERT_EQ(bp.Status(), ACL_SUCCESS);
    ASSERT_EQ(a.Upload({{2.f, 0.f}}), ACL_SUCCESS);
    ASSERT_EQ(ap.Upload({a.Data()}), ACL_SUCCESS);
    ASSERT_EQ(bp.Upload({b.Data()}), ACL_SUCCESS);
    for (const auto value : {aclblasComplex{1.f, 0.f}, {0.f, 0.f}, {2.f, -1.f}}) {
        ASSERT_EQ(alpha.Upload({value}), ACL_SUCCESS);
        ASSERT_EQ(b.Upload({{4.f, 2.f}, {123.f, -321.f}}), ACL_SUCCESS);
        EXPECT_EQ(
            aclblasCtrsmBatched(
                handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, 1, alpha.Data(),
                value.real == 0 ? nullptr : ap.Data(), 1, bp.Data(), 2, 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        std::vector<aclblasComplex> result(2);
        ASSERT_EQ(b.Download(result), ACL_SUCCESS);
        EXPECT_FLOAT_EQ(result[0].real, 2 * value.real - value.imag);
        EXPECT_FLOAT_EQ(result[0].imag, value.real + 2 * value.imag);
        EXPECT_FLOAT_EQ(result[1].real, 123.f);
        EXPECT_FLOAT_EQ(result[1].imag, -321.f);
    }
}
TEST_F(CtrsmBatchedSmallInverseTest, DeviceAlphaFollowsHandleStream)
{
    const std::vector<aclblasComplex> values{{1.f, 0.f}, {0.f, 0.f}, {2.f, -1.f}, {1.f, 0.f}};
    DeviceArray<aclblasComplex> alpha(1), source(values.size()), a(1), b(2);
    DeviceArray<const aclblasComplex*> ap(1);
    DeviceArray<aclblasComplex*> bp(1);
    ASSERT_EQ(alpha.Status(), ACL_SUCCESS);
    ASSERT_EQ(source.Status(), ACL_SUCCESS);
    ASSERT_EQ(a.Status(), ACL_SUCCESS);
    ASSERT_EQ(b.Status(), ACL_SUCCESS);
    ASSERT_EQ(ap.Status(), ACL_SUCCESS);
    ASSERT_EQ(bp.Status(), ACL_SUCCESS);
    ASSERT_EQ(source.Upload(values), ACL_SUCCESS);
    ASSERT_EQ(alpha.Upload({{99.f, -99.f}}), ACL_SUCCESS);
    ASSERT_EQ(a.Upload({{2.f, 0.f}}), ACL_SUCCESS);
    ASSERT_EQ(ap.Upload({a.Data()}), ACL_SUCCESS);
    ASSERT_EQ(bp.Upload({b.Data()}), ACL_SUCCESS);
    for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_EQ(b.Upload({{4.f, 2.f}, {123.f, -321.f}}), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpyAsync(
                alpha.Data(), sizeof(aclblasComplex), source.Data() + i, sizeof(aclblasComplex),
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream_),
            ACL_SUCCESS);
        ASSERT_EQ(
            aclblasCtrsmBatched(
                handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, 1, alpha.Data(),
                values[i].real == 0 ? nullptr : ap.Data(), 1, bp.Data(), 2, 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        std::vector<aclblasComplex> result(2);
        ASSERT_EQ(b.Download(result), ACL_SUCCESS);
        EXPECT_FLOAT_EQ(result[0].real, 2 * values[i].real - values[i].imag);
        EXPECT_FLOAT_EQ(result[0].imag, values[i].real + 2 * values[i].imag);
        EXPECT_FLOAT_EQ(result[1].real, 123.f);
        EXPECT_FLOAT_EQ(result[1].imag, -321.f);
    }
}
TEST_F(CtrsmBatchedSmallInverseTest, InvalidLeadingDimension)
{
    const aclblasComplex alpha{1.f, 0.f};
    EXPECT_EQ(
        aclblasCtrsmBatched(
            handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 23, 193, &alpha, nullptr, 22,
            nullptr, 23, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
}

struct FallbackGeometry {
    int order;
    bool right;
    bool upper;
    aclblasOperation_t trans;
    bool unit;
    bool constrainedWorkspace = false;
    int rhs = 3;
    bool fastPath = false;
};
void PrepareFallback(
    const FallbackGeometry& g, int ld, int m, int n, std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    a.assign(size_t(ld) * g.order, {nan, nan});
    b.assign(size_t(ld) * n, {-999.f, 999.f});
    for (int col = 0; col < g.order; col++) {
        for (int row = 0; row < g.order; row++) {
            if (g.upper ? row < col : row > col)
                a[size_t(col) * ld + row] = {0.0005f * ((row + col) % 7 - 3), 0.0003f * ((row + col) % 5 - 2)};
            else if (row == col && !g.unit)
                a[size_t(col) * ld + row] = {1.25f, 0.1f};
        }
    }
    for (int col = 0; col < n; col++)
        for (int row = 0; row < m; row++)
            b[size_t(col) * ld + row] = {0.05f * (row % 13 - 6), 0.04f * (col % 11 - 5)};
}
void CompareFallback(
    int m, int n, int ld, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden)
{
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < ld; row++) {
            const size_t at = size_t(col) * ld + row;
            if (row >= m) {
                ASSERT_EQ(std::memcmp(&actual[at], &golden[at], sizeof(aclblasComplex)), 0);
            } else {
                ASSERT_TRUE(std::isfinite(actual[at].real));
                ASSERT_TRUE(std::isfinite(actual[at].imag));
                ASSERT_NEAR(actual[at].real, golden[at].real, 0.0001220703125f * (1.f + std::abs(golden[at].real)));
                ASSERT_NEAR(actual[at].imag, golden[at].imag, 0.0001220703125f * (1.f + std::abs(golden[at].imag)));
            }
        }
    }
}
class CtrsmBatchedFallbackTest : public BlasTest<FallbackGeometry> {};
TEST_P(CtrsmBatchedFallbackTest, MatchesNetlibWithPadding)
{
    const auto g = GetParam();
    const int m = g.right ? g.rhs : g.order, n = g.right ? g.order : g.rhs, ld = std::max(g.order, m) + 3;
    std::vector<aclblasComplex> a, b;
    PrepareFallback(g, ld, m, n, a, b);
    auto golden = b;
    const aclblasComplex alpha = g.order > 4096 || g.constrainedWorkspace || g.fastPath ? aclblasComplex{1.f, 0.f} :
                                                                                          aclblasComplex{0.5f, -0.25f};
    cblas_ctrsm(
        CblasColMajor, g.right ? CblasRight : CblasLeft, g.upper ? CblasUpper : CblasLower,
        g.trans == ACLBLAS_OP_N ? CblasNoTrans : (g.trans == ACLBLAS_OP_C ? CblasConjTrans : CblasTrans),
        g.unit ? CblasUnit : CblasNonUnit, m, n, &alpha, a.data(), ld, golden.data(), ld);
    DeviceArray<aclblasComplex> da(a.size()), db(b.size());
    DeviceArray<aclblasComplex> limitedWorkspace(1);
    if (g.constrainedWorkspace) {
        ASSERT_EQ(limitedWorkspace.Status(), ACL_SUCCESS);
        ASSERT_EQ(
            aclblasSetWorkspace(handle_, limitedWorkspace.Data(), sizeof(aclblasComplex)), ACLBLAS_STATUS_SUCCESS);
    }
    DeviceArray<const aclblasComplex*> ap(1);
    DeviceArray<aclblasComplex*> bp(1);
    ASSERT_EQ(da.Status(), ACL_SUCCESS);
    ASSERT_EQ(db.Status(), ACL_SUCCESS);
    ASSERT_EQ(ap.Status(), ACL_SUCCESS);
    ASSERT_EQ(bp.Status(), ACL_SUCCESS);
    ASSERT_EQ(da.Upload(a), ACL_SUCCESS);
    ASSERT_EQ(db.Upload(b), ACL_SUCCESS);
    ASSERT_EQ(ap.Upload({da.Data()}), ACL_SUCCESS);
    ASSERT_EQ(bp.Upload({db.Data()}), ACL_SUCCESS);
    ASSERT_EQ(
        aclblasCtrsmBatched(
            handle_, g.right ? ACLBLAS_SIDE_RIGHT : ACLBLAS_SIDE_LEFT, g.upper ? ACLBLAS_UPPER : ACLBLAS_LOWER, g.trans,
            g.unit ? ACLBLAS_UNIT : ACLBLAS_NON_UNIT, m, n, &alpha, ap.Data(), ld, bp.Data(), ld, 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    std::vector<aclblasComplex> actual(b.size());
    ASSERT_EQ(db.Download(actual), ACL_SUCCESS);
    CompareFallback(m, n, ld, actual, golden);
}
INSTANTIATE_TEST_SUITE_P(
    TileBoundaries, CtrsmBatchedFallbackTest,
    ::testing::Values(
        FallbackGeometry{1, false, false, ACLBLAS_OP_N, false}, FallbackGeometry{127, false, true, ACLBLAS_OP_T, true},
        FallbackGeometry{128, false, false, ACLBLAS_OP_C, false},
        FallbackGeometry{129, true, true, ACLBLAS_OP_N, false}, FallbackGeometry{129, true, false, ACLBLAS_OP_T, true},
        FallbackGeometry{257, true, true, ACLBLAS_OP_C, false},
        FallbackGeometry{4097, false, false, ACLBLAS_OP_N, true},
        FallbackGeometry{12, true, false, ACLBLAS_OP_N, true, true},
        FallbackGeometry{8, false, false, ACLBLAS_OP_N, true, true}));
INSTANTIATE_TEST_SUITE_P(
    RightBlockBoundaries, CtrsmBatchedFallbackTest,
    ::testing::Values(
        FallbackGeometry{65, true, true, ACLBLAS_OP_C, false, false, 67, true},
        FallbackGeometry{65, true, false, ACLBLAS_OP_N, true, false, 131, true},
        FallbackGeometry{128, true, false, ACLBLAS_OP_T, false, false, 64, true},
        FallbackGeometry{129, true, true, ACLBLAS_OP_N, true, false, 127, true}));
} // namespace

namespace {
template <typename T>
class CtrsmDeviceBuffer {
public:
    explicit CtrsmDeviceBuffer(size_t count) : count_(count)
    {
        if (count)
            status_ = aclrtMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ~CtrsmDeviceBuffer()
    {
        if (data_)
            aclrtFree(data_);
    }
    CtrsmDeviceBuffer(const CtrsmDeviceBuffer&) = delete;
    CtrsmDeviceBuffer& operator=(const CtrsmDeviceBuffer&) = delete;
    T* Data() const { return data_; }
    aclError Status() const { return status_; }
    aclError Upload(const std::vector<T>& v)
    {
        return aclrtMemcpy(data_, count_ * sizeof(T), v.data(), v.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    aclError Download(std::vector<T>& v)
    {
        return aclrtMemcpy(v.data(), v.size() * sizeof(T), data_, count_ * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    }

private:
    size_t count_;
    T* data_ = nullptr;
    aclError status_ = ACL_SUCCESS;
};
void FillCtrsmInputs(const CtrsmBatchedParam& p, std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    const int order = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    for (int batch = 0; batch < p.batchCount; batch++) {
        auto x = makeBlasComplexMatrix(order, order, p.lda, p.aFill, p.randomSeed + batch * 1000U);
        auto y = makeBlasComplexMatrix(p.m, p.n, p.ldb, p.bFill, p.randomSeed + batch * 1000U + 500U);
        for (int col = 0; col < order; col++)
            for (int row = 0; row < order; row++) {
                if (x.empty())
                    continue;
                auto& z = x[col * p.lda + row];
                if ((p.uplo == ACLBLAS_UPPER && row > col) || (p.uplo == ACLBLAS_LOWER && row < col))
                    z = {0, 0};
                if (row == col && p.diag == ACLBLAS_NON_UNIT)
                    z.real += (z.real >= 0 ? 1 : -1) * float(std::max(5, order));
            }
        a.insert(a.end(), x.begin(), x.end());
        b.insert(b.end(), y.begin(), y.end());
    }
}
void ComputeCtrsmGolden(
    const CtrsmBatchedParam& p, const std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    const int order = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    for (int batch = 0; batch < p.batchCount; batch++) {
        cblas_ctrsm(
            CblasColMajor, p.side == ACLBLAS_SIDE_LEFT ? CblasLeft : CblasRight,
            p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower,
            p.trans == ACLBLAS_OP_N ? CblasNoTrans : (p.trans == ACLBLAS_OP_T ? CblasTrans : CblasConjTrans),
            p.diag == ACLBLAS_UNIT ? CblasUnit : CblasNonUnit, p.m, p.n, &p.alpha,
            a.empty() ? nullptr : a.data() + size_t(batch) * p.lda * order, p.lda,
            b.data() + size_t(batch) * p.ldb * p.n, p.ldb);
    }
}
void CompareCtrsmComponent(float actual, float ref, size_t& mismatched, size_t& limitBad)
{
    if (std::isnan(ref)) {
        EXPECT_TRUE(std::isnan(actual));
        return;
    }
    if (std::isinf(ref)) {
        EXPECT_EQ(actual, ref);
        return;
    }
    double error = std::abs(double(actual) - ref);
    int exponent = 0;
    std::frexp(std::abs(double(ref)), &exponent);
    double ulp = std::ldexp(1., std::max(exponent - 1, -126) - 23);
    mismatched += !std::isfinite(actual) || error > std::ldexp(1., -13) * (1 + std::abs(double(ref)));
    limitBad += !std::isfinite(actual) || error > std::max(.01, 32 * ulp);
}
void CompareCtrsmOutput(
    const CtrsmBatchedParam& p, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y)
{
    size_t mismatched = 0, limitBad = 0, count = size_t(p.batchCount) * p.m * p.n * 2;
    for (int batch = 0; batch < p.batchCount; batch++)
        for (int col = 0; col < p.n; col++)
            for (int row = 0; row < p.ldb; row++) {
                size_t at = (size_t(batch) * p.n + col) * p.ldb + row;
                if (row >= p.m) {
                    EXPECT_EQ(std::memcmp(&x[at], &y[at], sizeof(aclblasComplex)), 0);
                    continue;
                }
                for (int q = 0; q < 2; q++) {
                    float actual = q ? x[at].imag : x[at].real, ref = q ? y[at].imag : y[at].real;
                    CompareCtrsmComponent(actual, ref, mismatched, limitBad);
                }
            }
    EXPECT_LE(mismatched, count / 100) << p.caseName;
    EXPECT_EQ(limitBad, 0U) << p.caseName;
}
aclblasStatus_t InvokeCtrsm(
    const CtrsmBatchedParam& p, aclblasHandle_t h, const aclblasComplex* const* a, aclblasComplex* const* b)
{
    return aclblasCtrsmBatched(
        p.description.find("handle_null") != std::string::npos ? nullptr : h, p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        p.nullAlpha ? nullptr : &p.alpha, a, p.lda, b, p.ldb, p.batchCount);
}
class CtrsmBatchedCsvTest : public BlasTest<CtrsmBatchedParam> {
protected:
    void CheckStatus(const CtrsmBatchedParam& p)
    {
        const aclblasComplex* ap = nullptr;
        aclblasComplex* bp = nullptr;
        EXPECT_EQ(
            InvokeCtrsm(
                p, handle_, BlasFillMode(p.aFill).method == BlasFillMode::M_NULLPTR ? nullptr : &ap,
                BlasFillMode(p.bFill).method == BlasFillMode::M_NULLPTR ? nullptr : &bp),
            p.expectResult);
    }
};
TEST_P(CtrsmBatchedCsvTest, CsvDriven)
{
    const auto& p = GetParam();
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m == 0 || p.n == 0) {
        CheckStatus(p);
        return;
    }
    std::vector<aclblasComplex> a, b;
    FillCtrsmInputs(p, a, b);
    CtrsmDeviceBuffer<aclblasComplex> da(a.size()), db(b.size());
    CtrsmDeviceBuffer<const aclblasComplex*> ap(p.batchCount);
    CtrsmDeviceBuffer<aclblasComplex*> bp(p.batchCount);
    ASSERT_EQ(da.Status(), ACL_SUCCESS);
    ASSERT_EQ(db.Status(), ACL_SUCCESS);
    ASSERT_EQ(ap.Status(), ACL_SUCCESS);
    ASSERT_EQ(bp.Status(), ACL_SUCCESS);
    const int order = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    std::vector<const aclblasComplex*> av(p.batchCount);
    std::vector<aclblasComplex*> bv(p.batchCount);
    for (int batch = 0; batch < p.batchCount; batch++) {
        av[batch] = a.empty() ? nullptr : da.Data() + size_t(batch) * p.lda * order;
        bv[batch] = db.Data() + size_t(batch) * p.ldb * p.n;
    }
    if (!a.empty())
        ASSERT_EQ(da.Upload(a), ACL_SUCCESS);
    ASSERT_EQ(db.Upload(b), ACL_SUCCESS);
    ASSERT_EQ(ap.Upload(av), ACL_SUCCESS);
    ASSERT_EQ(bp.Upload(bv), ACL_SUCCESS);
    const bool timing = p.caseName.find("TC_PF_") == 0;
    if (timing) {
        for (int sample = 0; sample < 15; sample++) {
            ASSERT_EQ(db.Upload(b), ACL_SUCCESS);
            ASSERT_EQ(InvokeCtrsm(p, handle_, ap.Data(), bp.Data()), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        }
        return;
    }
    auto golden = b;
    ComputeCtrsmGolden(p, a, golden);
    ASSERT_EQ(InvokeCtrsm(p, handle_, a.empty() ? nullptr : ap.Data(), bp.Data()), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    std::vector<aclblasComplex> actual(b.size());
    ASSERT_EQ(db.Download(actual), ACL_SUCCESS);
    CompareCtrsmOutput(p, actual, golden);
}
void AppendOrderedRegressions(std::vector<CtrsmBatchedParam>& cases)
{
    for (int order : {1024, 1025, 2047, 2048}) {
        for (bool right : {false, true}) {
            csv_map row{
                {"case_name", std::string("Ordered_") + std::to_string(order) + (right ? "_Right" : "_Left")},
                {"side", right ? "RIGHT" : "LEFT"},
                {"uplo", order % 2 ? "LOWER" : "UPPER"},
                {"trans", right ? "T" : "N"},
                {"diag", "UNIT"},
                {"m", std::to_string(right ? 3 : order)},
                {"n", std::to_string(right ? order : 3)},
                {"lda", std::to_string(order + 3)},
                {"ldb", std::to_string((right ? 3 : order) + 5)},
                {"alpha_real", "100"},
                {"alpha_imag", "0"},
                {"batchCount", "1"},
                {"a", "RANDOM_NORM_P0.1_P0.1"},
                {"b", "RANDOM_NORM_5_5"},
                {"random_seed", std::to_string(20261019U + order)}};
            cases.emplace_back(row);
        }
    }
}
void AppendSpecialValueRegressions(std::vector<CtrsmBatchedParam>& cases)
{
    for (bool upper : {false, true}) {
        for (const char* fill : {"RANDOM_EXTREME", "VALUE_NORM_INF"}) {
            csv_map row{
                {"case_name", std::string(upper ? "Upper_" : "Lower_") + fill},
                {"side", "LEFT"},
                {"uplo", upper ? "UPPER" : "LOWER"},
                {"trans", "N"},
                {"diag", "NON_UNIT"},
                {"m", "32"},
                {"n", "32"},
                {"lda", "32"},
                {"ldb", "32"},
                {"alpha_real", "1"},
                {"alpha_imag", "0"},
                {"batchCount", "3"},
                {"a", "RANDOM_NORM_5_5"},
                {"b", fill},
                {"random_seed", "20261019"}};
            cases.emplace_back(row);
        }
    }
}
void AppendRightPaddingRegressions(std::vector<CtrsmBatchedParam>& cases)
{
    for (int order : {4, 5, 7, 9, 15, 17, 31, 33, 47, 49, 63, 65, 95}) {
        const int pad = order <= 32 ? 32 : (order <= 48 ? 48 : (order <= 64 ? 64 : 96));
        for (bool wide : {false, true}) {
            for (const char* mode : {"N", "T", "C"}) {
                const int rows = wide ? pad * 32 + 3 : 3;
                csv_map row{
                    {"case_name",
                     std::string("RightPadding_") + std::to_string(order) + (wide ? "_Wide_" : "_Narrow_") + mode},
                    {"side", "RIGHT"},
                    {"uplo", order % 2 ? "LOWER" : "UPPER"},
                    {"trans", mode},
                    {"diag", "NON_UNIT"},
                    {"m", std::to_string(rows)},
                    {"n", std::to_string(order)},
                    {"lda", std::to_string(order + 3)},
                    {"ldb", std::to_string(rows + 5)},
                    {"alpha_real", "1"},
                    {"alpha_imag", "0"},
                    {"batchCount", "2"},
                    {"a", "RANDOM_NORM_5_5"},
                    {"b", "RANDOM_NORM_5_5"},
                    {"random_seed", std::to_string(20261019U + order)}};
                cases.emplace_back(row);
            }
        }
    }
}
std::vector<CtrsmBatchedParam> PrecisionRegressionCases()
{
    std::vector<CtrsmBatchedParam> cases;
    AppendOrderedRegressions(cases);
    AppendSpecialValueRegressions(cases);
    AppendRightPaddingRegressions(cases);
    return cases;
}
INSTANTIATE_TEST_SUITE_P(
    PrecisionRegression, CtrsmBatchedCsvTest, ::testing::ValuesIn(PrecisionRegressionCases()),
    PrintCaseInfoString<CtrsmBatchedParam>);
const std::vector<CtrsmBatchedParam>& OfficialCases()
{
    static const auto cases = GetCasesFromCsv<CtrsmBatchedParam>(ReplaceFileExtension2Csv(__FILE__));
    return cases;
}
TEST(CtrsmBatchedCsvCoverage, CsvCaseCount)
{
    const auto& cases = OfficialCases();
    const auto performance =
        std::count_if(cases.begin(), cases.end(), [](const auto& p) { return p.caseName.find("TC_PF_") == 0; });
    EXPECT_EQ(cases.size(), 1200U);
    EXPECT_EQ(performance, 200);
    EXPECT_EQ(cases.size() - performance, 1000U);
}
INSTANTIATE_TEST_SUITE_P(
    Official, CtrsmBatchedCsvTest, ::testing::ValuesIn(OfficialCases()), PrintCaseInfoString<CtrsmBatchedParam>);
} // namespace
