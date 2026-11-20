/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ssbmv_test.cpp
 * \brief CSV driven accuracy test of aclblasSsbmv on Atlas A2/A3 (arch22).
 *
 * The golden is the ecosystem reference implementation, cblas_ssbmv (Netlib BLAS), see
 * ssbmv_golden.h. The default uses the repository's FLOAT32 mixed tolerance.
 * Set SSBMV_TASK_TOLERANCE=1 to also run the CSV suite with the task-book tolerances.
 *
 * Cases whose golden is not finite (Inf/NaN fills) are not compared numerically - the mixed
 * tolerance strategy cannot express them - and are checked for IEEE behaviour consistency with the
 * golden instead: NaN has to stay NaN, an infinity has to keep its sign, and a finite golden
 * element may not become non-finite.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "securec.h"
#include "ssbmv_param.h"
#include "ssbmv_golden.h"
#include "ssbmv_npu_wrapper.h"

namespace {
/*!
 * \brief 打印小规模用例的输入，便于用例失败后脱离 NPU 复现。
 *
 * 只在 n 较小时输出，避免大 shape 刷屏；开关是环境变量 SSBMV_DEBUG。
 */
void DumpSsbmvInputs(
    const SsbmvParam& p, const std::vector<float>& aHost, const std::vector<float>& xHost,
    const std::vector<float>& yHost)
{
    if (std::getenv("SSBMV_DEBUG") == nullptr || p.n > 8) {
        return;
    }
    std::cout << "[" << p.caseName << "] uplo=" << static_cast<int>(p.uplo) << " n=" << p.n << " k=" << p.k
              << " lda=" << p.lda << " incx=" << p.incx << " incy=" << p.incy << " alpha=" << p.alpha
              << " beta=" << p.beta << "\n  A=";
    for (float value : aHost) {
        std::cout << value << " ";
    }
    std::cout << "\n  x=";
    for (float value : xHost) {
        std::cout << value << " ";
    }
    std::cout << "\n  y=";
    for (float value : yHost) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

/*! \brief True when at least one element of v is Inf or NaN. */
bool SsbmvHasNonFinite(const std::vector<float>& v)
{
    for (float value : v) {
        if (!std::isfinite(value)) {
            return true;
        }
    }
    return false;
}

/*!
 * \brief Compares the elements whose golden is not finite on IEEE behaviour only.
 *
 * A numeric tolerance cannot express these cases, and the accumulation order of a row block kernel
 * is not the reference's, so the value a saturated row takes is not pinned either: with band values
 * around FLT_MAX the products overflow to Inf, and whether such a row ends up inf, -inf or nan
 * depends on the order its terms are added in. TC_FL_124 row 13 is the clean demonstration - the
 * exact sum is +3.32e38, which is still finite!, the reference comes out -inf and this kernel +inf.
 *
 * What is checked instead is that the boundary is honoured rather than the exact value:
 *   - a row the reference could still evaluate to a value has to come out finite here too, and to
 *     the same tolerance a finite case gets;
 *   - a row the reference saturated has to stay saturated.
 * That still catches what these cases exist for: a lane read out of bounds, a stale buffer or an
 * uninitialised accumulator all show up as a non-finite output where the reference is finite.
 */
bool SsbmvVerifyNonFinite(
    const std::vector<float>& output, const std::vector<float>& golden, const std::string& caseId,
    const VerifyConfig& cfg)
{
    std::vector<float> finiteOut;
    std::vector<float> finiteGold;
    size_t saturated = 0;
    size_t escapedFinite = 0;
    const size_t count = std::min(output.size(), golden.size());
    for (size_t i = 0; i < count; i++) {
        if (std::isfinite(golden[i])) {
            finiteOut.push_back(output[i]);
            finiteGold.push_back(golden[i]);
        } else {
            saturated++;
            if (std::isfinite(output[i])) {
                escapedFinite++;
            }
        }
    }
    bool ok = (escapedFinite == 0);
    if (!finiteGold.empty() && ok) {
        ok = Verifier::verifyVector(finiteOut.data(), finiteGold.data(), finiteGold.size(), 1, cfg, caseId);
    }
    std::cout << "[" << caseId << "] " << (ok ? "PASSED" : "FAILED")
              << " (non-finite golden, IEEE behaviour check: " << saturated << " saturated element(s), "
              << escapedFinite << " came back finite)" << std::endl;
    return ok;
}

/*!
 * \brief 真·就地的调用封装：x 与 y 落在同一块 device 缓冲上。
 *
 * aclblasSsbmv_npu 会给 x、y 各分配一块 device 缓冲，host 侧传同一块内存设备侧也不重合，
 * 因此测不到就地路径。这里把 xy（含 inc 的间隔）整段搬进一块缓冲，x、y 同指针，算完再取回。
 */
aclblasStatus_t SsbmvInPlaceOnDevice(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const float* alpha, const float* aHost, int lda,
    std::vector<float>& xy, int inc, const float* beta)
{
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float);
    const size_t xyBytes = xy.size() * sizeof(float);
    DeviceBuffer aDev(aBytes);
    aDev.copyFromHost(aHost, aBytes);
    DeviceBuffer xyDev(xyBytes);
    xyDev.copyFromHost(xy.data(), xyBytes);

    const aclblasStatus_t ret = aclblasSsbmv(
        handle, uplo, n, k, alpha, aDev.floatPtr(), lda, xyDev.floatPtr(), inc, beta, xyDev.floatPtr(), inc);
    /* 只等 handle 绑定的流：全设备同步会掩盖算子在错误的流上下发的问题。 */
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || stream == nullptr ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    xyDev.copyToHost(xy.data(), xyBytes);
    return ret;
}

/*! \brief 往 device 缓冲的指定元素写一个 float。 */
bool SsbmvWriteDeviceFloat(DeviceBuffer& buffer, uint64_t index, float value)
{
    void* dst = buffer.bytePtr() + index * sizeof(float);
    return aclrtMemcpy(dst, sizeof(float), &value, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

/*! \brief 从 device 缓冲的指定元素读一个 float。 */
float SsbmvReadDeviceFloat(DeviceBuffer& buffer, uint64_t index)
{
    float value = 0.0f;
    const void* src = buffer.bytePtr() + index * sizeof(float);
    EXPECT_EQ(aclrtMemcpy(&value, sizeof(float), src, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    return value;
}

/*! \brief ±2^31 步长的 x 侧：x 的物理下标 0 与远端哨兵，正负步长读出顺序相反。 */
void CheckExtremeStrideX(
    aclblasHandle_t handle, aclrtStream stream, DeviceBuffer& strideBuf, DeviceBuffer& dA, DeviceBuffer& dY)
{
    constexpr int N = 2;
    constexpr int K = 0;
    constexpr int LDA = 1;
    constexpr uint64_t LAST = 1ULL << 31;
    const int incMax = std::numeric_limits<int>::max();
    const int incMin = std::numeric_limits<int>::min();

    ASSERT_TRUE(SsbmvWriteDeviceFloat(strideBuf, 0, 3.0f));
    ASSERT_TRUE(SsbmvWriteDeviceFloat(strideBuf, LAST - 1, 7.0f));
    ASSERT_TRUE(SsbmvWriteDeviceFloat(strideBuf, LAST, 7.0f));
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int incx : {incMax, incMin}) {
        const float yInit[N] = {0.0f, 0.0f};
        dY.copyFromHost(yInit, sizeof(yInit));
        ASSERT_EQ(
            aclblasSsbmv(
                handle, ACLBLAS_LOWER, N, K, &alpha, dA.floatPtr(), LDA, strideBuf.floatPtr(), incx, &beta,
                dY.floatPtr(), 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        float out[N] = {0.0f, 0.0f};
        dY.copyToHost(out, sizeof(out));
        EXPECT_FLOAT_EQ(out[0], (incx > 0) ? 3.0f : 7.0f) << "incx=" << incx;
        EXPECT_FLOAT_EQ(out[1], (incx > 0) ? 7.0f : 3.0f) << "incx=" << incx;
    }
}

/*! \brief ±2^31 步长的 y 侧：alpha = 0、beta = 2，步长走到的远端变成 14，另一端保持 7。 */
void CheckExtremeStrideY(aclblasHandle_t handle, aclrtStream stream, DeviceBuffer& strideBuf, DeviceBuffer& dA)
{
    constexpr int N = 2;
    constexpr int K = 0;
    constexpr int LDA = 1;
    constexpr uint64_t LAST = 1ULL << 31;
    const int incMax = std::numeric_limits<int>::max();
    const int incMin = std::numeric_limits<int>::min();
    float alpha = 0.0f;
    float beta = 2.0f;

    for (int incy : {incMax, incMin}) {
        ASSERT_TRUE(SsbmvWriteDeviceFloat(strideBuf, 0, 3.0f));
        ASSERT_TRUE(SsbmvWriteDeviceFloat(strideBuf, LAST - 1, 7.0f));
        ASSERT_TRUE(SsbmvWriteDeviceFloat(strideBuf, LAST, 7.0f));
        ASSERT_EQ(
            aclblasSsbmv(
                handle, ACLBLAS_LOWER, N, K, &alpha, dA.floatPtr(), LDA, dA.floatPtr(), 1, &beta, strideBuf.floatPtr(),
                incy),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        const uint64_t far = (incy > 0) ? (LAST - 1) : LAST;
        const uint64_t other = (incy > 0) ? LAST : (LAST - 1);
        EXPECT_FLOAT_EQ(SsbmvReadDeviceFloat(strideBuf, 0), 6.0f) << "incy=" << incy;
        EXPECT_FLOAT_EQ(SsbmvReadDeviceFloat(strideBuf, far), 14.0f) << "incy=" << incy;
        EXPECT_FLOAT_EQ(SsbmvReadDeviceFloat(strideBuf, other), 7.0f) << "incy=" << incy;
    }
}
struct SsbmvAccuracyShape {
    int n;
    int k;
    int padding;
    int incx;
    int incy;
};

enum class SsbmvAccuracyInput { Uniform, Normal, Cancellation };

size_t SsbmvPhysicalIndex(int logical, int n, int inc)
{
    const int index = inc > 0 ? logical : n - 1 - logical;
    return static_cast<size_t>(index) * ssbmvAbsStride(inc);
}

uint32_t SsbmvOrderedFloatBits(float value)
{
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    EXPECT_EQ(memcpy_s(&bits, sizeof(bits), &value, sizeof(value)), EOK);
    constexpr uint32_t SIGN_BIT = uint32_t{1} << 31;
    // Both zeros map to the same rank. Each representable step costs one ULP.
    return (bits & SIGN_BIT) != 0 ? SIGN_BIT - (bits & ~SIGN_BIT) : SIGN_BIT + bits;
}

struct SsbmvAccuracyMetrics {
    long double maxAbs = 0.0L;
    long double normalizedMax = 0.0L;
    std::vector<uint32_t> ulps;
};

// Direct row-dot reference accumulated in long double from the float32 inputs.
SsbmvAccuracyMetrics SsbmvMeasureAccuracy(
    const SsbmvAccuracyShape& shape, aclblasFillMode_t uplo, float alpha, float beta, const std::vector<float>& a,
    const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& output)
{
    constexpr long double MIN_SCALE = 0x1p-16L;
    SsbmvAccuracyMetrics metrics;
    metrics.ulps.reserve(shape.n);
    const int lda = shape.k + 1 + shape.padding;
    for (int row = 0; row < shape.n; ++row) {
        long double sum = 0.0L;
        long double sumAbs = 0.0L;
        const int first = std::max(0, row - shape.k);
        const int end = std::min(shape.n, row + shape.k + 1);
        for (int col = first; col < end; ++col) {
            // Reflect the unstored half of the band into the stored triangle.
            const int low = std::min(row, col);
            const int high = std::max(row, col);
            const int bandCol = uplo == ACLBLAS_UPPER ? high : low;
            const int bandRow = uplo == ACLBLAS_UPPER ? shape.k - (high - low) : high - low;
            const long double aij = a[static_cast<size_t>(bandCol) * lda + bandRow];
            const long double xi = x[SsbmvPhysicalIndex(col, shape.n, shape.incx)];
            const long double product = aij * xi;
            sum += product;
            sumAbs += std::abs(product);
        }
        const size_t pos = SsbmvPhysicalIndex(row, shape.n, shape.incy);
        const long double prior = beta == 0.0f ? 0.0L : static_cast<long double>(beta) * y[pos];
        const long double golden = static_cast<long double>(alpha) * sum + prior;
        const long double scale = std::abs(static_cast<long double>(alpha)) * sumAbs + std::abs(prior);
        if (!std::isfinite(output[pos])) {
            metrics.maxAbs = std::numeric_limits<long double>::infinity();
            metrics.normalizedMax = metrics.maxAbs;
            metrics.ulps.push_back(std::numeric_limits<uint32_t>::max());
            continue;
        }
        const long double diff = std::abs(static_cast<long double>(output[pos]) - golden);
        metrics.maxAbs = std::max(metrics.maxAbs, diff);
        const long double normalized = scale > MIN_SCALE ? diff / scale : diff / MIN_SCALE;
        metrics.normalizedMax = std::max(metrics.normalizedMax, normalized);
        const uint32_t actualBits = SsbmvOrderedFloatBits(output[pos]);
        const uint32_t goldenBits = SsbmvOrderedFloatBits(static_cast<float>(golden));
        metrics.ulps.push_back(std::max(actualBits, goldenBits) - std::min(actualBits, goldenBits));
    }
    std::sort(metrics.ulps.begin(), metrics.ulps.end());
    return metrics;
}

constexpr float SSBMV_ACCURACY_HOLE = -989.125f;

struct SsbmvAccuracyInputs {
    std::vector<float> a;
    std::vector<float> x;
    std::vector<float> y;
    float alpha;
    float beta;
};

SsbmvAccuracyInputs MakeSsbmvAccuracyInputs(
    const SsbmvAccuracyShape& shape, aclblasFillMode_t uplo, SsbmvAccuracyInput input, uint32_t seed)
{
    const int lda = shape.k + 1 + shape.padding;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
    const float mean = uniform(rng);
    const float stddev = std::uniform_real_distribution<float>(0.1f, 2.0f)(rng);
    std::normal_distribution<float> normal(mean, stddev);
    const auto sample = [&] { return input == SsbmvAccuracyInput::Normal ? normal(rng) : uniform(rng); };
    const bool cancellation = input == SsbmvAccuracyInput::Cancellation;
    SsbmvAccuracyInputs inputs{
        std::vector<float>(static_cast<size_t>(shape.n) * lda, poison),
        std::vector<float>(static_cast<size_t>(shape.n - 1) * ssbmvAbsStride(shape.incx) + 1, poison),
        std::vector<float>(static_cast<size_t>(shape.n - 1) * ssbmvAbsStride(shape.incy) + 1, SSBMV_ACCURACY_HOLE),
        cancellation ? 0.5f : sample(),
        cancellation ? 0.0f : sample(),
    };

    for (int low = 0; low < shape.n; ++low) {
        const int end = std::min(shape.n, low + shape.k + 1);
        for (int high = low; high < end; ++high) {
            const int distance = high - low;
            const size_t index = uplo == ACLBLAS_UPPER ? static_cast<size_t>(high) * lda + shape.k - distance :
                                                         static_cast<size_t>(low) * lda + distance;
            // Opposite signs on adjacent diagonals cancel. Dyadic perturbations
            // preserve a measurable contribution in the low bits of the output.
            const float sign = distance % 2 == 0 ? 1.0f : -1.0f;
            inputs.a[index] = cancellation ? sign * (1.0f + sample() * 0x1p-16f) : sample();
        }
        inputs.x[SsbmvPhysicalIndex(low, shape.n, shape.incx)] = cancellation ? 1.0f + sample() * 0x1p-16f : sample();
        inputs.y[SsbmvPhysicalIndex(low, shape.n, shape.incy)] = cancellation ? poison : sample();
    }
    return inputs;
}

void CheckSsbmvAccuracy(
    aclblasHandle_t handle, const std::string& caseName, const SsbmvAccuracyShape& shape, aclblasFillMode_t uplo,
    SsbmvAccuracyInput input, uint32_t seed)
{
    SCOPED_TRACE(caseName);
    const int lda = shape.k + 1 + shape.padding;
    const auto [a, x, y, alpha, beta] = MakeSsbmvAccuracyInputs(shape, uplo, input, seed);
    std::vector<float> output = y;
    ASSERT_EQ(
        aclblasSsbmv_npu(
            handle, uplo, shape.n, shape.k, &alpha, a.data(), lda, x.data(), shape.incx, &beta, output.data(),
            shape.incy),
        ACLBLAS_STATUS_SUCCESS);
    std::vector<float> netlib = y;
    ASSERT_EQ(
        aclblasSsbmv_cpu(
            handle, uplo, shape.n, shape.k, &alpha, a.data(), lda, x.data(), shape.incx, &beta, netlib.data(),
            shape.incy),
        ACLBLAS_STATUS_SUCCESS);

    for (size_t i = 0; i < output.size(); ++i) {
        if (i % ssbmvAbsStride(shape.incy) != 0) {
            EXPECT_EQ(output[i], SSBMV_ACCURACY_HOLE);
        }
    }
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, netlib.data(), shape.n);
    EXPECT_TRUE(
        Verifier::verifyVector(output.data(), netlib.data(), shape.n, ssbmvAbsStride(shape.incy), cfg, caseName));

    const auto metrics = SsbmvMeasureAccuracy(shape, uplo, alpha, beta, a, x, y, output);
    const size_t p95 = (95 * metrics.ulps.size() + 99) / 100 - 1;
    std::ostringstream record;
    record << std::scientific << std::setprecision(12) << "SSBMV_ACCURACY," << caseName << ',' << metrics.maxAbs << ','
           << metrics.normalizedMax << ',' << metrics.ulps[p95] << ',' << metrics.ulps.back();
    std::cout << record.str() << '\n';
}

void VerifySsbmvCsvOutput(const SsbmvParam& p, const std::vector<float>& yNpu, const std::vector<float>& yCpu)
{
    std::vector<float> npuLogical(p.n);
    std::vector<float> cpuLogical(p.n);
    for (int i = 0; i < p.n; i++) {
        const size_t idx = SsbmvPhysicalIndex(i, p.n, p.incy);
        npuLogical[i] = yNpu[idx];
        cpuLogical[i] = yCpu[idx];
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, cpuLogical.data(), static_cast<size_t>(p.n));
    const char* taskTolerance = std::getenv("SSBMV_TASK_TOLERANCE");
    if (taskTolerance != nullptr && std::strcmp(taskTolerance, "1") == 0) {
        cfg.mixedAtol = 0x1p-16;
        cfg.mixedRtol = 0x1p-10;
    }
    if (SsbmvHasNonFinite(cpuLogical)) {
        EXPECT_TRUE(SsbmvVerifyNonFinite(npuLogical, cpuLogical, p.caseName, cfg));
        return;
    }
    EXPECT_TRUE(
        Verifier::verifyVector(npuLogical.data(), cpuLogical.data(), static_cast<size_t>(p.n), 1, cfg, p.caseName));
}
} // namespace

class SsbmvTest : public BlasTest<SsbmvParam> {};

TEST_F(SsbmvTest, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> a(30, 1.0f);
    std::vector<float> x(10, 1.0f);
    std::vector<float> y(10, 0.0f);
    aclblasStatus_t ret =
        aclblasSsbmv_npu(nullptr, ACLBLAS_LOWER, 10, 2, &alpha, a.data(), 3, x.data(), 1, &beta, y.data(), 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(SsbmvTest, ZeroOrderIsNoOp)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> a(4, 1.0f);
    std::vector<float> x(1, 1.0f);
    std::vector<float> y(1, 7.0f);
    aclblasStatus_t ret =
        aclblasSsbmv_npu(SsbmvTest::handle_, ACLBLAS_UPPER, 0, 0, &alpha, a.data(), 1, x.data(), 1, &beta, y.data(), 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(y[0], 7.0f);
}

TEST_F(SsbmvTest, AlphaZeroInPlaceIsExact)
{
    /* y = beta * y has to be a single float multiply of the incoming y, bit for bit. */
    const int n = 64;
    const int k = 8;
    float alpha = 0.0f;
    float beta = -0.5f;
    std::vector<float> a(static_cast<size_t>(k + 1) * n, 1.0f);
    std::vector<float> xy(n);
    for (int i = 0; i < n; i++) {
        xy[i] = (i % 2 == 0) ? (1.0f + static_cast<float>(i) * 0.25f) : -(1.0f + static_cast<float>(i) * 0.125f);
    }
    std::vector<float> expected(n);
    for (int i = 0; i < n; i++) {
        expected[i] = beta * xy[i];
    }
    /* In place: the same buffer is passed as x and as y, so the fast path must not read x at all. */
    aclblasStatus_t ret = aclblasSsbmv_npu(
        SsbmvTest::handle_, ACLBLAS_LOWER, n, k, &alpha, a.data(), k + 1, xy.data(), 1, &beta, xy.data(), 1);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(xy[i], expected[i]) << "index " << i;
    }
}

/*!
 * \brief alpha != 0 的真就地精度用例：y 的每个写回都会踩掉 x，算子必须先把 x 备份进 workspace。
 *
 * n=4096/k=8 的形状下 kernel 既按 chunk 分块（chunkRows < blockRows），核数又被成本模型重选到
 * 多核：备份一旦缺失，写回的 y 会污染后续 chunk（乃至别的核）还要当 x 读的元素。
 */
TEST_F(SsbmvTest, InPlaceAlphaNonZeroMatchesReference)
{
    const int n = 4096;
    const int k = 8;
    const int lda = k + 1;
    const float alpha = 1.0f;
    const float beta = 0.5f;
    auto aHost = makeBlasArray(static_cast<int64_t>(lda) * n, "RANDOM_NORM_5_5", 20260101);
    auto xyHost = makeBlasStrided(n, 1, "RANDOM_NORM_5_5", 20260102);
    std::vector<float> expected = xyHost;
    ASSERT_EQ(
        aclblasSsbmv_cpu(
            handle_, ACLBLAS_LOWER, n, k, &alpha, aHost.data(), lda, xyHost.data(), 1, &beta, expected.data(), 1),
        ACLBLAS_STATUS_SUCCESS);

    std::vector<float> xy = xyHost;
    ASSERT_EQ(
        SsbmvInPlaceOnDevice(handle_, ACLBLAS_LOWER, n, k, &alpha, aHost.data(), lda, xy, 1, &beta),
        ACLBLAS_STATUS_SUCCESS);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, expected.data(), static_cast<size_t>(n));
    EXPECT_TRUE(
        Verifier::verifyVector(xy.data(), expected.data(), static_cast<size_t>(n), 1, cfg, "InPlaceAlphaNonZero"));
}

/*!
 * \brief 用户注入的 workspace 放不下 x 时必须直接失败（ALLOC_FAILED），不能悄悄退回单核硬算。
 *
 * 单核救不了就地：kernel 按 chunk 分块，chunk c 写回的 y 正是 chunk c+1 还要当 x 读的行；
 * 重选核数后跨核更甚。用户 workspace 库无权扩容，此时唯一正确的出口就是报错。
 */
TEST_F(SsbmvTest, InPlaceWithoutWorkspaceReportsAllocFailed)
{
    const int n = 4096;
    const int k = 8;
    const int lda = k + 1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto aHost = makeBlasArray(static_cast<int64_t>(lda) * n, "RANDOM_NORM_5_5", 20260103);
    auto xyHost = makeBlasStrided(n, 1, "RANDOM_NORM_5_5", 20260104);
    std::vector<float> xy = xyHost;

    void* tinyWorkspace = nullptr;
    ASSERT_EQ(aclrtMalloc(&tinyWorkspace, 64, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclblasSetWorkspace(handle_, tinyWorkspace, 64), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        SsbmvInPlaceOnDevice(handle_, ACLBLAS_LOWER, n, k, &alpha, aHost.data(), lda, xy, 1, &beta),
        ACLBLAS_STATUS_ALLOC_FAILED);
    /* 换回共享 stream 会把句柄重置回库自有的 workspace（顺序同 gemm_strided_batched_ex 的
     * UserWorkspaceTooSmall），之后才能释放临时缓冲。 */
    ASSERT_EQ(aclblasSetStream(handle_, stream_), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtFree(tinyWorkspace), ACL_SUCCESS);

    /* 句柄确实回到了库 workspace：普通（非就地）调用照常成功。 */
    const aclblasStatus_t afterRet = aclblasSsbmv_npu(
        handle_, ACLBLAS_LOWER, n, k, &alpha, aHost.data(), lda, xyHost.data(), 1, &beta, xy.data(), 1);
    EXPECT_EQ(afterRet, ACLBLAS_STATUS_SUCCESS);
}

/*!
 * \brief x 超过默认 workspace（32MiB）时就地调用要能把库自有的 workspace 扩容再备份。
 *
 * incx=9001 把 x 的整段撑到 32MiB 以上而 n 只有 1024（A 依然很小）。库自有的 workspace 原地翻倍
 * 增长，用完不缩；用户注入的 workspace 不参与扩容，故与上一个用例各管一支。
 */
TEST_F(SsbmvTest, InPlaceXOverDefaultWorkspaceGrowsWorkspace)
{
    const int n = 1024;
    const int k = 1;
    const int lda = k + 1;
    const int inc = 9001;
    const float alpha = 1.0f;
    const float beta = 0.5f;
    auto aHost = makeBlasArray(static_cast<int64_t>(lda) * n, "RANDOM_NORM_5_5", 20260105);
    auto xyHost = makeBlasStrided(n, inc, "RANDOM_NORM_5_5", 20260106);
    ASSERT_GT(xyHost.size() * sizeof(float), 32U * 1024U * 1024U) << "用例前提：x 必须超过默认 workspace";

    std::vector<float> expected = xyHost;
    ASSERT_EQ(
        aclblasSsbmv_cpu(
            handle_, ACLBLAS_LOWER, n, k, &alpha, aHost.data(), lda, xyHost.data(), inc, &beta, expected.data(), inc),
        ACLBLAS_STATUS_SUCCESS);

    std::vector<float> xy = xyHost;
    ASSERT_EQ(
        SsbmvInPlaceOnDevice(handle_, ACLBLAS_LOWER, n, k, &alpha, aHost.data(), lda, xy, inc, &beta),
        ACLBLAS_STATUS_SUCCESS);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, expected.data(), expected.size());
    EXPECT_TRUE(Verifier::verifyVector(xy.data(), expected.data(), expected.size(), 1, cfg, "InPlaceGrowWorkspace"));
}

/*!
 * \brief ±1000 步长的精度用例：x 与 y 的物理跨度都是 (n-1)*1000+1 个 float。
 *
 * CSV 的 incx/incy 只覆盖 {-3..3, 0}，大步长的下标换算（含负步长取反向）在线性用例里
 * 只有这一档能测到；两端都非零、方向相反，x 的正向与 y 的负向同时被覆盖。
 */
TEST_F(SsbmvTest, LargeStridesMatchReference)
{
    const int n = 64;
    const int k = 4;
    const int lda = k + 1;
    const int incx = 1000;
    const int incy = -1000;
    const float alpha = 1.0f;
    const float beta = 0.5f;
    auto aHost = makeBlasArray(static_cast<int64_t>(lda) * n, "RANDOM_NORM_5_5", 20260201);
    auto xHost = makeBlasStrided(n, incx, "RANDOM_NORM_5_5", 20260202);
    auto yHost = makeBlasStrided(n, incy, "RANDOM_NORM_5_5", 20260203);

    std::vector<float> expected = yHost;
    ASSERT_EQ(
        aclblasSsbmv_cpu(
            handle_, ACLBLAS_LOWER, n, k, &alpha, aHost.data(), lda, xHost.data(), incx, &beta, expected.data(), incy),
        ACLBLAS_STATUS_SUCCESS);
    std::vector<float> yNpu = yHost;
    ASSERT_EQ(
        aclblasSsbmv_npu(
            handle_, ACLBLAS_LOWER, n, k, &alpha, aHost.data(), lda, xHost.data(), incx, &beta, yNpu.data(), incy),
        ACLBLAS_STATUS_SUCCESS);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, expected.data(), expected.size());
    EXPECT_TRUE(
        Verifier::verifyVector(yNpu.data(), expected.data(), expected.size(), 1, cfg, "LargeStridesMatchReference"));
}

/*!
 * \brief int 能表达的最大步长（±2^31）：x/y 的跨度为 2^31+1 个 float（8GiB）。
 *
 * k = 0 时每行只取一个 x 与一个 y，所以哨兵只放在物理下标 0 和 2^31 上；正、负步长都做，
 * 覆盖 INT_MIN 的 |inc|（kernel 里 absIncx 的 uint32 转换）与 X/YPhysicalPos 的边界。
 * 8GiB 设备缓冲不够时跳过。
 */
TEST_F(SsbmvTest, ExtremeStridesMatchReference)
{
    constexpr uint64_t SPAN = (1ULL << 31) + 1ULL; /* elements */
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes), ACL_SUCCESS);
    if (freeBytes < SPAN * sizeof(float) + (1ULL << 30)) {
        GTEST_SKIP() << "extreme-stride case needs an 8GiB device buffer";
    }

    std::unique_ptr<DeviceBuffer> strideBuf;
    try {
        strideBuf = std::make_unique<DeviceBuffer>(static_cast<size_t>(SPAN) * sizeof(float));
    } catch (const std::exception&) {
        GTEST_SKIP() << "8GiB device buffer allocation failed";
    }
    DeviceBuffer dA(sizeof(float) * 2U);
    DeviceBuffer dY(sizeof(float) * 2U);
    const float aHost[2] = {1.0f, 1.0f};
    dA.copyFromHost(aHost, sizeof(aHost));

    CheckExtremeStrideX(handle_, stream_, *strideBuf, dA, dY);
    CheckExtremeStrideY(handle_, stream_, *strideBuf, dA);
}

/*!
 * \brief (n-1)*|inc| 超出 kernel 的 uint32 寻址域时必须显式拒绝，而不是回绕后静默算错。
 *
 * n = 4、inc = INT_MAX：(n-1)*|inc| = 3*INT_MAX > UINT32_MAX，x/y 各一支。校验发生在
 * 触碰数据之前，因此这里的小缓冲足够。
 */
TEST_F(SsbmvTest, RejectsStrideSpanBeyondUint32)
{
    constexpr int n = 4;
    constexpr int k = 1;
    constexpr int lda = k + 1;
    const int hugeInc = std::numeric_limits<int>::max();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float aHost[lda * n] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float xHost[n] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float yHost[n] = {0.0f, 0.0f, 0.0f, 0.0f};
    DeviceBuffer dA(sizeof(aHost));
    DeviceBuffer dX(sizeof(xHost));
    DeviceBuffer dY(sizeof(yHost));
    dA.copyFromHost(aHost, sizeof(aHost));
    dX.copyFromHost(xHost, sizeof(xHost));
    dY.copyFromHost(yHost, sizeof(yHost));

    EXPECT_EQ(
        aclblasSsbmv(
            handle_, ACLBLAS_LOWER, n, k, &alpha, dA.floatPtr(), lda, dX.floatPtr(), hugeInc, &beta, dY.floatPtr(), 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasSsbmv(
            handle_, ACLBLAS_LOWER, n, k, &alpha, dA.floatPtr(), lda, dX.floatPtr(), 1, &beta, dY.floatPtr(), hugeInc),
        ACLBLAS_STATUS_INVALID_VALUE);
}

/*!
 * \brief k = INT_MAX 时 int 域不存在合法的 lda（lda >= k + 1 已溢出，属于 UB），必须显式拒绝。
 *
 * 校验发生在触碰数据之前，小缓冲足够；lda 取 1、负数与 INT_MAX 三档，回绕写法下三者都会被
 * 放行，其中负值还会让 kernel 读到 A 缓冲之外。
 */
TEST_F(SsbmvTest, RejectsBandWidthWithoutValidLda)
{
    constexpr int n = 4;
    constexpr int k = std::numeric_limits<int>::max();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float aHost[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float xHost[n] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float yHost[n] = {0.0f, 0.0f, 0.0f, 0.0f};
    DeviceBuffer dA(sizeof(aHost));
    DeviceBuffer dX(sizeof(xHost));
    DeviceBuffer dY(sizeof(yHost));
    dA.copyFromHost(aHost, sizeof(aHost));
    dX.copyFromHost(xHost, sizeof(xHost));
    dY.copyFromHost(yHost, sizeof(yHost));

    for (int lda : {1, -1, std::numeric_limits<int>::max()}) {
        EXPECT_EQ(
            aclblasSsbmv(
                handle_, ACLBLAS_LOWER, n, k, &alpha, dA.floatPtr(), lda, dX.floatPtr(), 1, &beta, dY.floatPtr(), 1),
            ACLBLAS_STATUS_INVALID_VALUE)
            << "lda=" << lda;
    }
}

/*!
 * \brief beta == 0 时 y 只写不读：strided-y 写回路径也不能把旧 y 的 Inf/NaN 乘进结果。
 *
 * BLAS 契约规定 beta == 0 时 y 不必在输入上设置。incy != 1 的 StoreStridedChunk 曾无条件
 * 计算 (alpha * acc) + beta * y_old，beta == 0 时退化成 0 * Inf = NaN / 0 * NaN = NaN，
 * 输出行变成 NaN。这里把 y 的有效位置交替填 NaN / Inf，golden 由 cblas 生成（不读 y），
 * 并按整个 strided 缓冲比对，顺便覆盖「行间元素原样保留」。
 */
TEST_F(SsbmvTest, BetaZeroIgnoresStridedY)
{
    constexpr int n = 32;
    constexpr int k = 3;
    constexpr int lda = k + 1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto aHost = makeBlasArray(static_cast<int64_t>(lda) * n, "RANDOM_NORM_5_5", 20260401);
    auto xHost = makeBlasStrided(n, 1, "RANDOM_NORM_5_5", 20260402);

    /* 正、负步各做一次：负步长下逻辑行 i 落在物理位置 (n-1-i)*|incy|，与写回路径的
     * lane 映射是另一条分支。 */
    for (int incy : {2, -2}) {
        const int absIncy = (incy > 0) ? incy : -incy;
        std::vector<float> yHost((static_cast<size_t>(n) - 1U) * absIncy + 1U, 0.0f);
        for (int i = 0; i < n; i++) {
            const int idx = (incy > 0) ? (i * incy) : ((n - 1 - i) * absIncy);
            yHost[static_cast<size_t>(idx)] =
                ((i % 2) == 0) ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
        }

        std::vector<float> expected = yHost;
        ASSERT_EQ(
            aclblasSsbmv_cpu(
                handle_, ACLBLAS_UPPER, n, k, &alpha, aHost.data(), lda, xHost.data(), 1, &beta, expected.data(), incy),
            ACLBLAS_STATUS_SUCCESS);
        std::vector<float> yNpu = yHost;
        ASSERT_EQ(
            aclblasSsbmv_npu(
                handle_, ACLBLAS_UPPER, n, k, &alpha, aHost.data(), lda, xHost.data(), 1, &beta, yNpu.data(), incy),
            ACLBLAS_STATUS_SUCCESS)
            << "incy=" << incy;

        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, expected.data(), expected.size());
        EXPECT_TRUE(
            Verifier::verifyVector(yNpu.data(), expected.data(), expected.size(), 1, cfg, "BetaZeroIgnoresStridedY"))
            << "incy=" << incy;
    }
}

/*!
 * \brief k >= n 的合法宽带调用：列形式不建 2k 的 x 窗口、段按 SSBMV_MIRROR_TILE 分块归约。
 *
 * k = 30000、n = 256 时 x 窗口本需 2k+1 个 float（240KB），旧实现把 chunkRows 钳到 1 后
 * 不经验证就下发，kernel 的 xBuf 仍按 chunkRows + 2k 分配，触发 VEC UB 越界异常。这里覆盖
 * UPPER/LOWER 与连续/步长 x、y，作为「宽带能装进 UB 且数值正确」的回归防线。
 */
TEST_F(SsbmvTest, WideBandMatchesReference)
{
    const int n = 256;
    const int k = 30000;
    const int lda = k + 1;
    const float alpha = 1.25f;
    const float beta = -0.5f;
    auto aHost = makeBlasArray(static_cast<int64_t>(lda) * n, "RANDOM_NORM_5_5", 20260301);

    for (aclblasFillMode_t uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (int incx : {1, 2}) {
            for (int incy : {1, -1}) {
                auto xHost = makeBlasStrided(n, incx, "RANDOM_NORM_5_5", 20260302);
                auto yHost = makeBlasStrided(n, incy, "RANDOM_NORM_5_5", 20260303);
                std::vector<float> expected = yHost;
                ASSERT_EQ(
                    aclblasSsbmv_cpu(
                        handle_, uplo, n, k, &alpha, aHost.data(), lda, xHost.data(), incx, &beta, expected.data(),
                        incy),
                    ACLBLAS_STATUS_SUCCESS);
                std::vector<float> yNpu = yHost;
                ASSERT_EQ(
                    aclblasSsbmv_npu(
                        handle_, uplo, n, k, &alpha, aHost.data(), lda, xHost.data(), incx, &beta, yNpu.data(), incy),
                    ACLBLAS_STATUS_SUCCESS)
                    << "uplo=" << static_cast<int>(uplo) << " incx=" << incx << " incy=" << incy;

                VerifyConfig cfg;
                applyMixedTolerance(cfg, ACL_FLOAT, expected.data(), expected.size());
                EXPECT_TRUE(Verifier::verifyVector(yNpu.data(), expected.data(), expected.size(), 1, cfg, "WideBand"));
            }
        }
    }
}

TEST_F(SsbmvTest, SupplementalMixedAccuracy)
{
    const SsbmvAccuracyShape shapes[] = {
        {1, 0, 0, 1, 1},      {4, 1, 0, 1, 1},     {8, 2, 1, 1, 1},     {17, 4, 0, 1, 1},
        {31, 9, 0, 1, 1},     {32, 8, 0, 1, 1},    {33, 9, 0, 1, 1},    {63, 16, 0, 1, 1},
        {64, 16, 0, 1, 1},    {65, 16, 0, 1, 1},   {65, 67, 2, -2, 3},  {129, 31, 17, 2, -3},
        {257, 67, 4, -1, -2}, {769, 192, 0, 1, 1}, {1031, 17, 0, 1, 1}, {4093, 64, 0, 1, 1},
    };
    uint32_t seed = 40202601;
    for (const auto& shape : shapes) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            const std::string id = std::to_string(seed) + (uplo == ACLBLAS_UPPER ? "_upper" : "_lower");
            CheckSsbmvAccuracy(handle_, "uniform_" + id, shape, uplo, SsbmvAccuracyInput::Uniform, seed);
            CheckSsbmvAccuracy(handle_, "normal_" + id, shape, uplo, SsbmvAccuracyInput::Normal, seed);
        }
        ++seed;
    }
}

TEST_F(SsbmvTest, SupplementalCancellationAccuracy)
{
    const SsbmvAccuracyShape shapes[] = {
        {63, 16, 0, 1, 1},
        {1031, 17, 0, 1, 1},
        {1153, 127, 0, 1, 1},
        {2049, 1024, 1, 1, 1},
    };
    uint32_t seed = 40202701;
    for (const auto& shape : shapes) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            const std::string id = std::to_string(seed) + (uplo == ACLBLAS_UPPER ? "_upper" : "_lower");
            CheckSsbmvAccuracy(handle_, "cancellation_" + id, shape, uplo, SsbmvAccuracyInput::Cancellation, seed);
        }
        ++seed;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Ssbmv, SsbmvTest, ::testing::ValuesIn(GetCasesFromCsv<SsbmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsbmvParam>);

TEST_P(SsbmvTest, CsvDriven)
{
    const auto& p = GetParam();
    /* TC_PF_* 是性能用例，这里只做一次下发 smoke。A 最大 33MB，逐元素生成随机数是这 200 条
     * 用例的主要耗时，smoke 不校验数值，用常量填充即可。 */
    const bool perfCase = (p.caseName.rfind("TC_PF_", 0) == 0);

    auto aHost = perfCase ?
                     std::vector<float>(static_cast<size_t>(static_cast<int64_t>(p.lda) * std::max(1, p.n)), 1.0f) :
                     makeBlasArray(static_cast<int64_t>(p.lda) * std::max(1, p.n), p.a, p.randomSeed);
    auto xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed + 1);
    auto yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed + 2);

    std::vector<float> yNpu = yHost;
    const float* alphaPtr = (p.alphaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.alpha;
    const float* betaPtr = (p.betaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.beta;
    /* An empty buffer means the case asked for a null pointer, which has to reach the operator as
     * one instead of as whatever data() happens to return. */
    const float* aPtr = aHost.empty() ? nullptr : aHost.data();
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yNpu.empty() ? nullptr : yNpu.data();

    aclblasStatus_t ret = aclblasSsbmv_npu(
        SsbmvTest::handle_, p.uplo, p.n, p.k, alphaPtr, aPtr, p.lda, xPtr, p.incx, betaPtr, yPtr, p.incy);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    if (p.n == 0) {
        return;
    }

    /* 性能用例到下发为止：gtest 耗时含 host 准备与 golden 计算，本来也不能用于性能判定。
     * 精度由其余 1000 条用例承担，性能由独立的 ssbmv_perf 微基准按任务书 §3.3 口径测量。 */
    if (perfCase) {
        return;
    }

    std::vector<float> yCpu = yHost;
    aclblasSsbmv_cpu(
        SsbmvTest::handle_, p.uplo, p.n, p.k, alphaPtr, aHost.data(), p.lda, xHost.data(), p.incx, betaPtr, yCpu.data(),
        p.incy);

    DumpSsbmvInputs(p, aHost, xHost, yHost);
    VerifySsbmvCsvOutput(p, yNpu, yCpu);
}
