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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "chemm_param.h"
#include "chemm_golden.h"
#include "chemm_npu_wrapper.h"
#include "cblas_compat.h"

ChemmParam::ChemmParam(const csv_map& map) : BlasTestParamBase(map)
{
    side = parseSideMode(ReadMap(map, "side", "LEFT"));
    uplo = parseFillMode(ReadMap(map, "uplo", "LOWER"));
    m = static_cast<int64_t>(parseInt64(ReadMap(map, "m", "0")));
    n = static_cast<int64_t>(parseInt64(ReadMap(map, "n", "0")));

    std::string alphaRealStr = ReadMap(map, "alpha_real", "1.0");
    std::string alphaImagStr = ReadMap(map, "alpha_imag", "0.0");
    nullAlpha = (alphaRealStr == "null" || alphaRealStr == "nullptr");
    alphaReal = nullAlpha ? 0.0f : parseFloat(alphaRealStr, 1.0f);
    alphaImag = nullAlpha ? 0.0f : parseFloat(alphaImagStr, 0.0f);

    aFillSpec = ReadMap(map, "a_fill", "RANDOM_NORM_1E3_1E3");
    aNormal = (aFillSpec == "NORMAL_NORM");
    aFill = parseFill(aFillSpec);
    const std::string ldaStr = ReadMap(map, "lda", "");
    lda = ldaStr.empty() ? (side == ACLBLAS_SIDE_LEFT ? m : n) : static_cast<int64_t>(parseInt64(ldaStr));

    bFillSpec = ReadMap(map, "b_fill", "RANDOM_NORM_1E3_1E3");
    bNormal = (bFillSpec == "NORMAL_NORM");
    bFill = parseFill(bFillSpec);
    const std::string ldbStr = ReadMap(map, "ldb", "");
    ldb = ldbStr.empty() ? m : static_cast<int64_t>(parseInt64(ldbStr));

    std::string betaRealStr = ReadMap(map, "beta_real", "0.5");
    std::string betaImagStr = ReadMap(map, "beta_imag", "0.0");
    nullBeta = (betaRealStr == "null" || betaRealStr == "nullptr");
    betaReal = nullBeta ? 0.0f : parseFloat(betaRealStr, 0.5f);
    betaImag = nullBeta ? 0.0f : parseFloat(betaImagStr, 0.0f);

    cFillSpec = ReadMap(map, "c_fill", "RANDOM_NORM_1E3_1E3");
    cNormal = (cFillSpec == "NORMAL_NORM");
    cFill = parseFill(cFillSpec);
    const std::string ldcStr = ReadMap(map, "ldc", "");
    ldc = ldcStr.empty() ? m : static_cast<int64_t>(parseInt64(ldcStr));
}

static_assert(
    sizeof(aclblasComplex) == 2 * sizeof(float),
    "CBLAS complex64 oracle requires aclblasComplex to be two float components");

inline CBLAS_SIDE ChemmToCblasSide(aclblasSideMode_t side)
{
    return side == ACLBLAS_SIDE_LEFT ? CblasLeft : CblasRight;
}

inline aclblasStatus_t ChemmValidateCpuArgs(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if ((side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) || (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr ||
        lda < std::max<int64_t>(1, aDim) || ldb < std::max<int64_t>(1, m) || ldc < std::max<int64_t>(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasChemm_cpu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    aclblasStatus_t status = ChemmValidateCpuArgs(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return status;
    }

    constexpr int64_t kCblasIntMax = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (m > kCblasIntMax || n > kCblasIntMax || lda > kCblasIntMax || ldb > kCblasIntMax || ldc > kCblasIntMax) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // OpenBLAS partitions K differently when its process-wide thread count
    // changes, so the last-bit result is otherwise machine-load dependent.
    // Netlib CBLAS is single-threaded and does not expose this extension.
#ifdef OPENBLAS_VERSION
    openblas_set_num_threads(1);
#endif
    cblas_chemm(
        CblasColMajor, ChemmToCblasSide(side), ToCblasUplo(uplo), static_cast<int>(m), static_cast<int>(n), alpha, A,
        static_cast<int>(lda), B, static_cast<int>(ldb), beta, C, static_cast<int>(ldc));

    return ACLBLAS_STATUS_SUCCESS;
}

inline std::mutex& ChemmTimeoutMutex()
{
    static std::mutex mutex;
    return mutex;
}

inline std::unordered_set<aclblasHandle_t>& ChemmTimedOutHandles()
{
    static std::unordered_set<aclblasHandle_t> timedOutHandles;
    return timedOutHandles;
}

inline bool ChemmIsHandleTimedOut(aclblasHandle_t handle)
{
    std::lock_guard<std::mutex> lock(ChemmTimeoutMutex());
    return ChemmTimedOutHandles().count(handle) != 0U;
}

inline void ChemmMarkHandleTimedOut(aclblasHandle_t handle)
{
    std::lock_guard<std::mutex> lock(ChemmTimeoutMutex());
    (void)ChemmTimedOutHandles().insert(handle);
}

struct ChemmDeviceBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
};

inline void ChemmReleaseDeviceBuffers(ChemmDeviceBuffers& buffers)
{
    (void)aclrtFree(buffers.a);
    (void)aclrtFree(buffers.b);
    (void)aclrtFree(buffers.c);
    buffers = {};
}

inline aclblasStatus_t ChemmAllocAndCopy(void** device, size_t bytes, const void* host)
{
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        (void)aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ChemmPrepareDeviceBuffers(
    ChemmDeviceBuffers& buffers, const aclblasComplex* a, size_t aBytes, const aclblasComplex* b, size_t bBytes,
    const aclblasComplex* c, size_t cBytes)
{
    aclblasStatus_t status = ChemmAllocAndCopy(&buffers.a, aBytes, a);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = ChemmAllocAndCopy(&buffers.b, bBytes, b);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = ChemmAllocAndCopy(&buffers.c, cBytes, c);
    }
    if (status != ACLBLAS_STATUS_SUCCESS) {
        ChemmReleaseDeviceBuffers(buffers);
    }
    return status;
}

inline aclblasStatus_t ChemmRestoreTimeout(uint32_t savedTimeout, bool& changed)
{
    if (!changed) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtSetOpExecuteTimeOutWithMs(savedTimeout) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    changed = false;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ChemmRunDeviceOperation(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc, size_t aBytes, size_t bBytes, size_t cBytes)
{
    ChemmDeviceBuffers buffers;
    aclblasStatus_t status = ChemmPrepareDeviceBuffers(buffers, A, aBytes, B, bBytes, C, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Keep the handle in Library workspace mode.  aclblasChemm derives the
    // exact workspace from its current tiling and can grow it as required.
    uint32_t savedTimeout = 0U;
    bool timeoutChanged = false;
    if (aclrtGetOpExecuteTimeout(&savedTimeout) != ACL_SUCCESS ||
        aclrtSetOpExecuteTimeOutWithMs(5000U) != ACL_SUCCESS) {
        ChemmReleaseDeviceBuffers(buffers);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    timeoutChanged = true;
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS) {
        (void)ChemmRestoreTimeout(savedTimeout, timeoutChanged);
        ChemmReleaseDeviceBuffers(buffers);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclblasStatus_t ret = aclblasChemm(
        handle, side, uplo, m, n, alpha, static_cast<const aclblasComplex*>(buffers.a), lda,
        static_cast<const aclblasComplex*>(buffers.b), ldb, beta, static_cast<aclblasComplex*>(buffers.c), ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeStreamWithTimeout(stream, 10000) == ACL_SUCCESS) {
        if (aclrtMemcpy(C, cBytes, buffers.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else if (ret == ACLBLAS_STATUS_SUCCESS) {
        ChemmMarkHandleTimedOut(handle);
        ret = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    aclblasStatus_t restore = ChemmRestoreTimeout(savedTimeout, timeoutChanged);
    ChemmReleaseDeviceBuffers(buffers);
    if (restore != ACLBLAS_STATUS_SUCCESS) {
        return restore;
    }
    return ret;
}

aclblasStatus_t aclblasChemm_npu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    if (handle != nullptr && ChemmIsHandleTimedOut(handle)) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (handle == nullptr || m <= 0 || n <= 0 || A == nullptr || B == nullptr || C == nullptr) {
        return aclblasChemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t aElemCount = static_cast<size_t>(aDim) * static_cast<size_t>(lda);
    const size_t bElemCount = static_cast<size_t>(n) * static_cast<size_t>(ldb);
    const size_t cElemCount = static_cast<size_t>(n) * static_cast<size_t>(ldc);
    const size_t aBytes = aElemCount * 2U * sizeof(float);
    const size_t bBytes = bElemCount * 2U * sizeof(float);
    const size_t cBytes = cElemCount * 2U * sizeof(float);

    return ChemmRunDeviceOperation(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, aBytes, bBytes, cBytes);
}

template <typename Launch>
inline aclblasStatus_t ChemmWarmupLaunches(Launch& launch, aclrtStream stream, int warmup)
{
    for (int i = 0; i < warmup; ++i) {
        const aclblasStatus_t status = launch();
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return aclrtSynchronizeStream(stream) == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
}

inline void ChemmDestroyEvents(aclrtEvent start, aclrtEvent end)
{
    if (start != nullptr) {
        (void)aclrtDestroyEvent(start);
    }
    if (end != nullptr) {
        (void)aclrtDestroyEvent(end);
    }
}

template <typename Launch>
inline aclblasStatus_t ChemmTimedLaunches(Launch& launch, aclrtStream stream, int iterations, float& elapsedMs)
{
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    if (aclrtCreateEvent(&start) != ACL_SUCCESS || aclrtCreateEvent(&end) != ACL_SUCCESS) {
        ChemmDestroyEvents(start, end);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclblasStatus_t status =
        aclrtRecordEvent(start, stream) == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
    for (int i = 0; i < iterations && status == ACLBLAS_STATUS_SUCCESS; ++i) {
        status = launch();
    }
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtRecordEvent(end, stream) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeEvent(end) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtEventElapsedTime(&elapsedMs, start, end) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ChemmDestroyEvents(start, end);
    return status;
}

aclblasStatus_t aclblasChemm_perf(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc, int warmup, int iterations, float& averageUs)
{
    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(n) * static_cast<size_t>(ldb) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(n) * static_cast<size_t>(ldc) * sizeof(aclblasComplex);
    ChemmDeviceBuffers buffers;
    aclblasStatus_t status = ChemmPrepareDeviceBuffers(buffers, A, aBytes, B, bBytes, C, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Performance launches must use the same tiling-owned Library workspace.
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS) {
        ChemmReleaseDeviceBuffers(buffers);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    auto launch = [&]() {
        return aclblasChemm(
            handle, side, uplo, m, n, alpha, static_cast<const aclblasComplex*>(buffers.a), lda,
            static_cast<const aclblasComplex*>(buffers.b), ldb, beta, static_cast<aclblasComplex*>(buffers.c), ldc);
    };
    float elapsedMs = 0.0f;
    status = ChemmWarmupLaunches(launch, stream, warmup);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = ChemmTimedLaunches(launch, stream, iterations, elapsedMs);
    }
    ChemmReleaseDeviceBuffers(buffers);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        averageUs = elapsedMs * 1000.0f / static_cast<float>(iterations);
    }
    return status;
}

class ChemmArch22Test : public BlasTest<ChemmParam> {};

constexpr int64_t kMaxSafeTestElems = 64LL * 1024 * 1024;

inline int64_t ChemmSafeElemCount(int64_t rows, int64_t cols)
{
    if (cols == 0 || rows <= 0) {
        return 0;
    }
    if (rows > kMaxSafeTestElems / cols) {
        return 0;
    }
    return rows * cols;
}

inline std::vector<float> ChemmMakeHostArray(int64_t count, const BlasFillMode& fill, bool normal, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count <= 0) {
        return {};
    }
    if (!normal) {
        return makeBlasArray(count, fill, seed);
    }

    // NORMAL_NORM is Chemm-local: each matrix gets a reproducible
    // mu in [-5, 5] and sigma in [0.1, 2], then both complex components
    // are sampled from that normal distribution.
    std::mt19937 generator(seed ? seed : 42);
    std::uniform_real_distribution<float> meanDistribution(-5.0f, 5.0f);
    std::uniform_real_distribution<float> stddevDistribution(0.1f, 2.0f);
    std::normal_distribution<float> valueDistribution(meanDistribution(generator), stddevDistribution(generator));
    std::vector<float> values(static_cast<size_t>(count));
    for (float& value : values) {
        value = valueDistribution(generator);
    }
    return values;
}

inline size_t ChemmComplexOffset(int64_t row, int64_t col, int64_t ld)
{
    return 2 * static_cast<size_t>(row + col * ld);
}

inline void ChemmMakeHermitian(std::vector<float>& values, const ChemmParam& p, int64_t aDim)
{
    if (p.m <= 0 || p.n <= 0) {
        return;
    }
    for (int64_t d = 0; d < aDim; ++d) {
        size_t diagonal = ChemmComplexOffset(d, d, p.lda);
        if (diagonal + 1 < values.size()) {
            values[diagonal + 1] = 0.0f;
        }
    }
    for (int64_t col = 0; col < aDim; ++col) {
        for (int64_t row = 0; row < aDim; ++row) {
            bool reflected = (p.uplo == ACLBLAS_UPPER) ? (row > col) : (row < col);
            if (!reflected) {
                continue;
            }
            size_t destination = ChemmComplexOffset(row, col, p.lda);
            size_t source = ChemmComplexOffset(col, row, p.lda);
            if (destination + 1 < values.size() && source + 1 < values.size()) {
                values[destination] = values[source];
                values[destination + 1] = -values[source + 1];
            }
        }
    }
}

inline void ChemmPrepareHostData(
    const ChemmParam& p, std::vector<float>& a, std::vector<float>& b, std::vector<float>& c,
    std::vector<float>& result)
{
    int64_t aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    int64_t aCount = ChemmSafeElemCount(aDim, p.lda);
    int64_t bCount = ChemmSafeElemCount(p.n, p.ldb);
    int64_t cCount = ChemmSafeElemCount(p.n, p.ldc);
    if (aCount > 0 && p.aFill.method != BlasFillMode::M_NULLPTR) {
        a = ChemmMakeHostArray(2 * aCount, p.aFill, p.aNormal, p.randomSeed);
        ChemmMakeHermitian(a, p, aDim);
    }
    if (bCount > 0 && p.bFill.method != BlasFillMode::M_NULLPTR) {
        b = ChemmMakeHostArray(2 * bCount, p.bFill, p.bNormal, p.randomSeed + 1);
    }
    if (cCount > 0 && p.cFill.method != BlasFillMode::M_NULLPTR) {
        c = ChemmMakeHostArray(2 * cCount, p.cFill, p.cNormal, p.randomSeed + 2);
        result = c;
    }
}

inline void ChemmBuildPointers(
    const ChemmParam& p, std::vector<float>& a, std::vector<float>& b, std::vector<float>& result,
    aclblasComplex& alpha, aclblasComplex& beta, const aclblasComplex*& alphaPtr, const aclblasComplex*& betaPtr,
    const aclblasComplex*& aPtr, const aclblasComplex*& bPtr, aclblasComplex*& cPtr)
{
    alpha = {p.alphaReal, p.alphaImag};
    beta = {p.betaReal, p.betaImag};
    alphaPtr = p.nullAlpha ? nullptr : &alpha;
    betaPtr = p.nullBeta ? nullptr : &beta;
    aPtr = a.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(a.data());
    bPtr = b.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(b.data());
    cPtr = result.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(result.data());
}

inline bool ChemmCheckExpectedStatus(const ChemmParam& p, aclblasStatus_t ret)
{
    if (p.expectResult == ACLBLAS_STATUS_SUCCESS) {
        return true;
    }
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    return false;
}

inline bool ChemmHasExtremeFill(const ChemmParam& p)
{
    return p.aFill.pattern == BlasFillMode::P_EXTREME || p.bFill.pattern == BlasFillMode::P_EXTREME;
}

inline bool ChemmPerfTimingRequested()
{
    const char* mode = std::getenv("CBLAS3_PERF_MODE");
    return mode != nullptr && mode[0] == '1';
}

inline void ChemmBuildGolden(
    const ChemmParam& p, const std::vector<float>& cHost, const aclblasComplex* alpha, const aclblasComplex* beta,
    const aclblasComplex* a, const aclblasComplex* b, std::vector<float>& golden, aclblasHandle_t handle)
{
    golden = cHost;
    aclblasComplex* output = golden.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(golden.data());
    ASSERT_EQ(
        aclblasChemm_cpu(handle, p.side, p.uplo, p.m, p.n, alpha, a, p.lda, b, p.ldb, beta, output, p.ldc),
        ACLBLAS_STATUS_SUCCESS);
}

inline bool ChemmSameNonFinite(float output, float golden)
{
    if (std::isnan(golden)) {
        return std::isnan(output);
    }
    return std::isinf(golden) && std::isinf(output) && std::signbit(golden) == std::signbit(output);
}

inline void ChemmVerifyComponent(
    const std::vector<float>& result, const std::vector<float>& golden, const std::string& caseName)
{
    ASSERT_EQ(result.size(), golden.size());
    std::vector<float> finiteResult;
    std::vector<float> finiteGolden;
    finiteResult.reserve(result.size());
    finiteGolden.reserve(golden.size());
    for (size_t i = 0; i < result.size(); ++i) {
        if (std::isfinite(result[i]) && std::isfinite(golden[i])) {
            finiteResult.push_back(result[i]);
            finiteGolden.push_back(golden[i]);
        } else {
            EXPECT_TRUE(ChemmSameNonFinite(result[i], golden[i])) << caseName << " non-finite component at " << i;
        }
    }
    if (finiteResult.empty()) {
        return;
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, finiteGolden.data(), finiteGolden.size());
    EXPECT_TRUE(
        Verifier::verifyVector(finiteResult.data(), finiteGolden.data(), finiteGolden.size(), 1, config, caseName));
}

inline void ChemmVerifyResult(const ChemmParam& p, const std::vector<float>& result, const std::vector<float>& golden)
{
    std::vector<float> resultReal;
    std::vector<float> resultImag;
    std::vector<float> goldenReal;
    std::vector<float> goldenImag;
    size_t logicalCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.n);
    resultReal.reserve(logicalCount);
    resultImag.reserve(logicalCount);
    goldenReal.reserve(logicalCount);
    goldenImag.reserve(logicalCount);
    for (int64_t col = 0; col < p.n; ++col) {
        for (int64_t row = 0; row < p.m; ++row) {
            size_t offset = ChemmComplexOffset(row, col, p.ldc);
            ASSERT_LT(offset + 1, result.size());
            ASSERT_LT(offset + 1, golden.size());
            resultReal.push_back(result[offset]);
            resultImag.push_back(result[offset + 1]);
            goldenReal.push_back(golden[offset]);
            goldenImag.push_back(golden[offset + 1]);
        }
    }
    ChemmVerifyComponent(resultReal, goldenReal, p.caseName + "_real");
    ChemmVerifyComponent(resultImag, goldenImag, p.caseName + "_imag");
}

TEST(ChemmValidationTest, NullHandle)
{
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.5f, 0.0f};
    // Minimal dummy buffers to satisfy non-null pointer check; handle=nullptr
    // is caught before any pointer deref.
    std::vector<float> aBuf(32, 0.0f); // 4*4*2 floats
    std::vector<float> bBuf(32, 0.0f);
    std::vector<float> cBuf(32, 0.0f);
    aclblasStatus_t ret = aclblasChemm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4, &alpha, reinterpret_cast<aclblasComplex*>(aBuf.data()), 4,
        reinterpret_cast<aclblasComplex*>(bBuf.data()), 4, &beta, reinterpret_cast<aclblasComplex*>(cBuf.data()), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Chemm, ChemmArch22Test, ::testing::ValuesIn(GetCasesFromCsv<ChemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ChemmParam>);

TEST_P(ChemmArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    const bool perfCase = p.caseName.rfind("TC_PF_", 0) == 0;
    if (perfCase && !ChemmPerfTimingRequested()) {
        GTEST_SKIP() << "Set CBLAS3_PERF_MODE=1 to run performance sampling";
    }
    std::vector<float> aHostFlt;
    std::vector<float> bHostFlt;
    std::vector<float> cHostFlt;
    std::vector<float> cResultFlt;

    ChemmPrepareHostData(p, aHostFlt, bHostFlt, cHostFlt, cResultFlt);
    aclblasComplex alphaVal;
    aclblasComplex betaVal;
    const aclblasComplex* alphaPtr;
    const aclblasComplex* betaPtr;
    const aclblasComplex* aPtr;
    const aclblasComplex* bPtr;
    aclblasComplex* cPtr;
    ChemmBuildPointers(p, aHostFlt, bHostFlt, cResultFlt, alphaVal, betaVal, alphaPtr, betaPtr, aPtr, bPtr, cPtr);
    if (perfCase) {
        constexpr int kWarmup = 5;
        constexpr int kIterations = 60;
        float averageUs = 0.0f;
        aclblasStatus_t perfStatus = aclblasChemm_perf(
            ChemmArch22Test::handle_, p.side, p.uplo, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr,
            p.ldc, kWarmup, kIterations, averageUs);
        ASSERT_EQ(perfStatus, ACLBLAS_STATUS_SUCCESS);
        std::cout << "[PERF] case=" << p.caseName << " average_us=" << averageUs << " warmup=" << kWarmup
                  << " iterations=" << kIterations << std::endl;
        return;
    }
    aclblasStatus_t ret = aclblasChemm_npu(
        ChemmArch22Test::handle_, p.side, p.uplo, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);

    if (!ChemmCheckExpectedStatus(p, ret)) {
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    // EXTREME intentionally mixes FLT_MAX with signed/zero values. Different
    // legal FP32 reduction orders can produce different NaN/Inf classes after
    // overflow, so this case validates successful execution rather than a
    // finite tolerance that is undefined for non-finite results.
    if (ChemmHasExtremeFill(p)) {
        return;
    }
    std::vector<float> cGoldenFlt;
    ChemmBuildGolden(p, cHostFlt, alphaPtr, betaPtr, aPtr, bPtr, cGoldenFlt, ChemmArch22Test::handle_);
    ChemmVerifyResult(p, cResultFlt, cGoldenFlt);
}
