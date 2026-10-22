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

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "chemm_param.h"

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

/*!
 * \brief Netlib/OpenBLAS CBLAS reference implementation for aclblasChemm.
 *
 * Parameters are checked in the public int64_t ABI before conversion to the
 * CBLAS int ABI.  This prevents an out-of-range test parameter from silently
 * truncating before it reaches the independently implemented CBLAS oracle.
 */
inline aclblasStatus_t aclblasChemm_cpu(
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

// ── golden cache: avoid recomputing CPU golden every test run ──

namespace chemm_golden_cache {

constexpr uint32_t CHEMM_GOLDEN_VERSION = 4;
constexpr uint32_t CHEMM_GOLDEN_MAGIC = 0x4D474843;

inline void ensureDir(const std::string& path)
{
    std::string dir = path;
    if (!dir.empty() && dir.back() == '/') {
        dir.pop_back();
    }
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
    }
}

inline uint64_t computeParamHash(const ChemmParam& p)
{
    struct HashAccum {
        uint64_t h = 14695981039346656037ULL;
        void feed(const void* data, size_t len)
        {
            const auto* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < len; ++i) {
                h ^= static_cast<uint64_t>(bytes[i]);
                h *= 1099511628211ULL;
            }
        }
    } acc;

    acc.feed(&p.side, sizeof(p.side));
    acc.feed(&p.uplo, sizeof(p.uplo));
    acc.feed(&p.m, sizeof(p.m));
    acc.feed(&p.n, sizeof(p.n));
    acc.feed(&p.alphaReal, sizeof(p.alphaReal));
    acc.feed(&p.alphaImag, sizeof(p.alphaImag));
    acc.feed(&p.betaReal, sizeof(p.betaReal));
    acc.feed(&p.betaImag, sizeof(p.betaImag));
    acc.feed(&p.lda, sizeof(p.lda));
    acc.feed(&p.ldb, sizeof(p.ldb));
    acc.feed(&p.ldc, sizeof(p.ldc));
    acc.feed(&p.randomSeed, sizeof(p.randomSeed));
    acc.feed(&p.aNormal, sizeof(p.aNormal));
    acc.feed(&p.bNormal, sizeof(p.bNormal));
    acc.feed(&p.cNormal, sizeof(p.cNormal));
    acc.feed(&p.aFill.method, sizeof(p.aFill.method));
    acc.feed(&p.aFill.pattern, sizeof(p.aFill.pattern));
    acc.feed(&p.aFill.val1, sizeof(p.aFill.val1));
    acc.feed(&p.aFill.val2, sizeof(p.aFill.val2));
    acc.feed(&p.aFill.bandedKl, sizeof(p.aFill.bandedKl));
    acc.feed(&p.aFill.bandedKu, sizeof(p.aFill.bandedKu));
    acc.feed(&p.bFill.method, sizeof(p.bFill.method));
    acc.feed(&p.bFill.pattern, sizeof(p.bFill.pattern));
    acc.feed(&p.bFill.val1, sizeof(p.bFill.val1));
    acc.feed(&p.bFill.val2, sizeof(p.bFill.val2));
    acc.feed(&p.bFill.bandedKl, sizeof(p.bFill.bandedKl));
    acc.feed(&p.bFill.bandedKu, sizeof(p.bFill.bandedKu));
    acc.feed(&p.cFill.method, sizeof(p.cFill.method));
    acc.feed(&p.cFill.pattern, sizeof(p.cFill.pattern));
    acc.feed(&p.cFill.val1, sizeof(p.cFill.val1));
    acc.feed(&p.cFill.val2, sizeof(p.cFill.val2));
    acc.feed(&p.cFill.bandedKl, sizeof(p.cFill.bandedKl));
    acc.feed(&p.cFill.bandedKu, sizeof(p.cFill.bandedKu));

    return acc.h;
}

inline bool loadGolden(const std::string& filepath, std::vector<float>& data, uint64_t paramHash)
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t storedHash = 0;
    uint64_t storedCount = 0;

    if (!ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic)))
        return false;
    if (!ifs.read(reinterpret_cast<char*>(&version), sizeof(version)))
        return false;
    if (!ifs.read(reinterpret_cast<char*>(&storedHash), sizeof(storedHash)))
        return false;
    if (!ifs.read(reinterpret_cast<char*>(&storedCount), sizeof(storedCount)))
        return false;

    if (magic != CHEMM_GOLDEN_MAGIC || version != CHEMM_GOLDEN_VERSION || storedHash != paramHash) {
        return false;
    }

    data.resize(static_cast<size_t>(storedCount));
    if (!ifs.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(storedCount * sizeof(float)))) {
        data.clear();
        return false;
    }

    return ifs.good();
}

inline void saveGolden(const std::string& filepath, const std::vector<float>& data, uint64_t paramHash)
{
    size_t slash = filepath.rfind('/');
    if (slash != std::string::npos) {
        ensureDir(filepath.substr(0, slash));
    }

    std::ofstream ofs(filepath, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        return;
    }

    uint32_t magic = CHEMM_GOLDEN_MAGIC;
    uint32_t version = CHEMM_GOLDEN_VERSION;
    uint64_t count = static_cast<uint64_t>(data.size());

    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&paramHash), sizeof(paramHash));
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

} // namespace chemm_golden_cache

inline std::string makeGoldenDir(const std::string& filePath)
{
    size_t slash = filePath.rfind('/');
    if (slash == std::string::npos) {
        return "golden/";
    }
    return filePath.substr(0, slash + 1) + "golden/";
}
