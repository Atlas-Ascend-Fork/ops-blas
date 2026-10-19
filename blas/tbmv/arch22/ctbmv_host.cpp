/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ctbmv_host.cpp
 * \brief Single-precision complex triangular band matrix-vector multiply for Atlas A2/A3.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "ctbmv_kernel.h"
#include "ctbmv_tiling_data.h"

namespace {

constexpr uint32_t CTBMV_COMPACT_FAST_MAX_K = 32U;
constexpr uint32_t CTBMV_PANEL_TILE = 128U;
constexpr uint32_t CTBMV_PANEL_GATHER_PHASES = 8U;
constexpr uint32_t CTBMV_PANEL_WINDOW_WORDS = 8U * 160U;
constexpr uint32_t CTBMV_PANEL_BASE_WORDS = CTBMV_PANEL_TILE * (2U + CTBMV_PANEL_GATHER_PHASES);
constexpr uint32_t CTBMV_PANEL_TABLE_WORDS = CTBMV_PANEL_BASE_WORDS + 2U * CTBMV_PANEL_WINDOW_WORDS;
constexpr uint32_t CTBMV_WIDE_ATOMIC_MAX_N = 4096U;
constexpr uint32_t CTBMV_WIDE_ATOMIC_MAX_ROWS = 1152U;

const std::vector<uint32_t>& GetCtbmvWideGroupOffsets(uint32_t k)
{
    static std::map<uint32_t, std::vector<uint32_t>> tables;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = tables.find(k);
    if (found != tables.end()) {
        return found->second;
    }
    const uint32_t width = (k + 64U) & ~63U;
    const uint32_t rows = (k + 22U) & ~7U;
    std::vector<uint32_t> table(8U * rows * 8U);
    for (uint32_t phase = 0U; phase < 8U; ++phase) {
        for (uint32_t row = 0U; row < rows; ++row) {
            for (uint32_t column = 0U; column < 8U; ++column) {
                const int inner = static_cast<int>(row) - static_cast<int>(phase + column);
                const uint32_t source = inner >= 0 && inner <= static_cast<int>(k) ?
                                            column * width + static_cast<uint32_t>(inner) :
                                            8U * width;
                table[(phase * rows + row) * 8U + column] = source * sizeof(float);
            }
        }
    }
    return tables.emplace(k, std::move(table)).first->second;
}

bool IsRightBand(aclblasOperation_t trans, aclblasFillMode_t uplo)
{
    return (trans == ACLBLAS_OP_N && uplo == ACLBLAS_UPPER) || (trans != ACLBLAS_OP_N && uplo == ACLBLAS_LOWER);
}

const std::array<uint32_t, 544U>& GetCtbmvTinyOffsets(
    int n, int k, int lda, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    using Key = std::array<int, 6>;
    static std::map<Key, std::array<uint32_t, 544U>> tables;
    static std::mutex tableMutex;
    const Key key{n, k, lda, static_cast<int>(uplo), static_cast<int>(trans), static_cast<int>(diag)};
    std::lock_guard<std::mutex> lock(tableMutex);
    const auto found = tables.find(key);
    if (found != tables.end()) {
        return found->second;
    }
    std::array<uint32_t, 544U> table{};
    for (int row = 0; row < 16; ++row) {
        for (int inner = 0; inner < 8; ++inner) {
            const bool right = IsRightBand(trans, uplo);
            const int col = right ? row + inner : row - inner;
            uint32_t ai = static_cast<uint32_t>(n * lda * 2);
            uint32_t xi = static_cast<uint32_t>(n * 2);
            if (row < n && inner <= k && col >= 0 && col < n) {
                xi = static_cast<uint32_t>(col * 2);
                if (diag == ACLBLAS_UNIT && inner == 0) {
                    ai += 2U;
                } else {
                    const int mr = trans == ACLBLAS_OP_N ? row : col;
                    const int mc = trans == ACLBLAS_OP_N ? col : row;
                    const int band = uplo == ACLBLAS_UPPER ? k + mr - mc : mr - mc;
                    ai = static_cast<uint32_t>((mc * lda + band) * 2);
                }
            }
            const uint32_t pos = row * 8 + inner;
            table[pos] = ai * 4U;
            table[128U + pos] = (ai + 1U) * 4U;
            table[256U + pos] = xi * 4U;
            table[384U + pos] = (xi + 1U) * 4U;
        }
        table[512U + row * 2U] = row * 4U;
        table[513U + row * 2U] = (16U + row) * 4U;
    }
    return tables.emplace(key, table).first->second;
}

const std::vector<uint32_t>& GetCtbmvSmallPanelOffsets(
    int n, int k, int lda, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    using Key = std::array<int, 6>;
    static std::map<Key, std::vector<uint32_t>> tables;
    static std::mutex mutex;
    const Key key{n, k, lda, static_cast<int>(uplo), static_cast<int>(trans), static_cast<int>(diag)};
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = tables.find(key);
    if (found != tables.end()) {
        return found->second;
    }
    const uint32_t width = k < 8 ? 8U : (k < 16 ? 16U : (k < 32 ? 32U : 64U));
    const uint32_t paddedN = (static_cast<uint32_t>(n) + 7U) & ~7U;
    const uint32_t entries = paddedN * width;
    std::vector<uint32_t> table(entries * 4U + 16U);
    for (uint32_t row = 0U; row < paddedN; ++row) {
        for (uint32_t inner = 0U; inner < width; ++inner) {
            const bool right = IsRightBand(trans, uplo);
            const int col = right ? static_cast<int>(row + inner) : static_cast<int>(row) - static_cast<int>(inner);
            uint32_t ai = static_cast<uint32_t>(n * lda * 2);
            uint32_t xi = static_cast<uint32_t>(n * 2);
            if (row < static_cast<uint32_t>(n) && inner <= static_cast<uint32_t>(k) && col >= 0 && col < n) {
                xi = static_cast<uint32_t>(col * 2);
                if (diag == ACLBLAS_UNIT && inner == 0U) {
                    ai += 2U;
                } else {
                    const int mr = trans == ACLBLAS_OP_N ? static_cast<int>(row) : col;
                    const int mc = trans == ACLBLAS_OP_N ? col : static_cast<int>(row);
                    const int band = uplo == ACLBLAS_UPPER ? k + mr - mc : mr - mc;
                    ai = static_cast<uint32_t>((mc * lda + band) * 2);
                }
            }
            const uint32_t i = row * width + inner;
            table[i] = ai * 4U;
            table[entries + i] = (ai + 1U) * 4U;
            table[entries * 2U + i] = xi * 4U;
            table[entries * 3U + i] = (xi + 1U) * 4U;
        }
    }
    for (uint32_t i = 0; i < 8U; ++i) {
        table[entries * 4U + 2U * i] = i * 4U;
        table[entries * 4U + 2U * i + 1U] = (8U + i) * 4U;
    }
    return tables.emplace(key, std::move(table)).first->second;
}

const std::array<uint32_t, 8320U>& GetCtbmvTransposeNarrowOffsets(uint32_t k)
{
    static const std::array<std::array<uint32_t, 8320U>, 16U> tables = [] {
        std::array<std::array<uint32_t, 8320U>, 16U> values{};
        for (uint32_t t = 0U; t < 16U; ++t) {
            const uint32_t width = (t + 1U) * 64U;
            for (uint32_t i = 0U; i < 4096U; ++i) {
                const uint32_t offset = (i / width + i % width) * 8U;
                values[t][i] = offset;
                values[t][4096U + i] = offset + 4U;
            }
            for (uint32_t i = 0U; i < 64U; ++i) {
                values[t][8192U + i * 2U] = i * 4U;
                values[t][8192U + i * 2U + 1U] = (64U + i) * 4U;
            }
        }
        return values;
    }();
    return tables[k / 64U];
}

const std::array<uint32_t, 6272U>& GetCtbmvTransposeWindowOffsets(uint32_t k)
{
    static const std::array<std::array<uint32_t, 6272U>, 16U> tables = [] {
        std::array<std::array<uint32_t, 6272U>, 16U> values{};
        for (uint32_t t = 0U; t < 16U; ++t) {
            const uint32_t width = (t + 1U) * 64U;
            for (uint32_t i = 0U; i < 6144U; ++i) {
                const uint32_t offset = (i / width + i % width) * 4U;
                values[t][i] = offset;
            }
            for (uint32_t i = 0U; i < 64U; ++i) {
                values[t][6144U + i * 2U] = i * 4U;
                values[t][6144U + i * 2U + 1U] = (64U + i) * 4U;
            }
        }
        return values;
    }();
    return tables[k / 64U];
}

const std::array<uint32_t, CTBMV_WIDE_ATOMIC_MAX_ROWS * 2U>& GetCtbmvWideAtomicInterleaveOffsets()
{
    static const std::array<uint32_t, CTBMV_WIDE_ATOMIC_MAX_ROWS* 2U> offsets = [] {
        std::array<uint32_t, CTBMV_WIDE_ATOMIC_MAX_ROWS * 2U> values{};
        for (uint32_t i = 0U; i < CTBMV_WIDE_ATOMIC_MAX_ROWS; ++i) {
            values[i * 2U] = i * sizeof(float);
            values[i * 2U + 1U] = (CTBMV_WIDE_ATOMIC_MAX_N + i) * sizeof(float);
        }
        return values;
    }();
    return offsets;
}

const std::array<uint32_t, CTBMV_PANEL_TABLE_WORDS>& GetCtbmvPanelOffsets(uint32_t lda)
{
    static const std::array<std::array<uint32_t, CTBMV_PANEL_TABLE_WORDS>, CTBMV_COMPACT_FAST_MAX_K + 1U> tables = [] {
        std::array<std::array<uint32_t, CTBMV_PANEL_TABLE_WORDS>, CTBMV_COMPACT_FAST_MAX_K + 1U> values{};
        for (uint32_t tableLda = 1U; tableLda <= CTBMV_COMPACT_FAST_MAX_K + 1U; ++tableLda) {
            auto& table = values[tableLda - 1U];
            for (uint32_t i = 0U; i < CTBMV_PANEL_WINDOW_WORDS; ++i) {
                const uint32_t position = std::min<uint32_t>(159U, i / 160U + i % 160U);
                table[CTBMV_PANEL_BASE_WORDS + i] = position * 8U;
                table[CTBMV_PANEL_BASE_WORDS + CTBMV_PANEL_WINDOW_WORDS + i] = position * 8U + 4U;
            }
            for (uint32_t i = 0U; i < CTBMV_PANEL_TILE; ++i) {
                table[i * 2U] = i * sizeof(float);
                table[i * 2U + 1U] = (CTBMV_PANEL_TILE + i) * sizeof(float);
                for (uint32_t phase = 0U; phase < CTBMV_PANEL_GATHER_PHASES; ++phase) {
                    table[CTBMV_PANEL_TILE * (2U + phase) + i] =
                        i * tableLda * 2U * sizeof(float) + phase * sizeof(float);
                }
            }
        }
        return values;
    }();
    return tables[lda - 1U];
}

aclblasStatus_t ValidateCtbmvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, const aclblasComplex* a,
    int lda, const aclblasComplex* x, int incx)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: invalid diag=%d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0 || k < 0 || lda <= k || incx == 0) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: invalid shape: n=%d k=%d lda=%d incx=%d", n, k, lda, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && (a == nullptr || x == nullptr)) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: A and x must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

CtbmvTilingData MakeCtbmvTiling(
    int n, int k, int lda, int incx, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    uint32_t coreNum, CtbmvPath path)
{
    CtbmvTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.useCoreNum = static_cast<uint16_t>(std::min<uint32_t>(static_cast<uint32_t>(n), coreNum));
    tiling.rowsPerCore = CeilDiv<uint32_t>(tiling.n, tiling.useCoreNum);
    tiling.uplo = static_cast<uint8_t>(uplo);
    tiling.trans = static_cast<uint8_t>(trans);
    tiling.diag = static_cast<uint8_t>(diag);
    tiling.path = path;
    tiling.incx = static_cast<int64_t>(incx);
    return tiling;
}

bool UseCompactPanel(
    int n, int k, int lda, int incx, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t availableCoreNum)
{
    const uint32_t panelRows = n > 2048 ? 128U : 64U;
    const uint32_t panelCores = CeilDiv<uint32_t>(static_cast<uint32_t>(n), panelRows);
    const bool lowerBandGroup = trans == ACLBLAS_OP_N && uplo == ACLBLAS_LOWER && n >= 512 && n <= 4096 && k == 32;
    return incx == 1 && lda == k + 1 && n <= 4096 && k <= 32 && !lowerBandGroup &&
           !(trans != ACLBLAS_OP_N && uplo == ACLBLAS_UPPER) && n >= 128 && availableCoreNum >= panelCores;
}

CtbmvPath SelectLargeCtbmvPath(
    int n, int k, int incx, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t availableCoreNum)
{
    if (incx != 1) {
        return CtbmvPath::Generic;
    }
    if (trans == ACLBLAS_OP_N && n >= 65 && n <= 4096 && k >= 32 && k < 1024) {
        return uplo == ACLBLAS_LOWER && n >= 512 && k <= 256 && availableCoreNum >= 32U ? CtbmvPath::WideGroup :
                                                                                          CtbmvPath::WideColumn;
    }
    if (trans != ACLBLAS_OP_N && k < 1024) {
        return n >= 17 && k >= 1 && k < n ? CtbmvPath::Transpose : CtbmvPath::Row;
    }
    if (trans == ACLBLAS_OP_N && n >= 65 && k >= 33 && k < 1024) {
        return CtbmvPath::Row;
    }
    return CtbmvPath::Generic;
}

CtbmvPath SelectCtbmvPath(
    int n, int k, int lda, int incx, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t availableCoreNum)
{
    const uint32_t absIncx = static_cast<uint32_t>(incx < 0 ? -static_cast<int64_t>(incx) : incx);
    const bool contiguous = incx == 1;
    const bool packedBand = lda == k + 1;
    if (n >= 8 && n <= 16 && k <= 4 && packedBand && contiguous) {
        return CtbmvPath::TinyVector;
    }
    if (n <= 16 && absIncx <= 3U) {
        return CtbmvPath::SmallScalar;
    }
    if (n >= 17 && n <= 128 && k < 64 && packedBand && contiguous) {
        return CtbmvPath::SmallPanel;
    }
    if (UseCompactPanel(n, k, lda, incx, uplo, trans, availableCoreNum)) {
        // Includes the three task-book targets; no exact n/k specialization.
        return CtbmvPath::CompactPanel;
    }
    return SelectLargeCtbmvPath(n, k, incx, uplo, trans, availableCoreNum);
}

bool IsWideCtbmvPath(CtbmvPath path) { return path == CtbmvPath::WideColumn || path == CtbmvPath::WideGroup; }

bool IsGenericColumn(const CtbmvTilingData& tiling)
{
    return tiling.path == CtbmvPath::Generic && tiling.incx == 1 && tiling.trans == ACLBLAS_OP_N;
}

uint32_t SelectCtbmvCores(CtbmvPath path, uint32_t n, int incx, aclblasOperation_t trans, uint32_t availableCoreNum)
{
    const bool genericColumn = path == CtbmvPath::Generic && incx == 1 && trans == ACLBLAS_OP_N;
    if (path == CtbmvPath::TinyVector || path == CtbmvPath::SmallScalar ||
        (path == CtbmvPath::Generic && !genericColumn)) {
        return 1U;
    }
    if (path == CtbmvPath::SmallPanel) {
        return std::min<uint32_t>(availableCoreNum, CeilDiv<uint32_t>(n, 8U));
    }
    if (path == CtbmvPath::CompactPanel) {
        return CeilDiv<uint32_t>(n, n > 2048U ? 128U : 64U);
    }
    if (IsWideCtbmvPath(path)) {
        return std::min<uint32_t>(availableCoreNum, n);
    }
    return std::min<uint32_t>(n, availableCoreNum);
}

struct CtbmvWorkspace {
    const void* metadata = nullptr;
    size_t metadataBytes = 0U;
    size_t metadataOffset = 0U;
    size_t workspaceBytes = 0U;
    const void* groupMetadata = nullptr;
    size_t groupBytes = 0U;
};

void SetSmallCtbmvMetadata(const CtbmvTilingData& t, CtbmvWorkspace& plan)
{
    const auto uplo = static_cast<aclblasFillMode_t>(t.uplo);
    const auto trans = static_cast<aclblasOperation_t>(t.trans);
    const auto diag = static_cast<aclblasDiagType_t>(t.diag);
    if (t.path == CtbmvPath::TinyVector) {
        const auto& offsets = GetCtbmvTinyOffsets(t.n, t.k, t.lda, uplo, trans, diag);
        plan.metadata = offsets.data();
        plan.metadataBytes = offsets.size() * sizeof(uint32_t);
        plan.metadataOffset = 0U;
    } else {
        const auto& offsets = GetCtbmvSmallPanelOffsets(t.n, t.k, t.lda, uplo, trans, diag);
        plan.metadata = offsets.data();
        plan.metadataBytes = offsets.size() * sizeof(uint32_t);
    }
}

void SetCtbmvMetadata(const CtbmvTilingData& t, CtbmvWorkspace& plan)
{
    switch (t.path) {
        case CtbmvPath::TinyVector:
        case CtbmvPath::SmallPanel:
            SetSmallCtbmvMetadata(t, plan);
            break;
        case CtbmvPath::CompactPanel: {
            const auto& offsets = GetCtbmvPanelOffsets(t.lda);
            plan.metadata = offsets.data();
            plan.metadataBytes = offsets.size() * sizeof(uint32_t);
            break;
        }
        case CtbmvPath::Transpose:
            plan.metadata =
                t.k <= 128 ? GetCtbmvTransposeNarrowOffsets(t.k).data() : GetCtbmvTransposeWindowOffsets(t.k).data();
            plan.metadataBytes = (t.k <= 128 ? 8320U : 6272U) * sizeof(uint32_t);
            break;
        case CtbmvPath::WideColumn:
        case CtbmvPath::WideGroup: {
            const auto& offsets = GetCtbmvWideAtomicInterleaveOffsets();
            plan.metadata = offsets.data();
            plan.metadataBytes = offsets.size() * sizeof(uint32_t);
            if (t.path == CtbmvPath::WideGroup) {
                const auto& groupOffsets = GetCtbmvWideGroupOffsets(t.k);
                plan.groupMetadata = groupOffsets.data();
                plan.groupBytes = groupOffsets.size() * sizeof(uint32_t);
            }
            break;
        }
        case CtbmvPath::SmallScalar:
        case CtbmvPath::Generic:
        case CtbmvPath::Row:
            break;
    }
}

CtbmvWorkspace MakeCtbmvWorkspace(const CtbmvTilingData& tiling)
{
    CtbmvWorkspace plan;
    const size_t vectorBytes = static_cast<size_t>(tiling.n) * sizeof(aclblasComplex);
    plan.metadataOffset = CeilAlign<size_t>(vectorBytes, 32U);
    SetCtbmvMetadata(tiling, plan);
    plan.workspaceBytes = vectorBytes;
    if (plan.metadataBytes > 0U) {
        plan.workspaceBytes = plan.metadataOffset + plan.metadataBytes + plan.groupBytes;
        if (IsWideCtbmvPath(tiling.path)) {
            plan.workspaceBytes += vectorBytes * tiling.useCoreNum;
        }
    } else if (tiling.path == CtbmvPath::SmallScalar) {
        plan.workspaceBytes = 0U;
    } else if (IsGenericColumn(tiling)) {
        plan.workspaceBytes = vectorBytes * tiling.useCoreNum;
    }
    return plan;
}

aclblasStatus_t UploadCtbmvWorkspace(
    _aclblas_handle* h, aclblasComplex* x, uint8_t* workspace, const CtbmvWorkspace& plan,
    const CtbmvTilingData& tiling)
{
    const size_t vectorBytes = static_cast<size_t>(tiling.n) * sizeof(aclblasComplex);
    const bool wide = IsWideCtbmvPath(tiling.path);
    const auto path = tiling.path;
    const bool snapshotX = path == CtbmvPath::SmallPanel || path == CtbmvPath::CompactPanel ||
                           path == CtbmvPath::Transpose || path == CtbmvPath::Row || wide;
    if (snapshotX) {
        const auto status =
            aclrtMemcpyAsync(workspace, plan.workspaceBytes, x, vectorBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, h->stream);
        if (status != ACL_SUCCESS) {
            OP_LOGE("aclblasCtbmv", "aclblasCtbmv: x snapshot aclrtMemcpyAsync failed, status=%d", status);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    if (plan.metadataBytes > 0U &&
        aclrtMemcpyAsync(
            workspace + plan.metadataOffset, plan.workspaceBytes - plan.metadataOffset, plan.metadata,
            plan.metadataBytes, ACL_MEMCPY_HOST_TO_DEVICE, h->stream) != ACL_SUCCESS) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: metadata aclrtMemcpyAsync failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (plan.groupBytes > 0U &&
        aclrtMemcpyAsync(
            workspace + plan.metadataOffset + plan.metadataBytes, plan.groupBytes, plan.groupMetadata, plan.groupBytes,
            ACL_MEMCPY_HOST_TO_DEVICE, h->stream) != ACL_SUCCESS) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: group metadata aclrtMemcpyAsync failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (wide && aclrtMemsetAsync(x, vectorBytes, 0, vectorBytes, h->stream) != ACL_SUCCESS) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: output aclrtMemsetAsync failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (wide) {
        const size_t partialOffset = plan.metadataOffset + plan.metadataBytes + plan.groupBytes;
        const size_t partialBytes = vectorBytes * tiling.useCoreNum;
        if (aclrtMemsetAsync(workspace + partialOffset, partialBytes, 0, partialBytes, h->stream) != ACL_SUCCESS) {
            OP_LOGE("aclblasCtbmv", "aclblasCtbmv: partial workspace aclrtMemsetAsync failed");
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    if (IsGenericColumn(tiling) &&
        aclrtMemsetAsync(workspace, plan.workspaceBytes, 0, plan.workspaceBytes, h->stream) != ACL_SUCCESS) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: workspace aclrtMemsetAsync failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchCtbmv(
    aclblasHandle_t handle, const aclblasComplex* a, aclblasComplex* x, const CtbmvTilingData& tiling)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    auto* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    const auto plan = MakeCtbmvWorkspace(tiling);
    if (plan.workspaceBytes > 0U && (workspace == nullptr || plan.workspaceBytes > GetEffectiveWorkspaceSize(h))) {
        OP_LOGE(
            "aclblasCtbmv", "aclblasCtbmv: workspace is null or insufficient: required=%zu available=%zu",
            plan.workspaceBytes, GetEffectiveWorkspaceSize(h));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const auto status = UploadCtbmvWorkspace(h, x, workspace, plan, tiling);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    ctbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)), reinterpret_cast<uint8_t*>(x), workspace, tiling,
        tiling.useCoreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCtbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const aclblasStatus_t status = ValidateCtbmvParams(uplo, trans, diag, n, k, A, lda, x, incx);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n == 0 || (k == 0 && diag == ACLBLAS_UNIT)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t availableCoreNum = GetAivCoreCount();
    if (availableCoreNum == 0U) {
        OP_LOGE("aclblasCtbmv", "aclblasCtbmv: no available AIV cores");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const auto path = SelectCtbmvPath(n, k, lda, incx, uplo, trans, availableCoreNum);
    const auto cores = SelectCtbmvCores(path, static_cast<uint32_t>(n), incx, trans, availableCoreNum);
    auto tiling = MakeCtbmvTiling(n, k, lda, incx, uplo, trans, diag, cores, path);
    if (path == CtbmvPath::CompactPanel) {
        tiling.rowsPerCore = n > 2048 ? 128U : 64U;
    }
    return LaunchCtbmv(handle, A, x, tiling);
}
