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
 * \file aclblaslt_matmul_algo.cpp
 * \brief Public C API: matmul algorithm init / config / id enumeration / heuristic.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_algo_heuristic.h"
#include "aclblaslt_handle_impl.h"
#include "aclblaslt_layout_impl.h"
#include "host_utils.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

// ---- Gate-off MX feature alignment (single source for the three query entries below) ----
// With the MXFP8/MXFP4 subtrees excluded from this build (ENABLE_BLASLT_MXFP8 undefined or 0),
// a well-formed MX request must be rejected at the query entries with the same NOT_SUPPORTED that
// the engine fallback (matmul_engine.cpp) reports at execution time, instead of advertising algos
// that can never execute. The pair condition mirrors the engine dispatch checks (E4M3/E5M2 pair,
// E2M1 pair). In gate-on builds this helper is constant-false, so every call site keeps its
// pre-gate behavior exactly (the MX rejection becomes a dead branch).
inline bool IsGatedOffMxPair(aclDataType typeA, aclDataType typeB)
{
#if defined(ENABLE_BLASLT_MXFP8) && (ENABLE_BLASLT_MXFP8)
    (void)typeA;
    (void)typeB;
    return false;
#else
    return (IsMxfp8Type(typeA) && IsMxfp8Type(typeB)) || (IsMxfp4Type(typeA) && IsMxfp4Type(typeB));
#endif
}

// ---- AlgoInit helpers ----

aclblasStatus_t ValidateAlgoInitArgs(
    aclblasLtHandle_t lightHandle, aclblasComputeType_t computeType, aclDataType scaleType, aclDataType Atype,
    aclDataType Btype, aclDataType Ctype, aclDataType Dtype, int algoId)
{
    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (algoId < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!CheckComputeTypeCompatibility(computeType, Atype, Btype) || IsGatedOffMxPair(Atype, Btype)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (Ctype != Dtype) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (Dtype == ACL_FLOAT && scaleType != ACL_FLOAT && scaleType != ACL_DT_UNDEFINED) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Populate packed with the default tiling, then override from a non-zero algoId. Returns false when
// the algoId cannot be decoded into a valid tiling configuration.
bool InitPackedAlgoForId(int algoId, PackedAlgo& packed)
{
    packed.magic = ACLBLASLT_ALGO_MAGIC;
    packed.l1mDiv16 = static_cast<uint16_t>(DEFAULT_L1_M / 16);
    packed.l1nDiv16 = static_cast<uint16_t>(DEFAULT_L1_N / 16);
    packed.policy = static_cast<uint8_t>(DISPATCH_POLICY_MMAD_SYNC);
    packed.numBuffers = 1;
    packed.splitK = 1;
    packed.flags = 0;

    if (algoId == 0) {
        packed.algoId = DEFAULT_ALGO_ID;
        packed.flags = EncodeL1KToFlags(packed.flags, DEFAULT_L1_K);
        return true;
    }

    packed.algoId = static_cast<uint32_t>(algoId);
    uint8_t decPolicy = 0;
    uint16_t decL1mDiv16 = 0;
    uint16_t decL1nDiv16 = 0;
    uint32_t decL1k = DEFAULT_L1_K;
    uint8_t decSplitK = 0;
    if (!DecodeAlgoIdFields(packed.algoId, &decPolicy, &decL1mDiv16, &decL1nDiv16, &decL1k, &decSplitK)) {
        return false;
    }
    packed.policy = decPolicy;
    packed.l1mDiv16 = decL1mDiv16;
    packed.l1nDiv16 = decL1nDiv16;
    packed.splitK = decSplitK;
    packed.flags = EncodeL1KToFlags(packed.flags, decL1k);
    return true;
}

// ---- ConfigSetAttribute helpers ----

aclblasStatus_t ReadConfigU32(const void* buf, size_t sizeInBytes, uint32_t& out)
{
    if (sizeInBytes != sizeof(uint32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!MemcpySSucceeds(&out, sizeof(out), buf, sizeof(uint32_t))) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ConfigSetTileId(PackedAlgo& packed, const void* buf, size_t sizeInBytes)
{
    uint32_t tileId = 0;
    const aclblasStatus_t st = ReadConfigU32(buf, sizeInBytes, tileId);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (tileId == ACLBLASLT_MATMUL_TILE_UNDEFINED) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    uint16_t l1mDiv16 = 0;
    uint16_t l1nDiv16 = 0;
    if (!TileIdToL1MN(tileId, &l1mDiv16, &l1nDiv16)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    packed.l1mDiv16 = l1mDiv16;
    packed.l1nDiv16 = l1nDiv16;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ConfigSetStages(PackedAlgo& packed, const void* buf, size_t sizeInBytes)
{
    uint32_t stages = 0;
    const aclblasStatus_t st = ReadConfigU32(buf, sizeInBytes, stages);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (stages == ACLBLASLT_MATMUL_STAGES_UNDEFINED) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (stages < 1 || stages > 4) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    packed.numBuffers = static_cast<uint8_t>(stages);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ConfigSetSplitK(PackedAlgo& packed, const void* buf, size_t sizeInBytes)
{
    uint32_t splitK = 0;
    const aclblasStatus_t st = ReadConfigU32(buf, sizeInBytes, splitK);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (splitK < 1 || splitK > 255) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    packed.splitK = static_cast<uint8_t>(splitK);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ConfigSetReduction(PackedAlgo& packed, const void* buf, size_t sizeInBytes)
{
    uint32_t rs = 0;
    const aclblasStatus_t st = ReadConfigU32(buf, sizeInBytes, rs);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (rs > 2) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    packed.flags = static_cast<uint8_t>(
        (packed.flags & static_cast<uint8_t>(~FLAG_REDUCTION_MASK)) | static_cast<uint8_t>(rs & FLAG_REDUCTION_MASK));
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ConfigSetCustomOption(PackedAlgo& packed, const void* buf, size_t sizeInBytes)
{
    uint32_t opt = 0;
    const aclblasStatus_t st = ReadConfigU32(buf, sizeInBytes, opt);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (opt > 2) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    packed.policy = static_cast<uint8_t>(opt);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ApplyAlgoConfigSetAttr(
    PackedAlgo& packed, aclblasLtMatmulAlgoConfigAttributes_t attr, const void* buf, size_t sizeInBytes)
{
    switch (attr) {
        case ACLBLASLT_ALGO_CONFIG_TILE_ID:
            return ConfigSetTileId(packed, buf, sizeInBytes);
        case ACLBLASLT_ALGO_CONFIG_STAGES_ID:
            return ConfigSetStages(packed, buf, sizeInBytes);
        case ACLBLASLT_ALGO_CONFIG_SPLITK_NUM:
            return ConfigSetSplitK(packed, buf, sizeInBytes);
        case ACLBLASLT_ALGO_CONFIG_REDUCTION_SCHEME:
            return ConfigSetReduction(packed, buf, sizeInBytes);
        case ACLBLASLT_ALGO_CONFIG_CUSTOM_OPTION:
            return ConfigSetCustomOption(packed, buf, sizeInBytes);
        case ACLBLASLT_ALGO_CONFIG_ID:
        case ACLBLASLT_ALGO_CONFIG_INNER_SHAPE_ID:
        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
}

// ---- ConfigGetAttribute helper ----

aclblasStatus_t ReadAlgoConfigValue(
    const PackedAlgo& packed, aclblasLtMatmulAlgoConfigAttributes_t attr, uint32_t& value)
{
    switch (attr) {
        case ACLBLASLT_ALGO_CONFIG_ID:
            value = packed.algoId;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_ALGO_CONFIG_TILE_ID:
            value = L1MNToTileId(packed.l1mDiv16 * 16, packed.l1nDiv16 * 16);
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_ALGO_CONFIG_STAGES_ID:
            value = (packed.numBuffers == 0) ? ACLBLASLT_MATMUL_STAGES_UNDEFINED : packed.numBuffers;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_ALGO_CONFIG_SPLITK_NUM:
            value = packed.splitK;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_ALGO_CONFIG_REDUCTION_SCHEME:
            value = packed.flags & FLAG_REDUCTION_MASK;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_ALGO_CONFIG_CUSTOM_OPTION:
            value = packed.policy;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_ALGO_CONFIG_INNER_SHAPE_ID:
        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
}

// ---- GetHeuristic helpers ----

struct RankedCandidate {
    AlgoCandidate cand;
    float score;
};

aclblasStatus_t ValidateHeuristicArgs(
    int requestedAlgoCount, const aclblasLtMatmulHeuristicResult_t heuristicResultsArray[],
    aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t computeDesc, aclblasLtMatrixLayout_t Adesc,
    aclblasLtMatrixLayout_t Bdesc, aclblasLtMatrixLayout_t Cdesc, aclblasLtMatrixLayout_t Ddesc)
{
    if (requestedAlgoCount <= 0 || heuristicResultsArray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lightHandle == nullptr || computeDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Adesc == nullptr || Bdesc == nullptr || Cdesc == nullptr || Ddesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Score a single L1 tile: blends multi-core load balance, L1 utilization and the caller's L0 tile
// preference. Higher is better.
float ScoreTileCandidate(
    uint64_t m, uint64_t n, uint32_t cm, uint32_t cn, uint32_t ck, uint32_t prefL0M, uint32_t prefL0N, uint32_t prefL0K,
    const AscendHardwareCaps& caps)
{
    const uint32_t tilesM = CeilDiv<uint32_t>(m, cm);
    const uint32_t tilesN = CeilDiv<uint32_t>(n, cn);
    const uint32_t totalTiles = tilesM * tilesN;
    const float balanceScore =
        1.0f - static_cast<float>(totalTiles % std::max(1u, caps.numAICores)) / std::max(1u, caps.numAICores);
    const size_t l1Usage = static_cast<size_t>(cm) * ck * sizeof(float) + static_cast<size_t>(cn) * ck * sizeof(float);
    const float l1Util = static_cast<float>(l1Usage) / static_cast<float>(L1_SIZE);
    float l0Match = 0.0f;
    if (prefL0M > 0 && cm % prefL0M == 0) {
        l0Match += 0.3f;
    }
    if (prefL0N > 0 && cn % prefL0N == 0) {
        l0Match += 0.3f;
    }
    if (prefL0K > 0 && ck % prefL0K == 0) {
        l0Match += 0.4f;
    }
    return balanceScore * 0.4f + std::min(l1Util, 1.0f) * 0.3f + l0Match * 0.3f;
}

std::vector<RankedCandidate> BuildRankedCandidates(
    uint64_t m, uint64_t n, const aclblasLtMatrixLayoutImpl* A, const aclblasLtMatrixLayoutImpl* B,
    const aclblasLtMatmulDescImpl* desc, uint32_t prefL0M, uint32_t prefL0N, uint32_t prefL0K, int32_t deviceId)
{
    AscendHardwareCaps caps;
    GetAscendHardwareCaps(deviceId, &caps);

    // L1 tile candidates (M, N, K); mirrors the heuristic search space.
    const uint32_t tileCandidates[][3] = {
        {128, 256, 256}, {256, 128, 256}, {256, 256, 128}, {128, 128, 128}, {256, 256, 64}};

    std::vector<RankedCandidate> ranked;
    ranked.reserve(sizeof(tileCandidates) / sizeof(tileCandidates[0]));

    for (const auto& tc : tileCandidates) {
        const uint32_t cm = tc[0];
        const uint32_t cn = tc[1];
        const uint32_t ck = tc[2];
        const float score = ScoreTileCandidate(m, n, cm, cn, ck, prefL0M, prefL0N, prefL0K, caps);

        AlgoCandidate cand;
        cand.l1TileM = cm;
        cand.l1TileN = cn;
        cand.l1TileK = ck;
        SelectL0TileShape(cm, cn, ck, 0, 0, A->type, B->type, &cand.l0TileM, &cand.l0TileN, &cand.l0TileK);
        cand.policy = DISPATCH_POLICY_MMAD_SYNC;
        cand.numBuffers = 1;
        // splitK stays 1: the current kernel set has no split-K reduction stage, so a real split
        // would produce partial sums without accumulation. Heuristic keeps single-K for correctness.
        cand.splitKFactor = 1;
        cand.algoId = GenerateAlgoId(cand.policy, cm, cn, ck, cand.splitKFactor);
        cand.workspaceSize = CalculateWorkspaceForAscend(m, n, cand.splitKFactor, desc->epilogue);
        ranked.push_back({cand, score});
    }

    std::sort(ranked.begin(), ranked.end(), [](const RankedCandidate& a, const RankedCandidate& b) {
        return a.score > b.score;
    });
    return ranked;
}

aclblasStatus_t FillHeuristicResults(
    const std::vector<RankedCandidate>& ranked, int requestedAlgoCount, size_t maxWorkspace,
    aclblasLtMatmulHeuristicResult_t heuristicResultsArray[], int& filled)
{
    filled = 0;
    for (const auto& rc : ranked) {
        if (filled >= requestedAlgoCount) {
            break;
        }
        // Respect the caller's workspace budget (cuBLAS semantics: skip algos that exceed it).
        if (maxWorkspace != 0 && rc.cand.workspaceSize > maxWorkspace) {
            continue;
        }
        heuristicResultsArray[filled].algo = BuildAlgoFromCandidate(rc.cand);
        heuristicResultsArray[filled].algo.max_workspace_bytes = rc.cand.workspaceSize;
        heuristicResultsArray[filled].workspaceSize = rc.cand.workspaceSize;
        heuristicResultsArray[filled].state = ACLBLAS_STATUS_SUCCESS;
        heuristicResultsArray[filled].wavesCount = 1.0f;
        if (CheckedMemsetS(
                heuristicResultsArray[filled].reserved, sizeof(heuristicResultsArray[filled].reserved),
                sizeof(heuristicResultsArray[filled].reserved)) != ACLBLAS_STATUS_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        ++filled;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclblasStatus_t aclblasLtMatmulAlgoInit(
    aclblasLtHandle_t lightHandle, aclblasComputeType_t computeType, aclDataType scaleType, aclDataType Atype,
    aclDataType Btype, aclDataType Ctype, aclDataType Dtype, int algoId, aclblasLtMatmulAlgo_t* algo)
{
    if (algo == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const aclblasStatus_t argStatus =
        ValidateAlgoInitArgs(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId);
    if (argStatus != ACLBLAS_STATUS_SUCCESS) {
        return argStatus;
    }

    PackedAlgo packed{};
    if (!InitPackedAlgoForId(algoId, packed)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (!MemcpySSucceeds(algo->data, sizeof(algo->data), &packed, sizeof(packed))) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    algo->max_workspace_bytes = DEFAULT_WORKSPACE_SIZE;

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulAlgoConfigSetAttribute(
    aclblasLtMatmulAlgo_t* algo, aclblasLtMatmulAlgoConfigAttributes_t attr, const void* buf, size_t sizeInBytes)
{
    if (algo == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    PackedAlgo packed;
    if (!DecodeAlgo(*algo, &packed)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const aclblasStatus_t setStatus = ApplyAlgoConfigSetAttr(packed, attr, buf, sizeInBytes);
    if (setStatus != ACLBLAS_STATUS_SUCCESS) {
        return setStatus;
    }

    // Preserve the L1 K-tile encoded in flags when recomputing the derived algoId, so
    // tweaking one attribute does not silently reset the K tile selected by the heuristic.
    const uint32_t configL1K = DecodeL1KFromFlags(packed.flags);
    packed.algoId = GenerateAlgoId(
        static_cast<DispatchPolicyType>(packed.policy), packed.l1mDiv16 * 16, packed.l1nDiv16 * 16, configL1K,
        packed.splitK);

    if (packed.splitK > 1) {
        algo->max_workspace_bytes = DEFAULT_WORKSPACE_SIZE;
    }

    if (!MemcpySSucceeds(algo->data, sizeof(algo->data), &packed, sizeof(packed))) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulAlgoConfigGetAttribute(
    const aclblasLtMatmulAlgo_t* algo, aclblasLtMatmulAlgoConfigAttributes_t attr, void* buf, size_t sizeInBytes,
    size_t* sizeWritten)
{
    if (algo == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (sizeInBytes == 0 && sizeWritten == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (sizeInBytes != 0 && buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    PackedAlgo packed;
    if (!DecodeAlgo(*algo, &packed)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const size_t requiredSize = sizeof(uint32_t);
    uint32_t value = 0;
    const aclblasStatus_t readStatus = ReadAlgoConfigValue(packed, attr, value);
    if (readStatus != ACLBLAS_STATUS_SUCCESS) {
        return readStatus;
    }

    if (sizeWritten != nullptr) {
        *sizeWritten = requiredSize;
    }

    if (sizeInBytes != 0) {
        if (sizeInBytes < requiredSize) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (!MemcpySSucceeds(buf, sizeInBytes, &value, requiredSize)) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulAlgoCapGetAttribute(
    const aclblasLtMatmulAlgo_t* algo, aclblasLtMatmulAlgoCapAttributes_t cap, void* capAttribute,
    size_t capAttributeSize, size_t* sizeWritten)
{
    if (algo == nullptr) { // error-code matrix row 1
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (capAttributeSize == 0 && sizeWritten == nullptr) { // error-code matrix row 2
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (capAttributeSize != 0 && capAttribute == nullptr) { // error-code matrix row 3
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    PackedAlgo packed;
    if (!DecodeAlgo(*algo, &packed)) { // error-code matrix row 4
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Capability values are the full static validity domains (dtype/algo independent): packed only
    // carries the "initialized algo" contract check (magic); its decoded fields never participate
    // in the capability value derivation below.

    uint32_t values[ALGO_CAP_MAX_ELEMS] = {0};
    const size_t requiredSize = ResolveAlgoCapValue(cap, values); // internal-layer table lookup
    if (requiredSize == 0) {                                      // error-code matrix row 5
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    // Error-code matrix rows 6~9 (probe / undersized buffer / copy / memcpy_s failure) are all
    // handled by the shared GetAttribute tail; sizeWritten reports the required size even when the
    // caller buffer is too small.
    return CopyAttributeOut(capAttribute, capAttributeSize, values, requiredSize, sizeWritten);
}

aclblasStatus_t aclblasLtMatmulAlgoGetIds(
    aclblasLtHandle_t lightHandle, aclblasComputeType_t computeType, aclDataType scaleType, aclDataType Atype,
    aclDataType Btype, aclDataType Ctype, aclDataType Dtype, int* algoIdsArray, int algoIdsArrayLength, int* numAlgoIds)
{
    (void)scaleType;
    if (numAlgoIds == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *numAlgoIds = 0;

    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (algoIdsArray == nullptr || algoIdsArrayLength <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!CheckComputeTypeCompatibility(computeType, Atype, Btype) || IsGatedOffMxPair(Atype, Btype)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (Ctype != Dtype) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    const std::vector<int> ids = EnumerateValidAlgoIds();
    const int count = std::min(algoIdsArrayLength, static_cast<int>(ids.size()));
    for (int i = 0; i < count; ++i) {
        algoIdsArray[i] = ids[static_cast<size_t>(i)];
    }
    *numAlgoIds = count;
    return ACLBLAS_STATUS_SUCCESS;
}

// issue #748: preference 读取提取为 helper，控制 GetHeuristic 函数规模
struct HeuristicPreference {
    size_t maxWorkspace = 0;
    uint32_t prefL0M = 0;
    uint32_t prefL0N = 0;
    uint32_t prefL0K = 0;
};

HeuristicPreference ReadHeuristicPreference(const aclblasLtMatmulPreference_t preference)
{
    HeuristicPreference pref;
    if (preference != nullptr) {
        auto* p = reinterpret_cast<aclblasLtMatmulPreferenceImpl*>(preference);
        pref.maxWorkspace = p->maxWorkspaceBytes;
        pref.prefL0M = p->preferredL0M;
        pref.prefL0N = p->preferredL0N;
        pref.prefL0K = p->preferredL0K;
    }
    return pref;
}

// issue #748: 类型对与维度校验提取为 helper，控制 GetHeuristic 函数规模。
// 维度超 uint32 表示范围时拒绝（与 matmul 执行路径的 IsRepresentableAsUint32 同口径）。
aclblasStatus_t ValidateHeuristicTypePair(
    aclblasLtMatmulHeuristicResult_t* heuristicResultsArray, const aclblasLtMatrixLayoutImpl* A,
    const aclblasLtMatrixLayoutImpl* B, const aclblasLtMatrixLayoutImpl* D, const aclblasLtMatmulDescImpl* desc)
{
    constexpr uint64_t kUint32Max = 0xFFFFFFFFULL;
    if (D->rows > kUint32Max || D->cols > kUint32Max) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!CheckComputeTypeCompatibility(desc->computeType, A->type, B->type)) {
        heuristicResultsArray[0].state = ACLBLAS_STATUS_INVALID_VALUE;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (IsGatedOffMxPair(A->type, B->type)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulAlgoGetHeuristic(
    aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t computeDesc, aclblasLtMatrixLayout_t Adesc,
    aclblasLtMatrixLayout_t Bdesc, aclblasLtMatrixLayout_t Cdesc, aclblasLtMatrixLayout_t Ddesc,
    aclblasLtMatmulPreference_t preference, int requestedAlgoCount,
    aclblasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount)
{
    // Validate input parameters
    if (returnAlgoCount == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *returnAlgoCount = 0;

    const aclblasStatus_t argStatus = ValidateHeuristicArgs(
        requestedAlgoCount, heuristicResultsArray, lightHandle, computeDesc, Adesc, Bdesc, Cdesc, Ddesc);
    if (argStatus != ACLBLAS_STATUS_SUCCESS) {
        return argStatus;
    }

    // Read the caller's workspace budget and preferred L0 tile shape (all optional).
    const HeuristicPreference pref = ReadHeuristicPreference(preference);
    const size_t maxWorkspace = pref.maxWorkspace;

    // Get matrix dimensions for GEMM D = A * B + C (A: m x k, B: k x n, C/D: m x n).
    auto* A = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
    auto* B = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
    auto* D = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Ddesc);
    auto* desc = reinterpret_cast<aclblasLtMatmulDescImpl*>(computeDesc);

    const uint64_t m = D->rows;
    const uint64_t n = D->cols;

    const aclblasStatus_t typeStatus = ValidateHeuristicTypePair(heuristicResultsArray, A, B, D, desc);
    if (typeStatus != ACLBLAS_STATUS_SUCCESS) {
        return typeStatus;
    }

    // Materialize ranked tiling candidates into serializable algo blobs. Each returned
    // aclblasLtMatmulAlgo_t carries a valid PackedAlgo (magic + tile + K index in flags), so
    // aclblasLtMatmul() can DecodeAlgo() it and honor the selected tiling.
    auto* ltHandle = reinterpret_cast<const _aclblaslt_handle*>(lightHandle);
    const std::vector<RankedCandidate> ranked =
        BuildRankedCandidates(m, n, A, B, desc, pref.prefL0M, pref.prefL0N, pref.prefL0K, ltHandle->deviceId);

    int filled = 0;
    const aclblasStatus_t fillStatus =
        FillHeuristicResults(ranked, requestedAlgoCount, maxWorkspace, heuristicResultsArray, filled);
    if (fillStatus != ACLBLAS_STATUS_SUCCESS) {
        return fillStatus;
    }

    if (filled == 0) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    *returnAlgoCount = filled;
    return ACLBLAS_STATUS_SUCCESS;
}

} // extern "C"
