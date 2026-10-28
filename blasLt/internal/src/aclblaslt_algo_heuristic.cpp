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
 * \file aclblaslt_algo_heuristic.cpp
 * \brief Matmul algo encode/decode, PackedAlgo (de)serialization, FP32 tiling override, and the
 *        heuristic scoring helpers.
 */

#include "aclblaslt_algo_heuristic.h"

#include "host_utils.h"
#include "matmul_fp32_tiling_data.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

// Valid tiling parameter domains shared by algoId encode/decode and AlgoGetIds enumeration.
constexpr uint32_t VALID_L1_M[] = {128, 256};
constexpr uint32_t VALID_L1_N[] = {128, 256};
constexpr uint32_t VALID_L1_K[] = {64, 128, 256};
constexpr uint8_t VALID_POLICIES[] = {0, 1, 2};
constexpr uint8_t VALID_SPLIT_K[] = {1, 2, 3};

struct TileEntry {
    uint32_t m;
    uint32_t n;
};

const TileEntry kTileTable[] = {
    /* [UNDEFINED] */ {0, 0},
    /* [128x128]   */ {128, 128},
    /* [128x256]   */ {128, 256},
    /* [256x128]   */ {256, 128},
    /* [256x256]   */ {256, 256},
};

// Scan the (n, k, splitK) tiling sub-domain for a fixed (policy, l1m) and report the first field set
// whose GenerateAlgoId matches algoId. Extracted from DecodeAlgoIdFields to cap nesting depth.
bool MatchAlgoIdForPolicyM(
    uint32_t algoId, uint8_t p, uint32_t m, uint8_t* policy, uint16_t* l1mDiv16, uint16_t* l1nDiv16, uint32_t* l1k,
    uint8_t* splitK)
{
    for (uint32_t n : VALID_L1_N) {
        for (uint32_t kk : VALID_L1_K) {
            for (uint8_t s : VALID_SPLIT_K) {
                if (GenerateAlgoId(static_cast<DispatchPolicyType>(p), m, n, kk, s) == algoId) {
                    *policy = p;
                    *l1mDiv16 = static_cast<uint16_t>(m / 16);
                    *l1nDiv16 = static_cast<uint16_t>(n / 16);
                    *l1k = kk;
                    *splitK = s;
                    return true;
                }
            }
        }
    }
    return false;
}

// Append every algoId over the (m, n, k, splitK) sub-domain for a fixed policy. Extracted from
// EnumerateValidAlgoIds to cap nesting depth.
void AppendAlgoIdsForPolicy(uint8_t p, std::vector<int>& ids)
{
    for (uint32_t m : VALID_L1_M) {
        for (uint32_t n : VALID_L1_N) {
            for (uint32_t kk : VALID_L1_K) {
                for (uint8_t s : VALID_SPLIT_K) {
                    ids.push_back(static_cast<int>(GenerateAlgoId(static_cast<DispatchPolicyType>(p), m, n, kk, s)));
                }
            }
        }
    }
}

// ---- Algo capability value derivation (aclblasLtMatmulAlgoCapGetAttribute) ----
// Capability values are the full static validity domains declared above; they never depend on the
// packed algo fields. Each helper materializes one ascending uint32_t array (capacity bounded by
// ALGO_CAP_MAX_ELEMS) and returns the required byte count (elementCount * sizeof(uint32_t)).

// Anti-silent-truncation guards: the helpers below materialize into a caller buffer of
// ALGO_CAP_MAX_ELEMS elements. Growing the tile domains or the stages range beyond that capacity
// must fail the build here instead of silently under-reporting the arrays (the runtime
// `count >= ALGO_CAP_MAX_ELEMS` guards stay as defense in depth).
static_assert(sizeof(VALID_L1_M) / sizeof(VALID_L1_M[0]) * (sizeof(VALID_L1_N) / sizeof(VALID_L1_N[0])) <=
                  ALGO_CAP_MAX_ELEMS,
              "TILE_IDS domain (VALID_L1_M x VALID_L1_N) exceeds ALGO_CAP_MAX_ELEMS: raise the capacity");
static_assert(static_cast<uint32_t>(ACLBLASLT_MATMUL_STAGES_4) - static_cast<uint32_t>(ACLBLASLT_MATMUL_STAGES_1) +
                      1u <= ALGO_CAP_MAX_ELEMS,
              "STAGES_IDS domain (STAGES_1..STAGES_4) exceeds ALGO_CAP_MAX_ELEMS: raise the capacity");

// TILE_IDS: every VALID_L1_M x VALID_L1_N combination mapped through L1MNToTileId (UNDEFINED
// filtered out), ascending and deduplicated. The current domains {128,256} x {128,256} derive to
// {1,2,3,4}.
size_t FillCapTileIds(uint32_t* outValues)
{
    size_t count = 0;
    for (uint32_t m : VALID_L1_M) {
        for (uint32_t n : VALID_L1_N) {
            const uint32_t tileId = L1MNToTileId(m, n);
            if (tileId == ACLBLASLT_MATMUL_TILE_UNDEFINED || count >= ALGO_CAP_MAX_ELEMS) {
                continue;
            }
            outValues[count] = tileId;
            ++count;
        }
    }
    std::sort(outValues, outValues + count);
    // Deduplicate: distinct (m, n) pairs may map onto the same tile id if the tile table ever
    // gains aliases; the reported array must stay a strictly ascending id set.
    count = static_cast<size_t>(std::unique(outValues, outValues + count) - outValues);
    return count * sizeof(uint32_t);
}

// STAGES_IDS: the ConfigSetStages acceptance domain [1,4] (enforced in the api layer), materialized
// from the public ACLBLASLT_MATMUL_STAGES_1.._4 enumerators as the single source of truth here.
// Assumes STAGES_1..STAGES_4 are contiguous enum values (currently 1..4; the static_assert above
// bounds the range). If a non-contiguous stages set is ever introduced, replace the loop with an
// explicit table of the supported enumerators.
size_t FillCapStagesIds(uint32_t* outValues)
{
    size_t count = 0;
    for (uint32_t stages = ACLBLASLT_MATMUL_STAGES_1; stages <= static_cast<uint32_t>(ACLBLASLT_MATMUL_STAGES_4);
         ++stages) {
        if (count >= ALGO_CAP_MAX_ELEMS) {
            break;
        }
        outValues[count] = stages;
        ++count;
    }
    return count * sizeof(uint32_t);
}

} // namespace

uint8_t EncodeL1KToFlags(uint8_t flags, uint32_t l1k)
{
    uint8_t idx = 0;
    switch (l1k) {
        case 64:
            idx = 1;
            break;
        case 128:
            idx = 2;
            break;
        case 256:
            idx = 3;
            break;
        default:
            idx = 0;
            break;
    }
    return static_cast<uint8_t>(
        (flags & static_cast<uint8_t>(~FLAG_L1K_MASK)) | static_cast<uint8_t>(idx << FLAG_L1K_SHIFT));
}

uint32_t DecodeL1KFromFlags(uint8_t flags)
{
    switch ((flags & FLAG_L1K_MASK) >> FLAG_L1K_SHIFT) {
        case 1:
            return 64u;
        case 2:
            return 128u;
        case 3:
            return 256u;
        default:
            return DEFAULT_L1_K;
    }
}

uint32_t GenerateAlgoId(DispatchPolicyType policy, uint32_t l1m, uint32_t l1n, uint32_t l1k, uint32_t splitKFactor)
{
    return (static_cast<uint32_t>(policy) << 28) ^ (l1m << 16) ^ (l1n << 8) ^ (l1k << 2) ^ splitKFactor;
}

aclblasLtMatmulAlgo_t BuildAlgoFromCandidate(const AlgoCandidate& cand)
{
    aclblasLtMatmulAlgo_t out{};
    PackedAlgo packed{};
    packed.magic = ACLBLASLT_ALGO_MAGIC;
    packed.algoId = cand.algoId;
    packed.l1mDiv16 = static_cast<uint16_t>(cand.l1TileM / 16);
    packed.l1nDiv16 = static_cast<uint16_t>(cand.l1TileN / 16);
    packed.policy = static_cast<uint8_t>(cand.policy);
    packed.numBuffers = static_cast<uint8_t>(cand.numBuffers);
    packed.splitK = static_cast<uint8_t>(cand.splitKFactor);
    // flags default: reduction scheme NONE (low bits 0) + L1 K-tile index in the high bits.
    packed.flags = EncodeL1KToFlags(0u, cand.l1TileK);
    (void)MemcpySSucceeds(out.data, sizeof(out.data), &packed, sizeof(packed));
    out.max_workspace_bytes = cand.workspaceSize;
    return out;
}

bool DecodeAlgo(const aclblasLtMatmulAlgo_t& algo, PackedAlgo* packed)
{
    if (packed == nullptr) {
        return false;
    }
    if (!MemcpySSucceeds(packed, sizeof(PackedAlgo), algo.data, sizeof(PackedAlgo))) {
        return false;
    }
    return packed->magic == ACLBLASLT_ALGO_MAGIC;
}

// Apply the decoded algorithm's L1 tile onto the FP32 kernel tiling. The FP32 tiling payload only
// exposes base tiles (baseM/baseN/baseK) and the L1 K depth (kL1); dispatch policy and pipeline
// stages have no corresponding field, so they are decoded and recorded but do not alter this path.
// Because the default baseM already equals min(m, 128) and every valid L1 M tile is >= 128, only the
// N and K tiles produce an observable effect here.
void ApplyAlgoTilingOverrideFp32(
    const PackedAlgo& packed, uint64_t m, uint64_t n, uint64_t k, MatmulFp32TilingData& tiling)
{
    (void)m;
    const uint32_t l1n = static_cast<uint32_t>(packed.l1nDiv16) * 16u;
    const uint32_t l1k = DecodeL1KFromFlags(packed.flags);
    if (l1n > 0U && tiling.baseN > 0U) {
        tiling.baseN = std::min<uint32_t>(tiling.baseN, static_cast<uint32_t>(std::min<uint64_t>(n, l1n)));
    }
    if (l1k > 0U) {
        tiling.kL1 = static_cast<uint32_t>(std::min<uint64_t>(k, l1k));
        if (tiling.baseK > tiling.kL1) {
            tiling.baseK = tiling.kL1;
        }
    }
}

bool TileIdToL1MN(uint32_t tileId, uint16_t* l1mDiv16, uint16_t* l1nDiv16)
{
    if (tileId == ACLBLASLT_MATMUL_TILE_UNDEFINED || tileId > static_cast<uint32_t>(ACLBLASLT_MATMUL_TILE_256x256)) {
        return false;
    }
    const TileEntry& entry = kTileTable[tileId];
    *l1mDiv16 = static_cast<uint16_t>(entry.m / 16);
    *l1nDiv16 = static_cast<uint16_t>(entry.n / 16);
    return true;
}

uint32_t L1MNToTileId(uint32_t l1m, uint32_t l1n)
{
    for (uint32_t i = 1; i <= static_cast<uint32_t>(ACLBLASLT_MATMUL_TILE_256x256); ++i) {
        if (kTileTable[i].m == l1m && kTileTable[i].n == l1n) {
            return i;
        }
    }
    return ACLBLASLT_MATMUL_TILE_UNDEFINED;
}

bool DecodeAlgoIdFields(
    uint32_t algoId, uint8_t* policy, uint16_t* l1mDiv16, uint16_t* l1nDiv16, uint32_t* l1k, uint8_t* splitK)
{
    for (uint8_t p : VALID_POLICIES) {
        for (uint32_t m : VALID_L1_M) {
            if (MatchAlgoIdForPolicyM(algoId, p, m, policy, l1mDiv16, l1nDiv16, l1k, splitK)) {
                return true;
            }
        }
    }
    return false;
}

// Enumerate every algoId reachable through GenerateAlgoId over the supported tiling domains.
// Used by aclblasLtMatmulAlgoGetIds and to validate non-zero algoIds in aclblasLtMatmulAlgoInit.
std::vector<int> EnumerateValidAlgoIds()
{
    std::vector<int> ids;
    for (uint8_t p : VALID_POLICIES) {
        AppendAlgoIdsForPolicy(p, ids);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

// Resolve one capability attribute of aclblasLtMatmulAlgoCapGetAttribute from the static validity
// domains (array-valued caps are derived by the Fill* helpers in the anonymous namespace above).
// Returns the required byte count, or 0 for unknown / unsupported caps; callers map 0 to
// ACLBLAS_STATUS_NOT_SUPPORTED. No side effects, no heap allocation, reentrant; outValues is left
// untouched whenever 0 is returned.
size_t ResolveAlgoCapValue(aclblasLtMatmulAlgoCapAttributes_t cap, uint32_t* outValues)
{
    if (outValues == nullptr) {
        return 0;
    }
    switch (cap) {
        case ACLBLASLT_ALGO_CAP_SPLITK_SUPPORT:
            // Implementation-capability view (cuBLASLt semantics: "split-K implementation is
            // supported"): the current kernel set has no split-K reduction stage (see the api-layer
            // note in BuildRankedCandidates) and the FP32 tiling path does not consume splitK.
            // ConfigSetSplitK still accepts [1,255] and the algoId encoding domain is {1,2,3}:
            // splitK > 1 is recorded but not executed as a real split. Self-consistent with
            // REDUCTION_SCHEME_MASK = 0 (no reduction scheme is implemented).
            outValues[0] = 0u;
            return sizeof(uint32_t);
        case ACLBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK:
            // Mask semantics aligned with cuBLASLt: the value is the bitwise OR of the supported
            // aclblasLtReductionScheme_t enumerator values, and NONE (= 0) contributes no bit.
            // INPLACE / WORKSPACE have no implementation and are unsupported, so the mask is 0.
            // Self-consistent with SPLITK_SUPPORT = 0 (no reduction implementation implies no
            // split-K execution).
            outValues[0] = 0u;
            return sizeof(uint32_t);
        case ACLBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT:
            // This library has no CTA swizzling concept.
            outValues[0] = 0u;
            return sizeof(uint32_t);
        case ACLBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT:
            // The layout attributes BATCH_COUNT / STRIDED_BATCH_OFFSET are storable and readable,
            // but the aclblasLtMatmul compute path never consumes the batch fields (single matrix
            // only), so strided batching is reported as unsupported.
            outValues[0] = 0u;
            return sizeof(uint32_t);
        case ACLBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT:
            // aclblasLtMatmul supports both in-place (C == D) and out-of-place (C != D) results.
            outValues[0] = 1u;
            return sizeof(uint32_t);
        case ACLBLASLT_ALGO_CAP_UPLO_SUPPORT:
            // Matmul descriptors carry no uplo / triangular attribute.
            outValues[0] = 0u;
            return sizeof(uint32_t);
        case ACLBLASLT_ALGO_CAP_TILE_IDS:
            return FillCapTileIds(outValues);
        case ACLBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX: {
            const uint8_t* policiesEnd = VALID_POLICIES + sizeof(VALID_POLICIES) / sizeof(VALID_POLICIES[0]);
            outValues[0] = static_cast<uint32_t>(*std::max_element(VALID_POLICIES, policiesEnd));
            return sizeof(uint32_t);
        }
        case ACLBLASLT_ALGO_CAP_STAGES_IDS:
            return FillCapStagesIds(outValues);
        case ACLBLASLT_ALGO_CAP_MAX:
        default:
            // Values outside the enumerated subset (cuBLASLt 8~12 MATHMODE_IMPL/GAUSSIAN_IMPL/
            // CUSTOM_MEMORY_ORDER/POINTER_MODE_MASK/EPILOGUE_MASK, 14~23 LD_NEGATIVE/
            // NUMERICAL_IMPL_FLAGS/MIN_ALIGNMENT_*/ATOMIC_SYNC/CUDA 13.x additions, the MAX
            // sentinel, or any out-of-range int passed through the C ABI) are unsupported.
            return 0;
    }
}

// Return lightweight default hardware caps until real device queries are wired up.
void GetAscendHardwareCaps(int32_t, AscendHardwareCaps* caps)
{
    if (caps == nullptr) {
        return;
    }
    caps->numAICores = DEFAULT_AI_CORES;
    caps->l0CubeSize = L0_SIZE;
    caps->l1BufferSize = L1_SIZE;
    caps->memoryBandwidthGBps = DEFAULT_PEAK_GBPS;
    caps->peakTFlops = DEFAULT_PEAK_TFLOPS;
    caps->bandwidthBoundThreshold = 32.0;
}

void SelectL0TileShape(
    uint32_t l1M, uint32_t l1N, uint32_t l1K, size_t, size_t, aclDataType, aclDataType, uint32_t* l0M, uint32_t* l0N,
    uint32_t* l0K)
{
    *l0K = std::min(64u, l1K);
    *l0M = std::min(128u, l1M);
    *l0N = std::min(256u, l1N);

    while (*l0M > 16 && (l1M % *l0M != 0)) {
        --(*l0M);
    }
    while (*l0N > 16 && (l1N % *l0N != 0)) {
        --(*l0N);
    }
}

size_t CalculateWorkspaceForAscend(uint64_t m, uint64_t n, uint32_t splitKFactor, aclblasLtEpilogue_t epilogue)
{
    size_t workspace = 0;
    if (splitKFactor > 1) {
        workspace +=
            static_cast<size_t>(splitKFactor) * static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
    }

    switch (epilogue) {
        case ACLBLASLT_EPILOGUE_BIAS:
        case ACLBLASLT_EPILOGUE_RELU_BIAS:
        case ACLBLASLT_EPILOGUE_GELU_BIAS:
            workspace += static_cast<size_t>(m) * sizeof(float);
            break;
        case ACLBLASLT_EPILOGUE_GELU:
        case ACLBLASLT_EPILOGUE_RELU:
            workspace += 64 * 1024;
            break;
        default:
            break;
    }
    return workspace;
}
