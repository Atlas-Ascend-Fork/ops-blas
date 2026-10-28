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
 * \file aclblaslt_algo_heuristic.h
 * \brief Internal matmul algo encode/decode, the serialized PackedAlgo layout, tiling override, and
 *        the heuristic candidate scoring helpers shared by the algo API and the matmul engine.
 */

#pragma once

#include "aclblaslt_handle_impl.h"
#include "matmul_fp32_tiling_data.h"

#include "cann_ops_blasLt.h"

#include <acl/acl.h>
#include <cstdint>
#include <vector>

constexpr uint32_t ACLBLASLT_ALGO_MAGIC = 0xACBD1234;

constexpr uint32_t DEFAULT_L1_M = 128;
constexpr uint32_t DEFAULT_L1_N = 128;
constexpr uint32_t DEFAULT_L1_K = 128;
// Default algoId for policy=MMAD_SYNC, L1 tile 128x128x128, splitK=1. The bit shifts mirror
// GenerateAlgoId(): [28:31] policy | [16:23] l1m | [8:15] l1n | [2:9] l1k | [0:1] splitK.
constexpr uint32_t DEFAULT_ALGO_ID = (0u << 28) ^ (128u << 16) ^ (128u << 8) ^ (128u << 2) ^ 1u;

// PackedAlgo.flags bit layout:
//   bits [0:1] reduction scheme (aclblasLtReductionScheme_t)
//   bits [4:5] L1 K-tile index (0 = unset/default 128, 1 = 64, 2 = 128, 3 = 256)
// Encoding l1k in the spare flag bits lets heuristic/matmul distinguish tiles that share the same
// M x N but differ in K without growing the 16-byte PackedAlgo layout, so the aclblasLtMatmulAlgo_t
// ABI stays unchanged.
constexpr uint8_t FLAG_REDUCTION_MASK = 0x03;
constexpr uint8_t FLAG_L1K_SHIFT = 4;
constexpr uint8_t FLAG_L1K_MASK = 0x30;

enum DispatchPolicyType : uint8_t {
    DISPATCH_POLICY_MMAD_SYNC = 0,
    DISPATCH_POLICY_MMAD_PINGPONG = 1,
    DISPATCH_POLICY_MMAD_MULTI_STAGE = 2,
};

struct AscendHardwareCaps {
    uint32_t numAICores = DEFAULT_AI_CORES;
    uint32_t l0CubeSize = L0_SIZE;
    size_t l1BufferSize = L1_SIZE;
    double memoryBandwidthGBps = DEFAULT_PEAK_GBPS;
    double peakTFlops = DEFAULT_PEAK_TFLOPS;
    double bandwidthBoundThreshold = 32.0;
};

struct AlgoCandidate {
    uint32_t algoId = 0;
    uint32_t l1TileM = 128;
    uint32_t l1TileN = 128;
    uint32_t l1TileK = 128;
    uint32_t l0TileM = 64;
    uint32_t l0TileN = 64;
    uint32_t l0TileK = 64;
    DispatchPolicyType policy = DISPATCH_POLICY_MMAD_SYNC;
    uint32_t numBuffers = 1;
    uint32_t splitKFactor = 1;
    size_t workspaceSize = 0;
    double peakPerformance = DEFAULT_PEAK_TFLOPS;
};

struct PackedAlgo {
    uint32_t magic;
    uint32_t algoId;
    uint16_t l1mDiv16;
    uint16_t l1nDiv16;
    uint8_t policy;
    uint8_t numBuffers;
    uint8_t splitK;
    uint8_t flags;
};
static_assert(sizeof(PackedAlgo) == 16, "PackedAlgo must fit algo.data");

// ---- algoId / flags codec ----
uint8_t EncodeL1KToFlags(uint8_t flags, uint32_t l1k);
uint32_t DecodeL1KFromFlags(uint8_t flags);
uint32_t GenerateAlgoId(DispatchPolicyType policy, uint32_t l1m, uint32_t l1n, uint32_t l1k, uint32_t splitKFactor);

// ---- PackedAlgo (de)serialization ----
aclblasLtMatmulAlgo_t BuildAlgoFromCandidate(const AlgoCandidate& cand);
bool DecodeAlgo(const aclblasLtMatmulAlgo_t& algo, PackedAlgo* packed);

// Apply the decoded algorithm's L1 tile onto the FP32 kernel tiling (N and K tiles only take effect;
// see the definition for the rationale).
void ApplyAlgoTilingOverrideFp32(
    const PackedAlgo& packed, uint64_t m, uint64_t n, uint64_t k, MatmulFp32TilingData& tiling);

// ---- ACLBLASLT_MATMUL_TILE_* <-> L1 M/N mapping ----
bool TileIdToL1MN(uint32_t tileId, uint16_t* l1mDiv16, uint16_t* l1nDiv16);
uint32_t L1MNToTileId(uint32_t l1m, uint32_t l1n);

// ---- algoId enumeration / validation ----
bool DecodeAlgoIdFields(
    uint32_t algoId, uint8_t* policy, uint16_t* l1mDiv16, uint16_t* l1nDiv16, uint32_t* l1k, uint8_t* splitK);
std::vector<int> EnumerateValidAlgoIds();

// ---- Algo capability query (aclblasLtMatmulAlgoCapGetAttribute) ----
// Maximum element count of any supported array-valued cap (TILE_IDS / STAGES_IDS: 4 elements each).
constexpr size_t ALGO_CAP_MAX_ELEMS = 4;

// Resolve the capability value of cap from the static validity domains: returns the required byte
// count (scalar 4 / array elementCount * 4) and writes the value(s) into outValues (capacity >=
// ALGO_CAP_MAX_ELEMS). Returns 0 for any cap outside the enumerated subset (8~12, 14+, the MAX
// sentinel, or any out-of-range int) without writing outValues; the caller maps 0 to
// ACLBLAS_STATUS_NOT_SUPPORTED.
size_t ResolveAlgoCapValue(aclblasLtMatmulAlgoCapAttributes_t cap, uint32_t* outValues);

// ---- heuristic scoring helpers ----
void GetAscendHardwareCaps(int32_t deviceId, AscendHardwareCaps* caps);
void SelectL0TileShape(
    uint32_t l1M, uint32_t l1N, uint32_t l1K, size_t, size_t, aclDataType, aclDataType, uint32_t* l0M, uint32_t* l0N,
    uint32_t* l0K);
size_t CalculateWorkspaceForAscend(uint64_t m, uint64_t n, uint32_t splitKFactor, aclblasLtEpilogue_t epilogue);
