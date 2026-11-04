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
#include "aclblaslt_logger_impl.h"
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

// ---- aclblasLtMatmulAlgoCheck helpers ----
// The generic workload predicates (IsRepresentableAsUint32 / IsSupportedPointerMode /
// HasSupportedTransposeModes / IsLayoutCompatible / HasSingleNonStridedBatch /
// AreOperandLayoutsCompatible / AreOutputLayoutsCompatible / IsMatmulLayoutValid /
// HasRowMajorLayouts) moved to host_utils.h and are shared with aclblasltMatmul()
// (they were formerly file-local mirrors kept in sync by hand). What remains local:
// HasSupportedDtypesForAlgoCheck and the Stage helpers below.
// Divergences by design (AlgoCheck-only semantics, not inconsistencies with matmul):
//   * Stage-0 null-argument checking precedes the Stage-1 handle check, so a
//     (lightHandle == nullptr && operationDesc == nullptr) call returns ACLBLAS_STATUS_INVALID_VALUE
//     here instead of matmul's ACLBLAS_STATUS_NOT_INITIALIZED (the result hard null-check
//     promotes argument-shape validity ahead of handle availability; single-fault outcomes stay
//     identical to matmul).
//   * HasSupportedDtypesForAlgoCheck applies the strict pointer-mode rule (alpha/beta are
//     not visible to AlgoCheck); see the FP32xFP32 branch inside that mirror.

// Mirror of HasSupportedDtypes in aclblaslt_matmul.cpp（该处为文件局部实现，无法跨文件复用；
// 修改彼处须同步此处）。唯一实质改造点：alpha/beta 对 AlgoCheck 不可见（本接口无
// 运行时标量入参），FP32×FP32 分支的 epilogueRuns 改为纯 pointerMode 判定，故改名
// ...ForAlgoCheck 以防与「完全一致」的预期混淆；其余分支逐行相同。
// 内核写出宽度由 A/B dtype 路径决定，C/D dtype 必须落在对应路径实际支持的白名单内，
// 否则设备端按错误元素宽度读写 D 造成越界或静默错误结果：
//   FP32 路径  ：MMAD 按 float 写原始乘积；HOST 模式 alpha=1 且 beta=0 时不启动 epilogue，
//                D 必须与原始乘积同宽（FP32），其余场景 epilogue 支持 FP32/BF16 输出；
//   MXFP8 路径 ：内核固定按 bfloat16 写出，D 须为 BF16；
//   MXFP4 路径 ：内核按 D dtype 选择 bf16/float 实例，D 须为二者之一。
// C 由 epilogue 读取，仅支持 FP32/BF16 解释。
bool HasSupportedDtypesForAlgoCheck(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b,
    const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d)
{
    const aclDataType dtypeA = a->type;
    const aclDataType dtypeB = b->type;
    const aclDataType dtypeC = c->type;
    const aclDataType dtypeD = d->type;
    // Gate-off MX rejection (see IsGatedOffMxPair and the invariant stated at the top of this file):
    // AlgoCheck is a query entry just like AlgoInit / AlgoGetIds / AlgoGetHeuristic, so a well-formed
    // MX request must be rejected here with the same NOT_SUPPORTED that the engine fallback reports
    // at execution time, instead of advertising an algo that can never execute. Without this gate a
    // gate-off build (ENABLE_BLASLT_MXFP8 FALSE, the default when asc-devkit < 9.1) reports
    // SUCCESS + workspaceSize for MXFP8/MXFP4 while aclblasLtMatmul returns NOT_SUPPORTED for the
    // very same arguments. Reachable without API abuse: the header documents that algo may be
    // restored from a previously serialized blob, so an MXFP8 algo serialized on a gate-on host
    // bypasses AlgoInit's gate when checked on a gate-off host.
    if (IsGatedOffMxPair(dtypeA, dtypeB)) {
        return false;
    }
    // C 白名单与 MXFP 输出 dtype 规则与 aclblasLtMatmul 共享（host_utils.h）；
    // FP32 分支两处有意分叉——本侧看不到 alpha/beta，只能用严格 pointerMode 判定（下同）。
    switch (CheckCAndMxDtypeRules(dtypeA, dtypeB, dtypeC, dtypeD)) {
        case AclblasMxDtypeRule::kUnsupported:
            return false;
        case AclblasMxDtypeRule::kSupported:
            return true;
        case AclblasMxDtypeRule::kNotMxPath:
            break;
    }
    if (dtypeA == ACL_FLOAT && dtypeB == ACL_FLOAT) {
        // 严格判定：alpha/beta 不可见，无法复现 matmul 的运行时 epilogueRuns 判定。
        // - DEVICE 模式：epilogue 必然运行（引擎 needEpilogue 含 devicePointerMode，
        //   见 matmul_engine.cpp 的 needEpilogue）→ D ∈ {FP32, BF16}；
        // - HOST 模式：按「epilogue 可能不运行」（alpha=1/beta=0 的最常见默认调用）判定
        //   → D == FP32。HOST + D=BF16 仅在 alpha≠1/beta≠0 时实际可跑，AlgoCheck 报
        //   NOT_SUPPORTED 属保守方向的已知假阴性（避免对默认调用假阳性），非缺陷。
        if (desc->pointerMode == ACLBLASLT_POINTER_MODE_DEVICE) {
            return dtypeD == ACL_FLOAT || dtypeD == ACL_BF16;
        }
        return dtypeD == ACL_FLOAT;
    }
    return false;
}

// Mirror of the MXFP k%32 check inlined in MatmulLaunch（matmul_engine.cpp，该处为文件局部实现，
// 修改彼处须同步此处）。调用点已过 dtype 门，A/B 同族 MXFP 时才可能命中。
// k=0 会通过 k%32 检查，但量化 tiling 的 CalScaleFactors 随后执行 0/0 整数除零
// （主机端 SIGFPE），MXFP 路径对 k=0 显式拒绝。
bool ViolatesMxfpKConstraint(aclDataType dtypeA, uint64_t k)
{
    return (IsMxfp8Type(dtypeA) || IsMxfp4Type(dtypeA)) && (k == 0 || k % 32 != 0);
}

// Mirror of the scaleA/scaleB null checks inlined in LaunchMmadKernel 各量化分支
// （matmul_engine.cpp，该处为文件局部实现，无法跨文件复用；修改彼处须同步此处）。引擎在 MXFP8 / MXFP4 分支内各自判空
// scaleA/scaleB；AlgoCheck 在 dtype 门通过后统一判定，二者等价。非 MXFP 路径无需 scale 指针，
// 不检查。
bool MissingMxfpScalePointers(const aclblasLtMatmulDescImpl* desc, aclDataType dtypeA, aclDataType dtypeB)
{
    const bool mxfpPair = (IsMxfp8Type(dtypeA) && IsMxfp8Type(dtypeB)) || (IsMxfp4Type(dtypeA) && IsMxfp4Type(dtypeB));
    return mxfpPair && (desc->scaleA == nullptr || desc->scaleB == nullptr);
}

// Mirror of the rowOrder inline variable in aclblaslt_matmul.cpp（该处为文件局部实现，无法跨文件
// 复用；修改彼处须同步此处）；引擎侧兜底见 matmul_engine.cpp 的 ORDER_ROW 检查。
// 字段填法与本文件 FillHeuristicResults（语义独立实现）
// 一致：state=SUCCESS、wavesCount=1.0f、reserved 清零；但 algo 来源不同——AlgoCheck 原样
// 回显调用方传入的 algo（memcpy_s 整体 24B 拷贝），而非从 candidate 构造。不重写
// max_workspace_bytes，用户可安全复用/再序列化原 algo，后续 aclblasLtMatmul 的
// workspaceSizeInBytes >= algo->max_workspace_bytes 校验仍以回显 algo 携带的原值为准。
// 「失败不回写 result」指校验失败路径——全部校验失败分支在触达 result 之前返回；
// SecureC 拷贝/清零失败属运行时异常路径（映射 INTERNAL_ERROR，固定 24B/16B 拷贝实际不会失败），
// 返回时 result 可能部分写入——与 FillHeuristicResults 一致。
aclblasStatus_t FillAlgoCheckResult(
    const aclblasLtMatmulAlgo_t& algo, size_t workspaceSize, aclblasLtMatmulHeuristicResult_t* result)
{
    if (!MemcpySSucceeds(&result->algo, sizeof(result->algo), &algo, sizeof(algo))) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    result->workspaceSize = workspaceSize;
    result->state = ACLBLAS_STATUS_SUCCESS;
    result->wavesCount = 1.0f;
    if (CheckedMemsetS(result->reserved, sizeof(result->reserved), sizeof(result->reserved)) !=
        ACLBLAS_STATUS_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}


// ── AlgoCheck Stage helpers（每个 helper 各自打日志，主体只剩 Stage 调用与早退）──

// Stage 0: name the first null argument precisely instead of
// lumping seven faults into one message. Multi-fault priority (AlgoCheck-specific convention,
// the one deliberate divergence from aclblasLtMatmul): argument-shape validity precedes handle
// availability, so (lightHandle == nullptr && operationDesc == nullptr) returns INVALID_VALUE here
// instead of matmul's NOT_INITIALIZED.
bool AlgoCheckArgsAreNonNull(
    const aclblasLtMatmulHeuristicResult_t* result, const aclblasLtMatmulAlgo_t* algo,
    const aclblasLtMatmulDesc_t operationDesc, const aclblasLtMatrixLayout_t Adesc,
    const aclblasLtMatrixLayout_t Bdesc, const aclblasLtMatrixLayout_t Cdesc,
    const aclblasLtMatrixLayout_t Ddesc)
{
    const char* firstNull = nullptr;
    if (result == nullptr) {
        firstNull = "result";
    } else if (algo == nullptr) {
        firstNull = "algo";
    } else if (operationDesc == nullptr) {
        firstNull = "operationDesc";
    } else if (Adesc == nullptr) {
        firstNull = "Adesc";
    } else if (Bdesc == nullptr) {
        firstNull = "Bdesc";
    } else if (Cdesc == nullptr) {
        firstNull = "Cdesc";
    } else if (Ddesc == nullptr) {
        firstNull = "Ddesc";
    }
    if (firstNull != nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck", "%s is null", firstNull);
        return false;
    }
    return true;
}

// Stage 2, dual check — the magic must be valid and a non-zero
// algoId must be decodable; algoId == 0 counts as the unset/default tiling and is accepted
// (same default semantics as InitPackedAlgoForId). Deliberate divergence from the engine,
// which silently falls back to auto-tiling for a valid-magic blob with an undecodable algoId
// (matmul_engine.cpp decodes the magic only): AlgoCheck is a checker and rejects it explicitly.
aclblasStatus_t CheckAlgoBlobStructural(const aclblasLtMatmulAlgo_t* algo, PackedAlgo* packed)
{
    if (!DecodeAlgo(*algo, packed)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck", "algo magic is invalid");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (packed->algoId != 0U) {
        uint8_t decPolicy = 0;
        uint16_t decL1mDiv16 = 0;
        uint16_t decL1nDiv16 = 0;
        uint32_t decL1k = 0;
        uint8_t decSplitK = 0;
        if (!DecodeAlgoIdFields(
                packed->algoId, &decPolicy, &decL1mDiv16, &decL1nDiv16, &decL1k, &decSplitK)) {
            AclBlasLt::LoggerManager::GetInstance().Log(
                ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck", "algoId %u cannot be decoded",
                static_cast<unsigned int>(packed->algoId));
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Stage 3 part 1: pointer mode, scale type, transpose modes — same order
// and same codes as the aclblasLtMatmul chain.
aclblasStatus_t CheckScalarSemantics(const aclblasLtMatmulDescImpl* desc)
{
    // pointerMode must be HOST or DEVICE; out-of-range values such as 2
    // (DEVICE_VECTOR) are rejected.
    if (!IsSupportedPointerMode(desc->pointerMode)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck", "pointerMode %d is not supported",
            static_cast<int>(desc->pointerMode));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Only ACL_FLOAT scalars are parsed.
    if (desc->scaleType != ACL_FLOAT) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck",
            "scaleType %d is not supported, only ACL_FLOAT scalars are parsed", static_cast<int>(desc->scaleType));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
// No conjugate path.
    if (!HasSupportedTransposeModes(desc)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck",
            "transA/transB ACLBLAS_OP_C is not supported (no conjugate path)");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Stage 3 part 2: layout/batch/ranges, dtype whitelist, row-major order,
// MXFP k constraint, MXFP scale pointers — same order and same codes as aclblasLtMatmul.
aclblasStatus_t CheckMatrixSemantics(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* aLayout,
    const aclblasLtMatrixLayoutImpl* bLayout, const aclblasLtMatrixLayoutImpl* cLayout,
    const aclblasLtMatrixLayoutImpl* dLayout, uint64_t m, uint64_t n, uint64_t k)
{
    // Dimensions / ld / batch metadata /
    // uint32 ranges / FP4 ld fold.
    if (!IsMatmulLayoutValid(desc, aLayout, bLayout, cLayout, dLayout, m, n, k)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck",
            "matrix dimensions, leading dimensions, batch metadata, or integer ranges are invalid");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // C/D dtype whitelist, strict pointer-mode rule.
    if (!HasSupportedDtypesForAlgoCheck(desc, aLayout, bLayout, cLayout, dLayout)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck",
            "dtype combination A=%d B=%d C=%d D=%d is not supported by any matmul kernel",
            static_cast<int>(aLayout->type), static_cast<int>(bLayout->type), static_cast<int>(cLayout->type),
            static_cast<int>(dLayout->type));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    // Row-major only.
    if (!HasRowMajorLayouts(aLayout, bLayout, cLayout, dLayout)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck",
            "only ACLBLASLT_ORDER_ROW layouts are supported (A=%d, B=%d, C=%d, D=%d)",
            static_cast<int>(aLayout->order), static_cast<int>(bLayout->order), static_cast<int>(cLayout->order),
            static_cast<int>(dLayout->order));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    // MXFP k must be a non-zero multiple of 32.
    if (ViolatesMxfpKConstraint(aLayout->type, k)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck",
            "MXFP path requires k to be a non-zero multiple of 32 (k=%lu)", static_cast<unsigned long>(k));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // MXFP paths require both scale pointers; non-MXFP paths need no
    // scale and are not checked.
    if (MissingMxfpScalePointers(desc, aLayout->type, bLayout->type)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck",
            "MXFP path requires non-null scaleA and scaleB pointers");
        return ACLBLAS_STATUS_INVALID_VALUE;
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

// Host-side usability check of one algo against one matmul workload (parameter-aligned with
// cuBLAS cublasLtMatmulAlgoCheck). Two-stage validation:
//   P_algo — structural validity of the algo blob (magic + decodable non-zero algoId);
//   P_work — workload support domain, same order and same codes as the aclblasLtMatmul
//            validation chain (pointerMode -> scaleType -> transpose -> m/n/k read -> no-op ->
//            layout -> dtype -> order -> MXFP-K -> MXFP scale pointers), minus the runtime-only
//            items AlgoCheck does not have (alpha/beta, device pointers, workspace, stream).
// On success the algo is echoed back unchanged and workspaceSize reports
// max(the algo's intrinsic demand, algo->max_workspace_bytes); on any validation failure result
// is left untouched. No device query, no kernel launch, no allocation: the handle is checked
// for presence only.
aclblasStatus_t aclblasLtMatmulAlgoCheck(
    aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t operationDesc, aclblasLtMatrixLayout_t Adesc,
    aclblasLtMatrixLayout_t Bdesc, aclblasLtMatrixLayout_t Cdesc, aclblasLtMatrixLayout_t Ddesc,
    const aclblasLtMatmulAlgo_t* algo, aclblasLtMatmulHeuristicResult_t* result)
{
    // Stage 0: argument-shape validity precedes handle availability
    // (the one deliberate divergence from aclblasLtMatmul; see AlgoCheckArgsAreNonNull,
    // which also names the offending argument precisely in its log). Stage 1 (row 2): handle
    // presence only — never dereferenced, no device query; matches aclblasLtMatmul's
    // NOT_INITIALIZED behaviour.
    if (!AlgoCheckArgsAreNonNull(result, algo, operationDesc, Adesc, Bdesc, Cdesc, Ddesc)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lightHandle == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmulAlgoCheck", "lightHandle is null");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    // Stage 2: P_algo — structural validity of the algo blob (
    // see CheckAlgoBlobStructural).
    PackedAlgo packed;
    const aclblasStatus_t algoStatus = CheckAlgoBlobStructural(algo, &packed);
    if (algoStatus != ACLBLAS_STATUS_SUCCESS) {
        return algoStatus;
    }

    // Stage 3: P_work — same order and same codes as the aclblasLtMatmul chain. The scalar
    // checks (CheckScalarSemantics) precede the dimension read; the matrix checks
    // (CheckMatrixSemantics) follow the no-op early
    // return — exactly the aclblasLtMatmul order.
    auto* ALayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
    auto* BLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
    auto* CLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Cdesc);
    auto* DLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Ddesc);
    auto* desc = reinterpret_cast<aclblasLtMatmulDescImpl*>(operationDesc);

    const aclblasStatus_t scalarStatus = CheckScalarSemantics(desc);
    if (scalarStatus != ACLBLAS_STATUS_SUCCESS) {
        return scalarStatus;
    }

    const uint64_t m = DLayout->rows;
    const uint64_t n = DLayout->cols;
    const uint64_t k = (desc->transA == ACLBLAS_OP_N) ? ALayout->cols : ALayout->rows;

    AclBlasLt::LoggerManager::GetInstance().Log(
        ACLBLASLT_LOG_MASK_TRACE, "aclblasLtMatmulAlgoCheck", "m=%lu, n=%lu, k=%lu", static_cast<unsigned long>(m),
        static_cast<unsigned long>(n), static_cast<unsigned long>(k));

    // No-op: m==0 or n==0 is an empty workload; succeed early (placed
    // before layout/dtype/order/MXFP checks, matching aclblasLtMatmul) with workspaceSize
    // forced to 0 — CalculateWorkspaceForAscend's RELU/GELU add-on would report 64KB for m=0,
    // which misrepresents a no-op.
    if (m == 0U || n == 0U) {
        return FillAlgoCheckResult(*algo, 0, result);
    }

    const aclblasStatus_t matrixStatus = CheckMatrixSemantics(desc, ALayout, BLayout, CLayout, DLayout, m, n, k);
    if (matrixStatus != ACLBLAS_STATUS_SUCCESS) {
        return matrixStatus;
    }

    // Stage 4: all checks passed — fill result. splitK comes from
    // the decoded PackedAlgo blob field (packed.splitK), never re-derived from the algoId.
    // The reported workspaceSize is the amount the caller should allocate:
    // max(intrinsic demand, algo->max_workspace_bytes). The intrinsic part (split-K staging +
    // epilogue bias, same source as FillHeuristicResults / BuildRankedCandidates) alone can be
    // 0 for the default algo, while aclblasLtMatmul() separately validates
    // ``workspaceSizeInBytes >= algo->max_workspace_bytes`` (AlgoInit stamps
    // DEFAULT_WORKSPACE_SIZE = 32MB), so reporting only the intrinsic demand made the natural
    // "alloc(result->workspaceSize) then run Matmul" flow fail with INVALID_VALUE. Taking the
    // max matches AlgoGetHeuristic's behaviour (BuildAlgoFromCandidate stamps
    // max_workspace_bytes = intrinsic) and cuBLASLt, whose AlgoCheck reports the allocatable
    // amount. The raw-product staging surcharge (RequiredRawWorkspace, matmul_engine.cpp)
    // still depends on runtime alpha/beta and C/D buffer overlap, which are invisible here.
    const size_t intrinsic = CalculateWorkspaceForAscend(m, n, packed.splitK, desc->epilogue);
    const size_t workspaceSize = intrinsic > algo->max_workspace_bytes ? intrinsic : algo->max_workspace_bytes;
    return FillAlgoCheckResult(*algo, workspaceSize, result);
}

} // extern "C"
