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
 * \file quant_matmul_mx_tiling_swat_host.h
 * \brief Standalone SWAT tiling for MX matmul (ops-blas, no cann-samples dependency).
 *
 * Logic aligned with quant_matmul_mx_tiling_swat.h in matmul_recipes.
 */

#pragma once

#include <algorithm>
#include <cstdint>

#include "quant_matmul_tiling_data.h"

namespace quant_matmul_mx_tiling {

constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t WINDOW_LEN = 4UL;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t FP8_C0_SIZE = 32UL;
constexpr uint64_t BASEK_LIMIT = 4095UL;
constexpr uint64_t DATA_SIZE_L0C = 4UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t TILING_MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t TILING_MXFP_MULTI_BASE_SIZE = 2UL;
constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
constexpr uint64_t SCALER_FACTOR_MIN = 1UL;
constexpr uint64_t SCALER_FACTOR_MAX = 127UL;
constexpr uint64_t MTE2_MIN_LOAD_SIZE = 32768UL;
constexpr uint64_t MTE2_CACHELINE_SIZE = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;
constexpr uint64_t L1_ALIGN_SIZE = 32UL;
constexpr uint64_t L2_ALIGN_SIZE = 128UL;

// Ascend950 default on-chip memory sizes (bytes).
constexpr uint64_t DEFAULT_L1_SIZE = 512UL * 1024UL;
constexpr uint64_t DEFAULT_L0A_SIZE = 64UL * 1024UL;
constexpr uint64_t DEFAULT_L0C_SIZE = 256UL * 1024UL;
constexpr uint64_t DEFAULT_AIC_NUM = 24UL;

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b + static_cast<T>(a % b != 0);
}

template <typename T>
inline T Align(T a, T b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
inline T FloorAlign(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

// MXFP8 element: shape count equals byte size (one byte per element).
template <typename T>
constexpr T GetShapeMxfp8(T size)
{
    return size;
}

template <typename T>
constexpr T GetSizeMxfp8(T shape)
{
    return shape;
}

// MXFP4: shape uses fp4x2 slots; byte size is half of logical element count (see matmul_recipes common_utils).
template <typename T>
constexpr T GetShapeMxfp4(T size)
{
    return size << 1;
}

template <typename T>
constexpr T GetSizeMxfp4(T shape)
{
    return (shape + 1) >> 1;
}

struct QuantMatmulPlatformInfo {
    uint64_t aicNum{DEFAULT_AIC_NUM};
    uint64_t l1Size{DEFAULT_L1_SIZE};
    uint64_t l0aSize{DEFAULT_L0A_SIZE};
    uint64_t l0cSize{DEFAULT_L0C_SIZE};
};

struct QuantMatmulArgs {
    uint64_t m{0UL};
    uint64_t n{0UL};
    uint64_t k{0UL};
    bool transA{false};
    bool transB{true};
};

struct QuantMatmulRunInfo {
    uint64_t baseM{0UL};
    uint64_t baseN{0UL};
    uint64_t baseK{0UL};
    uint64_t stepKa{0UL};
    uint64_t stepKb{0UL};
    uint64_t depthA1{0UL};
    uint64_t depthB1{0UL};
    uint64_t dbL0c{0UL};
    uint64_t mBlockCnt{0UL};
    uint64_t nBlockCnt{0UL};
    uint64_t totalBlockCnt{0UL};
    uint64_t mTailTile{1UL};
    uint64_t nTailTile{1UL};
    uint64_t mTailSize{0UL};
    uint64_t nTailSize{0UL};
    uint64_t tailBlockCnt{0UL};
    uint64_t mBaseTailSplitCnt{1UL};
    uint64_t mTailMain{0UL};
    uint64_t nBaseTailSplitCnt{1UL};
    uint64_t nTailMain{0UL};
    uint64_t scaleFactorA{0UL};
    uint64_t scaleFactorB{0UL};
};

class QuantMatmulMxfp8TilingSwat {
public:
    void GetTilingData(
        uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, uint64_t aicNum,
        QuantMatmulTilingData& tilingData)
    {
        args_ = {};
        platformInfo_ = {};
        runInfo_ = {};
        platformInfo_.aicNum = aicNum > 0UL ? aicNum : DEFAULT_AIC_NUM;
        args_.m = m;
        args_.n = n;
        args_.k = k;
        args_.transA = transA;
        args_.transB = transB;

        CalcBasicBlock();
        OptimizeEdgeBasicBlock();
        CalcTailBasicBlock();
        CalcPathSpecificL1();

        const uint32_t scaleKL1 = CalcScaleKL1();
        BuildTilingData(tilingData, scaleKL1, static_cast<uint8_t>(DB_SIZE));
    }

private:
    QuantMatmulArgs args_{};
    QuantMatmulPlatformInfo platformInfo_{};
    QuantMatmulRunInfo runInfo_{};

    uint32_t CalcScaleKL1() const
    {
        return static_cast<uint32_t>(std::min(
            runInfo_.scaleFactorA * runInfo_.stepKa * runInfo_.baseK,
            runInfo_.scaleFactorB * runInfo_.stepKb * runInfo_.baseK));
    }

    void BuildTilingData(QuantMatmulTilingData& tilingData, uint32_t scaleKL1, uint8_t nBufferNum) const
    {
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.mTailTile = static_cast<uint32_t>(runInfo_.mTailTile);
        tilingData.nTailTile = static_cast<uint32_t>(runInfo_.nTailTile);
        tilingData.mBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.mBaseTailSplitCnt);
        tilingData.nBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.nBaseTailSplitCnt);
        tilingData.mTailMain = static_cast<uint32_t>(runInfo_.mTailMain);
        tilingData.nTailMain = static_cast<uint32_t>(runInfo_.nTailMain);
        tilingData.usedCoreNum = static_cast<uint32_t>(
            (runInfo_.totalBlockCnt > 1UL || runInfo_.tailBlockCnt == 0UL) ?
                platformInfo_.aicNum :
                runInfo_.tailBlockCnt * runInfo_.mTailTile * runInfo_.nTailTile);
        tilingData.dbL0c = static_cast<uint8_t>(runInfo_.dbL0c);
        tilingData.scaleKL1 = scaleKL1;
        tilingData.stepK = static_cast<uint8_t>(std::min(runInfo_.stepKa, runInfo_.stepKb));
        tilingData.nBufferNum = nBufferNum;
    }

    void CalcTailBasicBlock()
    {
        if (runInfo_.tailBlockCnt == 0UL) {
            return;
        }

        uint64_t mTile = 1UL;
        uint64_t nTile = 1UL;
        uint64_t preSplit = 1UL;
        uint64_t secSplit = 1UL;
        uint64_t& preSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? mTile : nTile;
        uint64_t& secSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? nTile : mTile;
        const uint64_t tileMax = platformInfo_.aicNum / runInfo_.tailBlockCnt;
        const uint64_t mTileMax = std::min(tileMax, CeilDiv(runInfo_.baseM, CUBE_BLOCK));
        const uint64_t nTileMax = std::min(tileMax, CeilDiv(runInfo_.baseN, CUBE_BLOCK));
        const uint64_t preSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? mTileMax : nTileMax;
        const uint64_t secSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? nTileMax : mTileMax;
        while ((CalUsedCoreNum(runInfo_, preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) ||
               (CalUsedCoreNum(runInfo_, preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax)) {
            if (CalUsedCoreNum(runInfo_, preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) {
                preSplitValid = ++preSplit;
            }
            if (CalUsedCoreNum(runInfo_, preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax) {
                secSplitValid = ++secSplit;
            }
        }

        runInfo_.mTailTile = mTile;
        runInfo_.nTailTile = nTile;
    }

    void CalcPathSpecificL1()
    {
        const uint64_t baseASize = GetSizeMxfp8(runInfo_.baseM * runInfo_.baseK);
        const uint64_t baseBSize = GetSizeMxfp8(runInfo_.baseN * runInfo_.baseK);

        const uint64_t baseScaleASize =
            Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE) * runInfo_.baseM;
        const uint64_t baseScaleBSize =
            Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE) * runInfo_.baseN;
        const uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        const uint64_t depthInit = GetDepthA1B1(runInfo_, platformInfo_.l1Size, baseL1Size, 1UL);
        const uint64_t leftL1SizeByDepthInit = platformInfo_.l1Size - depthInit * baseL1Size;
        const uint64_t depthASec =
            GetDepthA1B1(runInfo_, leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
        const uint64_t depthBSec =
            GetDepthA1B1(runInfo_, leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
        runInfo_.depthA1 = std::max(depthASec, depthBSec);
        runInfo_.depthB1 = runInfo_.depthA1;
        if (runInfo_.depthA1 * baseL1Size > platformInfo_.l1Size) {
            runInfo_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
            runInfo_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
        }
        CalStepKs(args_, runInfo_);
        CalScaleFactors(args_, platformInfo_, runInfo_, baseASize, baseBSize, baseScaleASize, baseScaleBSize);
    }

    void AdjustBasicBlock()
    {
        const uint64_t baseMAlignNum = args_.transA ? GetShapeMxfp8(L2_ALIGN_SIZE) : CUBE_BLOCK;
        const uint64_t baseNAlignNum = args_.transB ? CUBE_BLOCK : GetShapeMxfp8(L2_ALIGN_SIZE);
        uint64_t baseKAlignNum =
            (args_.transA && !args_.transB) ? GetShapeMxfp8(FP8_C0_SIZE) : GetShapeMxfp8(L2_ALIGN_SIZE);
        if (args_.transA || !args_.transB) {
            baseKAlignNum = GetShapeMxfp8(TILING_MXFP_DIVISOR_SIZE);
        }
        const uint64_t mMaxtile = CeilDiv(args_.m, baseMAlignNum);
        const uint64_t nMaxtile = CeilDiv(args_.n, baseNAlignNum);
        uint64_t tempBaseM = runInfo_.baseM;
        uint64_t tempBaseN = runInfo_.baseN;
        const uint64_t coreNumMN = platformInfo_.aicNum;

        if (mMaxtile * nMaxtile >= coreNumMN || (!args_.transA && args_.transB)) {
            uint64_t mCnt = CeilDiv(args_.m, runInfo_.baseM);
            uint64_t nCnt = CeilDiv(args_.n, runInfo_.baseN);

            if (mMaxtile > nMaxtile) {
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
                nCnt = CeilDiv(args_.n, tempBaseN);
                mCnt = platformInfo_.aicNum / nCnt;
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
            } else {
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
                mCnt = CeilDiv(args_.m, tempBaseM);
                nCnt = platformInfo_.aicNum / mCnt;
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
            }

            while (tempBaseN > tempBaseM * BASEM_BASEN_RATIO && nCnt < platformInfo_.aicNum / NUM_TWO &&
                   tempBaseN != baseNAlignNum) {
                nCnt = nCnt * NUM_TWO;
                mCnt = platformInfo_.aicNum / nCnt;
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
                mCnt = CeilDiv(args_.m, tempBaseM);
                nCnt = CeilDiv(args_.n, tempBaseN);
            }
            while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCnt < platformInfo_.aicNum / NUM_TWO &&
                   tempBaseM != baseMAlignNum) {
                mCnt = mCnt * NUM_TWO;
                nCnt = platformInfo_.aicNum / mCnt;
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
                mCnt = CeilDiv(args_.m, tempBaseM);
                nCnt = CeilDiv(args_.n, tempBaseN);
            }

            const uint64_t kAlignValue = Align(args_.k, baseKAlignNum);
            uint64_t kMaxValue = GetShapeMxfp8(platformInfo_.l0aSize / DB_SIZE) / std::max(tempBaseM, tempBaseN);
            kMaxValue = FloorAlign(kMaxValue, baseKAlignNum);
            if (kMaxValue >= baseKAlignNum) {
                runInfo_.baseM = tempBaseM;
                runInfo_.baseN = tempBaseN;
                runInfo_.baseK = std::min(kAlignValue, kMaxValue);
                runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT ? Align(runInfo_.baseK / NUM_TWO, BASIC_BLOCK_SIZE_256) :
                                                                runInfo_.baseK;
            }
        }
    }

    void CalcBasicBlock()
    {
        runInfo_.baseM = std::min(args_.m, BASIC_BLOCK_SIZE_256);
        runInfo_.baseM =
            !args_.transA ? Align(runInfo_.baseM, CUBE_BLOCK) : Align(runInfo_.baseM, GetShapeMxfp8(L1_ALIGN_SIZE));
        runInfo_.baseN = std::min(args_.n, BASIC_BLOCK_SIZE_256);
        runInfo_.baseN =
            args_.transB ? Align(runInfo_.baseN, CUBE_BLOCK) : Align(runInfo_.baseN, GetShapeMxfp8(L1_ALIGN_SIZE));
        runInfo_.baseK = Align(std::min(args_.k, BASIC_BLOCK_SIZE_128), TILING_MXFP_DIVISOR_SIZE);

        const uint64_t blockNum = CeilDiv(args_.m, runInfo_.baseM) * CeilDiv(args_.n, runInfo_.baseN);
        if (blockNum < platformInfo_.aicNum) {
            AdjustBasicBlock();
        }

        if (runInfo_.baseM == 0UL || runInfo_.baseN == 0UL || runInfo_.baseK == 0UL) {
            runInfo_.baseM = std::max(runInfo_.baseM, CUBE_BLOCK);
            runInfo_.baseN = std::max(runInfo_.baseN, CUBE_BLOCK);
            runInfo_.baseK = std::max(runInfo_.baseK, TILING_MXFP_DIVISOR_SIZE);
        }

        runInfo_.mBlockCnt = CeilDiv(args_.m, runInfo_.baseM);
        runInfo_.nBlockCnt = CeilDiv(args_.n, runInfo_.baseN);
        runInfo_.totalBlockCnt = runInfo_.mBlockCnt * runInfo_.nBlockCnt;
        runInfo_.tailBlockCnt = runInfo_.totalBlockCnt % platformInfo_.aicNum;
        runInfo_.mTailSize = args_.m - (runInfo_.mBlockCnt - 1UL) * runInfo_.baseM;
        runInfo_.nTailSize = args_.n - (runInfo_.nBlockCnt - 1UL) * runInfo_.baseN;
        runInfo_.dbL0c =
            runInfo_.baseM * runInfo_.baseN * DATA_SIZE_L0C * DB_SIZE <= platformInfo_.l0cSize ? DB_SIZE : 1U;
    }

    void OptimizeEdgeBasicBlock()
    {
        if (runInfo_.mBlockCnt == 1UL && runInfo_.nBlockCnt == 1UL) {
            return;
        }

        const bool isInnerAxisAlign = GetSizeMxfp8(args_.k) % MTE2_CACHELINE_SIZE == 0UL;

        const uint64_t mTailSize = args_.m % runInfo_.baseM;
        if (runInfo_.mBlockCnt > 1UL && mTailSize > 0UL && !args_.transA && isInnerAxisAlign) {
            const uint64_t baseTailCntMax =
                std::min((runInfo_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.mBlockCnt);
            const uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.mBlockCnt);
            const uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
            const uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
            uint64_t mergeWindowNum = 1UL;

            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
                 mergeLen += windowSize, ++mergeWindowNum) {
                const uint64_t newTailMain =
                    Align(CeilDiv((mergeLen * runInfo_.baseM + mTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                const uint64_t curPerf =
                    (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.mTailMain = newTailMain;
                    runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }

        const uint64_t nTailSize = args_.n % runInfo_.baseN;
        if (runInfo_.nBlockCnt > 1UL && nTailSize > 0UL && args_.transB && isInnerAxisAlign) {
            const uint64_t baseTailCntMax =
                std::min((runInfo_.baseN - nTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.nBlockCnt);
            const uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.nBlockCnt);
            const uint64_t mainWindowNum = runInfo_.nBlockCnt / windowSize - 1UL;
            const uint64_t tailWindowSize = runInfo_.nBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseN;
            uint64_t mergeWindowNum = 1UL;

            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
                 mergeLen += windowSize, ++mergeWindowNum) {
                const uint64_t newTailMain =
                    Align(CeilDiv((mergeLen * runInfo_.baseN + nTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                const uint64_t curPerf =
                    (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseN + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.nTailMain = newTailMain;
                    runInfo_.nBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
    }

    static uint64_t CalUsedCoreNum(const QuantMatmulRunInfo& runInfo, uint64_t mTile, uint64_t nTile)
    {
        return mTile * nTile * runInfo.tailBlockCnt;
    }

    static uint64_t GetDepthA1B1(
        const QuantMatmulRunInfo& runInfo, uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
    {
        if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE) {
            return depthInit;
        }
        uint64_t depthScale = leftSize / perDepthSize;
        if (depthInit > 1UL) {
            const uint64_t baseKSize = GetSizeMxfp8(runInfo.baseK);
            while ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
                   (depthScale * baseKSize) > BASIC_BLOCK_SIZE_512) {
                depthScale -= 1UL;
            }
            if ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
                (depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256) {
                depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
            }
            depthScale = std::max(depthScale, 1UL);
        } else {
            constexpr uint64_t SCALE_INDEX = 2UL;
            depthScale = 1UL;
            while (depthScale * perDepthSize < leftSize) {
                depthScale *= SCALE_INDEX;
            }
            depthScale = depthScale == 1UL ? depthScale : depthScale / SCALE_INDEX;
        }
        return depthInit * depthScale;
    }

    static void CalStepKs(const QuantMatmulArgs& args, QuantMatmulRunInfo& runInfo)
    {
        runInfo.stepKa = runInfo.depthA1 / DB_SIZE;
        runInfo.stepKb = runInfo.depthB1 / DB_SIZE;

        if (runInfo.stepKa * runInfo.baseK > args.k) {
            runInfo.stepKa = CeilDiv(args.k, runInfo.baseK);
        }

        if (runInfo.stepKb * runInfo.baseK > args.k) {
            runInfo.stepKb = CeilDiv(args.k, runInfo.baseK);
        }

        if (runInfo.stepKa > runInfo.stepKb) {
            runInfo.stepKa = runInfo.stepKa / runInfo.stepKb * runInfo.stepKb;
        }
        if (runInfo.stepKb > runInfo.stepKa) {
            runInfo.stepKb = runInfo.stepKb / runInfo.stepKa * runInfo.stepKa;
        }

        runInfo.stepKa = std::min(runInfo.stepKa, 4UL);
        runInfo.stepKb = std::min(runInfo.stepKb, 4UL);

        runInfo.depthA1 = runInfo.stepKa * DB_SIZE;
        runInfo.depthB1 = runInfo.stepKb * DB_SIZE;
    }

    static void CalScaleFactors(
        const QuantMatmulArgs& args, const QuantMatmulPlatformInfo& platformInfo, QuantMatmulRunInfo& runInfo,
        uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize)
    {
        const uint64_t scaleFactorAMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleASize, SCALER_FACTOR_MAX);
        const uint64_t scaleFactorBMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleBSize, SCALER_FACTOR_MAX);
        // issue #584: k=0 时 stepK 会被归零，分母为 0 的整数除法在主机端触发 SIGFPE，
        // 对分母做零值防护（k=0 已在引擎入口显式拒绝，本防护为纵深防御）。
        const uint64_t denomA = runInfo.stepKa * runInfo.baseK;
        const uint64_t denomB = runInfo.stepKb * runInfo.baseK;
        uint64_t scaleFactorA = (denomA > 0UL) ? args.k / denomA : 0UL;
        uint64_t scaleFactorB = (denomB > 0UL) ? args.k / denomB : 0UL;
        runInfo.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
        runInfo.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
        runInfo.scaleFactorA = std::min(scaleFactorAMax, runInfo.scaleFactorA);
        runInfo.scaleFactorB = std::min(scaleFactorBMax, runInfo.scaleFactorB);

        uint64_t leftL1Size = platformInfo.l1Size - (runInfo.depthA1 * baseASize + runInfo.depthB1 * baseBSize);
        const uint64_t scaleInit = leftL1Size / (runInfo.depthA1 * baseScaleASize + runInfo.depthB1 * baseScaleBSize);
        if (runInfo.scaleFactorA <= scaleInit && runInfo.scaleFactorB > scaleInit) {
            leftL1Size -= runInfo.scaleFactorA * runInfo.depthA1 * baseScaleASize;
            runInfo.scaleFactorB = std::min(leftL1Size / (runInfo.depthB1 * baseScaleBSize), runInfo.scaleFactorB);
        } else if (runInfo.scaleFactorB <= scaleInit && runInfo.scaleFactorA > scaleInit) {
            leftL1Size -= runInfo.scaleFactorB * runInfo.depthB1 * baseScaleBSize;
            runInfo.scaleFactorA = std::min(leftL1Size / (runInfo.depthA1 * baseScaleASize), runInfo.scaleFactorA);
        } else if (runInfo.scaleFactorA > scaleInit && runInfo.scaleFactorB > scaleInit) {
            leftL1Size -= scaleInit * runInfo.depthB1 * baseScaleBSize + scaleInit * runInfo.depthA1 * baseScaleASize;
            const uint64_t scaleASec =
                std::min(leftL1Size / (runInfo.depthA1 * baseScaleASize), runInfo.scaleFactorA - scaleInit);
            const uint64_t scaleBSec =
                std::min(leftL1Size / (runInfo.depthB1 * baseScaleBSize), runInfo.scaleFactorB - scaleInit);
            runInfo.scaleFactorA = scaleASec >= scaleBSec ? scaleASec + scaleInit : scaleInit;
            runInfo.scaleFactorB = scaleASec < scaleBSec ? scaleBSec + scaleInit : scaleInit;
        }
    }
};

inline void quant_matmul_mxfp8_swat_get_tiling(
    uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, uint64_t aicNum, QuantMatmulTilingData& tilingData)
{
    QuantMatmulMxfp8TilingSwat engine;
    engine.GetTilingData(m, n, k, transA, transB, aicNum, tilingData);
}

class QuantMatmulMxfp4TilingSwat {
public:
    void GetTilingData(
        uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, uint64_t aicNum,
        QuantMatmulTilingData& tilingData)
    {
        args_ = {};
        platformInfo_ = {};
        runInfo_ = {};
        platformInfo_.aicNum = aicNum > 0UL ? aicNum : DEFAULT_AIC_NUM;
        args_.m = m;
        args_.n = n;
        args_.k = k;
        args_.transA = transA;
        args_.transB = transB;

        CalcBasicBlock();
        OptimizeEdgeBasicBlock();
        CalcTailBasicBlock();
        CalcPathSpecificL1();

        const uint32_t scaleKL1 = CalcScaleKL1();
        BuildTilingData(tilingData, scaleKL1, static_cast<uint8_t>(DB_SIZE));
    }

private:
    QuantMatmulArgs args_{};
    QuantMatmulPlatformInfo platformInfo_{};
    QuantMatmulRunInfo runInfo_{};

    uint32_t CalcScaleKL1() const
    {
        return static_cast<uint32_t>(std::min(
            runInfo_.scaleFactorA * runInfo_.stepKa * runInfo_.baseK,
            runInfo_.scaleFactorB * runInfo_.stepKb * runInfo_.baseK));
    }

    void BuildTilingData(QuantMatmulTilingData& tilingData, uint32_t scaleKL1, uint8_t nBufferNum) const
    {
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.mTailTile = static_cast<uint32_t>(runInfo_.mTailTile);
        tilingData.nTailTile = static_cast<uint32_t>(runInfo_.nTailTile);
        tilingData.mBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.mBaseTailSplitCnt);
        tilingData.nBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.nBaseTailSplitCnt);
        tilingData.mTailMain = static_cast<uint32_t>(runInfo_.mTailMain);
        tilingData.nTailMain = static_cast<uint32_t>(runInfo_.nTailMain);
        tilingData.usedCoreNum = static_cast<uint32_t>(
            (runInfo_.totalBlockCnt > 1UL || runInfo_.tailBlockCnt == 0UL) ?
                platformInfo_.aicNum :
                runInfo_.tailBlockCnt * runInfo_.mTailTile * runInfo_.nTailTile);
        tilingData.dbL0c = static_cast<uint8_t>(runInfo_.dbL0c);
        tilingData.scaleKL1 = scaleKL1;
        tilingData.stepK = static_cast<uint8_t>(std::min(runInfo_.stepKa, runInfo_.stepKb));
        tilingData.nBufferNum = nBufferNum;
    }

    void CalcTailBasicBlock()
    {
        if (runInfo_.tailBlockCnt == 0UL) {
            return;
        }

        uint64_t mTile = 1UL;
        uint64_t nTile = 1UL;
        uint64_t preSplit = 1UL;
        uint64_t secSplit = 1UL;
        uint64_t& preSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? mTile : nTile;
        uint64_t& secSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? nTile : mTile;
        const uint64_t tileMax = platformInfo_.aicNum / runInfo_.tailBlockCnt;
        const uint64_t mTileMax = std::min(tileMax, CeilDiv(runInfo_.baseM, CUBE_BLOCK));
        const uint64_t nTileMax = std::min(tileMax, CeilDiv(runInfo_.baseN, CUBE_BLOCK));
        const uint64_t preSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? mTileMax : nTileMax;
        const uint64_t secSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? nTileMax : mTileMax;
        while ((CalUsedCoreNum(runInfo_, preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) ||
               (CalUsedCoreNum(runInfo_, preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax)) {
            if (CalUsedCoreNum(runInfo_, preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) {
                preSplitValid = ++preSplit;
            }
            if (CalUsedCoreNum(runInfo_, preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax) {
                secSplitValid = ++secSplit;
            }
        }

        runInfo_.mTailTile = mTile;
        runInfo_.nTailTile = nTile;
    }

    void CalcPathSpecificL1()
    {
        const uint64_t baseASize = GetSizeMxfp4(runInfo_.baseM * runInfo_.baseK);
        const uint64_t baseBSize = GetSizeMxfp4(runInfo_.baseN * runInfo_.baseK);

        const uint64_t baseScaleASize =
            Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE) * runInfo_.baseM;
        const uint64_t baseScaleBSize =
            Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE) * runInfo_.baseN;
        const uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        const uint64_t depthInit = GetDepthA1B1(runInfo_, platformInfo_.l1Size, baseL1Size, 1UL);
        const uint64_t leftL1SizeByDepthInit = platformInfo_.l1Size - depthInit * baseL1Size;
        const uint64_t depthASec =
            GetDepthA1B1(runInfo_, leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
        const uint64_t depthBSec =
            GetDepthA1B1(runInfo_, leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
        runInfo_.depthA1 = std::max(depthASec, depthBSec);
        runInfo_.depthB1 = runInfo_.depthA1;
        if (runInfo_.depthA1 * baseL1Size > platformInfo_.l1Size) {
            runInfo_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
            runInfo_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
        }
        CalStepKs(args_, runInfo_);
        CalScaleFactors(args_, platformInfo_, runInfo_, baseASize, baseBSize, baseScaleASize, baseScaleBSize);
    }

    void AdjustBasicBlock()
    {
        const uint64_t baseMAlignNum = args_.transA ? GetShapeMxfp4(L2_ALIGN_SIZE) : CUBE_BLOCK;
        const uint64_t baseNAlignNum = args_.transB ? CUBE_BLOCK : GetShapeMxfp4(L2_ALIGN_SIZE);
        uint64_t baseKAlignNum =
            (args_.transA && !args_.transB) ? GetShapeMxfp4(FP8_C0_SIZE) : GetShapeMxfp4(L2_ALIGN_SIZE);
        if (args_.transA || !args_.transB) {
            baseKAlignNum = GetShapeMxfp4(TILING_MXFP_DIVISOR_SIZE);
        }
        const uint64_t mMaxtile = CeilDiv(args_.m, baseMAlignNum);
        const uint64_t nMaxtile = CeilDiv(args_.n, baseNAlignNum);
        uint64_t tempBaseM = runInfo_.baseM;
        uint64_t tempBaseN = runInfo_.baseN;
        const uint64_t coreNumMN = platformInfo_.aicNum;

        if (mMaxtile * nMaxtile >= coreNumMN || (!args_.transA && args_.transB)) {
            uint64_t mCnt = CeilDiv(args_.m, runInfo_.baseM);
            uint64_t nCnt = CeilDiv(args_.n, runInfo_.baseN);

            if (mMaxtile > nMaxtile) {
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
                nCnt = CeilDiv(args_.n, tempBaseN);
                mCnt = platformInfo_.aicNum / nCnt;
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
            } else {
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
                mCnt = CeilDiv(args_.m, tempBaseM);
                nCnt = platformInfo_.aicNum / mCnt;
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
            }

            while (tempBaseN > tempBaseM * BASEM_BASEN_RATIO && nCnt < platformInfo_.aicNum / NUM_TWO &&
                   tempBaseN != baseNAlignNum) {
                nCnt = nCnt * NUM_TWO;
                mCnt = platformInfo_.aicNum / nCnt;
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
                mCnt = CeilDiv(args_.m, tempBaseM);
                nCnt = CeilDiv(args_.n, tempBaseN);
            }
            while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCnt < platformInfo_.aicNum / NUM_TWO &&
                   tempBaseM != baseMAlignNum) {
                mCnt = mCnt * NUM_TWO;
                nCnt = platformInfo_.aicNum / mCnt;
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
                mCnt = CeilDiv(args_.m, tempBaseM);
                nCnt = CeilDiv(args_.n, tempBaseN);
            }

            const uint64_t kAlignValue = Align(args_.k, baseKAlignNum);
            uint64_t kMaxValue = GetShapeMxfp4(platformInfo_.l0aSize / DB_SIZE) / std::max(tempBaseM, tempBaseN);
            kMaxValue = FloorAlign(kMaxValue, baseKAlignNum);
            if (kMaxValue >= baseKAlignNum) {
                runInfo_.baseM = tempBaseM;
                runInfo_.baseN = tempBaseN;
                runInfo_.baseK = std::min(kAlignValue, kMaxValue);
                runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT ? Align(runInfo_.baseK / NUM_TWO, BASIC_BLOCK_SIZE_256) :
                                                                runInfo_.baseK;
            }
        }
    }

    void CalcBasicBlock()
    {
        runInfo_.baseM = std::min(args_.m, BASIC_BLOCK_SIZE_256);
        runInfo_.baseM =
            !args_.transA ? Align(runInfo_.baseM, CUBE_BLOCK) : Align(runInfo_.baseM, GetShapeMxfp4(L1_ALIGN_SIZE));
        runInfo_.baseN = std::min(args_.n, BASIC_BLOCK_SIZE_256);
        runInfo_.baseN =
            args_.transB ? Align(runInfo_.baseN, CUBE_BLOCK) : Align(runInfo_.baseN, GetShapeMxfp4(L1_ALIGN_SIZE));
        runInfo_.baseK = Align(std::min(args_.k, BASIC_BLOCK_SIZE_256), TILING_MXFP_DIVISOR_SIZE);

        const uint64_t blockNum = CeilDiv(args_.m, runInfo_.baseM) * CeilDiv(args_.n, runInfo_.baseN);
        if (blockNum < platformInfo_.aicNum) {
            AdjustBasicBlock();
        }

        if (runInfo_.baseM == 0UL || runInfo_.baseN == 0UL || runInfo_.baseK == 0UL) {
            runInfo_.baseM = std::max(runInfo_.baseM, CUBE_BLOCK);
            runInfo_.baseN = std::max(runInfo_.baseN, CUBE_BLOCK);
            runInfo_.baseK = std::max(runInfo_.baseK, TILING_MXFP_DIVISOR_SIZE);
        }

        runInfo_.mBlockCnt = CeilDiv(args_.m, runInfo_.baseM);
        runInfo_.nBlockCnt = CeilDiv(args_.n, runInfo_.baseN);
        runInfo_.totalBlockCnt = runInfo_.mBlockCnt * runInfo_.nBlockCnt;
        runInfo_.tailBlockCnt = runInfo_.totalBlockCnt % platformInfo_.aicNum;
        runInfo_.mTailSize = args_.m - (runInfo_.mBlockCnt - 1UL) * runInfo_.baseM;
        runInfo_.nTailSize = args_.n - (runInfo_.nBlockCnt - 1UL) * runInfo_.baseN;
        runInfo_.dbL0c =
            runInfo_.baseM * runInfo_.baseN * DATA_SIZE_L0C * DB_SIZE <= platformInfo_.l0cSize ? DB_SIZE : 1U;
    }

    void OptimizeEdgeBasicBlock()
    {
        if (runInfo_.mBlockCnt == 1UL && runInfo_.nBlockCnt == 1UL) {
            return;
        }

        const bool isInnerAxisAlign = GetSizeMxfp4(args_.k) % MTE2_CACHELINE_SIZE == 0UL;

        const uint64_t mTailSize = args_.m % runInfo_.baseM;
        if (runInfo_.mBlockCnt > 1UL && mTailSize > 0UL && !args_.transA && isInnerAxisAlign) {
            const uint64_t baseTailCntMax =
                std::min((runInfo_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.mBlockCnt);
            const uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.mBlockCnt);
            const uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
            const uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
            uint64_t mergeWindowNum = 1UL;

            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
                 mergeLen += windowSize, ++mergeWindowNum) {
                const uint64_t newTailMain =
                    Align(CeilDiv((mergeLen * runInfo_.baseM + mTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                const uint64_t curPerf =
                    (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.mTailMain = newTailMain;
                    runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }

        const uint64_t nTailSize = args_.n % runInfo_.baseN;
        if (runInfo_.nBlockCnt > 1UL && nTailSize > 0UL && args_.transB && isInnerAxisAlign) {
            const uint64_t baseTailCntMax =
                std::min((runInfo_.baseN - nTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.nBlockCnt);
            const uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.nBlockCnt);
            const uint64_t mainWindowNum = runInfo_.nBlockCnt / windowSize - 1UL;
            const uint64_t tailWindowSize = runInfo_.nBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseN;
            uint64_t mergeWindowNum = 1UL;

            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
                 mergeLen += windowSize, ++mergeWindowNum) {
                const uint64_t newTailMain =
                    Align(CeilDiv((mergeLen * runInfo_.baseN + nTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                const uint64_t curPerf =
                    (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseN + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.nTailMain = newTailMain;
                    runInfo_.nBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
    }

    static uint64_t CalUsedCoreNum(const QuantMatmulRunInfo& runInfo, uint64_t mTile, uint64_t nTile)
    {
        return mTile * nTile * runInfo.tailBlockCnt;
    }

    static uint64_t GetDepthA1B1(
        const QuantMatmulRunInfo& runInfo, uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
    {
        if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE) {
            return depthInit;
        }
        uint64_t depthScale = leftSize / perDepthSize;
        if (depthInit > 1UL) {
            const uint64_t baseKSize = GetSizeMxfp4(runInfo.baseK);
            while ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
                   (depthScale * baseKSize) > BASIC_BLOCK_SIZE_512) {
                depthScale -= 1UL;
            }
            if ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
                (depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256) {
                depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
            }
            depthScale = std::max(depthScale, 1UL);
        } else {
            constexpr uint64_t SCALE_INDEX = 2UL;
            depthScale = 1UL;
            while (depthScale * perDepthSize < leftSize) {
                depthScale *= SCALE_INDEX;
            }
            depthScale = depthScale == 1UL ? depthScale : depthScale / SCALE_INDEX;
        }
        return depthInit * depthScale;
    }

    static void CalStepKs(const QuantMatmulArgs& args, QuantMatmulRunInfo& runInfo)
    {
        runInfo.stepKa = runInfo.depthA1 / DB_SIZE;
        runInfo.stepKb = runInfo.depthB1 / DB_SIZE;

        if (runInfo.stepKa * runInfo.baseK > args.k) {
            runInfo.stepKa = CeilDiv(args.k, runInfo.baseK);
        }

        if (runInfo.stepKb * runInfo.baseK > args.k) {
            runInfo.stepKb = CeilDiv(args.k, runInfo.baseK);
        }

        if (runInfo.stepKa > runInfo.stepKb) {
            runInfo.stepKa = runInfo.stepKa / runInfo.stepKb * runInfo.stepKb;
        }
        if (runInfo.stepKb > runInfo.stepKa) {
            runInfo.stepKb = runInfo.stepKb / runInfo.stepKa * runInfo.stepKa;
        }

        runInfo.stepKa = std::min(runInfo.stepKa, 4UL);
        runInfo.stepKb = std::min(runInfo.stepKb, 4UL);

        runInfo.depthA1 = runInfo.stepKa * DB_SIZE;
        runInfo.depthB1 = runInfo.stepKb * DB_SIZE;
    }

    static void CalScaleFactors(
        const QuantMatmulArgs& args, const QuantMatmulPlatformInfo& platformInfo, QuantMatmulRunInfo& runInfo,
        uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize)
    {
        const uint64_t scaleFactorAMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleASize, SCALER_FACTOR_MAX);
        const uint64_t scaleFactorBMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleBSize, SCALER_FACTOR_MAX);
        // issue #584: k=0 时 stepK 会被归零，分母为 0 的整数除法在主机端触发 SIGFPE，
        // 对分母做零值防护（k=0 已在引擎入口显式拒绝，本防护为纵深防御）。
        const uint64_t denomA = runInfo.stepKa * runInfo.baseK;
        const uint64_t denomB = runInfo.stepKb * runInfo.baseK;
        uint64_t scaleFactorA = (denomA > 0UL) ? args.k / denomA : 0UL;
        uint64_t scaleFactorB = (denomB > 0UL) ? args.k / denomB : 0UL;
        runInfo.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
        runInfo.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
        runInfo.scaleFactorA = std::min(scaleFactorAMax, runInfo.scaleFactorA);
        runInfo.scaleFactorB = std::min(scaleFactorBMax, runInfo.scaleFactorB);

        uint64_t leftL1Size = platformInfo.l1Size - (runInfo.depthA1 * baseASize + runInfo.depthB1 * baseBSize);
        const uint64_t scaleInit = leftL1Size / (runInfo.depthA1 * baseScaleASize + runInfo.depthB1 * baseScaleBSize);
        if (runInfo.scaleFactorA <= scaleInit && runInfo.scaleFactorB > scaleInit) {
            leftL1Size -= runInfo.scaleFactorA * runInfo.depthA1 * baseScaleASize;
            runInfo.scaleFactorB = std::min(leftL1Size / (runInfo.depthB1 * baseScaleBSize), runInfo.scaleFactorB);
        } else if (runInfo.scaleFactorB <= scaleInit && runInfo.scaleFactorA > scaleInit) {
            leftL1Size -= runInfo.scaleFactorB * runInfo.depthB1 * baseScaleBSize;
            runInfo.scaleFactorA = std::min(leftL1Size / (runInfo.depthA1 * baseScaleASize), runInfo.scaleFactorA);
        } else if (runInfo.scaleFactorA > scaleInit && runInfo.scaleFactorB > scaleInit) {
            leftL1Size -= scaleInit * runInfo.depthB1 * baseScaleBSize + scaleInit * runInfo.depthA1 * baseScaleASize;
            const uint64_t scaleASec =
                std::min(leftL1Size / (runInfo.depthA1 * baseScaleASize), runInfo.scaleFactorA - scaleInit);
            const uint64_t scaleBSec =
                std::min(leftL1Size / (runInfo.depthB1 * baseScaleBSize), runInfo.scaleFactorB - scaleInit);
            runInfo.scaleFactorA = scaleASec >= scaleBSec ? scaleASec + scaleInit : scaleInit;
            runInfo.scaleFactorB = scaleASec < scaleBSec ? scaleBSec + scaleInit : scaleInit;
        }
    }
};

inline void quant_matmul_mxfp4_swat_get_tiling(
    uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, uint64_t aicNum, QuantMatmulTilingData& tilingData)
{
    QuantMatmulMxfp4TilingSwat engine;
    engine.GetTilingData(m, n, k, transA, transB, aicNum, tilingData);
}

} // namespace quant_matmul_mx_tiling
