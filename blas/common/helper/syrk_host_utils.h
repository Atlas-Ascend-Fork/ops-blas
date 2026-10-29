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

#include "host_utils.h"

static inline uint32_t GetUsedAivCoreNum(uint32_t n, uint32_t aivCoreNum)
{
    return std::max<uint32_t>(std::min<uint32_t>(n, aivCoreNum), 1);
}

static inline uint32_t GetUsedAicCoreNum(uint32_t n, uint32_t baseM, const char* opTag)
{
    uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE(opTag, "cube core count is 0");
        return 0;
    }
    uint32_t coreDim = CeilDiv<uint32_t>(n, aicCoreNum);
    uint32_t singleCoreDim = std::max<uint32_t>(coreDim, baseM);
    uint64_t tileCount = static_cast<uint64_t>(CeilDiv<uint32_t>(n, singleCoreDim)) *
                         static_cast<uint64_t>(CeilDiv<uint32_t>(n, singleCoreDim));
    return std::max<uint32_t>(std::min<uint64_t>(tileCount, aicCoreNum), 1);
}

static inline aclblasStatus_t ReadAlphaBetaFromDevice(
    const float* alpha, const float* beta, float& alphaVal, float& betaVal, aclrtStream stream, const char* opTag)
{
    alphaVal = 0.0f;
    betaVal = 0.0f;
    // Resolve pointer locations first: host scalars are dereferenced directly,
    // device scalars go through D2H, and mixing host/device is rejected — the
    // same contract as cherkx/cherk (issue #642).
    bool alphaIsDevice = false;
    bool betaIsDevice = false;
    aclblasStatus_t locRet = CheckPtrLocation(alpha, &alphaIsDevice);
    if (locRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(opTag, "alpha pointer location query failed, st=%d", static_cast<int>(locRet));
        return locRet;
    }
    locRet = CheckPtrLocation(beta, &betaIsDevice);
    if (locRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(opTag, "beta pointer location query failed, st=%d", static_cast<int>(locRet));
        return locRet;
    }
    if (alphaIsDevice != betaIsDevice) {
        OP_LOGE(opTag, "alpha and beta must be both host or both device pointers");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!alphaIsDevice) {
        alphaVal = *alpha;
        betaVal = *beta;
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError aclRet =
        aclrtMemcpyAsync(&alphaVal, sizeof(float), alpha, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opTag, "aclrtMemcpyAsync alpha D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtMemcpyAsync(&betaVal, sizeof(float), beta, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opTag, "aclrtMemcpyAsync beta D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opTag, "aclrtSynchronizeStream for D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
static inline T CalScaleTilingData(
    uint32_t usedAivCoreNum, uint32_t n, uint32_t ldc, uint8_t isAlphaZero, uint8_t isKZero, uint8_t isBetaZero,
    uint32_t tempRowStride, uint8_t uploMode, float alphaVal, float betaVal)
{
    T tiling{};
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempRowStride = tempRowStride;
    tiling.rowsPerCore = CeilDiv<uint32_t>(n, usedAivCoreNum);
    tiling.alphaVal = alphaVal;
    tiling.betaVal = betaVal;
    tiling.uploMode = uploMode;
    tiling.isAlphaZero = isAlphaZero;
    tiling.isKZero = isKZero;
    tiling.isBetaZero = isBetaZero;
    return tiling;
}

struct SyrkLaunchCtx {
    float alphaVal;
    float betaVal;
    bool isAlphaZero;
    bool isKZero;
    bool isBetaZero;
    uint32_t usedAivCoreNum;
    uint32_t usedAicCoreNum;
    uint32_t tempRowStride;
    size_t tempAligned;
};

static inline aclblasStatus_t PrepareSyrkLaunch(
    aclrtStream stream, uint32_t n, uint32_t k, aclblasOperation_t& trans, const float* alpha, const float* beta,
    uint32_t baseM, uint32_t fixpipeAlign, const char* opTag, SyrkLaunchCtx& ctx)
{
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (trans == ACLBLAS_OP_C) {
        trans = ACLBLAS_OP_T;
    }
    aclblasStatus_t readRet = ReadAlphaBetaFromDevice(alpha, beta, ctx.alphaVal, ctx.betaVal, stream, opTag);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }

    ctx.isAlphaZero = (ctx.alphaVal == 0.0f);
    ctx.isKZero = (k == 0);
    ctx.isBetaZero = (ctx.betaVal == 0.0f);

    OP_LOGD(
        opTag, "alpha=%.6f beta=%.6f isAlphaZero=%d isKZero=%d isBetaZero=%d", ctx.alphaVal, ctx.betaVal,
        ctx.isAlphaZero, ctx.isKZero, ctx.isBetaZero);

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE(opTag, "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ctx.usedAivCoreNum = GetUsedAivCoreNum(n, aivCoreNum);
    ctx.usedAicCoreNum = GetUsedAicCoreNum(n, baseM, opTag);
    if (ctx.usedAicCoreNum == 0) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    ctx.tempRowStride = CeilAlign<uint32_t>(n, fixpipeAlign);
    constexpr size_t GM_ALIGN = 32;
    size_t tempSize = static_cast<size_t>(n) * static_cast<size_t>(ctx.tempRowStride) * sizeof(float);
    ctx.tempAligned = (tempSize + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    return ACLBLAS_STATUS_SUCCESS;
}
