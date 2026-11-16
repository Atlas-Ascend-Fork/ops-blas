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
 * \file isamax_host.cpp
 * \brief Host side implementation for isamax operator (arch22)
 */

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void isamax_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling,
                      uint32_t numBlocks, void *stream);

constexpr int32_t MAXNUMF32ELEEACHCORE = 23040;
constexpr int32_t BYTESPERREPEAT = 256;
constexpr int32_t F32LEN = 4;
constexpr uint32_t MAXVECTORNUM = 40;

constexpr int32_t GM_RESULT_LEN = 2;
constexpr int32_t BYTE_LEN_4 = 4;
constexpr uint64_t ELEMENTS_IN_BLOCK = 8;
constexpr int32_t MAX_REPEATS = 255;

constexpr int32_t DEAL_TIMES_EACH_CORE_REDUCE = 63;

constexpr uint32_t ELEMENTS_PER_REPEAT = BYTESPERREPEAT / F32LEN;
static_assert(ELEMENTS_PER_REPEAT > 0, "ELEMENTS_PER_REPEAT must be greater than zero");

struct IsamaxTilingData {
    uint32_t incx;
    uint32_t needVecCoreNum;
    uint32_t dtypeFlag;
    uint32_t rstLenAllCoreBytes;
    uint32_t tailCount;
    uint32_t maxRepeatLen;
    uint32_t startOffset[MAXVECTORNUM];
    uint32_t eleTotalEachCore[MAXVECTORNUM];
    uint32_t dealTimesEachCore[MAXVECTORNUM];
    uint32_t dealLenEachTime[MAXVECTORNUM];
    uint32_t reduceMaxRstsLenEachCore[MAXVECTORNUM];
    uint32_t dealLenUpBlockEachTime[MAXVECTORNUM];
    uint32_t totalRptCntNor[MAXVECTORNUM];
    uint32_t totalRptCntNorRemainder[MAXVECTORNUM];
    uint32_t rptBatchCntNor[MAXVECTORNUM];
    uint32_t rptBatchCntNorRemainder[MAXVECTORNUM];
    uint32_t rmdRptLenNor[MAXVECTORNUM];
};

struct CoreTilingBuffers {
    std::array<uint32_t, MAXVECTORNUM> startOffset{};
    std::array<uint32_t, MAXVECTORNUM> endOffset{};
    std::array<uint32_t, MAXVECTORNUM> eleTotalEachCore{};
    std::array<uint32_t, MAXVECTORNUM> dealLenEachTime{};
    std::array<uint32_t, MAXVECTORNUM> dealTimesEachCore{};
    std::array<uint32_t, MAXVECTORNUM> reduceMaxRstsLenEachCore{};
    std::array<uint32_t, MAXVECTORNUM> dealLenUpBlockEachTime{};
    std::array<uint32_t, MAXVECTORNUM> totalRptCntNor{};
    std::array<uint32_t, MAXVECTORNUM> totalRptCntNorRemainder{};
    std::array<uint32_t, MAXVECTORNUM> rptBatchCntNor{};
    std::array<uint32_t, MAXVECTORNUM> rptBatchCntNorRemainder{};
    std::array<uint32_t, MAXVECTORNUM> rmdRptLenNor{};
};

struct TilingCommonParams {
    uint32_t needVecCoreNum;
    uint32_t rstLenAllCoreBytes;
    uint32_t minEleRepeatTail;
    uint32_t minEleRepeatsNumberEachCore;
    uint32_t minEleRepeatsNumbeTail;
};

uint32_t CeilA2B(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

uint32_t GetNeedVecCoreNum(uint32_t tensorLen, uint32_t elements)
{
    if (elements == 0) {
        return 1;
    }
    if (tensorLen > MAXVECTORNUM * elements) {
        return MAXVECTORNUM;
    }
    if (tensorLen > elements) {
        return tensorLen / elements;
    }
    return 1;
}

TilingCommonParams CalcIsamaxCommonParams(uint32_t n)
{
    TilingCommonParams params{};
    uint32_t minEleEachCore = ELEMENTS_PER_REPEAT;
    uint32_t needVecCoreNum = GetNeedVecCoreNum(n, minEleEachCore);
    if (needVecCoreNum == 0) {
        needVecCoreNum = 1;
    }
    uint32_t minEleRepeatsNumber = n / minEleEachCore;
    uint32_t minEleRepeatTail = n % minEleEachCore;

    params.needVecCoreNum = needVecCoreNum;
    params.rstLenAllCoreBytes = needVecCoreNum * GM_RESULT_LEN * BYTE_LEN_4;
    params.minEleRepeatTail = minEleRepeatTail;
    params.minEleRepeatsNumberEachCore = minEleRepeatsNumber / needVecCoreNum;
    params.minEleRepeatsNumbeTail = minEleRepeatsNumber % needVecCoreNum;
    return params;
}

void CalcCoreTiling(uint32_t i, TilingCommonParams &params,
                    CoreTilingBuffers &buf)
{
    uint32_t minEleEachCore = ELEMENTS_PER_REPEAT;
    uint32_t eleLenEachCore = params.minEleRepeatsNumberEachCore * minEleEachCore;

    buf.startOffset[i] = (i == 0) ? 0 : buf.endOffset[i - 1];

    if (params.minEleRepeatsNumbeTail > 0) {
        eleLenEachCore += minEleEachCore;
        params.minEleRepeatsNumbeTail--;
    }
    buf.dealTimesEachCore[i] = 0;
    buf.dealLenEachTime[i] = eleLenEachCore;
    if (eleLenEachCore > 0 && eleLenEachCore <= MAXNUMF32ELEEACHCORE) {
        buf.dealTimesEachCore[i] = 1;
    } else if (eleLenEachCore > MAXNUMF32ELEEACHCORE) {
        buf.dealTimesEachCore[i] = CeilA2B(eleLenEachCore, MAXNUMF32ELEEACHCORE);
        buf.dealLenEachTime[i] = MAXNUMF32ELEEACHCORE;
    }

    uint32_t dealLenEachTimeAttachTail = buf.dealLenEachTime[i];
    if (i == 0 && params.minEleRepeatTail != 0) {
        eleLenEachCore += params.minEleRepeatTail;
        if (buf.dealTimesEachCore[i] == 0) {
            buf.dealTimesEachCore[i] = 1;
        }
        dealLenEachTimeAttachTail += params.minEleRepeatTail;
    }
    buf.endOffset[i] = buf.startOffset[i] + eleLenEachCore;
    buf.eleTotalEachCore[i] = eleLenEachCore;

    buf.reduceMaxRstsLenEachCore[i] = DEAL_TIMES_EACH_CORE_REDUCE * ELEMENTS_IN_BLOCK + ELEMENTS_IN_BLOCK;
    buf.dealLenUpBlockEachTime[i] = CeilA2B(dealLenEachTimeAttachTail, ELEMENTS_IN_BLOCK) * ELEMENTS_IN_BLOCK;

    buf.totalRptCntNor[i] = buf.dealLenEachTime[i] / ELEMENTS_PER_REPEAT;
    buf.totalRptCntNorRemainder[i] = buf.dealLenEachTime[i] % ELEMENTS_PER_REPEAT;
    buf.rptBatchCntNor[i] = buf.totalRptCntNor[i] / MAX_REPEATS;
    buf.rptBatchCntNorRemainder[i] = buf.totalRptCntNor[i] % MAX_REPEATS;
    buf.rmdRptLenNor[i] = buf.rptBatchCntNorRemainder[i] * ELEMENTS_PER_REPEAT;
}

void CopyCoreTilingToOutput(const CoreTilingBuffers &buf, IsamaxTilingData &tiling)
{
    std::copy(buf.startOffset.begin(), buf.startOffset.end(), tiling.startOffset);
    std::copy(buf.eleTotalEachCore.begin(), buf.eleTotalEachCore.end(), tiling.eleTotalEachCore);
    std::copy(buf.dealTimesEachCore.begin(), buf.dealTimesEachCore.end(), tiling.dealTimesEachCore);
    std::copy(buf.dealLenEachTime.begin(), buf.dealLenEachTime.end(), tiling.dealLenEachTime);
    std::copy(buf.reduceMaxRstsLenEachCore.begin(), buf.reduceMaxRstsLenEachCore.end(),
              tiling.reduceMaxRstsLenEachCore);
    std::copy(buf.dealLenUpBlockEachTime.begin(), buf.dealLenUpBlockEachTime.end(), tiling.dealLenUpBlockEachTime);
    std::copy(buf.totalRptCntNor.begin(), buf.totalRptCntNor.end(), tiling.totalRptCntNor);
    std::copy(buf.totalRptCntNorRemainder.begin(), buf.totalRptCntNorRemainder.end(),
              tiling.totalRptCntNorRemainder);
    std::copy(buf.rptBatchCntNor.begin(), buf.rptBatchCntNor.end(), tiling.rptBatchCntNor);
    std::copy(buf.rptBatchCntNorRemainder.begin(), buf.rptBatchCntNorRemainder.end(),
              tiling.rptBatchCntNorRemainder);
    std::copy(buf.rmdRptLenNor.begin(), buf.rmdRptLenNor.end(), tiling.rmdRptLenNor);
}

IsamaxTilingData CalIsamaxTilingData(uint32_t n, uint32_t incx)
{
    IsamaxTilingData tilingData{};
    CoreTilingBuffers buf{};

    TilingCommonParams params = CalcIsamaxCommonParams(n);

    for (uint32_t i = 0; i < params.needVecCoreNum; i++) {
        CalcCoreTiling(i, params, buf);
    }

    tilingData.incx = incx;
    tilingData.needVecCoreNum = params.needVecCoreNum;
    tilingData.dtypeFlag = 0;
    tilingData.rstLenAllCoreBytes = params.rstLenAllCoreBytes;
    tilingData.tailCount = params.minEleRepeatTail;
    tilingData.maxRepeatLen = MAX_REPEATS * ELEMENTS_PER_REPEAT;

    CopyCoreTilingToOutput(buf, tilingData);
    return tilingData;
}

static aclblasStatus_t ValidateIsamaxParams(
    aclblasHandle_t handle, int n, const float* x, int incx, int* result, bool &earlyReturn)
{
    earlyReturn = false;
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || incx < 1) {
        if (result == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        aclError aclRet = aclrtMemsetAsync(result, sizeof(int32_t), 0, sizeof(int32_t), handle->stream);
        CHECK_RET(
            aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemsetAsync failed. ERROR: %d\n", aclRet);
            return ACLBLAS_STATUS_EXECUTION_FAILED);
        aclRet = aclrtSynchronizeStream(handle->stream);
        CHECK_RET(
            aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
            return ACLBLAS_STATUS_EXECUTION_FAILED);
        earlyReturn = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || result == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchIsamax(
    aclrtStream useStream, const float* x, int* result, const IsamaxTilingData &tiling)
{
    uint32_t numBlocks = CeilA2B(tiling.needVecCoreNum, 2);
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;
    size_t workspaceSize = 16 * 1024 * 1024 + MAXVECTORNUM * GM_RESULT_LEN * BYTE_LEN_4;

    aclError aclRet = aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(IsamaxTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workspaceDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(IsamaxTilingData), &tiling, sizeof(IsamaxTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    isamax_kernel_do(reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(result),
                     workspaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasIsamax(aclblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    bool earlyReturn = false;
    aclblasStatus_t status = ValidateIsamaxParams(handle, n, x, incx, result, earlyReturn);
    if (status != ACLBLAS_STATUS_SUCCESS || earlyReturn) {
        return status;
    }

    IsamaxTilingData tiling = CalIsamaxTilingData(static_cast<uint32_t>(n), static_cast<uint32_t>(incx));
    return LaunchIsamax(handle->stream, x, result, tiling);
}
