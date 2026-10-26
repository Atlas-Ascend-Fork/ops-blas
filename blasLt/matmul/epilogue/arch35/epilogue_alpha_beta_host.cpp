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
 * \file epilogue_alpha_beta_host.cpp
 * \brief Host-side launch helper for alpha*D_raw + beta*C epilogue.
 */

#include <algorithm>
#include <cstdint>

#include <acl/acl.h>

#include "epilogue_alpha_beta_tiling_data.h"
#include "epilogue_alpha_beta_host.h"
#include "epilogue_alpha_beta_kernel.h"

namespace {

constexpr uint32_t SIMT_MIN_THREAD_NUM = 128U;
constexpr uint32_t SIMT_MAX_THREAD_NUM = 2048U;

// issue #596: m*n 用 uint32 相乘在 m*n >= 2^32 时回绕，元素总数按 64 位计算。
constexpr uint64_t CeilDivU64(uint64_t a, uint64_t b)
{
    if (a == 0 || b == 0) {
        return 0ULL;
    }
    return (a + b - 1) / b;
}

uint32_t GetVectorCoreCount()
{
    int32_t deviceId = 0;
    int64_t vecCoreNum = 0;
    if (aclrtGetDevice(&deviceId) != ACL_SUCCESS) {
        return 1U;
    }
    if (aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &vecCoreNum) != ACL_SUCCESS ||
        vecCoreNum <= 0) {
        return 1U;
    }
    return static_cast<uint32_t>(vecCoreNum);
}

uint8_t ToEpilogueDtype(aclDataType dtype) { return (dtype == ACL_BF16) ? 1U : 0U; }

} // namespace

void epilogue_alpha_beta_get_tiling(uint32_t m, uint32_t n, uint32_t numBlocks, EpilogueAlphaBetaTilingData& tilingData)
{
    tilingData = {};
    tilingData.m = m;
    tilingData.n = n;
    tilingData.usedCoreNum = std::max(1U, numBlocks);
    // issue #596: m*n 以 64 位中间量计算，避免超大矩阵下乘法回绕低估线程需求
    const uint64_t totalElements = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    const uint64_t perCore = CeilDivU64(totalElements, tilingData.usedCoreNum);
    const uint64_t alignedThreads = CeilDivU64(perCore, SIMT_MIN_THREAD_NUM) * SIMT_MIN_THREAD_NUM;
    tilingData.numThreads = static_cast<uint32_t>(std::min<uint64_t>(alignedThreads, SIMT_MAX_THREAD_NUM));
}

void epilogue_alpha_beta_do(
    uint8_t* dRaw, uint8_t* c, uint8_t* d, uint32_t m, uint32_t n, uint32_t ldc, uint32_t ldd, uint32_t lddRaw,
    float alpha, float beta, aclDataType dtypeC, aclDataType dtypeDRaw, aclDataType dtypeD, void* stream,
    const float* alphaDevice, const float* betaDevice)
{
    EpilogueAlphaBetaTilingData tiling{};
    epilogue_alpha_beta_get_tiling(m, n, GetVectorCoreCount(), tiling);
    tiling.ldc = ldc;
    tiling.ldd = ldd;
    tiling.lddRaw = lddRaw;
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.dtypeDRaw = ToEpilogueDtype(dtypeDRaw);
    tiling.dtypeC = ToEpilogueDtype(dtypeC);
    tiling.dtypeD = ToEpilogueDtype(dtypeD);
    tiling.useC = (beta != 0.0f && c != nullptr) ? 1U : 0U;
    tiling.deviceScalarMode = (alphaDevice != nullptr && betaDevice != nullptr) ? 1U : 0U;
    if (tiling.deviceScalarMode != 0U) {
        tiling.useC = 1U;
    }
    epilogue_alpha_beta_kernel_do(dRaw, c, d, tiling, stream, alphaDevice, betaDevice);
}
