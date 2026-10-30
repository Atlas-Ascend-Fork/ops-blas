/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Arch22 grouped batched SGEMM host dispatch.
#include <exception>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "sgemm_gb_plan.h"
#include "sgemm_gb_cube_host.h"
void sgemm_gb_vector_do(sgemm_gb::Packet packet, uint32_t blocks, void* stream);
void sgemm_gb_small8_do(sgemm_gb::Small8Packet packet, uint32_t blocks, void* stream);
void sgemm_gb_small8_strided_do(sgemm_gb::Small8StridedPacket packet, uint32_t blocks, void* stream);
namespace {
void DispatchSmall8(const sgemm_gb::Arguments& args, const std::vector<sgemm_gb::Small8Packet>& packets, void* stream)
{
    const uint32_t aiv = GetAivCoreCount();
    sgemm_gb::Small8StridedPacket strided{};
    if (sgemm_gb::PlanSmall8Strided(args, packets, strided)) {
        sgemm_gb_small8_strided_do(strided, std::min<uint32_t>(aiv, strided.count), stream);
        return;
    }
    for (const auto& packet : packets)
        sgemm_gb_small8_do(packet, sgemm_gb::Small8Blocks(packet, aiv), stream);
}
void DispatchValidated(const sgemm_gb::Arguments& args, void* stream)
{
    // Snapshot before the first enqueue: invalid later groups cannot partially modify output.
    std::vector<sgemm_gb::Small8Packet> smallPackets;
    if (sgemm_gb::PlanSmall8(args, smallPackets)) {
        DispatchSmall8(args, smallPackets, stream);
        return;
    }
    const auto packets = sgemm_gb::PlanValidated(args);
    const uint32_t aiv = GetAivCoreCount();
    const auto cubePlans = sgemm_gb::PrepareCubePlans(packets);
    for (size_t i = 0; i < packets.size(); ++i) {
        const size_t bundled = sgemm_gb::LaunchSmallGroups(packets, cubePlans, i, stream);
        if (bundled) {
            i += bundled - 1;
            continue;
        }
        if (cubePlans[i].enabled) {
            if (i + 1 < packets.size() &&
                sgemm_gb::LaunchCombinedCube(packets[i], packets[i + 1], cubePlans[i], cubePlans[i + 1], stream)) {
                ++i;
                continue;
            }
            sgemm_gb::LaunchCube(packets[i], cubePlans[i], stream);
            continue;
        }
        sgemm_gb_vector_do(packets[i], sgemm_gb::VectorBlocks(packets[i], aiv), stream);
    }
}
} // namespace
aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount, const aclblasOperation_t* transaArray,
    const aclblasOperation_t* transbArray, const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* const* Aarray, const int* ldaArray, const float* const* Barray,
    const int* ldbArray, const float* betaArray, float* const* Carray, const int* ldcArray, const int* groupSizeArray)
{
    if (!handle)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    sgemm_gb::Arguments args{groupCount, transaArray, transbArray, mArray,    nArray, kArray,   alphaArray,    Aarray,
                             ldaArray,   Barray,      ldbArray,    betaArray, Carray, ldcArray, groupSizeArray};
    const auto valid = sgemm_gb::Validate(args);
    if (valid != ACLBLAS_STATUS_SUCCESS || groupCount == 0)
        return valid;
    try {
        DispatchValidated(args, handle->stream);
    } catch (const std::exception&) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // Device errors are surfaced by the caller's stream synchronization, as with upstream launchers.
    return ACLBLAS_STATUS_SUCCESS;
}
