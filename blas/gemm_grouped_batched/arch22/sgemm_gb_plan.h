/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Host-only validation and launch planning. Compiled unchanged in portable tests.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include "cann_ops_blas.h"
#include "sgemm_gb_packet.h"
namespace sgemm_gb {
struct Arguments {
    int groups;
    const aclblasOperation_t *ta, *tb;
    const int *m, *n, *k;
    const float* alpha;
    const float* const* a;
    const int* lda;
    const float* const* b;
    const int* ldb;
    const float* beta;
    float* const* c;
    const int *ldc, *batch;
};
inline bool SafeMul(uint64_t a, uint64_t b, uint64_t& out)
{
    if (b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b)
        return false;
    out = a * b;
    return true;
}
inline bool ExtentOK(int rows, int cols, int ld)
{
    if (rows == 0 || cols == 0)
        return true;
    uint64_t span = 0;
    if (!SafeMul(static_cast<uint64_t>(cols - 1), static_cast<uint64_t>(ld), span))
        return false;
    span += static_cast<uint64_t>(rows);
    // Pointer arithmetic and byte extents must also fit ptrdiff_t, not just uint64_t.
    return span <= static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
}
inline bool HasRequiredArrays(const Arguments& x)
{
    return x.ta && x.tb && x.m && x.n && x.k && x.alpha && x.a && x.lda && x.b && x.ldb && x.beta && x.c && x.ldc &&
           x.batch;
}
inline bool ValidGroupMetadata(const Arguments& x, int g)
{
    if ((x.ta[g] != ACLBLAS_OP_N && x.ta[g] != ACLBLAS_OP_T) || (x.tb[g] != ACLBLAS_OP_N && x.tb[g] != ACLBLAS_OP_T))
        return false; // OP_C is invalid under this task.
    if (x.m[g] < 0 || x.n[g] < 0 || x.k[g] < 0 || x.batch[g] < 0)
        return false;
    const int ar = x.ta[g] == ACLBLAS_OP_N ? x.m[g] : x.k[g];
    const int ac = x.ta[g] == ACLBLAS_OP_N ? x.k[g] : x.m[g];
    const int br = x.tb[g] == ACLBLAS_OP_N ? x.k[g] : x.n[g];
    const int bc = x.tb[g] == ACLBLAS_OP_N ? x.n[g] : x.k[g];
    return x.lda[g] >= std::max(1, ar) && x.ldb[g] >= std::max(1, br) && x.ldc[g] >= std::max(1, x.m[g]) &&
           ExtentOK(ar, ac, x.lda[g]) && ExtentOK(br, bc, x.ldb[g]) && ExtentOK(x.m[g], x.n[g], x.ldc[g]);
}
inline aclblasStatus_t Validate(const Arguments& x)
{
    if (x.groups < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (x.groups == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (!HasRequiredArrays(x))
        return ACLBLAS_STATUS_INVALID_VALUE;
    uint64_t total = 0;
    // Validate ALL scalar metadata before reading any matrix-pointer element.
    for (int g = 0; g < x.groups; ++g) {
        if (!ValidGroupMetadata(x, g))
            return ACLBLAS_STATUS_INVALID_VALUE;
        total += static_cast<uint64_t>(x.batch[g]); // int32 groupCount * int32 batch fits uint64.
        if (total > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(void*))
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Strict task policy: non-null element pointers are required even for alpha=0 / k=0 / empty shapes.
    // No payload is dereferenced here. Host pointer-array residency remains a caller precondition.
    for (uint64_t i = 0; i < total; ++i)
        if (!x.a[i] || !x.b[i] || !x.c[i])
            return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}
inline bool CubeEligible(const Packet& p);
inline bool SameSmallGroupShape(const Group& g, const Group& h)
{
    return g.m == h.m && g.n == h.n && g.k == h.k && g.lda == h.lda && g.ldb == h.ldb && g.ldc == h.ldc &&
           g.ta == h.ta && g.tb == h.tb;
}
inline bool MergeCountsEligible(const Packet& first, const Packet& second)
{
    return first.count && second.count && first.count <= kPacketBatches &&
           second.count <= kPacketBatches - first.count && CubeEligible(first) && CubeEligible(second) &&
           first.g.m <= 256 && first.g.n <= 256 && first.g.k <= 256 && SameSmallGroupShape(first.g, second.g);
}
inline bool OverlapRanges(uint64_t a, uint64_t na, uint64_t b, uint64_t nb) { return a <= b ? b - a < na : a - b < nb; }
inline bool PacketPairIndependent(const Packet& p, const Packet& q, uint64_t bytesA, uint64_t bytesB, uint64_t bytesC)
{
    for (uint32_t i = 0; i < p.count; ++i)
        for (uint32_t j = 0; j < q.count; ++j)
            if (OverlapRanges(p.c[i], bytesC, q.a[j], bytesA) || OverlapRanges(p.c[i], bytesC, q.b[j], bytesB) ||
                ((&p != &q || i != j) && OverlapRanges(p.c[i], bytesC, q.c[j], bytesC)))
                return false;
    return true;
}
inline bool SmallPacketsIndependent(const Packet& first, const Packet& second)
{
    const auto& g = first.g;
    const uint64_t bytesA = uint64_t(g.m) * g.k * sizeof(float);
    const uint64_t bytesB = uint64_t(g.k) * g.n * sizeof(float);
    const uint64_t bytesC = uint64_t(g.m) * g.n * sizeof(float);
    const Packet* packets[] = {&first, &second};
    for (const auto* p : packets)
        for (uint32_t i = 0; i < p->count; ++i)
            if (p->a[i] > std::numeric_limits<uint64_t>::max() - bytesA ||
                p->b[i] > std::numeric_limits<uint64_t>::max() - bytesB ||
                p->c[i] > std::numeric_limits<uint64_t>::max() - bytesC)
                return false;
    for (const auto* p : packets)
        for (const auto* q : packets)
            if (!PacketPairIndependent(*p, *q, bytesA, bytesB, bytesC))
                return false;
    return true;
}
inline bool MergeIndependentSmallPackets(Packet& first, const Packet& second)
{
    if (!MergeCountsEligible(first, second))
        return false;
    // Compact storage is guaranteed by CubeEligible. Check every output against
    // every input/output before removing an inter-group stream-order boundary.
    if (!SmallPacketsIndependent(first, second))
        return false;
    for (uint32_t i = 0; i < second.count; ++i) {
        first.a[first.count + i] = second.a[i];
        first.b[first.count + i] = second.b[i];
        first.c[first.count + i] = second.c[i];
    }
    first.count += second.count;
    // Keep the first group id: offset-packet fusion must not cross a new boundary.
    return true;
}
inline std::vector<Packet> PlanValidated(const Arguments& x, const std::vector<uint8_t>* skippedGroups = nullptr)
{
    std::vector<Packet> plan;
    uint64_t offset = 0;
    for (int g = 0; g < x.groups; ++g) {
        const uint64_t batch = static_cast<uint64_t>(x.batch[g]);
        // Even an empty group consumes its flattened pointer-array range.
        const bool skipped = skippedGroups != nullptr && (*skippedGroups)[g] != 0;
        if (!skipped && x.m[g] && x.n[g] && batch && !((x.k[g] == 0 || x.alpha[g] == 0.0f) && x.beta[g] == 1.0f)) {
            for (uint64_t j = 0; j < batch; j += kPacketBatches) {
                Packet p{};
                p.g = {
                    static_cast<uint32_t>(x.m[g]),
                    static_cast<uint32_t>(x.n[g]),
                    static_cast<uint32_t>(x.k[g]),
                    static_cast<uint32_t>(x.lda[g]),
                    static_cast<uint32_t>(x.ldb[g]),
                    static_cast<uint32_t>(x.ldc[g]),
                    x.ta[g] == ACLBLAS_OP_T ? 1U : 0U,
                    x.tb[g] == ACLBLAS_OP_T ? 1U : 0U,
                    x.alpha[g],
                    x.beta[g]};
                p.count = static_cast<uint32_t>(std::min<uint64_t>(batch - j, kPacketBatches));
                p.reserved = static_cast<uint32_t>(g);
                for (uint32_t s = 0; s < p.count; ++s) {
                    p.a[s] = reinterpret_cast<uint64_t>(x.a[offset + j + s]);
                    p.b[s] = reinterpret_cast<uint64_t>(x.b[offset + j + s]);
                    p.c[s] = reinterpret_cast<uint64_t>(x.c[offset + j + s]);
                }
                if (!plan.empty() && MergeIndependentSmallPackets(plan.back(), p))
                    continue;
                plan.push_back(p);
            }
        }
        offset += batch;
    }
    return plan;
}
inline bool Small8RangesIndependent(const Arguments& x, uint64_t total)
{
    // Combining groups removes stream-order boundaries. Require disjoint
    // output/input ranges, including partial overlap, before using this path.
    // Sort once to avoid quadratic validation for large batches.
    constexpr uint64_t bytes = 64 * sizeof(float);
    std::vector<uint64_t> inputs, outputs;
    for (uint64_t i = 0; i < total; ++i) {
        inputs.push_back(reinterpret_cast<uint64_t>(x.a[i]));
        inputs.push_back(reinterpret_cast<uint64_t>(x.b[i]));
        outputs.push_back(reinterpret_cast<uint64_t>(x.c[i]));
    }
    std::sort(inputs.begin(), inputs.end());
    std::sort(outputs.begin(), outputs.end());
    for (size_t i = 0; i < outputs.size(); ++i) {
        const uint64_t output = outputs[i];
        if (i && OverlapRanges(outputs[i - 1], bytes, output, bytes))
            return false;
        const auto next = std::lower_bound(inputs.begin(), inputs.end(), output);
        if ((next != inputs.end() && OverlapRanges(output, bytes, *next, bytes)) ||
            (next != inputs.begin() && OverlapRanges(output, bytes, *(next - 1), bytes)))
            return false;
    }
    return true;
}
inline bool PlanSmall8(const Arguments& x, std::vector<Small8Packet>& packets)
{
    uint64_t total = 0;
    for (int g = 0; g < x.groups; ++g) {
        if (x.m[g] != 8 || x.n[g] != 8 || x.k[g] != 8 || x.lda[g] != 8 || x.ldb[g] != 8 || x.ldc[g] != 8 ||
            x.ta[g] != ACLBLAS_OP_N || x.tb[g] != ACLBLAS_OP_N || x.alpha[g] != 1.0f || x.beta[g] != 0.0f)
            return false;
        total += static_cast<uint64_t>(x.batch[g]);
    }
    if (!total || !Small8RangesIndependent(x, total))
        return false;
    packets.reserve(static_cast<size_t>(CeilDiv(total, kSmall8Batches)));
    for (uint64_t start = 0; start < total; start += kSmall8Batches) {
        Small8Packet packet{};
        packet.g = {8, 8, 8, 8, 8, 8, 0, 0, 1.0f, 0.0f};
        packet.count = static_cast<uint32_t>(std::min<uint64_t>(total - start, kSmall8Batches));
        for (uint32_t i = 0; i < packet.count; ++i) {
            packet.a[i] = reinterpret_cast<uint64_t>(x.a[start + i]);
            packet.b[i] = reinterpret_cast<uint64_t>(x.b[start + i]);
            packet.c[i] = reinterpret_cast<uint64_t>(x.c[start + i]);
        }
        packets.push_back(packet);
    }
    return true;
}
inline bool PlanSmall8Strided(const Arguments& x, const std::vector<Small8Packet>& packets, Small8StridedPacket& out)
{
    // Called only after PlanSmall8 accepted the entire invocation and checked
    // distinct output pointers. A partial/mismatched progression falls back
    // to its independent pointer-list packets without changing semantics.
    uint64_t total = 0;
    for (int g = 0; g < x.groups; ++g)
        total += static_cast<uint64_t>(x.batch[g]);
    if (packets.empty() || total <= kSmall8Batches || total > std::numeric_limits<uint32_t>::max())
        return false;
    const uint64_t baseA = reinterpret_cast<uint64_t>(x.a[0]);
    const uint64_t baseB = reinterpret_cast<uint64_t>(x.b[0]);
    const uint64_t baseC = reinterpret_cast<uint64_t>(x.c[0]);
    const uint64_t secondA = reinterpret_cast<uint64_t>(x.a[1]);
    const uint64_t secondB = reinterpret_cast<uint64_t>(x.b[1]);
    const uint64_t secondC = reinterpret_cast<uint64_t>(x.c[1]);
    if (secondA <= baseA || secondB <= baseB || secondC <= baseC)
        return false;
    const uint64_t strideA = secondA - baseA;
    const uint64_t strideB = secondB - baseB;
    const uint64_t strideC = secondC - baseC;
    const uint64_t last = total - 1;
    if (strideA > (std::numeric_limits<uint64_t>::max() - baseA) / last ||
        strideB > (std::numeric_limits<uint64_t>::max() - baseB) / last ||
        strideC > (std::numeric_limits<uint64_t>::max() - baseC) / last)
        return false;
    for (uint64_t i = 2; i < total; ++i)
        if (reinterpret_cast<uint64_t>(x.a[i]) != baseA + i * strideA ||
            reinterpret_cast<uint64_t>(x.b[i]) != baseB + i * strideB ||
            reinterpret_cast<uint64_t>(x.c[i]) != baseC + i * strideC)
            return false;
    out = {};
    out.g = packets[0].g;
    out.count = static_cast<uint32_t>(total);
    out.baseA = baseA;
    out.baseB = baseB;
    out.baseC = baseC;
    out.strideA = strideA;
    out.strideB = strideB;
    out.strideC = strideC;
    return true;
}
inline uint32_t Small8Blocks(const Small8Packet& p, uint32_t cores)
{
    return std::min<uint32_t>(std::max(1U, cores), p.count);
}
inline uint32_t VectorBlocks(const Packet& p, uint32_t cores)
{
    const uint64_t tasks = CeilDiv(p.g.m, kVectorRows) * p.g.n * p.count;
    return static_cast<uint32_t>(std::min<uint64_t>(std::max(1U, cores), tasks));
}
inline bool CubeEligible(const Packet& p)
{
    const auto& g = p.g;
    // Experimental 16-tile route for packed matrices with 16-aligned M/N/K.
    // Keep this outside the product branch until A2 precision/performance tests pass.
    return g.alpha == 1.0f && g.beta == 0.0f && g.m >= 16 && g.n >= 16 && g.k >= 16 && g.m % 16 == 0 && g.n % 16 == 0 &&
           g.k % 16 == 0 && g.lda == (g.ta ? g.k : g.m) && g.ldb == (g.tb ? g.n : g.k) && g.ldc == g.m;
}
} // namespace sgemm_gb
