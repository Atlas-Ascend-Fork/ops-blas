/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ctbmv_kernel.h"
#include "ctbmv_panel.h"
#include "ctbmv_panel_compact.h"
#include "ctbmv_column.h"
#include "ctbmv_column_wide.h"
#include "ctbmv_transpose.h"
#include "ctbmv_transpose_fast.h"

using namespace AscendC;
using namespace Ctbmv;

__global__ __aicore__ void ctbmv_kernel_small_panel(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    CtbmvSmallPanelAiv op;
    op.Process(a, x, workspace, tiling);
    ResetMask();
#endif
}

__global__ __aicore__ void ctbmv_kernel_tiny_vector(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    CtbmvTinyVectorAiv vectorOp;
    if (!vectorOp.Process(a, x, workspace, tiling)) {
        CtbmvSmallScalarAiv scalarOp;
        scalarOp.Init(a, x, tiling);
        scalarOp.Process();
    }
    ResetMask();
#endif
}

__global__ __aicore__ void ctbmv_kernel_wide_group(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    CtbmvWideGroupAiv op;
    op.Process(a, x, workspace, tiling);
    ResetMask();
#endif
}

__global__ __aicore__ void ctbmv_kernel_transpose(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    if (tiling.k <= 128U) {
        CtbmvTransposeNarrowAiv op;
        op.Init(a, x, workspace, tiling);
        op.Process();
    } else {
        CtbmvTransposeBatchAiv batchOp;
        batchOp.Init(a, x, workspace, tiling);
        batchOp.Process();
    }
    ResetMask();
#endif
}

__global__ __aicore__ void ctbmv_kernel_compact(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    CtbmvCompactPanelAiv op;
    op.Init(a, x, workspace, tiling);
    op.Process();
    ResetMask();
#endif
}

__global__ __aicore__ void ctbmv_kernel_row(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    CtbmvTransposeFastAiv op;
    op.Init(a, x, workspace, tiling);
    op.Process();
    ResetMask();
#endif
}

__global__ __aicore__ void ctbmv_kernel_small(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    CtbmvSmallScalarAiv op;
    op.Init(a, x, tiling);
    op.Process();
    ResetMask();
#endif
}

__global__ __aicore__ void ctbmv_kernel_wide(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    CtbmvWideColumnSliceAiv op;
    op.Init(a, x, workspace, tiling);
    op.Process();
    ResetMask();
#endif
}

__global__ __aicore__ void ctbmv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    if (tiling.incx == 1 && tiling.trans == ACLBLAS_OP_N) {
        CtbmvColumnFastAiv op;
        op.Init(a, x, workspace, tiling);
        op.Process();
        ResetMask();
        return;
    }
#endif
    CtbmvAiv op;
    op.Init(a, x, workspace, tiling);
    op.Process();
    ResetMask();
}

__global__ __aicore__ void ctbmv_copy_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtbmvAiv op;
    op.Init(a, x, workspace, tiling);
    op.CopyBack();
    ResetMask();
}

__global__ __aicore__ void ctbmv_wide_reduce_kernel(GM_ADDR x, GM_ADDR partial, CtbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __DAV_C220_VEC__
    constexpr uint32_t tile = 64U;
    const uint32_t row = GetBlockIdx() * tile;
    if (row >= tiling.n) {
        return;
    }
    const uint32_t count = tiling.n - row < tile ? tiling.n - row : tile;
    const uint32_t padded = (count * 2U + 7U) & ~7U;
    GlobalTensor<float> source;
    GlobalTensor<float> output;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial), tiling.n * 2U * tiling.useCoreNum);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling.n * 2U);
    LocalTensor<float> values(TPosition::VECCALC, 0U, padded * tiling.useCoreNum);
    DataCopyExtParams copy{tiling.useCoreNum, count * 8U, (tiling.n - count) * 8U, 0U, 0U};
    DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>(padded - count * 2U), 0.0F};
    DataCopyPad(values, source[row * 2U], copy, pad);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t active = tiling.useCoreNum; active > 1U;) {
        const uint32_t next = (active + 1U) / 2U;
        Add(values, values, values[next * padded], (active - next) * padded);
        PipeBarrier<PIPE_V>();
        active = next;
    }
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams store{1U, count * 8U, 0U, 0U, 0U};
    DataCopyPad(output[row * 2U], values, store);
    PipeBarrier<PIPE_ALL>();
    ResetMask();
#endif
}

void ctbmv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    switch (tiling.path) {
        case CtbmvPath::Transpose:
            ctbmv_kernel_transpose<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
            break;
        case CtbmvPath::TinyVector:
            ctbmv_kernel_tiny_vector<<<1U, nullptr, stream>>>(a, x, workspace, tiling);
            break;
        case CtbmvPath::WideGroup: {
            ctbmv_kernel_wide_group<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
            ctbmv_wide_reduce_kernel<<<(tiling.n + 63U) / 64U, nullptr, stream>>>(
                x, workspace + ((tiling.n * 8U + 31U) & ~31U) + 9216U + 8U * ((tiling.k + 22U) & ~7U) * 32U, tiling);
            break;
        }
        case CtbmvPath::SmallPanel:
            ctbmv_kernel_small_panel<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
            break;
        case CtbmvPath::CompactPanel:
            ctbmv_kernel_compact<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
            break;
        case CtbmvPath::Row:
            ctbmv_kernel_row<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
            break;
        case CtbmvPath::SmallScalar:
            ctbmv_kernel_small<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
            break;
        case CtbmvPath::WideColumn: {
            ctbmv_kernel_wide<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
            ctbmv_wide_reduce_kernel<<<(tiling.n + 63U) / 64U, nullptr, stream>>>(
                x, workspace + ((tiling.n * 8U + 31U) & ~31U) + 9216U, tiling);
            break;
        }
        case CtbmvPath::Generic:
            ctbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, workspace, tiling);
            ctbmv_copy_kernel<<<1U, nullptr, stream>>>(a, x, workspace, tiling);
            break;
    }
}
