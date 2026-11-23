/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CTPTTR_KERNEL_IMPL_H
#define CTPTTR_KERNEL_IMPL_H

#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "ctpttr_kernel.h"

using namespace AscendC;

namespace ctpttr_arch35 {
// A complex64 element is copied as a pair of 32-bit words, so no floating point
// instruction ever touches the payload and NaN/Inf bit patterns stay intact.
constexpr uint32_t WORDS_PER_COMPLEX = 2U;
constexpr uint32_t CHUNK_WORDS = 8192U; // 32KB staging buffer

class CtpttrKernel {
public:
    __aicore__ inline explicit CtpttrKernel(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR aPacked, GM_ADDR aFull, const CtpttrTilingData& tiling)
    {
        tiling_ = tiling;
        apGM_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(aPacked));
        aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(aFull));
        pipe_.InitBuffer(stageBuf_, CHUNK_WORDS * sizeof(uint32_t));
        eventM2M3_ = static_cast<int32_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        eventM3M2_ = static_cast<int32_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        // Round-robin over columns keeps both uplo branches balanced: LOWER columns
        // shrink with the column index, UPPER columns grow with it.
        for (uint32_t col = blockIdx; col < tiling_.n; col += blockNum) {
            CopyColumn(col);
        }
    }

private:
    __aicore__ inline uint64_t PackedStart(uint32_t col) const
    {
        const uint64_t c = static_cast<uint64_t>(col);
        const uint64_t n = static_cast<uint64_t>(tiling_.n);
        return (tiling_.upper != 0U) ? (c * (c + 1U) / 2U) : (c * (2U * n - c + 1U) / 2U);
    }

    __aicore__ inline void CopyColumn(uint32_t col)
    {
        // One column of the triangle is a contiguous run in AP and a contiguous run in
        // A, so the whole operator is a sequence of independent block copies.
        const uint32_t colLen = (tiling_.upper != 0U) ? (col + 1U) : (tiling_.n - col);
        const uint64_t firstRow = (tiling_.upper != 0U) ? 0U : static_cast<uint64_t>(col);
        const uint64_t srcBase = WORDS_PER_COMPLEX * PackedStart(col);
        const uint64_t dstBase =
            WORDS_PER_COMPLEX * (static_cast<uint64_t>(col) * static_cast<uint64_t>(tiling_.lda) + firstRow);
        const uint32_t totalWords = colLen * WORDS_PER_COMPLEX;
        LocalTensor<uint32_t> stage = stageBuf_.Get<uint32_t>();

        for (uint32_t done = 0; done < totalWords;) {
            uint32_t chunkWords = totalWords - done;
            if (chunkWords > CHUNK_WORDS) {
                chunkWords = CHUNK_WORDS;
            }
            const DataCopyExtParams copyParams{1, chunkWords * static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
            const DataCopyPadExtParams<uint32_t> loadParams{false, 0, 0, 0};

            DataCopyPad(stage, apGM_[srcBase + done], copyParams, loadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventM2M3_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventM2M3_);
            // blockLen carries the exact byte count, so the UB tail is never written back.
            DataCopyPad(aGM_[dstBase + done], stage, copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventM3M2_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventM3M2_);
            done += chunkWords;
        }
    }

    TPipe& pipe_;
    CtpttrTilingData tiling_{};
    GlobalTensor<uint32_t> apGM_;
    GlobalTensor<uint32_t> aGM_;
    TBuf<TPosition::VECIN> stageBuf_;
    int32_t eventM2M3_ = 0;
    int32_t eventM3M2_ = 0;
};
} // namespace ctpttr_arch35

extern "C" __global__ __aicore__ void ctpttr_aiv_kernel(GM_ADDR source, GM_ADDR destination, CtpttrTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ctpttr_arch35::CtpttrKernel op(pipe);
    op.Init(source, destination, tiling);
    op.Process();
}

// issue #1181: launch through aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs
// so the runtime status of the launch is returned to the caller: the <<<>>>
// statement form discards it (same pattern as stbmv/chbmv direct-return).
aclError ctpttr_kernel_do(const uint8_t* source, uint8_t* destination, const CtpttrTilingData& tiling, void* stream)
{
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(ctpttr_aiv_kernel), &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        uint8_t* source;
        uint8_t* destination;
        CtpttrTilingData tiling;
    } args{const_cast<uint8_t*>(source), destination, tiling};
    static_assert(offsetof(Arguments, tiling) == 2U * sizeof(uint8_t*), "kernel ABI mismatch");
    return aclrtLaunchKernelWithHostArgs(
        function, tiling.useCoreNum, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}

#endif // CTPTTR_KERNEL_IMPL_H
