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

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"
#include "ssbmv_kernel.h"

namespace ssbmv {
using AscendC::Adds;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::WaitFlag;

/*!< Local spelling of the shared element size; the value itself lives in ssbmv_tiling_data.h so
 *   the host sizes its UB budget with the same number. */
constexpr uint32_t BYTES_PER_FLOAT32 = SSBMV_BYTES_PER_FLOAT32;
/*!< Upper bound, in elements, of one contiguous A load. The pieces are kept a multiple of the
 *   vector alignment so that every piece starts on a 32B boundary in UB. */
constexpr uint32_t SSBMV_SPAN_CHUNK = 4096;

/*! \brief Round val up to the next multiple of SSBMV_ALIGN_ELEMS. */
__aicore__ inline uint32_t AlignUp(uint32_t val)
{
    return (val + SSBMV_ALIGN_ELEMS - 1U) / SSBMV_ALIGN_ELEMS * SSBMV_ALIGN_ELEMS;
}

/*!
 * \brief x 窗口片段的 32B 对齐搬运（三个窗口装载路径共用的尾巴）。
 *
 * 目的地址落在「lead 向下取整到 8 的倍数」处（32B 对齐），错位 lead % 8 由 leftPadding 吸收；
 * 尾部不足一个 32B 块的部分由 rightPadding 清零。lead = vLo - winLo，为该片段相对窗口起点的元素数。
 */
__aicore__ inline void CopyXWindowAligned(
    const LocalTensor<float>& xWin, const GlobalTensor<float>& xGM, int64_t vLo, uint32_t lead, uint32_t payload)
{
    const uint32_t leadAligned = lead - (lead % SSBMV_ALIGN_ELEMS);
    const uint8_t leftPad = static_cast<uint8_t>(lead - leadAligned);
    const uint32_t used = leftPad + payload;
    const uint8_t rightPad = static_cast<uint8_t>(AlignUp(used) - used);
    DataCopyExtParams copyParams(1, payload * BYTES_PER_FLOAT32, 0, 0, 0);
    DataCopyPadExtParams<float> padParams(true, leftPad, rightPad, 0.0f);
    DataCopyPad(xWin[leadAligned], xGM[vLo], copyParams, padParams);
}

/*!
 * \brief 行-对角形式需要的 band span 元素个数（与 host 的预算公式必须逐字一致）。
 *
 * span 覆盖本 chunk 的行块加上 k 列延伸：UPPER 向右（列 [c0, c1+k)），LOWER 向左（[c0-k, c1)）。
 * 最宽的一段是 lda * (chunkRows + k) + k + 1（最后一行再往前 k 个元素的尾巴）。
 */
/*! \brief 行-对角形式所需的 span 元素个数（含前置零哨兵）。两个 kernel 共用，故做成自由函数。
 *
 * 全程用 32 位算术：AI Core 的标量单元没有 64 位乘法指令，`uint64_t` 乘法要调软件例程，
 * 而这里是在每次下发的 Init 路径上（小 shape 的固定开销里这一项可观）。
 * issue #841 修正安全依据：host 的 ValidateSsbmvArgs 并不存在 n/lda <= 8192 的
 * 规格约束（n=3e8、k=7、lda=8 的形状可合法进入 diag-only 入口）；32 位不回绕的
 * 真实依据是准入链自身的乘积上界——本函数返回的 span 必须完整装入 UB
 * （与 host 的 SsbmvDiagSpanFloats 逐字一致、按 UB 容量装箱准入），即
 * lda * (chunkRows + k) + k + 1 连同 guard/对齐/余量以 UB 量级（约 50K floats）
 * 为上界，远小于 2^32；后续放宽准入或改 UB 容量时须同步复核此处的 32 位域。 */
__aicore__ inline uint32_t DiagSpanFloats(uint32_t chunkRows, uint32_t k, uint32_t lda)
{
    const uint32_t spanMax = lda * (chunkRows + k) + k + 1U;
    const uint32_t guard = SSBMV_ALIGN_ELEMS * lda;
    return guard + AlignUp(spanMax) + SSBMV_UB_SLACK;
}

/*!< 偏移表递推的步长（lanes）。取 64 是因为一次 Adds 正好覆盖一个 repeat（4 字节元素、
 *   8 个 block × 8 lane），既不让标量头部变长，也让大表只需要几十条指令。 */
constexpr uint32_t SSBMV_OFFSET_BLOCK = 64;

/*!
 * \brief 生成 Gather 用的字节偏移表，每组从下一行的首地址开始。
 *
 * 不用库里的 CreateVecIndex / ArithProgression —— 无 TPipe 的入口里它们必崩：两者的实现
 * （kernel_operator_vec_createvecindex_impl.h 的 CreateVecIndexOneBlk、arithprogression_
 * common_impl.h）内部都会调用 GetTPipePtr()->FetchEventID()，而没有构造过 TPipe 时那个
 * 全局指针是 null，设备侧解引用就是 507035（vector core exception）。
 *
 * 先标量写入一个对齐块，再用向量加法复制已生成的前缀。
 * 单对角线按 64 lane 递推，分组使用长向量批量扩展偏移。
 */
template <uint32_t groupWidth = 1U>
__aicore__ inline void SeedOffsetTable(
    const LocalTensor<uint32_t>& table, uint32_t rowStrideBytes, uint32_t count, uint32_t laneStrideBytes)
{
    LocalTensor<int32_t> tbl = table.ReinterpretCast<int32_t>();
    const uint32_t head = (count < SSBMV_ALIGN_ELEMS) ? count : SSBMV_ALIGN_ELEMS;
    for (uint32_t i = 0; i < head; i++) {
        tbl.SetValue(i, static_cast<int32_t>((i / groupWidth) * rowStrideBytes + (i % groupWidth) * laneStrideBytes));
    }
}

template <uint32_t groupWidth = 1U>
__aicore__ inline void ExtendOffsetTable(const LocalTensor<uint32_t>& table, uint32_t rowStrideBytes, uint32_t count)
{
    const auto tbl = table.ReinterpretCast<int32_t>();
    const uint32_t head = (count < SSBMV_ALIGN_ELEMS) ? count : SSBMV_ALIGN_ELEMS;
    for (uint32_t off = head; off < count;) {
        const uint32_t block = (groupWidth == 1U && off > SSBMV_OFFSET_BLOCK) ? SSBMV_OFFSET_BLOCK : off;
        const uint32_t lanes = (count - off < block) ? (count - off) : block;
        Adds(
            tbl[off], tbl[off - block], static_cast<int32_t>((block / groupWidth) * rowStrideBytes),
            static_cast<int32_t>(lanes));
        off += lanes;
    }
}

template <uint32_t groupWidth = 1U>
__aicore__ inline void BuildOffsetTable(
    const LocalTensor<uint32_t>& table, uint32_t rowStrideBytes, uint32_t count, uint32_t laneStrideBytes = 0U)
{
    SeedOffsetTable<groupWidth>(table, rowStrideBytes, count, laneStrideBytes);
    SetFlag<HardEvent::S_V>(EVENT_ID4);
    WaitFlag<HardEvent::S_V>(EVENT_ID4);
    ExtendOffsetTable<groupWidth>(table, rowStrideBytes, count);
}

inline aclError LaunchKernel(
    const void* symbol, uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks,
    const SsbmvTilingData& tiling, void* stream)
{
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(symbol, &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    if (function == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    struct Arguments {
        uint8_t* a;
        uint8_t* x;
        uint8_t* y;
        uint8_t* workspace;
        SsbmvTilingData tiling;
    } args{a, x, y, workSpace, tiling};
    static_assert(offsetof(Arguments, tiling) == 4U * sizeof(uint8_t*));
    static_assert(sizeof(Arguments) == 4U * sizeof(uint8_t*) + sizeof(SsbmvTilingData));
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}
} // namespace ssbmv

aclError ssbmv_diag_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks, const SsbmvTilingData& tiling,
    void* stream);
