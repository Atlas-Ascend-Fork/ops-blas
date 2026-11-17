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
#include "kernel_operator.h"

// AoS complex tile, stored as [column][row][real,imag]. Every source column
// starts on a 32-byte boundary. Only selected GM addresses are destinations.
// scratch and indexStorage are dead computation buffers supplied by the caller;
// the caller must wait for MTE3 before reusing either the output or scratch.
namespace triangle_store {
__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }
// Rectangular UB layout, exact selected GM range. ROWS is a multiple of
// four complexes, so left-padding can align every suffix within its column.
__aicore__ inline void LoadComplexTile(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float>& gm, uint32_t pitch, uint32_t ldc, uint32_t rowBase,
    uint32_t colBase, uint32_t rows, uint32_t cols, bool upper, event_t vToMte2)
{
    AscendC::Duplicate(dst, 0.0f, pitch * cols * 2U);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    for (uint32_t col = 0U; col < cols; ++col) {
        const uint32_t absoluteCol = colBase + col;
        const uint32_t begin = upper || absoluteCol < rowBase ? 0U : MinU32(rows, absoluteCol - rowBase);
        const uint32_t end = upper ? (absoluteCol < rowBase ? 0U : MinU32(rows, absoluteCol - rowBase + 1U)) : rows;
        if (end <= begin) {
            continue;
        }
        const AscendC::DataCopyExtParams cp{1U, (end - begin) * 8U, 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{true, static_cast<uint8_t>((begin & 3U) * 2U), 0U, 0.0f};
        const uint64_t offset = (static_cast<uint64_t>(absoluteCol) * ldc + rowBase + begin) * 2ULL;
        AscendC::DataCopyPad(dst[col * pitch * 2U + (begin & ~3U) * 2U], gm[offset], cp, pad);
    }
}
template <uint32_t ROWS>
__aicore__ inline void LoadComplexSelected(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float>& gm, uint32_t ldc, uint32_t rowBase, uint32_t colBase,
    uint32_t rows, uint32_t cols, bool upper, event_t vToMte2)
{
    LoadComplexTile(dst, gm, ROWS, ldc, rowBase, colBase, rows, cols, upper, vToMte2);
}
template <uint32_t ROWS, bool SYNC_ALL = false>
__aicore__ inline void StoreComplexDiagonal(
    AscendC::GlobalTensor<float>& gm, AscendC::LocalTensor<float> output, AscendC::LocalTensor<float> scratch,
    AscendC::LocalTensor<int32_t> indexStorage, uint32_t ldc, uint32_t rowBase, uint32_t colBase, uint32_t rows,
    uint32_t cols, bool upper, event_t vToMte3)
{
    constexpr uint32_t FLOATS = ROWS * 2U;
    if (!upper) {
        // On arch22 CreateVecIndex seeds its first block with scalar
        // SetValue stores. Finish prior vector reads of this reused buffer
        // before the scalar pipeline overwrites it.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vToMte3);
        // Four residue tables remove an unaligned complex prefix without
        // floating-point arithmetic. All gathers have aligned UB bases.
        AscendC::CreateVecIndex(indexStorage, 0, FLOATS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(indexStorage, indexStorage, static_cast<int32_t>(sizeof(float)), FLOATS);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t residue = 1U; residue < 4U; ++residue) {
            AscendC::Adds(
                indexStorage[residue * FLOATS], indexStorage, static_cast<int32_t>(residue * 2U * sizeof(float)),
                FLOATS);
        }
        AscendC::PipeBarrier<PIPE_V>();
        auto indices = indexStorage.ReinterpretCast<uint32_t>();
        for (uint32_t col = 0U; col < cols; ++col) {
            const uint32_t begin = MinU32(rows, colBase + col - rowBase);
            const uint32_t count = rows - begin;
            if (count != 0U) {
                AscendC::Gather(
                    scratch[col * FLOATS], output[col * FLOATS + (begin & ~3U) * 2U], indices[(begin & 3U) * FLOATS],
                    0U, count * 2U);
            }
        }
        if constexpr (SYNC_ALL) {
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
        }
    }
    for (uint32_t col = 0U; col < cols; ++col) {
        const uint32_t absCol = colBase + col;
        const uint32_t begin = upper ? 0U : MinU32(rows, absCol - rowBase);
        const uint32_t count = upper ? MinU32(rows, absCol - rowBase + 1U) : rows - begin;
        if (count == 0U) {
            continue;
        }
        const uint64_t offset = (static_cast<uint64_t>(absCol) * ldc + rowBase + begin) * 2ULL;
        const AscendC::DataCopyExtParams cp{1U, count * 8U, 0U, 0U, 0U};
        AscendC::DataCopyPad(gm[offset], upper ? output[col * FLOATS] : scratch[col * FLOATS], cp);
    }
}
} // namespace triangle_store
