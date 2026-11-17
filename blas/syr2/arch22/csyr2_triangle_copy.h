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
#include "common/helper/kernel_utils.h"

namespace Csyr2Triangle {
using namespace AscendC;

// Upper bands retain aligned UB columns but never transfer rows below c.
__aicore__ inline void UpperIn(
    LocalTensor<float> dst, GlobalTensor<float> a, uint32_t col, uint32_t lda, uint32_t columns = 4U)
{
    const uint32_t pitch = 2U * (col + 4U);
    for (uint32_t j = 0; j < columns; ++j) {
        DataCopyExtParams copy{1, 8U * (col + j + 1U), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0F};
        DataCopyPad(dst[pitch * j], a[2ULL * (col + j) * lda], copy, pad);
    }
}
__aicore__ inline void UpperOut(
    GlobalTensor<float> a, LocalTensor<float> src, uint32_t col, uint32_t lda, uint32_t columns = 4U)
{
    const uint32_t pitch = 2U * (col + 4U);
    for (uint32_t j = 0; j < columns; ++j) {
        DataCopyExtParams copy{1, 8U * (col + j + 1U), 0, 0, 0};
        DataCopyPad(a[2ULL * (col + j) * lda], src[pitch * j], copy);
    }
}

__aicore__ inline void LowerIn(
    LocalTensor<float> dst, GlobalTensor<float> a, uint32_t col, uint32_t row, uint32_t rows, uint32_t columns,
    uint32_t lda)
{
    if (row > col) {
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            dst, a[2ULL * (uint64_t(col) * lda + row)], 0, columns, 8U * rows, 0, 0, 8U * (lda - rows), 0);
        return;
    }
    const uint32_t pitch = (2U * rows + 7U) & ~7U;
    const uint32_t headRows = rows < 4U ? rows : 4U;
    DataCopyExtParams first{1, 8U * rows, 0, 0, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0F};
    DataCopyPad(dst, a[2ULL * (uint64_t(col) * lda + row)], first, noPad);
    for (uint32_t j = 1; j < columns; ++j) {
        DataCopyExtParams copy{1, 8U * (headRows - j), 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, static_cast<uint8_t>(2U * j), 0, 0.0F};
        DataCopyPad(dst[pitch * j], a[2ULL * (uint64_t(col + j) * lda + row + j)], copy, pad);
    }
    if (rows > 4U && columns > 1U) {
        DataCopyExtParams copy{static_cast<uint16_t>(columns - 1U), 8U * (rows - 4U), 8U * (lda - rows + 4U), 1U, 0};
        DataCopyPad(dst[pitch + 8U], a[2ULL * (uint64_t(col + 1U) * lda + row + 4U)], copy, noPad);
    }
}

__aicore__ inline void LowerOut(
    GlobalTensor<float> a, LocalTensor<float> src, LocalTensor<float> head, uint32_t col, uint32_t row, uint32_t rows,
    uint32_t columns, uint32_t lda, bool sharedHead, event_t ready)
{
    if (row > col) {
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            a[2ULL * (uint64_t(col) * lda + row)], src, 0, columns, 8U * rows, 0, 0, 0, 8U * (lda - rows));
        return;
    }
    const uint32_t pitch = (2U * rows + 7U) & ~7U;
    const uint32_t headRows = rows < 4U ? rows : 4U;
    // Slot zero holds constant byte offsets 0,4,...,28. The other three
    // aligned slots gather only valid diagonal values, entirely on the V pipe.
    for (uint32_t j = 1; j < columns; ++j)
        Gather(head[8U * j], src[pitch * j], head.ReinterpretCast<uint32_t>(), 8U * j, 2U * (headRows - j));
    set_flag(PIPE_V, PIPE_MTE3, ready);
    wait_flag(PIPE_V, PIPE_MTE3, ready);
    // Column zero already starts on its diagonal and needs no head packing.
    DataCopyExtParams first{1, 8U * rows, 0, 0, 0};
    DataCopyPad(a[2ULL * (uint64_t(col) * lda + row)], src, first);
    for (uint32_t j = 1; j < columns; ++j) {
        DataCopyExtParams copy{1, 8U * (headRows - j), 0, 0, 0};
        DataCopyPad(a[2ULL * (uint64_t(col + j) * lda + row + j)], head[8U * j], copy);
    }
    if (rows > 4U && columns > 1U) {
        DataCopyExtParams copy{static_cast<uint16_t>(columns - 1U), 8U * (rows - 4U), 1U, 8U * (lda - rows + 4U), 0};
        DataCopyPad(a[2ULL * (uint64_t(col + 1U) * lda + row + 4U)], src[pitch + 8U], copy);
    }
    if (sharedHead) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    }
}
__aicore__ inline void LowerOut(
    GlobalTensor<float> a, LocalTensor<float> src, LocalTensor<float> head, uint32_t col, uint32_t row, uint32_t rows,
    uint32_t columns, uint32_t lda, bool sharedHead = true)
{
    LowerOut(a, src, head, col, row, rows, columns, lda, sharedHead, EVENT_ID5);
}
} // namespace Csyr2Triangle
