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
#include "chbmv_vector.h"

__aicore__ inline void ChbmvVector::LoadY()
{
    if (t.betaReal == 0 && t.betaImag == 0) {
        Duplicate(yr, 0.0f, 64);
        Duplicate(yi, 0.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        // Scalar GM loads are legal for both 4-byte views and arbitrary strides.
        for (uint32_t i = 0; i < rows; ++i) {
            const int64_t at = t.yStart + (static_cast<int64_t>(start) + i) * t.incy;
            yr.SetValue(i, yGm.GetValue(2 * at));
            yi.SetValue(i, yGm.GetValue(2 * at + 1));
        }
        Sync<AscendC::HardEvent::S_V>();
        if (t.betaReal != 1 || t.betaImag != 0) {
            Scale(yr, yi, t.betaReal, t.betaImag);
            Adds(yr, vr, 0.0f, rows);
            Adds(yi, vi, 0.0f, rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

__aicore__ inline void ChbmvVector::Store()
{
    Sync<AscendC::HardEvent::V_S>();
    const uint64_t stride = t.incy < 0 ? -t.incy : t.incy;
    const bool contiguous = stride == 1;
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t lane = t.incy < 0 ? rows - 1 - i : i;
        const uint32_t pos = i * (contiguous ? 2 : 8);
        float real = yr.GetValue(lane);
        float imag = yi.GetValue(lane);
        if (!chbmv_extended::Finite(real) || !chbmv_extended::Finite(imag)) {
            chbmv_extended::Result extended{0, 0};
            chbmv_extended::Row(aGm, xGm, yGm, t, start + lane, extended);
            real = extended.real;
            imag = extended.imag;
        }
        packed.SetValue(pos, real);
        packed.SetValue(pos + 1, imag);
    }
    Sync<AscendC::HardEvent::S_MTE3>();
    const int64_t first = t.yStart + static_cast<int64_t>(t.incy < 0 ? start + rows - 1 : start) * t.incy;
    if (contiguous)
        DataCopyPad(yGm[first * 2], packed, AscendC::DataCopyExtParams{1, rows * 8, 0, 0, 0});
    else if (stride <= 0x1fffffffULL)
        DataCopyPad(
            yGm[first * 2], packed,
            AscendC::DataCopyExtParams{static_cast<uint16_t>(rows), 8, 0, static_cast<uint32_t>((stride - 1) * 8), 0});
    else
        for (uint32_t i = 0; i < rows; ++i)
            DataCopyPad(yGm[(first + i * stride) * 2], packed[i * 8], AscendC::DataCopyExtParams{1, 8, 0, 0, 0});
    Sync<AscendC::HardEvent::MTE3_S>();
    Sync<AscendC::HardEvent::S_V>();
}
