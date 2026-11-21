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
 * \file ctpsv_kernel_utils.h
 * \brief Shared helpers for ctpsv kernel files. Included only from kernel compilation units.
 */

#pragma once

#include <cstdint>

__aicore__ inline uint32_t CtpsvPackedUpperIdx(uint32_t i, uint32_t j) { return i + j * (j + 1) / 2; }

__aicore__ inline uint32_t CtpsvPackedLowerIdx(uint32_t i, uint32_t j, uint32_t n)
{
    return i + (2 * n - j - 1) * j / 2;
}

__simt_callee__ inline uint32_t CtpsvPackedUpperIdxSimt(uint32_t i, uint32_t j) { return i + j * (j + 1) / 2; }

__simt_callee__ inline uint32_t CtpsvPackedLowerIdxSimt(uint32_t i, uint32_t j, uint32_t n)
{
    return i + (2 * n - j - 1) * j / 2;
}

// Shared complex-division implementation for the scalar (CtpsvCdiv) and SIMT (CtpsvSimtCdiv)
// paths. ABSF is the abs helper available in the calling context (CtpsvAbsf on the scalar path,
// CtpsvSimtAbsf on the SIMT path). A single definition keeps the Smith divide (matching libgcc
// __divsc3, hence the reference BLAS golden) identical across both paths and avoids duplicated code.
// The defensive zero-divisor guards are unreachable in normal operation: the divisor is always a
// diagonal element kept non-zero by the host-side boost = max(5, n).
#define CTPSV_DEFINE_CDIV(MODIFIER, NAME, ABSF)                                             \
    MODIFIER inline void NAME(float ar, float ai, float br, float bi, float& cr, float& ci) \
    {                                                                                       \
        if (bi == 0.0f) {                                                                   \
            if (br == 0.0f) {                                                               \
                cr = 0.0f;                                                                  \
                ci = 0.0f;                                                                  \
                return;                                                                     \
            }                                                                               \
            cr = ar / br;                                                                   \
            ci = ai / br;                                                                   \
            return;                                                                         \
        }                                                                                   \
        if (br == 0.0f) {                                                                   \
            cr = ai / bi;                                                                   \
            ci = -ar / bi;                                                                  \
            return;                                                                         \
        }                                                                                   \
        float ratio;                                                                        \
        float den;                                                                          \
        if (ABSF(br) < ABSF(bi)) {                                                          \
            ratio = br / bi;                                                                \
            den = br * ratio + bi;                                                          \
            cr = (ar * ratio + ai) / den;                                                   \
            ci = (ai * ratio - ar) / den;                                                   \
        } else {                                                                            \
            ratio = bi / br;                                                                \
            den = bi * ratio + br;                                                          \
            cr = (ai * ratio + ar) / den;                                                   \
            ci = (ai - ar * ratio) / den;                                                   \
        }                                                                                   \
    }
