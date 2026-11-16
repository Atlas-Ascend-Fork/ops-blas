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

#include <cmath>

// ==========================================================================
// CROTG_ALGO — single-source-of-truth for the crotg algorithm body.
//
// Invoked as: CROTG_ALGO(ar, ai, br, bi, cr, sr, si, rr, ri)
// where the last 5 arguments are float* output pointers.
//
// Defined as a macro so it can be invoked from both __simt_callee__ (device)
// and regular (host) functions without SIMT calling-convention restrictions.
// Reference: https://www.netlib.org/blas/crotg.f
// ==========================================================================
// Keep the combined norm in a scaled domain: its unscaled value can exceed
// FLT_MAX even when both components of r are representable. Normalize a's phase
// separately so a much larger b does not erase a nonzero a before normalization.
// Evaluate each operand's hypot after its own scaling, so a large range gap
// does not feed two tiny scaled components to the device hypotf implementation.
// For finite inputs, the nonzero branches imply positive scales and norms.
// Check each divisor explicitly; an exceptional zero divisor leaves NaN outputs.
#define CROTG_ALGO(ar, ai, br, bi, cr, sr, si, rr, ri)                    \
    do {                                                               \
        *(cr) = NAN;                                                   \
        *(sr) = NAN;                                                   \
        *(si) = NAN;                                                   \
        *(rr) = NAN;                                                   \
        *(ri) = NAN;                                                   \
        if ((br) == 0.0f && (bi) == 0.0f) {                             \
            *(cr) = 1.0f;                                              \
            *(sr) = 0.0f;                                              \
            *(si) = 0.0f;                                              \
            *(rr) = (ar);                                              \
            *(ri) = (ai);                                              \
        } else {                                                       \
            const float aScale = fmaxf(fabsf(ar), fabsf(ai));            \
            const float bScale = fmaxf(fabsf(br), fabsf(bi));            \
            const float scale = fmaxf(aScale, bScale);                  \
            if (bScale == 0.0f || scale == 0.0f) {                      \
                break;                                                 \
            }                                                          \
            const float bsr = (br) / scale;                             \
            const float bsi = (bi) / scale;                             \
            const float bnorm = (bScale / scale) *                     \
                hypotf((br) / bScale, (bi) / bScale);                   \
            if ((ar) == 0.0f && (ai) == 0.0f) {                         \
                if (bnorm == 0.0f) {                                   \
                    break;                                             \
                }                                                      \
                *(cr) = 0.0f;                                          \
                *(sr) = bsr / bnorm;                                   \
                *(si) = -bsi / bnorm;                                  \
                *(rr) = hypotf(br, bi);                                \
                *(ri) = 0.0f;                                          \
            } else {                                                   \
                if (aScale == 0.0f) {                                  \
                    break;                                             \
                }                                                      \
                const float asr = (ar) / aScale;                        \
                const float asi = (ai) / aScale;                        \
                const float anorm = hypotf(asr, asi);                   \
                if (anorm == 0.0f) {                                   \
                    break;                                             \
                }                                                      \
                const float sgnr = asr / anorm;                         \
                const float sgni = asi / anorm;                         \
                const float scaledA = (aScale / scale) * anorm;         \
                const float norm = hypotf(scaledA, bnorm);              \
                if (norm == 0.0f) {                                    \
                    break;                                             \
                }                                                      \
                *(cr) = scaledA / norm;                                \
                *(sr) = (sgnr * bsr + sgni * bsi) / norm;              \
                *(si) = (sgni * bsr - sgnr * bsi) / norm;              \
                *(rr) = (sgnr * norm) * scale;                         \
                *(ri) = (sgni * norm) * scale;                         \
            }                                                          \
        }                                                              \
    } while (0)
