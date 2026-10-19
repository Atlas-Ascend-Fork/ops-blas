/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
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
#define CROTG_ALGO(ar, ai, br, bi, cr, sr, si, rr, ri)               \
    do {                                                               \
        float anorm = hypotf(ar, ai);                                  \
        float bnorm = hypotf(br, bi);                                  \
                                                                       \
        *(cr) = 0.0f;                                                  \
        *(sr) = 0.0f;                                                  \
        *(si) = 0.0f;                                                  \
        *(rr) = 0.0f;                                                  \
        *(ri) = 0.0f;                                                  \
                                                                       \
        if (anorm == 0.0f) {                                           \
            if (bnorm == 0.0f) {                                       \
                *(cr) = 1.0f;                                          \
            } else {                                                   \
                *(cr) = 0.0f;                                          \
                *(sr) = (br) / (bnorm);                                \
                *(si) = -(bi) / (bnorm);                               \
                *(rr) = (bnorm);                                       \
                *(ri) = 0.0f;                                          \
            }                                                          \
        } else {                                                       \
            float rnorm = hypotf(anorm, bnorm);                       \
                                                                       \
            *(cr) = (anorm) / (rnorm);                                 \
            float sgnr = (ar) / (anorm);                               \
            float sgni = (ai) / (anorm);                               \
            *(sr) = ((sgnr) * (br) + (sgni) * (bi)) / (rnorm);         \
            *(si) = ((sgni) * (br) - (sgnr) * (bi)) / (rnorm);         \
            *(rr) = (sgnr) * (rnorm);                                  \
            *(ri) = (sgni) * (rnorm);                                  \
        }                                                              \
    } while (0)
