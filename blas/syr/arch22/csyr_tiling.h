/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file csyr_tiling.h
 * \brief Shared arch22 csyr tiling definition: the by-value kernel tiling struct and the
 *        kernel launcher prototype live HERE, in one copy included by both TUs
 *        (csyr_kernel.cpp / csyr_host.cpp). Pure POD header -- only <stdint.h>.
 */

#ifndef CSYR_ARCH22_TILING_H
#define CSYR_ARCH22_TILING_H

#include <stdint.h>

// Kernel-side tiling passed BY VALUE as a launch parameter. The per-core split tables
// (startOffset/calNum) are not shipped at all -- calNum has an exact closed form under
// the zigzag map (kernel twin: csyr_cal_num_of in csyr_kernel.cpp): with full as the
// quotient of n divided by coreNum and r as the remainder, core c takes full columns
// plus exactly one extra column when r is nonzero and (full odd: c is at least coreNum
// minus r; full even: c is below r). CSYR_MAX_N is the host constant (csyr_host.cpp).
//
// This header carries the SINGLE copy of the tiling struct consumed by both TUs. The
// csyr_kernel_do launcher prototype stays in the host TU (csyr_host.cpp) -- the kernel
// TU defines the launcher with GM_ADDR (__gm__ uint8_t*) parameters, an
// address-space-qualified type that is not the same C++ type as the host-facing
// uint8_t* prototype (both TUs are bisheng-compiled with the arch macros on, so a
// macro-guarded prototype cannot separate them either).
struct CsyrKernelTiling {
    uint32_t uplo;       // 1=ACLBLAS_UPPER, 0=ACLBLAS_LOWER (kernel dispatch bit)
    uint32_t n;          // order (<= CSYR_MAX_N, checked on host); n==0 never reaches the kernel
    uint32_t lda;        // column-major leading dimension
    uint32_t incxAbs;    // |incx|; consumed HOST-side (staged read-back resolves the stride);
                         // retained in the descriptor for launch stability
    uint32_t negx;       // 1=incx<0 (host read-back maps phys=(n-1-k)*incxAbs); host-side only
    float alphaReal;     // real(alpha)
    float alphaImag;     // imag(alpha)
    uint32_t batchCols;  // columns per double-buffered batch (UB budget derived)
    uint32_t coreNum;    // cores launched (host: CSYR_VEC_CORES or the CSYR_FORCE_CORES sweep
                         // value; the kernel clamps to CSYR_MAX_CORES)
    uint32_t dbgMode;    // 0=normal; 1=V-only (no MTE3); 2=MTE3-only (no V compute);
                         // 3=prologue-only (attribution only). PERF attribution ONLY (results
                         // invalid when != 0); set via env CSYR_DBG_MODE.
    uint32_t lowerIlv;   // 1=LOWER uses the interleaved compute domain (with the head-split
                         // store fix); 0=SoA path. Host-staged form: the host sets 1 for
                         // every LOWER case and the kernel forces the same value
                         // defensively -- the field is kept for descriptor stability and
                         // budget-twin documentation.
    uint32_t barrierGate; // gated barrier-only schedule: 1 = the barrier-only column
                          // schedule (per-batch PIPE_BARRIER(ALL), no event quad --
                          // csyr_sched_barrier in csyr_kernel.cpp), 0 = the production
                          // schedule (csyr_sched_events). Host sets 1 ONLY on the band
                          // where it measures net wins: LOWER && 900<=n<=2048 (band twins
                          // CSYR_BARRIER_GATE_NMIN/NMAX). The kernel dispatch re-validates
                          // the band defensively (fail-safe: a desync falls back to gate=0).
};

#endif  // CSYR_ARCH22_TILING_H
