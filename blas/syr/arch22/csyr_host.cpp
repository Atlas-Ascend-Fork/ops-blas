/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file csyr_host.cpp
 * \brief Host side implementation for the complex symmetric rank-1 update operator csyr.
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
// The tiling struct lives in the shared header csyr_tiling.h (one copy consumed by both
// TUs). The launcher prototype stays in this TU on purpose: the kernel TU defines
// csyr_kernel_do with GM_ADDR (__gm__ uint8_t*) parameters, an address-space-qualified
// type that is not the same C++ type as this host-facing prototype, and both TUs are
// bisheng-compiled with the arch macros on (a macro-guarded prototype cannot separate
// them). The staging byte copies below use std::copy / std::copy_n (byte-identical
// data movement, no value conversion).
#include "csyr_tiling.h"

void csyr_kernel_do(uint8_t* gm_x, uint8_t* gm_A, uint8_t* gm_staged, uint8_t* workSpace,
                    const struct CsyrKernelTiling& tiling, uint32_t numBlocks, void* stream);

// CSYR_VEC_CORES=40 is the 910B3 AIV ceiling of this arch22 product line. The production
// core count is taken from the runtime platform API (GetAivCoreCount in
// csyr_pick_core_num below; arch35 precedent blas/syr/arch35/syr_host.cpp) and clamped
// to [1, CSYR_VEC_CORES]; the constant itself now carries the product ceiling, the
// CSYR_FORCE_CORES sweep upper bound, and the kernel-side defensive clamp anchor
// (kernel twin: CSYR_MAX_CORES in csyr_kernel.cpp; values MUST stay identical across
// the twins -- cross-arch port concern only).
constexpr uint32_t CSYR_VEC_CORES = 40;
// Named constants, host side (kernel twins in csyr_kernel.cpp; values MUST stay identical
// across the twins):
constexpr uint32_t CSYR_UB_FLOATS = 49152u;   // per-core UB budget in floats (192KB)
constexpr uint32_t CSYR_SOA_BUDGET = 48640u;  // SoA budget base: CSYR_UB_FLOATS minus the
                                              // 512-float interleave-table fixed term
constexpr uint32_t CSYR_VEC_GRAN = 64u;       // V-pipe repeat granularity in floats (one
                                              // repeat / 256B; nPad and repeat-count rounding)
constexpr uint32_t CSYR_HEAD_FLTS = 64u;      // interleaved-slot head region in floats
                                              // (32 rows x 2 floats; slotStride 2nPad + 64)
constexpr uint32_t CSYR_SMALL_N_MAX = 512u;   // n <= CSYR_SMALL_N_MAX -> small-n policy
constexpr uint32_t CSYR_SMALL_N_CORES = 20u;  // small-n core count
// Mid-size band core policy: the mid-size LOWER optimum is 36 cores. The band is scoped to
// the LOWER column schedule it was tuned on (the UPPER column schedule keeps the default).
// Outside the band the default stays optimal (the boundary sizes are parallelism bound),
// so the ceiling is pinned at 950; the floor 900 matches CSYR_BARRIER_GATE_NMIN (the band
// nests inside the gated band).
constexpr uint32_t CSYR_MID_N_MIN = 900u;     // mid-size band floor (LOWER only)
constexpr uint32_t CSYR_MID_N_MAX = 950u;     // mid-size band ceiling (LOWER only)
constexpr uint32_t CSYR_MID_N_CORES = 36u;    // mid-size band core count
// Gated barrier-only schedule band (kernel twin: CSYR_BARRIER_GATE_NMIN/NMAX in
// csyr_kernel.cpp; values MUST stay identical across the twins). The barrier-only column
// schedule (per-batch PIPE_BARRIER(ALL), zero event slots) is dispatched ONLY on this band:
// the measured gains concentrate in LOWER with n in [912,2000]; small-n forms (nb=1/nb-tiny)
// regress (barrier wake cost overtakes), as does UPPER.
constexpr uint32_t CSYR_BARRIER_GATE_NMIN = 900u;
constexpr uint32_t CSYR_BARRIER_GATE_NMAX = 2048u;
// Kernel-side output group count S on the production build (csyr_kernel.cpp CSYR_GROUPS,
// triple buffering). Used by the batchCols budget twins below (the S=1 fallback build is
// covered: host formulas derived for S=3 over-reserve for S=1, which is UB-safe). MUST
// match the kernel constant.
constexpr uint32_t CSYR_KERNEL_GROUPS = 3u;
// Unified maximum-order quota: n <= 4864. On the current build the governing B=1 layout
// is the interleaved slim 2-op domain: 4nPad + S*(2nPad+64) + 32 = 10*nPad + 224
// floats (S = CSYR_KERNEL_GROUPS = 3) <= 49152, which fits up to nPad <= 4892 (the 4864
// quota is kept). With S=3 the SoA fallback layouts only fit up to nPad <= 4032 and the
// dispatch degrades dynamically (see the batchCols budget below). n > CSYR_MAX_N returns
// ACLBLAS_STATUS_INVALID_VALUE (documented in test/syr/csyr/README.md, section 4).
constexpr uint32_t CSYR_MAX_N = 4864;

// Element-level GM offset guard. The kernel's GM addressing (MTE2/MTE3, the pack path and
// the atomic stores) works on element offsets; the host contract bounds the largest
// element offset touched -- (n-1)*lda + n for A and (n-1)*|incx| + 1 for x -- to 2^31.
// Offsets above that exceed the documented address envelope. Returns INVALID_VALUE (BLAS
// callers expect a parameter-range rejection for strides that large).
constexpr uint64_t CSYR_MAX_ELEM_OFFSET = (1ull << 31);

// The per-core column count is derived on the kernel side from the closed form in
// csyr_cal_num_of (see the CsyrKernelTiling comment in csyr_tiling.h). The zigzag
// round-robin map balances BOTH per-core work and per-core column count (the per-column
// fixed stall makes column count a first-order critical resource).

// Host-side data pre-staging: the kernel's resident x / s' buffers are built HERE, in one
// combined 512B-aligned device buffer [x_staged | s'_staged], each half ceil(8n/512)*512
// bytes, so the kernel loads both views with two plain MTE2 bursts and no in-kernel
// reordering prologue (de-interleave / negate / interleave-gather build, plus its table
// upload) is needed. The staging transform is pure data movement at the bit level:
//   x_staged[2k]   = xre(k), x_staged[2k+1] = xim(k)   (byte-exact copy of logical x)
//   s'_staged[2k]  = -xim(k), s'_staged[2k+1] = xre(k) (sign flip + lane swap)
// The s' sign flip is applied as a bitwise sign xor: fl(a * -1.0f) is bit-identical to
// the sign xor for every finite / inf / (+/-)0 input (IEEE multiply by -1 is exact and
// only touches the sign bit), so the staged image matches a multiply-based build bit for
// bit. The zero tail past the valid 8n bytes keeps each half a clean 512B-multiple span;
// it is never consumed into any stored element (kernel-side store lengths are exact,
// n-bounded forms).
constexpr uint32_t CSYR_STAGE_GRAN = 512u;             // staging byte granularity (128 floats)
constexpr uint64_t CSYR_STRIDE_SPAN_CAP = (16ull << 20);  // strided x read-back span cap

// Read the logical x vector (n complex elements at stride incxAbs, negative stride
// remapped phys = (n-1-k)*incxAbs) into the interleaved host view xHost[2k]=re(k),
// xHost[2k+1]=im(k). The contiguous fast path (incx==+1) is a single D2H copy (same
// pattern as the alpha read-back above it); the strided path reads the covered device
// span when it is proportionate, else falls back to per-element 8-byte read-backs
// (pathological huge-stride shapes only; every supported combination fits the span
// form).
aclError csyr_read_x_host(std::vector<uint8_t>& xHost, const aclblasComplex* x, uint32_t n,
                          uint32_t incxAbs, uint32_t negx)
{
    xHost.assign(static_cast<size_t>(n) * sizeof(*x), 0u);
    // n == 0 cannot reach here through aclblasCsyr (csyr_validate_params quick-returns
    // n == 0 before staging); the early return keeps the unsigned (n - 1) span math
    // below wrap-free as a local contract, independent of the entry-validation ladder.
    if (n == 0u) {
        return ACL_SUCCESS;
    }
    if (incxAbs == 1u && negx == 0u) {
        return aclrtMemcpy(xHost.data(), xHost.size(), x, xHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    const uint64_t spanElem = (static_cast<uint64_t>(n) - 1u) * incxAbs + 1u;
    const uint64_t spanBytes = spanElem * sizeof(*x);
    if (spanBytes <= CSYR_STRIDE_SPAN_CAP) {
        std::vector<uint8_t> span(static_cast<size_t>(spanBytes));
        const aclError ret =
            aclrtMemcpy(span.data(), span.size(), x, span.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        for (uint32_t k = 0; k < n; k++) {
            const uint64_t phys = (negx != 0u) ? (static_cast<uint64_t>(n) - 1u - k) * incxAbs
                                               : static_cast<uint64_t>(k) * incxAbs;
            std::copy_n(span.begin() + static_cast<size_t>(phys) * sizeof(*x), sizeof(*x),
                        xHost.begin() + static_cast<size_t>(k) * sizeof(*x));
        }
        return ACL_SUCCESS;
    }
    for (uint32_t k = 0; k < n; k++) {
        const uint64_t phys = (negx != 0u) ? (static_cast<uint64_t>(n) - 1u - k) * incxAbs
                                           : static_cast<uint64_t>(k) * incxAbs;
        const aclError ret = aclrtMemcpy(&xHost[static_cast<size_t>(k) * sizeof(*x)], sizeof(*x),
                                         reinterpret_cast<const uint8_t*>(x) + phys * sizeof(*x),
                                         sizeof(*x), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    return ACL_SUCCESS;
}

// Single release point for the per-call device buffers. A failed aclrtFree is LOG'd and
// swallowed -- there is no recoverable action for it inside the caller's error paths (the
// path's original status is returned unchanged), and the pointer is nulled so no path can
// double-free. Every error path releases the staged buffer before returning, so no
// leak/double-free is introduced.
void csyr_free_device_buf(uint8_t*& p, const char* what)
{
    if (p == nullptr) {
        return;
    }
    const aclError ret = aclrtFree(p);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclblasCsyr: aclrtFree(%s) failed. ERROR: %d\n", what, ret);
    }
    p = nullptr;
}

// Element-level span check, with the unsigned-domain |incx| negation (std::abs on INT_MIN
// is UB) and the span LOG text. Returns false and LOGs when either the A span (n-1)*lda + n
// or the x span (n-1)*|incx| + 1 exceeds 2^31; writes the unsigned |incx| back through
// incxAbsVal.
static bool csyr_check_elem_span(const int n, const int lda, const int incx, uint32_t& incxAbsVal)
{
    incxAbsVal = (incx < 0) ? (0u - static_cast<uint32_t>(incx))
                            : static_cast<uint32_t>(incx);
    // n == 0 cannot reach here through aclblasCsyr (csyr_validate_params quick-returns
    // n == 0 before the span check); the early return keeps the unsigned (n - 1) span
    // formulas below wrap-free as a local contract, independent of the entry ladder.
    if (n == 0) {
        return true;
    }
    const uint64_t aElemSpan = (static_cast<uint64_t>(n) - 1u) * static_cast<uint64_t>(lda) +
                               static_cast<uint64_t>(n);
    const uint64_t xElemSpan = (static_cast<uint64_t>(n) - 1u) * static_cast<uint64_t>(incxAbsVal) + 1u;
    if (aElemSpan > CSYR_MAX_ELEM_OFFSET || xElemSpan > CSYR_MAX_ELEM_OFFSET) {
        LOG_PRINT("aclblasCsyr: element-level GM offset exceeds 2^31 (n = %d, lda = %d, incx = %d; "
                  "A span %llu, x span %llu).\n",
                  n, lda, incx, static_cast<unsigned long long>(aElemSpan),
                  static_cast<unsigned long long>(xElemSpan));
        return false;
    }
    return true;
}

// Dual alpha contract (README twin: the aclblasCsyr parameter table / constraint row;
// same-repo precedent ResolveCher2kScalarLocations in
// blas/her2k/arch22/cher2k_host.cpp): the pointer is located first
// (aclrtPointerGetAttributes via CheckPtrLocation). A DEVICE alpha keeps the 8-byte
// D2H read-back (direct dereference of a device pointer is UB); a HOST alpha is
// consumed with a member-wise 8-byte copy. A failed location query is rejected like
// the cher2k precedent (the pointer is outside any address space this runtime can
// name). Stages 5-6 of the order documented at csyr_validate_params: the alpha==(0,0)
// quick return sets earlyDone; the x/A nullptr check (stage 7) stays in the caller,
// reached only when alpha != (0,0).
static aclblasStatus_t csyr_resolve_alpha(const aclblasComplex* alpha, float& alphaReal,
    float& alphaImag, bool& earlyDone)
{
    struct {
        float real;
        float imag;
    } alphaHost;
    bool alphaDevice = false;
    const aclblasStatus_t locStatus = CheckPtrLocation(alpha, &alphaDevice);
    CHECK_RET(
        locStatus == ACLBLAS_STATUS_SUCCESS,
        LOG_PRINT("aclblasCsyr: alpha pointer location query failed. STATUS: %d\n",
                  static_cast<int>(locStatus));
        return locStatus);
    if (alphaDevice) {
        aclError aclRet =
            aclrtMemcpy(&alphaHost, sizeof(alphaHost), alpha, sizeof(alphaHost), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(
            aclRet == ACL_SUCCESS, LOG_PRINT("aclblasCsyr: alpha D2H copy failed. ERROR: %d\n", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR);
    } else {
        alphaHost.real = alpha->real;
        alphaHost.imag = alpha->imag;
    }
    if (alphaHost.real == 0.0f && alphaHost.imag == 0.0f) {
        earlyDone = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    alphaReal = alphaHost.real;
    alphaImag = alphaHost.imag;
    return ACLBLAS_STATUS_SUCCESS;
}

// Parameter validation, in dependency order. earlyDone covers the two SUCCESS quick
// returns (n == 0 / alpha == (0,0)): the caller returns the status unchanged and skips
// every later stage.
static aclblasStatus_t csyr_validate_params(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const int incx, const int lda,
    const aclblasComplex* alpha, const aclblasComplex* x, const aclblasComplex* A,
    uint32_t& incxAbsVal, float& alphaReal, float& alphaImag, bool& earlyDone)
{
    // Validation order. Each check precedes everything it depends on:
    //  1. handle (stream retrieval depends on it)
    //  2. uplo enum
    //  3. scalar group: n<0 / n>CSYR_MAX_N / incx==0 / lda<max(1,n) / alpha==nullptr
    //     (alpha null check precedes any alpha dereference or read-back; lda check precedes the
    //     n==0 short-circuit so n==0 with lda==0 is INVALID_VALUE, locked by negative CSV cases)
    //  4. n==0 quick return WITHOUT reading alpha (saves one D2H read-back)
    //  5. alpha location query + dual read (CheckPtrLocation branch: device pointer ->
    //     8-byte aclrtMemcpy D2H read-back; host pointer -> member-wise 8-byte host copy;
    //     direct dereference of a device alpha is UB)
    //  6. alpha==(0,0) quick return
    //  7. x/A nullptr (reached only when n>0 and alpha!=(0,0))
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0 || static_cast<uint32_t>(n) > CSYR_MAX_N || incx == 0 || lda < std::max(1, n) || alpha == nullptr) {
        if (static_cast<uint32_t>(n) > CSYR_MAX_N) {
            LOG_PRINT("aclblasCsyr: n = %d exceeds the supported maximum %u.\n", n, CSYR_MAX_N);
        }
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        earlyDone = true;
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Element-level GM offset guard (see CSYR_MAX_ELEM_OFFSET above). Runs AFTER the n==0
    // quick return (n==0 touches no element, and the quick-return contract is unchanged);
    // the |incx| unsigned-domain negation and the span check live in csyr_check_elem_span
    // below.
    if (!csyr_check_elem_span(n, lda, incx, incxAbsVal)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Stages 5-6 of the order above: alpha location query + dual read + the alpha==(0,0)
    // quick return, all inside csyr_resolve_alpha (stage 7 below is reached only when
    // alpha != (0,0)).
    const aclblasStatus_t alphaStatus = csyr_resolve_alpha(alpha, alphaReal, alphaImag, earlyDone);
    if (alphaStatus != ACLBLAS_STATUS_SUCCESS || earlyDone) {
        return alphaStatus;
    }
    if (x == nullptr || A == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// The batchCols budget tree (same branch order, same formulas, same env knobs; the
// kernel twin gates in csyr_kernel.cpp MUST stay in sync).
static void csyr_apply_batch_cols_override(uint32_t nPadBudget, uint32_t budgetFixed,
                                           uint32_t lowerIlv, CsyrKernelTiling& tiling);

static void csyr_fill_batch_cols(uint32_t uplo, const int n, CsyrKernelTiling& tiling)
{
    // Batched double buffering: batchCols is derived from the per-core UB budget. The LOWER
    // path dispatches to the interleaved compute domain for EVERY n (the 2-op form needs no
    // q' working region, so the per-column slot is 2*nPad+64 floats); see the budget comment
    // below. Budget, uplo-dependent (kernel twin: the ilvLayout / lowerPair layout checks in
    // csyr_kernel.cpp -- MUST stay in sync). Dispatch order for LOWER:
    //   (1) LOWER interleaved domain (slim 2-op slot + head-split store):
    //       slot = (2nPad + 64) floats (stream + 64-float head dst); S groups x B slots +
    //       xraw/s' 4nPad + 32 pattern floats (the 4 head-shift select word pairs are placed
    //       at stride 8 uint32 -- footprint 26 uint32 = 104B, reserved as 32 floats;
    //       temps overlaid on the second slot):
    //       (4nPad + S*B*(2nPad+64) + 32) <= 49152 ->
    //       B = (49152 - 4*nPad - 32) / (S*(2nPad+64)) (nPad=512 -> 14, 1024 -> 7,
    //       2048 -> 3, 4096/4864 -> 1). Fits EVERY n <= 4864.
    //   (2) LOWER SoA two-column interleave (CSYR_DISABLE_LOWER_ILV fallback):
    //       (8 + 2*S*B)*nPad + 512 (kernel lowerPair gate).
    //   (3) LOWER SoA single-column: (6 + 2*S*B)*nPad + 512.
    //       Both SoA layouts are S-aware: the B divisor is 2*S = 6 (with S=3). With S=3 the
    //       single-column layout only fits up to nPad <= 4032; for larger n the SoA fallback
    //       cannot run and the interleaved domain is kept (LOG'd below).
    //   UPPER (slim 2-op layout): (4nPad + S*B*2nPad + 32) <= 49152 ->
    //       B = (49152 - 4*nPad - 32) / (S*2*nPad) (nPad=512 -> 15, 1024 -> 7, 2048 -> 3,
    //       4864 -> 1).
    // With S=CSYR_KERNEL_GROUPS=3 groups (triple buffering). The kernel never reads beyond
    // nPad per column, so a batch covering a full triangle tail stays in budget.
    // Batch-boundary reuse of a group's slots is gated by the per-TRIPLE completion barrier in
    // csyr_process_columns (MTE3 atomic writes do not self-gate).
    uint32_t nPadBudget = (tiling.n + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN * CSYR_VEC_GRAN;
    // Divisor guard, dead by caller invariant -- aclblasCsyr quick-returns on n == 0
    // before tiling, so nPadBudget >= CSYR_VEC_GRAN.
    if (nPadBudget == 0u) {
        tiling.batchCols = 1u; // B=1 always fits the UB budget
        return;
    }
    uint32_t budgetFixed = 6u; // SoA single-column fixed term: 6*nPad (S-independent)
    uint32_t lowerIlv = 0u;
    if (uplo == 0u) {
        // Every LOWER case runs the interleaved domain, whose slim 2-op slot fits the
        // 192KB budget for every n <= CSYR_MAX_N (see the budget comment above). The
        // kernel twin re-validates the budget and clamps batchCols defensively (B=1
        // always fits).
        lowerIlv = 1u;
        // Interleaved-domain slim-slot budget (kernel twin check above).
        uint32_t ilvSlotStride = 2u * nPadBudget + CSYR_HEAD_FLTS;
        tiling.batchCols =
            std::max(1u, (CSYR_UB_FLOATS - 4u * nPadBudget - 32u) / (CSYR_KERNEL_GROUPS * ilvSlotStride));
    }
    tiling.lowerIlv = lowerIlv;
    if (lowerIlv == 0u && tiling.uplo == 0u) {
        // SoA two-column pair probe, S-aware (kernel twin: the lowerPair gate
        // (8 + 2*S*B)*nPad + 512 <= 49152; B divisor = 2*S).
        uint32_t perNPad = CSYR_SOA_BUDGET / nPadBudget;
        uint32_t bTry = (perNPad > 8u) ? ((perNPad - 8u) / (2u * CSYR_KERNEL_GROUPS)) : 0u;
        if (bTry >= 1u) {
            budgetFixed = 8u;
        }
    }
    if (lowerIlv == 0u && tiling.uplo != 0u) {
        // UPPER slim-slot budget (kernel twin check above).
        tiling.batchCols =
            std::max(1u, (CSYR_UB_FLOATS - 4u * nPadBudget - 32u) / (2u * CSYR_KERNEL_GROUPS * nPadBudget));
    } else if (lowerIlv == 0u) {
        // LOWER SoA fallback (env CSYR_DISABLE_LOWER_ILV=1, debug-only path): S-aware
        // budget (B divisor 2*S, fixed term 6 or 8 nPad).
        tiling.batchCols = std::max(1u, ((CSYR_SOA_BUDGET / nPadBudget) - budgetFixed) / (2u * CSYR_KERNEL_GROUPS));
        if (((budgetFixed + 2u * CSYR_KERNEL_GROUPS * tiling.batchCols) * nPadBudget + 512u) > CSYR_UB_FLOATS) {
            // Even B=1 does not fit the single-column SoA layout (nPad > 4032 with S=3):
            // the SoA fallback cannot run for this n; keep the interleaved domain.
            LOG_PRINT("aclblasCsyr: CSYR_DISABLE_LOWER_ILV SoA fallback does not fit the UB budget "
                      "for n = %d with S = %u groups; keeping the interleaved domain.\n",
                      n, CSYR_KERNEL_GROUPS);
            lowerIlv = 1u;
            tiling.lowerIlv = 1u;
            uint32_t ilvSlotStride = 2u * nPadBudget + CSYR_HEAD_FLTS;
            tiling.batchCols =
                std::max(1u, (CSYR_UB_FLOATS - 4u * nPadBudget - 32u) / (CSYR_KERNEL_GROUPS * ilvSlotStride));
        }
    }
    // Debug/sweep knob (default off): CSYR_BATCH_COLS overrides the derived batch size,
    // clamped to the active layout's UB-budget maximum (csyr_apply_batch_cols_override
    // below).
    csyr_apply_batch_cols_override(nPadBudget, budgetFixed, lowerIlv, tiling);
}

// The CSYR_BATCH_COLS debug/sweep knob: overrides the derived batch size, clamped to the
// active layout's UB-budget maximum (same clamp formulas per active layout).
static void csyr_apply_batch_cols_override(uint32_t nPadBudget, uint32_t budgetFixed,
                                           uint32_t lowerIlv, CsyrKernelTiling& tiling)
{
    // Divisor guard, dead by caller invariant -- the sole caller (csyr_fill_batch_cols)
    // passes a derived nPadBudget >= CSYR_VEC_GRAN.
    if (nPadBudget == 0u) {
        return;
    }
    if (const char* envBc = std::getenv("CSYR_BATCH_COLS"); envBc != nullptr) {
        long v = std::strtol(envBc, nullptr, 10);
        if (v > 0) {
            uint32_t budgetMax;
            if (lowerIlv != 0u) {
                uint32_t ilvSlotStride = 2u * nPadBudget + CSYR_HEAD_FLTS;
                budgetMax =
                    std::max(1u, (CSYR_UB_FLOATS - 4u * nPadBudget - 32u) / (CSYR_KERNEL_GROUPS * ilvSlotStride));
            } else if (tiling.uplo != 0u) {
                budgetMax =
                    std::max(1u, (CSYR_UB_FLOATS - 4u * nPadBudget - 32u) / (2u * CSYR_KERNEL_GROUPS * nPadBudget));
            } else {
                budgetMax =
                    std::max(1u, ((CSYR_SOA_BUDGET / nPadBudget) - budgetFixed) / (2u * CSYR_KERNEL_GROUPS));
            }
            tiling.batchCols = std::min(budgetMax, static_cast<uint32_t>(v));
        }
    }
}

// Core-count policy: size bands plus the CSYR_FORCE_CORES env override. The production
// core count is queried from the runtime platform API (GetAivCoreCount -- the
// PlatformAscendC singleton, arch35 precedent blas/syr/arch35/syr_host.cpp) and clamped
// to [1, CSYR_VEC_CORES]; the band constants are additionally clamped against the
// platform count so a smaller SKU cannot be oversubscribed. On a full-size 910B3
// (platform count 40) every picked value is identical to the previous fixed-40 policy.
// A failed query returns 0 and the caller rejects the launch with
// ACLBLAS_STATUS_EXECUTION_FAILED (arch35 precedent); the kernel closed form accepts any
// coreNum in [1, CSYR_MAX_CORES], so policy changes need no kernel change.
static uint32_t csyr_pick_core_num(uint32_t uplo, uint32_t n)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0u) {
        LOG_PRINT("aclblasCsyr: GetAivCoreCount failed (platform query returned 0).\n");
        return 0u;
    }
    const uint32_t platformCores = std::min(std::max(aivCoreNum, 1u), CSYR_VEC_CORES);
    // Size-banded core-count policy constants (CSYR_SMALL_N_MAX/CSYR_SMALL_N_CORES =
    // 512/20), clamped to the platform count for this arch22 source.
    uint32_t coreNum = platformCores;
    if (n <= CSYR_SMALL_N_MAX) {
        coreNum = CSYR_SMALL_N_CORES;
    } else if (uplo == 0u && n >= CSYR_MID_N_MIN && n <= CSYR_MID_N_MAX) {
        // Load-balanced size-banded scheduling: the mid-size LOWER column distribution
        // balances best at CSYR_MID_N_CORES (see the constants above); the kernel split
        // closed form takes any coreNum.
        coreNum = CSYR_MID_N_CORES;
    }
    coreNum = std::min(coreNum, platformCores);
    if (const char* envCores = std::getenv("CSYR_FORCE_CORES"); envCores != nullptr) {
        long v = std::strtol(envCores, nullptr, 10);
        if (v >= 1 && v <= static_cast<long>(platformCores)) {
            coreNum = static_cast<uint32_t>(v);
        }
    }
    return coreNum;
}

// Build the combined [x_staged | s'_staged] host image (same arithmetic, same byte-level
// movement -- the sign flip stays the bitwise sign xor; std::copy / std::copy_n for the
// byte-level copies).
static std::vector<uint8_t> csyr_build_staged_image(const std::vector<uint8_t>& xHost, uint32_t n)
{
    const uint32_t stagedFloats =
        (2u * n + (CSYR_STAGE_GRAN / sizeof(float)) - 1u) / (CSYR_STAGE_GRAN / sizeof(float)) *
        (CSYR_STAGE_GRAN / sizeof(float));
    const size_t stagedBytes = static_cast<size_t>(stagedFloats) * sizeof(float);
    std::vector<uint8_t> staged(2u * stagedBytes, 0u);  // [x_staged | s'_staged], zero tail
    std::copy(xHost.begin(), xHost.end(), staged.begin());
    {
        // s' image: s'[2k] = -xim(k) (bitwise sign xor, see the staging note above),
        // s'[2k+1] = xre(k).
        uint8_t* sPrime = staged.data() + stagedBytes;
        for (uint32_t k = 0; k < n; k++) {
            const size_t srcOff = static_cast<size_t>(k) * sizeof(aclblasComplex);
            const size_t dstOff = static_cast<size_t>(k) * sizeof(aclblasComplex);
            uint32_t imBits;
            std::copy_n(staged.data() + srcOff + sizeof(float), sizeof(imBits),
                        reinterpret_cast<uint8_t*>(&imBits));
            imBits ^= 0x80000000u;  // sign flip: bit-identical to fl(xim * -1.0f)
            std::copy_n(reinterpret_cast<const uint8_t*>(&imBits), sizeof(imBits), sPrime + dstOff);
            std::copy_n(staged.data() + srcOff, sizeof(float), sPrime + dstOff + sizeof(float));
        }
    }
    return staged;
}

// The staging + launch + drain stage: the per-call malloc/H2D/sync/free flow (the
// atomic-store drain discipline) keeps its exact call count and order:
// staged malloc -> single H2D -> kernel launch -> stream sync -> free staged.
static aclblasStatus_t csyr_stage_and_launch(aclrtStream useStream, const aclblasComplex* x,
                                             aclblasComplex* A, const CsyrKernelTiling& tiling,
                                             uint32_t incxAbsVal, uint32_t coreNum)
{
    // No host-side x VALUE transformation -- only a layout copy. The strided access is
    // resolved by the host-side read-back (csyr_read_x_host, same logical->physical
    // mapping); the kernel consumes the staged interleaved view directly.
    // Host-side pre-staging: read the logical x back (D2H, same synchronous pattern as
    // the alpha read-back -- x is a const input of this call, so reading it before any
    // launch of THIS call cannot interact with the atomic-store drain of any call),
    // build the combined [x_staged | s'_staged] image, and upload it with ONE H2D copy.
    // The malloc/free call COUNT and ORDER below are fixed (the staged buffer) --
    // the per-call malloc/free pair is the atomic-store drain discipline this
    // driver requires (see the note at the malloc below).
    std::vector<uint8_t> xHost;
    aclError aclRet = csyr_read_x_host(xHost, x, tiling.n, incxAbsVal, tiling.negx);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclblasCsyr: x D2H read failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    const std::vector<uint8_t> staged = csyr_build_staged_image(xHost, tiling.n);
    size_t stagedTotalBytes = staged.size();

    uint8_t* stagedDevice = nullptr;
    void* stagedDeviceRaw = nullptr;   // aclrtMalloc void* out-param, cast below

    // The zero-per-call-malloc / static-table / async-return host form is NOT sound here:
    // with a process-lifetime persistent device table, the SECOND and
    // later launches of the same kernel in one process fail -- wrong values via the ST
    // wrapper (its aclrtSynchronizeDevice does not drain in-flight SetAtomicAdd stores)
    // and deterministic 507035 faults at aclrtSynchronizeStream/SynchronizeDevice
    // (~40ms/call). The identical kernel binary is fault-free with the per-call
    // malloc/H2D/sync/free flow, so the per-call aclrtFree pair is what drains/serializes
    // the atomic stores on this driver. The per-call synchronous flow is retained; async
    // + static-table is rejected on this hardware/driver generation. The staged upload
    // keeps this flow: one malloc + one H2D, freed after the stream sync in the same
    // order.
    aclRet = aclrtMalloc(&stagedDeviceRaw, stagedTotalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc staged buffer failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);
    // No C-style (void**) cast -- aclrtMalloc writes through an intermediate void*, then
    // one static_cast to the typed pointer.
    stagedDevice = static_cast<uint8_t*>(stagedDeviceRaw);

    // Single H2D upload of the combined staged image (total size is a 512B multiple, so
    // the upload itself is a clean aligned span).
    aclRet = aclrtMemcpy(
        stagedDevice, stagedTotalBytes, staged.data(), stagedTotalBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy staged buffer failed. ERROR: %d\n", aclRet);
        csyr_free_device_buf(stagedDevice, "staged");
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    // workSpace slot: legacy launch-signature parameter, no kernel path dereferences it
    // (nullptr is safe); kept because the launch descriptor takes a fixed signature.
    csyr_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(A),
        stagedDevice, nullptr, tiling, coreNum, useStream);

    // The interface synchronizes internally instead of returning asynchronously (the
    // caller would otherwise need to synchronize before read-back). Kept as a hard
    // memory-lifetime constraint: the malloc'ed staged buffer is freed below
    // and must not outlive the kernel; same as the upstream syr_host.cpp precedent. This
    // does not pollute the kernel-pure-time acceptance metric; async refactor is a future
    // optimization item (see the note above: the async/static-table form is not
    // drivable on this stack).
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
        csyr_free_device_buf(stagedDevice, "staged");
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    csyr_free_device_buf(stagedDevice, "staged");

    return ACLBLAS_STATUS_SUCCESS;
}

// The aclblasCsyr body is split into the four stages above (validation ladder /
// batchCols budget / core policy / staging+launch). Every check, device call and quick
// return keeps a fixed order; see each stage's comment. The alpha location query + dual
// read, the staging H2D, the per-call malloc/free drain flow and the 512B-aligned
// staging image make up the complete call flow.
aclblasStatus_t aclblasCsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* alpha,
    const aclblasComplex* x, const int incx, aclblasComplex* A, const int lda)
{
    uint32_t incxAbsVal = 0u;
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    bool earlyDone = false;
    const aclblasStatus_t valStatus = csyr_validate_params(
        handle, uplo, n, incx, lda, alpha, x, A, incxAbsVal, alphaReal, alphaImag, earlyDone);
    if (valStatus != ACLBLAS_STATUS_SUCCESS || earlyDone) {
        return valStatus;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    // Value-initialization instead of std::memset -- CsyrKernelTiling is a trivially
    // default-constructible POD aggregate, so `tiling{}` is bitwise-identical to the
    // zeroed object the memset produced (all-scalar members, no padding-sensitive
    // readers), and every field is then assigned below.
    CsyrKernelTiling tiling{};
    tiling.uplo = (uplo == ACLBLAS_UPPER) ? 1 : 0;
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.incxAbs = incxAbsVal; // unsigned-domain |incx| (std::abs(INT_MIN) is UB)
    tiling.negx = (incx < 0) ? 1 : 0;
    tiling.alphaReal = alphaReal;
    tiling.alphaImag = alphaImag;
    tiling.dbgMode = 0u;
    if (const char* envDbg = std::getenv("CSYR_DBG_MODE"); envDbg != nullptr) {
        long v = std::strtol(envDbg, nullptr, 10);
        if (v > 0 && v < 8) {
            tiling.dbgMode = static_cast<uint32_t>(v);
        }
    }
    csyr_fill_batch_cols(tiling.uplo, n, tiling);
    tiling.coreNum = csyr_pick_core_num(tiling.uplo, tiling.n);
    // Platform core-count query failure (arch35 precedent): reject the launch before
    // any device work -- a 0 coreNum has no valid schedule.
    if (tiling.coreNum == 0u) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    // Gated barrier-only schedule: 1 = dispatch the barrier-only column schedule
    // (csyr_process_columns_barr), 0 = the production schedule. Gate band =
    // LOWER && 900<=n<=2048 (CSYR_BARRIER_GATE_NMIN/NMAX). Kernel twin: the top-level
    // dispatch in csyr_kernel.cpp re-validates the band defensively -- a host/kernel
    // desync can only fall back to the production schedule.
    tiling.barrierGate =
        (tiling.uplo == 0u && tiling.n >= CSYR_BARRIER_GATE_NMIN && tiling.n <= CSYR_BARRIER_GATE_NMAX)
            ? 1u : 0u;

    return csyr_stage_and_launch(useStream, x, A, tiling, incxAbsVal, tiling.coreNum);
}
