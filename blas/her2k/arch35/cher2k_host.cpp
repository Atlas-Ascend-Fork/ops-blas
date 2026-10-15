/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cher2k_host.cpp
 * \brief CHER2K Host implementation for ascend950 (DAV_3510)
 *
 *        Phase 0 (AIV): Deinterleave complex A, B into Ar, Ai, Br, Bi via
 *                       cher2k_deinterleave_kernel_do (core-split: half the cores per matrix)
 *        Phase 1 (AIC × 4): 4 real GEMM launches via gemm_kernel_do (4M decomposition)
 *        Phase 2 (AIV): Combine via cher2k_combine_kernel_do
 *        Small-n (AIV SIMT): for orders up to 8, a fused direct path via
 *                            cher2k_simt_small_do (CHER2K_ARCH35_SIMT_N_MAX, P1-F1)
 *
 *        The update follows the M plus M-conjugate identity on the non-transposed (N) path:
 *        C is updated by alpha times A times the conjugate transpose of B, plus
 *        the conjugate of alpha times B times the conjugate transpose of A, plus
 *        beta times C.
 *          The intermediate M, equal to alpha times A times the conjugate
 *          transpose of B, splits into a real part PR and an imaginary part PI;
 *          PR is the sum of the first two real products and PI is the difference
 *          of the remaining two.
 *          The second term (the conjugate of alpha times B times the conjugate
 *          transpose of A) is exactly the conjugate transpose of M, so the update
 *          becomes M plus the conjugate transpose of M plus beta times C.
 *          On the non-transposed (N) path the four real products are: t1 is Ar times the
 *          transpose of Br, t2 is Ai times the transpose of Bi, t3 is Ai times
 *          the transpose of Br, and t4 is Ar times the transpose of Bi.
 *          On the conjugate-transposed (C) path they are: t1 is the transpose of Ar times Br, t2
 *          is the transpose of Ai times Bi, t3 is the transpose of Ar times Bi,
 *          and t4 is the transpose of Ai times Br.
 */

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cher2k_kernel.h"
#include "gemm/arch35/gemm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/syrk_host_utils.h"

// ==========================================================================
//  [codecheck #1/#2/#5] Single environment read point for the CHER2K_* tuning knobs
//  used in this file (CHER2K_L1_SHAPE / CHER2K_L2_TILE / CHER2K_FUSED_GEMM). The
//  source is /proc/self/environ, the execve-time snapshot that stays constant for
//  the process lifetime, read once into a C++11 magic static; this avoids the getenv
//  dangling-pointer race with a concurrent setenv/putenv (G.STD.18-CPP). Only a set
//  variable whose first character is '0' is OFF; unset or any other value keeps the
//  default (ON):
//    CHER2K_L1_SHAPE = 0   force the legacy 32x16 fused shape (SelectCher2kFusedShape)
//    CHER2K_L2_TILE = 0    force the row-band combine (CalCombineTilingData)
//    CHER2K_FUSED_GEMM = 0 fall back to the unfused 4-launch GEMM (RunCher2kGemmPhase)
//  The snapshot is read-only after execve, so a runtime setenv has no effect.
// ==========================================================================
namespace {
// Initial process environment snapshot: "name=value" records separated by NUL (empty on open failure, knobs default)
const std::string& Cher2kInitialEnviron()
{
    static const std::string snapshot = []() -> std::string {
        std::string buf;
        int fd = open("/proc/self/environ", O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            OP_LOGW(
                "aclblasCher2k", "Cher2kInitialEnviron: /proc/self/environ unreadable (errno=%d); knobs default to ON",
                errno);
            return buf;
        }
        char chunk[4096] = {0};
        ssize_t n = 0;
        while ((n = read(fd, chunk, sizeof(chunk))) > 0) {
            buf.append(chunk, static_cast<size_t>(n));
        }
        (void)close(fd);
        return buf;
    }();
    return snapshot;
}

// Exact name match inside the "name=value" record block; return the value address or nullptr
const char* Cher2kEnvironLookup(const std::string& env, const char* name)
{
    const size_t nameLen = strlen(name);
    size_t pos = 0;
    while (pos < env.size()) {
        const char* entry = env.data() + pos;
        const char* nul = static_cast<const char*>(memchr(entry, '\0', env.size() - pos));
        const size_t entryLen = (nul != nullptr) ? static_cast<size_t>(nul - entry) : (env.size() - pos);
        if (entryLen > nameLen && entry[nameLen] == '=' && memcmp(entry, name, nameLen) == 0) {
            return entry + nameLen + 1;
        }
        if (nul == nullptr) {
            break;
        } // the last record has no trailing '\0'
        pos += entryLen + 1;
    }
    return nullptr;
}
} // namespace

static bool Cher2kReadTuningKnob(const char* name)
{
    const char* v = Cher2kEnvironLookup(Cher2kInitialEnviron(), name);
    return v != nullptr && v[0] == '0';
}

// Inverse knob: ON only when the variable is set and its first character is '1'
// (the dual of the "first char '0' means OFF" rule in Cher2kReadTuningKnob). Used by
// the stage2 Step 1 pair-enumeration diagnostic path:
//   CHER2K_PAIR_ENUM = 1 enables the symmetric tile-pair enumeration self-check and log (default off)
// When off this knob has no side effect; the existing behaviour is unchanged.
static bool Cher2kTuningKnobIsOne(const char* name)
{
    const char* v = Cher2kEnvironLookup(Cher2kInitialEnviron(), name);
    return v != nullptr && v[0] == '1';
}

// [perf-iter87] Default-on knob: true when unset, false when explicitly set to '0'.
// Used for optimizations measured and promoted to default-on (currently
// CHER2K_PAIR_WALK). The explicit off keeps A/B regression and a field fallback
// without a rebuild: setting CHER2K_PAIR_WALK to 0 reverts to the grid walk.
static bool Cher2kTuningKnobDefaultOn(const char* name)
{
    const char* v = Cher2kEnvironLookup(Cher2kInitialEnviron(), name);
    return v == nullptr || v[0] != '0';
}

// [perf-iter58] Numeric tuning knob: parse the decimal prefix of the variable and
// return defaultVal when unset or without digits. Like the bool knobs it goes through
// the Cher2kInitialEnviron() snapshot (no getenv), used for shape-parameter A/B sweeps
// so one binary scans many values without a rebuild per constant change.
static uint32_t Cher2kReadTuningKnobU32(const char* name, uint32_t defaultVal)
{
    const char* v = Cher2kEnvironLookup(Cher2kInitialEnviron(), name);
    if (v == nullptr) {
        return defaultVal;
    }
    uint32_t acc = 0U;
    bool any = false;
    for (const char* p = v; *p >= '0' && *p <= '9'; ++p) {
        acc = acc * 10U + static_cast<uint32_t>(*p - '0');
        any = true;
    }
    return any ? acc : defaultVal;
}

// [MIX fold] Sink selector (same /proc/self/environ snapshot, no getenv):
//   CHER2K_MIX_FOLD = 0   force the legacy AIC-only fixpipe path (escape hatch)
//   unset / any other     MIX on-chip fold (default)
//   CHER2K_MIX_STORE4 = 1 MIX but AIV writes only t1..t4 (M1 hand-off gate, bit-exact check)
// Returns a CHER2K_FOLD_* value from cher2k_tiling_data.h.
static uint32_t Cher2kSelectFoldMode()
{
    if (Cher2kReadTuningKnob("CHER2K_MIX_FOLD")) {
        return CHER2K_FOLD_LEGACY;
    }
    const char* store4 = Cher2kEnvironLookup(Cher2kInitialEnviron(), "CHER2K_MIX_STORE4");
    if (store4 != nullptr && store4[0] == '1') {
        return CHER2K_FOLD_STORE4;
    }
    return CHER2K_FOLD_PRPI;
}

// --------------------------------------------------------------------------
//  #6/#7/#8 leading-dimension checks, split out of ValidateCher2kParams to keep
//  its cyclomatic complexity (at most 20) and nbnc line count (at most 50)
//  within the codecheck limits. Semantics are unchanged: each of lda and ldb
//  must be at least the larger of one and n on the non-transposed path, or the
//  larger of one and k otherwise, while ldc must be at least the larger of one
//  and n.
// --------------------------------------------------------------------------
static aclblasStatus_t ValidateCher2kLeadingDims(aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    int minLda = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < minLda) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: lda must be >= %d, got lda=%d", minLda, lda);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldb < minLda) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: ldb must be >= %d, got ldb=%d", minLda, ldb);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, n)) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: ldc must be >= max(1,n), got ldc=%d n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// --------------------------------------------------------------------------
//  #9/#10 pointer non-null checks, split out for the same reason.
//  alpha and beta must be non-null; A and B must be non-null whenever n and k
//  are both positive.
// --------------------------------------------------------------------------
static aclblasStatus_t ValidateCher2kPointers(
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* B, const float* beta)
{
    if (alpha == nullptr) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr && n > 0 && k > 0) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: A must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (B == nullptr && n > 0 && k > 0) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: B must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Parameter validation (design §4.1 — order is normative, #1..#11)
//  All checks run before any quick return.
// ==========================================================================
static aclblasStatus_t ValidateCher2kParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc,
    const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* B, const float* beta, aclblasComplex* C)
{
    // #2 uplo must be either UPPER (121) or LOWER (122), otherwise INVALID_ENUM
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    // #3 trans must be one of OP_N (111), OP_T (112) or OP_C (113), otherwise INVALID_ENUM
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: trans must be OP_N/OP_T/OP_C, got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    // #4 trans being OP_T is a legal enum but unsupported, so it yields INVALID_VALUE
    if (trans == ACLBLAS_OP_T) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: trans=OP_T is not supported by Cher2k, use OP_N or OP_C");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // #5 both n and k must be non-negative
    if (n < 0) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k < 0) {
        OP_LOGE("aclblasCher2k", "ValidateCher2kParams: k must be >= 0, got %d", k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // #6/#7/#8 leading dimensions: lda and ldb must be at least the larger of
    // one and n on the non-transposed path (the larger of one and k otherwise),
    // and ldc must be at least the larger of one and n
    aclblasStatus_t ldSt = ValidateCher2kLeadingDims(trans, n, k, lda, ldb, ldc);
    if (ldSt != ACLBLAS_STATUS_SUCCESS) {
        return ldSt;
    }
    // #9/#10 pointer non-null checks
    aclblasStatus_t ptrSt = ValidateCher2kPointers(n, k, alpha, A, B, beta);
    if (ptrSt != ACLBLAS_STATUS_SUCCESS) {
        return ptrSt;
    }
    // #11 a null C together with a positive n: resolved in aclblasCher2k after a
    // staging read of beta (needs the handle's stream), see ResolveCher2kCNull.
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Staging read of alpha (complex, 2 floats) and beta (1 float) from device.
//  Only used for the branch decision when C is null and n is positive.
//
//  [review: stream sync in library code] Uses the SYNCHRONOUS aclrtMemcpy for
//  this one-shot scalar read, instead of an async copy followed by an explicit
//  stream wait: the library's normal path must not wait on the stream, and the
//  value is needed before returning (same shape as sasum_host.cpp:137 /
//  scopy_host.cpp:106, which copy scalars synchronously).
//  Because the copy is synchronous the values are landed on return, so no
//  explicit sync is needed and no work of the caller's stream is waited on.
// ==========================================================================
static aclblasStatus_t ReadCher2kScalars(
    const aclblasComplex* alpha, const float* beta, float& alphaReal, float& alphaImag, float& betaVal,
    aclrtStream stream, bool readAlpha = true)
{
    // [perf-iter30] Stage alpha.real/alpha.imag in ONE 8B D2H instead of two
    // separate 4B copies. Every tiny D2H costs ~16us of stream-timeline latency
    // on this platform (measured: three 4-byte copies plus a sync cost 34.5us,
    // against 18.0us for an 8-byte plus a 4-byte copy with a sync), and this
    // staging read sits inside the caller's timed window, so the extra copy is
    // pure fixed per-call overhead that dominates small-n cases.
    (void)stream; // kept for signature stability; the copy below is synchronous
    float alphaPair[2] = {0.0f, 0.0f};
    if (readAlpha) {
        aclError aclRet =
            aclrtMemcpy(alphaPair, sizeof(alphaPair), alpha, sizeof(alphaPair), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasCher2k", "ReadCher2kScalars: aclrtMemcpy alpha D2H failed, ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    aclError aclRet = aclrtMemcpy(&betaVal, sizeof(float), beta, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCher2k", "ReadCher2kScalars: aclrtMemcpy beta D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (readAlpha) {
        alphaReal = alphaPair[0];
        alphaImag = alphaPair[1];
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Design §4.1 #11: a null C together with a positive n
//    a non-zero beta gives INVALID_VALUE; a zero beta gives SUCCESS without any write.
// ==========================================================================
static aclblasStatus_t ResolveCher2kCNull(_aclblas_handle* h, const float* beta, int n)
{
    float betaVal = 0.0f;
    aclblasStatus_t st = ReadCher2kScalars(nullptr, beta, betaVal, betaVal, betaVal, h->stream, false);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k", "ResolveCher2kCNull: ReadCher2kScalars beta failed, st=%d", static_cast<int>(st));
        return st;
    }
    if (betaVal != 0.0f) {
        OP_LOGE("aclblasCher2k", "ResolveCher2kCNull: C must not be nullptr when n>0 and beta!=0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    OP_LOGD("aclblasCher2k", "C is nullptr with beta=0 and n=%d: success without write", n);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  GEMM tiling computation (reuses gemm pattern, same as cherk)
// ==========================================================================
// ==========================================================================
//  [L1] K2f fused-kernel tile shape selection
// ==========================================================================
//  [L1-shape-v2] Two measured levers, both from the arch35 Mmad micro-benchmark
//  (the probe-decision finalization report §1) plus a fresh 200-case A/B sweep (see
//  the delivery gemm-optimization note §performance-gain table):
//  1) The block row size is 64 for every order of magnitude 64 and above. The
//     probe put the legacy 32-row Mmad at ~15 TF/s, against the 23.0~23.5 TF/s
//     platform rate that every 64-row-block shape reaches; the previous rule
//     granted the 64-row block only to orders that are exact multiples of 64,
//     leaving 144 of the 200 reference cases (all non-multiples of 64) on the
//     32x16 fallback. The A/B sweep measured a 2.3 to 2.7 times end-to-end gain
//     on those orders (TC_PF_1118, order 3885 and inner dim 3334: 81.5ms down
//     to 31.6ms) purely from lifting that guard — it is a shape-rate lever, not
//     an L1-traffic one, so it applies to every order of magnitude 64 and above.
//  2) The block column size is 64 only in the large gemm-bound regime (order at
//     least 1200 and inner dim at least 640). Per-core L1 traffic grows in
//     proportion to the square of the order times the inner dim, scaled by the
//     sum of the reciprocals of the two block sizes: each A row-panel is re-read
//     once per column tile and vice versa, so widening the column block from 32
//     to 64 drops the coefficient from 0.0469 to 0.03125 (a 33% cut) — but only
//     helps while that load traffic, not the k-independent store of the 4 n by n
//     temp frames, is the limiter (both scale with the squared order, so the
//     crossover keys on the inner dim alone). The sweep puts it at an order of
//     about 1200 on the n axis (below it the 64-wide column block under-fills
//     the 24 AIC cores: order 1024 takes 606us against 542us, and the formal gate
//     TC_PF_1003 regresses 722us above 680us) and at an inner dim of about 640 on
//     the k axis (order 4096 with inner dim 512 is store-bound and the 32-wide
//     column block is 1.8% faster — that case sits at ratio 0.403, so it must
//     stay on the 32-wide block).
//     Above both, the 64-wide block wins big: order 2048 with inner dim 2048
//     gains 7%, order 2714 with inner dim 4035 gains 30%, order 3885 with inner
//     dim 3334 gains 36%, and order 1744 with inner dim 851 gains 8.7% (the last
//     one flips its ratio from 0.394 to 0.432 across the 0.4 gate).
//  Fallback shape 32x16 (the legacy constants, bit-identical tile geometry
//  incl. tail tiles) is reserved for orders below 64 (no full 64x64 block
//  partition) and the CHER2K_L1_SHAPE escape hatch.
// ==========================================================================
constexpr uint32_t CHER2K_BN64_MIN_N = 1160; // 64-wide column block crossover on the n axis (A/B refined)
constexpr uint32_t CHER2K_BN64_MIN_K = 640;  // 64-wide column block crossover on the k axis
// [perf-iter58] Upper k bound of the tall/narrow low-k band that also takes
// a 64-wide column block (see SelectCher2kFusedShape). Measured on the full 200;
// case 1193 (order 2872, inner dim 256) regresses 4.9% just above this bound.
constexpr uint32_t CHER2K_BN64_TALLK_MAX = 160;
// [perf-iter93] Lower n bound for the default-on pair walk (see the gate in
// LaunchCher2kKernel). Same value as CHER2K_BN64_MIN_N by construction: both mark
// the point where a large-order shape starts paying for its tile count, and the
// pair-walk crossover was measured to sit in the same place — TC_PF_1089
// (order 1024) regresses, while every case the pair walk flips (order 1579 and
// above) gains.
constexpr uint32_t CHER2K_PAIR_WALK_MIN_N = 1160;

static void SelectCher2kFusedShape(uint32_t n, uint32_t k, int32_t& baseM, int32_t& baseN)
{
    if (Cher2kReadTuningKnob("CHER2K_L1_SHAPE")) {
        baseM = 32;
        baseN = 16;
        return; // [l123c A/B] force legacy shape
    }
    if (n < 64U) {
        baseM = 32; // CHER2K_FUSED_BM_FALLBACK, the same value as GEMM_BASE_M
        baseN = 16; // CHER2K_FUSED_BN_FALLBACK, the same value as GEMM_BASE_N
        return;
    }
    baseM = 64; // CHER2K_FUSED_BM_MAIN
    // [perf-iter58] Wide-N for the tall/narrow low-k band. Measured on the full
    // 200 (the column-block override sweep, /tmp/pf200_bn64.log): forcing a
    // 64-wide column block globally is
    // net NEGATIVE (189 against 191 — it loses cases 1102, 1123 and 1179 and
    // regresses dozens of already-passing shapes by 60 to 85%), but the subset
    // with order at least 1160 and inner dim at most 160 improves uniformly and
    // touches no passing case adversely. That subset speeds up case 1156 (order
    // 3100, inner dim 149) by 1.0% and flips it to PASS, case 1149 (3105,120) by
    // 1.3%, case 1091 (2048,64) by 2.9%, case 1122 (2143,137) by 1.3%, and case
    // 1144 (1290,35) by 0.6%, while case 1093 (4096,64) slows by 1.5% (already
    // far below the gate either way).
    // Rationale: at a 32-wide column block a 64-row A band is re-read nLoopCount
    // times; for these shapes the tile count is large enough that halving the
    // B-side N blocks wins, while the narrow-order shapes (order below 1160) lose
    // from the wider 64x64 L1 footprint. Both bounds are required — dropping the
    // k bound pulls in case 1193 (2872,256), a 4.9% regression, and dropping the
    // n bound pulls in case 1089, an 8.2% regression.
    // Default off (knob 0) keeps the original crossover rule unchanged.
    const uint32_t bnOverride = Cher2kReadTuningKnobU32("CHER2K_BN_OVERRIDE", 0U);
    if (bnOverride == 32U || bnOverride == 64U) {
        baseN = static_cast<int32_t>(bnOverride);
        return;
    }
    const bool gemmBound = (n >= CHER2K_BN64_MIN_N) && (k >= CHER2K_BN64_MIN_K);
    const bool narrowKBand = (n >= CHER2K_BN64_MIN_N) && (k <= CHER2K_BN64_TALLK_MAX);
    baseN = (gemmBound || narrowKBand) ? 64 : 32; // CHER2K_FUSED_BN_WIDE : CHER2K_FUSED_BN_MAIN
}

static GemmTilingData CalCher2kGemmTilingData(
    int n, int k, int lda, int ldb, int tempLdc, aclblasOperation_t transa, aclblasOperation_t transb)
{
    GemmTilingData tiling{};
    tiling.m = n; // CHER2K: the row count of C is the order n (C is n×n)
    tiling.n = n; // CHER2K: the column count of C is the order n
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = tempLdc;
    tiling.cLdc = tempLdc;
    // [L1] per-shape tile geometry (see SelectCher2kFusedShape above). The
    // unfused 4-launch path shares this tiling, and the shared gemm kernel
    // ignores baseM/baseN/baseK (it uses its own compile-time constants) —
    // but CalcMultiCorePartition DOES key on baseM/baseN for the block
    // partition, so both paths get a partition consistent with the launch
    // that consumes it.
    SelectCher2kFusedShape(static_cast<uint32_t>(n), static_cast<uint32_t>(k), tiling.baseM, tiling.baseN);
    tiling.baseK = GEMM_BASE_K;
    tiling.tileKChunk = GEMM_TILE_K_CHUNK;
    tiling.c0Size = GEMM_C0_SIZE;
    tiling.isTransA = (transa != ACLBLAS_OP_N) ? 1 : 0;
    tiling.isTransB = (transb != ACLBLAS_OP_N) ? 1 : 0;
    // alpha/beta are folded into Phase 2; GEMM runs without post-process
    tiling.alphaReal = 1.0f;
    tiling.alphaImag = 0.0f;
    tiling.betaReal = 0.0f;
    tiling.betaImag = 0.0f;
    tiling.hasBeta = 0;
    return tiling;
}

static void ApplyColMajorSwap(GemmTilingData& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
}

static void CalcMultiCorePartition(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    int32_t mTiles = (tiling.m + tiling.baseM - 1) / tiling.baseM;
    int32_t nTiles = (tiling.n + tiling.baseN - 1) / tiling.baseN;
    // [perf-iter1] Prefer a square-ish m/n block split over a pure utilization
    // search. The original loop only maximized the product of the m-block and
    // n-block counts (core count) and always picked a single m block, which gives
    // every core the FULL m range (mTiles/baseM tiles tall) and a fractional n
    // range; with 64 n tiles and 28 cores the n tiles split 3/3/.../2 unevenly
    // (8 cores take 3 tiles against 20 cores taking 2 tiles, a 1.5-fold
    // imbalance) and the long m run serializes the per-tile K pipeline.
    // A balanced m-block count closest to the square root of maxCores keeps
    // utilization AND evens out the tile counts on both axes: measured on 950PR
    // with order and inner dim both 1024 (32 m tiles, 64 n tiles, 28 cores) the
    // single m block took 630us, against 525us for four m blocks (a 17% cut);
    // order and inner dim both 2048 behave similarly.
    int32_t lo = 1;
    int32_t hi = std::min(mTiles, maxCores);
    // Selection rule (perf-iter1 fix, finalized in perf-iter2):
    // 1) maximize utilization, the product of the m-block and n-block counts
    //    (keep all cores busy; the original rule only did this and always picked
    //    a single m block, giving every core the FULL m range and a 1.5-fold
    //    imbalance on the n axis);
    // 2) break utilization TIES by the most square-ish pair of block counts —
    //    with the strict 'greater than' comparison of the first perf-iter1
    //    attempt, the candidate with one m block and all cores on n locked in
    //    first and silently reverted the balanced pick, which is why that attempt
    //    measured no end-to-end change. On 950PR with order and inner dim both
    //    1024 and 28 AIC cores, the one-by-28 split took 630us against 525us for
    //    the four-by-seven split per gemm (a 17% cut).
    // Integer sqrt only (bisheng ASC hides ::sqrt behind a half intrinsic);
    // maxCores is at most 56 on 950PR so a loop over one through eight is exact.
    int32_t sqrtHint = 1;
    while ((sqrtHint + 1) * (sqrtHint + 1) <= maxCores) {
        sqrtHint++;
    }
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = std::min(nTiles, maxCores);
    if (bestNBlocks < 1) {
        bestNBlocks = 1;
    }
    int32_t bestUtilization = -1;
    int32_t bestImbalance = INT32_MAX;
    const int32_t mbLimit = std::min(mTiles, maxCores);
    for (int32_t mb = 1; mb <= mbLimit; mb++) {
        int32_t nb = std::min(nTiles, maxCores / mb);
        if (nb < 1) {
            nb = 1;
        }
        int32_t utilization = mb * nb;
        if (utilization > maxCores) {
            continue;
        }
        int32_t imbalance = (mb > nb) ? (mb - nb) : (nb - mb);
        bool better = (utilization > bestUtilization) || (utilization == bestUtilization && imbalance < bestImbalance);
        // Deterministic tie-break inside an equal utilization and imbalance:
        // keep the earlier (smaller m-block) candidate, except let the square-root
        // vicinity win so the common square case (for instance five-by-four for 20
        // cores) is stable.
        if (better) {
            bestUtilization = utilization;
            bestImbalance = imbalance;
            bestMBlocks = mb;
            bestNBlocks = nb;
        } else if (
            utilization == bestUtilization && imbalance == bestImbalance && mb <= sqrtHint && bestMBlocks > sqrtHint) {
            bestMBlocks = mb;
            bestNBlocks = nb;
        }
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    tiling.usedCoreNum = bestMBlocks * bestNBlocks;
}

static void PrepareCubeTiling(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    ApplyColMajorSwap(tiling);
    CalcMultiCorePartition(tiling, cubeCoreNum);
}

// ==========================================================================
//  Combine tiling computation
// ==========================================================================
// [perf-iter3 final] The K3 combine core count is the original rule, the smaller
// of the order and the AIV core count.
// The perf-iter2/3 core-count tuning (16 cores at order 1024 and 32 cores at
// order 2048, with a power-of-two clamp so the rows per core stay a multiple of
// 64) measured a real K3-only win (at order 1024, 900us with 56 cores down to
// 67us with 16 cores; end-to-end 2256us against 3040us) BUT corrupts C for
// orders 1449 and above: with 64 rows per core the combine writes wrong values
// across the whole uplo triangle (a maximum absolute error near 6.7, on 50% of
// the elements; probe tmp/perfiter3/vecprobe*.cpp, sweeps
// tmp/perfiter3/k3core_sweep*.txt). The threshold is the C byte span, which
// exceeds 16MB once the order passes about 1449. A rows-per-core below 64 (for
// instance 37 across 56 cores) hides it, which is why the original 56-core rule
// is suite-clean (CP3 2026-09-11 1000+120+66 all PASS) while every reduced-core
// variant fails TC_SQ_031/077 at order 2048. The root cause (suspected te::Copy
// 2D strided GM offset handling above the 16MB span) was not fixed this round,
// so the rule was reverted; see the perf-iteration log §negative-results.
static uint32_t TuneCombineCoreNum(uint32_t n, uint32_t usedAivCoreNum)
{
    // [r2/G2] Kernel side: the 16MB-plus GM-offset hazard is fixed in
    // cher2k_kernel.cpp (Cher2kLoadTempBlock / [r2/G2 >16MB offset fix]) — the
    // K3 combine used to Slice one big-frame GM tensor (8.4MB at order 1449
    // through 67MB at order 4096); the offset now advances the base pointer so no
    // stride field can wrap. With that fix the reduced-core rule is numerically
    // clean on every shape of order 1449 and above that was tested (at order
    // 2048: TC_SQ_031/054/077/100, TC_EX_0384/0872, TC_TK_143 all PASS).
    //
    // Host side: reduce cores ONLY when the row band is EXACT, i.e. when the
    // order is a multiple of 64 and the order-over-64 core count fits (so the
    // rows per core are exactly 64 with no partial last band). Reason: r2
    // measured a HANG (not corruption) at order 1466 — rounding 1466 over 64 up
    // gives 23 cores, and 23 cores of 64 rows each cover 1472 rows, 6 more than
    // the 1466 requested, leaving the last core a 58-row band, and that
    // rows-per-core-of-64-with-a-partial-tail configuration deadlocks in the
    // combine (reproduced with BOTH the fused and unfused K2, so it is a K3
    // issue, not a G1 regression; tmp/r2/g2iso*/). The exact-multiple subset is
    // validated: order 1024 with 16 cores and order 2048 with 32 cores were clean
    // in the perf-iter3 sweeps and in the r2 whitebox/main spot runs, and
    // exact multiples up to order 1448 were already covered by the perf-iter4
    // rule for that range that passed the full main table 983/984. Every other
    // order keeps the CP3-original full-core rule (suite-clean, perfiter4
    // 983/984).
    // Measured K3-only win on the exact subset: order 1024 from 900us to 67us
    // (16 cores), order 2048 from 3238us to 142us (32 cores) —
    // tmp/perfiter3/k3core_sweep*.txt.
    //
    // [k3core FINAL] The vector chain is now CORRECT (in-UB 64x64 Reg::Gather
    // transpose of the partner s-tiles before the 1D chain — the previous
    // version assumed the transposed offset coincides with the linear offset
    // when rows and cols are both 64, which only holds on the diagonal and
    // corrupted every full tile; probe tmp/k3core/k3probe check, 46 of 46 shapes
    // clean after the fix). Reduced cores are therefore SAFE for the
    // exact-multiple subset:
    //   when the order is a multiple of 64, the core count is the smaller of
    //                  the order over 64 and the used AIV core count (64 rows
    //                  per core, every row band is full 64x64 vector tiles)
    //   otherwise, the full-core original rule applies (row bands straddle tile
    //                  boundaries; the scalar tail path is the cliff:
    //                  measured order 1000 with 16 cores at 2711us and order 2040
    //                  with 32 cores on LOWER at 4969us, against 147/201us for the
    //                  multiple-of-64 neighbours — tmp/k3core/time_sweep.log)
    // Measured (fixed kernel, tmp/k3core/time_sweep.log):
    //   order 1024 UPPER/LOWER: 868/860 down to 147/201  (16 cores)
    //   order 2048 UPPER/LOWER: 3307/3307 down to 294/340 (32 cores)
    //   order 1152 UPPER/LOWER: 18 cores 163/210
    //   order 4096 UPPER: 64 cores 735 (clamped to 56 AIV cores giving 74 rows
    //                     per core, scalar; keep full-core for orders above
    //                     64 times the used AIV core count)
    // [L2] Exact-multiple subset: use the FULL AIV count and let the combine
    // walk the uplo triangle as a 2D tile list (tileMode 1, see
    // CalCombineTilingData below) instead of a reduced-core row band. The
    // reduced-core rule (core count equal to the order over 64) exists ONLY
    // because a row band is the unit of allocation: capping the cores was the
    // only way to shorten the 16-tile critical path a row band implies. With the
    // tile list each core owns the tiles whose index leaves that core's number as
    // remainder when divided by the core count, so more cores is strictly better
    // (at order 1024 UPPER: 136 tiles over 56 cores is about 2.4 tiles per core,
    // a critical path of 3 tiles against 16; LOWER also has 136 tiles). Cores are
    // still capped by the tile count (more cores than tiles would idle) and by
    // the multiple-of-64 guard, which keeps every tile a full 64x64 square (the
    // vector chain shape) and keeps the 16MB-plus GM-offset guard of
    // Cher2kLoadTempBlock engaged (base-pointer advance, unchanged).
    if (n % CHER2K_ARCH35_SCALE_BLOCK == 0) {
        const uint32_t byRow = n / CHER2K_ARCH35_SCALE_BLOCK;
        const uint32_t tileEstimate = byRow * (byRow + 1) / 2; // uplo tile count
        return std::max<uint32_t>(std::min<uint32_t>(tileEstimate, usedAivCoreNum), 1U);
    }
    return std::max<uint32_t>(std::min<uint32_t>(usedAivCoreNum, n), 1U);
}

static Cher2kCombineTilingData CalCombineTilingData(
    uint32_t usedAivCoreNum, uint32_t n, uint32_t ldc, uint32_t tempLdc, uint8_t isAlphaZero, uint8_t isKZero,
    uint8_t isBetaZero, uint8_t uploMode, float alphaReal, float alphaImag, float betaVal)
{
    Cher2kCombineTilingData tiling{};
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempLdc = tempLdc;
    const uint32_t combineCores = TuneCombineCoreNum(n, usedAivCoreNum);
    tiling.rowsPerCore = CeilDiv<uint32_t>(n, combineCores);
    tiling.alphaReal = alphaReal;
    tiling.alphaImag = alphaImag;
    tiling.betaVal = betaVal;
    tiling.uploMode = uploMode;
    tiling.isAlphaZero = isAlphaZero;
    tiling.isKZero = isKZero;
    tiling.isBetaZero = isBetaZero;
    // [perf-iter96] isSimt selects the combine tile walk: 0 (default) is the
    // O(q+owned) owned-only walk, 1 is the legacy replay-all walk (A/B knob).
    tiling.isSimt = Cher2kTuningKnobIsOne("CHER2K_COMBINE_REPLAY") ? 1 : 0;
    // [L2] mode selection. tileMode 1 enumerates the uplo triangle as a q by q
    // tile grid with q equal to the order rounded up over 64; the edge tiles are
    // clipped to the smaller of 64 and the remaining rows or cols (P1
    // generalization), so every order — not only multiples of 64 — is served by
    // the 2D tile list. Interior full 64x64 tiles keep the vector-chain
    // precondition of 64 rows and 64 cols; only the last row or column band runs
    // the scalar tail. Requires the tile count to fit the uint16 field: the
    // triangle tile total (q times q-plus-one over two) must be at most 65535,
    // i.e. the order must be at most 23104 (the previous 2048 cap only covered
    // orders up to 4032 and left order 4096 — whose triangle tile total is 2080 —
    // on the scalar row-band fallback). Everything else keeps mode 0: identical
    // row-band walk to the pre-L2 binary. CHER2K_L2_TILE forces mode 0.
    const uint32_t q = CeilDiv<uint32_t>(n, CHER2K_ARCH35_SCALE_BLOCK);
    const uint32_t tileTotal = q * (q + 1) / 2;
    const bool l2ForceOff = Cher2kReadTuningKnob("CHER2K_L2_TILE");
    // Orders below 64 stay on mode 0: such a shape has no interior 64x64 tile to
    // vectorize, and the row-band walk is measurably faster for tiny orders
    // (TC_PF_1010 at order 32: mode 0 takes 52us against 99us for mode 1).
    if (!l2ForceOff && n >= CHER2K_ARCH35_SCALE_BLOCK && tileTotal <= 65535U) {
        tiling.tileMode = 1;
        tiling.numCores = static_cast<uint8_t>(combineCores);
        tiling.tileTotal = static_cast<uint16_t>(tileTotal);
    } else {
        tiling.tileMode = 0;
        tiling.numCores = 0;
        tiling.tileTotal = 0;
    }
    return tiling;
}

// ==========================================================================
//  [stage2 Step 1] Gated pair-enumeration diagnostic (design §6 Step 1).
//  With CHER2K_PAIR_ENUM set to 1: enumerate the symmetric tile pairs on the 64x64
//  combine grid and self-check coverage (every in-triangle tile exactly once; an
//  off-diagonal pair covers both (i,j) and its transpose (j,i), qc^2 product tiles
//  in total) and load balance (the per-core weight sums to the total; the max minus
//  min core load stays within the largest single pair weight), then log the per-core
//  quota. Default off: return at once, zero side effect, existing behaviour unchanged.
//  This function only reads tiling params; it touches no GM/UB/signal and changes no
//  launch path.
// ==========================================================================
// [codecheck B2/C4] Coverage self-check: rebuild the whole qc×qc product tile
// grid and assert every tile is covered exactly once (triangle representative +
// its symmetric partner).
static bool Cher2kPairEnumCoverage(uint32_t qc, bool upper, bool hasTail, uint32_t numPairs, uint32_t& coveredTiles)
{
    coveredTiles = 0;
    for (uint32_t I = 0; I < qc; ++I) {
        for (uint32_t J = 0; J < qc; ++J) {
            uint32_t hit = 0;
            for (uint32_t p = 0; p < numPairs; ++p) {
                const Cher2kTilePair pr = Cher2kBuildPair(qc, p, upper, hasTail);
                if (pr.i == I && pr.j == J) {
                    hit += 1; // in-triangle representative
                } else if (pr.i != pr.j && pr.j == I && pr.i == J) {
                    hit += 1; // symmetric partner (J,I) covered by the representative (I,J)
                }
            }
            if (hit != 1) {
                return false;
            }
            coveredTiles += 1;
        }
    }
    return true;
}

// [codecheck B2/C4] Load-balance self-check: replay the pair cursor and collect
// the per-core load statistics (sum / max / min / largest single pair weight).
static void Cher2kPairEnumBalance(
    uint32_t qc, bool upper, bool hasTail, uint32_t cores, uint32_t numPairs, uint32_t& loadSum, uint32_t& loadMax,
    uint32_t& loadMin, uint32_t& maxPairW)
{
    Cher2kPairCursor cur;
    Cher2kPairCursorInit(cur, cores);
    loadMax = 0;
    loadMin = 0xFFFFFFFFU;
    maxPairW = 0;
    Cher2kTilePair pr{0, 0, 0, 0};
    for (uint32_t p = 0; p < numPairs; ++p) {
        Cher2kPairCursorNext(cur, qc, upper, hasTail, cores, pr);
        const uint32_t w = Cher2kPairWeight(qc, p, upper, hasTail);
        if (w > maxPairW) {
            maxPairW = w;
        }
    }
    loadSum = 0;
    for (uint32_t c = 0; c < cores; ++c) {
        const uint32_t cl = cur.loads[c];
        loadSum += cl;
        if (cl > loadMax) {
            loadMax = cl;
        }
        if (cl < loadMin) {
            loadMin = cl;
        }
    }
}

static void Cher2kReportPairEnum(uint32_t n, uint8_t uploMode, uint32_t numCores)
{
    if (!Cher2kTuningKnobIsOne("CHER2K_PAIR_ENUM")) {
        return; // default off: no side effect
    }
    const uint32_t qc = CeilDiv<uint32_t>(n, CHER2K_ARCH35_SCALE_BLOCK);
    const bool upper = (uploMode == ACLBLAS_UPPER);
    const bool hasTail = (n % CHER2K_ARCH35_SCALE_BLOCK != 0);
    const uint32_t cores = std::max<uint32_t>(numCores, 1U);
    const uint32_t numPairs = Cher2kPairCount(qc);
    const uint32_t total = Cher2kPairTotalWeight(qc, upper, hasTail);

    // Coverage self-check: rebuild the whole qc x qc product tile grid and assert it is covered exactly once.
    uint32_t coveredTiles = 0;
    const bool coverOk = Cher2kPairEnumCoverage(qc, upper, hasTail, numPairs, coveredTiles);

    // Load-balance self-check: per-core weight sum.
    uint32_t loadSum = 0;
    uint32_t loadMax = 0;
    uint32_t loadMin = 0;
    uint32_t maxPairW = 0;
    Cher2kPairEnumBalance(qc, upper, hasTail, cores, numPairs, loadSum, loadMax, loadMin, maxPairW);
    const bool balanceOk = (loadSum == total) && (loadMax - loadMin <= maxPairW);

    OP_LOGI(
        "aclblasCher2k",
        "[PAIR_ENUM] n=%u qc=%u uplo=%s tail=%d cores=%u pairs=%u totalWeight=%u (qc^2=%u) covered=%u "
        "coverOk=%d load[min=%u max=%u spread=%u maxPairW=%u] balanceOk=%d",
        n, qc, upper ? "UPPER" : "LOWER", hasTail ? 1 : 0, cores, numPairs, total, qc * qc, coveredTiles,
        coverOk ? 1 : 0, loadMin, loadMax, loadMax - loadMin, maxPairW, balanceOk ? 1 : 0);
}

// ==========================================================================
//  PrepareCher2kParams — derive the branch flags WITHOUT a device staging read.
//  [iter34] The old version staged alpha (8B) + beta (4B) through two tiny async
//  device-to-host copies followed by an explicit stream wait, which costs ~24us
//  per call and sits INSIDE the caller's timed window (measured: two copies plus
//  a wait took 23.39us, one copy plus a wait took 12.01us, and the wait alone
//  took 0.9us). That staging is now gone
//  for the DEVICE case: the K3 combine kernel and the K4 SIMT kernel read
//  alpha/beta straight from GM (the device pointers are threaded to them as
//  kernel arguments) and derive isAlphaZero / isBetaZero / skipTemp on chip.
//
//  The scalar pointers may also be HOST pointers (the operator contract, same as
//  srot/csyrk: aclrtPointerGetAttributes decides). In that case there is no
//  device read to eliminate anyway — the host dereferences them once and the
//  kernel uses the tiling scalars (with both the alpha and beta device flags
//  clear). The mirror / quick-return host unit tests take this path.
//
//  host-side skipTemp therefore only covers a zero inner dim; a zero alpha on the
//  device path is handled by the kernel (the alpha multiply zeroes the product
//  term, and the combine's skipTemp ternary is kernel-derived).
// ==========================================================================
static aclblasStatus_t PrepareCher2kParams(
    _aclblas_handle* h, uint32_t k, const aclblasComplex* alpha, const float* beta, float& alphaReal, float& alphaImag,
    float& betaVal, bool& isAlphaZero, bool& isKZero, bool& isBetaZero, bool& skipTemp, bool& isAlphaDev,
    bool& isBetaDev)
{
    (void)h;
    bool alphaOnDev = false;
    bool betaOnDev = false;
    aclblasStatus_t locSt = CheckPtrLocation(alpha, &alphaOnDev);
    if (locSt != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCher2k", "PrepareCher2kParams: alpha pointer location query failed, st=%d",
            static_cast<int>(locSt));
        return locSt;
    }
    locSt = CheckPtrLocation(beta, &betaOnDev);
    if (locSt != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCher2k", "PrepareCher2kParams: beta pointer location query failed, st=%d", static_cast<int>(locSt));
        return locSt;
    }

    // Host pointers: dereference once here (no device read). Device pointers:
    // leave placeholders — the kernel reads the real values from GM.
    alphaReal = alphaOnDev ? 0.0f : alpha->real;
    alphaImag = alphaOnDev ? 0.0f : alpha->imag;
    betaVal = betaOnDev ? 0.0f : (*beta);

    isAlphaZero = alphaOnDev ? false : (alphaReal == 0.0f && alphaImag == 0.0f);
    isKZero = (k == 0);
    isBetaZero = betaOnDev ? false : (betaVal == 0.0f);
    // Host-side skipTemp only covers a zero inner dim (and the host-visible zero
    // alpha). On the device path a zero alpha is decided in-kernel.
    skipTemp = isAlphaZero || isKZero;
    isAlphaDev = alphaOnDev;
    isBetaDev = betaOnDev;

    OP_LOGD(
        "aclblasCher2k", "isAlphaDev=%d isBetaDev=%d isKZero=%d skipTemp=%d", isAlphaDev ? 1 : 0, isBetaDev ? 1 : 0,
        isKZero ? 1 : 0, skipTemp ? 1 : 0);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  PrepareCher2kWorkspace — workspace size (design §2.4) + ensure allocation
//    arBytes   is the buffer size of each of Ar/Ai/Br/Bi, i.e. rows times cols
//              times 4 bytes (four buffers of equal size)
//    tempBytes is tempLdc times n times 4 bytes, where tempLdc is the order
//              rounded up to a multiple of 16
//    ws        is zero when skipTemp is set, otherwise four arBytes plus four
//              tempBytes, rounded up to a multiple of 512
//    layout: the four temp frames first, then Ar, Ai, Br, Bi (same as cherk)
// ==========================================================================
static aclblasStatus_t PrepareCher2kWorkspace(
    _aclblas_handle* h, uint32_t n, uint32_t k, aclblasOperation_t trans, bool skipTemp, uint32_t& aicCoreNum,
    uint32_t& aivCoreNum, uint32_t& usedAivCoreNum, uint32_t& tempLdc, uint32_t& logicalRows, uint32_t& physCols,
    size_t& arBytes, size_t& tempBytes)
{
    aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE("aclblasCher2k", "PrepareCher2kWorkspace: GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCher2k", "PrepareCher2kWorkspace: GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    usedAivCoreNum = GetUsedAivCoreNum(n, aivCoreNum);
    tempLdc = CeilAlign<uint32_t>(n, GEMM_FRACTAL);

    // A and B share the same shape (rows by cols); Ar/Ai/Br/Bi are packed
    // column-major with a row stride equal to the row count.
    uint32_t physRows = (trans == ACLBLAS_OP_N) ? n : k;
    physCols = (trans == ACLBLAS_OP_N) ? k : n;
    logicalRows = physRows;
    arBytes = static_cast<size_t>(logicalRows) * physCols * sizeof(float);
    tempBytes = static_cast<size_t>(tempLdc) * n * sizeof(float);
    size_t workspaceNeed = 0;
    if (!skipTemp) {
        // [p1l R1] back to 4 temp frames plus the deinterleaved Ar/Ai/Br/Bi
        // operands. The 4 transposed twins (P1-D, which cost four more tempBytes)
        // were removed: their builder kernel's GM store was unfixable for column
        // tails below 64.
        workspaceNeed = arBytes * 4 + tempBytes * 4;
    }
    constexpr size_t GM_ALIGN = 512;
    workspaceNeed = (workspaceNeed + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;

    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, workspaceNeed);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCher2k", "PrepareCher2kWorkspace: workspace ensure failed, required=%zu, ret=%d", workspaceNeed,
            wsRet);
        return wsRet;
    }
    OP_LOGD(
        "aclblasCher2k", "workspace n=%u k=%u tempLdc=%u arBytes=%zu tempBytes=%zu need=%zu", n, k, tempLdc, arBytes,
        tempBytes, workspaceNeed);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  RunCher2kGemmPhase — Phase 0+1: AIV deinterleave (A and B) + 4 GEMM launches
//  4-launch mapping (design §3.2, M2-verified on 950PR):
//  The launch helper, given the two operand pointers X and Y and the output
//  frame dTi, stores Y times the transpose of X on the N path and the transpose
//  of Y times X on the C path (empirically confirmed by the M2 probe; the
//  fixpipe/ApplyColMajorSwap storage convention transposes the logical product).
//    N path: dT1 is launched from Br,Ar and yields Ar times the transpose of Br;
//            dT2 is launched from Bi,Ai and yields Ai times the transpose of Bi;
//            dT3 is launched from Br,Ai and yields Ai times the transpose of Br;
//            dT4 is launched from Bi,Ar and yields Ar times the transpose of Bi.
//    C path: dT1 is launched from Br,Ar and yields the transpose of Ar times Br;
//            dT2 is launched from Bi,Ai and yields the transpose of Ai times Bi;
//            dT3 is launched from Bi,Ar and yields the transpose of Ar times Bi;
//            dT4 is launched from Br,Ai and yields the transpose of Ai times Br.
//  NOTE (risk #1 resolved): the M1 skeleton launched the N path with the
//  operand pairs reversed (the pairs Ar,Br / Ai,Br / Ar,Bi), which stores the
//  transposed products and breaks the first two products (unsymmetric). The M2
//  smoke gate (orders 4 and 8 against the CPU golden) caught it; operands were
//  swapped per the §risk#1 contingency. No architecture change.
// ==========================================================================
// [codecheck #4] RunCher2kGemmPhase was a 66-line oversized function, split into four
// parts along its original paragraphs:
//   Cher2kLaunchDeinterleave  — Phase 0 deinterleave launch (tiling build + launch)
//   CalCher2kProductTiling    — Phase 1 tiling build (incl. the fused-path tkc clamp)
//   Cher2kLaunchFused4        — 4-launch dispatch, fused default path (one cher2k_fused4_do)
//   Cher2kLaunchUnfused4      — 4-launch dispatch, unfused escape hatch (4 gemm_kernel_do)
// Pure code motion: every statement, evaluation order and operand order is kept
// verbatim (numerically zero change); the t1..t4 mapping table stays reviewable as a
// local array with per-port comments inside the two dispatch functions.

// Phase 0 part: core-split deinterleave. The Ar/Ai/Br/Bi four-buffer layout is given by
// this comment (in workspace after the 4 temp areas); the caller derives the pointers
// from the same layout.
static void Cher2kLaunchDeinterleave(
    const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb, uint32_t logicalRows,
    uint32_t physCols, uint32_t aivCoreNum, uint8_t* dAr, uint8_t* dAi, uint8_t* dBr, uint8_t* dBi, aclrtStream stream)
{
    // Phase 0: core-split deinterleave — the cores below the split point handle
    // A, the remaining cores handle B
    Cher2kDeinterleaveTilingData deintTiling{};
    deintTiling.rows = logicalRows;
    deintTiling.cols = physCols;
    deintTiling.lda = lda;
    deintTiling.ldb = ldb;
    uint32_t splitCore = CeilDiv<uint32_t>(aivCoreNum, 2);
    deintTiling.splitCore = splitCore;
    // [deint-2d] Default to the 2D tile-list split (tileMode 1): it removes the
    // row-band starvation when the row count falls below 64 times the split core
    // count (on the conjugate-transposed path with inner dim 64 this gives a single 64-row
    // band, so only 8 of the 28 cores per matrix get work and 40 of the 56 sit
    // idle). Setting CHER2K_DEINT_2D to 0 restores the legacy 1D row band (escape
    // hatch; the output is bit-identical either way).
    deintTiling.tileMode = Cher2kReadTuningKnob("CHER2K_DEINT_2D") ? 0U : 1U;
    // [r2/deint-bulk] Round the per-core row band UP to a multiple of 8
    // (CHER2K_ARCH35_ELEMENTS_PER_BLOCK) so every full tile's blockRows is
    // 8-aligned: the kernel then takes the single-instruction multi-burst
    // load/store (Cher2kLoadCplxBlockBulk / Cher2kStoreRealBlocksBulk) for the
    // whole tile instead of one MTE2/MTE3 call per column. K1 is MTE3-bound
    // (59% on TC_PF_1104) and the per-column path issued two store instructions
    // per block column for each tile — on the conjugate-transposed path the band is tiny (the
    // row count and the inner dim are both 416 across 28 cores, giving 15 rows)
    // so it was ALWAYS padded and never reached the bulk path; the 8-alignment
    // makes the band 16 and the bulk path live for every shape. Cores past the
    // aligned extent go idle (their row start reaches their row end).
    const uint32_t rowsPerCoreRaw = CeilDiv<uint32_t>(logicalRows, splitCore);
    deintTiling.rowsPerCore = CeilAlign<uint32_t>(rowsPerCoreRaw, CHER2K_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t deintBlocks = std::min(aivCoreNum, splitCore * 2);

    OP_LOGI("aclblasCher2k", "launching cher2k deinterleave kernel: aivBlocks=%u splitCore=%u", deintBlocks, splitCore);
    cher2k_deinterleave_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)), reinterpret_cast<GM_ADDR>(dAr),
        reinterpret_cast<GM_ADDR>(dAi), reinterpret_cast<GM_ADDR>(dBr), reinterpret_cast<GM_ADDR>(dBi), deintTiling,
        deintBlocks, stream);
}

// tiling build part: cube tiling + the fused-path L1 half-boundary / P1-E tileKChunk
// clamp. Every clamp and assignment is kept verbatim, only moved out of the main function.
static GemmTilingData CalCher2kProductTiling(
    uint32_t n, uint32_t k, uint32_t logicalRows, uint32_t tempLdc, uint32_t aicCoreNum, aclblasOperation_t transaGemm,
    aclblasOperation_t transbGemm, int unfusedEnv)
{
    GemmTilingData cubeTiling =
        CalCher2kGemmTilingData(n, k, logicalRows, logicalRows, tempLdc, transaGemm, transbGemm);
    PrepareCubeTiling(cubeTiling, aicCoreNum);
    cubeTiling.ldc = static_cast<int32_t>(tempLdc);
    if (unfusedEnv == 0) {
        // Fused kernel L1 layout holds A1, A2, B1 and B2, which together take
        // twice the sum of (BM times tkc) and (tkc times BN) floats, ping-ponged
        // over TWO halves of the L1 size (32KB device L1, so a 16KB half).
        //   fallback 32x16: a tkc of at most 32 needs 12KB, which fits the 16KB
        //                   half.
        //   primary 64x32: a tkc of 32 would need 24KB per half -- 8KB PAST the
        //                  16KB half boundary. [l123d ROOT CAUSE] That overflow
        //                  is what made the L1 landing non-deterministically
        //                  wrong: the second half (l1BufId 1) writes the range
        //                  from 16K to 40K, stomping past the nominal 32KB L1
        //                  while the other half's data may still be live -- the
        //                  corruption depends on MTE1/MTE2 timing, which is
        //                  exactly the "same seed, different output" signature the
        //                  four-quadrant A/B isolated to the 64x32 shape (l123c).
        //                  The probe binary "worked" only because its golden
        //                  spot-check itself reported 15 of 12 bad elements at
        //                  order 1024 (tmp/probe-decision pb_s64x32.log line 1-2)
        //                  -- the probe never had a clean accuracy gate at 64x32.
        // Fix: clamp tkc so the 4-operand chunk fits the half:
        //   the 64x32 shape allows a tkc of at most 16 (needs 12KB, which fits the 16KB half)
        //   the 32x16 shape allows a tkc of at most 32 (12KB, fits, unchanged)
        const uint32_t tkcLimit = (cubeTiling.baseM == 64) ? 16U : 32U;
        if (static_cast<uint32_t>(cubeTiling.tileKChunk) > tkcLimit) {
            cubeTiling.tileKChunk = static_cast<int32_t>(tkcLimit);
        }
        // [L1][P1-E] per-(shape,order,trans) tileKChunk lookup, extending the
        // P1-E E-table mechanism. History: the global tkc is shape-blind and
        // the r2final global sweeps showed opposite optima on the two
        // transposes at the OLD 32x16 shape (with order and inner dim both 1024,
        // the non-transposed path preferred a tkc of 24 while the conjugate-transposed path regressed
        // with it), which is why the knob is keyed on the order and the transpose.
        // L1 re-keyed those optima at the new primary shape: the probe tkc
        // micro-sweep at 64x32 measured, at order 1024 on UPPER/N, tkc values of
        // 8, 16, 24 and 32 giving 653.9, 605.3, 607.3 and 601.7, and at order
        // 2048 on LOWER/C, tkc values of 16 and 32 giving 3824.8 and 3795.4, i.e.
        // a tkc of 32 is optimal or tied on the primary shape — so the P1-E
        // order-1024/N exception (a tkc of 24, a 32x16-era result) is DROPPED for
        // the primary shape and kept only for the fallback shape where it was
        // measured. Every other shape keeps a tkc of 32.
        if (cubeTiling.baseM == 32 && n == 1024 && transaGemm == ACLBLAS_OP_N) {
            cubeTiling.tileKChunk = 24;
        }
    }
    return cubeTiling;
}

// 4-launch dispatch part (fused default path). t1..t4 mapping table [codecheck #4: local array + comments]:
//   dC item i points at the temp frame written by the (i+1)-th kernel output port (c1..c4).
//   Kernel pairing: output port c1 is X1 times Y1 (stored in t1), c2 is X2 times Y2 (stored in t2),
//     c3 is X1 times Y2, c4 is X2 times Y1.
//     N path: t3 comes from X1 times Y2, t4 from X2 times Y1, so c3 lands in dT3 and c4 in dT4
//     C path: t3 comes from X2 times Y1, t4 from X1 times Y2, so c3 lands in dT4 and c4 in dT3
static void Cher2kLaunchFused4(
    uint32_t numBlocks, aclrtStream stream, uint8_t* dAr, uint8_t* dAi, uint8_t* dBr, uint8_t* dBi, uint8_t* dT1,
    uint8_t* dT2, uint8_t* dT3, uint8_t* dT4, const GemmTilingData& cubeTiling, aclblasOperation_t trans,
    uint32_t foldMode, const Cher2kCombineParams& combineParams)
{
    uint8_t* dC[4] = {dT1, dT2, nullptr, nullptr}; // output ports c1->t1, c2->t2 (fixed)
    dC[2] = (trans == ACLBLAS_OP_N) ? dT3 : dT4;   // c3 from X1*Y2: N path -> t3, C path -> t4
    dC[3] = (trans == ACLBLAS_OP_N) ? dT4 : dT3;   // c4 from X2*Y1: N path -> t4, C path -> t3
    // [MIX fold] foldMode 0xFFFFFFFF selects the legacy AIC-only fixpipe path (the default).
    // Any other value selects the MIX kernel with that AIV sink; under
    // CHER2K_FOLD_PRPI the c2/c4 pointers are unused (PR to t1, PI to t3).
    if (foldMode == CHER2K_FOLD_LEGACY) {
        cher2k_fused4_do(
            numBlocks, stream, reinterpret_cast<GM_ADDR>(dBr), reinterpret_cast<GM_ADDR>(dBi),
            reinterpret_cast<GM_ADDR>(dAr), reinterpret_cast<GM_ADDR>(dAi), reinterpret_cast<GM_ADDR>(dC[0]),
            reinterpret_cast<GM_ADDR>(dC[1]), reinterpret_cast<GM_ADDR>(dC[2]), reinterpret_cast<GM_ADDR>(dC[3]),
            cubeTiling);
        return;
    }
    // [stage2 Step 3a] The combine parameter channel is threaded into the MIX
    // kernel here (stored, not yet consumed). The legacy AIC-only path above
    // takes no such parameter, so its behaviour is untouched.
    cher2k_fused4_mix_do(
        numBlocks, stream, reinterpret_cast<GM_ADDR>(dBr), reinterpret_cast<GM_ADDR>(dBi),
        reinterpret_cast<GM_ADDR>(dAr), reinterpret_cast<GM_ADDR>(dAi), reinterpret_cast<GM_ADDR>(dC[0]),
        reinterpret_cast<GM_ADDR>(dC[1]), reinterpret_cast<GM_ADDR>(dC[2]), reinterpret_cast<GM_ADDR>(dC[3]),
        cubeTiling, foldMode, combineParams);
    // [p1l R1] no twin fill: K3 reads the (j,i) partner tile directly
    // from the t frames (swapped-origin 2D strided copy).
}

// 4-launch dispatch part (unfused escape hatch, CHER2K_FUSED_GEMM set to 0). t1..t4 operand mapping
// table [codecheck #4: local table + comments]: on the N path gemm_kernel_do(X, Y, dTi) stores Y times
// the transpose of X and on the C path the transpose of Y times X (M2 probe, see the
// RunCher2kGemmPhase header comment):
//     N path: t1 from Br,Ar gives Ar * Br^T; t2 from Bi,Ai gives Ai * Bi^T
//             t3 from Br,Ai gives Ai * Br^T; t4 from Bi,Ar gives Ar * Bi^T
//     C path: t1 from Br,Ar gives Ar^T * Br; t2 from Bi,Ai gives Ai^T * Bi
//             t3 from Bi,Ar gives Ar^T * Bi; t4 from Br,Ai gives Ai^T * Br
// t1/t2 operands match on both the N and C paths; only t3/t4 swap with trans (the same difference
// table as the c3/c4 output-port swap of the fused path).
static void Cher2kLaunchUnfused4(
    uint32_t numBlocks, aclrtStream stream, uint8_t* dAr, uint8_t* dAi, uint8_t* dBr, uint8_t* dBi, uint8_t* dT1,
    uint8_t* dT2, uint8_t* dT3, uint8_t* dT4, const GemmTilingData& cubeTiling, aclblasOperation_t trans)
{
    OP_LOGI("aclblasCher2k", "launching 4 gemm kernels: aicBlocks=%u", numBlocks);

    // t1/t2 operand order is identical on both paths (§3.2 table);
    // only the t3/t4 operands differ between N and C.
    gemm_kernel_do(numBlocks, stream, dBr, dAr, dT1, cubeTiling);     // t1: Ar * Br^T | Ar^T * Br
    gemm_kernel_do(numBlocks, stream, dBi, dAi, dT2, cubeTiling);     // t2: Ai * Bi^T | Ai^T * Bi
    if (trans == ACLBLAS_OP_N) {
        gemm_kernel_do(numBlocks, stream, dBr, dAi, dT3, cubeTiling); // t3: Ai * Br^T
        gemm_kernel_do(numBlocks, stream, dBi, dAr, dT4, cubeTiling); // t4: Ar * Bi^T
    } else {
        gemm_kernel_do(numBlocks, stream, dBi, dAr, dT3, cubeTiling); // t3: Ar^T * Bi
        gemm_kernel_do(numBlocks, stream, dBr, dAi, dT4, cubeTiling); // t4: Ai^T * Br
    }
}

static aclblasStatus_t RunCher2kGemmPhase(
    _aclblas_handle* h, const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb,
    uint32_t logicalRows, uint32_t physCols, size_t arBytes, uint32_t n, uint32_t k, aclblasOperation_t trans,
    uint32_t aicCoreNum, uint32_t aivCoreNum, uint32_t tempLdc, uint8_t* wsBase, size_t tempBytes, uint8_t* dT1,
    uint8_t* dT2, uint8_t* dT3, uint8_t* dT4, uint32_t foldMode, const Cher2kCombineParams& combineParams)
{
    aclrtStream stream = h->stream;

    // Ar, Ai, Br, Bi allocated in workspace after the 4 temp areas
    uint8_t* dAr = wsBase + tempBytes * 4;
    uint8_t* dAi = dAr + arBytes;
    uint8_t* dBr = dAi + arBytes;
    uint8_t* dBi = dBr + arBytes;

    // Phase 0: core-split deinterleave (A/B deinterleaved into Ar/Ai/Br/Bi).
    Cher2kLaunchDeinterleave(A, lda, B, ldb, logicalRows, physCols, aivCoreNum, dAr, dAi, dBr, dBi, stream);

    // Phase 1: 4 real products.
    // Two execution paths, byte-identical outputs (validated by the accuracy
    // gate + whitebox suite; see the perf-iteration log):
    //   fused  (default): one cher2k_fused4_do launch; each K-tile loads the
    //                     4 real operands once and issues 4 Mmad.
    //   unfused (when CHER2K_FUSED_GEMM is set to 0): the original 4
    //                     gemm_kernel_do launches (A/B-comparison escape hatch).
    // Operand mapping derivation (post-swap row-major kernel view — the
    // shared kernel's first arg is the row-major "X" side, second the "Y"
    // side; both paths launch t1/t2 as (Br,Ar) and (Bi,Ai)):
    //   on the non-transposed path: t3 is (Br,Ai) and t4 is (Bi,Ar), so t3 is X1
    //            times Y2 and t4 is X2 times Y1
    //   on the conjugate-transposed path: t3 is (Bi,Ar) and t4 is (Br,Ai), so t3
    //            is X2 times Y1 and t4 is X1 times Y2
    // so the (X1,X2,Y1,Y2) tuple is (Br,Bi,Ar,Ai) on BOTH paths and only the
    // t3/t4 output pointers swap for the conjugate-transposed path.
    // Fused 4-product kernel (perf-iter2, deadlocked then; FIXED in r2):
    // the deadlock was a missing M-to-FIX handshake in FusedProcessTile — see
    // the [r2/G1 deadlock fix] comment in cher2k_kernel.cpp for the line-level
    // flag-accounting root cause (the trailing WaitFlag<FIX_M> consumed the
    // id-0 credit the next tile's leading Wait needed). It is now the DEFAULT
    // path; setting CHER2K_FUSED_GEMM to 0 falls back to the 4-launch path as an
    // emergency escape hatch (A/B-comparison / regression bisect).
    // Read once (magic static, see Cher2kReadTuningKnob): the process environment is
    // constant, so the first and every later call return the same value, as the former
    // function-local static lambda did.
    const int unfusedEnv = Cher2kReadTuningKnob("CHER2K_FUSED_GEMM") ? 1 : 0;
    aclblasOperation_t transaGemm = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_N : ACLBLAS_OP_T;
    aclblasOperation_t transbGemm = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_T : ACLBLAS_OP_N;

    GemmTilingData cubeTiling =
        CalCher2kProductTiling(n, k, logicalRows, tempLdc, aicCoreNum, transaGemm, transbGemm, unfusedEnv);

    uint32_t numBlocks = static_cast<uint32_t>(cubeTiling.usedCoreNum);
    if (!unfusedEnv) {
        OP_LOGI(
            "aclblasCher2k",
            "launching fused 4-product gemm kernel: aicBlocks=%u tile=%dx%d tkc=%d mb=%d nb=%d fold=%u", numBlocks,
            cubeTiling.baseM, cubeTiling.baseN, cubeTiling.tileKChunk, cubeTiling.mBlocks, cubeTiling.nBlocks,
            foldMode);
        Cher2kLaunchFused4(
            numBlocks, stream, dAr, dAi, dBr, dBi, dT1, dT2, dT3, dT4, cubeTiling, trans, foldMode, combineParams);
        return ACLBLAS_STATUS_SUCCESS;
    }
    Cher2kLaunchUnfused4(numBlocks, stream, dAr, dAi, dBr, dBi, dT1, dT2, dT3, dT4, cubeTiling, trans);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  LaunchCher2kSmallKernel — K4 fused SIMT path (orders up to CHER2K_ARCH35_SIMT_N_MAX)
//  Fused direct compute, no workspace, no Phase0/Phase1.
//  The skipTemp case (a zero alpha or a zero inner dim) is served naturally: the
//  k-loop contributes nothing and only beta times C survives (with a zero inner
//  dim the loop body never runs; a zero alpha zeroes the dot products through the
//  alpha multiply).
// ==========================================================================
static aclblasStatus_t LaunchCher2kSmallKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb, aclblasComplex* C, uint32_t ldc,
    const aclblasComplex* alpha, const float* beta, float alphaReal, float alphaImag, float betaVal, bool isAlphaDev,
    bool isBetaDev, uint32_t aivCoreNum, uint32_t usedAivCoreNum)
{
    Cher2kSimtTilingData tiling{};
    tiling.n = n;
    tiling.k = k;
    tiling.ldc = ldc;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.rowsPerCore = CeilDiv<uint32_t>(n, usedAivCoreNum);
    tiling.transN = (trans == ACLBLAS_OP_N) ? 1U : 0U;
    // [iter34] scalars: real values when the host dereferenced HOST pointers,
    // placeholders when the kernel reads the DEVICE pointer from GM.
    tiling.alphaReal = alphaReal;
    tiling.alphaImag = alphaImag;
    tiling.betaVal = betaVal;
    tiling.uploMode = static_cast<uint8_t>(uplo);
    tiling.isAlphaDev = static_cast<uint8_t>(isAlphaDev ? 1 : 0);
    tiling.isBetaDev = static_cast<uint8_t>(isBetaDev ? 1 : 0);

    OP_LOGI(
        "aclblasCher2k", "launching cher2k simt small kernel: n=%u k=%u aivCores=%u trans=%d", n, k, usedAivCoreNum,
        static_cast<int>(trans));
    (void)aivCoreNum;
    cher2k_simt_small_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)), reinterpret_cast<GM_ADDR>(C),
        isAlphaDev ? reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(alpha)) : nullptr,
        isBetaDev ? reinterpret_cast<GM_ADDR>(const_cast<float*>(beta)) : nullptr, tiling, usedAivCoreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  LaunchCher2kKernel — general pipeline: Phase0 → Phase1(×4) → Phase2
//  Branch table (design §4.2), evaluated after validation:
//    an order of zero                        → SUCCESS (API entry)
//    a null C with a positive order          → a non-zero beta: INVALID_VALUE /
//                                              a zero beta: SUCCESS with no write (API entry)
//    a zero alpha or a zero inner dim,
//    together with a beta of one             → SUCCESS, no data access (below)
//    a zero alpha or a zero inner dim,
//    together with a zero beta               → K3/K4 skipTemp, isBetaZero (zero uplo)
//    a zero alpha or a zero inner dim,
//    together with any other beta            → K3/K4 skipTemp (beta scale only)
//    a general product with a zero beta      → K1 + 4×K2 + K3 (no old C read)
//    a general product with a non-zero beta  → K1 + 4×K2 + K3
//    an order up to 8 (CHER2K_ARCH35_SIMT_N_MAX) → K4
// ==========================================================================
// [codecheck #6] tiling assembly part: workspace preparation + temp frame pointer layout.
// The output parameters match PrepareCher2kWorkspace's out-params; dT1..dT4 derive from wsBase.
struct Cher2kLaunchAssembly {
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    uint32_t usedAivCoreNum;
    uint32_t tempLdc;
    uint32_t logicalRows;
    uint32_t physCols;
    size_t arBytes;
    size_t tempBytes;
    uint8_t* wsBase;
    uint8_t* dT1;
    uint8_t* dT2;
    uint8_t* dT3;
    uint8_t* dT4;
};

static aclblasStatus_t AssembleCher2kLaunch(
    _aclblas_handle* h, uint32_t n, uint32_t k, aclblasOperation_t trans, bool skipTemp, Cher2kLaunchAssembly& asm_)
{
    aclblasStatus_t st = PrepareCher2kWorkspace(
        h, n, k, trans, skipTemp, asm_.aicCoreNum, asm_.aivCoreNum, asm_.usedAivCoreNum, asm_.tempLdc, asm_.logicalRows,
        asm_.physCols, asm_.arBytes, asm_.tempBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k", "LaunchCher2kKernel: PrepareCher2kWorkspace failed, st=%d", static_cast<int>(st));
        return st;
    }

    asm_.wsBase = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    asm_.dT1 = asm_.wsBase;
    asm_.dT2 = asm_.dT1 + asm_.tempBytes;
    asm_.dT3 = asm_.dT2 + asm_.tempBytes;
    asm_.dT4 = asm_.dT3 + asm_.tempBytes;
    // [p1l R1] the transposed twins (P1-D, which started at dT4 plus tempBytes)
    // were removed; the workspace shrank back to four tempBytes plus four arBytes.
    //
    // [skipTemp safety] When skipTemp is set, workspaceNeed is 0 bytes — the
    // dT1..dT4 pointers above are still derived from wsBase (which may be a
    // 0-length/absent buffer) and passed into K3. This is safe: K3 with both the
    // alpha-zero and inner-dim-zero flags set never dereferences the t frames (the
    // M plus M-conjugate term is identically 0, so only the beta-times-C
    // accumulator is read and written), and SetGlobalBuffer merely stores the
    // pointer without touching GM.
    return ACLBLAS_STATUS_SUCCESS;
}

// [codecheck #6] Path 1 of the three-way dispatch: K4 small-n fused SIMT (order at most 8).
// The skipTemp cases are covered naturally (a zero k leaves the loop body unexecuted; a zero
// alpha zeroes the result through the alpha multiply).
// [iter34] The DEVICE pointers for alpha/beta pass straight through to the K4 kernel (the
// kernel reads them from GM and decides quick-return / scalar itself); the host no longer
// does a D2H staging read.
static aclblasStatus_t DispatchCher2kSmallPath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb, aclblasComplex* C, uint32_t ldc,
    const aclblasComplex* alpha, const float* beta, float alphaReal, float alphaImag, float betaVal, bool isAlphaDev,
    bool isBetaDev)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCher2k", "DispatchCher2kSmallPath: GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t usedAivCoreNum = GetUsedAivCoreNum(n, aivCoreNum);
    return LaunchCher2kSmallKernel(
        h, uplo, trans, n, k, A, lda, B, ldb, C, ldc, alpha, beta, alphaReal, alphaImag, betaVal, isAlphaDev, isBetaDev,
        aivCoreNum, usedAivCoreNum);
}

// [codecheck B4/C3] foldMode resolution: Phase-1 sink selector + the pair-walk
// control bits. Split out of DispatchCher2kPipelinePaths (pure derivation).
//
// [MIX fold] select the Phase-1 sink. Under CHER2K_FOLD_PRPI the K2f MIX
// kernel already folded t1..t4 into PR and PI, so the combine must NOT fold
// again (the pr-pi-folded flag is set) and only reads PR (t1) and PI (t3).
//
// [stage2 Step 2] When CHER2K_PAIR_WALK is set to 1: switch the MIX kernel's
// tile walk from the mi by ni full grid to the symmetric tile-PAIR walk (design
// §2.3/§2.6). The uplo order bit and the pair-walk enable bit ride along in
// foldMode (the launch signature is unchanged). Gated to orders of 64 and above
// (the pair grid needs the 64x64 fused shape; the kernel also degrades to the
// grid walk if the block row size is not 64). Default: off.
//
// [perf-iter83] The AIV sink is NOT forced any more. The original Step 2
// pinned it to CHER2K_FOLD_STORE4 ("immediate drain", no combine yet), which
// writes FOUR frames per tile instead of two — measured with msprof Memory
// (TC_PF_1144): the fused4 write grew from 16.06MB to 32.12MB and the combine
// read from 16.97MB to 33.81MB, i.e. exactly double the intermediate traffic,
// and the fused4 duration went from 37.6us to 90.8us. That is the bulk of the
// 89% pair-walk regression this slice is meant to remove. Keeping the default
// PRPI sink writes PR and PI only (two frames) and leaves the pr-pi-folded flag
// set for the combine, so the pair walk pays no traffic penalty versus the grid
// walk. The pair-walk bits still ride in foldMode; only the sink is now
// inherited.
// [perf-iter93] Lower order bound. The pair walk reorders tiles so that a tile
// (I,J) and its partner (J,I) are emitted back-to-back on one core, which breaks
// the grid walk's A-panel reuse (the grid walk keeps mi outer and ni inner, so A
// stays fixed across the whole inner loop; the pair walk alternates between two
// different A rows). For large orders the win from pairing dominates, but at
// small orders the tile count is too low for that to pay off. Measured on
// TC_PF_1089 (order 1024, inner dim 64): the pair walk took 55.00 to 55.21us
// against 54.67 to 54.94us for the grid, across 5 runs each — a clean,
// reproducible separation (not noise) that drops the ratio from 0.402 to 0.399
// and flips a passing case to FAIL. The three cases the pair walk does flip
// (1040/1097 at order 1818, 1113 at order 1579) are all far above this bound, so
// gating here keeps every gain and removes the regression.
static uint32_t Cher2kResolveFoldMode(aclblasFillMode_t uplo, uint32_t n, bool skipTemp)
{
    uint32_t foldMode = skipTemp ? CHER2K_FOLD_LEGACY : Cher2kSelectFoldMode();
    if (!skipTemp && foldMode != CHER2K_FOLD_LEGACY && n >= CHER2K_PAIR_WALK_MIN_N &&
        Cher2kTuningKnobDefaultOn("CHER2K_PAIR_WALK")) {
        foldMode |= CHER2K_FOLD_PAIRWALK_BIT;
        if (uplo == ACLBLAS_LOWER) {
            foldMode |= CHER2K_FOLD_LOWER_BIT;
        }
        OP_LOGI(
            "aclblasCher2k", "[PAIR_WALK] enabled: n=%u uplo=%s sink=%s", n, uplo == ACLBLAS_LOWER ? "LOWER" : "UPPER",
            (foldMode & ~(CHER2K_FOLD_PAIRWALK_BIT | CHER2K_FOLD_LOWER_BIT)) == CHER2K_FOLD_PRPI ? "PRPI" : "STORE4");
    }
    return foldMode;
}

// [codecheck B4/C3] Phase 0+1 dispatch (the K1+K2 4-launch GEMM). Split out of
// DispatchCher2kPipelinePaths; the launch arguments/order are unchanged.
static aclblasStatus_t Cher2kDispatchGemmPhase(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb, aclblasComplex* C, uint32_t ldc,
    float alphaReal, float alphaImag, float betaVal, bool isAlphaZero, bool isKZero, bool isBetaZero, uint32_t foldMode,
    const Cher2kLaunchAssembly& asm_)
{
    // [stage2 Step 3a] Assemble the combine parameter channel for the MIX
    // kernel (C ptr / ldc / alpha / beta / uplo / zero flags). Pure pipe:
    // the kernel stores it but does not consume it yet, so the launched
    // computation is unchanged.
    Cher2kCombineParams combineParams{};
    combineParams.cGm = reinterpret_cast<uint8_t*>(C);
    combineParams.ldc = ldc;
    combineParams.tempLdc = asm_.tempLdc;
    combineParams.alphaReal = alphaReal;
    combineParams.alphaImag = alphaImag;
    combineParams.betaVal = betaVal;
    combineParams.uploMode = static_cast<uint8_t>(uplo);
    combineParams.isAlphaZero = static_cast<uint8_t>(isAlphaZero ? 1 : 0);
    combineParams.isKZero = static_cast<uint8_t>(isKZero ? 1 : 0);
    combineParams.isBetaZero = static_cast<uint8_t>(isBetaZero ? 1 : 0);
    aclblasStatus_t st = RunCher2kGemmPhase(
        h, A, lda, B, ldb, asm_.logicalRows, asm_.physCols, asm_.arBytes, n, k, trans, asm_.aicCoreNum, asm_.aivCoreNum,
        asm_.tempLdc, asm_.wsBase, asm_.tempBytes, asm_.dT1, asm_.dT2, asm_.dT3, asm_.dT4, foldMode, combineParams);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k", "LaunchCher2kKernel: RunCher2kGemmPhase failed, st=%d", static_cast<int>(st));
    }
    return st;
}

// [codecheck B4/C3] K3 combine dispatch. Split out of DispatchCher2kPipelinePaths;
// the tiling derivation, the diagnostic call and the launch order are unchanged.
static void Cher2kDispatchCombine(
    _aclblas_handle* h, aclblasFillMode_t uplo, uint32_t n, uint32_t ldc, const aclblasComplex* alpha,
    const float* beta, aclblasComplex* C, float alphaReal, float alphaImag, float betaVal, bool isAlphaZero,
    bool isKZero, bool isBetaZero, bool skipTemp, bool isAlphaDev, bool isBetaDev, uint32_t foldMode,
    const Cher2kLaunchAssembly& asm_)
{
    // [iter34] alphaReal/alphaImag/betaVal and the isAlphaZero/isBetaZero fields
    // passed to CalCombineTilingData are the real host-dereferenced values on the
    // HOST-scalar path and placeholders on the DEVICE-scalar path; the combine
    // kernel overwrites them from the GM alpha/beta pointers (isAlphaDev/
    // isBetaDev) and derives isAlphaZero/isBetaZero/skipTemp itself.
    Cher2kCombineTilingData combineTiling = CalCombineTilingData(
        asm_.usedAivCoreNum, n, ldc, asm_.tempLdc, static_cast<uint8_t>(isAlphaZero ? 1 : 0),
        static_cast<uint8_t>(isKZero ? 1 : 0), static_cast<uint8_t>(isBetaZero ? 1 : 0), static_cast<uint8_t>(uplo),
        alphaReal, alphaImag, betaVal);
    // [perf-iter83] Derive the combine's sink from the foldMode CONTROL bits
    // only. foldMode now carries the pair-walk and lower-order bits alongside the
    // sink selector, so a plain equality test against CHER2K_FOLD_PRPI is false
    // whenever the pair walk is on — which would tell the combine to fold PR and
    // PI itself even though the MIX kernel already did, and it would then read the
    // un-written t2/t4 frames.
    combineTiling.isPrPiFolded = static_cast<uint8_t>(
        (foldMode & ~(CHER2K_FOLD_PAIRWALK_BIT | CHER2K_FOLD_LOWER_BIT)) == CHER2K_FOLD_PRPI ? 1 : 0);
    combineTiling.isAlphaDev = static_cast<uint8_t>(isAlphaDev ? 1 : 0);
    combineTiling.isBetaDev = static_cast<uint8_t>(isBetaDev ? 1 : 0);

    const uint32_t combineCores = TuneCombineCoreNum(n, asm_.usedAivCoreNum);
    // [stage2 Step 1] Symmetric tile-pair enumeration diagnostic (default off; set CHER2K_PAIR_ENUM to 1 to enable).
    // Purely read-only; no change to the launch path.
    Cher2kReportPairEnum(n, static_cast<uint8_t>(uplo), combineCores);
    OP_LOGI(
        "aclblasCher2k",
        "launching cher2k combine kernel: aivCores=%u (tuned from %u) tileMode=%d tiles=%u skipTemp=%d isBetaZero=%d",
        combineCores, asm_.usedAivCoreNum, combineTiling.tileMode, combineTiling.tileTotal, skipTemp ? 1 : 0,
        isBetaZero ? 1 : 0);
    cher2k_combine_kernel_do(
        reinterpret_cast<GM_ADDR>(asm_.dT1), reinterpret_cast<GM_ADDR>(asm_.dT2), reinterpret_cast<GM_ADDR>(asm_.dT3),
        reinterpret_cast<GM_ADDR>(asm_.dT4), reinterpret_cast<GM_ADDR>(C),
        isAlphaDev ? reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(alpha)) : nullptr,
        isBetaDev ? reinterpret_cast<GM_ADDR>(const_cast<float*>(beta)) : nullptr, combineTiling, combineCores,
        h->stream);
}

// [codecheck #6] Paths 2/3 of the three-way dispatch: the Phase0+Phase1 4-launch GEMM (K1+K2,
// skipped under skipTemp) and the K3 combine. Both share the same workspace/temp frame layout,
// so they are launched in their original order inside this function (GEMM before combine; the
// order cannot be swapped).
static aclblasStatus_t DispatchCher2kPipelinePaths(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb, aclblasComplex* C, uint32_t ldc,
    const aclblasComplex* alpha, const float* beta, float alphaReal, float alphaImag, float betaVal, bool isAlphaZero,
    bool isKZero, bool isBetaZero, bool skipTemp, bool isAlphaDev, bool isBetaDev, const Cher2kLaunchAssembly& asm_)
{
    const uint32_t foldMode = Cher2kResolveFoldMode(uplo, n, skipTemp);
    if (!skipTemp) {
        const aclblasStatus_t st = Cher2kDispatchGemmPhase(
            h, uplo, trans, n, k, A, lda, B, ldb, C, ldc, alphaReal, alphaImag, betaVal, isAlphaZero, isKZero,
            isBetaZero, foldMode, asm_);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    Cher2kDispatchCombine(
        h, uplo, n, ldc, alpha, beta, C, alphaReal, alphaImag, betaVal, isAlphaZero, isKZero, isBetaZero, skipTemp,
        isAlphaDev, isBetaDev, foldMode, asm_);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCher2kKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* alpha, const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb,
    const float* beta, aclblasComplex* C, uint32_t ldc)
{
    float alphaReal;
    float alphaImag;
    float betaVal;
    bool isAlphaZero;
    bool isKZero;
    bool isBetaZero;
    bool skipTemp;
    bool isAlphaDev;
    bool isBetaDev;
    aclblasStatus_t st = PrepareCher2kParams(
        h, k, alpha, beta, alphaReal, alphaImag, betaVal, isAlphaZero, isKZero, isBetaZero, skipTemp, isAlphaDev,
        isBetaDev);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k", "LaunchCher2kKernel: PrepareCher2kParams failed, st=%d", static_cast<int>(st));
        return st;
    }

    // [iter34] On the DEVICE-scalar path the BLAS quick-return (a zero alpha or
    // a zero inner dim, together with a beta of one) cannot be decided on the
    // host (alpha/beta are not staged any more); it is enforced in-kernel (both
    // K4 and K3 early-return without touching C when they read a zero alpha, a
    // zero inner dim and a beta of one from GM). On the HOST-scalar path the
    // flags above are already exact and the same early-return fires in-kernel from
    // the tiling scalars, so C is byte-unchanged in both cases. (The GEMM phase
    // for a zero inner dim is still skipped host-side, since skipTemp tracks the
    // inner-dim-zero flag, so nothing writes the temp frames.)

    // Path 1: K4 small-n fused SIMT path, no workspace (skipTemp cases are
    // also served: a zero alpha or a zero inner dim reduce to beta scaling, which
    // K4 applies directly)
    if (n <= CHER2K_ARCH35_SIMT_N_MAX) {
        return DispatchCher2kSmallPath(
            h, uplo, trans, n, k, A, lda, B, ldb, C, ldc, alpha, beta, alphaReal, alphaImag, betaVal, isAlphaDev,
            isBetaDev);
    }

    // Paths 2+3: tiling assembly (workspace + temp frame layout) -> Phase0/1 4-launch -> K3.
    Cher2kLaunchAssembly asm_{};
    st = AssembleCher2kLaunch(h, n, k, trans, skipTemp, asm_);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k", "LaunchCher2kKernel: AssembleCher2kLaunch failed, st=%d", static_cast<int>(st));
        return st;
    }
    return DispatchCher2kPipelinePaths(
        h, uplo, trans, n, k, A, lda, B, ldb, C, ldc, alpha, beta, alphaReal, alphaImag, betaVal, isAlphaZero, isKZero,
        isBetaZero, skipTemp, isAlphaDev, isBetaDev, asm_);
}

// ==========================================================================
//  Public API entry
// ==========================================================================
extern "C" aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const float* beta, aclblasComplex* C, int ldc)
{
    // #1 the handle must not be null
    if (handle == nullptr) {
        OP_LOGE("aclblasCher2k", "aclblasCher2k: handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // #2..#10 (full validation before any quick return)
    aclblasStatus_t st = ValidateCher2kParams(uplo, trans, n, k, lda, ldb, ldc, alpha, A, B, beta, C);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);

    // #11 a null C together with a positive order (needs the stream for the
    // staging read of beta)
    if (C == nullptr && n > 0) {
        return ResolveCher2kCNull(h, beta, n);
    }

    // quick return when the order is zero
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchCher2kKernel(
        h, uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha, A, static_cast<uint32_t>(lda), B,
        static_cast<uint32_t>(ldb), beta, C, static_cast<uint32_t>(ldc));
}
