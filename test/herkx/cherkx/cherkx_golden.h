/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHERKX_GOLDEN_H
#define CHERKX_GOLDEN_H

#include <algorithm>
#include <cstddef>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

// Sentinel: all validations passed, proceed to compute.
static const int CHERKX_CPU_VALID_OK = 0x7FFFFFFF;

// ── workspace 上限镜像常量（检视 #1：workspace 无符号回绕修复，golden 同步）──
// 权威出处（golden 不直接 include 算子内部头——其含 log/log.h 依赖，会引入 dlog 符号；
// 按既有镜像常量惯例引用并标注出处，联调时与 host 终态对齐）：
//   - ACLBLAS_MAX_WORKSPACE_SIZE = 2 GiB     ← blas/common/helper/aclblas_handle_internal.h:29
//   - 单项（每矩阵）上限 = MAX/4 = 512 MiB    ← 检视 #1 修复语义（developer-code，
//     cherkx_host.cpp PrepareCherkxWorkspace 乘法前单项预检：realBytes/tempBytes 任一
//     单项 > MAX/4 → ACLBLAS_STATUS_ALLOC_FAILED；×4 组乘法（4 real + 4 temp）由此不再回绕）
//   - GEMM_FRACTAL = 16（tempLdc 对齐）       ← blas/gemm/arch35/gemm_tiling_data.h:19
static const size_t CHERKX_WS_MAX = 2ULL * 1024ULL * 1024ULL * 1024ULL;
static const size_t CHERKX_WS_PER_TERM_LIMIT = CHERKX_WS_MAX / 4;
static const size_t CHERKX_GEMM_FRACTAL = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Parameter validation (测试方案 §2.1 步骤 2；O2 决议：完整对齐 ssymm 终态校验顺序
// 与错误码语义；权威源 = 需求文档 §2.3 参数规格 / §4 差距表)
//   Key difference from cherk: trans accepts ONLY OP_N / OP_C — OP_T is rejected
//   outright (HERKX is a cuBLAS extension that mandates conjugate-transpose or
//   no-transpose; a pure transpose breaks Hermitian semantics). OP_T is NOT
//   mapped to CblasConjTrans here, matching the NPU operator and her2k behaviour.
//   非法 uplo / trans（含 OP_T、任意非法值）→ INVALID_ENUM（差距 2，原 INVALID_VALUE）。
// ─────────────────────────────────────────────────────────────────────────────
static bool HerkxIsValidUplo(aclblasFillMode_t uplo)
{
    return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER;
}

static bool HerkxIsValidTrans(aclblasOperation_t trans)
{
    // OP_T(112) is explicitly rejected — only OP_N(111) / OP_C(113) are valid.
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_C;
}

// 复数零判定（差距 1 引入的关键新边界，测试方案 §2.1 步骤 3 / §3.2 特殊值）：
// α==0 当且仅当 实部==0 且 虚部==0；纯虚 α=(0,±1) 为**非零**（TC_AL_13 专测）。
static inline bool HerkxAlphaIsComplexZero(const aclblasComplex* alpha)
{
    return alpha->real == 0.0f && alpha->imag == 0.0f;
}

// 校验第 1 段：handle / 枚举 / 维度 / 前导维度（ld）。
// 顺序即 ssymm 终态顺序，**不得重排**（CP2.1 复检实证 + 测试方案 §2.1 步骤 2 明示）：
//   handle → uplo/trans 枚举 → n<0/k<0 → n==0 早退 → lda/ldb/ldc。
// n==0 早退位于 ld 与指针校验**之前**：其后不再校验指针与 ld（TC_AL_15 验证点）。
static int ValidateHerkxEnumsAndDims(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, int lda, int ldb, int ldc)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (!HerkxIsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (!HerkxIsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // n==0 fast return — no access to A/B/C, no ld validation (测试方案 §4.2 S-B1).
    if (n == 0) {
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
    }
    int minLdab = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < minLdab) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (ldb < minLdab) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CHERKX_CPU_VALID_OK;
}

// 校验第 2 段：标量与矩阵指针的条件化非空（n>0 已由第 1 段保证）。
//   - alpha/beta 为 nullptr → INVALID_VALUE；
//   - A/B 条件化非空：仅当 α≠0（复数零判定）且 n>0 且 k>0 时要求非空；
//     α==0 或 k==0 时 A/B 可为 nullptr（TC_AL_11/12/16 允许侧、TC_AL_13 拒绝侧）；
//   - C 条件化非空：仅当 β≠0 且 n>0 时要求非空；**β==0 且 C==nullptr → SUCCESS
//     早退**（差距 3；Host β：TC_L0_15/AL_08，Device β：AL_09——golden 一律按
//     Host 语义解引用标量真值，指针位置是 NPU 侧被测行为，golden 不感知）。
static int ValidateHerkxScalarsAndPtrs(
    int n, int k, const aclblasComplex* alpha, const float* beta,
    const aclblasComplex* A, const aclblasComplex* B,
    aclblasComplex* C, const aclblasComplex* originalC)
{
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    const bool alphaNonZero = !HerkxAlphaIsComplexZero(alpha);
    if (A == nullptr && alphaNonZero && n > 0 && k > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (B == nullptr && alphaNonZero && n > 0 && k > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (C == nullptr) {
        // β==0：C 不是必需输入 → 合法早退（不读不写 C）；β≠0：拒绝。
        return (*beta == 0.0f) ? static_cast<int>(ACLBLAS_STATUS_SUCCESS)
                               : static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (originalC == nullptr && n > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CHERKX_CPU_VALID_OK;
}

// 校验第 3 段：workspace 单项预检（检视 #1 修复镜像，2026-10-15；2026-10-15 晚对齐 host 终态）。
//   host 终态（cherkx_host.cpp PrepareCherkxWorkspace，developer-code 已落地）：预检位于
//   **if (!skipTemp) 分支内**（字节乘法所在处；skipTemp 时 workspaceNeed=0、无乘法无回绕，
//   不预检）。golden 同构：**仅 !skipTemp（α 非复数零 且 k>0）执行预检**，skipTemp 用例
//   跳过预检放行、走既有后续语义（短路径/正常路径不变）。
//   预检内容：任一单项（realBytes = logicalRows×physCols×4B 每实矩阵 / tempBytes =
//   tempLdc×n×4B 每临时矩阵）> ACLBLAS_MAX_WORKSPACE_SIZE/4 → ALLOC_FAILED。镜像实现用
//   **因子除法比较**（a > L/D ⟺ a×D > L，正整数域严格等价），自身不回绕——n=k=2^30 时
//   乘积 2^62、×4 = 2^64 在 size_t 回绕为 0（原缺陷），除法式无此问题。
//   顺序镜像：接线于 ValidateHerkxCpuParams 第 3 段（全部校验通过、C==null&&β==0 早退
//   之后——与 host Launch→PrepareCherkxWorkspace 同序）。
//   另镜像既有总量上限（aclblas_handle_internal.h:201 EnsureDefaultWorkspace：
//   requiredSize > MAX → ALLOC_FAILED；workspaceNeed 仅 !skipTemp 非零，随预检同分支）——
//   单项预检通过后各单项 ≤ 512MiB，组乘 4×单项 ≤ 2GiB、总和 ≤ 4GiB 均在 size_t 内不回绕，
//   可安全直算。512B 对齐粒度不镜像（边界 512B 内差异属实现细节）。
//   前置条件：n > 0（n==0/n<0 已由第 1 段早退/拒绝）。
static int ValidateHerkxWorkspacePrecheck(
    aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha)
{
    // 预检仅 !skipTemp（α 非复数零 且 k>0）——镜像 host 终态：预检在 PrepareCherkxWorkspace
    // 的 if (!skipTemp) 分支内（skipTemp 无乘法无回绕，不预检、放行）。
    if (HerkxAlphaIsComplexZero(alpha) || k == 0) {
        return CHERKX_CPU_VALID_OK;
    }
    const size_t physRows = static_cast<size_t>((trans == ACLBLAS_OP_N) ? n : k);
    const size_t physCols = static_cast<size_t>((trans == ACLBLAS_OP_N) ? k : n);
    // 单项 realBytes = physRows × physCols × sizeof(float)（每实矩阵）
    if (physRows > 0 && physCols > 0 &&
        physRows > CHERKX_WS_PER_TERM_LIMIT / (physCols * sizeof(float))) {
        return static_cast<int>(ACLBLAS_STATUS_ALLOC_FAILED);
    }
    // 单项 tempBytes = tempLdc × n × sizeof(float)（tempLdc = CeilAlign(n, GEMM_FRACTAL)）
    const size_t tempLdc =
        (static_cast<size_t>(n) + CHERKX_GEMM_FRACTAL - 1) / CHERKX_GEMM_FRACTAL * CHERKX_GEMM_FRACTAL;
    if (tempLdc > CHERKX_WS_PER_TERM_LIMIT / (static_cast<size_t>(n) * sizeof(float))) {
        return static_cast<int>(ACLBLAS_STATUS_ALLOC_FAILED);
    }
    // 既有总量上限镜像（EnsureDefaultWorkspace :201）：workspaceNeed = 4×realBytes +
    // 4×tempBytes > MAX → ALLOC_FAILED（已在本分支内保证 !skipTemp；单项通过后乘法不回绕，可直算）
    const size_t realBytes = physRows * physCols * sizeof(float);
    const size_t tempBytes = tempLdc * static_cast<size_t>(n) * sizeof(float);
    if (realBytes * 4 + tempBytes * 4 > CHERKX_WS_MAX) {
        return static_cast<int>(ACLBLAS_STATUS_ALLOC_FAILED);
    }
    return CHERKX_CPU_VALID_OK;
}

static int ValidateHerkxCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, int lda, int ldb, int ldc,
    const aclblasComplex* alpha, const float* beta,
    const aclblasComplex* A, const aclblasComplex* B,
    aclblasComplex* C, const aclblasComplex* originalC)
{
    int st = ValidateHerkxEnumsAndDims(handle, uplo, trans, n, k, lda, ldb, ldc);
    if (st != CHERKX_CPU_VALID_OK) {
        return st; // 错误码，或 n==0 的 SUCCESS 早退（其后不再校验指针与 ld）
    }
    st = ValidateHerkxScalarsAndPtrs(n, k, alpha, beta, A, B, C, originalC);
    if (st != CHERKX_CPU_VALID_OK) {
        return st; // 错误码，或 C==null&&β==0 的 SUCCESS 早退（不发 kernel，host 同序）
    }
    // 第 3 段：workspace 单项预检（镜像 host Launch→PrepareCherkxWorkspace 顶部，检视 #1）
    return ValidateHerkxWorkspacePrecheck(trans, n, k, alpha);
}

// ─────────────────────────────────────────────────────────────────────────────
// uplo-triangle helpers (需求文档 §1 数学定义 / §2.6 精度区域②③④)
//   These operate on the uplo triangle ONLY — the non-uplo triangle is restored
//   separately by RestoreNonUploTriangleComplex. Column-major indexing: idx = i + j*ldc.
// ─────────────────────────────────────────────────────────────────────────────
static inline bool HerkxIsUplo(aclblasFillMode_t uplo, int i, int j)
{
    return (uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
}

// Zero the uplo triangle (including the diagonal) — used by alpha==0&&beta==0 and
// k==0&&beta==0 short-circuit paths. Diagonal imaginary part becomes 0 here too.
static inline void ZeroUploTriangleComplex(
    aclblasFillMode_t uplo, int n, int ldc, aclblasComplex* C)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (HerkxIsUplo(uplo, i, j)) {
                C[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc)] =
                    aclblasComplex{0.0f, 0.0f};
            }
        }
    }
}

// Scale the uplo triangle by a real beta — used by alpha==0&&beta!=0 and k==0&&beta!=0
// short-circuit paths. beta is real, so real and imag parts are both multiplied.
static inline void ScaleUploTriangleComplex(
    aclblasFillMode_t uplo, int n, int ldc, aclblasComplex* C, float beta)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (HerkxIsUplo(uplo, i, j)) {
                size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc);
                C[idx] = aclblasComplex{beta * C[idx].real, beta * C[idx].imag};
            }
        }
    }
}

// 短路径（α 复数零 或 k==0）的 uplo 三角 β 语义：A/B 项消失，C_uplo = β·C_old。
// β==0 走 Zero（与 0*x 不同：NaN/INF 亦归零，保持既有语义），否则走 Scale。
static inline void ApplyBetaOnlyUploComplex(
    aclblasFillMode_t uplo, int n, int ldc, aclblasComplex* C, float beta)
{
    if (beta == 0.0f) {
        ZeroUploTriangleComplex(uplo, n, ldc, C);
    } else {
        ScaleUploTriangleComplex(uplo, n, ldc, C, beta);
    }
}

// Restore the non-uplo triangle to the input-old C values. cblas_cgemm updates the
// whole n×n matrix, but HERKX only writes the uplo triangle — the non-uplo triangle
// must carry the original C values (需求文档 §2.6 区域②). Mirrors ssyrkx
// RestoreNonUploTriangle.
static inline void RestoreNonUploTriangleComplex(
    aclblasFillMode_t uplo, int n, int ldc,
    aclblasComplex* goldenC, const aclblasComplex* originalC)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (!HerkxIsUplo(uplo, i, j)) {
                size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc);
                goldenC[idx] = originalC[idx];
            }
        }
    }
}

// Force the diagonal imaginary part to zero (需求文档 §2.6 区域③ — herkx core).
// herkx has A≠B, so P_i[i][i] = Ai·Br^T[i][i] − Ar·Bi^T[i][i] is generally non-zero
// (unlike herk where anti-symmetry guarantees a zero diagonal imag). Per cuBLAS
// convention and the requirement doc, the diagonal imag must be explicitly cleared.
// This runs on ALL success paths (including short-circuits) regardless of beta.
static inline void ForceZeroDiagonalImag(int n, int ldc, aclblasComplex* C)
{
    for (int i = 0; i < n; i++) {
        size_t idx = static_cast<size_t>(i) + static_cast<size_t>(i) * static_cast<size_t>(ldc);
        C[idx].imag = 0.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU golden for aclblasCherkx (测试方案 §2.1 六步改造).
//   cblas_cherkx does not exist in standard CBLAS/OpenBLAS (HERKX is a cuBLAS
//   extension). The golden is built from cblas_cgemm:
//     1. Validate params (非法 uplo/trans → INVALID_ENUM; n==0 早退先于 ld/指针校验;
//        A/B 条件化非空按 α 复数零判定; C 条件化非空 + β==0&&C==null → SUCCESS 早退).
//     2. Short-circuit α 复数零 / k==0 paths (uplo triangle only, C_uplo = β·C_old).
//     3. cblas_cgemm computes the full matrix product alpha·op(A)·op(B)^H + beta·C.
//        ★ alpha 为复数标量，**直接透传**给 cblas_cgemm（其 void* 形参原生接受
//        complex<float>）；其内部复数乘法与需求文档 §1 的实/虚部交叉缩放展开
//        (Cr = αr·P_r − αi·P_i, Ci = αr·P_i + αi·P_r) 数学等价。
//        beta 为实数标量，按复数接口包装 {betaVal, 0.0f}（虚部 0，cuBLAS 同此）。
//     4. RestoreNonUploTriangleComplex puts the non-uplo triangle back to C_old.
//     5. ForceZeroDiagonalImag clears the diagonal imag on every success path.
//   The extra `originalC` param is a read-only snapshot of the input C used by step 4
//   (same pattern as aclblasSsyrkx_cpu).
//   标量一律按 Host 语义解引用：alpha/beta 的指针位置（Host/Device）是 NPU 侧被测
//   行为，golden 不感知（TEST_P/TEST_F 均以 host 真值调 golden）。
// ─────────────────────────────────────────────────────────────────────────────
inline aclblasStatus_t aclblasCherkx_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc,
    const aclblasComplex* originalC)
{
    int st = ValidateHerkxCpuParams(handle, uplo, trans, n, k, lda, ldb, ldc,
        alpha, beta, A, B, C, originalC);
    if (st != CHERKX_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    const float betaVal = *beta;

    // ── Short-circuit: α 复数零（实部虚部均 0）或 k==0 → A/B 项消失 ──
    if (HerkxAlphaIsComplexZero(alpha) || k == 0) {
        ApplyBetaOnlyUploComplex(uplo, n, ldc, C, betaVal);
        ForceZeroDiagonalImag(n, ldc, C);
        return ACLBLAS_STATUS_SUCCESS;
    }

    // ── Normal path: cblas_cgemm ──
    // trans=N: C = alpha·A·B^H      -> transA=CblasNoTrans,    transB=CblasConjTrans
    // trans=C: C = alpha·A^H·B       -> transA=CblasConjTrans,  transB=CblasNoTrans
    CBLAS_TRANSPOSE cblasTransA = (trans == ACLBLAS_OP_N) ? CblasNoTrans : CblasConjTrans;
    CBLAS_TRANSPOSE cblasTransB = (trans == ACLBLAS_OP_N) ? CblasConjTrans : CblasNoTrans;

    // beta is a real float; cblas_cgemm expects a complex scalar (void* to a 2-float
    // struct) → wrap as {betaVal, 0.0f}. alpha is already complex: pass it straight
    // through (no {val, 0} wrapper — 测试方案 §2.1 步骤 4). aclblasComplex is
    // binary-compatible with OpenBLAS's complex layout, so the casts are reinterprets.
    aclblasComplex betaC{betaVal, 0.0f};

    cblas_cgemm(
        CblasColMajor,
        cblasTransA, cblasTransB,
        n, n, k,
        static_cast<const void*>(alpha),
        static_cast<const void*>(A), lda,
        static_cast<const void*>(B), ldb,
        static_cast<const void*>(&betaC),
        static_cast<void*>(C), ldc);

    // cblas_cgemm updated the whole n×n; restore the non-uplo triangle to C_old.
    RestoreNonUploTriangleComplex(uplo, n, ldc, C, originalC);

    // Force the diagonal imag to zero on every success path (herkx core).
    ForceZeroDiagonalImag(n, ldc, C);

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CHERKX_GOLDEN_H
