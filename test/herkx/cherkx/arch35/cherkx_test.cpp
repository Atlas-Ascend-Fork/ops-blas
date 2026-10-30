/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cherkx_param.h"
#include "cherkx_golden.h"
#include "cherkx_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// 精度判定口径（**唯一取值处 = 测试方案 §2.3 精度判定口径表**；权威源 = 需求文档 §2.6）
//   精度判定以框架 verify.h 为唯一权威（2026-10-29 用户决策移除本地镜像常量与溯源自检，取代原
//   「O5 决议：本地镜像+溯源自检」约定，与其余 6 算子对齐；框架 FLOAT32 混合容差阈值对齐
//   mixed_tolerance_standard.md 2026-09-17 修订版，ops-blas 488f1cc，issue #292）。
//   区域① C uplo 三角（正常路径 α≠0 且 k>0）：混合容差逐元素 |actual−golden| ≤ atol+rtol·|golden|，
//        且 matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2（框架实现为逐元素 max(1e-2, 32×ULP)）
//        → VerifyUploPrecision + applyMixedTolerance(cfg, ACL_FLOAT)（既有保持）
//   区域② C 非 uplo 三角：与输入旧值（originalC 快照）bit-exact → VerifyNonUploExact（cfgNonRe/cfgNonIm）
//   区域③ C 对角线虚部（全部成功用例）：|imag| ≤ 2^-16 → VerifyDiagonalImagZero（既有保持）
//   区域④ C uplo 三角（快路径：α 复数零 或 k==0）：bit-exact → VerifyUploPrecisionExact（本次新增，
//        对齐 ssymm `alpha==0 → PrecisionMode::EXACT` 范式）
//   区域⑤ C ldc padding 区（ldc>n，含 sentinel 预填）：与 host 预填快照 bit-exact →
//        VerifyLdcPaddingExact（本次新增，覆盖「memset 破坏 padding」既有缺陷，AL_21/22 依赖）
//   返回值：aclblasStatus_t 整数相等 → EXPECT_EQ（逐用例 expect_result）+ TEST_F 断言
// 判定实现一律走框架 applyMixedTolerance(ACL_FLOAT) / PrecisionMode::EXACT，不在测试文件内复制
// 框架阈值做一致性对账（2026-10-29 用户决策：去除本地镜像+溯源自检，框架 verify.h 为唯一权威）。
// ═══════════════════════════════════════════════════════════════════════════════
static constexpr double kSpecDiagImagAtol = 1.52587890625e-5;  // 2^-16（区域③真实判定阈值，非框架镜像）

static const char* const kSpecSrcRegion1 =
    "测试方案§2.3区域①←需求§2.6;精度判定以框架verify.h为唯一权威"
    "(2026-10-29决策移除本地镜像自检;框架阈值见mixed_tolerance_standard.md 2026-09-17修订版,issue #292)";
static const char* const kSpecSrcRegion2 = "测试方案§2.3区域②←需求§2.6";
static const char* const kSpecSrcRegion3 = "测试方案§2.3区域③←需求§2.6";
static const char* const kSpecSrcRegion4 = "测试方案§2.3区域④←需求§2.6";
static const char* const kSpecSrcRegion5 = "测试方案§2.3区域⑤←需求§4.2(ldc>n逐列只清n个active行)";
static const char* const kSpecSrcRetCode = "测试方案§2.3返回值行←需求§2.3/§4差距表/O2决议";
// 方案 §2.3「辅助交叉核对」表取值为**无**（O5 决议：不引入第二套统计口径，避免判定歧义），
// 故本文件不自建宽松容差；台账 crosscheck 列如实记录该事实，不存在「两套结论分叉」的可能。
static const char* const kCrossCheckState =
    "NOT_ENABLED(方案§2.3辅助交叉核对=无;O5不引入第二套口径)";

// ═══════════════════════════════════════════════════════════════════════════════
// 权威精度断言台账（PM 口径：每条用例产出「输出张量 / 判定指标 / 实测值 / 阈值 / 结论」并随结果落盘）
//   落盘路径：环境变量 CHERKX_PRECISION_LEDGER，缺省 ./cherkx_precision_ledger.csv（追加模式，
//   多次运行以 run_id 列区分）。台账为**证据留痕**，通过依据仍是同名的 EXPECT/ASSERT 硬断言。
// ═══════════════════════════════════════════════════════════════════════════════
static const char* const kLedgerHeader =
    "run_id,case_name,output_tensor,judgement_metric,measured,threshold,verdict,crosscheck,authority_source,note";

static std::string CherkxLedgerPath()
{
    const char* env = std::getenv("CHERKX_PRECISION_LEDGER");
    return (env != nullptr && *env != '\0') ? std::string(env) : std::string("cherkx_precision_ledger.csv");
}

static const std::string& CherkxRunId()
{
    static const std::string runId = [] {
        const std::time_t now = std::time(nullptr);
        const std::tm tmUtc = *std::gmtime(&now);
        char buf[32] = {0};
        std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tmUtc);
        return std::string(buf) + "_pid" + std::to_string(static_cast<long>(getpid()));
    }();
    return runId;
}

static std::string CherkxQuote(const std::string& field)
{
    if (field.find_first_of(",\"\n") == std::string::npos) {
        return field;
    }
    std::string quoted = "\"";
    for (char c : field) {
        if (c == '"') {
            quoted += "\"\"";
        } else {
            quoted += c;
        }
    }
    return quoted + "\"";
}

static bool CherkxLedgerNeedHeader(const std::string& path)
{
    std::ifstream probe(path, std::ios::binary | std::ios::ate);
    return !(probe.is_open() && probe.tellg() > 0);
}

static std::ofstream& CherkxLedgerStream()
{
    const std::string path = CherkxLedgerPath();
    static const bool needHeader = CherkxLedgerNeedHeader(path);
    static std::ofstream stream(path, std::ios::app);
    static bool headerDone = false;
    if (needHeader && !headerDone) {
        if (stream.is_open()) {
            stream << kLedgerHeader << '\n';
        } else {
            std::cerr << "[LEDGER][WARN] cannot open " << path << " for append; 台账不落盘（断言仍生效）" << std::endl;
        }
        headerDone = true;
    }
    return stream;
}

// 台账单行：输出张量 / 判定指标 / 实测值 / 阈值 / 结论（+ 交叉核对状态 / 权威源出处 / 备注）
static void CherkxLedgerRecord(const std::string& caseName, const std::string& tensor,
    const std::string& metric, const std::string& measured, const std::string& threshold,
    bool pass, const std::string& source, const std::string& note = "")
{
    std::ofstream& stream = CherkxLedgerStream();
    if (!stream.is_open()) {
        return;
    }
    stream << CherkxQuote(CherkxRunId()) << ',' << CherkxQuote(caseName) << ','
           << CherkxQuote(tensor) << ',' << CherkxQuote(metric) << ','
           << CherkxQuote(measured) << ',' << CherkxQuote(threshold) << ','
           << (pass ? "PASS" : "FAIL") << ',' << CherkxQuote(kCrossCheckState) << ','
           << CherkxQuote(source) << ',' << CherkxQuote(note) << '\n';
    stream.flush();
}

static void CherkxLedgerSkip(const std::string& caseName, const std::string& reason)
{
    std::ofstream& stream = CherkxLedgerStream();
    if (!stream.is_open()) {
        return;
    }
    stream << CherkxQuote(CherkxRunId()) << ',' << CherkxQuote(caseName) << ",ALL,skipped,n/a,n/a,"
           << "SKIP," << CherkxQuote(kCrossCheckState) << ",-," << CherkxQuote(reason) << '\n';
    stream.flush();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 实测值度量（台账「实测值」列）：与 verify.h MixedToleranceStrategy / ExactStrategy 同式，
// 仅用于留痕；**通过依据**仍是 Verifier::verifyVector 的硬断言返回值。
// ═══════════════════════════════════════════════════════════════════════════════
struct CherkxRegionMetrics {
    size_t count = 0;
    size_t failCount = 0;
    double matchedRatio = 1.0;
    double maxAbsErr = 0.0;
};

static CherkxRegionMetrics CherkxMeasureMixed(
    const std::vector<float>& out, const std::vector<float>& gold, const VerifyConfig& cfg)
{
    CherkxRegionMetrics m;
    m.count = out.size();
    for (size_t i = 0; i < out.size(); i++) {
        const float o = out[i];
        const float g = gold[i];
        if (o == g || (std::isnan(o) && std::isnan(g))) {
            continue;
        }
        if (std::isnan(o) || std::isnan(g) || std::isinf(o) || std::isinf(g)) {
            m.failCount++;
            continue;
        }
        const double diff = std::abs(static_cast<double>(o) - static_cast<double>(g));
        m.maxAbsErr = std::max(m.maxAbsErr, diff);
        const double goldAbs = std::abs(static_cast<double>(g));
        if (diff > cfg.mixedAtol + cfg.mixedRtol * goldAbs) {
            m.failCount++;
        }
    }
    m.matchedRatio = (m.count > 0) ? static_cast<double>(m.count - m.failCount) / static_cast<double>(m.count) : 1.0;
    return m;
}

static CherkxRegionMetrics CherkxMeasureExact(const std::vector<float>& out, const std::vector<float>& gold)
{
    CherkxRegionMetrics m;
    m.count = out.size();
    for (size_t i = 0; i < out.size(); i++) {
        const bool equal = (out[i] == gold[i]) || (std::isnan(out[i]) && std::isnan(gold[i]));
        if (!equal) {
            m.failCount++;
            if (std::isfinite(out[i]) && std::isfinite(gold[i])) {
                m.maxAbsErr = std::max(m.maxAbsErr,
                    std::abs(static_cast<double>(out[i]) - static_cast<double>(gold[i])));
            }
        }
    }
    m.matchedRatio = (m.count > 0) ? static_cast<double>(m.count - m.failCount) / static_cast<double>(m.count) : 1.0;
    return m;
}

static std::string CherkxFmtMixed(const CherkxRegionMetrics& m)
{
    // %.6f → fixed(6)；%.6e → scientific(6)；%zu → 整型插入（不受浮点格式影响）
    std::ostringstream oss;
    oss << "matched_ratio=" << std::fixed << std::setprecision(6) << m.matchedRatio
        << ";max_abs_err=" << std::scientific << std::setprecision(6) << m.maxAbsErr
        << ";fail=" << m.failCount << "/" << m.count;
    return oss.str();
}

static std::string CherkxFmtExact(const CherkxRegionMetrics& m)
{
    std::ostringstream oss;
    oss << "mismatch=" << m.failCount << "/" << m.count
        << ";max_abs_diff=" << std::scientific << std::setprecision(6) << m.maxAbsErr;
    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Complex uplo-triangle verification helpers（四区域口径 + padding 保护）
//   Reuses cherk's VerifyUploTriangleComplex structure with the herkx core
//   adjustment: the diagonal imaginary-part zero check runs on ALL success cases
//   (herkx force-clears the diagonal imag regardless of beta, 需求 §2.6 区域③), whereas
//   herk only checks it when beta==0 (relying on anti-symmetry).
// ═══════════════════════════════════════════════════════════════════════════════

// Collect uplo / non-uplo element pairs from the NPU output and golden buffers.
// `isUplo` follows the BLAS uplo convention: UPPER ⇒ i <= j, LOWER ⇒ i >= j.
// `goldPtr` doubles as the "input old" reference for the non-uplo triangle because
// the golden restores the non-uplo triangle to originalC via RestoreNonUploTriangleComplex.
template <typename Param>
static inline void CollectUploNonUploElements(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr,
    std::vector<float>& npuUploRe, std::vector<float>& npuUploIm,
    std::vector<float>& goldUploRe, std::vector<float>& goldUploIm,
    std::vector<float>& npuNonRe, std::vector<float>& npuNonIm,
    std::vector<float>& oldNonRe, std::vector<float>& oldNonIm)
{
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                npuUploRe.push_back(cPtr[idx].real);
                npuUploIm.push_back(cPtr[idx].imag);
                goldUploRe.push_back(goldPtr[idx].real);
                goldUploIm.push_back(goldPtr[idx].imag);
            } else {
                npuNonRe.push_back(cPtr[idx].real);
                npuNonIm.push_back(cPtr[idx].imag);
                oldNonRe.push_back(goldPtr[idx].real);
                oldNonIm.push_back(goldPtr[idx].imag);
            }
        }
    }
}

// Collect the ldc padding region (rows n..ldc-1 of each of the n columns) — 区域⑤.
// Reference is the host pre-fill snapshot: padding must never be written nor memset
// （既有缺陷：ldc>n 时逐列只清 n 个 active 行，padding 区须保持预填值）。
template <typename Param>
static inline void CollectPaddingElements(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* prefillPtr,
    std::vector<float>& npuPadRe, std::vector<float>& npuPadIm,
    std::vector<float>& oldPadRe, std::vector<float>& oldPadIm)
{
    if (p.ldc <= p.n) {
        return;
    }
    for (int j = 0; j < p.n; j++) {
        for (int i = p.n; i < p.ldc; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            npuPadRe.push_back(cPtr[idx].real);
            npuPadIm.push_back(cPtr[idx].imag);
            oldPadRe.push_back(prefillPtr[idx].real);
            oldPadIm.push_back(prefillPtr[idx].imag);
        }
    }
}

static const char* const kThresholdMixed =
    "matched_ratio>=0.99;max_abs_err<=max(1e-2,32*ULP);per-elem |d|<=atol+rtol*|gold| (rtol=2^-13,atol=2^-13)";
static const char* const kThresholdExact = "mismatch==0 (bit-exact)";
static const char* const kEmptyRegionNote = "区域为空（count=0），无可判元素，vacuous pass";

// 空区域留痕：区域②/⑤ 在特定 shape 下无元素（n==1 无非 uplo 元素；ldc==n 无 padding），
// 台账仍如实记录该输出张量已被纳入判定，避免「用例未产出该张量结论」的追溯缺口。
static inline void CherkxRecordEmptyRegion(
    const std::string& caseName, const std::string& tensor, const std::string& metric, const char* source)
{
    CherkxLedgerRecord(caseName, tensor, metric, "count=0", kThresholdExact, true, source, kEmptyRegionNote);
}

// 区域①：uplo 三角混合容差（正常路径 α≠0 且 k>0）——real / imag 分别判定。
static inline void VerifyUploPrecision(
    const std::string& caseName,
    const std::vector<float>& npuUploRe, const std::vector<float>& goldUploRe,
    const std::vector<float>& npuUploIm, const std::vector<float>& goldUploIm)
{
    VerifyConfig cfgRe;
    applyMixedTolerance(cfgRe, ACL_FLOAT, goldUploRe.data(), goldUploRe.size());
    const bool passRe = Verifier::verifyVector(npuUploRe.data(), goldUploRe.data(), goldUploRe.size(), 1, cfgRe,
        caseName + "_uplo_real");
    const CherkxRegionMetrics mRe = CherkxMeasureMixed(npuUploRe, goldUploRe, cfgRe);
    CherkxLedgerRecord(caseName, "C_uplo_triangle.real", "mixed_tolerance(atol+rtol*|gold|,matched_ratio,max_abs_err)",
        CherkxFmtMixed(mRe), kThresholdMixed, passRe, kSpecSrcRegion1);
    EXPECT_TRUE(passRe);

    VerifyConfig cfgIm;
    applyMixedTolerance(cfgIm, ACL_FLOAT, goldUploIm.data(), goldUploIm.size());
    const bool passIm = Verifier::verifyVector(npuUploIm.data(), goldUploIm.data(), goldUploIm.size(), 1, cfgIm,
        caseName + "_uplo_imag");
    const CherkxRegionMetrics mIm = CherkxMeasureMixed(npuUploIm, goldUploIm, cfgIm);
    CherkxLedgerRecord(caseName, "C_uplo_triangle.imag", "mixed_tolerance(atol+rtol*|gold|,matched_ratio,max_abs_err)",
        CherkxFmtMixed(mIm), kThresholdMixed, passIm, kSpecSrcRegion1);
    EXPECT_TRUE(passIm);
}

// 区域④：快路径（α 复数零 或 k==0）uplo 三角 bit-exact —— 设备与 golden 均为单次 FP32
// 乘法/拷贝，bit-exact 可判（对齐 ssymm `alpha==0 → EXACT` 范式）。
static inline void VerifyUploPrecisionExact(
    const std::string& caseName,
    const std::vector<float>& npuUploRe, const std::vector<float>& goldUploRe,
    const std::vector<float>& npuUploIm, const std::vector<float>& goldUploIm)
{
    VerifyConfig cfgRe;
    cfgRe.mode = PrecisionMode::EXACT;
    const bool passRe = Verifier::verifyVector(npuUploRe.data(), goldUploRe.data(), goldUploRe.size(), 1, cfgRe,
        caseName + "_uplo_real_fastpath_exact");
    const CherkxRegionMetrics mRe = CherkxMeasureExact(npuUploRe, goldUploRe);
    CherkxLedgerRecord(caseName, "C_uplo_triangle.real(fastpath)", "bit_exact",
        CherkxFmtExact(mRe), kThresholdExact, passRe, kSpecSrcRegion4);
    EXPECT_TRUE(passRe);

    VerifyConfig cfgIm;
    cfgIm.mode = PrecisionMode::EXACT;
    const bool passIm = Verifier::verifyVector(npuUploIm.data(), goldUploIm.data(), goldUploIm.size(), 1, cfgIm,
        caseName + "_uplo_imag_fastpath_exact");
    const CherkxRegionMetrics mIm = CherkxMeasureExact(npuUploIm, goldUploIm);
    CherkxLedgerRecord(caseName, "C_uplo_triangle.imag(fastpath)", "bit_exact",
        CherkxFmtExact(mIm), kThresholdExact, passIm, kSpecSrcRegion4);
    EXPECT_TRUE(passIm);
}

// 区域②：非 uplo 三角必须等于输入旧值（EXACT，验证未被污染）。
static inline void VerifyNonUploExact(
    const std::string& caseName,
    const std::vector<float>& npuNonRe, const std::vector<float>& oldNonRe,
    const std::vector<float>& npuNonIm, const std::vector<float>& oldNonIm)
{
    if (npuNonRe.empty()) {
        CherkxRecordEmptyRegion(caseName, "C_non_uplo_triangle.real+imag", "bit_exact_vs_originalC",
            kSpecSrcRegion2);
        return;
    }
    VerifyConfig cfgNonRe;
    cfgNonRe.mode = PrecisionMode::EXACT;
    const bool passRe = Verifier::verifyVector(npuNonRe.data(), oldNonRe.data(), npuNonRe.size(), 1, cfgNonRe,
        caseName + "_nonuplo_real");
    const CherkxRegionMetrics mRe = CherkxMeasureExact(npuNonRe, oldNonRe);
    CherkxLedgerRecord(caseName, "C_non_uplo_triangle.real", "bit_exact_vs_originalC",
        CherkxFmtExact(mRe), kThresholdExact, passRe, kSpecSrcRegion2);
    EXPECT_TRUE(passRe);

    VerifyConfig cfgNonIm;
    cfgNonIm.mode = PrecisionMode::EXACT;
    const bool passIm = Verifier::verifyVector(npuNonIm.data(), oldNonIm.data(), npuNonIm.size(), 1, cfgNonIm,
        caseName + "_nonuplo_imag");
    const CherkxRegionMetrics mIm = CherkxMeasureExact(npuNonIm, oldNonIm);
    CherkxLedgerRecord(caseName, "C_non_uplo_triangle.imag", "bit_exact_vs_originalC",
        CherkxFmtExact(mIm), kThresholdExact, passIm, kSpecSrcRegion2);
    EXPECT_TRUE(passIm);
}

// 区域⑤：ldc padding 区（ldc>n）必须与 host 预填快照 bit-exact（不得被写/memset 触碰）。
static inline void VerifyLdcPaddingExact(
    const std::string& caseName,
    const std::vector<float>& npuPadRe, const std::vector<float>& oldPadRe,
    const std::vector<float>& npuPadIm, const std::vector<float>& oldPadIm)
{
    if (npuPadRe.empty()) {
        CherkxRecordEmptyRegion(caseName, "C_ldc_padding.real+imag", "bit_exact_vs_host_prefill",
            kSpecSrcRegion5);
        return;
    }
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    const bool passRe = Verifier::verifyVector(npuPadRe.data(), oldPadRe.data(), npuPadRe.size(), 1, cfg,
        caseName + "_ldc_padding_real");
    const CherkxRegionMetrics mRe = CherkxMeasureExact(npuPadRe, oldPadRe);
    CherkxLedgerRecord(caseName, "C_ldc_padding.real", "bit_exact_vs_host_prefill",
        CherkxFmtExact(mRe), kThresholdExact, passRe, kSpecSrcRegion5);
    EXPECT_TRUE(passRe);

    const bool passIm = Verifier::verifyVector(npuPadIm.data(), oldPadIm.data(), npuPadIm.size(), 1, cfg,
        caseName + "_ldc_padding_imag");
    const CherkxRegionMetrics mIm = CherkxMeasureExact(npuPadIm, oldPadIm);
    CherkxLedgerRecord(caseName, "C_ldc_padding.imag", "bit_exact_vs_host_prefill",
        CherkxFmtExact(mIm), kThresholdExact, passIm, kSpecSrcRegion5);
    EXPECT_TRUE(passIm);
}

// 区域③：对角虚部强制清零 —— herkx core difference（全部成功用例无条件执行，无 beta 守卫）。
template <typename Param>
static inline void VerifyDiagonalImagZero(const Param& p, const aclblasComplex* cPtr)
{
    if (p.n <= 0) {
        return;
    }
    double maxDiagImag = 0.0;
    for (int i = 0; i < p.n; i++) {
        size_t idx = static_cast<size_t>(i) + static_cast<size_t>(i) * p.ldc;
        double im = std::abs(static_cast<double>(cPtr[idx].imag));
        if (im > maxDiagImag) {
            maxDiagImag = im;
        }
    }
    // ACL_FLOAT atol = 2^-16 ≈ 1.5259e-5（口径表区域③）
    const bool pass = (maxDiagImag <= kSpecDiagImagAtol);
    std::ostringstream oss;
    oss << "max_abs_imag=" << std::scientific << std::setprecision(6) << maxDiagImag
        << ";n=" << p.n;
    const std::string measured = oss.str();
    CherkxLedgerRecord(p.caseName, "C_diagonal.imag", "max_abs_imag<=atol(强制清零语义)", measured,
        "<=2^-16(1.52587890625e-05)", pass, kSpecSrcRegion3);
    EXPECT_LE(maxDiagImag, kSpecDiagImagAtol)
        << "[" << p.caseName << "] diagonal imag not zero (max=" << maxDiagImag << ")";
}

// 返回值断言台账（口径表「返回值」行：整数相等）
static inline void CherkxRecordReturnCode(
    const std::string& caseName, aclblasStatus_t ret, aclblasStatus_t expect)
{
    // %d → std::to_string（纯整数字段）
    const std::string measured = "ret=" + std::to_string(static_cast<int>(ret));
    const std::string threshold = "expect_result=" + std::to_string(static_cast<int>(expect)) + "(整数相等)";
    CherkxLedgerRecord(caseName, "return_value", "integer_equal", measured, threshold,
        static_cast<int>(ret) == static_cast<int>(expect), kSpecSrcRetCode);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 统一四区域（+padding）校验入口
//   1. uplo 三角：快路径（α 复数零 || k==0）→ 区域④ EXACT；否则 → 区域① 混合容差
//   2. 非 uplo 三角：区域② EXACT（对照 originalC 旧值）
//   3. 对角虚部：区域③ |imag| ≤ 2^-16（全部成功用例）
//   4. ldc padding 区：区域⑤ EXACT（ldc>n 时生效，对照 host 预填快照）
// `goldPtr` already carries the input-old values in its non-uplo triangle because
// the golden restores them via RestoreNonUploTriangleComplex.
// ═══════════════════════════════════════════════════════════════════════════════
struct CherkxHostData {
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> bHost;
    std::vector<aclblasComplex> cHost;
    std::vector<aclblasComplex> originalC; // input-old C snapshot（非 uplo / padding 对照基准）
    std::vector<aclblasComplex> cGolden;   // golden output (starts from input-old C)
    const aclblasComplex* aPtr = nullptr;
    const aclblasComplex* bPtr = nullptr;
    aclblasComplex* cPtr = nullptr;
    aclblasComplex* cGoldenPtr = nullptr;
    const aclblasComplex* originalCPtr = nullptr;
    size_t cCount = 0;
};

// 区域划分自洽守卫：四区域（uplo + 非 uplo + padding）必须恰好覆盖 C 缓冲区（ldc×n），
// 防止「区域集合为空/漏元素却判 PASS」的空判风险；同时让台账的区域结论可追溯到元素总数。
template <typename Param>
static inline void CherkxExpectRegionPartition(
    const Param& p, const CherkxHostData& d, size_t uploCount, size_t nonCount, size_t padCount)
{
    const size_t active = static_cast<size_t>(p.n) * static_cast<size_t>(p.n);
    const size_t padding = (p.ldc > p.n) ? static_cast<size_t>(p.ldc - p.n) * static_cast<size_t>(p.n) : 0;
    const bool ok = (uploCount + nonCount == active) && (padCount == padding) &&
        (active + padding == d.cCount);
    // %zu → std::to_string（纯整数字段）
    const std::string measured = "uplo+nonuplo=" + std::to_string(uploCount + nonCount) +
        ";active(n*n)=" + std::to_string(active) + ";padding=" + std::to_string(padCount) +
        ";pad_expect=" + std::to_string(padding) + ";cCount=" + std::to_string(d.cCount);
    CherkxLedgerRecord(p.caseName, "region_partition_selfcheck", "四区域元素数覆盖 C 缓冲区(ldc*n)",
        measured, "uplo+nonuplo==n*n && padding==(ldc-n)*n && 合计==cCount", ok,
        "测试方案§2.3（四区域口径的划分自洽前提）");
    EXPECT_TRUE(ok) << "[" << p.caseName << "] 区域划分不自洽，四区域口径未完整覆盖 C 缓冲区";
}

template <typename Param>
static inline void VerifyUploTriangleComplex(
    const Param& p, const CherkxHostData& d, bool fastPath)
{
    if (p.n <= 0) {
        return;
    }

    std::vector<float> npuUploRe, npuUploIm, goldUploRe, goldUploIm;
    std::vector<float> npuNonRe, npuNonIm, oldNonRe, oldNonIm;
    CollectUploNonUploElements(p, d.cPtr, d.cGoldenPtr,
        npuUploRe, npuUploIm, goldUploRe, goldUploIm,
        npuNonRe, npuNonIm, oldNonRe, oldNonIm);

    std::vector<float> npuPadRe, npuPadIm, oldPadRe, oldPadIm;
    CollectPaddingElements(p, d.cPtr, d.originalCPtr, npuPadRe, npuPadIm, oldPadRe, oldPadIm);
    CherkxExpectRegionPartition(p, d, npuUploRe.size(), npuNonRe.size(), npuPadRe.size());

    if (fastPath) {
        VerifyUploPrecisionExact(p.caseName, npuUploRe, goldUploRe, npuUploIm, goldUploIm);
    } else {
        VerifyUploPrecision(p.caseName, npuUploRe, goldUploRe, npuUploIm, goldUploIm);
    }

    VerifyNonUploExact(p.caseName, npuNonRe, oldNonRe, npuNonIm, oldNonIm);

    // herkx core: diagonal imag zero check on every success case (no beta guard).
    VerifyDiagonalImagZero(p, d.cPtr);

    VerifyLdcPaddingExact(p.caseName, npuPadRe, oldPadRe, npuPadIm, oldPadIm);
}

// 快路径判定（区域④触发条件）：α 复数零（实部==0 且 虚部==0）或 k==0。
// 复用 golden 的判零式，避免测试侧另立一份判定语义。
static inline bool CherkxIsFastPath(const aclblasComplex& alpha, int k)
{
    return HerkxAlphaIsComplexZero(&alpha) || k == 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class CherkxArch35Test : public BlasTest<CherkxParam> {
};

// ── stream 健康探针（TEST_F 专用，超出 CSV 统一校验能力）──
// 在同 stream 追加 probe 任务（aclrtMemsetAsync 到临时 device buffer）并同步：
// 证明「β==0 && C==nullptr 早退未发 kernel、stream 未被 null GM 写毒化」（S-S2/S-S3）。
static void CherkxExpectStreamHealthy(aclrtStream stream, const std::string& caseName)
{
    constexpr size_t probeBytes = 32;
    void* probe = nullptr;
    const aclError mallocRet = aclrtMalloc(&probe, probeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(mallocRet, ACL_SUCCESS) << "[" << caseName << "] probe buffer alloc failed";
    const aclError memsetRet = aclrtMemsetAsync(probe, probeBytes, 0, probeBytes, stream);
    const aclError syncRet = aclrtSynchronizeStream(stream);
    const bool pass = (memsetRet == ACL_SUCCESS) && (syncRet == ACL_SUCCESS);
    // %d → std::to_string（纯整数字段）
    const std::string measured = "memsetAsync=" + std::to_string(static_cast<int>(memsetRet)) +
        ";syncStream=" + std::to_string(static_cast<int>(syncRet));
    CherkxLedgerRecord(caseName, "stream_health", "probe_aclrtMemsetAsync+aclrtSynchronizeStream",
        measured, "ACL_SUCCESS(0)", pass, "测试方案§5.3 TEST_F 断言要点②（S-S2/S-S3）");
    EXPECT_EQ(memsetRet, ACL_SUCCESS) << "[" << caseName << "] stream 已被毒化（probe memsetAsync 失败）";
    EXPECT_EQ(syncRet, ACL_SUCCESS) << "[" << caseName << "] stream 已被毒化（probe sync 失败）";
    aclrtFree(probe);
}

// ── TEST_F 1/4: null handle（既有保留，S-E1）──
TEST_F(CherkxArch35Test, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const float beta = 0.0f;
    aclblasStatus_t ret = aclblasCherkx_npu(
        nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4,
        &alpha, nullptr, 4, nullptr, 4, &beta, nullptr, 4);
    CherkxRecordReturnCode("TEST_F_NullHandle", ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── TEST_F 2/4: β==0 + C==nullptr（Host 标量）→ SUCCESS 且不发 kernel、stream 健康（S-S3）──
TEST_F(CherkxArch35Test, NullCBetaZero_NoKernelLaunch)
{
    constexpr int dim = 8;
    const std::string caseName = "TEST_F_NullCBetaZero_NoKernelLaunch";
    std::vector<aclblasComplex> aHost = makeBlasComplexMatrix(dim, dim, dim, BlasFillMode("RANDOM_NORM_1"), 20261101);
    std::vector<aclblasComplex> bHost = makeBlasComplexMatrix(dim, dim, dim, BlasFillMode("RANDOM_NORM_1"), 20261102);
    const aclblasComplex alpha{1.0f, 0.0f};
    const float beta = 0.0f;

    aclblasStatus_t ret = aclblasCherkx_npu(
        CherkxArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, dim, dim,
        &alpha, aHost.data(), dim, bHost.data(), dim, &beta, nullptr, dim);
    CherkxRecordReturnCode(caseName, ret, ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    CherkxExpectStreamHealthy(CherkxArch35Test::stream_, caseName);
}

// ── TEST_F 3/4: α/β 双 Device + β==0 + C==nullptr → SUCCESS（host D2H 读回真值后分支，终态语义）──
// α/β 须同置 Device：(H,D) 混合会于解引用前被拒绝 INVALID_VALUE，与期望 SUCCESS 矛盾
// 且到不了 β 读回分支（CP2.1 第 2/3 轮 R1b 修正）。
TEST_F(CherkxArch35Test, DeviceBetaZeroNullC_NoKernelLaunch)
{
    constexpr int dim = 8;
    const std::string caseName = "TEST_F_DeviceBetaZeroNullC_NoKernelLaunch";
    std::vector<aclblasComplex> aHost = makeBlasComplexMatrix(dim, dim, dim, BlasFillMode("RANDOM_NORM_1"), 20261103);
    std::vector<aclblasComplex> bHost = makeBlasComplexMatrix(dim, dim, dim, BlasFillMode("RANDOM_NORM_1"), 20261104);
    const aclblasComplex alpha{1.0f, 0.0f};
    const float beta = 0.0f;

    aclblasStatus_t ret = aclblasCherkx_npu(
        CherkxArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, dim, dim,
        &alpha, aHost.data(), dim, bHost.data(), dim, &beta, nullptr, dim,
        /*nullAlpha=*/false, /*nullBeta=*/false, /*alphaOnDevice=*/true, /*betaOnDevice=*/true);
    CherkxRecordReturnCode(caseName, ret, ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    CherkxExpectStreamHealthy(CherkxArch35Test::stream_, caseName);
}

// ── TEST_F 4/4: α/β 双 Device + α≠0 + A==nullptr → INVALID_VALUE（S-E10，禁「Device 标量猜测语义」的 α 侧反例）──
// host 必须 D2H 读回 α 真值后按 α≠0 校验 A 非空，禁止「Device α + null A 就假设 α==0」的猜测语义。
// β 同置 Device：(D,H) 混合会于解引用前被拒绝，返回码碰巧一致但目标验证点不可达（R1c 修正）。
TEST_F(CherkxArch35Test, DeviceAlphaNonZeroNullA)
{
    constexpr int dim = 8;
    const std::string caseName = "TEST_F_DeviceAlphaNonZeroNullA";
    std::vector<aclblasComplex> bHost = makeBlasComplexMatrix(dim, dim, dim, BlasFillMode("RANDOM_NORM_1"), 20261105);
    std::vector<aclblasComplex> cHost = makeBlasComplexMatrix(dim, dim, dim, BlasFillMode("RANDOM_NORM_1"), 20261106);
    const aclblasComplex alpha{1.0f, 0.0f};
    const float beta = 0.0f;

    aclblasStatus_t ret = aclblasCherkx_npu(
        CherkxArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, dim, dim,
        &alpha, nullptr, dim, bHost.data(), dim, &beta, cHost.data(), dim,
        /*nullAlpha=*/false, /*nullBeta=*/false, /*alphaOnDevice=*/true, /*betaOnDevice=*/true);
    CherkxRecordReturnCode(caseName, ret, ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

INSTANTIATE_TEST_SUITE_P(
    Cherkx, CherkxArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<CherkxParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CherkxParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// Host data preparation
// ═══════════════════════════════════════════════════════════════════════════════
static bool CherkxCheckHostMemBudget(const CherkxParam& p, size_t cBytes, size_t aBytes, size_t bBytes)
{
    // 3 copies of C (npu output + golden + input-old reference) + A + B.
    constexpr size_t kHostMemLimit = 8ULL * 1024ULL * 1024ULL * 1024ULL; // §8 遗留问题 6
    if (3 * cBytes + aBytes + bBytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << (3 * cBytes + aBytes + bBytes) / (1024 * 1024)
                  << " MB) exceeds limit for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }
    return true;
}

static bool PrepareHostData(const CherkxParam& p, CherkxHostData& d)
{
    // n<=0: wrapper fast-paths (n<=0 → passthrough); no host data needed. Return
    // true so the NPU call still runs and the return code is checked (n==0 →
    // SUCCESS, n<0 → INVALID_VALUE). cPtr stays null → no golden/verify.
    if (p.n <= 0) {
        return true;
    }

    // Guard A/B footprint with k>0 (mirrors wrapper): when k<=0 the operator
    // short-circuits (k==0) or rejects (k<0) without accessing A/B, and abCols
    // could be negative for k<0 which would overflow size_t.
    const int abCols = (p.k > 0) ? ((p.trans == ACLBLAS_OP_N) ? p.k : p.n) : 0;

    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(p.ldb) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    if (!CherkxCheckHostMemBudget(p, cBytes, aBytes, bBytes)) {
        return false;
    }

    // Seed strategy (测试方案 §8 遗留问题 3, approach B):
    //   seedA = randomSeed
    //   seedB = aEqB ? randomSeed : randomSeed + 1   (A==B → identical data)
    //   seedC = randomSeed + 2
    const uint32_t seedA = p.randomSeed;
    const uint32_t seedB = p.aEqB ? p.randomSeed : p.randomSeed + 1U;
    const uint32_t seedC = p.randomSeed + 2U;

    try {
        d.aHost = makeBlasComplexMatrix(p.lda, abCols, p.lda, p.aFill, seedA);
        d.bHost = makeBlasComplexMatrix(p.ldb, abCols, p.ldb, p.bFill, seedB);
        d.cHost = makeBlasComplexMatrix(p.n, p.n, p.ldc, p.cFill, seedC);
        d.aPtr = (d.aHost.empty() || p.nullA) ? nullptr : d.aHost.data();
        d.bPtr = (d.bHost.empty() || p.nullB) ? nullptr : d.bHost.data();
        d.cPtr = (d.cHost.empty() || p.nullC) ? nullptr : d.cHost.data();
        d.cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);
        if (d.cPtr == nullptr) {
            return true;
        }
        d.originalC.assign(d.cHost.begin(), d.cHost.end());
        d.cGolden.assign(d.cHost.begin(), d.cHost.end());
        d.cGoldenPtr = d.cGolden.data();
        d.originalCPtr = d.originalC.data();
    } catch (const std::bad_alloc&) {
        std::cout << "[SKIP] host memory allocation failed for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision（四区域口径 + padding 保护）
// ═══════════════════════════════════════════════════════════════════════════════
TEST_P(CherkxArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    CherkxHostData d;
    if (!PrepareHostData(p, d)) {
        CherkxLedgerSkip(p.caseName, "host memory limit");
        GTEST_SKIP() << "Skipped: host memory limit";
    }

    // 复数 alpha 由 CSV 的 alpha_real / alpha_imag 两列构造（§2.1 步骤 6）；
    // nullAlpha/nullBeta 由 wrapper 转发 nullptr；alphaOnDevice/betaOnDevice 透传指针模式。
    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    aclblasStatus_t ret = aclblasCherkx_npu(
        CherkxArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        &alpha, d.aPtr, p.lda, d.bPtr, p.ldb, &p.beta, d.cPtr, p.ldc,
        p.nullAlpha, p.nullBeta, p.alphaOnDevice, p.betaOnDevice);

    CherkxRecordReturnCode(p.caseName, ret, p.expectResult);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || d.cPtr == nullptr) {
        return;
    }

    // Golden: alpha/beta are always valid here (nullAlpha/nullBeta cases have
    // expectResult=INVALID_VALUE and returned above). originalC feeds the restore.
    // golden 一律按 Host 语义解引用标量真值（指针位置是 NPU 侧被测行为，golden 不感知）。
    aclblasStatus_t goldenRet = aclblasCherkx_cpu(
        CherkxArch35Test::handle_, p.uplo, p.trans, p.n, p.k,
        &alpha, d.aPtr, p.lda, d.bPtr, p.ldb, &p.beta,
        d.cGoldenPtr, p.ldc, d.originalCPtr);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    VerifyUploTriangleComplex(p, d, CherkxIsFastPath(alpha, p.k));
}
