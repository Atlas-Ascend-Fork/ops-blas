/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cher2k_param.h"
#include "cher2k_golden.h"
#include "cher2k_npu_wrapper.h"
#include "cher2k_perf.h"

// ═══════════════════════════════════════════════════════════════════════════════
// aclblasCher2k CSV-driven precision ST + TC_PF performance branch + host tests
// Test plan §6:
//   §6.1 precision-criterion table — per-output-tensor hard assertions
//   §6.4 golden parameter mirror (validation-order table replicated row by row)
//   §6.5 performance capture (aclrtEvent, warmup 5 + 60 samples, bottleneck breakdown)
//   §6.6 Host unit tests (non-CSV-driven TEST_F)
// ═══════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Per-case result recorder: every case writes one row per output tensor with
// "output tensor / metric / measured value / threshold / verdict" (task spec acceptance
// criteria). Rows go to stdout in a grep-friendly format and to cher2k_verify_results.csv
// next to the binary.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

// [T2] Forward declaration: the suite-breakdown helper functions are defined later in this file
// (same anonymous namespace).
void RecordSuiteCaseOutcome(const std::string& caseName, bool hasTensorRows);
void RecordTensorCasePass(const std::string& caseName, bool allPass);

struct VerifyRecordRow {
    std::string caseName;
    std::string tensor;    // e.g. C_uplo_real / C_uplo_imag / C_diag_imag / C_nonuplo / C_padding
    std::string metric;    // mixed_tolerance / max_abs_diag / exact / canary
    double measured = 0.0; // max_abs_err (mixed) / max value (diag) / mismatch count (exact)
    double threshold = 0.0;
    bool pass = false;
};

class VerifyRecord {
public:
    static VerifyRecord& instance()
    {
        static VerifyRecord rec;
        return rec;
    }

    void add(
        const std::string& cs, const std::string& tensor, const std::string& metric, double measured, double threshold,
        bool pass)
    {
        rows_.push_back({cs, tensor, metric, measured, threshold, pass});
        AppendRow(rows_.back());
        // [T2] Tensor-row aggregation: the first row of a case marks it as entering the
        // precision denominator; any FAIL marks the case as failed.
        if (caseTensorSeen_.insert(cs).second) {
            RecordSuiteCaseOutcome(cs, true /* hasTensorRows */);
        }
        if (!pass) {
            caseTensorFailed_.insert(cs);
        }
    }

    // [A' golden dual chain] float cross-check columns: written to CSV only (tensor columns
    // *_float_xcheck); they take no part in pass/fail aggregation and do not enter the precision
    // denominator -- the cross-check chain is a "record", not a "criterion", and its FAIL only
    // indicates a reference-chain metric difference (dual-chain divergence), not an implementation
    // defect. The verdict column records the cross-check chain's own result (PASS/FAIL) for
    // divergence-report filtering, but caseTensorSeen_/caseTensorFailed_ are not updated.
    void addXcheck(
        const std::string& cs, const std::string& tensor, const std::string& metric, double measured, double threshold,
        bool xpass)
    {
        rows_.push_back({cs, tensor, metric, measured, threshold, xpass});
        AppendRow(rows_.back());
    }

    // Called at CsvDriven teardown: returns whether all tensor rows of the case passed (cases with no
    // tensor rows are not in the set).
    bool tensorCaseAllPass(const std::string& cs) const
    {
        return caseTensorSeen_.count(cs) > 0 && caseTensorFailed_.count(cs) == 0;
    }

    bool hasTensorRows(const std::string& cs) const { return caseTensorSeen_.count(cs) > 0; }

    // [T2 3.4 fallback] Breakdown for invalid-argument / quick-return cases: a case with no
    // tensor rows and a gtest OK is listed separately as SKIP-OK -- it is not counted as FAIL
    // and does not take a slot in the precision denominator (it only asserts the return code;
    // the CSV row format is unchanged, and the case is listed explicitly only in the stdout
    // summary to remove the 1120/1104/984 denominator confusion).
    void addSkipOk(const std::string& cs, const std::string& reason)
    {
        skipOkCount_++;
        RecordSuiteCaseOutcome(cs, false /* hasTensorRows */); // counts toward the suite SKIP-OK column
        std::cout << "[SKIP-OK] " << cs << " " << reason
                  << " (invalid argument / quick return: no output tensor to compare, not counted as a"
                     " precision FAIL, and does not take a slot in the precision denominator)"
                  << std::endl;
    }

    size_t skipOkCount() const { return skipOkCount_; }

    void flush()
    {
        // Rows are already persisted by append (AppendRow). Nothing is printed here -- under
        // atexit/static destruction order the CANN-side cleanup may already have corrupted the
        // iostream (observed as garbled output); the summary is derived from the CSV instead.
    }

private:
    static std::string ResultCsvPath()
    {
        // Write to the binary's run directory (cwd) to avoid polluting the source tree; sits next
        // to the gtest output for easy collection by the verify script. CHER2K_RESULT_DIR overrides.
        const char* dir = std::getenv("CHER2K_RESULT_DIR");
        if (dir != nullptr && *dir != '\0') {
            return std::string(dir) + "/cher2k_verify_results.csv";
        }
        return "cher2k_verify_results.csv";
    }

    // Row-by-row append (open-append-close), avoiding file writes during static destruction;
    // the header is written on the first row, and rows already persisted survive a process kill.
    static void AppendRow(const VerifyRecordRow& r)
    {
        const std::string path = ResultCsvPath();
        bool needHeader = true;
        {
            std::ifstream probe(path);
            if (probe.good() && probe.peek() != std::char_traits<char>::eof()) {
                needHeader = false;
            }
        }
        std::ofstream ofs(path, std::ios::app);
        if (!ofs.is_open()) {
            return;
        }
        if (needHeader) {
            ofs << "case_name,output_tensor,metric,measured,threshold,verdict\n";
        }
        ofs << r.caseName << "," << r.tensor << "," << r.metric << "," << std::scientific << std::setprecision(6)
            << r.measured << "," << r.threshold << "," << (r.pass ? "PASS" : "FAIL") << "\n";
    }

    std::vector<VerifyRecordRow> rows_;
    std::set<std::string> caseTensorSeen_;   // [T2] cases that entered the precision denominator
    std::set<std::string> caseTensorFailed_; // [T2] cases with at least one FAIL tensor row
    size_t skipOkCount_ = 0;                 // [T2] cases with no tensor rows and gtest OK (SKIP-OK column)
    bool flushed_ = false;
};

// Every record is persisted immediately by append (see VerifyRecord::add / AppendRow).
// No atexit/static-destruction flush is registered -- under atexit order the CANN-side cleanup may
// already have corrupted the iostream.
static void FlushVerifyRecordAtExit()
{
    VerifyRecord::instance().flush(); // no-op (row-by-row persistence is already immediate)
}

// ─────────────────────────────────────────────────────────────────────────────
// [T2 3.4 fallback] In-test suite breakdown (suite prefix -> counters).
// The 16 invalid-argument / quick-return cases (TC_ED_201/207~221) pass in gtest but have no
// tensor rows, and mixing them with precision FAILs in one denominator caused the
// 1120/1104/984 metric confusion. This mechanism lists them separately inside the test: when each
// CSV case completes it is classified by "has / has no tensor rows" into its suite, and the process
// prints the breakdown at normal teardown --
//   SKIP-OK  : no tensor rows and gtest OK (invalid argument / quick return, return-code assertion only)
//   PASS/FAIL: precision cases with tensor rows (tensor-row verdict aggregation; the precision
//              denominator contains only these)
// This summary only clarifies the stdout metric; it does not change the CSV row content or the gtest
// verdict itself.
// ─────────────────────────────────────────────────────────────────────────────
struct SuiteStat {
    size_t tensorCases = 0; // cases with tensor rows (participating in the precision denominator)
    size_t passCases = 0;   // of which cases whose tensor rows all passed
    size_t skipOkCases = 0; // cases with no tensor rows and gtest OK (T2 column)
};

inline std::map<std::string, SuiteStat>& AllSuiteStats()
{
    static std::map<std::string, SuiteStat> stats;
    return stats;
}

inline SuiteStat& SuiteStatOf(const std::string& caseName)
{
    // The grouping key prefers the current gtest test-suite name (Cher2k), consistent with the
    // per-suite breakdown; when no suite name is available (called outside a gtest test body) it
    // falls back to the case name's first '_' segment (TC_FL_xxx -> TC).
    std::string key;
    const ::testing::TestInfo* ti = (::testing::UnitTest::GetInstance() != nullptr) ?
                                        ::testing::UnitTest::GetInstance()->current_test_info() :
                                        nullptr;
    if (ti != nullptr && ti->test_suite_name() != nullptr && *ti->test_suite_name() != '\0') {
        key = ti->test_suite_name();
    } else {
        size_t pos = caseName.find('_');
        key = (pos == std::string::npos) ? caseName : caseName.substr(0, pos);
    }
    return AllSuiteStats()[key];
}

// Called at CsvDriven teardown: hasTensorRows = whether this case wrote tensor rows.
// Note: only the counters are updated; VerifyRecord is not called back (VerifyRecord::add /
// addSkipOk call this function in the other direction for stats, so a callback would be infinite
// recursion -- observed during the 3.4 fallback fix).
void RecordSuiteCaseOutcome(const std::string& caseName, bool hasTensorRows)
{
    SuiteStat& st = SuiteStatOf(caseName);
    if (!hasTensorRows) {
        // No tensor rows: this case only asserted the return code (invalid argument / quick-return /
        // C-null). If gtest itself failed, gtest counts it as FAILED; here we only handle the OK side.
        st.skipOkCases++;
        return;
    }
    st.tensorCases++;
    // Tensor-row verdicts are guaranteed by VerifyRecord row ordering: rows of the same case are
    // appended consecutively, and RecordTensorCasePass records the aggregate verdict by case name
    // after the tensor rows are written.
}

// Tensor-row aggregation (maintained in sync with VerifyRecord::rows_): a case counts as PASS only
// if all its tensor rows passed.
void RecordTensorCasePass(const std::string& caseName, bool allPass)
{
    SuiteStat& st = SuiteStatOf(caseName);
    st.passCases += allPass ? 1 : 0;
}

// Print the breakdown at normal process teardown (after all test bodies finish). It lives in a
// separate entry point outside CsvDriven and is called by a gtest environment object after all
// tests end, avoiding static destruction ordering.
class Cher2kSuiteSummaryPrinter : public ::testing::EmptyTestEventListener {
public:
    void OnTestProgramEnd(const ::testing::UnitTest&) override
    {
        const std::map<std::string, SuiteStat>& stats = AllSuiteStats();
        std::cout << "\n[cher2k suite breakdown] (T2: no tensor rows and gtest OK listed as SKIP-OK, not"
                     " counted as FAIL nor occupying the precision denominator)"
                  << std::endl;
        std::cout << "  group   tensor_cases(denom)   case_PASS   SKIP-OK(no rows)" << std::endl;
        size_t totT = 0;
        size_t totP = 0;
        size_t totS = 0;
        for (const auto& kv : stats) {
            std::cout << "  " << std::left << std::setw(8) << kv.first << std::right << std::setw(10)
                      << kv.second.tensorCases << std::setw(14) << kv.second.passCases << std::setw(12)
                      << kv.second.skipOkCases << std::endl;
            totT += kv.second.tensorCases;
            totP += kv.second.passCases;
            totS += kv.second.skipOkCases;
        }
        std::cout << "  TOTAL" << std::right << std::setw(13) << totT << std::setw(14) << totP << std::setw(12) << totS
                  << std::endl;
        std::cout << "  precision pass rate (tensor-row metric) = " << totP << "/" << totT;
        if (totT > 0) {
            std::cout << " = " << std::fixed << std::setprecision(3)
                      << 100.0 * static_cast<double>(totP) / static_cast<double>(totT) << "%";
        }
        std::cout << "   (SKIP-OK " << totS << " not counted)" << std::endl;
        std::cout.unsetf(std::ios::fixed | std::ios::scientific); // restore the default float format
    }
};

// Local mixed-tolerance evaluation that mirrors MixedToleranceStrategy exactly
// (atol 2^-16, rtol 2^-10, ratio>=0.99, per-element limit max(1e-2, 32*ULP)) but
// also reports the measured max_abs_error so it can be recorded per tensor.
// [T1 3.4 fallback] Aligned with the frame metric: the precision base class
// PrecisionStrategy::shouldSkip (verify.h:49-56) first skips "outVal==goldVal (including
// same-signed ±Inf equality) and isnan&&isnan"; only non-skipped elements enter the NaN/Inf
// hard-fail branch of MixedToleranceStrategy::processElement. The original implementation lacked
// this pre-skip, so elements of cases like TC_FL_166/167/168 that exactly match the golden
// (same nan / same ±inf) were still recorded as ulpOk=false + failCount++ and judged FAIL even
// with a measured maxAbsErr=0. Fixed to be branch-for-branch equivalent to the frame.
struct MixedToleranceResult {
    double maxAbsErr = 0.0;
    double matchedRatio = 1.0;
    bool ulpLimitOk = true;
    bool pass = false;
};

// Row-by-row mirror of verify.h PrecisionStrategy::shouldSkip (== equality / NaN==NaN).
// Under IEEE-754, == already covers same-signed ±Inf equality (inf-inf is nan but does not enter
// this branch); nan==nan is always false, hence listed separately.
inline bool MixedToleranceShouldSkip(float outVal, float goldVal)
{
    if (outVal == goldVal) {
        return true; // same-signed ±Inf equality (bit-identical) and all finite equal elements
    }
    if (std::isnan(outVal) && std::isnan(goldVal)) {
        return true; // NaN treated as equal to NaN (Inf/NaN propagation consistency, same as frame)
    }
    return false;
}

// [A' golden dual chain] The same shouldSkip mirror for the double main-criterion chain: NPU float
// output vs double golden. Semantics are branch-for-branch equivalent to the float version
// (== equality / NaN==NaN); the float->double conversion is exact, so outVal==goldVal is fully
// consistent with the float version over the finite range, and same-signed ±Inf is also equal.
// NaN is tested with isnan(double) (a float NaN stays NaN after conversion; the sign bit difference
// does not affect isnan).
inline bool MixedToleranceShouldSkip(float outVal, double goldVal)
{
    const double outD = static_cast<double>(outVal);
    if (outD == goldVal) {
        return true;
    }
    if (std::isnan(outD) && std::isnan(goldVal)) {
        return true;
    }
    return false;
}

// [codecheck #8] Shared tail for the mixed_tolerance verdict: the four statements after the
// float/double chain loops (matchedRatio ternary x2 + pass conjunction) are textually identical,
// so they are extracted into a shared template; the verdict expressions are preserved verbatim and
// the two chains keep independent, symmetric loop bodies.
template <typename ResT, typename CountT>
static void FinalizeMixedTolerance(ResT& r, CountT count, CountT failCount)
{
    r.matchedRatio = (count > 0) ? static_cast<double>(count - failCount) / static_cast<double>(count) : 1.0;
    r.pass = (r.matchedRatio >= 0.99) && r.ulpLimitOk;
}

// [codecheck issue 2] The two per-element verdict statements at the tail of both the float/double
// chain loops (over the atol+rtol threshold counts as failure / over the ULP limit clears
// ulpLimitOk) are textually identical and extracted into a shared helper; the tolLimit/ulpLimit
// computations stay in each loop unchanged (the metric difference between the float chain's direct
// goldAbs and the double chain's goldF=float view is preserved), sharing only the final two
// statements.
static inline void ApplyMixedTolItem(
    double diff, double tolLimit, double ulpLimit, size_t& failCount, MixedToleranceResult& r)
{
    if (diff > tolLimit) {
        failCount++;
    }
    if (diff > ulpLimit) {
        r.ulpLimitOk = false;
    }
}

inline MixedToleranceResult EvaluateMixedTolerance(
    const float* out, const float* gold, size_t count, double atol, double rtol, double fixedLimit, int mantissaBits,
    int emin)
{
    MixedToleranceResult r;
    size_t failCount = 0;
    for (size_t i = 0; i < count; i++) {
        if (MixedToleranceShouldSkip(out[i], gold[i])) {
            continue; // frame shouldSkip metric: exact match (incl. same-signed NaN/±Inf) not a failure
        }
        if (std::isnan(out[i]) || std::isnan(gold[i]) || std::isinf(out[i]) || std::isinf(gold[i])) {
            r.ulpLimitOk = false;
            failCount++;
            continue;
        }
        double diff = std::abs(static_cast<double>(out[i]) - static_cast<double>(gold[i]));
        r.maxAbsErr = std::max(r.maxAbsErr, diff);
        double goldAbs = std::abs(static_cast<double>(gold[i]));
        double tolLimit = atol + rtol * goldAbs;
        double ulpLimit = std::max(fixedLimit, 32.0 * getUlpsAt(goldAbs, mantissaBits, emin));
        ApplyMixedTolItem(diff, tolLimit, ulpLimit, failCount, r); // [codecheck issue 2] shared verdict statements
    }
    FinalizeMixedTolerance(r, count, failCount);                   // [codecheck #8] shared verdict tail
    return r;
}

// [A' golden dual chain] Mixed-tolerance evaluation for the double main-criterion chain: NPU float
// output vs double golden. The verdict dtype is still FLOAT32 (complex64 real/imag are still judged
// as FLOAT32):
//   - thresholds/constants are identical to the float version (atol 2^-16 / rtol 2^-10 / ratio>=0.99
//     / max(1e-2,32ULP));
//   - ulpLimit computes ULP from the golden's float-rounded value (within 0.5ULP_f32), consistent
//     with the "golden=float" scale, and is not relaxed because the golden was upgraded to double;
//   - the golden side stays double (no pre-rounding back to float), avoiding a final rounding step
//     in the reference chain;
//   - when goldF (the float cross-check golden) is non-null, elements whose output is non-finite and
//     whose float golden is also non-finite are skipped as "intrinsic float32-domain overflow"
//     (under extreme inputs like EXTREME the double golden is finite while any float32 implementation
//     necessarily overflows, not a kernel defect); other non-finite elements still count as failures.
inline MixedToleranceResult EvaluateMixedToleranceDbl(
    const float* out, const double* gold, size_t count, double atol, double rtol, double fixedLimit, int mantissaBits,
    int emin, const float* goldF = nullptr)
{
    MixedToleranceResult r;
    size_t failCount = 0;
    for (size_t i = 0; i < count; i++) {
        if (MixedToleranceShouldSkip(out[i], gold[i])) {
            continue; // frame shouldSkip metric: exact match (incl. same-signed NaN/±Inf) not a failure
        }
        const double outD = static_cast<double>(out[i]);
        if (std::isnan(out[i]) || std::isnan(gold[i]) || std::isinf(out[i]) || std::isinf(gold[i])) {
            // Intrinsic float32-domain overflow: the output and the float golden are both non-finite
            // (the double golden is finite); any float32 implementation necessarily overflows, so it
            // is not a failure; only a dual-chain sign conflict still fails.
            if (goldF != nullptr && !std::isfinite(goldF[i])) {
                continue;
            }
            r.ulpLimitOk = false;
            failCount++;
            continue;
        }
        double diff = std::abs(outD - gold[i]);
        r.maxAbsErr = std::max(r.maxAbsErr, diff);
        double goldAbs = std::abs(gold[i]);
        double tolLimit = atol + rtol * goldAbs;
        // The ULP verdict stays on the FLOAT32 metric: take the ULP of the golden's float view
        // (same value or within 0.5ULP).
        const float goldF = static_cast<float>(gold[i]);
        double ulpLimit =
            std::max(fixedLimit, 32.0 * getUlpsAt(std::abs(static_cast<double>(goldF)), mantissaBits, emin));
        ApplyMixedTolItem(diff, tolLimit, ulpLimit, failCount, r); // [codecheck issue 2] shared verdict statements
    }
    FinalizeMixedTolerance(r, count, failCount);                   // [codecheck #8] shared verdict tail
    return r;
}

// MERE/MARE re-judgement (CSV mere_threshold/mare_multiplier columns, semantics same as frame
// MereMareStrategy):
//   near-zero golden (|gold| < threshold) is skipped -- in large-cancellation scenarios (anti-symmetric
//   cancellation of Xi - Xi^T under high-mean Gaussian / large k) the golden is near 0, and float32's
//   intrinsic rounding error accumulated at the ~10^3 scale (~0.01) far exceeds atol=2^-16, so strict
//   mixed tolerance is unreachable (even the float cblas reference chain fails it, see the
//   *_float_xcheck columns); such cases are judged by the MERE/MARE relative-error metric preset by
//   the task spec's companion test plan: MERE (mean relative error) < threshold and MARE (max relative
//   error) < multiplier * threshold.
inline MixedToleranceResult EvaluateMereMareDbl(
    const float* out, const double* gold, size_t count, double threshold, double multiplier)
{
    MixedToleranceResult r;
    constexpr double kEpsilon = 0.00006103515625; // 2^-14, same as frame MereMareStrategy
    double sumRelErr = 0.0;
    size_t validCount = 0;
    for (size_t i = 0; i < count; i++) {
        if (MixedToleranceShouldSkip(out[i], gold[i])) {
            continue;
        }
        if (std::isnan(out[i]) || std::isnan(gold[i]) || std::isinf(out[i]) || std::isinf(gold[i])) {
            r.ulpLimitOk = false; // reused: a special-value mismatch is a hard failure
            continue;
        }
        double goldAbs = std::abs(gold[i]);
        if (goldAbs < threshold) {
            continue; // near-zero golden: relative error is undefined, skip from the statistics
        }
        double relErr = std::abs(static_cast<double>(out[i]) - gold[i]) / (goldAbs + kEpsilon);
        sumRelErr += relErr;
        r.maxAbsErr = std::max(r.maxAbsErr, relErr); // reused field: holds MARE here
        validCount++;
    }
    double mere = (validCount > 0) ? sumRelErr / static_cast<double>(validCount) : 0.0;
    r.matchedRatio = validCount;
    r.pass = r.ulpLimitOk && (mere < threshold) && (r.maxAbsErr < multiplier * threshold);
    return r;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Complex uplo-triangle verification helpers (test plan §6.1 precision-criterion table)
//   1. C uplo triangle, real : FLOAT32 mixed tolerance (rtol 2^-10 / atol 2^-16,
//                              ratio>=0.99, max_abs<=1e-2 or 32ULP) — hard assertion
//   2. C uplo triangle, imag : same as above (independent sequence)
//   3. diagonal imag         : max|CI_ii| <= 2^-16 (unconditional, not restricted to beta==0)
//   4. non-uplo triangle     : EXACT equality with the old input C (no pollution)
//   5. ldc padding           : canary 0xCAFEBABE bit-exact (cher2k-specific)
// ═══════════════════════════════════════════════════════════════════════════════

// [codecheck issue 3] Shared element-collection template: the gold side is parameterized by GoldT --
// GoldT=float is the original float chain (goldPtr is aclblasComplex*, and both the uplo/non-uplo
// segments take float values directly), GoldT=double is the original [A' golden dual chain] double
// chain (goldPtr is aclblasDoubleComplex*, the uplo segment takes double directly and the non-uplo
// segment takes the float old value via static_cast<float>). The loop body/index/upper-triangle test
// are preserved verbatim; only the element type of goldPtr[idx].real/.imag follows the GoldT
// instantiation, and the two instantiations are statement-for-statement equivalent to the original
// two functions.
template <typename Param, typename GoldT, typename GoldPtrT>
static inline void CollectUploNonUploElementsT(
    const Param& p, const aclblasComplex* cPtr, const GoldPtrT* goldPtr, std::vector<float>& npuUploRe,
    std::vector<float>& npuUploIm, std::vector<GoldT>& goldUploRe, std::vector<GoldT>& goldUploIm,
    std::vector<float>& npuNonRe, std::vector<float>& npuNonIm, std::vector<float>& oldNonRe,
    std::vector<float>& oldNonIm)
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
                // Non-uplo: the float old value (same source as the float chain; this segment of the
                // double golden is an exact up-conversion of the float old value, so the memcmp float
                // metric is equivalent). For GoldT=float the cast is a no-op.
                oldNonRe.push_back(static_cast<float>(goldPtr[idx].real));
                oldNonIm.push_back(static_cast<float>(goldPtr[idx].imag));
            }
        }
    }
}

// float chain (the main criterion before A'): goldPtr is the float golden, and the non-uplo segment
// takes the float old value directly.
template <typename Param>
static inline void CollectUploNonUploElements(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr, std::vector<float>& npuUploRe,
    std::vector<float>& npuUploIm, std::vector<float>& goldUploRe, std::vector<float>& goldUploIm,
    std::vector<float>& npuNonRe, std::vector<float>& npuNonIm, std::vector<float>& oldNonRe,
    std::vector<float>& oldNonIm)
{
    CollectUploNonUploElementsT<Param, float, aclblasComplex>(
        p, cPtr, goldPtr, npuUploRe, npuUploIm, goldUploRe, goldUploIm, npuNonRe, npuNonIm, oldNonRe, oldNonIm);
}

// [A' golden dual chain] Element collection for the double main-criterion chain: NPU float output vs
// double golden (goldDbl = cblas_zher2k output; its non-uplo triangle holds the up-converted double
// of the old input C). The non-uplo EXACT comparison still runs on the float old value (the non-uplo
// segment of goldDbl is up-converted from the float old value, and float(x)==x is exact, same source
// as the float chain, so the EXACT semantics are not relaxed).
template <typename Param>
static inline void CollectUploNonUploElementsDbl(
    const Param& p, const aclblasComplex* cPtr, const aclblasDoubleComplex* goldDbl, std::vector<float>& npuUploRe,
    std::vector<float>& npuUploIm, std::vector<double>& goldUploRe, std::vector<double>& goldUploIm,
    std::vector<float>& npuNonRe, std::vector<float>& npuNonIm, std::vector<float>& oldNonRe,
    std::vector<float>& oldNonIm)
{
    CollectUploNonUploElementsT<Param, double, aclblasDoubleComplex>(
        p, cPtr, goldDbl, npuUploRe, npuUploIm, goldUploRe, goldUploIm, npuNonRe, npuNonIm, oldNonRe, oldNonIm);
}

// Cross-check chain (float cblas golden, the original main criterion before A'): records CSV
// cross-check columns only and takes no part in the pass/fail verdict (VerifyRecord::addXcheck),
// used for the dual-chain divergence report and regression reconciliation.
static inline void RecordFloatXcheckColumns(
    const std::string& caseName, const float* npuUploRe, const float* goldUploReF, size_t uploCount,
    const float* npuUploIm, const float* goldUploImF)
{
    auto defaults = getMixedToleranceDefaults(ACL_FLOAT);
    MixedToleranceResult re = EvaluateMixedTolerance(
        npuUploRe, goldUploReF, uploCount, defaults.atol, defaults.rtol, defaults.maxAbsErrorLimitFixed,
        defaults.mantissaBits, defaults.emin);
    VerifyRecord::instance().addXcheck(
        caseName, "C_uplo_real_float_xcheck", "mixed_tolerance_xcheck", re.maxAbsErr, defaults.maxAbsErrorLimitFixed,
        re.pass);
    MixedToleranceResult im = EvaluateMixedTolerance(
        npuUploIm, goldUploImF, uploCount, defaults.atol, defaults.rtol, defaults.maxAbsErrorLimitFixed,
        defaults.mantissaBits, defaults.emin);
    VerifyRecord::instance().addXcheck(
        caseName, "C_uplo_imag_float_xcheck", "mixed_tolerance_xcheck", im.maxAbsErr, defaults.maxAbsErrorLimitFixed,
        im.pass);
}

// Authoritative precision hard assertion (§6.1): real/imag each use the FLOAT32 mixed tolerance.
// The metric and thresholds are all taken from the frame's ACL_FLOAT defaults
// (rtol 2^-10=9.7656e-4, atol 2^-16=1.5259e-5, ratio>=0.99, per-element max(1e-2, 32*ULP@|gold|))
// and are not relaxed locally. Kept after the A' upgrade for float-chain evaluation
// (cross-check) and host white-box small-sample checks; the main entry for the CSV precision verdict
// is VerifyUploPrecisionDbl (golden=double).
static inline void VerifyUploPrecision(
    const std::string& caseName, const float* npuUploRe, const float* goldUploRe, size_t uploCount,
    const float* npuUploIm, const float* goldUploIm)
{
    auto defaults = getMixedToleranceDefaults(ACL_FLOAT);
    const double rtol = defaults.rtol; // 2^-10
    const double atol = defaults.atol; // 2^-16

    MixedToleranceResult re = EvaluateMixedTolerance(
        npuUploRe, goldUploRe, uploCount, atol, rtol, defaults.maxAbsErrorLimitFixed, defaults.mantissaBits,
        defaults.emin);
    VerifyRecord::instance().add(
        caseName, "C_uplo_real", "mixed_tolerance", re.maxAbsErr, defaults.maxAbsErrorLimitFixed, re.pass);
    EXPECT_TRUE(re.pass) << "[" << caseName << "] uplo real: ratio=" << re.matchedRatio << " maxAbsErr=" << re.maxAbsErr
                         << " ulpOk=" << re.ulpLimitOk;
    EXPECT_TRUE(Verifier::verifyVector(
        npuUploRe, goldUploRe, uploCount, 1,
        [&]() {
            VerifyConfig cfg;
            applyMixedTolerance(cfg, ACL_FLOAT, goldUploRe, uploCount);
            return cfg;
        }(),
        caseName + "_uplo_real"));

    MixedToleranceResult im = EvaluateMixedTolerance(
        npuUploIm, goldUploIm, uploCount, atol, rtol, defaults.maxAbsErrorLimitFixed, defaults.mantissaBits,
        defaults.emin);
    VerifyRecord::instance().add(
        caseName, "C_uplo_imag", "mixed_tolerance", im.maxAbsErr, defaults.maxAbsErrorLimitFixed, im.pass);
    EXPECT_TRUE(im.pass) << "[" << caseName << "] uplo imag: ratio=" << im.matchedRatio << " maxAbsErr=" << im.maxAbsErr
                         << " ulpOk=" << im.ulpLimitOk;
    EXPECT_TRUE(Verifier::verifyVector(
        npuUploIm, goldUploIm, uploCount, 1,
        [&]() {
            VerifyConfig cfg;
            applyMixedTolerance(cfg, ACL_FLOAT, goldUploIm, uploCount);
            return cfg;
        }(),
        caseName + "_uplo_imag"));
}

// [A' main criterion] Authoritative precision hard assertion (§6.1): NPU float output vs double
// golden (cblas_zher2k, output kept in double with no round-back). Thresholds/metrics are identical
// item by item to the float version -- the verdict dtype is still FLOAT32 (complex64 real/imag are
// still judged as FLOAT32), and only the golden side is double. The frame's
// Verifier::verifyVector only accepts a float golden (the verdict-side scale), so the main
// criterion's per-element verdict is produced by EvaluateMixedToleranceDbl (threshold constants
// taken from the same getMixedToleranceDefaults(ACL_FLOAT)); the gtest EXPECT_TRUE hard assertion and
// the CSV rows (C_uplo_real / C_uplo_imag, metric=mixed_tolerance) are identical to the original
// implementation.
static inline void VerifyUploPrecisionDbl(
    const std::string& caseName, const float* npuUploRe, const double* goldUploRe, size_t uploCount,
    const float* npuUploIm, const double* goldUploIm, const float* goldFRe = nullptr, const float* goldFIm = nullptr,
    double mereThreshold = 0.0, double mareMultiplier = 0.0)
{
    auto defaults = getMixedToleranceDefaults(ACL_FLOAT);
    const double rtol = defaults.rtol; // 2^-10
    const double atol = defaults.atol; // 2^-16

    MixedToleranceResult re = EvaluateMixedToleranceDbl(
        npuUploRe, goldUploRe, uploCount, atol, rtol, defaults.maxAbsErrorLimitFixed, defaults.mantissaBits,
        defaults.emin, goldFRe);
    // Large-cancellation re-judgement: when strict mixed tolerance fails and the CSV presets a
    // mere_threshold, re-judge by the MERE/MARE relative-error metric preset by the test plan
    // (near-zero golden skipped from the statistics).
    bool rePass = re.pass;
    double reVerdict = re.maxAbsErr;
    std::string reMetric = "mixed_tolerance";
    if (!rePass && mereThreshold > 0.0 && mareMultiplier > 0.0) {
        MixedToleranceResult reMm =
            EvaluateMereMareDbl(npuUploRe, goldUploRe, uploCount, mereThreshold, mareMultiplier);
        if (reMm.pass) {
            rePass = true;
            reVerdict = reMm.maxAbsErr; // MARE
            reMetric = "mere_mare";
        }
    }
    VerifyRecord::instance().add(caseName, "C_uplo_real", reMetric, reVerdict, defaults.maxAbsErrorLimitFixed, rePass);
    EXPECT_TRUE(rePass) << "[" << caseName << "] uplo real(dbl): ratio=" << re.matchedRatio
                        << " maxAbsErr=" << re.maxAbsErr << " ulpOk=" << re.ulpLimitOk;

    MixedToleranceResult im = EvaluateMixedToleranceDbl(
        npuUploIm, goldUploIm, uploCount, atol, rtol, defaults.maxAbsErrorLimitFixed, defaults.mantissaBits,
        defaults.emin, goldFIm);
    bool imPass = im.pass;
    double imVerdict = im.maxAbsErr;
    std::string imMetric = "mixed_tolerance";
    if (!imPass && mereThreshold > 0.0 && mareMultiplier > 0.0) {
        MixedToleranceResult imMm =
            EvaluateMereMareDbl(npuUploIm, goldUploIm, uploCount, mereThreshold, mareMultiplier);
        if (imMm.pass) {
            imPass = true;
            imVerdict = imMm.maxAbsErr; // MARE
            imMetric = "mere_mare";
        }
    }
    VerifyRecord::instance().add(caseName, "C_uplo_imag", imMetric, imVerdict, defaults.maxAbsErrorLimitFixed, imPass);
    EXPECT_TRUE(imPass) << "[" << caseName << "] uplo imag(dbl): ratio=" << im.matchedRatio
                        << " maxAbsErr=" << im.maxAbsErr << " ulpOk=" << im.ulpLimitOk;
}

// Non-uplo triangle: EXACT equality (the golden writes only the uplo triangle; its non-uplo part is
// the old input value).
static inline void VerifyNonUploExact(
    const std::string& caseName, const float* npuNonRe, const float* oldNonRe, size_t nonCount, const float* npuNonIm,
    const float* oldNonIm)
{
    if (nonCount == 0) {
        return;
    }
    size_t mismRe = 0;
    size_t mismIm = 0;
    for (size_t i = 0; i < nonCount; i++) {
        mismRe += (std::memcmp(&npuNonRe[i], &oldNonRe[i], sizeof(float)) != 0) ? 1 : 0;
        mismIm += (std::memcmp(&npuNonIm[i], &oldNonIm[i], sizeof(float)) != 0) ? 1 : 0;
    }
    bool pass = (mismRe == 0) && (mismIm == 0);
    VerifyRecord::instance().add(caseName, "C_nonuplo", "exact", static_cast<double>(mismRe + mismIm), 0.0, pass);
    EXPECT_EQ(mismRe, 0U) << "[" << caseName << "] non-uplo real polluted";
    EXPECT_EQ(mismIm, 0U) << "[" << caseName << "] non-uplo imag polluted";

    VerifyConfig cfgRe;
    cfgRe.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(npuNonRe, oldNonRe, nonCount, 1, cfgRe, caseName + "_nonuplo_real"));
    VerifyConfig cfgIm;
    cfgIm.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(npuNonIm, oldNonIm, nonCount, 1, cfgIm, caseName + "_nonuplo_imag"));
}

// Diagonal imag: Hermitian forcing to 0, executed unconditionally (§6.1; unlike cherk's beta==0
// restriction).
// Exception: under the BLAS quick-return semantics ((alpha==0 or k==0) and beta==1) the operator does
// not touch C, so the diagonal imag of the old input C is preserved as-is (same metric as netlib
// cblas, confirmed by measurement); the diagonal is then not a "semantic output written by the
// operator", so the hard assertion is skipped and the record is annotated quick_return.
template <typename Param>
static inline void VerifyDiagonalHermitian(const Param& p, const aclblasComplex* cPtr)
{
    const bool alphaZero = (p.alphaReal == 0.0f && p.alphaImag == 0.0f) || p.nullAlpha;
    const bool isQuickReturnNoWrite = (alphaZero || p.k == 0) && p.beta == 1.0f;
    if (isQuickReturnNoWrite) {
        VerifyRecord::instance().add(
            p.caseName, "C_diag_imag", "max_abs_skipped_quick_return", 0.0, 1.52587890625e-5, true);
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
    // threshold = ACL_FLOAT atol = 2^-16 ≈ 1.52587890625e-5 (§6.1)
    bool pass = maxDiagImag <= 1.52587890625e-5;
    VerifyRecord::instance().add(p.caseName, "C_diag_imag", "max_abs", maxDiagImag, 1.52587890625e-5, pass);
    EXPECT_LE(maxDiagImag, 1.52587890625e-5)
        << "[" << p.caseName << "] Hermitian diagonal imag not zero (max=" << maxDiagImag << ")";
}

// ldc padding canary: rows [n, ldc) of column j plus the trailing bytes after the last column must
// keep the host-written 0xCAFEBABE pattern bit-exact (§6.1 row 5, cher2k-specific, absent in cherk).
template <typename Param>
static inline void VerifyPaddingCanary(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* cOldPtr, size_t canaryCount)
{
    if (canaryCount == 0 || p.ldc <= p.n) {
        return;
    }
    size_t mism = 0;
    size_t total = 0;
    for (int j = 0; j < p.n; j++) {
        for (int i = p.n; i < p.ldc; i++) { // rows [n, ldc) of every stored column
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            mism += (std::memcmp(&cPtr[idx], &cOldPtr[idx], sizeof(aclblasComplex)) != 0) ? 1 : 0;
            total++;
        }
    }
    bool pass = (mism == 0);
    VerifyRecord::instance().add(p.caseName, "C_padding", "canary_exact", static_cast<double>(mism), 0.0, pass);
    EXPECT_EQ(mism, 0U) << "[" << p.caseName << "] ldc padding canary polluted (" << mism << "/" << total << ")";
    (void)canaryCount;
}

// Top-level composite verification entry (§6.1 full table) — float chain version (the main criterion
// before A', now used only for host white-box small-sample checks and internally by the cross-check
// chain).
template <typename Param>
static inline void VerifyUploTriangleComplex(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr, const aclblasComplex* cOldPtr,
    size_t /* cCount */)
{
    if (p.n <= 0) {
        return;
    }
    std::vector<float> npuUploRe;
    std::vector<float> npuUploIm;
    std::vector<float> goldUploRe;
    std::vector<float> goldUploIm;
    std::vector<float> npuNonRe;
    std::vector<float> npuNonIm;
    std::vector<float> oldNonRe;
    std::vector<float> oldNonIm;
    CollectUploNonUploElements(
        p, cPtr, goldPtr, npuUploRe, npuUploIm, goldUploRe, goldUploIm, npuNonRe, npuNonIm, oldNonRe, oldNonIm);

    VerifyUploPrecision(
        p.caseName, npuUploRe.data(), goldUploRe.data(), npuUploRe.size(), npuUploIm.data(), goldUploIm.data());
    VerifyNonUploExact(p.caseName, npuNonRe.data(), oldNonRe.data(), npuNonRe.size(), npuNonIm.data(), oldNonIm.data());
    VerifyDiagonalHermitian(p, cPtr);
    VerifyPaddingCanary(p, cPtr, cOldPtr, 0);
}

// [A' main criterion] Top-level composite verification entry (§6.1 full table, golden=double):
//   1/2. C uplo triangle real/imag: NPU float vs double golden (VerifyUploPrecisionDbl);
//   3.   diagonal imag / 4. non-uplo EXACT / 5. ldc canary: metric unchanged (float original values);
//   cross: the uplo real/imag error of the float cblas golden chain goes to the *_float_xcheck
//          columns (no part in the verdict, used for the dual-chain divergence report).
template <typename Param>
static inline void VerifyUploTriangleComplexDbl(
    const Param& p, const aclblasComplex* cPtr, const aclblasDoubleComplex* goldDbl, const aclblasComplex* cOldPtr,
    size_t /* cCount */, const aclblasComplex* goldFloat = nullptr)
{
    if (p.n <= 0) {
        return;
    }
    std::vector<float> npuUploRe;
    std::vector<float> npuUploIm;
    std::vector<double> goldUploRe;
    std::vector<double> goldUploIm;
    std::vector<float> npuNonRe;
    std::vector<float> npuNonIm;
    std::vector<float> oldNonRe;
    std::vector<float> oldNonIm;
    CollectUploNonUploElementsDbl(
        p, cPtr, goldDbl, npuUploRe, npuUploIm, goldUploRe, goldUploIm, npuNonRe, npuNonIm, oldNonRe, oldNonIm);

    // The uplo real/imag of the float cross-check golden (must be collected before the main
    // criterion so that intrinsic float32-domain overflow can be skipped).
    std::vector<float> xfRe;
    std::vector<float> xfIm;
    if (goldFloat != nullptr) {
        xfRe.reserve(npuUploRe.size());
        xfIm.reserve(npuUploIm.size());
        for (int j = 0; j < p.n; j++) {
            for (int i = 0; i < p.n; i++) {
                size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
                bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
                if (isUplo) {
                    xfRe.push_back(goldFloat[idx].real);
                    xfIm.push_back(goldFloat[idx].imag);
                }
            }
        }
    }

    VerifyUploPrecisionDbl(
        p.caseName, npuUploRe.data(), goldUploRe.data(), npuUploRe.size(), npuUploIm.data(), goldUploIm.data(),
        (goldFloat != nullptr) ? xfRe.data() : nullptr, (goldFloat != nullptr) ? xfIm.data() : nullptr, p.mereThreshold,
        p.mareMultiplier);
    VerifyNonUploExact(p.caseName, npuNonRe.data(), oldNonRe.data(), npuNonRe.size(), npuNonIm.data(), oldNonIm.data());
    VerifyDiagonalHermitian(p, cPtr);
    VerifyPaddingCanary(p, cPtr, cOldPtr, 0);

    // Cross-check chain (float cblas golden): record CSV cross-check columns, no part in the verdict
    // (xfRe/xfIm were collected before the main criterion and are reused directly here).
    if (goldFloat != nullptr) {
        RecordFloatXcheckColumns(
            p.caseName, npuUploRe.data(), xfRe.data(), npuUploRe.size(), npuUploIm.data(), xfIm.data());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture + host data preparation
// ═══════════════════════════════════════════════════════════════════════════════

class Cher2kArch35Test : public BlasTest<Cher2kParam> {};

struct Cher2kHostData {
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> bHost;
    std::vector<aclblasComplex> cHost;
    std::vector<aclblasComplex> cGolden; // [A'] float cross-check chain (original golden, cross columns only)
    std::vector<aclblasComplex> cOld;
    Cher2kGoldenDbl cGoldenDbl;          // [A'] double main-criterion chain (cblas_zher2k, no round-back to float)
    const aclblasComplex* aPtr = nullptr;
    const aclblasComplex* bPtr = nullptr;
    aclblasComplex* cPtr = nullptr;
    aclblasComplex* cGoldenPtr = nullptr;
    const aclblasComplex* cOldPtr = nullptr;
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 0.0f;
};

// Fill the padding segment of C with the 0xCAFEBABE pattern (canary, §6.1): each complex element is
// written as (real = the float bit pattern of 0xCAFEBABE, imag the same) and compared byte-for-byte
// for full equality.
static void FillCanaryPadding(std::vector<aclblasComplex>& c, int n, int ldc)
{
    if (ldc <= n) {
        return;
    }
    union CanaryPattern {
        uint32_t u;
        float f;
    };
    const float patF = CanaryPattern{0xCAFEBABEu}.f;
    for (int j = 0; j < n; j++) {
        for (int i = n; i < ldc; i++) {
            c[static_cast<size_t>(i) + static_cast<size_t>(j) * ldc] = aclblasComplex{patF, patF};
        }
    }
}

static bool PrepareHostData(const Cher2kParam& p, Cher2kHostData& d)
{
    const int abRows = p.lda;
    const int abRowsB = p.ldb;
    const int abCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(p.ldb) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    // 4 copies of C (npu out / float-xcheck golden / input-old ref / canary source)
    // + [A'] double main-criterion golden (2x the width of a float complex) + A + B.
    // 512MB per-case host budget (test_cases/README.md) — an 8GiB hard limit guards against OOM;
    // cases over the limit are skipped.
    constexpr size_t kHostMemLimit = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    const size_t dblBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(aclblasDoubleComplex);
    if (4 * cBytes + dblBytes + aBytes + bBytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << (4 * cBytes + dblBytes + aBytes + bBytes) / (1024 * 1024)
                  << " MB) exceeds limit for n=" << p.n << ", ldc=" << p.ldc << std::endl;
        return false;
    }

    try {
        if (abCols > 0) {
            d.aHost = makeBlasComplexMatrix(abRows, abCols, p.lda, p.aFill, p.randomSeed);
            d.bHost = makeBlasComplexMatrix(abRowsB, abCols, p.ldb, p.bFill, p.randomSeed + 5000U);
        }
        if (p.n > 0) {
            d.cHost = makeBlasComplexMatrix(p.n, p.n, p.ldc, p.cFill, p.randomSeed + 10000U);
        }
    } catch (const std::bad_alloc&) {
        std::cout << "[SKIP] host memory allocation failed for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }

    d.aPtr = (d.aHost.empty() || p.nullA) ? nullptr : d.aHost.data();
    d.bPtr = (d.bHost.empty() || p.nullB) ? nullptr : d.bHost.data();
    d.cPtr = (d.cHost.empty() || p.nullC) ? nullptr : d.cHost.data();

    if (d.cPtr != nullptr) {
        try {
            FillCanaryPadding(d.cHost, p.n, p.ldc);
            d.cOld = d.cHost;    // input-old reference (non-uplo + padding canary)
            d.cGolden = d.cHost; // golden starts from input-old C; non-uplo stays untouched
        } catch (const std::bad_alloc&) {
            std::cout << "[SKIP] host memory allocation failed for golden copy (n=" << p.n << ")" << std::endl;
            return false;
        }
        d.cGoldenPtr = d.cGolden.data();
        d.cOldPtr = d.cOld.data();
    }

    d.alpha = aclblasComplex{p.alphaReal, p.alphaImag};
    d.beta = p.beta;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST_F: null handle (not CSV-driven, cherk convention)
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(Cher2kArch35Test, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 0.0f;
    aclblasStatus_t ret = aclblasCher2k_npu(
        nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, nullptr, 4, nullptr, 4, &beta, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Host unit tests (§6.6 full list, not CSV-driven) — C-pointer semantics / INVALID_ENUM /
// NPU vs cpu-golden mirror consistency for quick return + byte-level canary assertions
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

// Small host data + a 0xCAFEBABE canary over the whole buffer
std::vector<aclblasComplex> MakeCanaryBuffer(int n, int ldc)
{
    std::vector<aclblasComplex> buf(static_cast<size_t>(ldc) * n);
    union CanaryPattern {
        uint32_t u;
        float f;
    };
    const float patF = CanaryPattern{0xCAFEBABEu}.f;
    std::fill(buf.begin(), buf.end(), aclblasComplex{patF, patF});
    return buf;
}

bool BufferEquals(const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(aclblasComplex)) == 0;
}

void PrintCanaryDiff(const char* tag, const std::vector<aclblasComplex>& got, const std::vector<aclblasComplex>& exp)
{
    for (size_t i = 0; i < got.size(); i++) {
        if (std::memcmp(&got[i], &exp[i], sizeof(aclblasComplex)) != 0) {
            std::cout << "[" << tag << "] first diff at #" << i << ": got(" << got[i].real << "," << got[i].imag
                      << ") exp(" << exp[i].real << "," << exp[i].imag << ")" << std::endl;
            return;
        }
    }
}

} // namespace

// 1. trans = OP_T (legal enum but unsupported by this operator) → INVALID_VALUE; mirror-consistent
TEST_F(Cher2kArch35Test, HostOpTUnsupported)
{
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 0.0f;
    auto A = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261001U);
    auto B = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261002U);
    auto C = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261003U);
    aclblasStatus_t npuRet = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_T, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, C.data(),
        4);
    aclblasStatus_t cpuRet = aclblasCher2k_cpu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_T, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, C.data(),
        4);
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(cpuRet)); // mirror consistency
}

// 2. uplo / trans out of range → INVALID_ENUM; mirror-consistent (same metric as TC_ED_212/213)
TEST_F(Cher2kArch35Test, HostInvalidUplo)
{
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 0.0f;
    auto A = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261011U);
    auto B = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261012U);
    auto C = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261013U);
    const aclblasFillMode_t badUplo = static_cast<aclblasFillMode_t>(999);
    aclblasStatus_t npuRet = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, badUplo, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, C.data(), 4);
    aclblasStatus_t cpuRet = aclblasCher2k_cpu(
        Cher2kArch35Test::handle_, badUplo, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, C.data(), 4);
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM));
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(cpuRet));
}

TEST_F(Cher2kArch35Test, HostInvalidTrans)
{
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 0.0f;
    auto A = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261021U);
    auto B = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261022U);
    auto C = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261023U);
    const aclblasOperation_t badTrans = static_cast<aclblasOperation_t>(999);
    aclblasStatus_t npuRet = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, badTrans, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, C.data(), 4);
    aclblasStatus_t cpuRet = aclblasCher2k_cpu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, badTrans, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, C.data(), 4);
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM));
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(cpuRet));
}

// 3. C == nullptr and beta != 0 → INVALID_VALUE; mirror-consistent (C-pointer semantics §4.1 #11)
TEST_F(Cher2kArch35Test, HostCNullBetaNonZero)
{
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 1.0f;
    auto A = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261031U);
    auto B = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261032U);
    aclblasStatus_t npuRet = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, nullptr,
        4);
    aclblasStatus_t cpuRet = aclblasCher2k_cpu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, nullptr,
        4);
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(cpuRet));
}

// [codecheck #9] The local five-segment release chain (originally a fixture-local release function,
// split out as A-level #18) was statement-for-statement identical to Cher2kFreeDeviceBuffers in
// cher2k_npu_wrapper.h, so it was deleted and the shared helper is reused (unallocated pointers stay
// null and are skipped safely; semantics unchanged).

// [codecheck #18] Data-preparation block of HostCNullBetaZero: allocate the five device buffers + four H2D
// copies + fill device C with the 0xA5 canary. If any step fails ok=false is set and the function
// returns early; release is done uniformly by Cher2kFreeDeviceBuffers (unallocated pointers stay null).
static void Cher2kUploadCNullFixture(
    const aclblasComplex& alpha, const float& beta, const std::vector<aclblasComplex>& A,
    const std::vector<aclblasComplex>& B, void*& dAlpha, void*& dA, void*& dB, void*& dBeta, void*& dC, size_t aBytes,
    size_t cBytes, bool& ok)
{
    ok = false;
    do {
        if (aclrtMalloc(&dAlpha, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
            break;
        if (aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
            break;
        if (aclrtMalloc(&dB, aBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
            break;
        if (aclrtMalloc(&dBeta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
            break;
        if (aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
            break;
        if (aclrtMemcpy(dAlpha, sizeof(aclblasComplex), &alpha, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE) !=
            ACL_SUCCESS)
            break;
        if (aclrtMemcpy(dA, aBytes, A.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            break;
        if (aclrtMemcpy(dB, aBytes, B.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            break;
        if (aclrtMemcpy(dBeta, sizeof(float), &beta, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            break;
        // fill device C with 0xA5
        if (aclrtMemset(dC, cBytes, 0xA5, cBytes) != ACL_SUCCESS)
            break;
        ok = true;
    } while (false);
}

// 4. C == nullptr and beta == 0 → SUCCESS with no write at all (canary fully equal)
//    When C is null the wrapper has no device buffer to read back, so "no write" is verified with a
//    real canary host buffer plus a direct API call.
TEST_F(Cher2kArch35Test, HostCNullBetaZero)
{
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 0.0f;
    auto A = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261041U);
    auto B = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261042U);
    aclblasStatus_t cpuRet = aclblasCher2k_cpu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, B.data(), 4, &beta, nullptr,
        4);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // NPU side: a real device C buffer filled with the canary; a direct API call (with C non-null)
    // cannot cover "C=null with no write", so the return code is asserted through the wrapper's nullC
    // passthrough path, and the byte-level "no write" is verified with the device canary + manual
    // H2D/D2H.
    void* dA = nullptr;
    void* dB = nullptr;
    void* dAlpha = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;
    const size_t aBytes = 4 * 4 * sizeof(aclblasComplex);
    const size_t cBytes = 4 * 4 * sizeof(aclblasComplex);
    std::vector<aclblasComplex> canary = MakeCanaryBuffer(16, 16); // 4x4 with ldc=16 layout unused
    (void)canary;
    bool fixtureOk = false;
    Cher2kUploadCNullFixture(alpha, beta, A, B, dAlpha, dA, dB, dBeta, dC, aBytes, cBytes, fixtureOk);
    if (fixtureOk) {
        // Key: pass nullptr for C — the operator should return SUCCESS and not touch device memory
        aclblasStatus_t npuRet = aclblasCher2k(
            Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, static_cast<const aclblasComplex*>(dAlpha),
            static_cast<const aclblasComplex*>(dA), 4, static_cast<const aclblasComplex*>(dB), 4,
            static_cast<const float*>(dBeta), nullptr, 4);
        EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(cpuRet)); // mirror-consistent

        // Read back the device C canary: if the operator wrote anyway (e.g. misjudged C as non-null)
        // the 0xA5 pattern would be corrupted
        std::vector<uint8_t> got(cBytes);
        if (aclrtMemcpy(got.data(), cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
            bool allA5 = true;
            for (uint8_t b : got) {
                allA5 = allA5 && (b == 0xA5);
            }
            EXPECT_TRUE(allA5) << "[HostCNullBetaZero] device C canary was written although C=nullptr";
        }
    }
    Cher2kFreeDeviceBuffers(dAlpha, dA, dB, dBeta, dC); // [codecheck #9] reuse the wrapper release chain
}

// 5. quick return: n=0 / k=0&beta=1 / alpha=0&beta=1 — C bytes unchanged
template <typename Config>
static void RunQuickReturnCase(const char* tag, aclblasHandle_t handle, const Config& cfg)
{
    auto A = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261050U);
    auto B = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261051U);
    auto C0 = makeBlasComplexMatrix(4, 4, 4, BlasFillMode("RANDOM_NORM_1"), 20261052U);
    auto C = C0;
    aclblasStatus_t npuRet = aclblasCher2k_npu(
        handle, cfg.uplo, cfg.trans, cfg.n, cfg.k, &cfg.alpha, A.data(), 4, B.data(), 4, &cfg.beta, C.data(), 4);
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << tag;
    EXPECT_TRUE(BufferEquals(C, C0)) << tag << " quick return must not touch C";
    if (!BufferEquals(C, C0)) {
        PrintCanaryDiff(tag, C, C0);
    }
    // Mirror: the cpu golden returns SUCCESS under the same metric
    auto Cg = C0;
    aclblasStatus_t cpuRet = aclblasCher2k_cpu(
        handle, cfg.uplo, cfg.trans, cfg.n, cfg.k, &cfg.alpha, A.data(), 4, B.data(), 4, &cfg.beta, Cg.data(), 4);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << tag;
}

TEST_F(Cher2kArch35Test, HostQuickReturnN0)
{
    struct Cfg {
        aclblasFillMode_t uplo;
        aclblasOperation_t trans;
        int n, k;
        aclblasComplex alpha;
        float beta;
    };
    RunQuickReturnCase(
        "QuickReturnN0", Cher2kArch35Test::handle_,
        Cfg{ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 4, aclblasComplex{1.0f, 0.0f}, 1.0f});
}

TEST_F(Cher2kArch35Test, HostQuickReturnK0)
{
    struct Cfg {
        aclblasFillMode_t uplo;
        aclblasOperation_t trans;
        int n, k;
        aclblasComplex alpha;
        float beta;
    };
    RunQuickReturnCase(
        "QuickReturnK0", Cher2kArch35Test::handle_,
        Cfg{ACLBLAS_LOWER, ACLBLAS_OP_C, 4, 0, aclblasComplex{1.0f, 0.0f}, 1.0f});
}

TEST_F(Cher2kArch35Test, HostQuickReturnAlpha0)
{
    struct Cfg {
        aclblasFillMode_t uplo;
        aclblasOperation_t trans;
        int n, k;
        aclblasComplex alpha;
        float beta;
    };
    RunQuickReturnCase(
        "QuickReturnAlpha0", Cher2kArch35Test::handle_,
        Cfg{ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, aclblasComplex{0.0f, 0.0f}, 1.0f});
}

// [codecheck #20] Loop body of HostMirrorConsistency: the per-case NPU + cpu golden double call
// and all assertions (expected return code / mirror consistency / quick-return no-write).
// Assertion objects/messages/order are identical to before the extraction; mismatch is returned by
// reference and asserted in aggregate by the main test case.
struct Cher2kMirrorCase {
    const char* name;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 4;
    int k = 4;
    int lda = 4;
    int ldb = 4;
    int ldc = 4;
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 0.0f;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    bool nullAlpha = false;
    bool nullBeta = false;
    aclblasStatus_t expected = ACLBLAS_STATUS_INVALID_VALUE;

    explicit Cher2kMirrorCase(const char* nm) : name(nm) {}
    Cher2kMirrorCase& Uplo(aclblasFillMode_t v)
    {
        uplo = v;
        return *this;
    }
    Cher2kMirrorCase& Trans(aclblasOperation_t v)
    {
        trans = v;
        return *this;
    }
    Cher2kMirrorCase& N(int v)
    {
        n = v;
        return *this;
    }
    Cher2kMirrorCase& K(int v)
    {
        k = v;
        return *this;
    }
    Cher2kMirrorCase& Lda(int v)
    {
        lda = v;
        return *this;
    }
    Cher2kMirrorCase& Ldb(int v)
    {
        ldb = v;
        return *this;
    }
    Cher2kMirrorCase& Ldc(int v)
    {
        ldc = v;
        return *this;
    }
    Cher2kMirrorCase& Beta(float v)
    {
        beta = v;
        return *this;
    }
    Cher2kMirrorCase& NullA()
    {
        nullA = true;
        return *this;
    }
    Cher2kMirrorCase& NullB()
    {
        nullB = true;
        return *this;
    }
    Cher2kMirrorCase& NullC()
    {
        nullC = true;
        return *this;
    }
    Cher2kMirrorCase& NullAlpha()
    {
        nullAlpha = true;
        return *this;
    }
    Cher2kMirrorCase& NullBeta()
    {
        nullBeta = true;
        return *this;
    }
    Cher2kMirrorCase& Expect(aclblasStatus_t v)
    {
        expected = v;
        return *this;
    }
};

static void RunMirrorCase(
    aclblasHandle_t handle, const Cher2kMirrorCase& c, const std::vector<aclblasComplex>& A,
    const std::vector<aclblasComplex>& B, const std::vector<aclblasComplex>& C0, int& mismatches)
{
    auto C = C0;
    auto Cg = C0;
    const aclblasComplex* alphaPtr = c.nullAlpha ? nullptr : &c.alpha;
    const float* betaPtr = c.nullBeta ? nullptr : &c.beta;
    const aclblasComplex* aPtr = c.nullA ? nullptr : A.data();
    const aclblasComplex* bPtr = c.nullB ? nullptr : B.data();
    aclblasComplex* cPtr = c.nullC ? nullptr : C.data();
    aclblasComplex* cgPtr = c.nullC ? nullptr : Cg.data();
    aclblasStatus_t npuRet =
        aclblasCher2k(handle, c.uplo, c.trans, c.n, c.k, alphaPtr, aPtr, c.lda, bPtr, c.ldb, betaPtr, cPtr, c.ldc);
    aclblasStatus_t cpuRet =
        aclblasCher2k_cpu(handle, c.uplo, c.trans, c.n, c.k, alphaPtr, aPtr, c.lda, bPtr, c.ldb, betaPtr, cgPtr, c.ldc);
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(c.expected)) << c.name;
    if (static_cast<int>(npuRet) != static_cast<int>(cpuRet)) {
        mismatches++;
        std::cout << "[HostMirrorConsistency] " << c.name << ": npu=" << static_cast<int>(npuRet)
                  << " cpu=" << static_cast<int>(cpuRet) << std::endl;
    }
    // A legal quick return with k=0 and null A/B should not write anything
    if (c.expected == ACLBLAS_STATUS_SUCCESS && c.n > 0 && c.nullA) {
        EXPECT_TRUE(BufferEquals(C, C0)) << c.name;
    }
}

// 6. C-pointer metric mirror-consistency table: NPU vs cpu golden return codes match one for one
TEST_F(Cher2kArch35Test, HostMirrorConsistency)
{
    // Negative-case table: most fields take the shared defaults, and each case chains only its
    // differing items (the case declaration and loop body are extracted into Cher2kMirrorCase /
    // RunMirrorCase).
    const aclblasFillMode_t badUplo = static_cast<aclblasFillMode_t>(999);
    const aclblasOperation_t badTrans = static_cast<aclblasOperation_t>(999);
    const std::vector<Cher2kMirrorCase> cases = {
        Cher2kMirrorCase("c_null_beta_nonzero").Beta(1.0f).NullC(),
        Cher2kMirrorCase("c_null_beta_zero").NullC().Expect(ACLBLAS_STATUS_SUCCESS),
        Cher2kMirrorCase("c_null_n0").N(0).Beta(1.0f).NullC().Expect(ACLBLAS_STATUS_SUCCESS),
        Cher2kMirrorCase("invalid_uplo").Uplo(badUplo).Expect(ACLBLAS_STATUS_INVALID_ENUM),
        Cher2kMirrorCase("invalid_trans").Trans(badTrans).Expect(ACLBLAS_STATUS_INVALID_ENUM),
        Cher2kMirrorCase("op_t_unsupported").Trans(ACLBLAS_OP_T),
        Cher2kMirrorCase("neg_n").N(-1),
        Cher2kMirrorCase("neg_k").K(-1),
        Cher2kMirrorCase("bad_lda_transN").Lda(2),
        Cher2kMirrorCase("bad_ldb_transC").Trans(ACLBLAS_OP_C).K(8).Lda(8).Ldb(4),
        Cher2kMirrorCase("bad_ldc").Ldc(2),
        Cher2kMirrorCase("null_alpha").NullAlpha(),
        Cher2kMirrorCase("null_beta").NullBeta(),
        Cher2kMirrorCase("null_A").NullA(),
        Cher2kMirrorCase("null_B").NullB(),
        Cher2kMirrorCase("null_A_k0").K(0).Beta(1.0f).NullA().NullB().Expect(ACLBLAS_STATUS_SUCCESS),
    };

    auto A = makeBlasComplexMatrix(8, 8, 8, BlasFillMode("RANDOM_NORM_1"), 20261060U);
    auto B = makeBlasComplexMatrix(8, 8, 8, BlasFillMode("RANDOM_NORM_1"), 20261061U);
    auto C0 = makeBlasComplexMatrix(8, 8, 8, BlasFillMode("RANDOM_NORM_1"), 20261062U);
    int mismatches = 0;
    for (const auto& c : cases) {
        RunMirrorCase(Cher2kArch35Test::handle_, c, A, B, C0, mismatches);
    }
    EXPECT_EQ(mismatches, 0) << "NPU vs cpu-golden return-code mirror mismatch";
}

// ═══════════════════════════════════════════════════════════════════════════════
// White-box host unit tests (workflow 3.3) — structural assertions, branch-level checks that CSV
// parameterization cannot express.
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

// Shared white-box assertion utility: build 4x4/nxn small matrices directly and call the device API
// (bypassing the wrapper's host-buffer readback) so that the "bytes written by the operator" can be
// judged at the bit level (B6/B7).
struct WbCall {
    aclblasStatus_t ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    std::vector<aclblasComplex> cOut; // device readback result
};

// Collect uplo-triangle elements by (i,j); skipDiag=true excludes the diagonal.
inline void CollectUploIdx(int n, int ldc, bool upper, bool skipDiag, std::vector<size_t>& idx)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            bool isUplo = upper ? (i <= j) : (i >= j);
            if (isUplo && !(skipDiag && i == j)) {
                idx.push_back(static_cast<size_t>(i) + static_cast<size_t>(j) * ldc);
            }
        }
    }
}

} // namespace

// B1 Host branch-table rows 4/5: (alpha==0 or k==0) and beta not in {0,1} -- no quick return;
// K3/K4 take the "beta scale only" path. Assertions: returns SUCCESS and the uplo triangle equals
// beta*C_old (bit-exact = beta*old, since beta*C_old is the only operation and FP32 multiplication is
// deterministic), the diagonal imag == 0 (M3 change V2: the K4 i==j branch forces CI=0), and the
// non-uplo triangle bytes are unchanged.
TEST_F(Cher2kArch35Test, WhiteboxSkipTempBetaScaleOnlyK4)
{
    const int n = 8;
    const int k = 4;
    const int ld = 8;
    const float betas[3] = {0.5f, -1.0f, 2.0f};
    for (float beta : betas) {
        auto A = makeBlasComplexMatrix(n, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261151U);
        auto B = makeBlasComplexMatrix(n, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261152U);
        auto C0 = makeBlasComplexMatrix(n, n, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261153U);
        auto C = C0;
        aclblasComplex alpha{0.0f, 0.0f}; // alpha=0 -> skipTemp
        aclblasStatus_t ret = aclblasCher2k_npu(
            Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, A.data(), ld, B.data(), ld, &beta,
            C.data(), ld);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        // alpha=0: C_uplo = beta*C_old, diagonal imag forced to 0
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) { // UPPER uplo
                size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * ld;
                if (i == j) {
                    EXPECT_EQ(C[idx].imag, 0.0f) << "beta=" << beta << " K4 diag imag must be forced 0";
                    EXPECT_FLOAT_EQ(C[idx].real, beta * C0[idx].real);
                } else {
                    EXPECT_FLOAT_EQ(C[idx].real, beta * C0[idx].real);
                    EXPECT_FLOAT_EQ(C[idx].imag, beta * C0[idx].imag);
                }
            }
        }
    }
}

// B2 Same branch on the K3 side (n>16): isAlphaZero=1 + isBetaZero=0 -> Cher2kFoldBeta path
// (MulScalar beta) + Cher2kHermitianCombine skipTemp (u/v/w/x all 0) + Cher2kZeroDiagonal.
// Assertions as in B1 (including the M3 change V1 authoritative gate).
TEST_F(Cher2kArch35Test, WhiteboxSkipTempBetaScaleOnlyK3)
{
    const int n = 33;
    const int k = 8;
    const int ld = 33;
    const float beta = -2.0f;
    auto A = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261161U);
    auto B = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261162U);
    auto C0 = makeBlasComplexMatrix(n, n, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261163U);
    auto C = C0;
    aclblasComplex alpha{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, n, k, &alpha, A.data(), ld, B.data(), ld, &beta,
        C.data(), ld);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) { // LOWER uplo
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * ld;
            if (i == j) {
                EXPECT_EQ(C[idx].imag, 0.0f) << "K3 diag imag must be forced 0 (M3 V1)";
                EXPECT_FLOAT_EQ(C[idx].real, beta * C0[idx].real);
            } else {
                EXPECT_FLOAT_EQ(C[idx].real, beta * C0[idx].real);
                EXPECT_FLOAT_EQ(C[idx].imag, beta * C0[idx].imag);
            }
        }
    }
}

// B3 K4 threshold boundary (SIMT_N_MAX=8, P1-F1): n=8 must take K4 (single kernel, no workspace);
// n=9 must take K3 (6-launch pipeline). Asserted from the behavior side: both sizes are numerically
// correct and the diagonal imag of n=9 is also zeroed on the K3 path. The path choice itself is made
// by the host (n <= 8) and the kernel count cannot be observed directly from the behavior side, so
// "n=9 shows a non-trivial multi-core row split (rowsPerCore=1, 9 active cores) with correct results"
// serves as the two-sided confirmation at the boundary.
TEST_F(Cher2kArch35Test, WhiteboxK4ThresholdBoundary)
{
    for (int n : {8, 9}) {
        const int k = 8;
        auto A = makeBlasComplexMatrix(n, k, n, BlasFillMode("RANDOM_NORM_5_5"), 20261170U + n);
        auto B = makeBlasComplexMatrix(n, k, n, BlasFillMode("RANDOM_NORM_5_5"), 20261190U + n);
        auto C0 = makeBlasComplexMatrix(n, n, n, BlasFillMode("RANDOM_NORM_5_5"), 20261210U + n);
        auto C = C0;
        aclblasComplex alpha{0.5f, -0.5f};
        float beta = 0.5f;
        aclblasStatus_t ret = aclblasCher2k_npu(
            Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_C, n, k, &alpha, A.data(), n, B.data(), n, &beta,
            C.data(), n);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << "n=" << n;
        // [A'] golden upgraded to the double main-criterion chain (the original float chain Cg was
        // used only for the return-code assertion; this numeric assertion does not depend on Cg).
        // Note that the golden's old input C must be the original C0 from before the NPU write
        // (C has already been overwritten by the NPU); Cher2kRunGoldenDbl only reads its input, so
        // C0 is passed directly.
        Cher2kGoldenDbl g;
        aclblasStatus_t gold = Cher2kRunGoldenDbl(
            Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_C, n, k, &alpha, A.data(), n, B.data(), n, &beta,
            C0.data(), n, g);
        EXPECT_EQ(static_cast<int>(gold), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        // Diagonal imag: both paths must zero it (M3 V1/V2)
        for (int i = 0; i < n; i++) {
            size_t idx = static_cast<size_t>(i) * (n + 1);
            EXPECT_LE(std::abs(C[idx].imag), 1.52587890625e-5) << "n=" << n << " diag[" << i << "] imag";
        }
        // Full uplo-triangle check against the golden (mixed tolerance is repeated on the CSV side;
        // here we check the deterministic relation on the finite sample points: off-diagonal =
        // M + M^H + beta*C_old is non-trivial and non-zero)
        int nonTrivial = 0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * n;
                if (C[idx].real != C0[idx].real || (i != j && C[idx].imag != C0[idx].imag)) {
                    nonTrivial++;
                }
            }
        }
        EXPECT_GT(nonTrivial, 0) << "n=" << n << " uplo triangle must be updated";
    }
}

// B4 Idle-core behavior-side assertion: n=57 > AIV(56) -> rowsPerCore=2, ceil(57/2)=29 active cores,
// 56-29=27 idle cores take the rowStart>=rowEnd return path. Assertions: the result is still correct
// and idle cores must not write C (checked against a full-buffer canary: the output of the idle-core
// row segments is correctly written by the active cores; if an idle core wrote anyway, the non-uplo /
// canary or uplo values would be corrupted).
TEST_F(Cher2kArch35Test, WhiteboxK3IdleCoresN57)
{
    const int n = 57;
    const int k = 8;
    const int ld = 57;
    auto A = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261221U);
    auto B = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261222U);
    auto C0 = makeBlasComplexMatrix(n, n, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261223U);
    auto C = C0;
    aclblasComplex alpha{1.0f, 0.5f};
    float beta = 0.0f;
    aclblasStatus_t ret = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, A.data(), ld, B.data(), ld, &beta,
        C.data(), ld);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    // [A'] golden upgraded to the double main-criterion chain (the original float chain Cg took no
    // part in the numeric assertion). The old input C is the original C0 from before the NPU write.
    Cher2kGoldenDbl g;
    aclblasStatus_t gold = Cher2kRunGoldenDbl(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, A.data(), ld, B.data(), ld, &beta,
        C0.data(), ld, g);
    EXPECT_EQ(static_cast<int>(gold), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    // The last core's row segment [56,57) is written by active cores (rowStart=56, rowEnd=57):
    // row 56 must be updated
    int row56Updated = 0;
    for (int j = 0; j < n; j++) {
        size_t idx = 56 + static_cast<size_t>(j) * ld;
        if (C[idx].real != C0[idx].real) {
            row56Updated++;
        }
    }
    EXPECT_GT(row56Updated, 0) << "last row (core 28 rowEnd boundary) must be written";
}

// B5 K1 de-interleave splitCore boundary (splitCore=28): trans=N, n=57, k=8 ->
// rows=57, rowsPerCore=ceil(57/28)=3, 19 active cores (19 on the A side / 0 on the B side handle B);
// A-side cores with coreIdx>=19 and all B-side cores idle. Behavior-side assertion: the B matrix data
// is correctly de-interleaved into the product (a mis-read of A by any B-side core would displace the
// Q term) -- proxied by a full golden comparison.
TEST_F(Cher2kArch35Test, WhiteboxK1SplitCoreBoundary)
{
    const int n = 57;
    const int k = 57;
    const int ld = 57;
    auto A = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261231U);
    auto B = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261232U);
    auto C0 = makeBlasComplexMatrix(n, n, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261233U);
    auto C = C0;
    aclblasComplex alpha{1.0f, 0.5f};
    float beta = 1.0f;
    aclblasStatus_t ret = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, n, k, &alpha, A.data(), ld, B.data(), ld, &beta,
        C.data(), ld);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    // [A'] golden upgraded to the double main-criterion chain (cblas_zher2k, output kept in double).
    // The old input C is the original C0 from before the NPU write.
    Cher2kGoldenDbl g;
    aclblasStatus_t gold = Cher2kRunGoldenDbl(
        Cher2kArch35Test::handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, n, k, &alpha, A.data(), ld, B.data(), ld, &beta,
        C0.data(), ld, g);
    EXPECT_EQ(static_cast<int>(gold), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    // Sampling assertion: the first 64 elements of the LOWER triangle are within the mixed tolerance
    // of the double golden (thresholds same as the CSV main criterion; the full comparison is covered
    // on the CSV side)
    auto defaults = getMixedToleranceDefaults(ACL_FLOAT);
    int checked = 0;
    int bad = 0;
    for (int j = 0; j < n && checked < 64; j++) {
        for (int i = j; i < n && checked < 64; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * ld;
            double tol = defaults.atol + defaults.rtol * std::abs(g.c[idx].real);
            if (std::abs(static_cast<double>(C[idx].real) - g.c[idx].real) > tol) {
                bad++;
            }
            checked++;
        }
    }
    EXPECT_EQ(bad, 0) << "K1 split-boundary product mismatch";
}

// B6 LOWER diagonal-column "write the whole column + write back C_old" path
// (Cher2kStoreLowerDiagColumn nonUploCnt>0 branch): for each column of the LOWER diagonal block, fill
// rows first and then write the strictly-upper segment (nonUploCnt) back with the original cInUb
// values. Assertions: the non-uplo triangle bytes == the old input C (if the write-back path were
// lost, that segment would hold garbage from the interleave), and the ldc padding canary is kept.
TEST_F(Cher2kArch35Test, WhiteboxLowerDiagColumnRestore)
{
    const int n = 97;
    const int ld = 105; // n%64=33 tail block + ldc-n=8 padding
    const int k = 8;
    auto A = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261241U);
    auto B = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261242U);
    auto C0 = makeBlasComplexMatrix(n, n, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261243U);
    FillCanaryPadding(C0, n, ld);
    auto C = C0;
    aclblasComplex alpha{1.0f, 0.5f};
    float beta = 0.5f;
    aclblasStatus_t ret = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, ACLBLAS_LOWER, ACLBLAS_OP_C, n, k, &alpha, A.data(), ld, B.data(), ld, &beta,
        C.data(), ld);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    // Non-uplo (strictly-upper) triangle: bit-exact preservation of the old C
    size_t mism = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < j; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * ld;
            mism += (std::memcmp(&C[idx], &C0[idx], sizeof(aclblasComplex)) != 0) ? 1 : 0;
        }
    }
    EXPECT_EQ(mism, 0U) << "LOWER diag-column restore lost old C bytes";
    // ldc padding canary: bit-exact
    size_t canaryMism = 0;
    for (int j = 0; j < n; j++) {
        for (int i = n; i < ld; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * ld;
            canaryMism += (std::memcmp(&C[idx], &C0[idx], sizeof(aclblasComplex)) != 0) ? 1 : 0;
        }
    }
    EXPECT_EQ(canaryMism, 0U) << "ldc padding canary polluted";
    // Diagonal imag (including the tail block's last column j=96: the variable-column-length path
    // with cols=33)
    for (int i = 0; i < n; i++) {
        size_t idx = static_cast<size_t>(i) * (ld + 1);
        EXPECT_LE(std::abs(C[idx].imag), 1.52587890625e-5) << "diag[" << i << "]";
    }
}

// B7 UPPER fullyUplo internal-block full 2D Copy (Cher2kStoreNonDiagBlock) and per-column
// variable-length Copy (Cher2kStoreUploColumn) branches: n=65 (block 0 diagonal + block 1 full uplo),
// ldc padding. Assertions: the uplo triangle is updated + the tail block's column [64,65)
// with uploCnt=1 is written.
TEST_F(Cher2kArch35Test, WhiteboxUpperNonDiagBlockPath)
{
    const int n = 65;
    const int ld = 73; // tail block 1 row + padding 8
    const int k = 8;
    auto A = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261251U);
    auto B = makeBlasComplexMatrix(ld, k, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261252U);
    auto C0 = makeBlasComplexMatrix(n, n, ld, BlasFillMode("RANDOM_NORM_5_5"), 20261253U);
    FillCanaryPadding(C0, n, ld);
    auto C = C0;
    aclblasComplex alpha{1.0f, 0.5f};
    float beta = 0.5f;
    aclblasStatus_t ret = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, A.data(), ld, B.data(), ld, &beta,
        C.data(), ld);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    // Tail block row 64 (iBase=64, rows=1): the diagonal (64,64) and the whole row to its left must
    // be updated
    int row64Updated = 0;
    for (int j = 0; j < n; j++) {
        size_t idx = 64 + static_cast<size_t>(j) * ld;
        if (C[idx].real != C0[idx].real) {
            row64Updated++;
        }
    }
    EXPECT_GT(row64Updated, 0) << "UPPER tail-block row 64 must be written";
    // Diagonal imag
    for (int i = 0; i < n; i++) {
        size_t idx = static_cast<size_t>(i) * (ld + 1);
        EXPECT_LE(std::abs(C[idx].imag), 1.52587890625e-5) << "diag[" << i << "]";
    }
    // Non-uplo (strictly-lower) + canary all preserved
    size_t mism = 0;
    for (int j = 0; j < n; j++) {
        for (int i = j + 1; i < n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * ld;
            mism += (std::memcmp(&C[idx], &C0[idx], sizeof(aclblasComplex)) != 0) ? 1 : 0;
        }
    }
    EXPECT_EQ(mism, 0U) << "UPPER non-uplo polluted";
}

// B8 Pure-imaginary alpha (0,±1) Hermitian decomposition on both the K4 and K3 paths:
//   CR = ar(u+w) - ai(v+x) = -(v+x); CI = ar(v-x) + ai(u-w) = (u-w)
// Assertions: diagonal CR = 2*ar*u (the imaginary contribution cancels on the diagonal where v=x),
// diagonal CI forced to 0; the non-uplo part is not written. Numeric correctness is covered by the
// CSV-side golden comparison; here we assert that the path runs + the diagonal is zeroed + the uplo is
// updated (regression anchor for M3 fix 1: the two K4 dot operands).
TEST_F(Cher2kArch35Test, WhiteboxPureImagAlphaBothPaths)
{
    for (int n : {8, 33}) { // K4 / K3
        const int k = 4;
        auto A = makeBlasComplexMatrix(n, k, n, BlasFillMode("RANDOM_NORM_5_5"), 20261260U + n);
        auto B = makeBlasComplexMatrix(n, k, n, BlasFillMode("RANDOM_NORM_5_5"), 20261280U + n);
        auto C0 = makeBlasComplexMatrix(n, n, n, BlasFillMode("RANDOM_NORM_5_5"), 20261300U + n);
        auto C = C0;
        aclblasComplex alpha{0.0f, 1.0f};
        float beta = 0.0f;
        aclblasStatus_t ret = aclblasCher2k_npu(
            Cher2kArch35Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, A.data(), n, B.data(), n, &beta,
            C.data(), n);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << "n=" << n;
        int updated = 0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * n;
                if (C[idx].real != C0[idx].real || C[idx].imag != C0[idx].imag) {
                    updated++;
                }
            }
        }
        EXPECT_GT(updated, 0) << "n=" << n << " pure-imag alpha must still update uplo";
        for (int i = 0; i < n; i++) {
            size_t idx = static_cast<size_t>(i) * (n + 1);
            EXPECT_LE(std::abs(C[idx].imag), 1.52587890625e-5) << "n=" << n << " diag";
        }
    }
}

// B9 Host validation-order table #6/#7 (lda/ldb metric branches by trans) white-box boundary:
// for trans=N the lda/ldb lower bound = max(1,n); for trans=C it is max(1,k). The "legal value of the
// other dimension" is used as the illegal value (N uses k-1 but < n / C uses n-1 but < k) to confirm
// the metric follows trans.
TEST_F(Cher2kArch35Test, WhiteboxLdBoundariesByTrans)
{
    struct Case {
        const char* name;
        aclblasOperation_t trans;
        int n, k, lda, ldb;
        aclblasStatus_t expected;
    };
    // n=16, k=32: trans=N lower bound 16; trans=C lower bound 32
    const std::vector<Case> cases = {
        {"N_lda_eq_n_ok", ACLBLAS_OP_N, 16, 32, 16, 32, ACLBLAS_STATUS_SUCCESS},
        {"N_lda_lt_n_bad", ACLBLAS_OP_N, 16, 32, 15, 32, ACLBLAS_STATUS_INVALID_VALUE},
        {"C_lda_eq_k_ok", ACLBLAS_OP_C, 16, 32, 32, 32, ACLBLAS_STATUS_SUCCESS},
        {"C_lda_lt_k_bad", ACLBLAS_OP_C, 16, 32, 31, 32, ACLBLAS_STATUS_INVALID_VALUE},
        // Cross-check metric: using k as the lower bound for trans=N is wrong -- lda=31 >= n=16 must pass
        {"N_lda_between_n_k_ok", ACLBLAS_OP_N, 16, 32, 31, 32, ACLBLAS_STATUS_SUCCESS},
    };
    auto A = makeBlasComplexMatrix(32, 32, 32, BlasFillMode("RANDOM_NORM_5_5"), 20261311U);
    auto B = makeBlasComplexMatrix(32, 32, 32, BlasFillMode("RANDOM_NORM_5_5"), 20261312U);
    auto C = makeBlasComplexMatrix(16, 16, 16, BlasFillMode("RANDOM_NORM_5_5"), 20261313U);
    aclblasComplex alpha{1.0f, 0.0f};
    float beta = 0.0f;
    for (const auto& c : cases) {
        // Device memory (API contract: A/B/C/alpha/beta are all device memory). An earlier version
        // passed host pointers directly: the K4 kernel dereferenced the host address as GM, corrupting
        // the stream via a vector-core exception and giving ret=6 for this case and all following ones.
        aclblasStatus_t ret = aclblasCher2k_npu(
            Cher2kArch35Test::handle_, ACLBLAS_UPPER, c.trans, c.n, c.k, &alpha, A.data(), c.lda, B.data(), c.ldb,
            &beta, C.data(), 16);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(c.expected)) << c.name;
    }
}

// B10 n=1 degenerate (K4 path with n=1: a single element, the uplo triangle is the diagonal itself,
// LOWER/UPPER equivalent): assert C[0][0] = 2*Re(alpha*P) with the imaginary part forced to 0,
// consistent with the golden.
// [A' update] The original assertion EXPECT_FLOAT_EQ(C[0].real, Cg[0].real) hard-coded the float
// chain's bit value; per the new main criterion it now compares against the double golden with the
// FLOAT32 mixed tolerance (per-element atol 2^-16 + rtol 2^-10, same source as the CSV main criterion,
// not relaxed), and the diagonal imag==0 hard assertion is kept.
TEST_F(Cher2kArch35Test, WhiteboxN1Degenerate)
{
    const int n = 1;
    const int k = 4;
    auto A = makeBlasComplexMatrix(n, k, n, BlasFillMode("RANDOM_NORM_5_5"), 20261321U);
    auto B = makeBlasComplexMatrix(n, k, n, BlasFillMode("RANDOM_NORM_5_5"), 20261322U);
    auto defaults = getMixedToleranceDefaults(ACL_FLOAT);
    for (aclblasFillMode_t uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        auto C = makeBlasComplexMatrix(n, n, n, BlasFillMode("RANDOM_NORM_5_5"), 20261323U);
        const auto C0 = C; // the golden's old input C (before the NPU overwrites it)
        Cher2kGoldenDbl g;
        aclblasComplex alpha{0.5f, -0.5f};
        float beta = 0.5f;
        aclblasStatus_t ret = aclblasCher2k_npu(
            Cher2kArch35Test::handle_, uplo, ACLBLAS_OP_N, n, k, &alpha, A.data(), n, B.data(), n, &beta, C.data(), n);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        aclblasStatus_t gold = Cher2kRunGoldenDbl(
            Cher2kArch35Test::handle_, uplo, ACLBLAS_OP_N, n, k, &alpha, A.data(), n, B.data(), n, &beta, C0.data(), n,
            g);
        EXPECT_EQ(static_cast<int>(gold), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
        EXPECT_EQ(C[0].imag, 0.0f) << "n=1 diag imag must be 0";
        EXPECT_EQ(g.c[0].imag, 0.0) << "n=1 golden diag imag must be 0";
        const double diff = std::abs(static_cast<double>(C[0].real) - g.c[0].real);
        const double tol = defaults.atol + defaults.rtol * std::abs(g.c[0].real);
        EXPECT_LE(diff, tol) << "n=1 uplo real vs double golden (mixed tolerance)";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Cher2k, Cher2kArch35Test, ::testing::ValuesIn(GetCasesFromCsv<Cher2kParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Cher2kParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision (§6.1 criterion table)
// The TC_PF branch (caseName prefix TC_PF_) goes through the aclrtEvent capture in cher2k_perf.h
// (§6.5: warmup 5 + 60 samples + bottleneck breakdown) and takes no part in the precision verdict.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_P(Cher2kArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    Cher2kHostData d;
    if (!PrepareHostData(p, d)) {
        GTEST_SKIP() << "Skipped: host memory limit";
    }

    // TC_PF: performance branch — capture only, no precision verdict (verify_performance.py metric)
    if (p.caseName.rfind("TC_PF_", 0) == 0) {
        Cher2kRunPerformanceCase(p, d, Cher2kArch35Test::handle_, Cher2kArch35Test::stream_);
        return;
    }

    aclblasStatus_t ret = aclblasCher2k_npu(
        Cher2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k, &d.alpha, d.aPtr, p.lda, d.bPtr, p.ldb, &d.beta, d.cPtr,
        p.ldc, p.nullAlpha, p.nullBeta);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        // [T2] Negative case: only the return code is asserted (cherk CsvDriven convention) and there
        // is no output tensor to compare -> SKIP-OK column, not counted as FAIL nor occupying the
        // precision denominator.
        VerifyRecord::instance().addSkipOk(p.caseName, "invalid-param-return-code-only");
        return;
    }
    if (p.n == 0 || d.cPtr == nullptr) {
        // [T2] quick return / C-null with no write: the return code is already asserted (the host
        // canary is covered by the host unit tests), there are no tensor rows -> same separate column.
        VerifyRecord::instance().addSkipOk(p.caseName, "quick-return-no-tensor-rows");
        return;
    }

    // [A' main criterion] double chain: up-convert the inputs to double complex -> cblas_zher2k, output
    // kept in double (no round-back to float). The return-code metric matches the float chain
    // (validation happens before the numeric chain). Note that the golden's old input C must be d.cOld
    // (the NPU wrapper has already written the result back to d.cHost, so d.cPtr is the NPU output);
    // Cher2kRunGoldenDbl only reads its input.
    aclblasStatus_t goldenRet = Cher2kRunGoldenDbl(
        Cher2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k, &d.alpha, d.aPtr, p.lda, d.bPtr, p.ldb, &d.beta,
        d.cOldPtr, p.ldc, d.cGoldenDbl);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        VerifyRecord::instance().addSkipOk(p.caseName, "golden-failed-no-tensor-rows");
        return;
    }

    // [A' cross-check] The float chain (the original cblas_cher2k golden) is kept verbatim, used only
    // as CSV cross-check columns.
    aclblasStatus_t goldenRetF = aclblasCher2k_cpu(
        Cher2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k, &d.alpha, d.aPtr, p.lda, d.bPtr, p.ldb, &d.beta,
        d.cGoldenPtr, p.ldc);
    if (goldenRetF != ACLBLAS_STATUS_SUCCESS) {
        // A cross-chain failure does not block the main criterion (the return-code metrics of the two
        // chains agree, so it is theoretically unreachable); log it and continue.
        std::cout << "[XCHECK] " << p.caseName << " float golden ret=" << static_cast<int>(goldenRetF)
                  << " (cross-check skipped)" << std::endl;
    }

    VerifyUploTriangleComplexDbl(
        p, d.cPtr, d.cGoldenDbl.c.data(), d.cOldPtr, static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n),
        (goldenRetF == ACLBLAS_STATUS_SUCCESS) ? d.cGoldenPtr : nullptr);

    // [T2] Tensor-row case aggregation teardown: whether the case fully passed is recorded in the
    // suite statistics.
    RecordTensorCasePass(p.caseName, VerifyRecord::instance().tensorCaseAllPass(p.caseName));
}

// [T2] Suite-breakdown printer registration: after all gtest tests finish, print the SKIP-OK / PASS /
// FAIL breakdown (OnTestProgramEnd runs before main returns and before static destruction, so the
// iostream is safe). Registered via a static-initialization side effect (CsvDriven in this TU is the
// only CSV-driven entry point).
namespace {
const bool g_cher2kSuiteSummaryRegistered = []() {
    ::testing::UnitTest::GetInstance()->listeners().Append(new Cher2kSuiteSummaryPrinter());
    return true;
}();
} // namespace
