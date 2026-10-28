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

#include <algorithm>
#include <array>
#include <cmath>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "types.h"

class PrecisionStrategy {
public:
    virtual ~PrecisionStrategy() = default;

    bool verify(const float* output, const float* golden, size_t count, int64_t stride, const std::string& caseId)
    {
        printHead(output, count, stride, "Output", caseId);
        printHead(golden, count, stride, "Golden", caseId);

        size_t skippedCount = 0;
        for (size_t i = 0; i < count; i++) {
            float outVal = output[static_cast<int64_t>(i) * stride];
            float goldVal = golden[static_cast<int64_t>(i) * stride];
            if (shouldSkip(outVal, goldVal)) {
                skippedCount++;
                continue;
            }
            processElement(outVal, goldVal);
        }

        return reportResult(count, skippedCount, caseId);
    }

    // Verify values supplied by a caller-owned traversal without materializing
    // a second output/golden array. The visitor receives a consumer callback
    // and must invoke it once for each element represented by `count`.
    template <typename Visit>
    bool verifyEach(Visit visit, size_t count, const std::string& caseId)
    {
        constexpr size_t kMaxPrint = 10;
        std::array<float, kMaxPrint> outputPreview{};
        std::array<float, kMaxPrint> goldenPreview{};
        size_t visited = 0;
        size_t skippedCount = 0;
        auto consume = [&](float outVal, float goldVal) {
            if (visited < kMaxPrint) {
                outputPreview[visited] = outVal;
                goldenPreview[visited] = goldVal;
            }
            ++visited;
            if (shouldSkip(outVal, goldVal)) {
                ++skippedCount;
                return;
            }
            processElement(outVal, goldVal);
        };
        visit(consume);

        printHead(outputPreview.data(), visited, 1, "Output", caseId);
        printHead(goldenPreview.data(), visited, 1, "Golden", caseId);
        if (visited != count) {
            std::cout << "[" << caseId << "] FAILED (visitor produced " << visited << "/" << count << " elements)"
                      << std::endl;
            return false;
        }
        return reportResult(count, skippedCount, caseId);
    }

protected:
    virtual bool shouldSkip(float outVal, float goldVal)
    {
        if (outVal == goldVal)
            return true;
        if (std::isnan(outVal) && std::isnan(goldVal))
            return true;
        if (std::isinf(outVal) && std::isinf(goldVal))
            return true;
        return false;
    }

    virtual void processElement(float outVal, float goldVal) = 0;
    virtual bool reportResult(size_t count, size_t skippedCount, const std::string& caseId) = 0;

    static void printHead(
        const float* data, size_t count, int64_t stride, const std::string& label, const std::string& caseId)
    {
        std::cout << std::fixed << std::setprecision(6);
        constexpr size_t kMaxPrint = 10;
        std::cout << "[" << caseId << "] " << label << ": ";
        for (size_t i = 0; i < count && i < kMaxPrint; i++) {
            std::cout << data[static_cast<int64_t>(i) * stride] << " ";
        }
        if (count > kMaxPrint)
            std::cout << "...";
        std::cout << std::endl;
    }
};

class AbsStrategy : public PrecisionStrategy {
public:
    explicit AbsStrategy(double absTol) : absTol_(absTol) {}

protected:
    void processElement(float outVal, float goldVal) override
    {
        // 单侧 NaN/Inf 不应被 abs(out-gold) 的比较语义漏判为通过（issue #536）
        if (!std::isfinite(outVal) || !std::isfinite(goldVal) || std::abs(outVal - goldVal) > absTol_)
            failCount_++;
    }

    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override
    {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << " (absTol=" << absTol_ << ", "
                  << failCount_ << "/" << count << " failures)" << std::endl;
        return pass;
    }

private:
    double absTol_;
    size_t failCount_ = 0;
};

class RelStrategy : public PrecisionStrategy {
public:
    RelStrategy(double relTol, double eps) : relTol_(relTol), eps_(eps) {}

protected:
    void processElement(float outVal, float goldVal) override
    {
        if (!std::isfinite(outVal) || !std::isfinite(goldVal)) {
            maxRelErr_ = std::numeric_limits<double>::infinity();
            return;
        }
        double relErr = std::abs(outVal - goldVal) / (std::abs(goldVal) + eps_);
        if (relErr > maxRelErr_)
            maxRelErr_ = relErr;
    }

    bool reportResult(size_t /*count*/, size_t /*skippedCount*/, const std::string& caseId) override
    {
        bool pass = (maxRelErr_ < relTol_);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << " (maxRelErr=" << maxRelErr_
                  << ", relTol=" << relTol_ << ")" << std::endl;
        return pass;
    }

private:
    double relTol_;
    double eps_;
    double maxRelErr_ = 0.0;
};

class CombinedStrategy : public PrecisionStrategy {
public:
    CombinedStrategy(double absTol, double relTol) : absTol_(absTol), relTol_(relTol) {}

protected:
    void processElement(float outVal, float goldVal) override
    {
        double diff = std::abs(outVal - goldVal);
        double scale = std::abs(goldVal) + 1e-7;
        if (!std::isfinite(outVal) || !std::isfinite(goldVal) || (diff > absTol_ && diff > relTol_ * scale))
            failCount_++;
    }

    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override
    {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << " (absTol=" << absTol_
                  << ", relTol=" << relTol_ << ", " << failCount_ << "/" << count << " failures)" << std::endl;
        return pass;
    }

private:
    double absTol_;
    double relTol_;
    size_t failCount_ = 0;
};

class MereMareStrategy : public PrecisionStrategy {
public:
    MereMareStrategy(double threshold, double multiplier)
        : threshold_(threshold), multiplier_(multiplier), outlierLimit_(multiplier * threshold)
    {}

protected:
    bool shouldSkip(float outVal, float goldVal) override { return PrecisionStrategy::shouldSkip(outVal, goldVal); }

    void processElement(float outVal, float goldVal) override
    {
        if ((std::isnan(outVal) && std::isnan(goldVal)) || (std::isinf(outVal) && std::isinf(goldVal))) {
            return;
        }
        if (std::isinf(outVal) || std::isinf(goldVal) || std::isnan(outVal) || std::isnan(goldVal)) {
            mismatchCount_++;
            return;
        }
        // Near-zero golden: relative error is undefined, skip from MERE/MARE statistics.
        // When |golden| < threshold_, the expected precision itself is below the noise floor.
        double absGold = std::abs(static_cast<double>(goldVal));
        if (absGold < threshold_) {
            skippedNearZero_++;
            return;
        }
        double relErr = std::abs(static_cast<double>(outVal) - static_cast<double>(goldVal)) / (absGold + kEpsilon);
        sumRelErr_ += relErr;
        if (relErr > maxRelErr_)
            maxRelErr_ = relErr;
        if (relErr > outlierLimit_)
            outlierCount_++;
        validCount_++;
    }

    bool reportResult(size_t count, size_t skippedCount, const std::string& caseId) override
    {
        double mere = (validCount_ > 0) ? sumRelErr_ / static_cast<double>(validCount_) : 0.0;

        std::cout << "[" << caseId << "] MERE=" << mere << " MARE=" << maxRelErr_ << " (threshold=" << threshold_
                  << ", outlier_limit=" << outlierLimit_;
        if (skippedCount > 0)
            std::cout << ", skipped " << skippedCount << " elements (exact/nan/inf-equal)";
        if (skippedNearZero_ > 0)
            std::cout << ", skipped " << skippedNearZero_ << " near-zero";
        if (mismatchCount_ > 0)
            std::cout << ", " << mismatchCount_ << " special-value mismatches";
        std::cout << ")" << std::endl;

        bool pass = (mismatchCount_ == 0) && (mere < threshold_) && (maxRelErr_ < outlierLimit_);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << " (MERE < threshold && MARE < "
                  << multiplier_ << "*threshold, " << outlierCount_ << " outliers, " << mismatchCount_
                  << " mismatches out of " << count << " elements)" << std::endl;
        return pass;
    }

private:
    static constexpr double kEpsilon = 0.00006103515625; // 2^-14
    double threshold_;
    double multiplier_;
    double outlierLimit_;
    double sumRelErr_ = 0.0;
    double maxRelErr_ = 0.0;
    size_t outlierCount_ = 0;
    size_t mismatchCount_ = 0;   // INF/NaN-related special-value mismatches
    size_t skippedNearZero_ = 0; // elements skipped because |golden| < threshold
    size_t validCount_ = 0;      // elements contributing to MERE/MARE statistics
};

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers: MIXED_TOLERANCE thresholds (mixed_tolerance_standard.md)
//   per-element max_abs_error_limit = max(fixed_value, 32 * ULP)
//   ULP(x) = 2^(floor(log2|x|) - mantissaBits)  for finite x != 0
//   ULP(0) = 2^(emin - mantissaBits)              (subnormal ULP)
// ═══════════════════════════════════════════════════════════════════════════════
inline double getUlpsAt(double magnitude, int mantissaBits, int emin)
{
    double absMag = std::abs(magnitude);
    if (absMag == 0.0)
        return std::pow(2.0, static_cast<double>(emin - mantissaBits));
    int exp = 0;
    std::frexp(absMag, &exp);
    return std::ldexp(1.0, std::max(exp - 1, emin) - mantissaBits);
}

struct MixedToleranceDefaults {
    double rtol;
    double atol;
    double maxAbsErrorLimitFixed;
    int mantissaBits;
    int emin;
};

inline MixedToleranceDefaults getMixedToleranceDefaults(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16:
            return {0.0009765625, 0.0009765625, 1e-1, 10, -14};
        case ACL_BF16:
            return {0.0078125, 0.0078125, 1e-1, 7, -126};
        case ACL_FLOAT:
            return {0.0001220703125, 0.0001220703125, 1e-2, 23, -126};
        case ACL_FLOAT8_E4M3FN:
            return {0.125, 0.125, 1.0, 3, -6};
        case ACL_FLOAT8_E5M2:
            return {0.25, 0.25, 1.0, 2, -14};
        default:
            return {0.0009765625, 0.0009765625, 1e-1, 10, -14};
    }
}

inline void applyMixedToleranceInternal(VerifyConfig& cfg, const MixedToleranceDefaults& d)
{
    cfg.mode = PrecisionMode::MIXED_TOLERANCE;
    cfg.mixedRtol = d.rtol;
    cfg.mixedAtol = d.atol;
    cfg.mixedRequiredMatchedRatio = 0.99;
    cfg.mixedMantissaBits = d.mantissaBits;
    cfg.mixedEmin = d.emin;
}

// Vector 场景：raw fixed 存入 cfg，per-element ulpLimit = max(fixed, 32*ULP_at_|gold|) 在 processElement 内逐元素算
inline void applyMixedTolerance(VerifyConfig& cfg, aclDataType dtype, const float* goldenData, size_t count)
{
    auto d = getMixedToleranceDefaults(dtype);
    applyMixedToleranceInternal(cfg, d);
    cfg.mixedMaxAbsErrorLimit = d.maxAbsErrorLimitFixed;
    (void)goldenData;
    (void)count;
}

// Scalar 场景：raw fixed 存入，per-element ulpLimit = max(fixed, 32*ULP) 在 verifyScalar 内逐元素算
inline void applyMixedTolerance(VerifyConfig& cfg, aclDataType dtype, float goldenScalar)
{
    auto d = getMixedToleranceDefaults(dtype);
    applyMixedToleranceInternal(cfg, d);
    cfg.mixedMaxAbsErrorLimit = d.maxAbsErrorLimitFixed;
    (void)goldenScalar;
}

class MixedToleranceStrategy : public PrecisionStrategy {
public:
    MixedToleranceStrategy(
        double absTol, double relTol, double requiredMatchedRatio, double fixedLimit, int mantissaBits, int emin)
        : absTol_(absTol),
          relTol_(relTol),
          requiredMatchedRatio_(requiredMatchedRatio),
          fixedLimit_(fixedLimit),
          mantissaBits_(mantissaBits),
          emin_(emin)
    {}

protected:
    // Mixed tolerance accepts matching NaNs and equal infinities, but must
    // reject opposite-sign infinities rather than skip them as a pair.
    bool shouldSkip(float outVal, float goldVal) override
    {
        return outVal == goldVal || (std::isnan(outVal) && std::isnan(goldVal));
    }

    void processElement(float outVal, float goldVal) override
    {
        if (std::isnan(outVal) || std::isnan(goldVal) || std::isinf(outVal) || std::isinf(goldVal)) {
            maxAbsErrorLimitFailed_ = true;
            failCount_++;
            return;
        }
        double diff = std::abs(static_cast<double>(outVal) - static_cast<double>(goldVal));
        if (diff > maxAbsError_)
            maxAbsError_ = diff;
        double goldAbs = std::abs(static_cast<double>(goldVal));
        double tolLimit = absTol_ + relTol_ * goldAbs;
        double ulpLimit = std::max(fixedLimit_, 32.0 * getUlpsAt(goldAbs, mantissaBits_, emin_));
        if (diff > tolLimit)
            failCount_++;
        if (diff > ulpLimit)
            maxAbsErrorLimitFailed_ = true;
    }

    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override
    {
        size_t matchedCount = (count >= failCount_) ? count - failCount_ : 0;
        double matchedRatio = (count > 0) ? static_cast<double>(matchedCount) / static_cast<double>(count) : 1.0;
        bool ratioOk = matchedRatio >= requiredMatchedRatio_;
        bool errorLimitOk = !maxAbsErrorLimitFailed_;
        bool pass = ratioOk && errorLimitOk;

        std::cout << std::scientific << std::setprecision(4);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << " (matchedRatio=" << matchedRatio
                  << " (req=" << requiredMatchedRatio_ << ")"
                  << ", maxAbsErr=" << maxAbsError_ << ", per-element-limit=max(" << fixedLimit_
                  << ", 32*ULP_at_each_element)"
                  << ", " << failCount_ << "/" << count << " failures"
                  << ", atol=" << absTol_ << ", rtol=" << relTol_ << ")" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        return pass;
    }

private:
    double absTol_;
    double relTol_;
    double requiredMatchedRatio_;
    double fixedLimit_;
    int mantissaBits_;
    int emin_;
    size_t failCount_ = 0;
    double maxAbsError_ = 0.0;
    bool maxAbsErrorLimitFailed_ = false;
};

class ExactStrategy : public PrecisionStrategy {
protected:
    void processElement(float /*outVal*/, float /*goldVal*/) override { failCount_++; }

    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override
    {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << " (exact match, " << failCount_ << "/"
                  << count << " mismatches)" << std::endl;
        return pass;
    }

private:
    size_t failCount_ = 0;
};

class IntegerStrategy : public PrecisionStrategy {
protected:
    bool shouldSkip(float /*outVal*/, float /*goldVal*/) override { return false; }

    void processElement(float outVal, float goldVal) override
    {
        if (static_cast<int64_t>(outVal) != static_cast<int64_t>(goldVal))
            failCount_++;
    }

    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override
    {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << " (integer match, " << failCount_ << "/"
                  << count << " mismatches)" << std::endl;
        return pass;
    }

private:
    size_t failCount_ = 0;
};

class Verifier {
public:
    static bool verifyVector(
        const float* output, const float* golden, size_t count, int64_t stride, const VerifyConfig& cfg,
        const std::string& caseId)
    {
        auto strategy = createStrategy(cfg);
        return strategy->verify(output, golden, count, stride, caseId);
    }

    template <typename Visit>
    static bool verifyEach(Visit visit, size_t count, const VerifyConfig& cfg, const std::string& caseId)
    {
        auto strategy = createStrategy(cfg);
        return strategy->verifyEach(visit, count, caseId);
    }

    static bool verifyScalar(float output, float golden, const VerifyConfig& cfg, const std::string& caseId)
    {
        std::cout << "[" << caseId << "] Output: " << output << std::endl;
        std::cout << "[" << caseId << "] Golden: " << golden << std::endl;

        if (output == golden) {
            std::cout << "[" << caseId << "] PASSED (exact match)" << std::endl;
            return true;
        }
        if (std::isnan(output) && std::isnan(golden)) {
            std::cout << "[" << caseId << "] PASSED (both nan)" << std::endl;
            return true;
        }
        bool pass = false;
        double diff = std::abs(static_cast<double>(output) - static_cast<double>(golden));
        switch (cfg.mode) {
            case PrecisionMode::ABS:
                pass = diff < cfg.absTol;
                break;
            case PrecisionMode::REL:
                pass = diff / (std::abs(static_cast<double>(golden)) + cfg.epsilonForRel) < cfg.relTol;
                break;
            case PrecisionMode::MIXED_TOLERANCE:
                if (std::isnan(output) || std::isnan(golden) || std::isinf(output) || std::isinf(golden)) {
                    pass = false;
                } else {
                    double tolLimit = cfg.mixedAtol + cfg.mixedRtol * std::abs(static_cast<double>(golden));
                    double ulpLimit = std::max(
                        cfg.mixedMaxAbsErrorLimit,
                        32.0 * getUlpsAt(std::abs(static_cast<double>(golden)), cfg.mixedMantissaBits, cfg.mixedEmin));
                    pass = (diff <= tolLimit) && (diff <= ulpLimit);
                }
                break;
            default:
                pass = diff < cfg.absTol;
                break;
        }

        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << " (diff=" << diff << ")" << std::endl;
        return pass;
    }

    static bool verifyInteger(int64_t output, int64_t golden, const std::string& caseId)
    {
        std::cout << "[" << caseId << "] Output: " << output << std::endl;
        std::cout << "[" << caseId << "] Golden: " << golden << std::endl;
        bool pass = (output == golden);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED") << std::endl;
        return pass;
    }

    // ── Complex Float MERE/MARE verification (real/imag separately) ──
    static bool verifyMereMareComplexFloat(
        const aclblasComplex* output, const aclblasComplex* golden, size_t count, double threshold, double multiplier,
        double epsilon, const std::string& caseId)
    {
        // Split into real and imaginary parts
        std::vector<float> outReal(count), outImag(count);
        std::vector<float> goldReal(count), goldImag(count);
        for (size_t i = 0; i < count; i++) {
            outReal[i] = output[i].real;
            outImag[i] = output[i].imag;
            goldReal[i] = golden[i].real;
            goldImag[i] = golden[i].imag;
        }

        MereMareStrategy realStrategy(threshold, multiplier);
        MereMareStrategy imagStrategy(threshold, multiplier);
        bool realPass = realStrategy.verify(outReal.data(), goldReal.data(), count, 1, caseId + "_real");
        bool imagPass = imagStrategy.verify(outImag.data(), goldImag.data(), count, 1, caseId + "_imag");
        return realPass && imagPass;
    }

private:
    static std::unique_ptr<PrecisionStrategy> createStrategy(const VerifyConfig& cfg)
    {
        switch (cfg.mode) {
            case PrecisionMode::ABS:
                return std::make_unique<AbsStrategy>(cfg.absTol);
            case PrecisionMode::REL:
                return std::make_unique<RelStrategy>(cfg.relTol, cfg.epsilonForRel);
            case PrecisionMode::COMBINED:
                return std::make_unique<CombinedStrategy>(cfg.absTol, cfg.relTol);
            case PrecisionMode::MERE_MARE:
                return std::make_unique<MereMareStrategy>(cfg.mereThreshold, cfg.mareMultiplier);
            case PrecisionMode::EXACT:
                return std::make_unique<ExactStrategy>();
            case PrecisionMode::INTEGER:
                return std::make_unique<IntegerStrategy>();
            case PrecisionMode::MIXED_TOLERANCE:
                return std::make_unique<MixedToleranceStrategy>(
                    cfg.mixedAtol, cfg.mixedRtol, cfg.mixedRequiredMatchedRatio, cfg.mixedMaxAbsErrorLimit,
                    cfg.mixedMantissaBits, cfg.mixedEmin);
            default:
                return std::make_unique<AbsStrategy>(cfg.absTol);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: get MERE threshold based on output dtype
//   FP16:  threshold = 2^-10 ≈ 0.000977, multiplier = 10
//   BF16:  threshold = 2^-7  ≈ 0.0078125, multiplier = 10
//   FP32:  threshold = 2^-13 ≈ 0.000122, multiplier = 10
//   FP8 E4M3FN: threshold = 2^-3 ≈ 0.125, multiplier = 10
//   FP8 E5M2:   threshold = 2^-2 ≈ 0.25, multiplier = 10
// ═══════════════════════════════════════════════════════════════════════════════

static double getMereThreshold(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16:
            return 0.0009765625; // 2^-10
        case ACL_BF16:
            return 0.0078125; // 2^-7
        case ACL_FLOAT:
            return 0.0001220703125; // 2^-13
        case ACL_FLOAT8_E4M3FN:
            return 0.125; // 2^-3
        case ACL_FLOAT8_E5M2:
            return 0.25; // 2^-2
        default:
            return 0.0009765625;
    }
}

static double getMareMultiplier(aclDataType dtype, int k, aclblasComputeType_t computeType)
{
    if (dtype == ACL_FLOAT16 && computeType == ACLBLAS_COMPUTE_16F && k > 160) {
        return std::max(10.0, static_cast<double>(k) / 3.0);
    }
    return 10.0;
}
