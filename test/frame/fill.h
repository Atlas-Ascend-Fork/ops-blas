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
#include <cfloat>
#include <cmath>
#include <limits>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "dtype_utils.h"
#include "cann_ops_blas_common.h"

constexpr float kBlasSentinel = -999.0f;
#ifndef FLT_TRUE_MIN
#define FLT_TRUE_MIN 1.401298464e-45F
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Layer 1: Configuration
//
// 命名规则: METHOD_PATTERN_VAL...
//   METHOD:  NULLPTR | INDEX | RANDOM | VALUE
//   PATTERN: NORM | UPPER | LOWER | DIAG | ALTER | EXTREME | ILLCOND | BANDED
//   VAL:     数值(N前缀=负, P可省略) 或特殊标记
//   参数化 PATTERN (如 BANDED) 先消耗结构参数, 剩余 VAL 用于填充值
//   从某位开始可不填(后续取默认值), 不允许跳位
// ─────────────────────────────────────────────────────────────────────────────

struct BlasFillMode {
    enum Method { M_NULLPTR, M_INDEX, M_RANDOM, M_GAUSS, M_VALUE } method = M_INDEX;
    enum Pattern { P_NORM, P_UPPER, P_LOWER, P_DIAG, P_ALTER, P_EXTREME, P_ILLCOND, P_BANDED } pattern = P_NORM;
    float val1 = 1.0f;
    float val2 = 1.0f;
    int bandedKl = 0;
    int bandedKu = 0;

    BlasFillMode() = default;
    BlasFillMode(const std::string& s) { parse(s); }

private:
    static const std::map<std::string, Method>& methodMap()
    {
        static const std::map<std::string, Method> m = {
            {"NULLPTR", M_NULLPTR}, {"INDEX", M_INDEX}, {"RANDOM", M_RANDOM}, {"GAUSS", M_GAUSS}, {"VALUE", M_VALUE}};
        return m;
    }

    static const std::map<std::string, Pattern>& patternMap()
    {
        static const std::map<std::string, Pattern> m = {
            {"NORM", P_NORM},   {"UPPER", P_UPPER},     {"LOWER", P_LOWER},     {"DIAG", P_DIAG},
            {"ALTER", P_ALTER}, {"EXTREME", P_EXTREME}, {"ILLCOND", P_ILLCOND}, {"BANDED", P_BANDED}};
        return m;
    }

    static int patternStructParams(Pattern p)
    {
        static const std::map<Pattern, int> m = {{P_BANDED, 2}};
        auto it = m.find(p);
        return (it != m.end()) ? it->second : 0;
    }

    static const std::map<std::string, float>& specialValues()
    {
        static const std::map<std::string, float> m = {
            {"INF", INFINITY}, {"NAN", NAN}, {"ALTER", 0.0f}, {"EXTREME", 0.0f}, {"ILLCOND", 0.0f}};
        return m;
    }

    static float parseNum(const std::string& s)
    {
        auto it = specialValues().find(s);
        if (it != specialValues().end())
            return it->second;

        bool neg = false;
        size_t pos = 0;
        if (pos < s.size() && s[pos] == 'N') {
            neg = true;
            pos++;
        } else if (pos < s.size() && s[pos] == 'P') {
            pos++;
        }

        std::string numStr = s.substr(pos);
        float val = 0.0f;
        if (!numStr.empty()) {
            size_t epos = numStr.find('E');
            if (epos != std::string::npos) {
                std::string mantissa = numStr.substr(0, epos);
                std::string expPart = numStr.substr(epos + 1);
                float mant = mantissa.empty() ? 1.0f : std::stof(mantissa);
                bool expNeg = false;
                size_t eidx = 0;
                if (eidx < expPart.size() && expPart[eidx] == 'N') {
                    expNeg = true;
                    eidx++;
                }
                std::string expNum = expPart.substr(eidx);
                float exp = expNum.empty() ? 0.0f : std::stof(expNum);
                if (expNeg)
                    exp = -exp;
                val = mant * std::pow(10.0f, exp);
            } else {
                val = std::stof(numStr);
            }
        }
        return neg ? -val : val;
    }

    // issue #779: 方法默认值初始化独立成函数，控制 parse() 函数体规模
    static void ApplyMethodDefaults(Method method, float* val1, float* val2)
    {
        static const std::map<Method, std::pair<float, float>> defaults = {
            {M_INDEX, {1.0f, 1.0f}}, {M_RANDOM, {1.0f, 1.0f}}, {M_GAUSS, {5.0f, 2.0f}}, {M_VALUE, {0.0f, 0.0f}}};
        auto dIt = defaults.find(method);
        if (dIt != defaults.end()) {
            *val1 = dIt->second.first;
            *val2 = dIt->second.second;
        }
    }

    void parse(const std::string& s)
    {
        std::vector<std::string> parts;
        std::istringstream iss(s);
        std::string token;
        while (std::getline(iss, token, '_'))
            if (!token.empty())
                parts.push_back(token);

        if (parts.empty())
            return;

        auto mIt = methodMap().find(parts[0]);
        if (mIt == methodMap().end()) {
            // issue #714: 方法名未命中时不再静默保持默认 M_INDEX——数据分布退化会让
            // 特例/极值覆盖无声消失；对齐 csv_loader 未知枚举的可见告警口径
            std::cout << "[fill] unknown method '" << parts[0] << "' in fill '" << s << "', fallback to INDEX\n";
            return;
        }
        method = mIt->second;

        if (method == M_NULLPTR)
            return;

        ApplyMethodDefaults(method, &val1, &val2);

        size_t idx = 1;
        if (idx < parts.size()) {
            auto pIt = patternMap().find(parts[idx]);
            if (pIt != patternMap().end()) {
                pattern = pIt->second;
                idx++;
            } else {
                // issue #714: 模式名未命中时不再静默保持默认 P_NORM
                std::cout << "[fill] unknown pattern '" << parts[idx] << "' in fill '" << s << "', fallback to NORM\n";
            }
        }

        int structParams = patternStructParams(pattern);
        if (structParams >= 1 && idx < parts.size()) {
            bandedKl = static_cast<int>(parseNum(parts[idx]));
            idx++;
        }
        if (structParams >= 2 && idx < parts.size()) {
            bandedKu = static_cast<int>(parseNum(parts[idx]));
            idx++;
        }

        if (idx < parts.size()) {
            val1 = parseNum(parts[idx]);
            val2 = val1;
            idx++;
        }
        if (idx < parts.size())
            val2 = parseNum(parts[idx]);
    }
};

inline BlasFillMode parseFill(const std::string& s) { return BlasFillMode(s); }

// ─────────────────────────────────────────────────────────────────────────────
// Layer 2: Strategy — 值生成策略
// ─────────────────────────────────────────────────────────────────────────────

class ValueGenerator {
public:
    virtual ~ValueGenerator() = default;
    virtual float at(size_t index) = 0;
};

class SequenceGenerator : public ValueGenerator {
    float start_;

public:
    explicit SequenceGenerator(float start) : start_(start) {}
    float at(size_t i) override { return (start_ < 0.0f ? -1.0f : 1.0f) * (std::abs(start_) + static_cast<float>(i)); }
};

class ConstantGenerator : public ValueGenerator {
    float value_;

public:
    explicit ConstantGenerator(float v) : value_(v) {}
    float at(size_t) override { return value_; }
};

class RandomGenerator : public ValueGenerator {
    std::mt19937& rng_;
    std::uniform_real_distribution<float> dist_;

public:
    RandomGenerator(std::mt19937& rng, float lo, float hi) : rng_(rng), dist_(lo, hi) {}
    float at(size_t) override { return dist_(rng_); }
};

// 正态分布：μ ∈ [-muHi, muHi]、σ ∈ [0.1, sigHi] 由 rng 随机采样一次，随后所有元素同分布生成；
// 元素值截断到 [-muHi, muHi]（与均匀分布值域一致），避免正态尾部大值导致三角求解 ill-conditioned
// （对齐生态精度标准 test_case_generation.md「值域控制在 [-5,5]，避免结果超出 32×ULP 硬上限」）。
class GaussGenerator : public ValueGenerator {
    std::mt19937& rng_;
    std::normal_distribution<float> dist_;
    float muHi_;

public:
    GaussGenerator(std::mt19937& rng, float muHi, float sigHi) : rng_(rng), dist_(0.0f, 1.0f), muHi_(muHi)
    {
        std::uniform_real_distribution<float> muDist(-muHi, muHi);
        std::uniform_real_distribution<float> sigDist(0.1f, sigHi);
        dist_.param(std::normal_distribution<float>::param_type(muDist(rng), sigDist(rng)));
    }
    float at(size_t) override
    {
        float v = dist_(rng_);
        return std::min(std::max(v, -muHi_), muHi_);
    }
};

class AlterGenerator : public ValueGenerator {
public:
    float at(size_t i) override { return (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1); }
};

class ExtremeGenerator : public ValueGenerator {
    static constexpr float kVals[] = {
        1.0f, 0.0f, -1.0f, FLT_MAX, FLT_MIN, -FLT_MAX, std::numeric_limits<float>::denorm_min()};

public:
    float at(size_t i) override { return kVals[i % 7]; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Layer 2.5: Factory — map 驱动的策略创建
// ─────────────────────────────────────────────────────────────────────────────

inline std::unique_ptr<ValueGenerator> createGenerator(const BlasFillMode& mode, std::mt19937& rng)
{
    static const std::map<BlasFillMode::Pattern, std::function<std::unique_ptr<ValueGenerator>()>> patternGens = {
        {BlasFillMode::P_ALTER, []() { return std::make_unique<AlterGenerator>(); }},
        {BlasFillMode::P_EXTREME, []() { return std::make_unique<ExtremeGenerator>(); }},
    };

    auto pIt = patternGens.find(mode.pattern);
    if (pIt != patternGens.end())
        return pIt->second();

    static const std::map<
        BlasFillMode::Method, std::function<std::unique_ptr<ValueGenerator>(const BlasFillMode&, std::mt19937&)>>
        methodGens = {
            {BlasFillMode::M_INDEX,
             [](const BlasFillMode& m, std::mt19937&) { return std::make_unique<SequenceGenerator>(m.val1); }},
            {BlasFillMode::M_VALUE,
             [](const BlasFillMode& m, std::mt19937&) { return std::make_unique<ConstantGenerator>(m.val1); }},
            {BlasFillMode::M_RANDOM,
             [](const BlasFillMode& m, std::mt19937& r) {
                 float lo = (m.val1 < 0.0f) ? -FLT_MAX : -m.val1;
                 float hi = (m.val2 < 0.0f) ? FLT_MAX : m.val2;
                 return std::make_unique<RandomGenerator>(r, lo, hi);
             }},
            {BlasFillMode::M_GAUSS,
             [](const BlasFillMode& m, std::mt19937& r) {
                 float muHi = (m.val1 > 0.0f) ? m.val1 : 5.0f;   // μ 半宽，默认 5（μ∈[-5,5]）
                 float sigHi = (m.val2 >= 0.1f) ? m.val2 : 2.0f; // σ 上限，默认 2（σ∈[0.1,2]）
                 return std::make_unique<GaussGenerator>(r, muHi, sigHi);
             }},
        };

    auto mIt = methodGens.find(mode.method);
    if (mIt != methodGens.end())
        return mIt->second(mode, rng);

    return std::make_unique<SequenceGenerator>(1.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 3: Layout — 内存布局函数
//   核心版本接受 BlasFillMode, string 版本构造后委托
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<float> makeBlasArray(int64_t size, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR || size <= 0)
        return {};

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);

    std::vector<float> data(static_cast<size_t>(size));
    for (size_t i = 0; i < data.size(); i++)
        data[i] = gen->at(i);
    return data;
}

inline std::vector<float> makeBlasArray(int64_t size, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasArray(size, BlasFillMode(fillStr), seed);
}

inline std::vector<float> makeBlasBanded(
    int m, int n, int kl, int ku, int lda, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR)
        return {};

    const size_t aSize = (n > 0) ? static_cast<size_t>(lda) * n : 1;
    std::vector<float> data(aSize, 0.0f);
    if (n <= 0)
        return data;

    // For lda < kl+ku+1 shapes the band layout formula (bandRow = ku+i-j,
    // which stays within [0, ku+kl]) can address beyond the lda*n logical
    // storage (formally UB on the heap write, one element past the buffer for
    // the negative-test shapes). Size the buffer to the band-reachable span
    // max(lda, kl+ku+1)*n: every band write stays in-bounds for any
    // m/n/kl/ku/lda (bandRow <= ku+kl <= bandLda-1 and
    // (bandLda-1)+(n-1)*lda <= bandLda*n-1), and the buffer then also matches
    // the sgbmv wrapper's H2D copy size max(lda, kl+ku+1)*n, keeping the host
    // read side of that copy in-bounds as well. Identical to aSize for every
    // valid shape (lda >= kl+ku+1) -- behavior unchanged for all valid rows.
    const int64_t bandLda = std::max<int64_t>(lda, static_cast<int64_t>(kl) + static_cast<int64_t>(ku) + 1);
    if (bandLda > lda && bandLda > 0) {
        data.resize(static_cast<size_t>(bandLda) * static_cast<size_t>(n), 0.0f);
    }

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);

    size_t seq = 0;
    for (int j = 0; j < n; j++) {
        int iStart = std::max(0, j - ku);
        int iEnd = std::min(m - 1, j + kl);
        for (int i = iStart; i <= iEnd; i++) {
            int bandRow = ku + i - j;
            data[bandRow + j * lda] = gen->at(seq++);
        }
    }
    return data;
}

inline std::vector<float> makeBlasBanded(
    int m, int n, int kl, int ku, int lda, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasBanded(m, n, kl, ku, lda, BlasFillMode(fillStr), seed);
}

inline std::vector<float> makeBlasBanded(int m, int n, int lda, const BlasFillMode& fill, uint32_t seed = 0)
{
    return makeBlasBanded(m, n, fill.bandedKl, fill.bandedKu, lda, fill, seed);
}

inline std::vector<float> makeBlasBanded(int m, int n, int lda, const std::string& fillStr, uint32_t seed = 0)
{
    BlasFillMode fill(fillStr);
    return makeBlasBanded(m, n, fill.bandedKl, fill.bandedKu, lda, fill, seed);
}

inline std::vector<float> makeBlasStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR)
        return {};

    const int absInc = std::abs(inc);
    const size_t size = (count > 0) ? static_cast<size_t>((count - 1) * absInc + 1) : 1;
    std::vector<float> data(size, 0.0f);
    if (count <= 0)
        return data;

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);

    for (int i = 0; i < count; i++) {
        int idx = (inc > 0) ? (i * inc) : ((count - 1 - i) * absInc);
        data[idx] = gen->at(i);
    }
    return data;
}

inline std::vector<float> makeBlasStrided(int count, int inc, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasStrided(count, inc, BlasFillMode(fillStr), seed);
}

// Mixed fill: first half RANDOM[-1, 1], second half special value (INF / NAN).
// fillStr format: "MIXED_RANDOM_INF" or "MIXED_RANDOM_NAN".
inline std::vector<float> makeBlasMixed(int count, int inc, const std::string& fillStr, uint32_t seed = 0)
{
    if (count <= 0) {
        return {};
    }
    const int absInc = std::abs(inc);
    const size_t size = static_cast<size_t>((count - 1) * absInc + 1);
    std::vector<float> data(size, 0.0f);
    float specialVal = INFINITY;
    if (fillStr.find("_NAN") != std::string::npos) {
        specialVal = NAN;
    }
    std::mt19937 rng(seed ? seed : 42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int halfCount = count / 2;
    for (int i = 0; i < count; i++) {
        int idx = (inc > 0) ? (i * inc) : ((count - 1 - i) * absInc);
        data[idx] = (i < halfCount) ? dist(rng) : specialVal;
    }
    return data;
}

inline std::vector<float> makeBlasPacked(int n, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR)
        return {};

    const size_t apLen = static_cast<size_t>(n) * (n + 1) / 2;
    std::vector<float> data(apLen);

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);

    bool upper = (fill.pattern == BlasFillMode::P_UPPER);
    size_t idx = 0;
    for (int j = 0; j < n; j++) {
        int iStart = upper ? 0 : j;
        int iEnd = upper ? j : (n - 1);
        for (int i = iStart; i <= iEnd; i++) {
            data[idx] = gen->at(idx);
            idx++;
        }
    }
    return data;
}

inline std::vector<float> makeBlasPacked(int n, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasPacked(n, BlasFillMode(fillStr), seed);
}

inline std::vector<float> makeBlasTriangular(int n, bool upper, const BlasFillMode& fill, uint32_t seed = 0)
{
    BlasFillMode adjusted = fill;
    adjusted.pattern = upper ? BlasFillMode::P_UPPER : BlasFillMode::P_LOWER;
    return makeBlasPacked(n, adjusted, seed);
}

inline std::vector<float> makeBlasTriangular(int n, bool upper, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasTriangular(n, upper, BlasFillMode(fillStr), seed);
}

inline void FillStructuredMatrix(
    std::vector<float>& data, int m, int n, int lda, BlasFillMode::Pattern pattern, std::mt19937& rng)
{
    static const std::map<BlasFillMode::Pattern, std::function<void(std::vector<float>&, int, int, int, std::mt19937&)>>
        layoutMap = {
            {BlasFillMode::P_UPPER,
             [](std::vector<float>& d, int m, int n, int lda, std::mt19937& r) {
                 std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                 for (int j = 0; j < n; j++)
                     for (int i = 0; i <= std::min(j, m - 1); i++)
                         d[i + static_cast<size_t>(j) * lda] = dist(r);
             }},
            {BlasFillMode::P_LOWER,
             [](std::vector<float>& d, int m, int n, int lda, std::mt19937& r) {
                 std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                 int mn = std::min(m, n);
                 for (int j = 0; j < mn; j++)
                     for (int i = j; i < m; i++)
                         d[i + static_cast<size_t>(j) * lda] = dist(r);
             }},
            {BlasFillMode::P_DIAG,
             [](std::vector<float>& d, int m, int n, int lda, std::mt19937& r) {
                 std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                 int mn = std::min(m, n);
                 for (int j = 0; j < mn; j++)
                     d[j + static_cast<size_t>(j) * lda] = dist(r);
             }},
            {BlasFillMode::P_ILLCOND,
             [](std::vector<float>& d, int m, int n, int lda, std::mt19937& r) {
                 std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
                 for (int j = 0; j < n; j++) {
                     float scale = (j % 2 == 0) ? 1e6f : 1e-6f;
                     for (int i = 0; i < m; i++)
                         d[i + static_cast<size_t>(j) * lda] = dist(r) * scale;
                 }
             }},
        };

    auto it = layoutMap.find(pattern);
    if (it != layoutMap.end())
        it->second(data, m, n, lda, rng);
}

inline std::vector<float> makeBlasMatrix(int m, int n, int lda, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR || m <= 0 || n <= 0 || lda <= 0)
        return {};

    const size_t storageSize = static_cast<size_t>(lda) * n;
    const size_t maxIndex = static_cast<size_t>(m - 1) + static_cast<size_t>(n - 1) * lda + 1;
    std::vector<float> data(std::max(storageSize, maxIndex), 0.0f);

    if (fill.method == BlasFillMode::M_VALUE) {
        if (fill.pattern == BlasFillMode::P_DIAG) {
            int mn = std::min(m, n);
            for (int j = 0; j < mn; j++)
                data[j + static_cast<size_t>(j) * lda] = fill.val1;
            return data;
        }
        std::fill(data.begin(), data.end(), fill.val1);
        return data;
    }

    static const std::map<BlasFillMode::Pattern, bool> structured = {
        {BlasFillMode::P_UPPER, true},
        {BlasFillMode::P_LOWER, true},
        {BlasFillMode::P_DIAG, true},
        {BlasFillMode::P_ILLCOND, true}};
    if (structured.count(fill.pattern)) {
        std::mt19937 rng(seed ? seed : 42);
        FillStructuredMatrix(data, m, n, lda, fill.pattern, rng);
        return data;
    }

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);
    for (size_t i = 0; i < storageSize; i++)
        data[i] = gen->at(i);
    return data;
}

inline std::vector<float> makeBlasMatrix(int m, int n, int lda, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasMatrix(m, n, lda, BlasFillMode(fillStr), seed);
}

inline void FillStructuredMatrixRM(
    std::vector<float>& data, int m, int n, int lda, BlasFillMode::Pattern pattern, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    if (pattern == BlasFillMode::P_UPPER) {
        for (int i = 0; i < m; i++)
            for (int j = i; j < n; j++)
                data[static_cast<size_t>(i) * lda + j] = dist(rng);
    } else if (pattern == BlasFillMode::P_LOWER) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= std::min(i, n - 1); j++)
                data[static_cast<size_t>(i) * lda + j] = dist(rng);
    } else if (pattern == BlasFillMode::P_DIAG) {
        int mn = std::min(m, n);
        for (int i = 0; i < mn; i++)
            data[static_cast<size_t>(i) * lda + i] = dist(rng);
    } else if (pattern == BlasFillMode::P_ILLCOND) {
        for (int i = 0; i < m; i++) {
            float scale = (i % 2 == 0) ? 1e6f : 1e-6f;
            for (int j = 0; j < n; j++)
                data[static_cast<size_t>(i) * lda + j] = dist(rng) * scale;
        }
    }
}

inline std::vector<float> makeBlasMatrixRM(int m, int n, int lda, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR || m <= 0 || n <= 0 || lda <= 0)
        return {};

    const size_t storageSize = static_cast<size_t>(m) * lda;
    std::vector<float> data(storageSize, 0.0f);

    if (fill.method == BlasFillMode::M_VALUE) {
        if (fill.pattern == BlasFillMode::P_DIAG) {
            int mn = std::min(m, n);
            for (int i = 0; i < mn; i++)
                data[static_cast<size_t>(i) * lda + i] = fill.val1;
            return data;
        }
        std::fill(data.begin(), data.end(), fill.val1);
        return data;
    }

    static const std::map<BlasFillMode::Pattern, bool> structured = {
        {BlasFillMode::P_UPPER, true},
        {BlasFillMode::P_LOWER, true},
        {BlasFillMode::P_DIAG, true},
        {BlasFillMode::P_ILLCOND, true}};
    if (structured.count(fill.pattern)) {
        std::mt19937 rng(seed ? seed : 42);
        FillStructuredMatrixRM(data, m, n, lda, fill.pattern, rng);
        return data;
    }

    std::mt19937 rng(seed ? seed : 42);
    auto gen = createGenerator(fill, rng);
    for (size_t i = 0; i < storageSize; i++)
        data[i] = gen->at(i);
    return data;
}

inline std::vector<float> makeBlasMatrixRM(int m, int n, int lda, const std::string& fillStr, uint32_t seed = 0)
{
    return makeBlasMatrixRM(m, n, lda, BlasFillMode(fillStr), seed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 3.5: LAPACK 风格矩阵生成 (getrf/getri/getrs 等 batched 算子共用)
//   列主序 n×n 方阵, 支持 batch 感知的 seed 偏移
// ─────────────────────────────────────────────────────────────────────────────

enum class BlasLapackMatrixType {
    RANDOM_NONSINGULAR,
    IDENTITY,
    DIAGONALLY_DOMINANT,
    ILL_CONDITIONED,
    DIAGONAL,
    UPPER_TRIANGULAR,
    LOWER_TRIANGULAR,
    SINGULAR_ZERO_COL,
    SINGULAR_DEPENDENT_ROW,
    MIXED
};

namespace detail {

inline void fillRandomNonsingular(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist01)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            mat[j * lda + i] = dist01(rng);
        mat[j * lda + j] += static_cast<float>(n);
    }
}

inline void fillIdentity(std::vector<float>& mat, int n, int lda)
{
    for (int j = 0; j < n; j++)
        mat[j * lda + j] = 1.0f;
}

inline void fillDiagonallyDominant(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist01)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            mat[j * lda + i] = dist01(rng);
        mat[j * lda + j] = static_cast<float>(n);
    }
}

inline void fillIllConditioned(std::vector<float>& mat, int n, int lda)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            mat[j * lda + i] = 1.0f / static_cast<float>(i + j + 1);
}

inline void fillDiagonal(std::vector<float>& mat, int n, int lda, std::mt19937& rng)
{
    std::uniform_real_distribution<float> diagDist(0.1f, 10.0f);
    for (int j = 0; j < n; j++)
        mat[j * lda + j] = diagDist(rng);
}

inline void fillTriangular(std::vector<float>& mat, int n, int lda, std::mt19937& rng, bool upper)
{
    std::uniform_real_distribution<float> diagDist(0.5f, 2.0f);
    std::uniform_real_distribution<float> offDist(0.0f, 1.0f);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            bool isOff = upper ? (i < j) : (i > j);
            if (isOff)
                mat[j * lda + i] = offDist(rng);
            else if (i == j)
                mat[j * lda + i] = diagDist(rng);
        }
    }
}

inline void fillSingularZeroCol(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist01)
{
    fillRandomNonsingular(mat, n, lda, rng, dist01);
    for (int i = 0; i < n; i++)
        mat[0 * lda + i] = 0.0f;
}

inline void fillSingularDependentRow(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist01)
{
    fillRandomNonsingular(mat, n, lda, rng, dist01);
    int depRow = std::min(n - 1, 2);
    if (depRow > 0)
        for (int j = 0; j < n; j++)
            mat[j * lda + depRow] = 2.0f * mat[j * lda + 0];
}

inline void fillMixed(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist01,
    int batchIdx)
{
    fillRandomNonsingular(mat, n, lda, rng, dist01);
    if (batchIdx % 2 == 1)
        for (int i = 0; i < n; i++)
            mat[0 * lda + i] = 0.0f;
}

} // namespace detail

inline std::vector<float> makeBlasLapackMatrix(
    int n, int lda, BlasLapackMatrixType type, uint32_t seed = 0, int batchIdx = 0)
{
    const size_t matSize = static_cast<size_t>(lda) * std::max(1, n);
    std::vector<float> mat(matSize, 0.0f);
    if (n <= 0)
        return mat;

    std::mt19937 rng(seed + static_cast<uint32_t>(batchIdx) * 1000);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    switch (type) {
        case BlasLapackMatrixType::RANDOM_NONSINGULAR:
            detail::fillRandomNonsingular(mat, n, lda, rng, dist01);
            break;
        case BlasLapackMatrixType::IDENTITY:
            detail::fillIdentity(mat, n, lda);
            break;
        case BlasLapackMatrixType::DIAGONALLY_DOMINANT:
            detail::fillDiagonallyDominant(mat, n, lda, rng, dist01);
            break;
        case BlasLapackMatrixType::ILL_CONDITIONED:
            detail::fillIllConditioned(mat, n, lda);
            break;
        case BlasLapackMatrixType::DIAGONAL:
            detail::fillDiagonal(mat, n, lda, rng);
            break;
        case BlasLapackMatrixType::UPPER_TRIANGULAR:
            detail::fillTriangular(mat, n, lda, rng, true);
            break;
        case BlasLapackMatrixType::LOWER_TRIANGULAR:
            detail::fillTriangular(mat, n, lda, rng, false);
            break;
        case BlasLapackMatrixType::SINGULAR_ZERO_COL:
            detail::fillSingularZeroCol(mat, n, lda, rng, dist01);
            break;
        case BlasLapackMatrixType::SINGULAR_DEPENDENT_ROW:
            detail::fillSingularDependentRow(mat, n, lda, rng, dist01);
            break;
        case BlasLapackMatrixType::MIXED:
            detail::fillMixed(mat, n, lda, rng, dist01, batchIdx);
            break;
    }
    return mat;
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 4: 低精度 / 整型专用填充 (FP8 / FP4 / INT 量化 ST 专用, 不经 BlasFillMode 策略)
//   这些函数产出落在目标 dtype 表示格上的值, 供 bit-exact / 量化往返 ST 直接调用。
// ─────────────────────────────────────────────────────────────────────────────

// integer-valued array (stored as float) in [lo, hi] — for INT8/INT32 bit-exact paths.
// Used by descriptor-style transform/quant ST that compare with EXACT precision.
inline std::vector<float> makeBlasIntArray(int64_t size, int lo, int hi, uint32_t seed = 0)
{
    if (size <= 0)
        return {};
    std::vector<float> data(static_cast<size_t>(size));
    std::mt19937 rng(seed ? seed : 42);
    if (lo > hi)
        std::swap(lo, hi);
    std::uniform_int_distribution<int> dist(lo, hi);
    for (auto& v : data)
        v = static_cast<float>(dist(rng));
    return data;
}

// integer extreme-value array (stored as float): cycles through the dtype's saturation extremes
// and a few interior values. Used by INT8/INT32 bit-exact ST that probe clamp/saturate edges
// (e.g. ±127 / int32 bounds). lo/hi are the saturation extremes for the target dtype.
inline std::vector<float> makeBlasIntExtreme(int64_t size, int lo, int hi)
{
    if (size <= 0)
        return {};
    const int pattern[] = {lo, hi, 0, hi - 1, lo + 1, 1, -1};
    std::vector<float> data(static_cast<size_t>(size));
    for (size_t i = 0; i < data.size(); i++)
        data[i] = static_cast<float>(pattern[i % 7]);
    return data;
}

// ── FP8 quantization-level fill (E4M3FN / E5M2 RINT check). ──
// Cycles through representable FP8 magnitudes plus inter-level midpoints (which exercise
// round-to-nearest-even). Values are kept inside each variant's finite range so the golden
// round-trip and device CAST_RINT agree on the landed level. `e5m2=false` → E4M3FN level set.
// Used by the FP8 quantization special-value cases (test plan S-06 / FP8 RINT check).
inline std::vector<float> makeBlasFp8Levels(int64_t size, bool e5m2, uint32_t seed = 0)
{
    if (size <= 0)
        return {};
    // exact representable magnitudes (subset spanning small/normal range) + inter-level midpoints.
    static const float kE4m3Levels[] = {
        0.0f, 0.015625f, 0.0625f, 0.09375f, 0.125f, 0.25f, 0.375f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f, 3.0f,
        4.0f, 6.0f, 8.0f,
        // midpoints (round-to-nearest-even probes):
        0.1875f, 0.4375f, 0.625f, 1.125f, 2.5f, 5.0f, 7.0f};
    static const float kE5m2Levels[] = {
        0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.375f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f,
        12.0f, 16.0f,
        // midpoints:
        0.3125f, 0.4375f, 0.875f, 1.75f, 2.5f, 5.0f, 10.0f};
    const float* levels = e5m2 ? kE5m2Levels : kE4m3Levels;
    const size_t n = e5m2 ? (sizeof(kE5m2Levels) / sizeof(float)) : (sizeof(kE4m3Levels) / sizeof(float));
    std::vector<float> data(static_cast<size_t>(size));
    std::mt19937 rng(seed ? seed : 42);
    std::uniform_int_distribution<int> signBit(0, 1);
    for (size_t i = 0; i < data.size(); i++) {
        float v = levels[i % n];
        data[i] = (signBit(rng) && v != 0.0f) ? -v : v;
    }
    return data;
}

// ── FP4 E2M1 quantization-level fill (RINT check). ──
// E2M1 has only 8 representable magnitudes {0,0.5,1,1.5,2,3,4,6}; this cycles those plus the
// inter-level midpoints {0.25,0.75,1.25,1.75,2.5,3.5,5.0} that probe round-to-nearest-even.
// Used by the FP4 quantization special-value case (test plan S-07 / FP4 RINT check).
inline std::vector<float> makeBlasFp4Levels(int64_t size, uint32_t seed = 0)
{
    if (size <= 0)
        return {};
    static const float levels[] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f, 3.0f, 4.0f, 6.0f, // exact E2M1 magnitudes
                                   0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};      // inter-level midpoints
    const size_t n = sizeof(levels) / sizeof(float);
    std::vector<float> data(static_cast<size_t>(size));
    std::mt19937 rng(seed ? seed : 42);
    std::uniform_int_distribution<int> signBit(0, 1);
    for (size_t i = 0; i < data.size(); i++) {
        float v = levels[i % n];
        data[i] = (signBit(rng) && v != 0.0f) ? -v : v;
    }
    return data;
}

// ── FP4 E2M1 general random fill (lands on representable lattice for low quantization noise). ──
// Random magnitudes drawn from the E2M1 representable set with random sign, so non-special-value
// FP4 cases do not accumulate large quantization error that could mask layout/permutation bugs.
inline std::vector<float> makeBlasFp4Random(int64_t size, uint32_t seed = 0)
{
    if (size <= 0)
        return {};
    static const float levels[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const size_t n = sizeof(levels) / sizeof(float);
    std::vector<float> data(static_cast<size_t>(size));
    std::mt19937 rng(seed ? seed : 42);
    std::uniform_int_distribution<int> idx(0, static_cast<int>(n) - 1);
    std::uniform_int_distribution<int> signBit(0, 1);
    for (auto& v : data) {
        float m = levels[idx(rng)];
        v = (signBit(rng) && m != 0.0f) ? -m : m;
    }
    return data;
}

inline std::vector<uint8_t> QuantizeMatrix(
    const std::vector<float>& floatData, const BlasFillMode& fill, aclDataType dtype, bool isNull)
{
    if (floatData.empty() || fill.method == BlasFillMode::M_NULLPTR || isNull) {
        return {};
    }
    return quantizeToBytes(floatData, dtype);
}

inline std::vector<float> PrepareGoldenData(const std::vector<float>& floatData, aclDataType dtype, bool isNull)
{
    std::vector<float> golden = floatData;
    if (!golden.empty() && !isNull) {
        quantizeRoundTrip(golden, dtype);
    }
    return golden;
}

inline std::pair<std::vector<std::vector<float>>, std::vector<const float*>> MakeDummyConstMatrices(
    int batchCount, int rows, int cols, float value, bool isNull)
{
    std::vector<std::vector<float>> matrices;
    std::vector<const float*> ptrs;
    if (!isNull) {
        matrices.resize(batchCount);
        ptrs.resize(batchCount);
        for (int b = 0; b < batchCount; b++) {
            matrices[b].resize(static_cast<size_t>(rows) * cols, value);
            ptrs[b] = matrices[b].data();
        }
    }
    return {matrices, ptrs};
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex matrix fill (for complex BLAS operators: cherk/chemm/zherk etc.)
//   fill.h::makeBlasMatrix only produces float buffers. For complex BLAS
//   operators we need aclblasComplex buffers. Strategy: the BlasFillMode string
//   applies to both real and imag parts with the SAME value range but
//   INDEPENDENT seeds — real uses `seed`, imag uses `seed + 1000`. This keeps
//   the real/imag ranges identical while avoiding degenerate correlation.
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<aclblasComplex> makeBlasComplexMatrix(int m, int n, int lda, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || m <= 0 || n <= 0 || lda <= 0) {
        return {};
    }
    std::vector<float> realPart = makeBlasMatrix(m, n, lda, fill, seed);
    std::vector<float> imagPart = makeBlasMatrix(m, n, lda, fill, seed + 1000U);
    const size_t sz = realPart.size();
    std::vector<aclblasComplex> data(sz);
    for (size_t i = 0; i < sz; i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}
