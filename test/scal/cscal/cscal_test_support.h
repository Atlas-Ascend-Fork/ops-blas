/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Common CSV, input and reference utilities. No NPU operations in this file.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <cblas.h>
#include "cann_ops_blas.h"
#include "../../../blas/scal/arch35/cscal_plan.h"
namespace cscaltest {
inline void Require(bool ok, const std::string& what)
{
    if (!ok)
        throw std::runtime_error(what);
}
inline std::string Env(const char* key, const std::string& fallback)
{
    auto v = std::getenv(key);
    return v && *v ? v : fallback;
}
inline std::string Trim(std::string s)
{
    const auto a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
inline std::vector<std::string> CsvLine(const std::string& line)
{
    std::vector<std::string> v;
    std::string cell;
    bool quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (quote && i + 1 < line.size() && line[i + 1] == '"') {
                cell += '"';
                ++i;
            } else
                quote = !quote;
        } else if (c == ',' && !quote) {
            v.push_back(Trim(cell));
            cell.clear();
        } else
            cell += c;
    }
    Require(!quote, "unterminated CSV quote");
    v.push_back(Trim(cell));
    return v;
}
inline int Int(const std::string& s)
{
    size_t used = 0;
    long long v = std::stoll(s, &used);
    Require(used == s.size() && v >= INT32_MIN && v <= INT32_MAX, "invalid int: " + s);
    return static_cast<int>(v);
}
inline float Float(const std::string& s)
{
    size_t used = 0;
    float v = std::stof(s, &used);
    Require(used == s.size(), "invalid float: " + s);
    return v;
}
inline bool IsNull(const std::string& s) { return s == "null" || s == "NULLPTR" || s == "nullptr"; }
struct Case {
    std::string id, description, fill = "RANDOM_NORM_5_5", expected = "ACLBLAS_STATUS_SUCCESS";
    int n = 1, incx = 1, offset = 0;
    aclblasComplex alpha{2.0f, -3.0f};
    bool nullAlpha = false, nullX = false, nullHandle = false, performance = false;
    uint32_t seed = 1;
    float mu = 0.0f, sigma = 1.0f;
};
inline std::vector<Case> LoadCases(const std::string& path)
{
    std::ifstream f(path);
    Require(bool(f), "cannot open cases: " + path);
    std::string line;
    Require(bool(std::getline(f, line)), "empty cases file");
    auto header = CsvLine(line);
    std::map<std::string, size_t> cols;
    for (size_t i = 0; i < header.size(); ++i)
        Require(cols.emplace(header[i], i).second, "duplicate column");
    for (const char* k : {"case_name", "n", "incx", "alpha_real", "alpha_imag", "x", "expect_result"})
        Require(cols.count(k) != 0, std::string("missing CSV column: ") + k);
    std::set<std::string> ids;
    std::vector<Case> result;
    while (std::getline(f, line)) {
        if (Trim(line).empty())
            continue;
        auto row = CsvLine(line);
        Require(row.size() == header.size(), "CSV column count mismatch");
        auto get = [&](const std::string& k, const std::string& d = "") { return cols.count(k) ? row[cols[k]] : d; };
        Case c;
        c.id = get("case_name");
        Require(!c.id.empty() && ids.insert(c.id).second, "empty/duplicate case id");
        c.description = get("description");
        c.n = Int(get("n"));
        c.incx = Int(get("incx"));
        c.fill = get("x");
        c.expected = get("expect_result");
        c.nullAlpha = IsNull(get("alpha_real")) || IsNull(get("alpha_imag"));
        if (!c.nullAlpha)
            c.alpha = {Float(get("alpha_real")), Float(get("alpha_imag"))};
        c.nullX = c.fill == "NULLPTR";
        c.nullHandle = get("null_handle", "0") == "1";
        c.offset = Int(get("offset_complex", "0"));
        Require(c.offset >= 0 && c.offset <= 31, "offset outside test range");
        const std::string seed = get("random_seed", "1");
        size_t used = 0;
        auto sv = std::stoull(seed, &used);
        Require(used == seed.size() && sv <= UINT32_MAX, "invalid seed");
        c.seed = uint32_t(sv);
        c.mu = Float(get("normal_mu", "0"));
        c.sigma = Float(get("normal_sigma", "1"));
        Require(std::isfinite(c.mu) && std::isfinite(c.sigma) && c.sigma > 0, "invalid distribution");
        c.performance = c.id.rfind("TC_PF_", 0) == 0 || get("stage") == "performance";
        result.push_back(c);
    }
    Require(!result.empty(), "no CSV cases loaded");
    return result;
}
inline std::string Status(aclblasStatus_t s)
{
    switch (s) {
        case ACLBLAS_STATUS_SUCCESS:
            return "ACLBLAS_STATUS_SUCCESS";
        case ACLBLAS_STATUS_INVALID_VALUE:
            return "ACLBLAS_STATUS_INVALID_VALUE";
        case ACLBLAS_STATUS_HANDLE_IS_NULLPTR:
            return "ACLBLAS_STATUS_HANDLE_IS_NULLPTR";
        case ACLBLAS_STATUS_EXECUTION_FAILED:
            return "ACLBLAS_STATUS_EXECUTION_FAILED";
        default:
            return "STATUS_" + std::to_string(static_cast<int>(s));
    }
}
inline uint32_t Bits(float x) { return cscal950::FloatBits(x); }
inline uint64_t Ulp(float a, float b)
{
    if (a == b)
        return 0;
    if (!std::isfinite(a) || !std::isfinite(b))
        return UINT64_MAX;
    auto ordered = [](uint32_t u) { return u & 0x80000000u ? ~u : u | 0x80000000u; };
    auto x = ordered(Bits(a)), y = ordered(Bits(b));
    return x > y ? uint64_t(x) - y : uint64_t(y) - x;
}
struct Input {
    std::vector<aclblasComplex> data;
    size_t prefix, span;
};
inline Input MakeInput(const Case& c)
{
    uint64_t bytes = 0;
    Require(cscal950::StorageBytes(c.n, c.incx, bytes), "test span overflow");
    Input in;
    in.prefix = 16 + c.offset;
    in.span = std::max<uint64_t>(1, bytes / 8);
    const uint64_t total = in.prefix + in.span + 16;
    // Explicit failure, not a silent skip, when a test would consume excessive memory.
    Require(total <= (uint64_t(1) << 27), "test allocation exceeds 1 GiB safety cap");
    in.data.assign(static_cast<size_t>(total), {123.25f, -456.5f});
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> uniform(-5, 5);
    std::normal_distribution<float> normal(c.mu, c.sigma);
    if (c.n <= 0 || c.incx <= 0 || c.nullX)
        return in;
    const std::set<std::string> modes{"RANDOM_NORM_5_5", "RANDOM_UNIFORM", "RANDOM_NORMAL",
                                      "VALUE_NORM_0",    "RANDOM_ALTER",   "RANDOM_EXTREME",
                                      "VALUE_NORM_INF",  "VALUE_NORM_NAN", "SPECIAL_MIX"};
    Require(modes.count(c.fill), "unsupported fill: " + c.fill);
    auto draw = [&](size_t j) -> float {
        if (c.fill == "VALUE_NORM_0")
            return 0.0f;
        if (c.fill == "RANDOM_NORMAL")
            return normal(rng);
        if (c.fill == "VALUE_NORM_INF")
            return (j & 1 ? -1 : 1) * std::numeric_limits<float>::infinity();
        if (c.fill == "VALUE_NORM_NAN")
            return std::numeric_limits<float>::quiet_NaN();
        if (c.fill == "RANDOM_ALTER")
            return j & 1 ? -5.0f : 5.0f;
        if (c.fill == "RANDOM_EXTREME")
            return j & 1 ? -1.0e30f : 1.0e-30f;
        if (c.fill == "SPECIAL_MIX") {
            float a[] = {0.0f, -0.0f, 1.0f, -2.0f, INFINITY, -INFINITY, NAN, 1.0e-30f};
            return a[j % 8];
        }
        return uniform(rng); // Original RANDOM_NORM_5_5 label is uniform, not Gaussian.
    };
    for (int i = 0; i < c.n; ++i) {
        auto& x = in.data[in.prefix + size_t(i) * c.incx];
        x.real = draw(2 * size_t(i));
        x.imag = draw(2 * size_t(i) + 1);
    }
    return in;
}
inline bool WillWrite(const Case& c)
{
    return !c.nullHandle && !c.nullAlpha && !c.nullX && c.n > 0 && c.incx > 0 &&
           !(c.alpha.real == 1.0f && c.alpha.imag == 0.0f);
}
struct Golden {
    std::vector<aclblasComplex> raw, expected;
    uint64_t policyDiff = 0;
};
inline Golden MakeGolden(const Case& c, const Input& in)
{
    Golden g{in.data, in.data, 0};
    if (!WillWrite(c))
        return g;
    cblas_cscal(c.n, &c.alpha, g.raw.data() + in.prefix, c.incx);
    g.expected = g.raw;
    if (CSCAL_ZERO_WRITES_ZERO && c.alpha.real == 0 && c.alpha.imag == 0) {
        for (int i = 0; i < c.n; ++i) {
            auto& z = g.expected[in.prefix + size_t(i) * c.incx];
            if (Bits(z.real) != 0 || Bits(z.imag) != 0)
                ++g.policyDiff;
            z = {0.0f, 0.0f};
        }
    }
    return g; // Raw CBLAS result is retained, policy override is explicitly counted.
}
struct Metrics {
    uint64_t count = 0, matched = 0, maxUlp = 0;
    double maxAbs = 0;
    bool hardLimit = true;
    void Add(float actual, float expected)
    {
        ++count;
        if (std::isnan(actual) || std::isnan(expected)) {
            const bool ok = std::isnan(actual) && std::isnan(expected);
            matched += ok;
            hardLimit &= ok;
            return;
        }
        if (std::isinf(actual) || std::isinf(expected)) {
            const bool ok = actual == expected;
            matched += ok;
            hardLimit &= ok;
            return;
        }
        const double err = std::abs(double(actual) - double(expected));
        const uint64_t ulp = Ulp(actual, expected);
        maxAbs = std::max(maxAbs, err);
        maxUlp = std::max(maxUlp, ulp);
        matched += err <= std::ldexp(1.0, -16) + std::ldexp(1.0, -10) * std::abs(double(expected));
        hardLimit &= err <= 1e-2 || ulp <= 32;
    }
    double Ratio() const { return count ? double(matched) / count : 1.0; }
    bool Pass() const { return hardLimit && Ratio() >= 0.99; }
};
struct Result {
    bool pass = false, guards = true, zeroExact = true;
    Metrics real, imag;
    uint64_t policyDiff = 0, bytes = 0;
    std::string actualStatus = "UNEXECUTED", error;
    double avgUs = 0, medianUs = 0, minUs = 0, maxUs = 0;
    int samples = 0;
};
inline Result Check(
    const Case& c, const Input& in, const Golden& g, const std::vector<aclblasComplex>& out, aclblasStatus_t status)
{
    Result r;
    r.actualStatus = Status(status);
    r.policyDiff = g.policyDiff;
    Require(out.size() == in.data.size(), "output size mismatch");
    const bool writes = WillWrite(c) && status == ACLBLAS_STATUS_SUCCESS;
    const bool diagnose = std::getenv("CSCAL_DIAGNOSTICS") != nullptr;
    bool diagnosed = false;
    for (size_t p = 0; p < out.size(); ++p) {
        const bool selected =
            writes && p >= in.prefix && p - in.prefix < in.span && (p - in.prefix) % size_t(c.incx) == 0;
        if (!selected)
            r.guards &= std::memcmp(&out[p], &in.data[p], sizeof(aclblasComplex)) == 0;
        else {
            if (diagnose && !diagnosed && std::isfinite(out[p].real) && std::isfinite(g.expected[p].real) &&
                std::abs(double(out[p].real) - double(g.expected[p].real)) > 1e-2 &&
                Ulp(out[p].real, g.expected[p].real) > 32) {
                const auto src = in.data[p];
                volatile float ac = c.alpha.real * src.real, bd = c.alpha.imag * src.imag;
                const float separate = ac - bd;
                const float fma1 = std::fma(c.alpha.real, src.real, -float(bd));
                const float fma2 = std::fma(-c.alpha.imag, src.imag, float(ac));
                std::fprintf(
                    stderr,
                    "CSCAL_DIAG %s p=%zu alpha=%08x,%08x input=%08x,%08x actual=%08x golden=%08x separate=%08x "
                    "fma1=%08x fma2=%08x\n",
                    c.id.c_str(), p, Bits(c.alpha.real), Bits(c.alpha.imag), Bits(src.real), Bits(src.imag),
                    Bits(out[p].real), Bits(g.expected[p].real), Bits(separate), Bits(fma1), Bits(fma2));
                diagnosed = true;
            }
            r.real.Add(out[p].real, g.expected[p].real);
            r.imag.Add(out[p].imag, g.expected[p].imag);
            if (CSCAL_ZERO_WRITES_ZERO && c.alpha.real == 0 && c.alpha.imag == 0)
                r.zeroExact &= Bits(out[p].real) == 0 && Bits(out[p].imag) == 0;
        }
    }
    r.pass = r.actualStatus == c.expected && r.guards && r.zeroExact && r.real.Pass() && r.imag.Pass();
    return r;
}
inline std::string Escape(const std::string& s)
{
    std::string x = "\"";
    for (char c : s) {
        if (c == '"')
            x += '"';
        x += c == '\n' || c == '\r' ? ' ' : c;
    }
    return x + '"';
}
inline void WriteResult(const std::string& path, const Case& c, const Result& r)
{
    std::ifstream old(path, std::ios::binary | std::ios::ate);
    bool header = !old || old.tellg() == std::streampos(0);
    old.close();
    std::ofstream f(path, std::ios::app);
    Require(bool(f), "cannot create result file: " + path);
    if (header)
        f << "case_name,stage,backend,n,incx,expected_status,actual_status,pass,guards_ok,zero_exact,"
             "real_matched_ratio,imag_matched_ratio,max_abs_real,max_abs_imag,max_ulp_real,max_ulp_imag,"
             "netlib_policy_diff,zero_policy,avg_device_us,median_device_us,min_device_us,max_device_us,samples,timing_"
             "method,bytes_allocated,error\n";
    f << Escape(c.id) << ',' << (c.performance ? "performance" : "accuracy") << ",NPU," << c.n << ',' << c.incx << ','
      << Escape(c.expected) << ',' << Escape(r.actualStatus) << ',' << r.pass << ',' << r.guards << ',' << r.zeroExact
      << ',' << std::setprecision(12) << r.real.Ratio() << ',' << r.imag.Ratio() << ',' << r.real.maxAbs << ','
      << r.imag.maxAbs << ',' << r.real.maxUlp << ',' << r.imag.maxUlp << ',' << r.policyDiff << ','
      << (CSCAL_ZERO_WRITES_ZERO ? "task_zero" : "netlib_arithmetic") << ',';
    if (r.samples)
        f << r.avgUs << ',' << r.medianUs << ',' << r.minUs << ',' << r.maxUs;
    else
        f << ",,,";
    f << ',' << r.samples << ',' << (r.samples ? "acl_event_single_launch_warm_reset" : "none") << ',' << r.bytes << ','
      << Escape(r.error) << '\n';
    f.flush();
    Require(bool(f), "failed writing result row");
}
} // namespace cscaltest
