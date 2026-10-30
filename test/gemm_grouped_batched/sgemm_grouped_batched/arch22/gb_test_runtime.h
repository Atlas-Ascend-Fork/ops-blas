/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "gb_test_support.h"
namespace gb_test {
class Fixture {
public:
    Parameters p;
    std::vector<DeviceMatrix> matrices;
    std::unique_ptr<Buffer> dummy;
    uint64_t bytes = 0;
    void SetupContract()
    {
#ifndef GB_CPU_MOCK
        throw std::runtime_error("contract mode is CPU metadata-only; NEVER enqueue fake addresses on NPU");
#else
        for (size_t i = 0; i < p.a.size(); ++i) {
            p.a[i] = reinterpret_cast<const float*>(uintptr_t(0x100000 + 256 * i));
            p.b[i] = reinterpret_cast<const float*>(uintptr_t(0x200000 + 256 * i));
            p.c[i] = reinterpret_cast<float*>(uintptr_t(0x300000 + 256 * i));
        }
#endif
    }
    void SetupInvalid()
    {
        dummy.reset(new Buffer(64, 0));
        dummy->Upload(std::vector<float>(64, kSentinel));
        std::fill(p.a.begin(), p.a.end(), dummy->Data());
        std::fill(p.b.begin(), p.b.end(), dummy->Data());
        std::fill(p.c.begin(), p.c.end(), dummy->Data());
        bytes = dummy->Bytes();
    }
    void BuildSpecs(const Row& row, size_t offset)
    {
        for (int g = 0; g < p.gc; ++g)
            for (int q = 0; q < p.batch[g]; ++q) {
                DeviceMatrix t;
                t.group = g;
                t.batch = q;
                uint32_t seed = static_cast<uint32_t>(std::stoull(Cell(row, "random_seed"))) ^
                                (static_cast<uint32_t>(g) * 0x9e3779b9U) ^ (static_cast<uint32_t>(q) * 0x85ebca6bU);
                t.as = {
                    p.ta[g] == ACLBLAS_OP_N ? p.m[g] : p.k[g], p.ta[g] == ACLBLAS_OP_N ? p.k[g] : p.m[g], p.lda[g],
                    Cell(row, "A_fill"), seed ^ 11U};
                t.bs = {
                    p.tb[g] == ACLBLAS_OP_N ? p.k[g] : p.n[g], p.tb[g] == ACLBLAS_OP_N ? p.n[g] : p.k[g], p.ldb[g],
                    Cell(row, "B_fill"), seed ^ 29U};
                t.cs = {p.m[g], p.n[g], p.ldc[g], Cell(row, "C_fill"), seed ^ 47U};
                const uint64_t add =
                    (Elements(t.as) + Elements(t.bs) + Elements(t.cs) + 3 * (16 + offset)) * sizeof(float);
                if (bytes > UINT64_MAX - add)
                    throw Skip("allocation sum overflow");
                bytes += add;
                matrices.push_back(std::move(t));
            }
    }
    void AllocateMatrices(size_t offset)
    {
        size_t idx = 0;
        // Only one matrix's initialization data is held on Host at a time, even for 24 GiB cases.
        for (auto& t : matrices) {
            t.a.reset(new Buffer(Elements(t.as), offset));
            t.a->Upload(Generate(t.as));
            t.b.reset(new Buffer(Elements(t.bs), offset));
            t.b->Upload(Generate(t.bs));
            t.c.reset(new Buffer(Elements(t.cs), offset));
            t.c->Upload(Generate(t.cs));
            p.a[idx] = t.a->Data();
            p.b[idx] = t.b->Data();
            p.c[idx] = t.c->Data();
            ++idx;
        }
        if (matrices.empty()) {
            dummy.reset(new Buffer(1, 0));
            dummy->Upload({kSentinel});
            p.a[0] = p.b[0] = dummy->Data();
            p.c[0] = dummy->Data();
        }
    }
    void SetupValid(const Row& row, const Options& o)
    {
        if (o.maxFlops && p.Flops() > o.maxFlops)
            throw Skip("FLOP budget; no accuracy result produced");
        const size_t offset = static_cast<size_t>(Int(Cell(row, "offset_floats", "0")));
        if (offset > 31)
            throw std::runtime_error("offset_floats must be 0..31");
        BuildSpecs(row, offset);
        if (o.maxBytes && bytes > o.maxBytes)
            throw Skip("allocation exceeds explicit max-bytes budget");
        size_t free = 0, total = 0;
        CheckACL(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total), "aclrtGetMemInfo");
        if (bytes > free - free / 10)
            throw Skip("insufficient free device memory (10% headroom reserved)");
        AllocateMatrices(offset);
    }
    Fixture(const Row& row, const Options& o) : p(row)
    {
        const auto expected = Expected(row);
        if (o.mode == "contract")
            SetupContract();
        else if (expected != ACLBLAS_STATUS_SUCCESS)
            SetupInvalid();
        else
            SetupValid(row, o);
        p.SetNullElements();
    }
    void CheckDummy() const
    {
        auto data = dummy->ReadChecked();
        for (float v : data)
            if (v != kSentinel)
                throw std::runtime_error("no-op/invalid call changed payload");
    }
    void MakeGolden(
        const DeviceMatrix& t, const std::vector<float>& ha, const std::vector<float>& hb,
        std::vector<float>& golden) const
    {
        const int g = t.group;
        if (!p.m[g] || !p.n[g])
            return;
        // Task explicitly defines k=0 as scale-only, including nonfinite alpha.
        if (p.k[g] == 0 || p.alpha[g] == 0.0f) {
            for (int j = 0; j < p.n[g]; ++j)
                for (int i = 0; i < p.m[g]; ++i) {
                    auto pos = static_cast<size_t>(j) * p.ldc[g] + i;
                    if (p.beta[g] == 0.0f)
                        golden[pos] = 0;
                    else if (p.beta[g] != 1.0f)
                        golden[pos] *= p.beta[g];
                }
            return;
        }
        cblas_sgemm(
            CblasColMajor, p.ta[g] == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans,
            p.tb[g] == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans, p.m[g], p.n[g], p.k[g], p.alpha[g], ha.data(),
            p.lda[g], hb.data(), p.ldb[g], p.beta[g], golden.data(), p.ldc[g]);
    }
    Metrics CheckOutput(
        const DeviceMatrix& t, const std::vector<float>& actual, const std::vector<float>& golden, bool ulpLimit) const
    {
        const int g = t.group;
        Metrics one;
        for (int j = 0; j < p.n[g]; ++j) {
            for (int i = 0; i < p.m[g]; ++i)
                one.Add(
                    actual[static_cast<size_t>(j) * p.ldc[g] + i], golden[static_cast<size_t>(j) * p.ldc[g] + i],
                    ulpLimit);
            for (int i = p.m[g]; i < p.ldc[g]; ++i)
                if (actual[static_cast<size_t>(j) * p.ldc[g] + i] != golden[static_cast<size_t>(j) * p.ldc[g] + i])
                    throw std::runtime_error("C leading-dimension padding modified");
        }
        if (p.m[g] == 0 || p.n[g] == 0)
            if (std::memcmp(actual.data(), golden.data(), actual.size() * sizeof(float)) != 0)
                throw std::runtime_error("empty matrix payload modified");
        return one;
    }
    void AccumulateMetrics(Result& result, const Metrics& one, const DeviceMatrix& t) const
    {
        // Per-matrix gate prevents a failed small group from being diluted by large groups.
        if (!one.Passed())
            result.message += "matrix failure g=" + std::to_string(t.group) + " b=" + std::to_string(t.batch) + "; ";
        result.metrics.total += one.total;
        result.metrics.matched += one.matched;
        result.metrics.nonfinite += one.nonfinite;
        result.metrics.mismatchedNonfinite += one.mismatchedNonfinite;
        result.metrics.hardFailures += one.hardFailures;
        result.metrics.maxAbs = std::max(result.metrics.maxAbs, one.maxAbs);
        result.metrics.maxRel = std::max(result.metrics.maxRel, one.maxRel);
    }
    void CheckInputs(const DeviceMatrix& t, const std::vector<float>& ha, const std::vector<float>& hb) const
    {
        auto da = t.a->ReadChecked(), db = t.b->ReadChecked();
        if (std::memcmp(da.data(), ha.data(), ha.size() * sizeof(float)) ||
            std::memcmp(db.data(), hb.data(), hb.size() * sizeof(float)))
            throw std::runtime_error("read-only A/B or their padding modified");
    }
    void Verify(Result& result, const Options& o) const
    {
        if (dummy) {
            CheckDummy();
            return;
        }
        for (const auto& t : matrices) {
            auto ha = Generate(t.as), hb = Generate(t.bs), golden = Generate(t.cs);
            MakeGolden(t, ha, hb, golden);
            const auto actual = t.c->ReadChecked();
            AccumulateMetrics(result, CheckOutput(t, actual, golden, o.ulpLimit), t);
            if (o.checkInputs)
                CheckInputs(t, ha, hb);
        }
    }
    void ResetC()
    {
        for (auto& t : matrices)
            t.c->Upload(Generate(t.cs));
    }
};
inline bool NeedsOutputReset(const Fixture& f)
{
    bool reset = false;
    for (int g = 0; g < f.p.gc; ++g)
        reset = reset || f.p.beta[g] != 0;
    return reset;
}
inline void MeasurePerformance(Fixture& f, Context& context, const Options& o, Result& result)
{
    // Nonzero beta needs reset outside every timed interval.
    const bool reset = NeedsOutputReset(f);
    for (int i = 0; i < o.warmup; ++i) {
        if (reset)
            f.ResetC();
        CheckBLAS(f.p.Call(context.handle), "warmup");
        CheckACL(aclrtSynchronizeStream(context.stream), "warmup sync");
    }
    aclrtEvent begin = nullptr, end = nullptr;
    try {
        CheckACL(aclrtCreateEvent(&begin), "event create");
        CheckACL(aclrtCreateEvent(&end), "event create");
        for (int i = 0; i < o.samples; ++i) {
            if (reset)
                f.ResetC();
            CheckACL(aclrtRecordEvent(begin, context.stream), "begin event");
            CheckBLAS(f.p.Call(context.handle), "timed call");
            CheckACL(aclrtRecordEvent(end, context.stream), "end event");
            CheckACL(aclrtSynchronizeEvent(end), "event sync");
            float ms = 0;
            CheckACL(aclrtEventElapsedTime(&ms, begin, end), "event elapsed");
            if (!std::isfinite(ms) || ms <= 0)
                throw std::runtime_error("invalid event timing");
            result.eventUs.push_back(double(ms) * 1000.0);
        }
    } catch (...) {
        aclrtSynchronizeStream(context.stream);
        if (begin)
            aclrtDestroyEvent(begin);
        if (end)
            aclrtDestroyEvent(end);
        throw;
    }
    CheckACL(aclrtDestroyEvent(begin), "event destroy");
    CheckACL(aclrtDestroyEvent(end), "event destroy");
    result.message = "Event wall-time only; msprof kernel-sum and separate accuracy gate still required";
}
inline void ClassifyResult(Result& result, const Options& o)
{
    if (!result.metrics.Passed() || (!result.message.empty() && o.mode != "perf"))
        result.status = "FAIL";
    else if (o.mode == "contract")
        result.status = "HOST_CONTRACT_PASS";
    else if (o.mode == "perf")
        result.status = "MEASURED_NOT_ACCEPTED";
#ifdef GB_CPU_MOCK
    else
        result.status = "CPU_MODEL_PASS";
#else
    else
        result.status = "PASS";
#endif
}
inline void ExecuteCase(const Row& row, Context& context, const Options& o, Result& result)
{
    if (o.mode == "perf") {
#ifdef GB_CPU_MOCK
        throw std::runtime_error("performance mode requires real NPU; CPU timings cannot be acceptance data");
#endif
        if (o.samples <= 10 || o.warmup < 5)
            throw std::runtime_error("need warmup >=5 and samples >10");
    }
    Fixture f(row, o);
    result.bytes = f.bytes;
    result.flops = f.p.Flops();
    const auto got = f.p.Call(context.handle), expected = Expected(row);
    // Synchronize even on a bad return so fixtures cannot free still-in-use device buffers.
    CheckACL(aclrtSynchronizeStream(context.stream), "post-call synchronization");
    if (got != expected)
        throw std::runtime_error(
            "status mismatch actual=" + std::to_string(int(got)) + " expected=" + std::to_string(int(expected)));
    if (o.mode == "contract") {
#ifdef GB_CPU_MOCK
        result.launches = gb_mock::LaunchCount();
        if (expected != ACLBLAS_STATUS_SUCCESS && result.launches)
            throw std::runtime_error("invalid call enqueued work");
#endif
    } else if (o.mode == "perf" && got == ACLBLAS_STATUS_SUCCESS)
        MeasurePerformance(f, context, o, result);
    else
        f.Verify(result, o);
#ifdef GB_CPU_MOCK
    result.launches = gb_mock::LaunchCount();
#endif
    ClassifyResult(result, o);
}
inline Result Run(const Row& row, Context& context, const Options& o)
{
    Result result;
    result.name = Cell(row, "case_name");
    result.mode = o.mode;
#ifdef GB_CPU_MOCK
    result.backend = "CPU_MODEL_NOT_NPU";
    gb_mock::SetMetadataOnly(o.mode == "contract");
    gb_mock::ResetCounters();
#else
    result.backend = "ASCEND_NPU";
#endif
    try {
        ExecuteCase(row, context, o, result);
    } catch (const Skip& e) {
        result.status = "SKIP_RESOURCE";
        result.message = e.what();
    } catch (const std::exception& e) {
        result.status = "FAIL";
        result.message = e.what();
    }
#ifdef GB_CPU_MOCK
    gb_mock::SetMetadataOnly(false);
    if (gb_mock::Pending(context.stream) || gb_mock::LiveBytes()) {
        result.status = "FAIL";
        result.message += "; outstanding queue or allocation leak";
    }
#endif
    return result;
}
inline std::string Escape(const std::string& s)
{
    std::ostringstream out;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            out << '\\' << char(c);
        else if (c == '\n')
            out << "\\n";
        else if (c == '\r')
            out << "\\r";
        else if (c == '\t')
            out << "\\t";
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else
            out << char(c);
    }
    return out.str();
}
inline std::string JSON(const Result& r)
{
    std::ostringstream s;
    s << std::setprecision(12);
    s << "{\"case_name\":\"" << Escape(r.name) << "\",\"status\":\"" << r.status << "\",\"backend\":\"" << r.backend
      << "\",\"mode\":\"" << r.mode << "\",\"message\":\"" << Escape(r.message) << "\",\"allocation_bytes\":" << r.bytes
      << ",\"flops\":" << r.flops << ",\"launches\":" << r.launches << ",\"total\":" << r.metrics.total
      << ",\"matched\":" << r.metrics.matched
      << ",\"matched_ratio\":" << (r.metrics.total ? double(r.metrics.matched) / r.metrics.total : 1.0)
      << ",\"max_abs\":" << r.metrics.maxAbs << ",\"max_relative\":" << r.metrics.maxRel
      << ",\"nonfinite\":" << r.metrics.nonfinite << ",\"nonfinite_mismatches\":" << r.metrics.mismatchedNonfinite
      << ",\"hard_failures\":" << r.metrics.hardFailures << ",\"event_us\":[";
    for (size_t i = 0; i < r.eventUs.size(); ++i) {
        if (i)
            s << ',';
        s << r.eventUs[i];
    }
    s << "]}";
    return s.str();
}
} // namespace gb_test
