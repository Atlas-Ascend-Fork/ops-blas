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

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "aclblaslt_algo_heuristic.h"
#include "aclblaslt_layout_impl.h"
#include "fill.h"
#include "lt_matmul_golden.h"
#include "lt_matmul_metadata.h"
#include "lt_matmul_device_buffers.h"

namespace LtMatmul22Test {

inline std::vector<float> SupplementalInput(int rows, int cols, int ld, const std::string& token, uint32_t seed)
{
    const std::string prefix = "NORMAL_RAW_";
    if (token.compare(0, prefix.size(), prefix) == 0) {
        const auto separator = token.find('_', prefix.size());
        if (separator == std::string::npos)
            throw std::runtime_error("NORMAL_RAW requires explicit mean and sigma");
        const float mean = std::stof(token.substr(prefix.size(), separator - prefix.size()));
        const float sigma = std::stof(token.substr(separator + 1));
        if (!std::isfinite(mean) || !std::isfinite(sigma) || mean < -5.0f || mean > 5.0f || sigma < 0.1f ||
            sigma > 2.0f) {
            throw std::runtime_error("NORMAL_RAW parameters violate task mean/sigma ranges");
        }
        std::mt19937 random(seed);
        std::normal_distribution<float> distribution(mean, sigma);
        std::vector<float> result(static_cast<size_t>(rows) * ld);
        for (auto& value : result)
            value = distribution(random); // Deliberately no clipping.
        return result;
    }
    if (token == "BITS_EXACT") {
        constexpr uint32_t values[] = {0x00000000U, 0x80000000U, 0x7fc12345U, 0xffc23456U,
                                       0x7f800000U, 0xff800000U, 0x3f800000U, 0xbf800000U};
        std::vector<float> result(static_cast<size_t>(rows) * ld);
        for (size_t index = 0; index < result.size(); ++index) {
            const uint32_t bits = values[index % 8U];
            CopyBytes(&result[index], sizeof(result[index]), &bits, sizeof(bits));
        }
        return result;
    }
    return makeBlasMatrixRM(rows, cols, ld, token, seed);
}

inline std::array<std::vector<float>, 4> Inputs(const Param& p)
{
    std::array<std::vector<float>, 4> input;
    const auto rows = p.Rows(), cols = p.Cols();
    for (size_t i = 0; i < input.size(); ++i) {
        if (p.fill[i] == "NULLPTR" || (i == 2 && p.inplace))
            continue;
        input[i] =
            SupplementalInput(std::max(0, rows[i]), std::max(0, cols[i]), p.StorageLd(i), p.fill[i], p.randomSeed + i);
        // makeBlasMatrixRM returns no elements for logical empty matrices. Their
        // physical rows/pitch still need storage before the padding walk below.
        const size_t storage = static_cast<size_t>(std::max(0, rows[i])) * p.StorageLd(i);
        if (input[i].empty())
            input[i].resize(std::max<size_t>(1, storage), PADDING_SENTINEL);
        for (int row = 0; row < rows[i]; ++row) {
            for (int col = std::max(0, cols[i]); col < p.StorageLd(i); ++col)
                input[i][static_cast<size_t>(row) * p.StorageLd(i) + col] = PADDING_SENTINEL;
        }
    }
    return input;
}

struct Outcome {
    aclblasStatus_t status = ACLBLAS_STATUS_INTERNAL_ERROR;
    Metrics metrics;
    size_t workspaceAllocated = 0, workspaceUsed = 0, workspaceGuardBytes = 0;
    uint32_t algorithm = 0x22000000U;
    bool hasInputFingerprints = false;
    size_t dataBytes = 0;
    size_t sampledCalls = 0;
    std::array<std::string, 4> inputFnv64;
    std::array<std::vector<uint32_t>, 4> inputSampleBits;
};

inline std::string InputFingerprint(const std::vector<float>& values)
{
    uint64_t hash = 14695981039346656037ULL;
    for (float value : values) {
        const uint32_t bits = FloatBits(value);
        for (uint32_t shift = 0; shift < 32U; shift += 8U) {
            hash ^= (bits >> shift) & 255U;
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << hash;
    return text.str();
}

inline size_t ExpectedH10Capacity(int m, int n, int k)
{
    const auto aligned = [](int x) -> __uint128_t { return (static_cast<__uint128_t>(x) + 15U) / 16U * 16U; };
    const __uint128_t rows = aligned(m), cols = aligned(n), reduction = aligned(k);
    const __uint128_t bytes = (rows * reduction + cols * reduction + rows * cols) * 4U + 496U;
    if (bytes > std::numeric_limits<size_t>::max())
        throw std::runtime_error("H10 test capacity overflow");
    return static_cast<size_t>(bytes);
}

inline size_t ExpectedH4Capacity(int m, int n, int k)
{
    return size_t(m) * ((size_t(n) + 15U) / 16U * 16U) * 4U +
           ((size_t(k) + 15U) / 16U * 16U) * ((size_t(m) + 7U) / 8U * 8U) * 4U + 16U;
}

inline size_t ExpectedH2Capacity(int m, int n, int k, bool ta, bool tb)
{
    const auto rounded = [](int value, unsigned alignment) -> __uint128_t {
        if (alignment == 0U) {
            throw std::invalid_argument("H2 test alignment must be nonzero");
        }
        return (static_cast<__uint128_t>(value) + alignment - 1U) / alignment * alignment;
    };
    const __uint128_t product = static_cast<__uint128_t>(m) * rounded(n, 16U) * 4U;
    const __uint128_t a = rounded(ta ? k : m, 16U) * rounded(ta ? m : k, 8U) * 4U;
    const __uint128_t b = rounded(tb ? n : k, 16U) * rounded(tb ? k : n, 8U) * 4U;
    const __uint128_t total = product + a + b + 16U;
    if (total > std::numeric_limits<size_t>::max())
        throw std::runtime_error("H2 test capacity overflow");
    return static_cast<size_t>(total);
}

class Invocation {
public:
    Invocation(const Param& p, aclblasLtHandle_t handle, aclrtStream stream) : p_(p), handle_(handle), stream_(stream)
    {
        RecordParameterMetadata(p, "invocation_setup");
        input_ = Inputs(p);
        descriptors_.Initialize(p);
        const char* fingerprints = std::getenv("LT_MATMUL_INPUT_FINGERPRINTS");
        result_.hasInputFingerprints =
            p.recordInputFingerprints || (fingerprints != nullptr && std::string(fingerprints) == "1");
        if (result_.hasInputFingerprints) {
            for (size_t i = 0; i < input_.size(); ++i) {
                const auto& values = input_[i == 2 && p.inplace ? 3 : i];
                result_.inputFnv64[i] = values.empty() ? "NULL" : InputFingerprint(values);
                for (size_t j = 0; j < std::min<size_t>(8, values.size()); ++j) {
                    result_.inputSampleBits[i].push_back(FloatBits(values[j]));
                }
            }
        }
        for (size_t i = 0; i < buffers_.size(); ++i) {
            buffers_[i].Initialize(input_[i].size() * sizeof(float), input_[i].data());
            result_.dataBytes += input_[i].size() * sizeof(float);
        }
        pointers_ = {
            buffers_[0].Data(), buffers_[1].Data(), p.inplace ? buffers_[3].Data() : buffers_[2].Data(),
            buffers_[3].Data()};
        if (p.algoMode == "heuristic" || p.algoMode == "explicit_h2" || p.algoMode == "explicit_h9" ||
            p.algoMode == "explicit_h5")
            PrepareHeuristic();
    }

    struct HeuristicQuery {
        Descriptors clean;
        const Descriptors* query = nullptr;
        int queryM = 0, queryN = 0, queryK = 0;
        bool queryTA = false, queryTB = false;
    };

    void PrepareQuery(HeuristicQuery& q)
    {
        // Invalid descriptors and no-op rows must still reach the Matmul API. Obtain an
        // ordinary supported algorithm from q.clean descriptors in those scenarios.
        q.query = &descriptors_;
        q.queryM = p_.m;
        q.queryN = p_.n;
        q.queryK = p_.k;
        q.queryTA = p_.transA != ACLBLAS_OP_N;
        q.queryTB = p_.transB != ACLBLAS_OP_N;
        if ((p_.expectResult != ACLBLAS_STATUS_SUCCESS && !p_.misalignWorkspace && !p_.forgeWorkspaceCapacity) ||
            p_.m == 0 || p_.n == 0) {
            auto sane = Supplemental("HeuristicSetup", 16, 16, 16);
            sane.transA = sane.transB = ACLBLAS_OP_N;
            sane.preferenceBytes = p_.preferenceBytes;
            q.clean.Initialize(sane);
            q.query = &q.clean;
            q.queryM = sane.m;
            q.queryN = sane.n;
            q.queryK = sane.k;
            q.queryTA = q.queryTB = false;
        }
    }

    void InitExplicitHeuristic(uint32_t id, size_t bytes, aclblasLtMatmulHeuristicResult_t& heuristic)
    {
        BlasCheck(
            aclblasLtMatmulAlgoInit(
                handle_, static_cast<aclblasComputeType_t>(p_.computeType), p_.scaleType, p_.dtype[0], p_.dtype[1],
                p_.dtype[2], p_.dtype[3], id, &heuristic.algo),
            "AlgoInit explicit");
        heuristic.workspaceSize = bytes;
        heuristic.algo.max_workspace_bytes = bytes;
        heuristic.state = ACLBLAS_STATUS_SUCCESS;
    }

    void QueryHeuristic(HeuristicQuery& q, aclblasLtMatmulHeuristicResult_t& heuristic, int& count)
    {
        if (p_.algoMode == "explicit_h9") {
            InitExplicitHeuristic(
                0x22000009,
                p_.n < 16 ?
                    0U :
                    static_cast<size_t>(p_.m) * ((static_cast<uint64_t>(p_.n) + 15U) / 16U * 16U) * sizeof(float),
                heuristic);
            q.queryM = p_.m;
            q.queryN = p_.n;
            q.queryK = p_.k;
            q.queryTA = p_.transA != ACLBLAS_OP_N;
            q.queryTB = p_.transB != ACLBLAS_OP_N;
            count = 1;
        } else if (p_.algoMode == "explicit_h5") {
            InitExplicitHeuristic(
                0x22000005,
                static_cast<size_t>(p_.m) * ((static_cast<uint64_t>(p_.n) + 15U) / 16U * 16U) * sizeof(float),
                heuristic);
            q.queryM = p_.m;
            q.queryN = p_.n;
            q.queryK = p_.k;
            count = 1;
        } else if (p_.algoMode == "explicit_h2") {
            InitExplicitHeuristic(
                0x22000002, ExpectedH2Capacity(p_.m, p_.n, p_.k, p_.transA != ACLBLAS_OP_N, p_.transB != ACLBLAS_OP_N),
                heuristic);
            q.queryM = p_.m;
            q.queryN = p_.n;
            q.queryK = p_.k;
            q.queryTA = p_.transA != ACLBLAS_OP_N;
            q.queryTB = p_.transB != ACLBLAS_OP_N;
            count = 1;
        } else {
            BlasCheck(
                aclblasLtMatmulAlgoGetHeuristic(
                    handle_, q.query->compute, q.query->layout[0], q.query->layout[1], q.query->layout[2],
                    q.query->layout[3], q.query->preference, 1, &heuristic, &count),
                "AlgoGetHeuristic");
        }
    }

    size_t QueryCapacity(const HeuristicQuery& q) const
    {
        return result_.algorithm == 0x2200000aU ?
                   ExpectedH10Capacity(q.queryM, q.queryN, q.queryK) :
               result_.algorithm == 0x22000004U ?
                   ExpectedH4Capacity(q.queryM, q.queryN, q.queryK) :
               result_.algorithm == 0x22000003U ?
                   static_cast<size_t>(q.queryM) * ((static_cast<uint64_t>(q.queryN) + 15U) / 16U * 16U) * 16U + 16U :
               result_.algorithm == 0x22000002U ?
                   ExpectedH2Capacity(q.queryM, q.queryN, q.queryK, q.queryTA, q.queryTB) :
               (result_.algorithm == 0x22000001U || result_.algorithm == 0x22000009U ||
                result_.algorithm == 0x22000005U || result_.algorithm == 0x2200000bU ||
                result_.algorithm == 0x2200000cU) ?
                   static_cast<size_t>(q.queryM) * ((static_cast<uint64_t>(q.queryN) + 15U) / 16U * 16U) *
                       sizeof(float) :
                   0;
    }

    bool TinyOrderedProduct() const
    {
        return p_.m > 0 && p_.n > 0 && p_.n <= 16 && p_.k > 0 &&
               static_cast<uint64_t>(p_.m) * p_.n <= 8192U / static_cast<uint32_t>(p_.k);
    }

    bool LongNZBeta() const
    {
        return result_.algorithm == 0x22000002U && p_.transA != ACLBLAS_OP_N && p_.transB == ACLBLAS_OP_N &&
               p_.m >= 1024 && p_.m <= 2048 && p_.n > 128 && p_.n <= 256 && p_.k >= 512 && p_.k <= 2048 &&
               p_.beta != 0.0f;
    }

    bool OrderedWorkspace() const
    {
        return (result_.algorithm == 0x2200000aU && (p_.alpha != 1.0f || p_.beta != 0.0f)) || TinyOrderedProduct() ||
               p_.k > 4096 || !std::isfinite(p_.alpha) || !std::isfinite(p_.beta) ||
               (std::abs(p_.alpha) > 16.0f && p_.k >= 256) || LongNZBeta();
    }

    void RecordNZWorkspace(size_t expectedCapacity, bool ordered)
    {
        if (result_.algorithm == 0x22000002U && p_.m > 0 && p_.n >= 16 && p_.k > 0 && p_.alpha != 0.0f && !ordered) {
            const size_t productBytes =
                static_cast<size_t>(p_.m) * ((static_cast<uint64_t>(p_.n) + 15U) / 16U * 16U) * 4U;
            result_.workspaceUsed = expectedCapacity - 16U - (p_.alpha == 1.0f && p_.beta == 0.0f ? productBytes : 0U);
        }
    }

    void RecordPackedWorkspace(size_t expectedCapacity, bool ordered)
    {
        if (result_.algorithm == 0x2200000aU && p_.alpha != 0.0f && p_.k > 0 && !ordered) {
            result_.workspaceUsed = expectedCapacity - 496U;
        }
        if (result_.algorithm == 0x22000004U && p_.alpha != 0.0f && !ordered) {
            const size_t productBytes = size_t(p_.m) * ((size_t(p_.n) + 15U) / 16U * 16U) * 4U;
            result_.workspaceUsed = expectedCapacity - 16U - (p_.alpha == 1.0f && p_.beta == 0.0f ? productBytes : 0U);
        }
        if (result_.algorithm == 0x22000003U && p_.alpha != 0.0f && !ordered) {
            result_.workspaceUsed = expectedCapacity - 16U;
        }
        RecordNZWorkspace(expectedCapacity, ordered);
    }

    void RecordWorkspaceUsage(size_t expectedCapacity)
    {
        const bool ordered = OrderedWorkspace();
        if ((result_.algorithm == 0x22000001U || result_.algorithm == 0x22000009U || result_.algorithm == 0x22000005U ||
             result_.algorithm == 0x2200000bU || result_.algorithm == 0x2200000cU) &&
            p_.m > 0 && p_.n >= 16 && p_.k > 0 && p_.alpha != 0.0f && !ordered &&
            !(p_.alpha == 1.0f && p_.beta == 0.0f)) {
            result_.workspaceUsed =
                static_cast<size_t>(p_.m) * ((static_cast<uint64_t>(p_.n) + 15U) / 16U * 16U) * sizeof(float);
        }
        RecordPackedWorkspace(expectedCapacity, ordered);
    }

    void AllocateHeuristicWorkspace(const aclblasLtMatmulHeuristicResult_t& heuristic)
    {
        size_t allocation =
            p_.workspaceBytes == "auto" ? heuristic.workspaceSize : static_cast<size_t>(std::stoull(p_.workspaceBytes));
        if (p_.forgeWorkspaceCapacity)
            algorithm_.max_workspace_bytes = 0;
        workspace_.Initialize(allocation + (p_.misalignWorkspace ? 1 : 0));
        workspacePointer_ = workspace_.Data();
        if (p_.misalignWorkspace && workspacePointer_ != nullptr)
            workspacePointer_ = static_cast<uint8_t*>(workspacePointer_) + 1;
        workspaceBytes_ = allocation;
        result_.workspaceAllocated = allocation;
        result_.workspaceGuardBytes = allocation == 0 ? 0 : 2 * GUARD_BYTES;
    }

    void PrepareHeuristic()
    {
        HeuristicQuery q;
        PrepareQuery(q);
        aclblasLtMatmulHeuristicResult_t heuristic{};
        int count = 0;
        QueryHeuristic(q, heuristic, count);
        if (count != 1 || heuristic.state != ACLBLAS_STATUS_SUCCESS)
            throw std::runtime_error("No usable heuristic result");
        algorithm_ = heuristic.algo;
        PackedAlgo packed{};
        CopyBytes(&packed, sizeof(packed), algorithm_.data, sizeof(packed));
        result_.algorithm = packed.algoId;
        const size_t expectedCapacity = QueryCapacity(q);
        if (heuristic.workspaceSize != expectedCapacity || heuristic.algo.max_workspace_bytes != expectedCapacity) {
            throw std::runtime_error("Algorithm capacity differs from the independent workspace contract");
        }
        AllocateHeuristicWorkspace(heuristic);
        RecordWorkspaceUsage(expectedCapacity);
    }

    aclblasStatus_t Launch()
    {
        auto d = descriptors_.layout;
        if (p_.nullDescriptor == "Adesc")
            d[0] = nullptr;
        if (p_.nullDescriptor == "Bdesc")
            d[1] = nullptr;
        if (p_.nullDescriptor == "Cdesc")
            d[2] = nullptr;
        if (p_.nullDescriptor == "Ddesc")
            d[3] = nullptr;
        const bool nullHandle = p_.description.find("handle_null") != std::string::npos;
        result_.status = aclblasLtMatmul(
            nullHandle ? nullptr : handle_, p_.nullDescriptor == "computeDesc" ? nullptr : descriptors_.compute,
            p_.alphaNull ? nullptr : &p_.alpha, pointers_[0], d[0], pointers_[1], d[1],
            p_.betaNull ? nullptr : &p_.beta, pointers_[2], d[2], pointers_[3], d[3],
            p_.algoMode == "nullptr" ? nullptr : &algorithm_, workspacePointer_, workspaceBytes_,
            p_.invalidStream ? reinterpret_cast<aclrtStream>(uintptr_t(1)) : stream_);
        return result_.status;
    }

    // Deliberate alias/capsule mutations are used only by white-box negative tests.
    void* Operand(size_t index) const { return pointers_.at(index); }
    void SetOperand(size_t index, void* value) { pointers_.at(index) = value; }
    void SetWorkspace(void* value) { workspacePointer_ = value; }
    aclblasLtMatmulAlgo_t& Algorithm() { return algorithm_; }
    void SetEpilogue(int32_t value)
    {
        BlasCheck(
            aclblasLtMatmulDescSetAttribute(
                descriptors_.compute, ACLBLASLT_MATMUL_DESC_EPILOGUE, &value, sizeof(value)),
            "set test epilogue");
    }

    Outcome Validate(bool benchmark, bool alreadyLaunched = false)
    {
        if (!alreadyLaunched)
            Launch();
        AclCheck(aclrtSynchronizeStream(stream_), "synchronize after Matmul");
        for (const auto& buffer : buffers_)
            buffer.CheckGuards();
        workspace_.CheckGuards();
        if (result_.status == ACLBLAS_STATUS_SUCCESS && p_.expectResult == ACLBLAS_STATUS_SUCCESS) {
            auto golden = Golden(p_, input_);
            auto actual = input_[3];
            buffers_[3].Download(actual);
            result_.metrics = Compare(p_, actual, golden);
            for (int i = 0; i < p_.m; ++i) {
                for (int j = std::max(0, p_.n); j < p_.StorageLd(3); ++j) {
                    if (FloatBits(actual[static_cast<size_t>(i) * p_.StorageLd(3) + j]) !=
                        FloatBits(PADDING_SENTINEL)) {
                        throw std::runtime_error("D row padding overwritten");
                    }
                }
            }
            if (p_.m == 0 || p_.n == 0) {
                if (actual.size() != input_[3].size() ||
                    (!actual.empty() &&
                     std::memcmp(actual.data(), input_[3].data(), actual.size() * sizeof(float)) != 0)) {
                    throw std::runtime_error("no-op changed D storage");
                }
            }
            std::vector<float>().swap(actual);
            std::vector<float>().swap(golden);
            for (size_t i = 0; i < 3; ++i) {
                if (buffers_[i].Bytes() == 0)
                    continue;
                std::vector<float> unchanged(input_[i].size());
                buffers_[i].Download(unchanged);
                if (std::memcmp(unchanged.data(), input_[i].data(), unchanged.size() * sizeof(float)) != 0) {
                    throw std::runtime_error("Read-only input " + std::to_string(i) + " changed");
                }
            }
            if (benchmark && result_.metrics.passed)
                Benchmark();
        }
        return result_;
    }

    void Benchmark()
    {
        if (p_.inplace)
            throw std::runtime_error("Performance CSV must use independent C/D");
        if (result_.workspaceAllocated > static_cast<double>(result_.dataBytes) * 1.2)
            throw std::runtime_error("Workspace exceeds 1.2x operand data bytes");
        std::cout << "LT_MATMUL_PROFILE case=" << p_.caseName << " accuracy_calls=1 warmup_calls=5 sample_calls=60"
                  << std::endl;
        for (int index = -5; index < 60; ++index) {
            BlasCheck(Launch(), "profile Matmul");
            AclCheck(aclrtSynchronizeStream(stream_), "profile synchronize");
            if (index >= 0)
                ++result_.sampledCalls;
            std::cout << "LT_MATMUL_SAMPLE case=" << p_.caseName << " phase=" << (index < 0 ? "warmup" : "sample")
                      << " index=" << (index < 0 ? index + 5 : index) << " status=0" << std::endl;
        }
    }

private:
    const Param& p_;
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    std::array<std::vector<float>, 4> input_;
    Descriptors descriptors_;
    std::array<DeviceBuffer, 4> buffers_;
    std::array<void*, 4> pointers_{};
    aclblasLtMatmulAlgo_t algorithm_{};
    DeviceBuffer workspace_;
    void* workspacePointer_ = nullptr;
    size_t workspaceBytes_ = 0;
    Outcome result_;
};

inline void Record(const Param& p, const Outcome& outcome)
{
    std::ostringstream record;
    record << std::setprecision(17) << "{\"case_id\":\"" << p.caseName
           << "\",\"gtest_name\":" << JsonString(CurrentGtestName()) << ",\"status\":" << outcome.status
           << ",\"expected_status\":" << p.expectResult << ",\"algorithm_id\":" << outcome.algorithm
           << ",\"workspace_allocated_bytes\":" << outcome.workspaceAllocated
           << ",\"workspace_used_bytes\":" << outcome.workspaceUsed
           << ",\"workspace_guard_bytes\":" << outcome.workspaceGuardBytes << ",\"operand_bytes\":" << outcome.dataBytes
           << ",\"matched_count\":" << outcome.metrics.matched << ",\"matched_ratio\":" << outcome.metrics.matchedRatio
           << ",\"max_abs_error\":" << outcome.metrics.maxAbs
           << ",\"max_abs_error_limit\":" << outcome.metrics.maxAbsLimit
           << ",\"finite_count\":" << outcome.metrics.finite << ",\"nonfinite_count\":" << outcome.metrics.nonfinite
           << ",\"nonfinite_matched\":" << outcome.metrics.nonfiniteMatched
           << ",\"exact_mismatches\":" << outcome.metrics.exactMismatches
           << ",\"sampled_api_calls\":" << outcome.sampledCalls
           << ",\"passed\":" << ((outcome.status == p.expectResult && outcome.metrics.passed) ? "true" : "false");
    if (outcome.hasInputFingerprints) {
        record << ",\"random_seed\":" << p.randomSeed
               << ",\"distribution_metadata\":" << std::quoted(p.distributionMetadata)
               << ",\"alpha_bits\":" << FloatBits(p.alpha) << ",\"beta_bits\":" << FloatBits(p.beta)
               << ",\"input_fingerprint_method\":\"FNV-1a-64 non-cryptographic; FP32 bits in little-endian byte order; "
                  "physical storage including padding\""
               << ",\"input_fnv1a64\":[";
        for (size_t i = 0; i < 4; ++i)
            record << (i == 0 ? "" : ",") << std::quoted(outcome.inputFnv64[i]);
        record << "],\"input_sample_bits\":[";
        for (size_t i = 0; i < 4; ++i) {
            record << (i == 0 ? "" : ",") << '[';
            for (size_t j = 0; j < outcome.inputSampleBits[i].size(); ++j) {
                record << (j == 0 ? "" : ",") << outcome.inputSampleBits[i][j];
            }
            record << ']';
        }
        record << ']';
    }
    record << '}';
    std::cout << "LT_MATMUL_RESULT " << record.str() << std::endl;
    if (const char* path = std::getenv("LT_MATMUL_RESULTS")) {
        std::ofstream output(path, std::ios::app);
        if (!output)
            throw std::runtime_error("Cannot append per-case result file");
        output << record.str() << '\n';
        if (!output)
            throw std::runtime_error("Cannot write per-case result file");
    }
}

} // namespace LtMatmul22Test
