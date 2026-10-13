/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <securec.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "blas_test.h"
#include "csv_loader.h"
#include "cgetrf_batched_golden.h"
#include "cgetrf_batched_precision.h"
#include "cgetrf_batched_boundary_extra.h"

namespace {
struct BoundaryParam : BlasTestParamBase {
    int n;
    int lda;
    int batchSize;
    int matrixOffset;
    int pointerOffset;
    int pivotOffset;
    int infoOffset;
    bool pivot;
    std::string category;
    explicit BoundaryParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n = std::stoi(ReadMap(m, "n"));
        lda = std::stoi(ReadMap(m, "lda"));
        batchSize = std::stoi(ReadMap(m, "batch_size"));
        matrixOffset = std::stoi(ReadMap(m, "matrix_offset_bytes"));
        pointerOffset = std::stoi(ReadMap(m, "pointer_offset_bytes"));
        pivotOffset = std::stoi(ReadMap(m, "pivot_offset_bytes"));
        infoOffset = std::stoi(ReadMap(m, "info_offset_bytes"));
        pivot = ReadMap(m, "pivot_mode") == "PIVOT";
        category = ReadMap(m, "category");
        if (category != "OFFSET" || n < 1 || lda < n || batchSize < 1 || matrixOffset < 0 || pointerOffset < 0 ||
            pivotOffset < 0 || infoOffset < 0 || matrixOffset % 8 || pointerOffset % 8 || pivotOffset % 4 ||
            infoOffset % 4)
            throw std::invalid_argument("Invalid boundary case");
    }
};

void CheckAcl(aclError status)
{
    if (status != ACL_SUCCESS)
        throw std::runtime_error("ACL boundary operation failed: " + std::to_string(status));
}

// Offsets preserve the element's natural alignment, but need not align to 32 bytes.
class GuardedBuffer {
public:
    GuardedBuffer(size_t bytes, size_t offset, const void* input = nullptr)
        : prefix_(64 + offset), bytes_(bytes), initial_(prefix_ + bytes + 64, 0xa5)
    {
        if (input && memcpy_s(initial_.data() + prefix_, initial_.size() - prefix_, input, bytes_) != EOK)
            throw std::runtime_error("Boundary input copy failed");
        CheckAcl(aclrtMalloc(&base_, initial_.size(), ACL_MEM_MALLOC_HUGE_FIRST));
        try {
            CheckAcl(aclrtMemcpy(base_, initial_.size(), initial_.data(), initial_.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        } catch (...) {
            aclrtFree(base_);
            base_ = nullptr;
            throw;
        }
    }
    ~GuardedBuffer()
    {
        if (base_)
            aclrtFree(base_);
    }
    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;
    void* Data() const { return static_cast<uint8_t*>(base_) + prefix_; }
    bool Read(void* output, size_t outputBytes)
    {
        std::vector<uint8_t> actual(initial_.size());
        CheckAcl(aclrtMemcpy(actual.data(), actual.size(), base_, actual.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        if (memcpy_s(output, outputBytes, actual.data() + prefix_, bytes_) != EOK)
            throw std::runtime_error("Boundary output copy failed");
        return std::equal(actual.begin(), actual.begin() + prefix_, initial_.begin()) &&
               std::equal(actual.begin() + prefix_ + bytes_, actual.end(), initial_.begin() + prefix_ + bytes_);
    }

private:
    void* base_ = nullptr;
    size_t prefix_;
    size_t bytes_;
    std::vector<uint8_t> initial_;
};

std::vector<aclblasComplex> MakeInput(const BoundaryParam& p, int batch)
{
    return CgetrfMakeBoundaryInput<aclblasComplex>(p, batch);
}

struct ComponentResult {
    bool classification = true;
    bool mixed = true;
    bool relative = true;
};

ComponentResult CompareComponent(
    const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name)
{
    ComponentResult result;
    std::vector<float> finiteActual;
    std::vector<float> finiteExpected;
    size_t nanCount = 0;
    size_t positiveInfCount = 0;
    size_t negativeInfCount = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        bool sameClass =
            std::isnan(expected[i]) ? std::isnan(actual[i]) :
            std::isinf(expected[i]) ? std::isinf(actual[i]) && std::signbit(actual[i]) == std::signbit(expected[i]) :
                                      std::isfinite(actual[i]);
        if (!sameClass && result.classification)
            std::cout << "CGETRF_CLASS_MISMATCH," << name << ',' << i << ',' << actual[i] << ',' << expected[i]
                      << std::endl;
        if (std::isnan(expected[i])) {
            ++nanCount;
            result.classification = std::isnan(actual[i]) && result.classification;
        } else if (std::isinf(expected[i])) {
            if (std::signbit(expected[i]))
                ++negativeInfCount;
            else
                ++positiveInfCount;
            result.classification = (std::isinf(actual[i]) && std::signbit(actual[i]) == std::signbit(expected[i])) &&
                                    result.classification;
        } else {
            result.classification = std::isfinite(actual[i]) && result.classification;
            finiteExpected.push_back(expected[i]);
            finiteActual.push_back(actual[i]);
        }
    }
    // A class mismatch is a failure even when the finite subset is empty.
    if (!finiteExpected.empty()) {
        if (std::all_of(finiteActual.begin(), finiteActual.end(), [](float x) { return std::isfinite(x); })) {
            auto verdict =
                CgetrfVerifyComponent(finiteActual.data(), finiteExpected.data(), finiteExpected.size(), name);
            result.mixed = verdict.mixed;
            result.relative = verdict.relative;
        } else {
            result.mixed = result.relative = false;
        }
    }
    std::cout << "CGETRF_BOUNDARY_COMPONENT," << name << ',' << result.classification << ',' << nanCount << ','
              << positiveInfCount << ',' << negativeInfCount << ',' << finiteExpected.size() << ',' << result.mixed
              << ',' << result.relative << std::endl;
    return result;
}

class CgetrfBoundaryTest : public BlasTest<BoundaryParam> {};
INSTANTIATE_TEST_SUITE_P(
    CgetrfBoundary, CgetrfBoundaryTest,
    ::testing::ValuesIn(GetCasesFromCsv<BoundaryParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<BoundaryParam>);

struct BoundaryCaseData {
    std::vector<std::vector<aclblasComplex>> inputs;
    std::vector<std::vector<aclblasComplex>> expected;
    std::vector<std::vector<aclblasComplex>> actual;
    std::vector<std::unique_ptr<GuardedBuffer>> matrices;
    std::vector<aclblasComplex*> pointers;
    std::vector<int> goldenPivot;
    std::vector<int> goldenInfo;
    explicit BoundaryCaseData(const BoundaryParam& p)
        : goldenPivot(static_cast<size_t>(p.n) * p.batchSize, -1), goldenInfo(p.batchSize, -1)
    {}
};

void PrepareBoundaryData(const BoundaryParam& p, BoundaryCaseData& data)
{
    const size_t matrixElements = static_cast<size_t>(p.lda) * p.n;
    for (int b = 0; b < p.batchSize; ++b) {
        data.inputs.push_back(MakeInput(p, b));
        data.expected.push_back(data.inputs.back());
        data.actual.push_back(data.inputs.back());
        FactorizeSingleMatrix(
            data.expected.back().data(), p.n, p.lda, p.pivot ? data.goldenPivot.data() + b * p.n : nullptr,
            data.goldenInfo[b], p.pivot);
        data.matrices.emplace_back(new GuardedBuffer(
            matrixElements * sizeof(aclblasComplex), p.matrixOffset + static_cast<size_t>(b) * 8,
            data.inputs.back().data()));
        data.pointers.push_back(static_cast<aclblasComplex*>(data.matrices.back()->Data()));
        std::cout << "CGETRF_ADDRESS," << p.caseName << ",matrix," << b << ','
                  << reinterpret_cast<uintptr_t>(data.pointers.back()) % 32 << std::endl;
    }
}

bool ReadBoundaryMatrices(const BoundaryParam& p, BoundaryCaseData& data, bool& guards)
{
    bool padding = true;
    for (int b = 0; b < p.batchSize; ++b) {
        guards =
            data.matrices[b]->Read(data.actual[b].data(), data.actual[b].size() * sizeof(aclblasComplex)) && guards;
        for (int j = 0; j < p.n; ++j)
            for (int i = p.n; i < p.lda; ++i) {
                size_t at = i + static_cast<size_t>(j) * p.lda;
                padding =
                    (std::memcmp(&data.actual[b][at], &data.inputs[b][at], sizeof(aclblasComplex)) == 0) && padding;
            }
    }
    return padding;
}

void LogBoundaryMismatches(
    const BoundaryParam& p, const BoundaryCaseData& data, const std::vector<int>& info, const std::vector<int>& pivot)
{
    for (int b = 0; b < p.batchSize; ++b)
        if (info[b] != data.goldenInfo[b])
            std::cout << "CGETRF_BOUNDARY_INFO_MISMATCH," << p.caseName << ',' << b << ',' << info[b] << ','
                      << data.goldenInfo[b] << std::endl;
    if (p.pivot)
        for (size_t k = 0; k < pivot.size(); ++k)
            if (pivot[k] != data.goldenPivot[k]) {
                std::cout << "CGETRF_BOUNDARY_PIVOT_MISMATCH," << p.caseName << ',' << k << ',' << pivot[k] << ','
                          << data.goldenPivot[k] << std::endl;
                break;
            }
}

int CompareBoundaryMatrices(const BoundaryParam& p, const BoundaryCaseData& data, ComponentResult (&components)[2])
{
    int skipped = 0;
    for (int b = 0; b < p.batchSize; ++b) {
        if (!CgetrfShouldCheckLu(data.goldenInfo[b])) {
            ++skipped;
            continue;
        }
        for (int part = 0; part < 2; ++part) {
            std::vector<float> got;
            std::vector<float> want;
            for (int j = 0; j < p.n; ++j)
                for (int i = 0; i < p.n; ++i) {
                    size_t at = i + static_cast<size_t>(j) * p.lda;
                    got.push_back(part ? data.actual[b][at].imag : data.actual[b][at].real);
                    want.push_back(part ? data.expected[b][at].imag : data.expected[b][at].real);
                }
            auto verdict =
                CompareComponent(got, want, p.caseName + "_batch" + std::to_string(b) + (part ? "_imag" : "_real"));
            components[part].classification &= verdict.classification;
            components[part].mixed &= verdict.mixed;
            components[part].relative &= verdict.relative;
        }
    }
    return skipped;
}

void RunBoundaryCase(const BoundaryParam& p, aclblasHandle_t handle, aclrtStream stream)
{
    BoundaryCaseData data(p);
    PrepareBoundaryData(p, data);
    GuardedBuffer pointerArray(data.pointers.size() * sizeof(data.pointers[0]), p.pointerOffset, data.pointers.data());
    GuardedBuffer pivotArray(data.goldenPivot.size() * sizeof(int), p.pivotOffset);
    GuardedBuffer infoArray(data.goldenInfo.size() * sizeof(int), p.infoOffset);
    std::cout << "CGETRF_ADDRESS," << p.caseName << ",arrays," << reinterpret_cast<uintptr_t>(pointerArray.Data()) % 32
              << ',' << reinterpret_cast<uintptr_t>(pivotArray.Data()) % 32 << ','
              << reinterpret_cast<uintptr_t>(infoArray.Data()) % 32 << std::endl;
    auto status = aclblasCgetrfBatched(
        handle, p.n, reinterpret_cast<aclblasComplex* const*>(pointerArray.Data()), p.lda,
        p.pivot ? static_cast<int*>(pivotArray.Data()) : nullptr, static_cast<int*>(infoArray.Data()), p.batchSize);
    if (status != ACLBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Operator launch failed: " + std::to_string(static_cast<int>(status)));
    CheckAcl(aclrtSynchronizeStream(stream));
    std::vector<int> info(p.batchSize);
    std::vector<int> pivot(data.goldenPivot.size());
    std::vector<aclblasComplex*> returnedPointers(data.pointers.size());
    bool guards = infoArray.Read(info.data(), info.size() * sizeof(int));
    guards =
        pointerArray.Read(returnedPointers.data(), returnedPointers.size() * sizeof(returnedPointers[0])) && guards;
    if (p.pivot)
        guards = pivotArray.Read(pivot.data(), pivot.size() * sizeof(int)) && guards;
    const bool padding = ReadBoundaryMatrices(p, data, guards);
    const bool infoPass = info == data.goldenInfo;
    const bool pivotPass = !p.pivot || pivot == data.goldenPivot;
    LogBoundaryMismatches(p, data, info, pivot);
    const bool pointersPass = returnedPointers == data.pointers;
    ComponentResult components[2];
    const int skipped = CompareBoundaryMatrices(p, data, components);
    bool passed = infoPass && pivotPass && guards && padding && pointersPass;
    for (const auto& c : components)
        passed &= c.classification && c.mixed && c.relative;
    std::cout << "CGETRF_BOUNDARY_CHECK," << p.caseName << ',' << passed << ',' << infoPass << ',' << pivotPass << ','
              << guards << ',' << padding << ',' << pointersPass;
    for (const auto& c : components)
        std::cout << ',' << c.classification << ',' << c.mixed << ',' << c.relative;
    std::cout << ',' << skipped << std::endl;
    EXPECT_TRUE(passed) << "Boundary check failed: " << p.caseName;
}

TEST_P(CgetrfBoundaryTest, CsvDriven)
{
    static bool deviceError = false;
    const auto& p = GetParam();
    if (deviceError) {
        std::cout << "CGETRF_BOUNDARY_BLOCKED," << p.caseName << std::endl;
        GTEST_SKIP() << "A preceding device operation failed; no further NPU calls are allowed.";
    }
    try {
        RunBoundaryCase(p, handle_, stream_);
    } catch (const std::exception& error) {
        deviceError = true;
        std::cout << "CGETRF_BOUNDARY_RUNTIME," << p.caseName << ',' << error.what() << std::endl;
        FAIL() << error.what();
    }
}
} // namespace
