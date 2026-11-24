/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "blas_test.h"
#include "device.h"
#include "cdgmm_data.h"
#include "cdgmm_golden.h"
#include "cdgmm_param.h"

namespace cdgmm_arch35_test {

constexpr size_t COPY_CHUNK = 131072U;
constexpr size_t ALIGNMENT_OFFSET = sizeof(float);
constexpr uint8_t GUARD_BYTE = 0xA5U;
constexpr int ALIGNMENT_ROWS = 3;
constexpr int ALIGNMENT_COLS = 2;

inline void CheckAcl(aclError status)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error("ACL failure: " + std::to_string(status));
    }
}

inline std::string CsvPath()
{
    const std::string source = __FILE__;
    return source.substr(0, source.rfind('.')) + ".csv";
}

inline bool InputUnchanged(const DeviceBuffer& device, const std::vector<aclblasComplex>& host)
{
    std::vector<aclblasComplex> scratch(std::min(COPY_CHUNK, host.size()));
    for (size_t start = 0; start < host.size(); start += COPY_CHUNK) {
        const size_t count = std::min(COPY_CHUNK, host.size() - start);
        const size_t bytes = count * sizeof(aclblasComplex);
        CheckAcl(aclrtMemcpy(
            scratch.data(), bytes, device.bytePtr() + start * sizeof(aclblasComplex), bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        if (std::memcmp(scratch.data(), host.data() + start, bytes) != 0) {
            return false;
        }
    }
    return true;
}

inline CdgmmArch35Param ManualCase(aclblasSideMode_t mode, int incx)
{
    CaseRow row = {
        {"case_name", "Manual"},
        {"description", "manual alias and stride coverage"},
        {"mode", mode == ACLBLAS_SIDE_LEFT ? "LEFT" : "RIGHT"},
        {"m", "33"},
        {"n", "7"},
        {"incx", std::to_string(incx)},
        {"lda", "36"},
        {"ldc", "36"},
        {"x_fill", "RANDOM_NORM_5_5"},
        {"a_fill", "GAUSS_NORM_5_2"},
        {"nullx", "0"},
        {"nullA", "0"},
        {"nullC", "0"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"mere_threshold", "0.0001220703125"},
        {"mare_multiplier", "10.0"},
        {"random_seed", "137"}};
    return CdgmmArch35Param(row);
}

inline std::vector<uint8_t> MakeGuardedStorage(const std::vector<aclblasComplex>& values)
{
    const size_t bytes = values.size() * sizeof(aclblasComplex);
    const auto* raw = reinterpret_cast<const unsigned char*>(values.data());
    std::vector<uint8_t> storage(bytes + 2U * ALIGNMENT_OFFSET, GUARD_BYTE);
    std::copy_n(raw, bytes, storage.begin() + ALIGNMENT_OFFSET);
    return storage;
}

class CdgmmTest : public BlasTest<CdgmmArch35Param> {
protected:
    void RunCase(const CdgmmArch35Param& p, bool inPlace = false)
    {
        SCOPED_TRACE(p.caseName);
        aclblasComplex placeholder{};
        const auto* aPointer = (p.nullA != 0 || p.aFill == "NULLPTR") ? nullptr : &placeholder;
        const auto* xPointer = (p.nullx != 0 || p.xFill == "NULLPTR") ? nullptr : &placeholder;
        auto* cPointer = p.nullC != 0 ? nullptr : &placeholder;
        // Keep independent dummy addresses; pointer equality is itself observable.
        aclblasComplex outputPlaceholder{};
        if (cPointer != nullptr && !inPlace) {
            cPointer = &outputPlaceholder;
        }
        const auto validation =
            CdgmmValidateParams(handle_, p.mode, p.m, p.n, aPointer, p.lda, xPointer, p.incx, cPointer, p.ldc);
        ASSERT_EQ(validation, p.expected);
        if (validation != ACLBLAS_STATUS_SUCCESS || p.m == 0 || p.n == 0) {
            EXPECT_EQ(
                aclblasCdgmm(handle_, p.mode, p.m, p.n, aPointer, p.lda, xPointer, p.incx, cPointer, p.ldc),
                p.expected);
            return;
        }
        RunDeviceCase(p, inPlace);
    }

    void RunAlignedCase(
        aclblasSideMode_t mode, bool inPlace, const std::vector<aclblasComplex>& a,
        const std::vector<aclblasComplex>& x, const std::vector<uint8_t>& aStorage,
        const std::vector<uint8_t>& xStorage)
    {
        const std::vector<uint8_t> cStorage(aStorage.size(), GUARD_BYTE);
        DeviceBuffer deviceA(aStorage.size());
        DeviceBuffer deviceX(xStorage.size());
        DeviceBuffer deviceC(cStorage.size());
        deviceA.copyFromHost(aStorage.data(), aStorage.size());
        deviceX.copyFromHost(xStorage.data(), xStorage.size());
        deviceC.copyFromHost(cStorage.data(), cStorage.size());
        auto& output = inPlace ? deviceA : deviceC;
        ASSERT_EQ(
            aclblasCdgmm(
                handle_, mode, ALIGNMENT_ROWS, ALIGNMENT_COLS,
                reinterpret_cast<const aclblasComplex*>(deviceA.bytePtr() + ALIGNMENT_OFFSET), ALIGNMENT_ROWS,
                reinterpret_cast<const aclblasComplex*>(deviceX.bytePtr() + ALIGNMENT_OFFSET), 1,
                reinterpret_cast<aclblasComplex*>(output.bytePtr() + ALIGNMENT_OFFSET), ALIGNMENT_ROWS),
            ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(stream_));
        std::vector<aclblasComplex> expected(a.size());
        ASSERT_EQ(
            aclblasCdgmm_cpu(
                handle_, mode, ALIGNMENT_ROWS, ALIGNMENT_COLS, a.data(), ALIGNMENT_ROWS, x.data(), 1, expected.data(),
                ALIGNMENT_ROWS),
            ACLBLAS_STATUS_SUCCESS);

        std::vector<uint8_t> actual(aStorage.size());
        output.copyToHost(actual.data(), actual.size());
        const size_t aBytes = a.size() * sizeof(aclblasComplex);
        EXPECT_EQ(std::memcmp(actual.data() + ALIGNMENT_OFFSET, expected.data(), aBytes), 0);
        for (size_t index = 0; index < ALIGNMENT_OFFSET; ++index) {
            EXPECT_EQ(actual[index], GUARD_BYTE);
            EXPECT_EQ(actual[ALIGNMENT_OFFSET + aBytes + index], GUARD_BYTE);
        }
        deviceX.copyToHost(actual.data(), xStorage.size());
        EXPECT_TRUE(std::equal(xStorage.begin(), xStorage.end(), actual.begin()));
        if (!inPlace) {
            deviceA.copyToHost(actual.data(), aStorage.size());
            EXPECT_EQ(actual, aStorage);
        }
    }

private:
    void RunDeviceCase(const CdgmmArch35Param& p, bool inPlace)
    {
        auto a = MakeMatrix(p.m, p.n, p.lda, p.aFill, p.randomSeed);
        auto x = MakeVector(p.mode == ACLBLAS_SIDE_LEFT ? p.m : p.n, p.incx, p.xFill, p.randomSeed + 1U);
        std::vector<aclblasComplex> golden(static_cast<size_t>(p.ldc) * p.n, SENTINEL);
        DeviceBuffer deviceA(a.size() * sizeof(aclblasComplex));
        DeviceBuffer deviceX(x.size() * sizeof(aclblasComplex));
        deviceA.copyFromHost(a.data(), deviceA.size());
        deviceX.copyFromHost(x.data(), deviceX.size());
        std::unique_ptr<DeviceBuffer> separateC;
        if (!inPlace) {
            separateC = std::make_unique<DeviceBuffer>(golden.size() * sizeof(aclblasComplex));
            separateC->copyFromHost(golden.data(), separateC->size());
        }
        DeviceBuffer& deviceC = inPlace ? deviceA : *separateC;
        ASSERT_EQ(Launch(p, deviceA, deviceX, deviceC), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(stream_));
        if (p.IsPerformance() && std::getenv("CDGMM_VERIFY_PERF") == nullptr) {
            CollectProfile(p, deviceA, deviceX, deviceC);
            return;
        }
        ASSERT_EQ(
            aclblasCdgmm_cpu(handle_, p.mode, p.m, p.n, a.data(), p.lda, x.data(), p.incx, golden.data(), p.ldc),
            ACLBLAS_STATUS_SUCCESS);
        EXPECT_TRUE(InputUnchanged(deviceX, x)) << "x was modified";
        if (!inPlace) {
            EXPECT_TRUE(InputUnchanged(deviceA, a)) << "A or its padding was modified";
        }
        RecordOutput(p, deviceC, golden);

        if (inPlace) {
            deviceA.copyFromHost(a.data(), deviceA.size());
        } else {
            ResetOutput(deviceC, golden.size());
        }
        ASSERT_EQ(Launch(p, deviceA, deviceX, deviceC), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(stream_));
        EXPECT_TRUE(InputUnchanged(deviceX, x)) << "repeat call modified x";
        if (!inPlace) {
            EXPECT_TRUE(InputUnchanged(deviceA, a)) << "repeat call modified A";
        }
        EXPECT_TRUE(InputUnchanged(deviceC, golden)) << "output is not bitwise deterministic";
    }

    aclblasStatus_t Launch(const CdgmmArch35Param& p, const DeviceBuffer& a, const DeviceBuffer& x, DeviceBuffer& c)
    {
        return aclblasCdgmm(
            handle_, p.mode, p.m, p.n, static_cast<const aclblasComplex*>(a.ptr()), p.lda,
            static_cast<const aclblasComplex*>(x.ptr()), p.incx, static_cast<aclblasComplex*>(c.ptr()), p.ldc);
    }

    void CollectProfile(const CdgmmArch35Param& p, const DeviceBuffer& a, const DeviceBuffer& x, DeviceBuffer& c)
    {
        // --launch-count limits captured application launches. msprof op supplies warmup separately.
        constexpr uint32_t PROFILE_SAMPLES = 10U;
        for (uint32_t sample = 1U; sample < PROFILE_SAMPLES; ++sample) {
            ASSERT_EQ(Launch(p, a, x, c), ACLBLAS_STATUS_SUCCESS);
            CheckAcl(aclrtSynchronizeStream(stream_));
        }
        std::cout << "CDGMM_PROFILE case=" << p.caseName << " application_launches=" << PROFILE_SAMPLES << '\n';
    }

    void RecordOutput(const CdgmmArch35Param& p, const DeviceBuffer& output, std::vector<aclblasComplex>& golden)
    {
        // A + golden are the only full-size host matrices. Read actual output
        // in chunks, then retain it in golden for the determinism comparison.
        std::vector<aclblasComplex> scratch(std::min(COPY_CHUNK, golden.size()));
        MatrixStats stats;
        for (size_t start = 0; start < golden.size(); start += COPY_CHUNK) {
            const size_t count = std::min(COPY_CHUNK, golden.size() - start);
            const size_t bytes = count * sizeof(aclblasComplex);
            CheckAcl(aclrtMemcpy(
                scratch.data(), bytes, output.bytePtr() + start * sizeof(aclblasComplex), bytes,
                ACL_MEMCPY_DEVICE_TO_HOST));
            for (size_t index = 0; index < count; ++index) {
                stats.Observe(
                    scratch[index], golden[start + index], (start + index) % p.ldc >= static_cast<size_t>(p.m));
            }
            std::copy_n(scratch.begin(), count, golden.begin() + start);
        }
        stats.real.Print(p.caseName, "real");
        stats.imag.Print(p.caseName, "imag");
        EXPECT_TRUE(stats.Passed()) << "padding_failures=" << stats.paddingFailures << " unwritten=" << stats.unwritten;
    }

    void ResetOutput(DeviceBuffer& output, size_t elements)
    {
        const std::vector<aclblasComplex> scratch(std::min(COPY_CHUNK, elements), SENTINEL);
        for (size_t start = 0; start < elements; start += COPY_CHUNK) {
            const size_t count = std::min(COPY_CHUNK, elements - start);
            const size_t bytes = count * sizeof(aclblasComplex);
            CheckAcl(aclrtMemcpy(
                output.bytePtr() + start * sizeof(aclblasComplex), bytes, scratch.data(), bytes,
                ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }
};

// Keep the registered GoogleTest suite name stable for existing filters and profiles.
using CdgmmArch35Test = CdgmmTest;

INSTANTIATE_TEST_SUITE_P(
    CdgmmArch35, CdgmmArch35Test, ::testing::ValuesIn(LoadCdgmmCases(CsvPath())),
    [](const ::testing::TestParamInfo<CdgmmArch35Param>& info) { return info.param.caseName; });

TEST_P(CdgmmArch35Test, CsvDriven) { RunCase(GetParam()); }

TEST_F(CdgmmArch35Test, NullHandleHasPriority)
{
    EXPECT_EQ(
        aclblasCdgmm(nullptr, static_cast<aclblasSideMode_t>(255), 0, 0, nullptr, 0, nullptr, 0, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CdgmmArch35Test, EmptyMatrixValidationOrder)
{
    EXPECT_EQ(
        aclblasCdgmm(handle_, ACLBLAS_SIDE_LEFT, 0, 5, nullptr, 1, nullptr, -2, nullptr, 1), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasCdgmm(handle_, ACLBLAS_SIDE_RIGHT, 3, 0, nullptr, 0, nullptr, 1, nullptr, 3),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCdgmm(handle_, ACLBLAS_SIDE_LEFT, 0, 5, nullptr, 1, nullptr, 0, nullptr, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCdgmm(handle_, static_cast<aclblasSideMode_t>(255), 0, 0, nullptr, 1, nullptr, 1, nullptr, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CdgmmArch35Test, InPlaceBothModesAndSignedStrides)
{
    for (const auto mode : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
        for (const int incx : {1, 2, -1, -2}) {
            RunCase(ManualCase(mode, incx), true);
        }
    }
}

TEST_F(CdgmmArch35Test, FourByteAlignedPointersPreserveGuards)
{
    const std::vector<aclblasComplex> a = {
        {1.0F, 2.0F}, {-3.0F, 4.0F}, {5.0F, -6.0F}, {7.0F, 8.0F}, {-9.0F, 10.0F}, {11.0F, -12.0F},
    };
    const std::vector<aclblasComplex> x = {
        {2.0F, -1.0F},
        {-3.0F, 2.0F},
        {4.0F, 3.0F},
    };
    const auto aStorage = MakeGuardedStorage(a);
    const auto xStorage = MakeGuardedStorage(x);
    for (const auto mode : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
        for (const bool inPlace : {false, true}) {
            ASSERT_NO_FATAL_FAILURE(RunAlignedCase(mode, inPlace, a, x, aStorage, xStorage));
        }
    }
}

TEST_F(CdgmmArch35Test, MultiTileCompactBothModesAndAliasing)
{
    for (const auto mode : {ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}) {
        for (const int incx : {1, -2}) {
            auto p = ManualCase(mode, incx);
            p.m = 257;
            p.n = 4097;
            p.lda = p.m;
            p.ldc = p.m;
            for (const bool inPlace : {false, true}) {
                RunCase(p, inPlace);
            }
        }
    }
}

TEST_F(CdgmmArch35Test, InPlaceMismatchedLeadingDimensions)
{
    aclblasComplex value{};
    EXPECT_EQ(
        aclblasCdgmm(handle_, ACLBLAS_SIDE_LEFT, 1, 1, &value, 1, &value, 1, &value, 2), ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CdgmmArch35Test, IntMinIncrementWithOneElement)
{
    auto p = ManualCase(ACLBLAS_SIDE_RIGHT, INT_MIN);
    p.m = 1;
    p.n = 1;
    p.lda = 1;
    p.ldc = 1;
    RunCase(p);
}

TEST_F(CdgmmArch35Test, MissingCsvColumnsAreRejected)
{
    const CaseRow emptyRow;
    EXPECT_THROW(CdgmmArch35Param{emptyRow}, std::runtime_error);
}

TEST_F(CdgmmArch35Test, MatrixDataRejectsInvalidDimensions)
{
    EXPECT_THROW(MakeMatrix(-1, 1, 1, "VALUE_NORM_0", 0U), std::invalid_argument);
    EXPECT_THROW(MakeMatrix(1, -1, 1, "VALUE_NORM_0", 0U), std::invalid_argument);
    EXPECT_THROW(MakeMatrix(2, 1, 1, "VALUE_NORM_0", 0U), std::invalid_argument);
}

static CdgmmArch35Param PaddedCase(aclblasSideMode_t mode, int incx, const char* caseName)
{
    auto p = ManualCase(mode, incx);
    p.caseName = caseName;
    p.m = 512;
    p.n = 512;
    p.lda = 515;
    p.ldc = 513;
    p.aFill = "RANDOM_NORM_5_5";
    return p;
}

TEST_F(CdgmmArch35Test, PaddedPerformanceShapesMatchGolden)
{
    RunCase(PaddedCase(ACLBLAS_SIDE_LEFT, 2, "TC_ED_EXTRA_LEFT"));
    RunCase(PaddedCase(ACLBLAS_SIDE_RIGHT, 3, "TC_ED_EXTRA_RIGHT"));
}

TEST_F(CdgmmArch35Test, TC_PF_EXTRA_LEFT) { RunCase(PaddedCase(ACLBLAS_SIDE_LEFT, 2, "TC_PF_EXTRA_LEFT")); }

TEST_F(CdgmmArch35Test, TC_PF_EXTRA_RIGHT) { RunCase(PaddedCase(ACLBLAS_SIDE_RIGHT, 3, "TC_PF_EXTRA_RIGHT")); }

} // namespace cdgmm_arch35_test
