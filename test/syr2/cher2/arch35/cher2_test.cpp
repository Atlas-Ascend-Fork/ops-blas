/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cstring>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <securec.h>
#include <dlfcn.h>
#include "blas_test.h"
#include "cher2_golden.h"

namespace {
void CheckCsvCases(const std::vector<Cher2Param>& cases)
{
    if (cases.size() != 1200) {
        throw std::runtime_error("Cher2 CSV must contain exactly 1200 cases");
    }
    std::unordered_set<std::string> names;
    size_t performance = 0;
    for (const auto& p : cases) {
        if (p.caseName.empty() || !names.insert(p.caseName).second) {
            throw std::runtime_error("Cher2 CSV contains an empty or duplicate case name");
        }
        performance += p.caseName.rfind("TC_PF_", 0) == 0;
    }
    if (performance != 200 || cases.size() - performance != 1000) {
        throw std::runtime_error("Cher2 CSV must contain 1000 accuracy and 200 performance cases");
    }
}

std::vector<Cher2Param> Cases()
{
    const auto raw = GetCasesFromCsv<Cher2Param>(ReplaceFileExtension2Csv(__FILE__));
    CheckCsvCases(raw);
    auto result = raw;
    // Keep the provided 1000 accuracy cases unchanged and pair each with an
    // explicit normal-distribution run. Performance inputs stay unchanged.
    for (auto p : raw) {
        if (p.caseName.find("TC_PF_") == 0) {
            continue;
        }
        p.normal = true;
        p.caseName += "_NORMAL";
        result.push_back(p);
    }
    // Alpha presets in the published CSV are fixed boundary values. Add
    // balanced random alpha cases with independent real/imaginary sampling.
    for (int i = 0; i < 32; ++i) {
        Cher2Param p = raw.front();
        p.caseName = "TC_DIST_" + std::to_string(i);
        p.normal = (i & 1) != 0;
        p.randomSeed = 20261028U + static_cast<uint32_t>(i);
        p.n = (i % 4 == 0) ? 257 : 33;
        p.lda = p.n + 3;
        p.uplo = (i % 4 < 2) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        p.incx = (i % 3 == 0) ? -2 : 1;
        p.incy = (i % 3 == 1) ? -3 : 1;
        std::mt19937 gen(p.randomSeed);
        std::uniform_real_distribution<float> uniform(-5.0f, 5.0f), sigma(0.1f, 2.0f);
        const float mean = uniform(gen), deviation = sigma(gen);
        std::normal_distribution<float> gaussian(mean, deviation);
        p.alpha = p.normal ? aclblasComplex{gaussian(gen), gaussian(gen)} : aclblasComplex{uniform(gen), uniform(gen)};
        result.push_back(p);
    }
    return result;
}

struct DeviceBuffer {
    void* data = nullptr;
    ~DeviceBuffer()
    {
        if (data) {
            aclrtFree(data);
        }
    }
    aclError CopyFrom(const std::vector<aclblasComplex>& host)
    {
        if (host.empty()) {
            return ACL_SUCCESS;
        }
        const size_t bytes = host.size() * sizeof(aclblasComplex);
        auto status = aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        return status == ACL_SUCCESS ? aclrtMemcpy(data, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) : status;
    }
    aclError CopyTo(std::vector<aclblasComplex>& host) const
    {
        if (host.empty()) {
            return ACL_SUCCESS;
        }
        const size_t bytes = host.size() * sizeof(aclblasComplex);
        return aclrtMemcpy(host.data(), bytes, data, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
};

size_t VectorSize(int n, int inc)
{
    return n > 0 && inc != 0 && inc != INT_MIN ? 1ULL + static_cast<uint64_t>(n - 1) * std::abs(inc) : 0;
}
} // namespace

class Cher2Test : public BlasTest<Cher2Param> {};
INSTANTIATE_TEST_SUITE_P(Cher2, Cher2Test, ::testing::ValuesIn(Cases()), PrintCaseInfoString<Cher2Param>);

namespace {
void PrintCase(const Cher2Param& p)
{
    std::printf(
        "[CASE] case=%s uplo=%d n=%d alpha_real=%.9g alpha_imag=%.9g incx=%d incy=%d lda=%d "
        "seed=%u normal=%d x_fill=%s y_fill=%s a_fill=%s expected=%d\n",
        p.caseName.c_str(), static_cast<int>(p.uplo), p.n, p.alpha.real, p.alpha.imag, p.incx, p.incy, p.lda,
        p.randomSeed, p.normal, p.xFill.c_str(), p.yFill.c_str(), p.aFill.c_str(), static_cast<int>(p.expectResult));
}

bool ValidStructure(const Cher2Param& p)
{
    return p.n >= 0 && p.incx != 0 && p.incy != 0 && p.incx != INT_MIN && p.incy != INT_MIN &&
           p.lda >= std::max(1, p.n) && !p.nullAlpha && (p.uplo == ACLBLAS_UPPER || p.uplo == ACLBLAS_LOWER);
}

void InitializeMatrix(const Cher2Param& p, std::vector<aclblasComplex>& a)
{
    // Distinct NaN payloads in untouched storage make accidental reads poison
    // arithmetic and make any write detectable with a byte comparison.
    uint32_t sentinelBits = 0x7fc12345U;
    float sentinel;
    ASSERT_EQ(EOK, memcpy_s(&sentinel, sizeof(sentinel), &sentinelBits, sizeof(sentinelBits)));
    if (!a.empty()) {
        for (int col = 0; col < p.n; ++col) {
            for (int row = 0; row < p.lda; ++row) {
                const size_t index = static_cast<size_t>(col) * p.lda + row;
                const bool selected = row < p.n && (p.uplo == ACLBLAS_UPPER ? row <= col : row >= col);
                if (!selected) {
                    a[index] = {sentinel, -0.0f};
                } else if (row == col) {
                    a[index].imag = 0.0f;
                }
            }
        }
    }
}

struct ProfileMarkers {
    using RangeStart = uint64_t (*)(const char*, void*);
    using RangeEnd = void (*)(uint64_t);
    const bool range = std::getenv("CHER2_PROFILE_RANGE") != nullptr;
    void* handle = nullptr;
    RangeStart start = nullptr;
    RangeEnd end = nullptr;
    ~ProfileMarkers()
    {
        if (handle) {
            dlclose(handle);
        }
    }
    void Init()
    {
        if (range) {
            const char* cann = std::getenv("ASCEND_HOME_PATH");
            ASSERT_NE(nullptr, cann);
            handle = dlopen((std::string(cann) + "/tools/mstx/lib64/libms_tools_ext.so").c_str(), RTLD_NOW);
            ASSERT_NE(nullptr, handle) << dlerror();
            start = reinterpret_cast<RangeStart>(dlsym(handle, "mstxRangeStartA"));
            end = reinterpret_cast<RangeEnd>(dlsym(handle, "mstxRangeEnd"));
            ASSERT_NE(nullptr, start);
            ASSERT_NE(nullptr, end);
        }
    }
};

template <typename Call>
void ProfileCase(
    const Cher2Param& p, aclrtStream stream, const DeviceBuffer& da, const std::vector<aclblasComplex>& original,
    const Call& call)
{
    DeviceBuffer backup;
    ASSERT_EQ(ACL_SUCCESS, backup.CopyFrom(original));
    const size_t bytes = original.size() * sizeof(aclblasComplex);
    ProfileMarkers markers;
    ASSERT_NO_FATAL_FAILURE(markers.Init());
    uint64_t rangeId = 0;
    for (int sample = 0; sample < 65; ++sample) {
        const bool beginRange = markers.range && sample >= 5 && (sample - 5) % 20 == 0;
        const bool endRange = markers.range && sample >= 5 && (sample - 5) % 20 == 19;
        if (!markers.range || sample < 5 || beginRange) {
            ASSERT_EQ(
                ACL_SUCCESS, aclrtMemcpyAsync(da.data, bytes, backup.data, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
            ASSERT_EQ(ACL_SUCCESS, aclrtSynchronizeStream(stream));
        }
        if (beginRange) {
            rangeId = markers.start(p.caseName.c_str(), stream);
        }
        ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, call());
        if (endRange) {
            markers.end(rangeId);
        }
        if (!markers.range || sample < 5 || endRange) {
            ASSERT_EQ(ACL_SUCCESS, aclrtSynchronizeStream(stream));
        }
        std::printf(
            "[PERF_CALL] case=%s sample=%d phase=%s status=0\n", p.caseName.c_str(), sample,
            sample < 5 ? "warmup" : "sample");
    }
}

void VerifyMatrix(
    const Cher2Param& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& golden,
    const std::vector<aclblasComplex>& original)
{
    cher2_test::Accuracy real, imag;
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.lda; ++row) {
            const size_t index = static_cast<size_t>(col) * p.lda + row;
            const bool selected = row < p.n && (p.uplo == ACLBLAS_UPPER ? row <= col : row >= col);
            if (!selected) {
                ASSERT_EQ(0, std::memcmp(&a[index], &original[index], sizeof(aclblasComplex))) << index;
            } else {
                real.Add(a[index].real, golden[index].real);
                imag.Add(a[index].imag, golden[index].imag);
                if (row == col) {
                    ASSERT_EQ(0.0f, a[index].imag);
                }
            }
        }
    }
    real.Print(p.caseName, "real");
    imag.Print(p.caseName, "imag");
    EXPECT_TRUE(real.Pass());
    EXPECT_TRUE(imag.Pass());
    EXPECT_LT(real.Mere(), p.mereThreshold);
    EXPECT_LT(imag.Mere(), p.mereThreshold);
    EXPECT_LT(real.maxRelative, p.mereThreshold * p.mareMultiplier);
    EXPECT_LT(imag.maxRelative, p.mereThreshold * p.mareMultiplier);
}

void VerifyReadOnly(const std::vector<aclblasComplex>& original, const DeviceBuffer& buffer)
{
    auto actual = original;
    ASSERT_EQ(ACL_SUCCESS, buffer.CopyTo(actual));
    EXPECT_EQ(0, std::memcmp(original.data(), actual.data(), original.size() * sizeof(aclblasComplex)));
}

void ExpectInvalidCase(const Cher2Param& p, aclblasHandle_t handle)
{
    EXPECT_EQ(
        p.expectResult,
        aclblasCher2(
            handle, p.uplo, p.n, p.nullAlpha ? nullptr : &p.alpha, nullptr, p.incx, nullptr, p.incy, nullptr, p.lda));
}

bool CheckNoOp(const Cher2Param& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& original)
{
    const bool noOp = p.n == 0 || (p.alpha.real == 0.0f && p.alpha.imag == 0.0f);
    if (noOp && !a.empty()) {
        EXPECT_EQ(0, std::memcmp(a.data(), original.data(), a.size() * sizeof(aclblasComplex)));
    }
    return noOp;
}

void VerifyReference(
    const Cher2Param& p, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y,
    const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& original)
{
    auto golden = original;
    cblas_cher2(
        CblasColMajor, p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, p.n, &p.alpha, x.data(), p.incx, y.data(),
        p.incy, golden.data(), p.lda);
    ASSERT_NO_FATAL_FAILURE(VerifyMatrix(p, a, golden, original));
}
} // namespace

TEST_P(Cher2Test, CsvDriven)
{
    const auto& p = GetParam();
    SCOPED_TRACE(p.caseName);
    PrintCase(p);
    const bool valid = ValidStructure(p);
    if (!valid) {
        ExpectInvalidCase(p, handle_);
        return;
    }
    auto x = cher2_test::MakeInput(VectorSize(p.n, p.incx), p.xFill, p.randomSeed, p.normal);
    auto y = cher2_test::MakeInput(VectorSize(p.n, p.incy), p.yFill, p.randomSeed + 19U, p.normal);
    auto a = cher2_test::MakeInput(static_cast<size_t>(p.lda) * p.n, p.aFill, p.randomSeed + 37U, p.normal);
    ASSERT_NO_FATAL_FAILURE(InitializeMatrix(p, a));
    const auto original = a;
    DeviceBuffer dx, dy, da;
    ASSERT_EQ(ACL_SUCCESS, dx.CopyFrom(x));
    ASSERT_EQ(ACL_SUCCESS, dy.CopyFrom(y));
    ASSERT_EQ(ACL_SUCCESS, da.CopyFrom(a));
    const auto call = [&]() {
        return aclblasCher2(
            handle_, p.uplo, p.n, &p.alpha, static_cast<const aclblasComplex*>(dx.data), p.incx,
            static_cast<const aclblasComplex*>(dy.data), p.incy, static_cast<aclblasComplex*>(da.data), p.lda);
    };
    const bool profiling = std::getenv("CHER2_PROFILE") && p.caseName.find("TC_PF_") == 0;
    if (profiling) {
        ASSERT_NO_FATAL_FAILURE(ProfileCase(p, stream_, da, original, call));
        return;
    }
    ASSERT_EQ(p.expectResult, call());
    ASSERT_EQ(ACL_SUCCESS, aclrtSynchronizeStream(stream_));
    ASSERT_EQ(ACL_SUCCESS, da.CopyTo(a));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (CheckNoOp(p, a, original)) {
        return;
    }
    ASSERT_NO_FATAL_FAILURE(VerifyReference(p, x, y, a, original));
    ASSERT_NO_FATAL_FAILURE(VerifyReadOnly(x, dx));
    ASSERT_NO_FATAL_FAILURE(VerifyReadOnly(y, dy));
}

TEST_F(Cher2Test, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    EXPECT_EQ(
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR,
        aclblasCher2(nullptr, ACLBLAS_UPPER, 0, &alpha, nullptr, 1, nullptr, 1, nullptr, 1));
}
