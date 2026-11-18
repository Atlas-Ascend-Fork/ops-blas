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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "blas_test.h"
#include "device.h"
#include "complex_triangular_case.h"
#include "complex_fill.h"

namespace complex_triangular_test {

constexpr uint64_t GUARD_ELEMENTS = 32U;
constexpr size_t CHUNK_ELEMENTS = 65536U;

inline void CheckAcl(aclError status)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error("ACL error: " + std::to_string(status));
    }
}

template <bool Pack>
inline aclblasStatus_t Invoke(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const void* input, void* output, int lda)
{
    static_assert(!Pack, "This fixture tests Ctpttr unpacked output");
    return aclblasCtpttr(
        handle, uplo, n, static_cast<const aclblasComplex*>(input), static_cast<aclblasComplex*>(output), lda);
}

inline aclblasStatus_t ReferenceStatus(
    aclblasHandle_t handle, const TriangularCase& testCase, const void* source, const void* destination)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (testCase.n < 0 || (testCase.uplo != ACLBLAS_UPPER && testCase.uplo != ACLBLAS_LOWER)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (testCase.lda < std::max(1, testCase.n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (testCase.n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return source == nullptr || destination == nullptr ? ACLBLAS_STATUS_INVALID_VALUE : ACLBLAS_STATUS_SUCCESS;
}

inline bool InputUnchanged(const DeviceBuffer& device, const std::vector<ComplexBits>& input)
{
    std::vector<ComplexBits> scratch(std::min(CHUNK_ELEMENTS, input.size()));
    for (size_t start = 0; start < input.size(); start += CHUNK_ELEMENTS) {
        const size_t count = std::min(CHUNK_ELEMENTS, input.size() - start);
        const size_t bytes = count * sizeof(ComplexBits);
        CheckAcl(aclrtMemcpy(
            scratch.data(), bytes, device.bytePtr() + start * sizeof(ComplexBits), bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        if (std::memcmp(scratch.data(), input.data() + start, bytes) != 0) {
            return false;
        }
    }
    return true;
}

inline void InitializeOutput(DeviceBuffer& device, uint64_t elements, const std::string& fillName, uint32_t seed)
{
    ComplexFill fill(fillName, seed);
    const uint64_t total = elements + GUARD_ELEMENTS;
    std::vector<ComplexBits> scratch(static_cast<size_t>(std::min<uint64_t>(CHUNK_ELEMENTS, total)));
    for (uint64_t start = 0; start < total; start += CHUNK_ELEMENTS) {
        const uint64_t count = std::min<uint64_t>(CHUNK_ELEMENTS, total - start);
        for (uint64_t index = 0; index < count; ++index) {
            const uint64_t position = start + index;
            scratch[index] = position < elements ? fill.At(position) : Guard(position - elements);
        }
        const size_t bytes = static_cast<size_t>(count) * sizeof(ComplexBits);
        CheckAcl(aclrtMemcpy(
            device.bytePtr() + start * sizeof(ComplexBits), bytes, scratch.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
}

template <bool Pack>
inline bool VerifyOutput(
    const DeviceBuffer& device, const std::vector<ComplexBits>& source, const TriangularCase& testCase,
    uint64_t elements, const std::string& fillName, bool print)
{
    GoldenCursor<Pack> reference(source.data(), testCase.n, testCase.lda, testCase.uplo == ACLBLAS_UPPER);
    ComplexFill original(fillName, testCase.seed + 1U);
    const uint64_t total = elements + GUARD_ELEMENTS;
    std::vector<ComplexBits> scratch(static_cast<size_t>(std::min<uint64_t>(CHUNK_ELEMENTS, total)));
    uint64_t realFailures = 0;
    uint64_t imagFailures = 0;
    uint64_t protectedFailures = 0;
    uint64_t checked = 0;
    for (uint64_t start = 0; start < total; start += CHUNK_ELEMENTS) {
        const uint64_t count = std::min<uint64_t>(CHUNK_ELEMENTS, total - start);
        const size_t bytes = static_cast<size_t>(count) * sizeof(ComplexBits);
        CheckAcl(aclrtMemcpy(
            scratch.data(), bytes, device.bytePtr() + start * sizeof(ComplexBits), bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        for (uint64_t index = 0; index < count; ++index) {
            const uint64_t position = start + index;
            const ComplexBits before = position < elements ? original.At(position) : Guard(position - elements);
            const bool selected = position < elements && reference.Selected(position);
            const ComplexBits expected = selected ? reference.Next() : before;
            if (selected) {
                realFailures += scratch[index].real != expected.real ? 1U : 0U;
                imagFailures += scratch[index].imag != expected.imag ? 1U : 0U;
                ++checked;
            } else {
                protectedFailures += std::memcmp(&scratch[index], &before, sizeof(before)) != 0 ? 1U : 0U;
            }
        }
    }
    if (print) {
        std::cout << "TRIANGULAR_BITS case=" << testCase.caseName << " elements=" << checked
                  << " real_mismatches=" << realFailures << " imag_mismatches=" << imagFailures
                  << " protected_mismatches=" << protectedFailures << '\n';
    }
    return realFailures == 0U && imagFailures == 0U && protectedFailures == 0U;
}

template <bool Pack>
class TriangularFixture : public BlasTest<TriangularCase> {
protected:
    void Run(const TriangularCase& testCase, const std::vector<ComplexBits>* custom = nullptr)
    {
        SCOPED_TRACE(testCase.caseName);
        const std::string& inputFill = Pack ? testCase.aFill : testCase.apFill;
        const std::string& outputFill = Pack ? testCase.apFill : testCase.aFill;
        ComplexBits inputDummy{};
        ComplexBits outputDummy{};
        const void* sourcePointer = inputFill == "NULLPTR" ? nullptr : &inputDummy;
        void* destinationPointer = outputFill == "NULLPTR" ? nullptr : &outputDummy;
        const auto expected = ReferenceStatus(this->handle_, testCase, sourcePointer, destinationPointer);
        ASSERT_EQ(expected, testCase.expected);
        if (expected != ACLBLAS_STATUS_SUCCESS || testCase.n == 0) {
            EXPECT_EQ(
                Invoke<Pack>(this->handle_, testCase.uplo, testCase.n, sourcePointer, destinationPointer, testCase.lda),
                testCase.expected);
            return;
        }

        RunDeviceCase(testCase, inputFill, outputFill, custom);
    }

    void RunDeviceCase(
        const TriangularCase& testCase, const std::string& inputFill, const std::string& outputFill,
        const std::vector<ComplexBits>* custom)
    {
        const uint64_t n = static_cast<uint64_t>(testCase.n);
        const uint64_t packed = n * (n + 1U) / 2U;
        const uint64_t matrix = n * static_cast<uint64_t>(testCase.lda);
        const uint64_t inputElements = Pack ? matrix : packed;
        const uint64_t outputElements = Pack ? packed : matrix;
        std::vector<ComplexBits> input(inputElements + GUARD_ELEMENTS);
        ComplexFill fill(inputFill, testCase.seed);
        for (uint64_t index = 0; index < inputElements; ++index) {
            input[index] = custom == nullptr ? fill.At(index) : custom->at(static_cast<size_t>(index));
        }
        for (uint64_t index = 0; index < GUARD_ELEMENTS; ++index) {
            input[inputElements + index] = Guard(index);
        }
        DeviceBuffer deviceInput(input.size() * sizeof(ComplexBits));
        DeviceBuffer deviceOutput((outputElements + GUARD_ELEMENTS) * sizeof(ComplexBits));
        deviceInput.copyFromHost(input.data(), deviceInput.size());
        InitializeOutput(deviceOutput, outputElements, outputFill, testCase.seed + 1U);
        ASSERT_EQ(Launch(testCase, deviceInput, deviceOutput), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(this->stream_));
        if (testCase.Performance() && std::getenv("TRIANGULAR_VERIFY_PERF") == nullptr) {
            CollectProfile(testCase, deviceInput, deviceOutput);
            return;
        }
        EXPECT_TRUE(InputUnchanged(deviceInput, input)) << "input or trailing guard was changed";
        EXPECT_TRUE(VerifyOutput<Pack>(deviceOutput, input, testCase, outputElements, outputFill, true));
        InitializeOutput(deviceOutput, outputElements, outputFill, testCase.seed + 1U);
        ASSERT_EQ(Launch(testCase, deviceInput, deviceOutput), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(this->stream_));
        EXPECT_TRUE(InputUnchanged(deviceInput, input)) << "repeat call changed input or trailing guard";
        EXPECT_TRUE(VerifyOutput<Pack>(deviceOutput, input, testCase, outputElements, outputFill, false));
    }

    aclblasStatus_t Launch(const TriangularCase& testCase, const DeviceBuffer& input, DeviceBuffer& output)
    {
        return Invoke<Pack>(this->handle_, testCase.uplo, testCase.n, input.ptr(), output.ptr(), testCase.lda);
    }

    void CollectProfile(const TriangularCase& testCase, const DeviceBuffer& input, DeviceBuffer& output)
    {
        // The first launch was issued by RunDeviceCase. msprof op supplies warmup separately.
        constexpr uint32_t PROFILE_SAMPLES = 10U;
        for (uint32_t sample = 1U; sample < PROFILE_SAMPLES; ++sample) {
            ASSERT_EQ(Launch(testCase, input, output), ACLBLAS_STATUS_SUCCESS);
            CheckAcl(aclrtSynchronizeStream(this->stream_));
        }
        std::cout << "TRIANGULAR_PROFILE case=" << testCase.caseName << " application_launches=" << PROFILE_SAMPLES
                  << '\n';
    }

    void RunNoOpProtection()
    {
        std::vector<ComplexBits> source(GUARD_ELEMENTS);
        std::vector<ComplexBits> destination(GUARD_ELEMENTS);
        for (size_t index = 0; index < source.size(); ++index) {
            source[index] = Guard(index);
            destination[index] = Guard(index + 7U);
        }
        DeviceBuffer deviceInput(source.size() * sizeof(ComplexBits));
        DeviceBuffer deviceOutput(destination.size() * sizeof(ComplexBits));
        deviceInput.copyFromHost(source.data(), deviceInput.size());
        deviceOutput.copyFromHost(destination.data(), deviceOutput.size());
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (const int lda : {0, 1}) {
                EXPECT_EQ(
                    Invoke<Pack>(this->handle_, uplo, 0, deviceInput.ptr(), deviceOutput.ptr(), lda),
                    lda == 0 ? ACLBLAS_STATUS_INVALID_VALUE : ACLBLAS_STATUS_SUCCESS);
                CheckAcl(aclrtSynchronizeStream(this->stream_));
                EXPECT_TRUE(InputUnchanged(deviceInput, source));
                EXPECT_TRUE(InputUnchanged(deviceOutput, destination));
            }
        }
    }

    // 任务书 §2.4 把 A/AP 指向同一内存定义为调用方前置条件（行为未定义），测试指导
    // 「无法用 CSV 表达的场景」要求以 TEST_F 手工构造。此处只断言调用能正常返回
    // （SUCCESS 或 INVALID_VALUE）、不破坏句柄状态，不校验输出内容。
    void RunOverlap()
    {
        constexpr int n = 5;
        constexpr int lda = 5;
        const uint64_t matrix = static_cast<uint64_t>(lda) * n;
        const uint64_t packed = static_cast<uint64_t>(n) * (n + 1U) / 2U;
        const uint64_t elements = matrix > packed ? matrix : packed;
        std::vector<ComplexBits> init(static_cast<size_t>(elements));
        for (size_t index = 0; index < init.size(); ++index) {
            init[index] = Guard(index);
        }
        DeviceBuffer shared(static_cast<size_t>(elements) * sizeof(ComplexBits));
        shared.copyFromHost(init.data(), shared.size());
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            const aclblasStatus_t status = Invoke<Pack>(this->handle_, uplo, n, shared.ptr(), shared.ptr(), lda);
            CheckAcl(aclrtSynchronizeStream(this->stream_));
            EXPECT_TRUE(status == ACLBLAS_STATUS_SUCCESS || status == ACLBLAS_STATUS_INVALID_VALUE)
                << "overlapping A/AP must still return a valid status, got " << static_cast<int>(status);
        }
        // 未定义行为之后句柄与设备上下文必须仍可用。
        Run(ManualCase(n, lda + 3, ACLBLAS_UPPER));
    }

    static size_t FillComparatorExpected(
        const TriangularCase& testCase, const std::vector<ComplexBits>& source, uint64_t outputElements,
        std::vector<ComplexBits>& expected)
    {
        ComplexFill fill("VALUE_NORM_N999", testCase.seed + 1U);
        GoldenCursor<Pack> reference(source.data(), testCase.n, testCase.lda, testCase.uplo == ACLBLAS_UPPER);
        size_t nanPosition = 0;
        for (size_t index = 0; index < expected.size(); ++index) {
            if (index >= outputElements) {
                expected[index] = Guard(index - outputElements);
                continue;
            }
            expected[index] = fill.At(index);
            if (reference.Selected(index)) {
                expected[index] = reference.Next();
                if (index != 0U) {
                    nanPosition = index;
                }
            }
        }
        return nanPosition;
    }

    void RunComparatorChecks()
    {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            auto testCase = ManualCase(3, 7, uplo);
            testCase.aFill = "VALUE_NORM_N999";
            testCase.apFill = "VALUE_NORM_N999";
            constexpr uint64_t PACKED = 6U;
            constexpr uint64_t MATRIX = 21U;
            const uint64_t outputElements = Pack ? PACKED : MATRIX;
            std::vector<ComplexBits> source(Pack ? MATRIX : PACKED);
            for (size_t index = 0; index < source.size(); ++index) {
                source[index] = Guard(index);
            }
            source[0] = {0x00000000U, 0x80000000U};
            std::vector<ComplexBits> expected(outputElements + GUARD_ELEMENTS);
            const size_t nanPosition = FillComparatorExpected(testCase, source, outputElements, expected);
            DeviceBuffer output(expected.size() * sizeof(ComplexBits));
            auto verify = [&]() {
                return VerifyOutput<Pack>(output, source, testCase, outputElements, "VALUE_NORM_N999", false);
            };
            output.copyFromHost(expected.data(), output.size());
            ASSERT_TRUE(verify());
            auto rejectChange = [&](size_t position, uint32_t realMask, uint32_t imagMask) {
                auto corrupted = expected;
                corrupted[position].real ^= realMask;
                corrupted[position].imag ^= imagMask;
                output.copyFromHost(corrupted.data(), output.size());
                EXPECT_FALSE(verify()) << "corruption escaped at " << position;
            };
            rejectChange(0U, 0x80000000U, 0U);
            rejectChange(0U, 0U, 0x80000000U);
            rejectChange(nanPosition, 1U, 0U);
            rejectChange(nanPosition, 0U, 1U);
            rejectChange(outputElements, 1U, 0U);
            if constexpr (!Pack) {
                rejectChange(static_cast<size_t>(testCase.n), 1U, 0U);
                rejectChange(uplo == ACLBLAS_UPPER ? 1U : static_cast<size_t>(testCase.lda), 0U, 1U);
            }
        }
    }

    void RunPayloads()
    {
        const std::vector<ComplexBits> patterns = {{0x00000000U, 0x80000000U}, {0x7F800000U, 0xFF800000U},
                                                   {0x7FC12345U, 0xFFC54321U}, {0x7F812345U, 0xFF854321U},
                                                   {0x00000001U, 0x80000001U}, {0x7F7FFFFFU, 0xFF7FFFFFU},
                                                   {0x3F800000U, 0xBF800000U}};
        for (const int n : {1, 2, 3, 5, 17}) {
            for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                auto testCase = ManualCase(n, n + 3, uplo);
                const size_t count =
                    Pack ? static_cast<size_t>(testCase.lda) * n : static_cast<size_t>(n) * (n + 1U) / 2U;
                std::vector<ComplexBits> input(count);
                for (size_t index = 0; index < count; ++index) {
                    input[index] = patterns[index % patterns.size()];
                }
                Run(testCase, &input);
            }
        }
    }
};

} // namespace complex_triangular_test
