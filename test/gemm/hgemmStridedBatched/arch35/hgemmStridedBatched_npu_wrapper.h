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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <gtest/gtest.h>
#include "hgemmStridedBatched_param.h"

namespace hgemm_test {
// Explicit high-bandwidth memory is required for reproducible device timing on
// the mixed-memory 950 platform. Ordinary pages remain a correctness option.
inline aclrtMemMallocPolicy DeviceMemoryPolicy()
{
    static const aclrtMemMallocPolicy policy = [] {
        const char* value = std::getenv("HGEMM_DEVICE_MEMORY");
        const std::string selected = value ? value : "high-bandwidth";
        if (selected == "normal") {
            std::cout << "[MEMORY] ACL_MEM_MALLOC_NORMAL_ONLY\n";
            return ACL_MEM_MALLOC_NORMAL_ONLY;
        }
        if (selected != "high-bandwidth")
            throw std::runtime_error("Unknown HGEMM_DEVICE_MEMORY");
        std::cout << "[MEMORY] ACL_MEM_MALLOC_HUGE_FIRST|ACL_MEM_TYPE_HIGH_BAND_WIDTH\n";
        return static_cast<aclrtMemMallocPolicy>(ACL_MEM_MALLOC_HUGE_FIRST | ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    }();
    return policy;
}

struct DeviceBuffer {
    void* base = nullptr;
    size_t bytes = 0, prefix = 0, allocated = 0, suffix = 0;
    ~DeviceBuffer()
    {
        if (base)
            aclrtFree(base);
    }
    void Upload(const std::vector<Half>& h, size_t offset = 0, bool guarded = false)
    {
        bytes = h.size() * sizeof(Half);
        if (!bytes)
            return;
        prefix = offset;
        std::vector<Half> storage;
        const Half* source = h.data();
        allocated = bytes;
        if (prefix || guarded) {
            suffix = GUARD_ELEMENTS;
            storage.assign(prefix + h.size() + suffix, GUARD_PATTERN);
            std::copy(h.begin(), h.end(), storage.begin() + prefix);
            source = storage.data();
            allocated = storage.size() * sizeof(Half);
        }
        if (aclrtMalloc(&base, allocated, DeviceMemoryPolicy()) != ACL_SUCCESS ||
            aclrtMemcpy(base, allocated, source, allocated, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            throw std::runtime_error("Device buffer upload failed");
    }
    void Download(std::vector<Half>& h)
    {
        if (bytes && aclrtMemcpy(h.data(), bytes, Ptr(), bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
            throw std::runtime_error("Device buffer download failed");
    }
    bool GuardsIntact()
    {
        if (!suffix)
            return true;
        std::vector<Half> all(allocated / sizeof(Half));
        if (aclrtMemcpy(all.data(), allocated, base, allocated, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
            throw std::runtime_error("Cannot check device guards");
        return std::all_of(all.begin(), all.begin() + prefix, [](Half v) { return v == GUARD_PATTERN; }) &&
               std::all_of(all.end() - suffix, all.end(), [](Half v) { return v == GUARD_PATTERN; });
    }
    Half* Ptr() { return base ? static_cast<Half*>(base) + prefix : nullptr; }
};

struct CaseData {
    const HgemmStridedBatchedParam& p;
    bool empty = false, negative = false;
    int ar = 0, ac = 0, br = 0, bc = 0;
    std::vector<Half> a, b, initial, actual;
    DeviceBuffer da, db, dc, workspace;

    explicit CaseData(const HgemmStridedBatchedParam& params) : p(params)
    {
        empty = p.m == 0 || p.n == 0 || p.batches == 0;
        // Invalid cases only need valid small allocations for the pointers that are
        // present; never allocate negative shapes before exercising API validation.
        negative = p.expected != ACLBLAS_STATUS_SUCCESS;
        ar = p.ta == ACLBLAS_OP_N ? p.m : p.k;
        ac = p.ta == ACLBLAS_OP_N ? p.k : p.m;
        br = p.tb == ACLBLAS_OP_N ? p.k : p.n;
        bc = p.tb == ACLBLAS_OP_N ? p.n : p.k;
        a = Input(negative ? 1 : Span(ar, ac, p.lda, p.sa, p.batches), p.af, p.seed);
        b = Input(negative ? 1 : Span(br, bc, p.ldb, p.sb, p.batches), p.bf, p.seed + B_SEED_OFFSET);
        initial = Input(negative ? 1 : Span(p.m, p.n, p.ldc, p.sc, p.batches), p.cf, p.seed + C_SEED_OFFSET);
        Restore();
        actual = initial;
    }

    void Restore()
    {
        if (const char* root = std::getenv("HGEMM_INPUT_DIR")) {
            if (!negative && !empty && !initial.empty()) {
                const auto directory = std::filesystem::path(root) / p.caseName;
                const auto restore = [&](const char* name, std::vector<Half>& data) {
                    std::ifstream file(std::filesystem::path(directory).append(name), std::ios::binary | std::ios::ate);
                    const size_t bytes = data.size() * sizeof(Half);
                    if (!file || file.tellg() != static_cast<std::streamoff>(bytes))
                        throw std::runtime_error("Missing or wrong-sized actual input attachment");
                    file.seekg(0);
                    file.read(reinterpret_cast<char*>(data.data()), bytes);
                    if (!file)
                        throw std::runtime_error("Cannot read actual input attachment");
                };
                restore("A.f16.bin", a);
                restore("B.f16.bin", b);
                restore("C_input.f16.bin", initial);
            }
        }
    }

    void Upload(aclblasHandle_t handle)
    {
        da.Upload(a);
        db.Upload(b);
        dc.Upload(initial, p.cOffset, true);
        if (p.workspaceBytes) {
            workspace.Upload(
                std::vector<Half>((p.workspaceBytes + p.workspaceOffset + sizeof(Half) - 1) / sizeof(Half)));
            ASSERT_EQ(
                aclblasSetWorkspace(
                    handle, reinterpret_cast<uint8_t*>(workspace.base) + p.workspaceOffset, p.workspaceBytes),
                ACLBLAS_STATUS_SUCCESS);
        }
    }

    aclblasStatus_t Call(aclblasHandle_t handle)
    {
        return aclblasHgemmStridedBatched(
            p.expected == ACLBLAS_STATUS_HANDLE_IS_NULLPTR ? nullptr : handle, p.ta, p.tb, p.m, p.n, p.k,
            p.alphaNull ? nullptr : &p.alpha, da.Ptr(), p.lda, p.sa, db.Ptr(), p.ldb, p.sb,
            p.betaNull ? nullptr : &p.beta, dc.Ptr(), p.ldc, p.sc, p.batches);
    }

    void Dump()
    {
        if (const char* root = std::getenv("HGEMM_DUMP_DIR")) {
            const auto directory = std::filesystem::path(root) / p.caseName;
            std::filesystem::create_directories(directory);
            const auto dump = [&](const char* name, const std::vector<Half>& data) {
                std::ofstream file(std::filesystem::path(directory).append(name), std::ios::binary);
                file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(Half));
                if (!file)
                    throw std::runtime_error("Cannot write actual input/output attachment");
            };
            dump("A.f16.bin", a);
            dump("B.f16.bin", b);
            dump("C_input.f16.bin", initial);
            dump("C_actual.f16.bin", actual);
        }
    }

    void CheckPadding() const
    {
        // Check all padding and batch gaps, independent of the numerical tolerance.
        if (p.sc >= static_cast<int64_t>(p.ldc) * p.n) {
            for (size_t i = 0; i < actual.size(); ++i) {
                const size_t local = p.sc ? i % p.sc : i;
                if (local >= static_cast<size_t>(p.ldc) * p.n || local % p.ldc >= static_cast<size_t>(p.m)) {
                    ASSERT_EQ(actual[i], initial[i]) << "C padding/gap changed at " << i;
                }
            }
        }
    }
};

} // namespace hgemm_test
