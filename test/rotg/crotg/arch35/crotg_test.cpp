/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "crotg_param.h"
#include "crotg_golden.h"
#include "crotg_npu_wrapper.h"

class CrotgArch35Test : public BlasTest<CrotgParam> {};

// Per-component accuracy verification (complex real/imag and float scalar all judged as FLOAT32)
static void VerifyCrotgResult(float result, float golden, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, caseName));
}

// Check whether a float is inf or nan
static inline bool IsInfOrNan(float v)
{
    return !std::isfinite(v);
}

// Mathematical property verification (orthonormality, rotation zeroing, norm preservation)
// Input: outputs c/s/r and the original inputs a/b
static void VerifyCrotgMathProps(float cr, float sr, float si, float rr, float ri,
                                 float origAr, float origAi, float origBr, float origBi,
                                 const std::string& caseName)
{
    // Skip mathematical property verification for cases with inf/nan
    // (design doc: expect SUCCESS without crash, properties not required)
    if (IsInfOrNan(cr) || IsInfOrNan(sr) || IsInfOrNan(si) || IsInfOrNan(rr) || IsInfOrNan(ri) ||
        IsInfOrNan(origAr) || IsInfOrNan(origAi) || IsInfOrNan(origBr) || IsInfOrNan(origBi)) {
        return;
    }

    const float atol = 1.5259e-5f; // 2^-16
    const float rtol = 9.7656e-4f; // 2^-10

    // Orthonormality: |c^2 + |s|^2 - 1| <= atol + rtol
    float ortho = std::fabs(cr * cr + (sr * sr + si * si) - 1.0f);
    EXPECT_LE(ortho, atol + rtol) << caseName << "_orthonormality";

    // Input norm sqrt(|a|^2+|b|^2), chained hypotf to avoid large-value square overflow
    float inNorm = hypotf(hypotf(origAr, origAi), hypotf(origBr, origBi));

    // Norm preservation: ||r| - sqrt(|a|^2+|b|^2)| <= atol + rtol*inNorm
    float rNorm = hypotf(rr, ri);
    float magErr = fabsf(rNorm - inNorm);
    EXPECT_LE(magErr, atol + rtol * inNorm) << caseName << "_magnitude";

    // Rotation zeroing 1: c*a + s*b - r ~= 0
    float caR = cr * origAr;
    float caI = cr * origAi;
    float sbR = sr * origBr - si * origBi;
    float sbI = sr * origBi + si * origBr;
    float rot1R = caR + sbR - rr;
    float rot1I = caI + sbI - ri;
    float rot1Err = hypotf(rot1R, rot1I);
    EXPECT_LE(rot1Err, atol + rtol * inNorm) << caseName << "_rotation1";

    // Rotation zeroing 2: -conjg(s)*a + c*b ~= 0
    float nsR = -sr;
    float nsI = si;
    float nsaR = nsR * origAr - nsI * origAi;
    float nsaI = nsR * origAi + nsI * origAr;
    float cbR = cr * origBr;
    float cbI = cr * origBi;
    float rot2R = nsaR + cbR;
    float rot2I = nsaI + cbI;
    float rot2Err = hypotf(rot2R, rot2I);
    EXPECT_LE(rot2Err, atol + rtol * inNorm) << caseName << "_rotation2";
}

// --- Parameter validation tests (standalone TEST_F, no CSV) ---

TEST_F(CrotgArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasCrotg_npu(nullptr, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CrotgArch35Test, NullA)
{
    aclblasComplex b{2.0f, 0.0f};
    float c = 0.0f;
    aclblasComplex s{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCrotg_npu(CrotgArch35Test::handle_, nullptr, &b, &c, &s);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CrotgArch35Test, NullB)
{
    aclblasComplex a{1.0f, 0.0f};
    float c = 0.0f;
    aclblasComplex s{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCrotg_npu(CrotgArch35Test::handle_, &a, nullptr, &c, &s);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CrotgArch35Test, NullC)
{
    aclblasComplex a{1.0f, 0.0f};
    aclblasComplex b{2.0f, 0.0f};
    aclblasComplex s{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCrotg_npu(CrotgArch35Test::handle_, &a, &b, nullptr, &s);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CrotgArch35Test, NullS)
{
    aclblasComplex a{1.0f, 0.0f};
    aclblasComplex b{2.0f, 0.0f};
    float c = 0.0f;
    aclblasStatus_t ret = aclblasCrotg_npu(CrotgArch35Test::handle_, &a, &b, &c, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// Mixed host/device pointers must be rejected (a is device, b/c/s are host)
TEST_F(CrotgArch35Test, MixedHostDevicePointers)
{
    aclblasComplex bHost{2.0f, 0.0f};
    float cHost = 0.0f;
    aclblasComplex sHost{0.0f, 0.0f};

    void* aDev = nullptr;
    ASSERT_EQ(aclrtMalloc(&aDev, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);

    aclblasStatus_t ret = aclblasCrotg(
        CrotgArch35Test::handle_, static_cast<aclblasComplex*>(aDev), &bHost, &cHost, &sHost);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);

    if (aDev) aclrtFree(aDev);
}

// --- CSV-driven functional tests ---

INSTANTIATE_TEST_SUITE_P(
    Crotg, CrotgArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<CrotgParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CrotgParam>);

// CSV-driven: all-device path (via NPU wrapper) - accuracy verification
TEST_P(CrotgArch35Test, CsvDrivenDevice)
{
    const auto& p = GetParam();

    // Prepare host data (handle nullptr negative cases)
    aclblasComplex aHost = p.a;
    aclblasComplex bHost = p.b;
    float cHost = 0.0f;
    aclblasComplex sHost{0.0f, 0.0f};

    aclblasComplex* aPtr = p.aIsNull ? nullptr : &aHost;
    aclblasComplex* bPtr = p.bIsNull ? nullptr : &bHost;
    float* cPtr = p.cIsNull ? nullptr : &cHost;
    aclblasComplex* sPtr = p.sIsNull ? nullptr : &sHost;

    aclblasStatus_t ret = aclblasCrotg_npu(CrotgArch35Test::handle_, aPtr, bPtr, cPtr, sPtr);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // Compute CPU golden
    aclblasComplex goldenA = p.a;
    aclblasComplex goldenB = p.b;
    float goldenC = 0.0f;
    aclblasComplex goldenS{0.0f, 0.0f};
    aclblasCrotg_cpu(CrotgArch35Test::handle_, &goldenA, &goldenB, &goldenC, &goldenS);

    // Verify each output scalar: r (a overwritten), c, s real/imag
    VerifyCrotgResult(aHost.real, goldenA.real, p.caseName + "_r_real");
    VerifyCrotgResult(aHost.imag, goldenA.imag, p.caseName + "_r_imag");
    VerifyCrotgResult(cHost, goldenC, p.caseName + "_c");
    VerifyCrotgResult(sHost.real, goldenS.real, p.caseName + "_s_real");
    VerifyCrotgResult(sHost.imag, goldenS.imag, p.caseName + "_s_imag");

    // Mathematical property verification (substitute original inputs)
    VerifyCrotgMathProps(cHost, sHost.real, sHost.imag, aHost.real, aHost.imag,
                         p.a.real, p.a.imag, p.b.real, p.b.imag, p.caseName + "_dev");
}

// CSV-driven: all-host path (aclrtMallocHost pinned memory, direct call) - accuracy verification
TEST_P(CrotgArch35Test, CsvDrivenHost)
{
    const auto& p = GetParam();

    aclblasComplex *aMem = nullptr;
    aclblasComplex *bMem = nullptr;
    float *cMem = nullptr;
    aclblasComplex *sMem = nullptr;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&aMem), sizeof(aclblasComplex)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&bMem), sizeof(aclblasComplex)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&cMem), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&sMem), sizeof(aclblasComplex)), ACL_SUCCESS);

    *aMem = p.a;
    *bMem = p.b;

    // Negative cases: pass nullptr for the corresponding field
    aclblasComplex* a = p.aIsNull ? nullptr : aMem;
    aclblasComplex* b = p.bIsNull ? nullptr : bMem;
    float* c = p.cIsNull ? nullptr : cMem;
    aclblasComplex* s = p.sIsNull ? nullptr : sMem;

    // Golden (computed only for success cases)
    aclblasComplex goldenA = p.a;
    aclblasComplex goldenB = p.b;
    float goldenC = 0.0f;
    aclblasComplex goldenS{0.0f, 0.0f};
    bool allValid = !p.aIsNull && !p.bIsNull && !p.cIsNull && !p.sIsNull;
    if (allValid) {
        EXPECT_EQ(aclblasCrotg_cpu(CrotgArch35Test::handle_, &goldenA, &goldenB, &goldenC, &goldenS),
                  ACLBLAS_STATUS_SUCCESS);
    }

    aclblasStatus_t ret = aclblasCrotg(CrotgArch35Test::handle_, a, b, c, s);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));

    if (ret == ACLBLAS_STATUS_SUCCESS && allValid) {
        VerifyCrotgResult(a->real, goldenA.real, p.caseName + "_host_r_real");
        VerifyCrotgResult(a->imag, goldenA.imag, p.caseName + "_host_r_imag");
        VerifyCrotgResult(*c, goldenC, p.caseName + "_host_c");
        VerifyCrotgResult(s->real, goldenS.real, p.caseName + "_host_s_real");
        VerifyCrotgResult(s->imag, goldenS.imag, p.caseName + "_host_s_imag");

        VerifyCrotgMathProps(*c, s->real, s->imag, a->real, a->imag,
                             p.a.real, p.a.imag, p.b.real, p.b.imag, p.caseName + "_host");
    }

    if (aMem) aclrtFreeHost(aMem);
    if (bMem) aclrtFreeHost(bMem);
    if (cMem) aclrtFreeHost(cMem);
    if (sMem) aclrtFreeHost(sMem);
}

// RAII device buffers for crotg perf test (4 scalars: a, b, c, s)
struct CrotgDevBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    void* s = nullptr;
    bool Init(size_t cBytes, size_t fBytes)
    {
        return aclrtMalloc(&a, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
            && aclrtMalloc(&b, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
            && aclrtMalloc(&c, fBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS
            && aclrtMalloc(&s, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    ~CrotgDevBuffers()
    {
        if (a) aclrtFree(a);
        if (b) aclrtFree(b);
        if (c) aclrtFree(c);
        if (s) aclrtFree(s);
    }
};

// Batch-launch crotg and measure avg per-call time (ms); resets inputs every 256 launches.
static double CrotgTimedBatchLaunch(void* dA, void* dB, void* dC, void* dS,
                                    const aclblasComplex& aHost, const aclblasComplex& bHost,
                                    size_t cBytes, int nLaunch,
                                    aclblasHandle_t handle, aclrtStream stream)
{
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < nLaunch; ++i) {
        if ((i & 0xFF) == 0 && i > 0) {
            aclrtMemcpy(dA, cBytes, &aHost, cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(dB, cBytes, &bHost, cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        aclblasCrotg(handle, static_cast<aclblasComplex*>(dA), static_cast<aclblasComplex*>(dB),
                     static_cast<float*>(dC), static_cast<aclblasComplex*>(dS));
    }
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::steady_clock::now();
    if (nLaunch > 0) {
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / nLaunch;
    }
    return 0.0;
}

// CSV-driven: performance test (TC_PF cases, high-precision batch-launch timing)
TEST_P(CrotgArch35Test, CsvPerfDevice)
{
    const auto& p = GetParam();
    if (p.caseName.find("TC_PF") == std::string::npos) return;
    if (p.aIsNull || p.bIsNull || p.cIsNull || p.sIsNull) return;

    CrotgDevBuffers buf;
    ASSERT_TRUE(buf.Init(sizeof(aclblasComplex), sizeof(float)));
    const size_t cBytes = sizeof(aclblasComplex);
    aclrtMemcpy(buf.a, cBytes, &p.a, cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.b, cBytes, &p.b, cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclblasCrotg(CrotgArch35Test::handle_, static_cast<aclblasComplex*>(buf.a),
                 static_cast<aclblasComplex*>(buf.b), static_cast<float*>(buf.c),
                 static_cast<aclblasComplex*>(buf.s));
    aclrtSynchronizeStream(CrotgArch35Test::stream_);
    aclrtMemcpy(buf.a, cBytes, &p.a, cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.b, cBytes, &p.b, cBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    int nLaunch = std::max(p.iters, 200);
    double avgMs = CrotgTimedBatchLaunch(buf.a, buf.b, buf.c, buf.s, p.a, p.b, cBytes,
                                         nLaunch, CrotgArch35Test::handle_, CrotgArch35Test::stream_);
    printf("[PERF] %s %.6f\n", p.caseName.c_str(), avgMs);
}
