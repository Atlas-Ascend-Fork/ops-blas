/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <cblas.h>
#include <dlfcn.h>

namespace OpsBlasTest {

const char* ReferenceBlasLibrary(const char* symbol)
{
    Dl_info info{};
    void* address = dlsym(RTLD_DEFAULT, symbol);
    return address != nullptr && dladdr(address, &info) != 0 ? info.dli_fname : "<static or unavailable>";
}

bool ReadReferenceBlasExpectation(const char*& expected)
{
    expected = std::getenv("OPS_BLAS_REFBLAS_NONFINITE");
    if (expected != nullptr && expected[0] == '\0') {
        expected = nullptr;
    }
#ifdef OPS_BLAS_REFERENCE_PROFILE
    if (expected != nullptr && std::strcmp(expected, "nan") != 0) {
        std::cerr
            << "[ReferenceBLAS] The compiled source profile requires nan; an environment variable cannot change it\n";
        return false;
    }
    expected = "nan";
    std::cout << "[ReferenceBLAS] profile=" << OPS_BLAS_REFERENCE_PROFILE << " linkage=static\n";
#else
    std::cout << "[ReferenceBLAS] profile=system (diagnostic only; reference build is not fixed)\n";
#endif
    if (expected != nullptr && std::strcmp(expected, "nan") != 0 && std::strcmp(expected, "recover-inf") != 0) {
        std::cerr << "[ReferenceBLAS] OPS_BLAS_REFBLAS_NONFINITE must be nan or recover-inf\n";
        return false;
    }
    return true;
}

const char* ProbeReferenceBlas(float (&outputs)[2][2][2])
{
    std::fenv_t environment;
    if (std::feholdexcept(&environment) != 0) {
        std::cerr << "[ReferenceBLAS] Cannot save the floating-point environment\n";
        return nullptr;
    }
    bool allNan = true;
    bool recoverInf = true;
    const float infinity = std::numeric_limits<float>::infinity();
    for (int imaginary = 0; imaginary < 2; ++imaginary) {
        const float alpha[2] = {imaginary == 0 ? 1.0f : 0.0f, imaginary == 0 ? 0.0f : 1.0f};
        const float input[2] = {infinity, infinity};
        float* scaled = outputs[imaginary][0];
        float* added = outputs[imaginary][1];
        scaled[0] = infinity;
        scaled[1] = infinity;
        added[0] = 1.0f;
        added[1] = -2.0f;
        cblas_cscal(1, alpha, scaled, 1);
        cblas_caxpy(1, alpha, input, 1, added, 1);
        for (const float* output : {scaled, added}) {
            allNan = allNan && std::isnan(output[0]) && std::isnan(output[1]);
            recoverInf = recoverInf && output[0] == (imaginary == 0 ? infinity : -infinity) && output[1] == infinity;
        }
    }
    if (std::fesetenv(&environment) != 0) {
        std::cerr << "[ReferenceBLAS] Cannot restore the floating-point environment\n";
        return nullptr;
    }
    return allNan ? "nan" : (recoverInf ? "recover-inf" : "other");
}

void LogReferenceBlas(const char* observed, const float (&outputs)[2][2][2])
{
#ifdef OPS_BLAS_REFERENCE_PROFILE
    std::cout << "[ReferenceBLAS] nonfinite=" << observed << '\n';
#else
    std::cout << "[ReferenceBLAS] cblas_cscal=" << ReferenceBlasLibrary("cblas_cscal")
              << " cblas_caxpy=" << ReferenceBlasLibrary("cblas_caxpy") << " nonfinite=" << observed << '\n';
#endif
    if (std::strcmp(observed, "other") == 0) {
        for (int imaginary = 0; imaginary < 2; ++imaginary) {
            std::cout << "[ReferenceBLAS] alpha=" << (imaginary == 0 ? "(1,0)" : "(0,1)") << " cblas_cscal=("
                      << outputs[imaginary][0][0] << ',' << outputs[imaginary][0][1] << ") cblas_caxpy=("
                      << outputs[imaginary][1][0] << ',' << outputs[imaginary][1][1] << ")\n";
        }
    }
}

bool CheckFiniteReferenceBlas()
{
    const float alpha[2] = {2.0f, -1.0f};
    const float input[2] = {3.0f, 4.0f};
    float scaled[2] = {3.0f, 4.0f};
    float added[2] = {1.0f, -2.0f};
    cblas_cscal(1, alpha, scaled, 1);
    cblas_caxpy(1, alpha, input, 1, added, 1);
    if (scaled[0] != 10.0f || scaled[1] != 5.0f || added[0] != 11.0f || added[1] != 3.0f) {
        std::cerr << "[ReferenceBLAS] Finite sanity check failed: cblas_cscal=(" << scaled[0] << ',' << scaled[1]
                  << ") cblas_caxpy=(" << added[0] << ',' << added[1] << ")\n";
        return false;
    }
    std::cout << "[ReferenceBLAS] finite=pass\n";
    return true;
}

// Diagnose only this independent CPU process; no operator test entry point is involved.
bool CheckReferenceBlas()
{
    const char* expected = nullptr;
    if (!ReadReferenceBlasExpectation(expected)) {
        return false;
    }
    float outputs[2][2][2]{};
    const char* observed = ProbeReferenceBlas(outputs);
    if (observed == nullptr) {
        return false;
    }
    LogReferenceBlas(observed, outputs);
    if (expected != nullptr && std::strcmp(expected, observed) != 0) {
        std::cerr << "[ReferenceBLAS] Reference environment mismatch: expected " << expected << ", observed "
                  << observed << ". Check the selected reference profile and the system-library configuration.\n";
        return false;
    }
    return CheckFiniteReferenceBlas();
}

} // namespace OpsBlasTest

int main() { return OpsBlasTest::CheckReferenceBlas() ? EXIT_SUCCESS : EXIT_FAILURE; }
