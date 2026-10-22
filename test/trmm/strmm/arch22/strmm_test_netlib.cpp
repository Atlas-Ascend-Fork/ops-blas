/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// The A2/A3 task specifies the Netlib BLAS STRMM reference. openEuler's
// libblas exports the Fortran reference routine but no cblas_strmm wrapper.
// Resolve strmm_ from the validated library handle: an ordinary external
// strmm_ reference can be interposed by OpenBLAS (including via LD_PRELOAD).

#include <cblas.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef STRMM_NETLIB_LIBRARY
#error "Configure arch22 STRMM through CMake to validate STRMM_NETLIB_LIB."
#endif

namespace {
using StrmmReference = void (*)(const char*, const char*, const char*, const char*,
                               const int*, const int*, const float*, const float*,
                               const int*, float*, const int*, size_t, size_t, size_t, size_t);

void CheckReferenceSemantics(StrmmReference reference)
{
    const int m = 2;
    const int n = 1;
    const float alpha = 1.0F;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    // The unstored triangle must not contribute 0*Inf to a dense GEMM.
    const float a[] = {2.0F, 3.0F, nan, 4.0F};
    float b[] = {inf, inf};
    reference("L", "L", "N", "N", &m, &n, &alpha, a, &m, b, &m, 1, 1, 1, 1);
    if (b[0] != inf || b[1] != inf) {
        throw std::runtime_error("STRMM golden does not preserve Netlib triangular Inf semantics");
    }
    // Netlib LEFT/N skips zero B before even reading its stored diagonal.
    const float zeroA[] = {nan, nan, nan, 4.0F};
    float zeroB[] = {0.0F, 1.0F};
    reference("L", "L", "N", "N", &m, &n, &alpha, zeroA, &m, zeroB, &m, 1, 1, 1, 1);
    if (zeroB[0] != 0.0F || zeroB[1] != 4.0F) {
        throw std::runtime_error("STRMM golden does not preserve Netlib zero-B semantics");
    }
}

StrmmReference LoadReference()
{
    void* library = dlopen(STRMM_NETLIB_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        throw std::runtime_error(std::string("Cannot load STRMM Netlib golden: ") + dlerror());
    }
    try {
        if (dlsym(library, "openblas_get_config") != nullptr ||
            dlsym(library, "bli_info_get_version_str") != nullptr) {
            throw std::runtime_error("STRMM golden requires Netlib reference BLAS, not OpenBLAS/BLIS");
        }
        auto reference = reinterpret_cast<StrmmReference>(dlsym(library, "strmm_"));
        if (reference == nullptr) {
            throw std::runtime_error("STRMM Netlib golden is missing the LP64 strmm_ symbol");
        }
        Dl_info origin{};
        if (dladdr(reinterpret_cast<void*>(reference), &origin) == 0 || origin.dli_fname == nullptr) {
            throw std::runtime_error("Cannot identify the STRMM golden library");
        }
        CheckReferenceSemantics(reference);
        std::fprintf(stderr, "[STRMM golden] Netlib strmm_: %s (configured: %s)\n",
                     origin.dli_fname, STRMM_NETLIB_LIBRARY);
        // Keep the handle alive for the lifetime of the cached function pointer.
        return reference;
    } catch (...) {
        dlclose(library);
        throw;
    }
}

StrmmReference GetReference()
{
    static const StrmmReference reference = LoadReference();
    return reference;
}
} // namespace

extern "C" void cblas_strmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag, const int m, const int n,
                             const float alpha, const float* a, const int lda, float* b, const int ldb)
{
    if (layout != CblasColMajor) {
        std::abort(); // STRMM tests in this repository use column-major only.
    }
    const char sideChar = side == CblasRight ? 'R' : 'L';
    const char uploChar = uplo == CblasLower ? 'L' : 'U';
    const char transChar = trans == CblasTrans ? 'T' : (trans == CblasConjTrans ? 'C' : 'N');
    const char diagChar = diag == CblasUnit ? 'U' : 'N';
    GetReference()(&sideChar, &uploChar, &transChar, &diagChar, &m, &n, &alpha, a, &lda, b, &ldb,
                   size_t{1}, size_t{1}, size_t{1}, size_t{1});
}

#ifdef STRMM_NETLIB_PROBE
// CMake runs the same loader and semantic checks before building NPU tests.
int main()
{
    try {
        GetReference();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
}
#endif
