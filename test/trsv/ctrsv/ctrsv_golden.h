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

#include <cstddef>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "cann_ops_blas.h"

namespace ctrsv_reference {

// GNU Fortran passes CHARACTER lengths after the explicit arguments.
using NetlibCtrsv = void (*)(
    const char*, const char*, const char*, const int*, const aclblasComplex*, const int*, aclblasComplex*, const int*,
    size_t, size_t, size_t);

inline NetlibCtrsv GetNetlibCtrsv()
{
    static const auto function = []() {
        // Resolve on this handle, not the global scope, which also contains OpenBLAS.
        void* library = dlopen(CTRSV_NETLIB_LIBRARY, RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            throw std::runtime_error(std::string("Cannot load Netlib BLAS: ") + dlerror());
        }
        auto symbol = reinterpret_cast<NetlibCtrsv>(dlsym(library, "ctrsv_"));
        if (symbol == nullptr) {
            throw std::runtime_error("Configured Netlib BLAS does not export ctrsv_");
        }
        if (dlsym(library, "openblas_get_config") != nullptr) {
            throw std::runtime_error("CTRSV_NETLIB_LIBRARY must be Netlib, not OpenBLAS");
        }
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(symbol), &info) == 0) {
            throw std::runtime_error("Cannot identify ctrsv_ reference library");
        }
        std::cout << "[GOLDEN] backend=Netlib symbol=ctrsv_ library=" << info.dli_fname << std::endl;
        return symbol;
    }();
    return function;
}

} // namespace ctrsv_reference

inline void aclblasCtrsv_cpu(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const aclblasComplex* matrix,
    int lda, aclblasComplex* x, int incx)
{
    static_assert(
        std::is_standard_layout<aclblasComplex>::value && sizeof(aclblasComplex) == 2 * sizeof(float) &&
            offsetof(aclblasComplex, real) == 0 && offsetof(aclblasComplex, imag) == sizeof(float),
        "Netlib COMPLEX requires interleaved float real/imaginary components");
    const char u = uplo == ACLBLAS_UPPER ? 'U' : 'L';
    const char t = trans == ACLBLAS_OP_N ? 'N' : (trans == ACLBLAS_OP_T ? 'T' : 'C');
    const char d = diag == ACLBLAS_UNIT ? 'U' : 'N';
    ctrsv_reference::GetNetlibCtrsv()(&u, &t, &d, &n, matrix, &lda, x, &incx, 1, 1, 1);
}
