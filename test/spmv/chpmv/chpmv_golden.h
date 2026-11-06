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
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <link.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include "chpmv_param.h"

namespace chpmv_test {
class Oracle {
public:
    using Function = void (*)(int, int, int, const void*, const void*, const void*, int, const void*, void*, int);
    Function run = nullptr;
    void* library = nullptr;
    static std::string Canonical(const char* path)
    {
        if (path == nullptr || path[0] == '\0') {
            throw std::runtime_error("invalid Netlib library path");
        }
        std::error_code error;
        const auto resolved = std::filesystem::canonical(path, error);
        if (error) {
            throw std::runtime_error("invalid Netlib library path");
        }
        return resolved.string();
    }
    Oracle()
    {
        const auto cblasPath = Canonical(std::getenv("CHPMV_NETLIB_CBLAS"));
        const auto blasPath = Canonical(std::getenv("CHPMV_NETLIB_BLAS"));
        // 独立加载命名空间，避免进程中其他 BLAS 抢占 Fortran 内部符号。
        library = dlmopen(LM_ID_NEWLM, cblasPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library)
            throw std::runtime_error(dlerror());
        run = reinterpret_cast<Function>(dlsym(library, "cblas_chpmv"));
        void* fortran = dlsym(library, "chpmv_");
        Dl_info ci{}, fi{};
        if (!run || !fortran || !dladdr(reinterpret_cast<void*>(run), &ci) || !dladdr(fortran, &fi) ||
            Canonical(ci.dli_fname) != cblasPath || Canonical(fi.dli_fname) != blasPath) {
            throw std::runtime_error("Netlib symbol provider does not match configured libraries");
        }
        std::cout << "NETLIB_CBLAS=" << ci.dli_fname << " NETLIB_CHPMV=" << fi.dli_fname << '\n';
        Complex ap[]{{2, NAN}, {1, 2}, {3, INFINITY}}, x[]{{1, 1}, {2, -1}}, y[2]{};
        const Complex alpha{1, 0}, beta{0, 0};
        run(102, 121, 2, &alpha, ap, x, 1, &beta, y, 1);
        if (y[0].real != 6 || y[0].imag != 5 || y[1].real != 9 || y[1].imag != -4)
            throw std::runtime_error("Netlib known-answer and diagonal-ignore check failed");
        Complex lower[]{{2, NAN}, {1, -2}, {3, INFINITY}}, reverse[]{{2, -1}, {1, 1}};
        y[0] = y[1] = {NAN, NAN};
        run(102, 122, 2, &alpha, lower, reverse, -1, &beta, y, -1);
        if (y[0].real != 9 || y[0].imag != -4 || y[1].real != 6 || y[1].imag != 5)
            throw std::runtime_error("Netlib lower/negative-stride/beta-zero check failed");
    }
    ~Oracle()
    {
        if (library)
            dlclose(library);
    }
    static Oracle& Instance()
    {
        static Oracle oracle;
        return oracle;
    }
};

} // namespace chpmv_test
