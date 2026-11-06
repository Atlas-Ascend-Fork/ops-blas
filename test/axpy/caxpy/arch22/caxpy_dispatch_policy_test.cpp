/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <iostream>

#include "caxpy_dispatch_policy.h"

namespace {
struct ClassificationCase {
    int64_t incx;
    int64_t incy;
    CaxpyStrideClass expected;
};

struct PolicyCase {
    uint64_t n;
    int64_t incx;
    int64_t incy;
    bool directStrided;
    uint32_t numBlocks;
    CaxpyKernelVariant expected;
};

constexpr std::array<ClassificationCase, 15> CASES{{
    {1, 1, CaxpyStrideClass::DENSE},
    {2, 1, CaxpyStrideClass::X_ONLY_POSITIVE},
    {9, 1, CaxpyStrideClass::X_ONLY_POSITIVE},
    {1, 2, CaxpyStrideClass::Y_ONLY_POSITIVE},
    {1, 10, CaxpyStrideClass::Y_ONLY_POSITIVE},
    {2, 2, CaxpyStrideClass::DUAL_SYMMETRIC_POSITIVE},
    {10, 10, CaxpyStrideClass::DUAL_SYMMETRIC_POSITIVE},
    {2, 3, CaxpyStrideClass::DUAL_ASYMMETRIC_POSITIVE},
    {10, 11, CaxpyStrideClass::DUAL_ASYMMETRIC_POSITIVE},
    {-1, 1, CaxpyStrideClass::NEGATIVE_MIXED},
    {1, -1, CaxpyStrideClass::NEGATIVE_MIXED},
    {-2, 3, CaxpyStrideClass::NEGATIVE_MIXED},
    {5, -3, CaxpyStrideClass::NEGATIVE_MIXED},
    {-2, -3, CaxpyStrideClass::NEGATIVE_MIXED},
    {-10, -10, CaxpyStrideClass::NEGATIVE_MIXED},
}};

constexpr std::array<PolicyCase, 23> POLICY_CASES{{
    {1, 1, 1, false, 1, CaxpyKernelVariant::DENSE_PIPELINED_ATOMIC},
    {8192, 1, 1, false, 40, CaxpyKernelVariant::DENSE_PIPELINED_ATOMIC},
    {640, 2, 1, true, 40, CaxpyKernelVariant::STRIDED_SCALAR},
    {640, 1, 3, true, 40, CaxpyKernelVariant::STRIDED_SCALAR},
    {640, 5, 5, true, 40, CaxpyKernelVariant::STRIDED_SCALAR},
    {640, 2, 3, true, 40, CaxpyKernelVariant::STRIDED_SCALAR},
    {640, -2, 3, true, 40, CaxpyKernelVariant::STRIDED_SCALAR},
    {641, 5, 1, true, 40, CaxpyKernelVariant::STRIDED_SCALAR},
    {679, 5, 1, true, 40, CaxpyKernelVariant::STRIDED_SCALAR},
    {17, 65537, -32769, true, 40, CaxpyKernelVariant::STRIDED_SCALAR},
    {680, 9, 1, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC},
    {680, 1, 9, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC},
    {680, 5, 5, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC},
    {680, 2, 3, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC},
    {680, -2, -3, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC},
    // incx=10 now stays on the shared span-read pipeline: it exceeds the old
    // fixed CAXPY_SHARED_X_SPAN_MAX_INCREMENT=9 cutoff but is well within
    // both the tile-capacity correctness floor and the measured performance
    // cap (CAXPY_SHARED_SPAN_PERFORMANCE_MAX_INCREMENT=60). This case used to
    // assert STRIDED_DISJOINT_PIPELINED_ATOMIC under the old fixed threshold;
    // that expectation is now wrong on purpose -- see caxpy_dispatch_policy.h
    // for the crossover measurement this depends on.
    {680, 10, 1, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC},
    // incx=61 exceeds the performance cap and correctly falls back to
    // disjoint even though its shrunk tile (capacity 80) would still be
    // usable by the correctness floor alone.
    {680, 61, 1, true, 40, CaxpyKernelVariant::STRIDED_DISJOINT_PIPELINED_ATOMIC},
    // incx=304 shrinks the tile capacity to 32, below
    // CAXPY_SHARED_MIN_USABLE_TILE_CAPACITY (33) -- the correctness floor,
    // not the performance cap, is what forces disjoint here. Kept as
    // defense-in-depth coverage of that floor in case the performance cap is
    // ever raised past this incx.
    {680, 304, 1, true, 40, CaxpyKernelVariant::STRIDED_DISJOINT_PIPELINED_ATOMIC},
    {327680, 2, 3, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC},
    {327681, 2, 3, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_SCALAR_TAIL},
    {328359, -2, 3, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_SCALAR_TAIL},
    {328360, 2, -3, true, 40, CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC},
    {640, 5, 5, false, 40, CaxpyKernelVariant::DENSE_PIPELINED_ATOMIC},
}};

} // namespace

int main()
{
    bool passed = true;
    for (const ClassificationCase& test : CASES) {
        const CaxpyStrideClass actual = ClassifyCaxpyStride(test.incx, test.incy);
        if (actual == test.expected)
            continue;
        std::cerr << "classification mismatch: incx=" << test.incx << " incy=" << test.incy
                  << " expected=" << static_cast<uint32_t>(test.expected) << " actual=" << static_cast<uint32_t>(actual)
                  << '\n';
        passed = false;
    }
    for (const PolicyCase& test : POLICY_CASES) {
        const CaxpyKernelVariant actual =
            SelectCaxpyKernelVariant(test.n, test.incx, test.incy, test.directStrided, test.numBlocks);
        if (actual == test.expected)
            continue;
        std::cerr << "policy mismatch: n=" << test.n << " incx=" << test.incx << " incy=" << test.incy
                  << " direct_strided=" << test.directStrided << " expected=" << static_cast<uint32_t>(test.expected)
                  << " actual=" << static_cast<uint32_t>(actual) << '\n';
        passed = false;
    }

    return passed ? 0 : 1;
}
