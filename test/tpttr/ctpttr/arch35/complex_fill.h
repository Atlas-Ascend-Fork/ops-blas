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

#include "golden_cursor.h"
#include "scalar_fill.h"

namespace complex_triangular_test {

class ComplexFill {
public:
    ComplexFill(const std::string& name, uint32_t seed) : real_(name, seed), imag_(name, seed + 1000U) {}

    ComplexBits At(uint64_t index) { return {FloatBits(real_.At(index)), FloatBits(imag_.At(index))}; }

private:
    ScalarFill real_;
    ScalarFill imag_;
};

inline ComplexBits Guard(uint64_t index)
{
    return {0x7FC12345U ^ static_cast<uint32_t>(index & 0xFFU), 0xFFA54321U ^ static_cast<uint32_t>(index & 0xFFU)};
}

} // namespace complex_triangular_test
