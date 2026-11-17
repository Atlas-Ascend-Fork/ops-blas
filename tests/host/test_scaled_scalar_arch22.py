# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check the device range fallback against FP64 arithmetic on the host."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r"""
#define __aicore__
#include "scaled_scalar_arch22.h"
#include <cmath>
#include <limits>
#include <random>

using namespace scaled_scalar;

bool Same(float a, float b)
{
    return (std::isnan(a) && std::isnan(b)) || a == b;
}

float Finite(std::mt19937& generator)
{
    Bits value;
    value.bits = generator();
    if ((value.bits & 0x7f800000U) == 0x7f800000U) {
        value.bits &= 0xfeffffffU;
    }
    return value.value;
}

int CheckRandom()
{
    std::mt19937 generator(20261112U);
    for (int sample = 0; sample < 200000; ++sample) {
        const float a = Finite(generator);
        const float b = Finite(generator);
        const float c = Finite(generator);
        const Value product = Product(a, b);
        if (!Same(ToFloat(product), static_cast<float>(double(a) * b)) ||
            !Same(ToFloat(Scale(product, c)), static_cast<float>((double(a) * b) * c))) {
            return 1;
        }
    }
    for (int sample = 0; sample < 10000; ++sample) {
        Value sum{0, 0};
        double golden = 0.0;
        for (int layer = 0; layer < 32; ++layer) {
            const float a = Finite(generator);
            const float b = Finite(generator);
            sum = Add(sum, Product(a, b));
            golden += double(a) * b;
        }
        if (!Same(ToFloat(sum), static_cast<float>(golden))) {
            return 2;
        }
    }
    return 0;
}

int main()
{
    const float maximum = std::numeric_limits<float>::max();
    const auto recovered = Add(Product(maximum, 2.0F), Neg(Product(maximum, 1.0F)));
    const auto diagonal = ConjugateProduct(maximum, maximum, maximum, maximum);
    const float infinity = std::numeric_limits<float>::infinity();
    if (ToFloat(recovered) != maximum || ToFloat(diagonal.imag) != 0.0F ||
        ToFloat(diagonal.real) != infinity || !std::isnan(ToFloat(Add(FromFloat(infinity), FromFloat(-infinity))))) {
        return 3;
    }
    return CheckRandom();
}
"""


@pytest.mark.parametrize("optimization", ("-O0", "-O2"))
def test_scaled_scalar_fp64_rounding(optimization, tmp_path):
    compiler = shutil.which("c++")
    assert compiler is not None, "a C++17 host compiler is required"
    source = tmp_path / "scaled_scalar_test.cpp"
    binary = tmp_path / "scaled_scalar_test"
    source.write_text(SOURCE, encoding="utf-8")
    command = [compiler, "-std=c++17", optimization, "-fsanitize=undefined", "-fno-sanitize-recover=undefined"]
    command += ["-ffp-contract=off", "-I", str(ROOT / "blas/common/helper"), str(source), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
