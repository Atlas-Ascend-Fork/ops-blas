# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile the real Host conversion header; never call an NPU runtime."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

DRIVER = r"""
#include "dtype_compat.h"
#include <cmath>
#include <cstdio>
#include <cstring>

int main()
{
    uint32_t mismatch = 0U;
    uint32_t nan_count = 0U;
    // Every nonzero FP32 fraction with exponent255 is a NaN. Enumerate all
    // 16777214 positive/negative payloads, including signaling and quiet NaNs.
    for (uint32_t sign : {0U, 0x80000000U}) {
        for (uint32_t payload = 1U; payload <= 0x7fffffU; ++payload) {
            const uint32_t bits = sign | 0x7f800000U | payload;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            const uint16_t converted = floatToBf16(value);
            const float expanded = bf16ToFloat(converted);
            ++nan_count;
            if (!std::isnan(expanded) || ((converted & 0x8000U) != (sign >> 16U))) {
                ++mismatch;
            }
        }
    }
    // Finite round-to-nearest-even oracle uses integer division/remainder,
    // independently of the production add-bias implementation.
    uint32_t state = 0x12345678U;
    uint32_t finite_count = 0U;
    for (uint32_t index = 0U; index < 200000U; ++index) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        const uint32_t magnitude = state % 0x7f800000U;
        const uint32_t sign = state & 0x80000000U;
        const uint32_t bits = magnitude | sign;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        uint32_t rounded = magnitude / 65536U;
        const uint32_t remainder = magnitude % 65536U;
        if (remainder > 32768U || (remainder == 32768U && (rounded % 2U != 0U))) {
            ++rounded;
        }
        const uint16_t expected = static_cast<uint16_t>((sign >> 16U) | rounded);
        mismatch += floatToBf16(value) != expected;
        ++finite_count;
    }
    // Signed zero, infinities and exact tie-even/tie-odd controls.
    const uint32_t boundary[] = {0U, 0x80000000U, 0x7f800000U, 0xff800000U,
        0x3f808000U, 0x3f818000U, 0xbf808000U, 0xbf818000U};
    const uint16_t expected[] = {0U, 0x8000U, 0x7f80U, 0xff80U,
        0x3f80U, 0x3f82U, 0xbf80U, 0xbf82U};
    for (uint32_t index = 0U; index < 8U; ++index) {
        float value;
        std::memcpy(&value, &boundary[index], sizeof(value));
        mismatch += floatToBf16(value) != expected[index];
    }
    std::printf("nan_payloads=%u finite_controls=%u mismatches=%u\n", nan_count, finite_count, mismatch);
    return mismatch == 0U ? 0 : 1;
}
"""


def test_bf16_preserves_all_nan_payloads_and_finite_rounding(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A real Host C++ compiler is required.")
    include_dirs = []
    if os.environ.get("ASCEND_HOME_PATH"):
        include_dirs.append(Path(os.environ["ASCEND_HOME_PATH"]) / "include")
    include_dirs += [Path(p) for p in os.environ.get("CANN_HOST_HEADER_INCLUDE", "").split(os.pathsep) if p]
    if not any((p / "acl/acl.h").is_file() for p in include_dirs):
        pytest.skip("Real CANN ACL headers are required; no replacement declarations are generated.")
    cpp = tmp_path / "real_dtype_header.cpp"
    cpp.write_text("#include <initializer_list>\n" + DRIVER)
    binary = tmp_path / "real_dtype_header"
    command = [compiler, "-std=c++17", "-O0", "-fno-fast-math", "-I" + str(ROOT / "test/utils")]
    command += ["-I" + str(p) for p in include_dirs]
    command += [str(cpp), "-o", str(binary)]
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "nan_payloads=16777214 finite_controls=200000 mismatches=0" in run.stdout
