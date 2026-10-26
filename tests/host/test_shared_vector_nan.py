# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the complete shared verifier using real ACL headers on the Host."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def real_acl_include():
    explicit = os.environ.get("CANN_HOST_HEADER_INCLUDE")
    home = os.environ.get("ASCEND_HOME_PATH")
    include = Path(explicit) if explicit else Path(home) / "include" if home else None
    assert include is not None and (include / "acl/acl.h").is_file(), (
        "real ACL headers are required: set CANN_HOST_HEADER_INCLUDE or ASCEND_HOME_PATH"
    )
    return include


@pytest.mark.parametrize("optimization", ("-O0", "-O2"))
def test_shared_vector_single_nan_and_existing_controls(optimization, tmp_path):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler is not None, "a real C++17 Host compiler is required"
    cpp = tmp_path / "shared_verifier.cpp"
    cpp.write_text(DRIVER)
    binary = tmp_path / "shared_verifier"
    command = [compiler, "-std=c++17", optimization, "-fno-fast-math", "-I", str(ROOT),
               "-I", str(ROOT / "include"), "-I", str(real_acl_include()), str(cpp), "-o", str(binary)]
    compiled = subprocess.run(command, capture_output=True, text=True, check=False)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cases=486 mismatches=0" in result.stdout


DRIVER = r"""#include <array>
#include <limits>
#include <vector>
#include "test/frame/verify.h"

struct InputPair {
    const char* name;
    float output;
    float golden;
    bool expectedVector;
    bool expectedScalar;
};

int RunCase(PrecisionMode mode, const InputPair& pair, int64_t stride, size_t position)
{
    std::array<float, 5> output = {1, 1, 1, 1, 1};
    std::array<float, 5> golden = {1, 1, 1, 1, 1};
    const int64_t start = stride < 0 ? 4 : 0;
    const size_t offset = static_cast<size_t>(start + static_cast<int64_t>(position) * stride);
    output[offset] = pair.output;
    golden[offset] = pair.golden;
    VerifyConfig cfg;
    cfg.mode = mode;
    const bool actualVector = Verifier::verifyVector(output.data() + start, golden.data() + start,
                                                    3, stride, cfg, pair.name);
    const bool actualScalar = Verifier::verifyScalar(pair.output, pair.golden, cfg, pair.name);
    const bool mismatch = actualVector != pair.expectedVector || actualScalar != pair.expectedScalar;
    if (mismatch) {
        std::cout << "MISMATCH mode=" << static_cast<int>(mode) << " name=" << pair.name
                  << " stride=" << stride << " position=" << position << " vector=" << actualVector
                  << " scalar=" << actualScalar << '\n';
    }
    return mismatch ? 1 : 0;
}

int main()
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::array<InputPair, 18> pairs = {{
        {"nan_output", nan, 1, false, false},
        {"nan_golden", 1, nan, false, false},
        {"negative_nan_output", -nan, 1, false, false},
        {"negative_nan_golden", 1, -nan, false, false},
        {"both_nan", nan, -nan, true, true},
        {"finite_equal", 1, 1, true, true},
        {"finite_bad", 2, 1, false, false},
        {"finite_inside_tolerance", 1.000001f, 1, true, true},
        {"signed_zero", 0, -0.0f, true, true},
        {"both_positive_inf", inf, inf, true, true},
        {"both_negative_inf", -inf, -inf, true, true},
        {"finite_positive_inf", 1, inf, false, false},
        {"positive_inf_finite", inf, 1, false, false},
        {"finite_negative_inf", 1, -inf, false, false},
        {"negative_inf_finite", -inf, 1, false, false},
        {"nan_inf", nan, inf, false, false},
        {"inf_nan", inf, nan, false, false},
        {"existing_opposite_inf_policy", inf, -inf, true, false}
    }};
    int cases = 0;
    int mismatches = 0;
    for (PrecisionMode mode : {PrecisionMode::ABS, PrecisionMode::REL, PrecisionMode::COMBINED}) {
        for (int64_t stride : {1, 2, -1}) {
            for (size_t position = 0; position < 3; ++position) {
                for (const auto& pair : pairs) {
                    mismatches += RunCase(mode, pair, stride, position);
                    ++cases;
                }
            }
        }
    }
    std::cout << "cases=" << cases << " mismatches=" << mismatches << '\n';
    return mismatches == 0 ? 0 : 1;
}
"""
