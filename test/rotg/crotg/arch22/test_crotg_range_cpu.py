#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU arithmetic regressions; no ACL/NPU runtime or device validation."""

import csv
import ctypes
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[4]
CXX = shutil.which("g++")
if CXX is not None:
    CXX = str(Path(CXX).resolve())


def fp32(value):
    return ctypes.c_float(value).value


def reference(ar, ai, br, bi):
    """Evaluate the rotation definition in double, then round outputs to FP32."""
    anorm = math.hypot(ar, ai)
    bnorm = math.hypot(br, bi)
    if bnorm == 0:
        return [ar, ai, 1.0, 0.0, 0.0]
    if anorm == 0:
        return [fp32(bnorm), 0.0, 0.0, fp32(br / bnorm), fp32(-bi / bnorm)]
    norm = math.hypot(ar, ai, br, bi)
    phase = complex(ar / anorm, ai / anorm)
    sine = phase * complex(br, -bi) / norm
    return [fp32(phase.real * norm), fp32(phase.imag * norm), fp32(anorm / norm),
            fp32(sine.real), fp32(sine.imag)]



def extract_function(source, signature):
    """Compile the actual Host arithmetic without requiring ACL runtime headers."""
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def production_source():
    source = (REPO / "blas/rotg/arch22/crotg_host.cpp").read_text(encoding="utf-8")
    signatures = ("static void SetCrotgZeroResult", "static void CrotgCpuCompute")
    functions = []
    for signature in signatures:
        functions.append(extract_function(source, signature))
    return "#include <cmath>\nstruct aclblasComplex { float real; float imag; };\n" + "\n".join(functions) + r'''
extern "C" void compute(float ar, float ai, float br, float bi, float* out)
{
    aclblasComplex a{ar, ai}, b{br, bi}, s{};
    CrotgCpuCompute(&a, &b, &out[2], &s);
    out[0] = a.real;
    out[1] = a.imag;
    out[3] = s.real;
    out[4] = s.imag;
}
'''


@unittest.skipUnless(CXX, "requires g++")
class CrotgRangeCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="crotg-range-")
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        source = directory / "test.cpp"
        source.write_text(production_source(), encoding="utf-8")
        library = directory / "test.so"
        subprocess.run([CXX, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", "-I", str(REPO), str(source), "-o", str(library)],
                       check=True, capture_output=True, text=True)
        cls.library = ctypes.CDLL(str(library))
        cls.compute = cls.library.compute
        cls.compute.argtypes = [ctypes.c_float] * 4 + [ctypes.POINTER(ctypes.c_float)]
        cls.compute.restype = None

    def check_case(self, values):
        values = tuple(map(fp32, values))
        actual = (ctypes.c_float * 5)()
        self.compute(*values, actual)
        expected = reference(*values)
        for index, (result, golden) in enumerate(zip(actual, expected)):
            with self.subTest(inputs=values, component=index):
                if math.isinf(golden):
                    self.assertEqual(result, golden)
                else:
                    self.assertTrue(math.isfinite(result), (result, golden))
                    # Component comparisons preserve the BLAS phase convention;
                    # an annihilation residual alone can accept the wrong phase.
                    self.assertLessEqual(abs(result - golden), 2**-16 + 2**-10 * abs(golden))
        return actual

    def test_zero_branches_and_ordinary_values(self):
        for values in [(0, 0, 0, 0), (3, -4, 0, 0), (0, 0, -3, 4),
                       (3, 4, -5, 12), (-1, 2, 3, -4)]:
            self.check_case(values)

    def test_large_finite_components(self):
        for values in [(2e38, 2e38, 2e38, 2e38), (3e38, 3e38, 1, -1),
                       (3e38, 3e38, 0, 0), (0, 0, 2e38, 2e38),
                       (-2e38, 2e38, 2e38, -2e38), (3e38, 0, 0, 3e38)]:
            self.check_case(values)

    def test_small_nonzero_phases(self):
        for ar, ai in [(0, 1e-30), (0, -1e-30), (-1e-30, 0), (1e-30, -1e-30)]:
            for br, bi in [(1, 0), (0, -1), (1e30, -1e30)]:
                self.check_case((ar, ai, br, bi))
        output = self.check_case((0, 1e-30, 1, 0))
        self.assertGreater(output[2], 0.0)
        self.assertAlmostEqual(output[2] / fp32(1e-30), 1.0, places=5)

    def test_existing_finite_csv_cases(self):
        count = 0
        with (REPO / "test/rotg/crotg/arch22/crotg_test.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS":
                    continue
                values = [float(row[key]) for key in ("a_real", "a_imag", "b_real", "b_imag")]
                if all(math.isfinite(value) for value in values):
                    self.check_case(values)
                    count += 1
        self.assertGreater(count, 0)

    def test_full_fp32_host_range(self):
        tiny = 2**-149
        largest = fp32(float.fromhex("0x1.fffffep+127"))
        for values in [(0, tiny, largest, 0), (tiny, -tiny, largest, -largest),
                       (tiny, tiny, -tiny, tiny), (largest, tiny, 0, 0)]:
            self.check_case(values)

    def test_logarithmic_random_inputs(self):
        generator = random.Random(20261115)
        for _ in range(3000):
            values = [math.ldexp(generator.uniform(-1, 1), generator.randint(-126, 127))
                      for _ in range(4)]
            self.check_case(values)


if __name__ == "__main__":
    unittest.main()
