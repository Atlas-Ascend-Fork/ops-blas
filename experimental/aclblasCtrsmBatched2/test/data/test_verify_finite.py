# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise the actual complex64 verifier CLI with Host binary files."""

import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("verify_result.py").resolve()


def run_verifier(golden, output):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        files = directory / "test/data/output"
        files.mkdir(parents=True)
        golden.tofile(files / "golden_0.bin")
        output.tofile(files / "output_b_0.bin")
        return subprocess.run(
            [str(Path(sys.executable).absolute()), str(SCRIPT), str(golden.shape[0]), str(golden.shape[1]), "1"],
            cwd=directory,
            text=True,
            capture_output=True,
            check=False,
        )


def finite_pair(shape=(2, 2)):
    golden = np.full(shape, complex(1, 2), dtype=np.complex64)
    return golden, golden.copy()


class TestVerifyFinite(unittest.TestCase):
    def test_nonfinite_real_or_imaginary_component(self):
        cases = product(("golden", "output"), ("real", "imag"), (np.nan, np.inf, -np.inf))
        for side, component, value in cases:
            with self.subTest(side=side, component=component, value=value):
                golden, output = finite_pair()
                target = golden if side == "golden" else output
                getattr(target, component)[0, 0] = value
                result = run_verifier(golden, output)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("[Failed]", result.stdout)

    def test_matching_nonfinite_components_are_not_accuracy_success(self):
        for component, value in product(("real", "imag"), (np.nan, np.inf, -np.inf)):
            with self.subTest(component=component, value=value):
                golden, output = finite_pair()
                getattr(golden, component)[0, 0] = value
                getattr(output, component)[0, 0] = value
                result = run_verifier(golden, output)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("[Failed]", result.stdout)

    def test_nonfinite_cannot_use_the_allowed_finite_mismatch_ratio(self):
        for value, expected in ((np.nan, 1), (1.005, 0)):
            with self.subTest(value=value):
                golden, output = finite_pair((1, 10001))
                output.real[0, 0] = value
                result = run_verifier(golden, output)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_existing_finite_tolerances(self):
        for delta, expected in ((0.0, 0), (1e-5, 0), (0.1, 1)):
            with self.subTest(delta=delta):
                golden, output = finite_pair()
                output.real[0, 0] += delta
                result = run_verifier(golden, output)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
