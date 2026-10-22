#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host CSV/report regressions using synthetic timings, without running an NPU."""

import csv
from pathlib import Path
import tempfile
import unittest

import verify_performance as verifier


class BaselineValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[3]
        cls.cases = verifier.parse_perf_params(repo, "ascend950")

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.path = Path(self.workspace.name) / "gpu_baseline.csv"

    def write_baseline(self, overrides=None, count=None):
        values = overrides or {}
        cases = (
            self.cases
            if count is None
            else (
                self.cases[:count]
                if count <= len(self.cases)
                else self.cases + [self.cases[-1]]
            )
        )
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_name", "gpu_ms"])
            writer.writeheader()
            writer.writerows(
                {"case_name": case["case_name"], "gpu_ms": values.get(index, "0.1")}
                for index, case in enumerate(cases)
            )
        return self.path

    def timings(self):
        return {
            case["case_name"]: {
                "n": int(case["n"]),
                "incx": int(case["incx"]),
                "warmup": 20,
                "samples": 100,
                "average_us": 100.0,
            }
            for case in self.cases
        }

    def test_accepts_finite_positive_millisecond_values(self):
        for value in ["0.1", "+0.1", " 0.2 ", "1e-6", "1e-200", "1e305"]:
            with self.subTest(gpu_ms=value):
                rows = verifier.load_baseline(self.write_baseline({0: value}))
                self.assertEqual(len(rows), 200)
                self.assertEqual(rows[0]["gpu_ms"], value)

    def test_complete_valid_csv_and_report(self):
        baseline = verifier.load_baseline(self.write_baseline())
        rows = verifier.collect_rows(self.cases, baseline, self.timings(), True)
        self.assertTrue(all(row["verdict"] == "PASS" for row in rows))
        self.assertTrue(all(row["ratio"] == "1.000000" for row in rows))
        output = Path(self.workspace.name) / "comparison.csv"
        verifier.write_rows(rows, output)
        with output.open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(
            saved, [{key: str(value) for key, value in row.items()} for row in rows]
        )
        self.assertEqual(verifier.report(rows, output, 0, []), 0)

    def test_rejects_nonfinite_and_unit_conversion_overflow(self):
        for value in ["nan", "NaN", "inf", "+inf", "-inf", "1e309", "1e307"]:
            with self.subTest(gpu_ms=value):
                with self.assertRaises(ValueError):
                    verifier.load_baseline(self.write_baseline({0: value}))

    def test_rejects_nonpositive_and_underflowed_times(self):
        for value in ["0", "-0", "-1", "-1e-20", "1e-400"]:
            with self.subTest(gpu_ms=value):
                with self.assertRaises(ValueError):
                    verifier.load_baseline(self.write_baseline({0: value}))

    def test_checks_first_middle_and_last_rows(self):
        for index in [0, 99, 199]:
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "gpu_ms"):
                    verifier.load_baseline(self.write_baseline({index: "nan"}))

    def test_rejects_empty_malformed_and_missing_cells(self):
        for value in ["", " ", "not-a-time"]:
            with self.subTest(gpu_ms=value):
                with self.assertRaises(ValueError):
                    verifier.load_baseline(self.write_baseline({0: value}))
        self.write_baseline()
        lines = self.path.read_text().splitlines()
        lines[-1] = self.cases[-1]["case_name"]
        self.path.write_text("\n".join(lines) + "\n")
        with self.subTest(missing_cell=True):
            with self.assertRaises(ValueError):
                verifier.load_baseline(self.path)
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_name"])
            writer.writeheader()
            writer.writerows({"case_name": case["case_name"]} for case in self.cases)
        with self.subTest(missing_column=True):
            with self.assertRaises(ValueError):
                verifier.load_baseline(self.path)

    def test_preserves_exact_baseline_row_count(self):
        for count in [199, 201]:
            with self.subTest(rows=count):
                with self.assertRaises(ValueError):
                    verifier.load_baseline(self.write_baseline(count=count))

    def test_preserves_performance_threshold(self):
        for gpu_ms, expected in [("0.04", "PASS"), ("0.039", "FAIL")]:
            with self.subTest(gpu_ms=gpu_ms):
                baseline = verifier.load_baseline(
                    self.write_baseline({index: gpu_ms for index in range(200)})
                )
                rows = verifier.collect_rows(self.cases, baseline, self.timings(), True)
                self.assertTrue(all(row["verdict"] == expected for row in rows))
                code = verifier.report(rows, self.path, 0, [])
                self.assertEqual(code, 0 if expected == "PASS" else 1)


if __name__ == "__main__":
    unittest.main()
