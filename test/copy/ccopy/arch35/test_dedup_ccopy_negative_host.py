# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host regressions for the actual ccopy deduplication CLI, without CANN/NPU."""

import csv
import io
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("dedup_ccopy_csv.py")
HEADER = [
    "case_name",
    "n",
    "incx",
    "incy",
    "x_fill",
    "y_fill",
    "x_align_offset",
    "y_align_offset",
    "expect_result",
    "random_seed",
]


def row(name, *, n=32, expect="SUCCESS", fill="RANDOM_NORM_5_5", seed=1):
    return [name, str(n), "1", "1", fill, "ZERO", "0", "0", expect, str(seed)]


def negative_fixtures():
    cases = []
    for expect in ["INVALID_VALUE", "ACLBLAS_STATUS_INVALID_VALUE", "NOT_SUPPORTED"]:
        cases.append(
            (
                expect,
                [
                    row("TC_EX_001", expect=expect),
                    row("TC_EX_002", expect=expect, seed=2),
                ],
                [0, 1],
            )
        )
    for n in [1, 32]:
        cases.append(
            (
                f"negative-prefix-{n}",
                [row("TC_ED_001", n=n), row("TC_ED_002", n=n, seed=2)],
                [0, 1],
            )
        )
    return cases


def mixed_fixtures():
    return [
        (
            "negative-before-positive",
            [row("TC_EX_001", expect="INVALID_VALUE"), row("TC_EX_002")],
            [0, 1],
        ),
        (
            "positive-before-negatives",
            [
                row("TC_EX_001"),
                row("TC_EX_002", expect="INVALID_VALUE"),
                row("TC_EX_003", expect="INVALID_VALUE", seed=2),
            ],
            [0, 1, 2],
        ),
    ]


def positive_fixtures():
    return [
        ("positive-exact", [row("TC_EX_001"), row("TC_EX_002", seed=2)], [0]),
        (
            "positive-equivalent",
            [row("TC_EX_001"), row("TC_EX_002", fill="VALUE_5")],
            [0],
        ),
        (
            "edge-positive-exact",
            [row("TC_EX_001", n=1), row("TC_EX_002", n=1, seed=2)],
            [0],
        ),
        (
            "edge-positive-equivalent",
            [row("TC_EX_001", n=1), row("TC_EX_002", n=1, fill="VALUE_5")],
            [0, 1],
        ),
        ("other-group-exact", [row("TC_PF_001"), row("TC_PF_002", seed=2)], [0]),
    ]


def fixtures():
    return negative_fixtures() + mixed_fixtures() + positive_fixtures()


class TestCcopyNegativeHost(unittest.TestCase):
    def check_cli(self, header, rows, expected, write):
        with tempfile.TemporaryDirectory(prefix="ccopy-negative-host-") as tmp:
            path = Path(tmp) / "cases.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
            before = path.read_bytes()
            command = [sys.executable, str(SCRIPT)]
            if write:
                command.append("--write")
            result = subprocess.run(
                command + [str(path)], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = re.search(r"total=(\d+) kept=(\d+) dropped=(\d+)", result.stdout)
            self.assertIsNotNone(summary, result.stdout)
            self.assertEqual(
                tuple(map(int, summary.groups())),
                (len(rows), len(expected), len(rows) - len(expected)),
            )
            if write:
                actual = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
                self.assertEqual(actual, [header, *expected])
            else:
                self.assertEqual(
                    path.read_bytes(), before, "preview must not modify its input"
                )

    def test_preview_preserves_negative_cases(self):
        for label, rows, indices in fixtures():
            with self.subTest(label=label):
                self.check_cli(HEADER, rows, [rows[i] for i in indices], False)

    def test_write_preserves_negative_cases(self):
        for label, rows, indices in fixtures():
            with self.subTest(label=label):
                self.check_cli(HEADER, rows, [rows[i] for i in indices], True)

    def test_current_repository_cases_remain_unchanged(self):
        with SCRIPT.with_name("ccopy_test.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            header, *rows = list(csv.reader(stream))
        for write in [False, True]:
            with self.subTest(write=write):
                self.check_cli(header, rows, rows, write)


if __name__ == "__main__":
    unittest.main()
