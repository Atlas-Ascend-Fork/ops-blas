#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-only validation that really imports run_self_test / run_memcheck (BL-34).

用法：
  python3 test_validation.py
"""
from __future__ import print_function

import csv
import os
import tempfile
import unittest

import run_memcheck
import run_self_test

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "arch35", "cher_test.csv")


class CsvSelfTestValidation(unittest.TestCase):
    def test_published_csv_counts(self):
        rows = run_self_test.audit_csv(CSV_PATH)
        self.assertEqual(len(rows), run_self_test.EXPECTED_CASES)
        self.assertEqual(
            sum(1 for row in rows if row["case_name"].startswith("TC_PF_")),
            run_self_test.EXPECTED_PERF,
        )

    def test_rejects_duplicate_names(self):
        with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
        rows[1]["case_name"] = rows[0]["case_name"]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.csv")
            with open(path, "w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                run_self_test.audit_csv(path)

    def test_rejects_wrong_total(self):
        with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
        rows = rows[:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "short.csv")
            with open(path, "w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                run_self_test.audit_csv(path)


class MemcheckSelfTestValidation(unittest.TestCase):
    def test_clean_requires_start_records(self):
        clean = "Start memcheck\nNo error detected\n[  PASSED  ] 1 test.\n"
        self.assertTrue(run_memcheck.audit_memcheck(clean, 0)["ok"])

    def test_zero_starts_is_fail(self):
        # False-green guard: returncode 0 with no instrumentation must FAIL.
        self.assertFalse(run_memcheck.audit_memcheck("[  PASSED  ] 0 tests.\n", 0)["ok"])

    def test_warning_is_fail(self):
        text = "Start memcheck\nWARNING: MemorySanitizer: use-of-uninitialized-value\n"
        self.assertFalse(run_memcheck.audit_memcheck(text, 0)["ok"])

    def test_nonzero_return_is_fail(self):
        text = "Start memcheck\nNo error detected\n"
        self.assertFalse(run_memcheck.audit_memcheck(text, 1)["ok"])


if __name__ == "__main__":
    unittest.main()
