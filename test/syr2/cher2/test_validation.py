#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-only failure-injection tests for CSV and sanitizer-result validation."""

import csv
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import run_memcheck
import verify_accuracy


class CsvIntegrityTest(unittest.TestCase):
    def check_rejected(self, mutate):
        rows = verify_accuracy.read_cases()
        fields = list(rows[0])
        mutate(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(RuntimeError):
                verify_accuracy.expected_cases(path)

    def test_published_counts(self):
        rows = verify_accuracy.read_cases()
        self.assertEqual(len(rows), 1200)
        self.assertEqual(sum(row["case_name"].startswith("TC_PF_") for row in rows), 200)
        self.assertEqual(len(verify_accuracy.expected_cases(verify_accuracy.HERE / "arch35/cher2_test.csv")), 2233)

    def test_missing_accuracy_row(self):
        self.check_rejected(lambda rows: rows.pop(0))

    def test_duplicate_name(self):
        self.check_rejected(lambda rows: rows[1].update(case_name=rows[0]["case_name"]))

    def test_wrong_category_count(self):
        self.check_rejected(lambda rows: rows[0].update(case_name="TC_PF_9999"))

    def test_generated_name_collision(self):
        self.check_rejected(lambda rows: rows[0].update(case_name="TC_DIST_0"))

    def test_short_and_extra_fields(self):
        source = (verify_accuracy.HERE / "arch35/cher2_test.csv").read_text(encoding="utf-8-sig").splitlines()
        for replacement in (source[1].rsplit(",", 1)[0], source[1] + ",unexpected"):
            with self.subTest(row=replacement), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.csv"
                path.write_text("\n".join([source[0], replacement] + source[2:]) + "\n", encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    verify_accuracy.read_cases(path)


class MemcheckValidationTest(unittest.TestCase):
    @staticmethod
    def clean_log():
        lines = ["[  PASSED  ] 2 tests."]
        for kernel in ("cher2_kernel", "cher2_aiv_kernel"):
            lines.extend([
                "Start memcheck sanitizer on kernel " + kernel,
                "Sanitizer finished on kernel " + kernel + ". No error detected.",
            ])
        return "\n".join(lines)

    def check_result(self, text=None, count=2, status=None, returncode=0):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "result.xml"
            suite = ET.Element("testsuite")
            for index in range(count):
                test = ET.SubElement(suite, "testcase", name=str(index), result="completed")
                if status:
                    ET.SubElement(test, status)
            ET.ElementTree(suite).write(xml)
            result = run_memcheck.check_group(
                "unit", 2, {"cher2_kernel", "cher2_aiv_kernel"}, returncode,
                self.clean_log() if text is None else text, xml,
            )
        return result["passed"]

    def test_clean_pair(self):
        self.assertTrue(self.check_result())

    def test_quoted_demangled_symbols(self):
        text = self.clean_log()
        for kernel in ("cher2_kernel", "cher2_aiv_kernel"):
            text = text.replace("kernel " + kernel, 'kernel "' + kernel + '(unsigned char*, Cher2TilingData)"')
        self.assertTrue(self.check_result(text=text))

    def test_missing_instrumentation_and_unpaired_records(self):
        clean = self.clean_log()
        cases = [
            "[  PASSED  ] 2 tests.",
            clean.replace("Start memcheck sanitizer on kernel cher2_aiv_kernel", ""),
            clean.replace("Sanitizer finished on kernel cher2_kernel. No error detected.", ""),
            clean.replace("cher2_aiv_kernel", "other_kernel"),
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(self.check_result(text=text))

    def test_failed_skipped_missing_and_warning(self):
        for status in ("failure", "skipped"):
            with self.subTest(status=status):
                self.assertFalse(self.check_result(status=status))
        self.assertFalse(self.check_result(count=1))
        self.assertFalse(self.check_result(returncode=1))
        self.assertFalse(self.check_result(text=self.clean_log() + "\nWARNING: memory error"))
        self.assertFalse(self.check_result(text=self.clean_log().replace("2 tests.", "1 tests.")))

    def test_fixed_group_sizes(self):
        groups = run_memcheck.test_groups()
        self.assertEqual([group[2] for group in groups], [27, 4])
        self.assertEqual(len(groups[0][1]), 27)


if __name__ == "__main__":
    unittest.main()
