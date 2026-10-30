# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Portable negative tests for the evidence gates (no NPU claims)."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import test_gate as gate
from run_memcheck import audit_memcheck


class GateTests(unittest.TestCase):
    def test_memcheck_requires_real_clean_instrumentation(self):
        clean = "Start memcheck\nNo error detected\nERROR: 0\n"
        self.assertTrue(audit_memcheck(clean, "program completed")["instrumentation_verified"])
        for text, run in (("No error detected", ""), ("Start memcheck", ""),
                          (clean + "ERROR out of bounds at 0x0", ""),
                          (clean, "stopping DBI")):
            with self.assertRaises(RuntimeError):
                audit_memcheck(text, run)

    def test_committed_csv_counts(self):
        self.assertEqual(gate.audit_csvs()["sgemm_grouped_batched_test_full.csv"]["performance"], 200)

    def test_truncated_full_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in gate.COUNTS:
                shutil.copyfile(gate.ARCH / name, root / name)
            full = root / "sgemm_grouped_batched_test_full.csv"
            full.write_bytes(full.read_bytes().rsplit(b"\n", 2)[0] + b"\n")
            with patch.object(gate, "ARCH", root), self.assertRaises(RuntimeError):
                gate.audit_csvs()

    def test_missing_discovery_case_rejected(self):
        names = ["TaskCSV/S.CSVAccuracy/" + row["case_name"]
                 for row in gate.read_cases(gate.ARCH / "sgemm_grouped_batched_test.csv")]
        names += ["Review/S.CSVAccuracy/" + row["case_name"]
                  for row in gate.read_cases(gate.ARCH / "review_regression.csv")]
        gate.check_mapping(names, gate.ARCH / "sgemm_grouped_batched_test.csv")
        with self.assertRaises(RuntimeError):
            gate.check_mapping(names[1:], gate.ARCH / "sgemm_grouped_batched_test.csv")

    def test_xml_missing_failure_skip_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "result.xml"
            for child in ("<failure/>", "<skipped/>"):
                xml.write_text(f'<testsuites><testsuite><testcase classname="S" name="T">{child}'
                               '</testcase></testsuite></testsuites>', encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    gate.check_xml(xml, ["S.T"])
            xml.write_text('<testsuites/>', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                gate.check_xml(xml, ["S.T"])
            xml.write_text('<testsuites><testsuite><testcase classname="S" name="T"/>'
                           '</testsuite></testsuites>', encoding="utf-8")
            self.assertEqual(gate.check_xml(xml, ["S.T"])["executed"], 1)

    def test_jsonl_mock_missing_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            for rows in ([], [{"case_name": "TC_1", "status": "CPU_MODEL_PASS"}],
                         [{"case_name": "TC_1", "status": "PASS"}] * 2):
                path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    gate.check_jsonl(path, ["TaskCSV/S.CSVAccuracy/TC_1"])


if __name__ == "__main__":
    unittest.main()
