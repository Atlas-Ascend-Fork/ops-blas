# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the HgemmBatched memcheck XML root-count contract."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HGEM = ROOT / "test/gemm/hgemmBatched"
sys.path.insert(0, str(HGEM))
try:
    saved_run_self_test = sys.modules.pop("run_self_test", None)
    SPEC = importlib.util.spec_from_file_location(
        "hgemm_batched_run_memcheck_regression", HGEM / "run_memcheck.py"
    )
    MEMCHECK = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MEMCHECK)
finally:
    sys.path.pop(0)
    if saved_run_self_test is not None:
        sys.modules["run_self_test"] = saved_run_self_test
    else:
        sys.modules.pop("run_self_test", None)

CASE = "CaseA"


class HgemmBatchedMemcheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="hgemm_batched_memcheck_")
        self.xml = Path(self._tmp.name) / "gtest.xml"

    def tearDown(self):
        self._tmp.cleanup()

    def write_xml(self, root_extra=None, case_hook=None):
        attrib = {"tests": "1", "failures": "0", "errors": "0", "disabled": "0", "skipped": "0"}
        attrib.update(root_extra or {})
        root = ET.Element("testsuites", attrib)
        suite = ET.SubElement(root, "testsuite", name="AllTests")
        case = ET.SubElement(
            suite,
            "testcase",
            classname="HgemmBatched/HgemmBatchedTest",
            name="CsvDriven/" + CASE,
            status="run",
            result="completed",
        )
        if case_hook is not None:
            case_hook(case)
        ET.ElementTree(root).write(self.xml)

    def test_clean_xml_passes(self):
        self.write_xml()
        self.assertEqual(MEMCHECK.check_xml(self.xml, CASE), 1)

    def test_root_failures_count_is_rejected(self):
        self.write_xml({"failures": "1"})
        with self.assertRaises(ValueError):
            MEMCHECK.check_xml(self.xml, CASE)

    def test_root_error_count_is_rejected(self):
        self.write_xml({"errors": "1"})
        with self.assertRaises(ValueError):
            MEMCHECK.check_xml(self.xml, CASE)

    def test_root_disabled_count_is_rejected(self):
        self.write_xml({"disabled": "1"})
        with self.assertRaises(ValueError):
            MEMCHECK.check_xml(self.xml, CASE)

    def test_root_skipped_count_is_rejected(self):
        self.write_xml({"skipped": "1"})
        with self.assertRaises(ValueError):
            MEMCHECK.check_xml(self.xml, CASE)

    def test_invalid_root_count_is_rejected(self):
        self.write_xml({"errors": "not-a-number"})
        with self.assertRaises(ValueError):
            MEMCHECK.check_xml(self.xml, CASE)


if __name__ == "__main__":
    unittest.main()
