# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the CgemmBatched memcheck XML contract."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
CGEM = ROOT / "test/gemm_batched/cgemm_batched"
sys.path.insert(0, str(CGEM))
try:
    SPEC = importlib.util.spec_from_file_location("cgemm_batched_run_memcheck_regression", CGEM / "run_memcheck.py")
    MEMCHECK = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MEMCHECK)
finally:
    sys.path.pop(0)

CASES = ["CaseA", "CaseB"]


class CgemmBatchedMemcheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cgemm_batched_memcheck_")
        self.xml = Path(self._tmp.name) / "case.xml"

    def tearDown(self):
        self._tmp.cleanup()

    def write_xml(self, root_extra=None, case_hook=None):
        attrib = {"tests": "2", "failures": "0", "errors": "0", "disabled": "0"}
        attrib.update(root_extra or {})
        root = ET.Element("testsuites", attrib)
        suite = ET.SubElement(root, "testsuite", name="AllTests")
        for case_name in CASES:
            case = ET.SubElement(
                suite,
                "testcase",
                classname="CgemmBatched/CgemmBatchedTest",
                name="CsvDriven/" + case_name,
                status="run",
                result="completed",
            )
            if case_hook is not None:
                case_hook(case_name, case)
        ET.ElementTree(root).write(self.xml)

    def check(self, expected=len(CASES)):
        return MEMCHECK.check_xml(self.xml, expected)

    def test_clean_xml_passes(self):
        self.write_xml()
        self.assertTrue(self.check())

    def test_missing_xml_is_rejected(self):
        self.assertFalse(self.check())

    def test_case_error_is_rejected(self):
        self.write_xml(case_hook=lambda name, case: ET.SubElement(case, "error", message="boom"))
        self.assertFalse(self.check())

    def test_case_not_run_is_rejected(self):
        self.write_xml(case_hook=lambda name, case: case.set("status", "notrun"))
        self.assertFalse(self.check())

    def test_root_error_count_is_rejected(self):
        self.write_xml({"errors": "1"})
        self.assertFalse(self.check())

    def test_invalid_root_count_is_rejected(self):
        self.write_xml({"errors": "not-a-number"})
        self.assertFalse(self.check())

    def test_unexpected_case_count_is_rejected(self):
        self.write_xml()
        self.assertFalse(self.check(len(CASES) + 1))


if __name__ == "__main__":
    unittest.main()
