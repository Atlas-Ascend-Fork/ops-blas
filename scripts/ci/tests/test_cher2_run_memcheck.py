# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the Cher2 memcheck XML and environment contract."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
CHER2 = ROOT / "test/syr2/cher2"
sys.path.insert(0, str(CHER2))
try:
    SPEC = importlib.util.spec_from_file_location("cher2_run_memcheck_regression", CHER2 / "run_memcheck.py")
    MEMCHECK = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MEMCHECK)
finally:
    sys.path.pop(0)

SELECTED = ["Cher2/Cher2Test.CsvDriven/*"]
CASES = ["CaseA", "CaseB"]


class Cher2MemcheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cher2_memcheck_")
        self.output = Path(self._tmp.name)
        self.xml = self.output / "case.xml"

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
                classname="Cher2/Cher2Test",
                name="CsvDriven/" + case_name,
                status="run",
                result="completed",
            )
            if case_hook is not None:
                case_hook(case_name, case)
        ET.ElementTree(root).write(self.xml)

    def check(self):
        return MEMCHECK.check_xml(self.xml, len(CASES), SELECTED)

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

    def test_main_forwards_clean_environment(self):
        captured = {}

        def fake_run_group(args, name, selected, expected, required, env):
            captured["env"] = env
            return dict(passed=True, expected_tests=expected)

        inherited = {
            "GTEST_REPEAT": "3",
            "GTEST_TOTAL_SHARDS": "2",
            "CHER2_PROFILE_SAMPLES": "5",
            "BLAS_TEST_CASES": "TC_L0_001",
            "KEEP_ME": "1",
        }
        with patch.object(
            MEMCHECK, "parse_args", return_value=SimpleNamespace(tool="mssanitizer", output=self.output)
        ), patch.object(MEMCHECK.shutil, "which", return_value="/bin/echo"), patch.object(
            MEMCHECK, "test_groups", return_value=[("csv", [], 0, set())]
        ), patch.object(MEMCHECK, "run_group", side_effect=fake_run_group), patch.dict(
            os.environ, inherited, clear=True
        ):
            MEMCHECK.main()
        env = captured["env"]
        for key in inherited:
            if key != "KEEP_ME":
                self.assertNotIn(key, env)
        self.assertEqual(env["KEEP_ME"], "1")


if __name__ == "__main__":
    unittest.main()
