# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the Chpr2 memcheck XML, environment and timeout contract."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "chpr2_run_memcheck_regression", ROOT / "test/spr2/chpr2/arch35/run_memcheck.py"
)
MEMCHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMCHECK)

SELECTED = ["*/CaseA", "*/CaseB"]
CASES = ["CaseA", "CaseB"]


class Chpr2MemcheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="chpr2_memcheck_")
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
                classname="Chpr2/Chpr2Arch35Test",
                name="CsvDriven/" + case_name,
                status="run",
                result="completed",
            )
            if case_hook is not None:
                case_hook(case_name, case)
        ET.ElementTree(root).write(self.xml)

    def check(self):
        return MEMCHECK.parse_xml_testcases(self.xml, SELECTED)

    def test_clean_xml_passes(self):
        self.write_xml()
        self.assertTrue(self.check()[0])

    def test_missing_xml_is_rejected(self):
        self.assertFalse(self.check()[0])

    def test_case_error_is_rejected(self):
        self.write_xml(case_hook=lambda name, case: ET.SubElement(case, "error", message="boom"))
        self.assertFalse(self.check()[0])

    def test_case_not_run_is_rejected(self):
        self.write_xml(case_hook=lambda name, case: case.set("status", "notrun"))
        self.assertFalse(self.check()[0])

    def test_root_error_count_is_rejected(self):
        self.write_xml({"errors": "1"})
        self.assertFalse(self.check()[0])

    def test_root_skipped_count_is_rejected(self):
        self.write_xml({"skipped": "1"})
        self.assertFalse(self.check()[0])

    def test_invalid_root_count_is_rejected(self):
        self.write_xml({"errors": "not-a-number"})
        self.assertFalse(self.check()[0])

    def test_environment_removes_inherited_gtest_vars(self):
        inherited = {
            "GTEST_FILTER": "*Chpr2*",
            "GTEST_REPEAT": "3",
            "GTEST_TOTAL_SHARDS": "2",
            "GTEST_ALSO_RUN_DISABLED_TESTS": "1",
            "CHPR2_PROFILE": "0",
            "CHPR2_PROFILE_CALLS": "0",
            "CHPR2_RESULTS": "0",
            "BLAS_TEST_CASES": "TC_L0_001",
            "KEEP_ME": "1",
        }
        with patch.dict(os.environ, inherited, clear=True):
            env = MEMCHECK.build_environment()
        self.assertFalse([key for key in env if key.startswith("GTEST_")])
        self.assertNotIn("BLAS_TEST_CASES", env)
        self.assertNotIn("CHPR2_PROFILE", env)
        self.assertNotIn("CHPR2_PROFILE_CALLS", env)
        self.assertNotIn("CHPR2_RESULTS", env)
        self.assertEqual(env["KEEP_ME"], "1")

    def test_non_positive_timeout_is_rejected(self):
        argv = ["run_memcheck.py", "--binary", "/bin/echo", "--output", str(self.output), "--timeout", "0"]
        with patch.object(sys, "argv", argv), patch.object(
            MEMCHECK.argparse.ArgumentParser, "error", side_effect=ValueError("invalid timeout")
        ) as parser_error, self.assertRaises(ValueError):
            MEMCHECK.main()
        parser_error.assert_called_once_with("--timeout must be a positive integer")


if __name__ == "__main__":
    unittest.main()
