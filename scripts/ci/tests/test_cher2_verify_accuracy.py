# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the Cher2 accuracy XML, environment and timeout contract."""

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
    "cher2_verify_accuracy_regression", ROOT / "test/syr2/cher2/verify_accuracy.py"
)
ACCURACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCURACY)

CASES = ["CaseA", "CaseB"]


class Cher2AccuracyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cher2_accuracy_")
        self.output = Path(self._tmp.name)
        self.xml = self.output / "accuracy.xml"

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

    def test_clean_xml_passes(self):
        self.write_xml()
        ACCURACY.validate_accuracy_xml(self.xml, set(CASES))

    def test_missing_xml_is_rejected(self):
        with self.assertRaises(RuntimeError):
            ACCURACY.validate_accuracy_xml(self.xml, set(CASES))

    def test_case_error_is_rejected(self):
        self.write_xml(case_hook=lambda name, case: ET.SubElement(case, "error", message="boom"))
        with self.assertRaises(RuntimeError):
            ACCURACY.validate_accuracy_xml(self.xml, set(CASES))

    def test_case_not_run_is_rejected(self):
        self.write_xml(case_hook=lambda name, case: case.set("status", "notrun"))
        with self.assertRaises(RuntimeError):
            ACCURACY.validate_accuracy_xml(self.xml, set(CASES))

    def test_root_error_count_is_rejected(self):
        self.write_xml({"errors": "1"})
        with self.assertRaises(RuntimeError):
            ACCURACY.validate_accuracy_xml(self.xml, set(CASES))

    def test_root_skipped_count_is_rejected(self):
        self.write_xml({"skipped": "1"})
        with self.assertRaises(RuntimeError):
            ACCURACY.validate_accuracy_xml(self.xml, set(CASES))

    def test_invalid_root_count_is_rejected(self):
        self.write_xml({"errors": "not-a-number"})
        with self.assertRaises(RuntimeError):
            ACCURACY.validate_accuracy_xml(self.xml, set(CASES))

    def test_environment_removes_inherited_gtest_vars(self):
        inherited = {
            "GTEST_FILTER": "*Cher2*",
            "GTEST_REPEAT": "3",
            "GTEST_TOTAL_SHARDS": "2",
            "GTEST_ALSO_RUN_DISABLED_TESTS": "1",
            "CHER2_PROFILE": "0",
            "CHER2_PROFILE_RANGE": "0",
            "CHER2_PROFILE_SAMPLES": "5",
            "BLAS_TEST_CASES": "TC_L0_001",
            "KEEP_ME": "1",
        }
        with patch.dict(os.environ, inherited, clear=True):
            env = ACCURACY.build_environment()
        self.assertFalse([key for key in env if key.startswith("GTEST_")])
        self.assertNotIn("BLAS_TEST_CASES", env)
        self.assertNotIn("CHER2_PROFILE", env)
        self.assertNotIn("CHER2_PROFILE_RANGE", env)
        self.assertNotIn("CHER2_PROFILE_SAMPLES", env)
        self.assertEqual(env["KEEP_ME"], "1")

    def test_non_positive_timeout_is_rejected(self):
        argv = ["verify_accuracy.py", "--timeout", "0"]
        with patch.object(sys, "argv", argv), patch.object(
            ACCURACY.argparse.ArgumentParser, "error", side_effect=ValueError("invalid timeout")
        ) as parser_error, self.assertRaises(ValueError):
            ACCURACY.main()
        parser_error.assert_called_once_with("--timeout must be a positive integer")


if __name__ == "__main__":
    unittest.main()
