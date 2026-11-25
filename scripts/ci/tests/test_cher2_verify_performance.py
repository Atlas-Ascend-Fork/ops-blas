# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the Cher2 performance XML and environment contract."""

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
    "cher2_verify_performance_regression", ROOT / "test/syr2/cher2/verify_performance.py"
)
PERF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PERF)

BATCH = [{"case_name": "CaseA"}, {"case_name": "CaseB"}]


class Cher2PerformanceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cher2_performance_")
        self.xml = Path(self._tmp.name) / "gtest.xml"

    def tearDown(self):
        self._tmp.cleanup()

    def write_xml(self, root_extra=None, case_hook=None):
        attrib = {"tests": "2", "failures": "0", "errors": "0", "disabled": "0"}
        attrib.update(root_extra or {})
        root = ET.Element("testsuites", attrib)
        suite = ET.SubElement(root, "testsuite", name="AllTests")
        for row in BATCH:
            case = ET.SubElement(
                suite,
                "testcase",
                classname="Cher2/Cher2Test",
                name="CsvDriven/" + row["case_name"],
                status="run",
                result="completed",
            )
            if case_hook is not None:
                case_hook(row["case_name"], case)
        ET.ElementTree(root).write(self.xml)

    def test_clean_profile_xml_passes(self):
        self.write_xml()
        PERF.validate_profile_xml(self.xml, BATCH)

    def test_case_error_is_rejected(self):
        self.write_xml(case_hook=lambda name, case: ET.SubElement(case, "error", message="boom"))
        with self.assertRaises(RuntimeError):
            PERF.validate_profile_xml(self.xml, BATCH)

    def test_case_not_run_is_rejected(self):
        self.write_xml(case_hook=lambda name, case: case.set("status", "notrun"))
        with self.assertRaises(RuntimeError):
            PERF.validate_profile_xml(self.xml, BATCH)

    def test_root_error_count_is_rejected(self):
        self.write_xml({"errors": "1"})
        with self.assertRaises(RuntimeError):
            PERF.validate_profile_xml(self.xml, BATCH)

    def test_environment_removes_inherited_gtest_vars(self):
        inherited = {
            "GTEST_FILTER": "*Cher2*",
            "GTEST_REPEAT": "3",
            "GTEST_TOTAL_SHARDS": "2",
            "BLAS_TEST_CASES": "TC_L0_001",
            "CHER2_PROFILE": "0",
            "KEEP_ME": "1",
        }
        with patch.dict(os.environ, inherited, clear=True):
            env = PERF.build_environment()
        self.assertFalse([key for key in env if key.startswith("GTEST_")])
        self.assertNotIn("BLAS_TEST_CASES", env)
        self.assertEqual(env["CHER2_PROFILE"], "1")
        self.assertEqual(env["CHER2_PROFILE_RANGE"], "1")
        self.assertEqual(env["KEEP_ME"], "1")

    def test_non_positive_timeout_is_rejected(self):
        argv = ["verify_performance.py", "--timeout", "0"]
        with patch.object(sys, "argv", argv), patch.object(
            PERF.argparse.ArgumentParser, "error", side_effect=ValueError("invalid timeout")
        ) as parser_error, self.assertRaises(ValueError):
            PERF.main()
        parser_error.assert_called_once_with("--timeout must be a positive integer")


if __name__ == "__main__":
    unittest.main()
