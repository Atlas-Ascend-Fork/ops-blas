# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the Ctrsmbatched self-test XML root-skipped, env and timeout contract."""

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
    "trsmbatched_run_self_test_regression", ROOT / "test/trsmbatched/ctrsmbatched/run_self_test.py"
)
SELFTEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELFTEST)

EXPECTED = ["CtrsmBatchedCsvCoverage.CsvCaseCount"]


class TrsmbatchedSelfTestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="trsmbatched_selftest_")
        self.output = Path(self._tmp.name)
        self.xml = self.output / "gtest.xml"

    def tearDown(self):
        self._tmp.cleanup()

    def write_xml(self, root_extra=None, case_hook=None):
        attrib = {"tests": "1", "failures": "0", "errors": "0", "disabled": "0"}
        attrib.update(root_extra or {})
        root = ET.Element("testsuites", attrib)
        suite = ET.SubElement(root, "testsuite", name="AllTests")
        case = ET.SubElement(
            suite,
            "testcase",
            classname="CtrsmBatchedCsvCoverage",
            name="CsvCaseCount",
            status="run",
            result="completed",
        )
        if case_hook is not None:
            case_hook(case)
        ET.ElementTree(root).write(self.xml)

    def test_clean_xml_passes(self):
        self.write_xml()
        result = SELFTEST.read_test_result(self.xml, EXPECTED)
        self.assertTrue(result["passed"])

    def test_root_failures_count_is_rejected(self):
        self.write_xml({"failures": "1"})
        result = SELFTEST.read_test_result(self.xml, EXPECTED)
        self.assertFalse(result["passed"])

    def test_root_error_count_is_rejected(self):
        self.write_xml({"errors": "1"})
        result = SELFTEST.read_test_result(self.xml, EXPECTED)
        self.assertFalse(result["passed"])

    def test_root_disabled_count_is_rejected(self):
        self.write_xml({"disabled": "1"})
        result = SELFTEST.read_test_result(self.xml, EXPECTED)
        self.assertFalse(result["passed"])

    def test_root_skipped_count_is_rejected(self):
        self.write_xml({"skipped": "1"})
        result = SELFTEST.read_test_result(self.xml, EXPECTED)
        self.assertFalse(result["passed"])

    def test_invalid_root_count_is_rejected(self):
        self.write_xml({"errors": "not-a-number"})
        with self.assertRaises(ValueError):
            SELFTEST.read_test_result(self.xml, EXPECTED)

    def test_environment_removes_inherited_gtest_vars(self):
        inherited = {
            "GTEST_FILTER": "*Ctrsm*",
            "GTEST_REPEAT": "3",
            "GTEST_TOTAL_SHARDS": "2",
            "GTEST_SHARD_INDEX": "1",
            "GTEST_ALSO_RUN_DISABLED_TESTS": "1",
            "GTEST_OUTPUT": "xml:/tmp/x.xml",
            "KEEP_ME": "1",
        }
        with patch.dict(os.environ, inherited, clear=True):
            env = SELFTEST.device_environment(0)
        for key in inherited:
            if key.startswith("GTEST_"):
                self.assertNotIn(key, env)
        self.assertEqual(env.get("GTEST_COLOR"), "no")
        self.assertEqual(env["KEEP_ME"], "1")

    def test_non_positive_timeout_is_rejected(self):
        argv = ["run_self_test.py", "--output-dir", str(self.output), "--timeout", "0"]
        with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
            SELFTEST.main()


if __name__ == "__main__":
    unittest.main()
