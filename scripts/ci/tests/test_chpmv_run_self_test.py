# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the Chpmv self-test XML root-count and env contract."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "chpmv_run_self_test_regression", ROOT / "test/spmv/chpmv/run_self_test.py"
)
SELFTEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELFTEST)

EXPECTED = ["Chpmv/ChpmvComparator.CaseA"]


class ChpmvSelfTestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="chpmv_selftest_")
        self.xml = Path(self._tmp.name) / "results.xml"

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
            classname="Chpmv/ChpmvComparator",
            name="CaseA",
            status="run",
            result="completed",
        )
        if case_hook is not None:
            case_hook(case)
        ET.ElementTree(root).write(self.xml)

    def test_clean_xml_passes(self):
        self.write_xml()
        SELFTEST.executed_records(self.xml, EXPECTED)

    def test_root_failures_count_is_rejected(self):
        self.write_xml({"failures": "1"})
        with self.assertRaises(ValueError):
            SELFTEST.executed_records(self.xml, EXPECTED)

    def test_root_error_count_is_rejected(self):
        self.write_xml({"errors": "1"})
        with self.assertRaises(ValueError):
            SELFTEST.executed_records(self.xml, EXPECTED)

    def test_root_disabled_count_is_rejected(self):
        self.write_xml({"disabled": "1"})
        with self.assertRaises(ValueError):
            SELFTEST.executed_records(self.xml, EXPECTED)

    def test_root_skipped_count_is_rejected(self):
        self.write_xml({"skipped": "1"})
        with self.assertRaises(ValueError):
            SELFTEST.executed_records(self.xml, EXPECTED)

    def test_invalid_root_count_is_rejected(self):
        self.write_xml({"errors": "not-a-number"})
        with self.assertRaises(ValueError):
            SELFTEST.executed_records(self.xml, EXPECTED)

    def test_environment_removes_inherited_gtest_vars(self):
        inherited = {
            "GTEST_FILTER": "*Chpmv*",
            "GTEST_REPEAT": "3",
            "GTEST_TOTAL_SHARDS": "2",
            "GTEST_SHARD_INDEX": "1",
            "GTEST_ALSO_RUN_DISABLED_TESTS": "1",
            "GTEST_OUTPUT": "xml:/tmp/x.xml",
            "BLAS_TEST_CASES": "TC_L0_001",
            "CHPMV_PROFILE_REPEAT": "5",
            "KEEP_ME": "1",
        }
        with patch.dict(os.environ, inherited, clear=True):
            env = SELFTEST.build_environment()
        self.assertFalse([key for key in env if key.startswith("GTEST_")])
        self.assertNotIn("BLAS_TEST_CASES", env)
        self.assertNotIn("CHPMV_PROFILE_REPEAT", env)
        self.assertEqual(env["KEEP_ME"], "1")


if __name__ == "__main__":
    unittest.main()
