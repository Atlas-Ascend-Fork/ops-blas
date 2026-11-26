# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the HgemmStridedBatched self-test XML root-count and env contract."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HGEM_STRIDED = ROOT / "test/gemm/hgemmStridedBatched"
sys.path.insert(0, str(HGEM_STRIDED))
try:
    SPEC = importlib.util.spec_from_file_location(
        "hgemm_strided_batched_run_self_test_regression", HGEM_STRIDED / "run_self_test.py"
    )
    SELFTEST = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(SELFTEST)
finally:
    sys.path.pop(0)


class HgemmStridedBatchedSelfTestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="hgemm_strided_selftest_")
        self.out = Path(self._tmp.name)

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
            classname="Hgemm/HgemmStridedBatchedTest",
            name="CsvDriven/CaseA",
            status="run",
            result="completed",
        )
        if case_hook is not None:
            case_hook(case)
        ET.ElementTree(root).write(self.out / "gtest.xml")

    def test_clean_xml_passes(self):
        self.write_xml()
        observed, failed, skipped, parse_error = SELFTEST.parse_cases(self.out, None)
        self.assertIsNone(parse_error)
        self.assertEqual(observed, ["CaseA"])
        self.assertEqual(failed, [])
        self.assertEqual(skipped, [])

    def test_root_failures_count_is_rejected(self):
        self.write_xml({"failures": "1"})
        _, _, _, parse_error = SELFTEST.parse_cases(self.out, None)
        self.assertIsNotNone(parse_error)

    def test_root_error_count_is_rejected(self):
        self.write_xml({"errors": "1"})
        _, _, _, parse_error = SELFTEST.parse_cases(self.out, None)
        self.assertIsNotNone(parse_error)

    def test_root_disabled_count_is_rejected(self):
        self.write_xml({"disabled": "1"})
        _, _, _, parse_error = SELFTEST.parse_cases(self.out, None)
        self.assertIsNotNone(parse_error)

    def test_root_skipped_count_is_rejected(self):
        self.write_xml({"skipped": "1"})
        _, _, _, parse_error = SELFTEST.parse_cases(self.out, None)
        self.assertIsNotNone(parse_error)

    def test_invalid_root_count_is_rejected(self):
        self.write_xml({"errors": "not-a-number"})
        _, _, _, parse_error = SELFTEST.parse_cases(self.out, None)
        self.assertIsNotNone(parse_error)

    def test_environment_removes_inherited_gtest_vars(self):
        inherited = {
            "GTEST_FILTER": "*Hgemm*",
            "GTEST_REPEAT": "3",
            "GTEST_TOTAL_SHARDS": "2",
            "GTEST_SHARD_INDEX": "1",
            "GTEST_ALSO_RUN_DISABLED_TESTS": "1",
            "BLAS_TEST_CASES": "TC_L0_001",
            "KEEP_ME": "1",
        }
        with patch.dict(os.environ, inherited, clear=True):
            env = SELFTEST.build_environment()
        self.assertFalse([key for key in env if key.startswith("GTEST_")])
        self.assertNotIn("BLAS_TEST_CASES", env)
        self.assertEqual(env["KEEP_ME"], "1")


if __name__ == "__main__":
    unittest.main()
