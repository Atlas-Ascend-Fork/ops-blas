# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for Chemv sanitizer logs and GTest argument forwarding."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/symv/chemv"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "chemv_memcheck_regression", ROOT / "test/symv/chemv/run_memcheck.py")
    MEMCHECK = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MEMCHECK)
finally:
    sys.path.pop(0)
KERNEL = '"chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"'
START = "[mssanitizer] Start memcheck sanitizer on kernel " + KERNEL + "\n"
FINISH = "[mssanitizer] Sanitizer finished on kernel " + KERNEL + ". No error detected.\n"


class TestChemvMemcheckLogs(unittest.TestCase):
    def test_demangled_kernel_name(self):
        record = MEMCHECK.sanitizer_records(START + FINISH, "")
        self.assertTrue(record["balanced"])
        self.assertEqual(record["started_kernels"], {KERNEL: 1})
        self.assertEqual(record["finished_kernels"], {KERNEL: 1})
        self.assertFalse(record["diagnostics"])

    def test_missing_finish_is_rejected(self):
        self.assertFalse(MEMCHECK.sanitizer_records(START, "")["balanced"])

    def test_clean_finish_does_not_hide_warning(self):
        for level in ("WARN", "WARNING", "ERROR", "FATAL"):
            with self.subTest(level=level):
                self.assertTrue(MEMCHECK.sanitizer_records("", "[" + level + "] " + FINISH)["diagnostics"])

    def check_forwarding(self, text, passed):
        with tempfile.TemporaryDirectory(prefix="chemv_gate_") as folder:
            output = Path(folder)
            args = SimpleNamespace(output=output, binary="fixture-binary", tool="fixture-sanitizer")
            group = dict(name="sample", tests=["Fixture.Case"], require_kernel=True)

            def fake_run(command, log, env):
                self.assertEqual(command[-1], args.binary)
                self.assertNotIn("--log-level=info", command)
                self.assertFalse(any(option.startswith("--gtest_") for option in command))
                self.assertEqual(env["GTEST_FILTER"], "Fixture.Case")
                self.assertEqual(env["GTEST_REPEAT"], "1")
                self.assertEqual(env["GTEST_OUTPUT"], "xml:" + str(output / "sample.xml"))
                log.write_text(text, encoding="utf-8")
                root = ET.Element("testsuites", tests="1", failures="0", disabled="0", errors="0")
                suite = ET.SubElement(root, "testsuite")
                ET.SubElement(suite, "testcase", classname="Fixture", name="Case", status="run", result="completed")
                ET.ElementTree(root).write(output / "sample.xml")
                return 0

            with patch.object(MEMCHECK, "run_logged", side_effect=fake_run):
                self.assertEqual(MEMCHECK.run_group(args, group, {})["passed"], passed)

    def test_gtest_options_use_environment(self):
        self.check_forwarding(START + FINISH, True)

    def test_gtest_pass_without_kernel_does_not_pass_memcheck(self):
        self.check_forwarding("[  PASSED  ] 1 test.\n", False)


if __name__ == "__main__":
    unittest.main()
