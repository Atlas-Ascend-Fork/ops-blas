# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# 运行时智能选卡脚本：为测试运行挑选空闲的 NPU 设备。

"""Host regressions for device selection argument validation."""

import contextlib
import importlib.util
import io
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import ANY, patch


SCRIPT = Path(__file__).resolve().parents[1] / "select_device.py"
SPEC = importlib.util.spec_from_file_location("select_device", SCRIPT)
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


class DeviceThresholdTests(unittest.TestCase):
    def check_invalid_threshold(self, threshold):
        # Capture argparse's exit boundary without terminating the test process.
        with contextlib.ExitStack() as stack:
            resolve_lib = stack.enter_context(patch.object(SELECTOR, "resolve_lib_path"))
            parser_exit = stack.enter_context(patch.object(
                SELECTOR.argparse.ArgumentParser, "exit", side_effect=RuntimeError("parser exit")
            ))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, "parser exit"):
                SELECTOR.main(["--threshold=" + threshold])
            parser_exit.assert_called_once_with(2, ANY)
            resolve_lib.assert_not_called()
        # Also exercise the real CLI and its unmodified argparse exit behavior.
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--threshold=" + threshold],
            capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("--threshold", result.stderr)

    def test_invalid_threshold_is_rejected_before_acl_access(self):
        for threshold in ("nan", "inf", "-inf", "-1", "100.1", "invalid"):
            with self.subTest(threshold=threshold):
                self.check_invalid_threshold(threshold)

    def check_valid_threshold(self, threshold, expected):
        stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            load_acl = stack.enter_context(patch.object(SELECTOR, "load_acl"))
            load_acl.return_value.aclInit.return_value = 0
            stack.enter_context(patch.object(SELECTOR.os.path, "isfile", return_value=True))
            stack.enter_context(patch.object(SELECTOR, "get_device_count", return_value=1))
            stack.enter_context(patch.object(SELECTOR, "get_device_utilization", return_value=100.0))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertEqual(SELECTOR.main(["--threshold=" + threshold]), expected)
            load_acl.return_value.aclFinalize.assert_called_once_with()
        self.assertEqual(stdout.getvalue(), "0\n" if expected == 0 else "ALL_BUSY\n")

    def test_valid_threshold_keeps_busy_and_boundary_decisions(self):
        for threshold, expected in (("0", 20), ("5.5", 20), ("100", 0)):
            with self.subTest(threshold=threshold):
                self.check_valid_threshold(threshold, expected)


if __name__ == "__main__":
    unittest.main()
