# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-only regression tests for pr_filelist EOF newline and CRLF handling in run_example.sh."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(os.environ.get(
    "RUN_EXAMPLE_SCRIPT", Path(__file__).resolve().parents[1] / "run_example.sh"
))


class RunExampleFilelistFormatTests(unittest.TestCase):
    def run_raw_filelist(self, raw_bytes, soc="ascend950"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "scripts/ci/run_example.sh"
            script.parent.mkdir(parents=True)
            script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
            script.chmod(0o755)
            for directory in ("axpy/caxpy", "blasLtMatmul", "frame"):
                target = root / "test" / directory
                target.mkdir(parents=True, exist_ok=True)
                (target / "CMakeLists.txt").touch()
            (root / "test/axpy/caxpy/arch35").mkdir(parents=True, exist_ok=True)
            (root / "build.sh").write_text(
                '#!/bin/bash\nprintf "BUILD_ARGS=%s\\n" "$*"\n', encoding="utf-8"
            )
            filelist = root / "files.txt"
            filelist.write_bytes(raw_bytes)
            result = subprocess.run(
                [str(script), str(filelist), soc], cwd=root,
                env={**os.environ, "RUN_FLAGS": "--no-smoke", "RUNNER_NPU_COUNT": "1"},
                capture_output=True, text=True, timeout=30, shell=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return next(line for line in result.stdout.splitlines()
                        if line.startswith("BUILD_ARGS="))

    def test_filelist_without_trailing_newline_keeps_final_entry(self):
        op_args = self.run_raw_filelist(b"test/axpy/caxpy/arch35/caxpy_test.cpp")
        self.assertIn("--ops=caxpy,blasLtMatmul", op_args)
        frame_args = self.run_raw_filelist(b"test/frame/test_main.cpp")
        self.assertNotIn("--ops=", frame_args)

    def test_filelist_with_crlf_line_endings_is_parsed(self):
        md_args = self.run_raw_filelist(b"blasLt/common/README.md\r\n")
        self.assertIn("--ops=blasLtMatmul", md_args)
        op_args = self.run_raw_filelist(
            b"README.md\r\ntest/axpy/caxpy/arch35/caxpy_test.cpp\r\n"
        )
        self.assertIn("--ops=caxpy,blasLtMatmul", op_args)


if __name__ == "__main__":
    unittest.main()
