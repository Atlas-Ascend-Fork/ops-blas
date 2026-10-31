# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise multi-header selection against real Git diffs."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(
    os.environ.get(
        "RUN_EXAMPLE_SCRIPT", Path(__file__).resolve().parents[1] / "run_example.sh"
    )
)


class IncludeHeaderFallbackTests(unittest.TestCase):
    def run_case(self, first_api, second_api, reverse=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts/ci").mkdir(parents=True)
            (root / "include").mkdir()
            (root / "test/gemm_ex").mkdir(parents=True)
            (root / "test/gemm_batched_ex").mkdir(parents=True)
            (root / "scripts/ci/run_example.sh").write_text(
                SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (root / "build.sh").write_text(
                '#!/bin/sh\nprintf "BUILD_ARGS=%s\\n" "$*"\n', encoding="utf-8"
            )
            one, two = (
                root / "include/cann_ops_blas.h",
                root / "include/cann_ops_blas_common.h",
            )
            one.write_text("/* baseline */\n", encoding="utf-8")
            two.write_text("/* baseline */\n", encoding="utf-8")

            def git(*args):
                subprocess.run(
                    ["git", *args], cwd=root, check=True, capture_output=True, text=True
                )

            git("init", "-q")
            git("config", "user.name", "test")
            git("config", "user.email", "test@example.invalid")
            git("add", ".")
            git("commit", "-qm", "baseline")
            git("update-ref", "refs/remotes/origin/master", "HEAD")
            one.write_text("/* baseline */\n" + first_api + "\n", encoding="utf-8")
            two.write_text("/* baseline */\n" + second_api + "\n", encoding="utf-8")
            git("add", "include")
            git("commit", "-qm", "change headers")
            names = ["include/cann_ops_blas.h", "include/cann_ops_blas_common.h"]
            if reverse:
                names.reverse()
            (root / "files.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
            script = root / "scripts/ci/run_example.sh"
            script.chmod(0o755)
            result = subprocess.run(
                [str(script), str(root / "files.txt"), "ascend950"],
                cwd=root,
                env={**os.environ, "TARGET_BRANCH": "master"},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return result.stdout

    def test_unresolved_header_falls_back_in_either_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                output = self.run_case(
                    "int aclblasGemmEx(void);", "#define SHARED_CHANGE 1", reverse
                )
                self.assertIn("Need run all: true", output)
                self.assertNotIn("--ops=", output)

    def test_fully_resolved_headers_keep_targeted_selection(self):
        output = self.run_case(
            "int aclblasGemmEx(void);", "int aclblasGemmBatchedEx(void);"
        )
        self.assertIn("Need run all: false", output)
        self.assertIn("--ops=", output)


if __name__ == "__main__":
    unittest.main()
