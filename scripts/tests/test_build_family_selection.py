# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check actual build.sh selection before the CANN environment check."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "build.sh"


class BuildFamilySelectionTest(unittest.TestCase):
    def run_selection(self, *options):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            script = root / "build.sh"
            script.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
            script.chmod(0o700)
            for family, op, arch in (
                ("geqrf_batched", "sgeqrf_batched", "arch35"),
                ("copy", "scopy", "arch22"),
            ):
                path = root / "test" / family / op
                (path / arch).mkdir(parents=True)
                (path / "CMakeLists.txt").touch()
            marker = root / "build" / "existing-output"
            marker.parent.mkdir()
            marker.write_text("keep")
            env = os.environ.copy()
            for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "SOC_VERSION"):
                env.pop(key, None)
            result = subprocess.run(
                [str(script), "--soc=ascend910b", *options],
                cwd=root, env=env, shell=False, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(marker.read_text(), "keep")
            self.assertNotEqual(result.returncode, 0)
            return result.stdout + result.stderr

    def test_unsupported_family_fails_before_full_discovery_or_build(self):
        for run in ([], ["--run"]):
            with self.subTest(run=run):
                output = self.run_selection("--ops=geqrf_batched", *run)
                self.assertIn("No requested operators support SOC", output)
                self.assertNotIn("Auto-discovered tests:", output)
                self.assertNotIn("environment is not configured", output)

    def test_supported_family_still_expands(self):
        output = self.run_selection("--ops=copy", "--run")
        self.assertIn("BUILD_OPS=scopy, RUN_TEST=ON", output)
        self.assertNotIn("Auto-discovered tests:", output)

    def test_mixed_families_keep_supported_members(self):
        output = self.run_selection("--ops=geqrf_batched,copy", "--run")
        self.assertIn("BUILD_OPS=scopy, RUN_TEST=ON", output)
        self.assertNotIn("Auto-discovered tests:", output)

    def test_no_selection_still_discovers_all_tests(self):
        output = self.run_selection("--run")
        self.assertIn("Auto-discovered tests: scopy,sgeqrf_batched", output)


if __name__ == "__main__":
    unittest.main()
