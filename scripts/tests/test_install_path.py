# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ops-blas run 包安装脚本，参考 ops-nn install.sh 实现

"""Filesystem regressions for packaged install-path argument parsing."""
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/package/install.sh"


class InstallPathTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="ops-blas-installer-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        package = self.root / "pkg"
        info = package / "share/info/ops_blas"
        info.mkdir(parents=True)
        (package / "lib64").mkdir()
        (package / "include").mkdir()
        self.script = (info / "install.sh").resolve()
        self.script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
        self.script.chmod(0o700)
        # Installation only copies these bytes; no shared library is loaded.
        (package / "lib64/libops_blas_fixture.so").write_text("host fixture", encoding="utf-8")
        (package / "include/cann_ops_blas_fixture.h").write_text("host fixture", encoding="utf-8")
        self.fallback = self.root / "fallback"

    def install(self, destination, extra=()):
        # The first explicit path keeps every baseline run in the temporary
        # directory even when the later path is incorrectly ignored.
        return subprocess.run(
            [str(self.script), *extra, "--install", "--quiet",
             "--install-path=" + str(self.fallback),
             "--install-path=" + str(destination)],
            capture_output=True, text=True, check=False, timeout=20, shell=False,
        )

    def assert_installed(self, destination, extra=()):
        result = self.install(destination, extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((destination / "cann/lib64/libops_blas_fixture.so").read_text(), "host fixture")
        self.assertFalse(self.fallback.exists())

    def test_ordinary_path_is_used(self):
        self.assert_installed(self.root / "selected")

    def test_run_substring_path_is_used(self):
        for name in ("selected.runtime", "selected.run"):
            with self.subTest(name=name):
                self.assert_installed(self.root / name)

    def test_makeself_run_argument_remains_ignored(self):
        self.assert_installed(self.root / "selected", ("./package.run",))

    def test_invalid_run_substring_path_is_rejected(self):
        result = self.install(self.root / "invalid.runtime path")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fallback.exists())

    def test_empty_path_is_rejected(self):
        result = self.install("")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fallback.exists())


if __name__ == "__main__":
    unittest.main()
