# Copyright (c) 2026 balckwell
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-only CI selector tests; build.sh only records generated arguments."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(os.environ.get(
    "RUN_EXAMPLE_SCRIPT", Path(__file__).resolve().parents[1] / "run_example.sh"
))


class ChangedPathSelectionTests(unittest.TestCase):
    def run_case(self, paths, smoke=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "scripts/ci/run_example.sh"
            script.parent.mkdir(parents=True)
            script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
            script.chmod(0o755)
            # Both flat and family layouts exist upstream. isamax's arch22
            # directory itself has CMakeLists.txt; gemm has flat AND nested tests.
            for directory in (
                "blasLtMatrixTransform", "gemm_ex", "isamax", "isamax/arch22",
                "axpy/caxpy", "gemm", "gemm/cgemm", "blasLtMatmul",
            ):
                target = root / "test" / directory
                target.mkdir(parents=True, exist_ok=True)
                (target / "CMakeLists.txt").touch()
            for directory in ("blasLtMatrixTransform/arch35", "gemm_ex/arch35",
                              "axpy/caxpy/arch35", "gemm/cgemm/arch35"):
                (root / "test" / directory).mkdir(parents=True, exist_ok=True)
            (root / "build.sh").write_text(
                '#!/bin/bash\nprintf "BUILD_ARGS=%s\\n" "$*"\n', encoding="utf-8"
            )
            filelist = root / "files.txt"
            filelist.write_text("\n".join(paths) + "\n", encoding="utf-8")
            result = subprocess.run(
                [str(script), str(filelist), "ascend950"], cwd=root,
                env={**os.environ, "RUN_FLAGS": "" if smoke else "--no-smoke",
                     "RUNNER_NPU_COUNT": "1"},
                capture_output=True, text=True, timeout=30, shell=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return next(line for line in result.stdout.splitlines()
                        if line.startswith("BUILD_ARGS="))

    def test_flat_architecture_paths_select_operator(self):
        for op, arch in (("blasLtMatrixTransform", "arch35"),
                         ("gemm_ex", "arch35"), ("isamax", "arch22")):
            with self.subTest(op=op, arch=arch):
                args = self.run_case([f"test/{op}/{arch}/{op}_test.cpp"])
                self.assertIn(f"--ops={op},blasLtMatmul", args)
                self.assertNotIn(f"--ops={arch}", args)

    def test_nested_members_keep_precedence_over_flat_family(self):
        for family, op in (("axpy", "caxpy"), ("gemm", "cgemm")):
            with self.subTest(family=family):
                args = self.run_case([f"test/{family}/{op}/arch35/{op}_test.cpp"])
                self.assertIn(f"--ops={op},blasLtMatmul", args)

    def test_blaslt_source_changes_fall_back_to_all(self):
        for path in ("blasLt/api/aclblaslt_transform.cpp",
                     "blasLt/matrix_transform/arch35/matrix_transform_kernel.cpp",
                     "blasLt/common/helper/host_utils.h", "blasLt/internal/types.h",
                     "blasLt/matmul/arch35/matmul.cpp"):
            with self.subTest(path=path):
                args = self.run_case([path])
                self.assertNotIn("--ops=", args)
                self.assertIn("--run --soc=ascend950", args)

    def test_blaslt_fallback_keeps_default_smoke_and_is_order_independent(self):
        paths = ["test/gemm_ex/arch35/gemm_ex_test.cpp",
                 "blasLt/api/aclblaslt_transform.cpp"]
        for ordered in (paths, paths[::-1]):
            with self.subTest(paths=ordered):
                args = self.run_case(ordered, smoke=True)
                self.assertNotIn("--ops=", args)
                self.assertIn("--smoke", args)

    def test_existing_blas_and_shared_test_paths(self):
        self.assertIn("--ops=caxpy,blasLtMatmul", self.run_case(
            ["blas/axpy/caxpy/arch35/caxpy.cpp"]))
        self.assertIn("--ops=axpy,blasLtMatmul", self.run_case(
            ["blas/axpy/arch22/caxpy_host.cpp"]))
        for path in ("blas/common/helper/host_utils.h", "test/utils/blas_test.h",
                     "test/frame/test_main.cpp"):
            with self.subTest(path=path):
                self.assertNotIn("--ops=", self.run_case([path]))
        self.assertIn("--ops=blasLtMatmul", self.run_case(
            ["blasLt/common/README.md", "other/blasLt/ignored.cpp"]))


if __name__ == "__main__":
    unittest.main()
