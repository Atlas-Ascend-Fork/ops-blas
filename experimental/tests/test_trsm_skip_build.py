#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise both documented run.sh entry points; CANN and numerical stages are isolated."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
VARIANTS = (
    ("aclblasTrsmBatched", "trsm_batched_test", ["1.0"]),
    ("aclblasCtrsmBatched2", "ctrsm_batched_test", ["1.0", "0.0"]),
)
REQUIRED = ["16", "16", "1", "0", "0", "0", "0"]


TOOL_PROGRAM = (
    "#!"
    + sys.executable
    + "\n"
    + """import json, os, sys
from pathlib import Path
command = Path(sys.argv[0]).name
args = sys.argv[1:]
kind = "build" if command in ("cmake", "make") else "kernel"
if command == "python3":
    kind = "generate" if args[0].endswith("gen_data.py") else "verify"
    args = args[1:]
if kind != "build":
    try:
        for value in args:
            float(value)
    except ValueError:
        raise SystemExit(2)
with open(os.environ["SYNTHETIC_TRACE"], "a") as output:
    output.write(json.dumps({"kind": kind, "args": args}) + "\\n")
"""
)


class TestTrsmSkipBuild(unittest.TestCase):
    def run_script(self, variant, arguments):
        name, kernel, _ = variant
        with tempfile.TemporaryDirectory(prefix="synthetic-trsm-cli-") as temp:
            directory = Path(temp)
            project = directory / "project"
            (project / "build/test").mkdir(parents=True)
            shutil.copy2(ROOT / name / "run.sh", project / "run.sh")
            toolkit = directory / "toolkit"
            toolkit.mkdir()
            (toolkit / "set_env.sh").write_text("# Synthetic empty CANN environment\n")
            tools = directory / "bin"
            tools.mkdir()
            trace = directory / "trace.jsonl"
            for target in [
                tools / "python3",
                tools / "cmake",
                tools / "make",
                project / "build/test" / kernel,
            ]:
                target.write_text(TOOL_PROGRAM)
                target.chmod(0o700)
            environment = os.environ.copy()
            environment.update(
                ASCEND_HOME_PATH=str(toolkit),
                SYNTHETIC_TRACE=str(trace),
                PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
            )
            result = subprocess.run(
                [BASH, str(project / "run.sh"), *arguments],
                env=environment,
                capture_output=True,
                text=True,
            )
            stages = (
                [json.loads(line) for line in trace.read_text().splitlines()]
                if trace.exists()
                else []
            )
            return result, stages

    def assert_run(self, variant, arguments, expected, skip):
        result, stages = self.run_script(variant, arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        generated = [
            stage.get("args") for stage in stages if stage.get("kind") == "generate"
        ]
        self.assertEqual(generated, [expected])
        kinds = [stage.get("kind") for stage in stages]
        self.assertEqual(kinds.count("kernel"), 1)
        self.assertEqual(kinds.count("verify"), 1)
        self.assertEqual(kinds.count("build"), 0 if skip else 2)
        for stage in stages:
            self.assertNotIn("--skip-build", stage.get("args", []))

    def test_skip_build_with_omitted_optional_values(self):
        for variant in VARIANTS:
            _, _, alpha = variant
            for args in (["--skip-build"], REQUIRED + ["--skip-build"]):
                with self.subTest(variant=variant[0], arguments=args):
                    self.assert_run(variant, args, REQUIRED + alpha, skip=True)
        complex_variant = VARIANTS[1]
        self.assert_run(
            complex_variant,
            REQUIRED + ["2.5", "--skip-build"],
            REQUIRED + ["2.5", "0.0"],
            skip=True,
        )

    def test_explicit_alpha_skip_build_controls(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant[0]):
                alpha = (
                    ["2.5"] if variant[0] == "aclblasTrsmBatched" else ["2.5", "-0.5"]
                )
                self.assert_run(
                    variant,
                    REQUIRED + alpha + ["--skip-build"],
                    REQUIRED + alpha,
                    skip=True,
                )

    def test_default_build_controls(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant[0]):
                self.assert_run(variant, [], REQUIRED + variant[2], skip=False)


if __name__ == "__main__":
    unittest.main()
