#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Memory-safety runner for the aclblasCrot GTest.

Runs the success-path cases that actually launch the kernel under the CANN mssanitizer
(msmemcheck) when available, then under valgrind, and otherwise falls back to a plain run
with asynchronous launch checking enabled. Because a wrong index conversion silently reads
or writes outside the stride slots, the selected filter must contain real kernel launches and
not only the negative cases.

Requires the CANN environment to be sourced first (for example
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`), otherwise the test binary cannot
load libplatform.so and the run exits with code 127.

Usage
  python3 test/rot/crot/run_memcheck.py --repo <repo-root> --device 0
  python3 test/rot/crot/run_memcheck.py --filter '*TC_L0_*' --timeout 3600
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from test_runner_utils import find_binary, parse_gtest_output, repo_root, report, run_binary, write_summary

DEFAULT_FILTER = "*TC_L0_*:*TC_INC_*:*TC_CS_*:*TC_EX_STBIG_*"
MIN_EXECUTED_CASES = 60
LOGGER = logging.getLogger("crot_memcheck")


def parse_args():
    parser = argparse.ArgumentParser(description="aclblasCrot memory-safety runner")
    parser.add_argument("--repo", default=None, help="ops-blas repo root")
    parser.add_argument("--filter", default=DEFAULT_FILTER, help="GTest filter")
    parser.add_argument("--device", type=int, default=None, help="ASCEND_DEVICE_ID")
    parser.add_argument("--timeout", type=int, default=3600, help="timeout in seconds")
    parser.add_argument("--output", default="memcheck_summary.json", help="summary JSON path")
    return parser.parse_args()


def find_tool():
    found = shutil.which("msmemcheck")
    if found:
        return [found, "--tool=memcheck", "--leak-check=yes", "--"]
    for tool in Path("/usr/local/Ascend").glob("*/tools/msmemcheck/msmemcheck"):
        if tool.is_file():
            return [str(tool), "--tool=memcheck", "--leak-check=yes", "--"]
    found = shutil.which("valgrind")
    if found:
        return [found, "--error-exitcode=9", "--leak-check=full", "--errors-for-leak-kinds=definite", "--"]
    return None


def check_run(proc, parsed, tool):
    errors = []
    if proc.returncode != 0:
        errors.append("runner exited with %d" % proc.returncode)
    if parsed["failed"]:
        errors.append("%d gtest failures" % parsed["failed"])
    if len(parsed["executed"]) < MIN_EXECUTED_CASES:
        errors.append("only %d cases executed, expected at least %d kernel-launching cases"
                      % (len(parsed["executed"]), MIN_EXECUTED_CASES))
    if tool and tool[0].endswith("msmemcheck") and "No error detected" not in proc.stdout:
        errors.append("mssanitizer did not report 'No error detected'")
    return errors


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    repo = repo_root(args.repo)
    binary = find_binary(repo)
    tool = find_tool()
    summary = {"binary": str(binary), "filter": args.filter, "tool": tool[0] if tool else "none (plain run)"}

    if binary is None:
        LOGGER.error("test binary not found, build it first")
        return 2

    env = dict(os.environ)
    env["ASCEND_LAUNCH_BLOCKING"] = "1"
    if args.device is not None:
        env["ASCEND_DEVICE_ID"] = str(args.device)
    command = (tool or []) + [str(binary), "--gtest_filter=" + args.filter]
    proc = run_binary(command, repo, env, args.timeout)
    parsed = parse_gtest_output(proc.stdout)

    summary.update({"return_code": proc.returncode, "executed": len(parsed["executed"]),
                    "gtest_passed": parsed["passed"], "gtest_failed": parsed["failed"]})
    summary["errors"] = check_run(proc, parsed, tool)
    write_summary(args.output, summary, LOGGER)
    report(LOGGER, summary)
    return 0 if not summary["errors"] else 4


if __name__ == "__main__":
    sys.exit(main())
