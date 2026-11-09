#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared helpers for the aclblasCrot python gates.

Keeping the gtest invocation and output parsing here avoids duplicating the same block in
run_self_test.py and run_memcheck.py.
"""

import json
import re
import subprocess
from pathlib import Path

RUN_RE = re.compile(r"^\[ RUN      \] (\S+)")
PASSED_RE = re.compile(r"^\[  PASSED  \] (\d+) tests?\.")
FAILED_RE = re.compile(r"^\[  FAILED  \] (\d+) tests?")


def repo_root(value):
    if value:
        return Path(value).resolve()
    return Path(__file__).resolve().parents[3]


def find_binary(repo):
    candidates = [
        repo / "build" / "test" / "rot" / "crot" / "crot_test",
        repo / "build" / "test" / "crot" / "crot_test",
    ]
    for path in candidates:
        if path.is_file() and os_access_x(path):
            return path
    return None


def os_access_x(path):
    try:
        return path.stat().st_mode & 0o111 != 0
    except OSError:
        return False


def run_binary(command, repo, env, timeout):
    return subprocess.run(
        command,
        cwd=str(repo),
        env=env,
        timeout=timeout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def parse_gtest_output(text):
    executed = []
    passed = None
    failed = None
    for line in text.splitlines():
        match = RUN_RE.match(line)
        if match:
            executed.append(match.group(1))
            continue
        match = PASSED_RE.match(line)
        if match:
            passed = int(match.group(1))
            continue
        match = FAILED_RE.match(line)
        if match:
            failed = int(match.group(1))
    return {"executed": executed, "passed": passed, "failed": failed}


def report(logger, summary):
    logger.info(json.dumps(summary, indent=2))


def write_summary(path, summary, logger):
    try:
        Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("cannot write %s (%s)", path, exc)
