#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared plumbing for the cgemv_batched arch35 gate scripts.

Both gates (run_memcheck.py / run_self_test.py) have to locate the test binary,
validate the path, and run it in its own directory so it can find the CSV. That
setup used to be duplicated verbatim, so it lives here once. Output goes through
the logging module rather than print(), per the repo logging rule.
"""

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("cgemv_batched")

OP_NAME = "cgemv_batched"

# The binary expects its own directory as argv[1] so it can locate the CSV.
BIN_REL_CANDIDATES = (
    os.path.join("build", "test", "gemv_batched", OP_NAME, OP_NAME + "_test"),
    os.path.join("out", "test", "gemv_batched", OP_NAME, OP_NAME + "_test"),
)


def here():
    """Absolute path of the directory holding this module (the arch35 dir)."""
    return os.path.dirname(os.path.abspath(__file__))


def repo_root():
    """Absolute path of the repository root, four levels above the arch35 dir."""
    return os.path.normpath(os.path.join(here(), "..", "..", "..", ".."))


def find_test_bin():
    """Search the build tree for cgemv_batched_test; None when not built."""
    root = repo_root()
    for rel in BIN_REL_CANDIDATES:
        cand = os.path.normpath(os.path.join(root, rel))
        if os.path.isfile(cand):
            return cand
    return None


def resolve_test_bin(explicit):
    """Return (test_bin, error). error is None when the binary is usable."""
    test_bin = explicit or find_test_bin()
    if test_bin is None:
        return None, "%s binary not found; pass --test-bin explicitly" % (OP_NAME + "_test")
    test_bin = os.path.normpath(os.path.abspath(test_bin))
    if not os.path.isfile(test_bin):
        return None, "test binary does not exist: %s" % test_bin
    return test_bin, None


def run_captured(cmd, cwd, env=None):
    """Run cmd with cwd, capturing merged stdout/stderr. Returns (rc, output)."""
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return proc.returncode, proc.stdout


def resolve_test_bin_or_exit(explicit):
    """Return test_bin, or None after logging the reason it is unusable."""
    test_bin, err = resolve_test_bin(explicit)
    if err is not None:
        log.info("[FAIL] %s", err)
        return None
    return test_bin
