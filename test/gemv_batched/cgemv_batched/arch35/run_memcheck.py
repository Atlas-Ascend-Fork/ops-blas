#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Memory-safety gate for the cgemv_batched arch35 kernel.

The kernel moves bare addresses around: strided DataCopyPad geometry
(srcStride = (lda - outCnt) * 8, padded dstStride), the column-tile UB layout
and the 2D path's SetAtomicAdd cross-core merge into y. An out-of-bounds GM
access there is silent memory corruption that a precision comparison does not
notice, so it is checked here with mssanitizer memcheck instead.

Each group runs in its own process, because mssanitizer aborts the process on
the first report and a group that shares a process with a failing one would
hide its own result. The groups cover every case-name prefix in the CSV, and at
least one success-path group (contract / extended) must actually run cases, so
the gate can never pass while checking nothing.

Usage:
    python3 run_memcheck.py [--test-bin PATH] [--json OUT] [--groups a,b,c]
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cgemv_batched_gate_common as common
from cgemv_batched_gate_common import log

# Group name -> case-name prefixes. "launch" holds the success-path cases that
# really enter the kernel, so the gate can never pass by only covering the
# early-return paths. Every prefix present in cgemv_batched_test.csv is assigned
# to exactly one group, so a run with all groups covers the whole CSV.
GROUPS = {
    # Basic shapes and value-range cases: the smallest successful kernel runs.
    "contract": ["TC_L0_", "TC_SQ_", "TC_CV_"],
    # Strided operands and lda padding: the DataCopyPad geometry under test.
    "layout": ["TC_INC_", "TC_LD_", "TC_MN_"],
    # Inf / NaN / alpha-beta special values: scalar folding edge cases.
    "special": ["TC_AB_", "TC_ED_", "TC_FL_"],
    # Extended sizes: larger tiles and the 2D column-split path.
    "extended": ["TC_EX_"],
    # Performance cases: timed kernel launches on the big shapes.
    "perf": ["TC_PF_"],
}

# At least one of these groups must have run cases, otherwise the gate would be
# vacuous (a filter that matches nothing also exits 0).
REQUIRED_LAUNCH_GROUPS = ["contract", "extended"]

MSSANITIZER_REL = os.path.join("tools", "mssanitizer", "bin", "mssanitizer")
MSSANITIZER_ROOT_HINTS = (
    "/usr/local/Ascend/cann-9.1.0",
    "/usr/local/Ascend/ascend-toolkit/latest",
)


def parse_args():
    parser = argparse.ArgumentParser(description="cgemv_batched arch35 memcheck gate")
    parser.add_argument(
        "--test-bin",
        default=None,
        help="path to cgemv_batched_test; default searches the build tree",
    )
    parser.add_argument("--json", default=None, help="write a JSON summary to this path")
    parser.add_argument(
        "--mssanitizer",
        default=None,
        help="path to the mssanitizer launcher; default searches ASCEND_HOME_PATH",
    )
    parser.add_argument(
        "--groups",
        default=",".join(GROUPS.keys()),
        help="comma-separated subset of groups to run",
    )
    parser.add_argument(
        "--skip-tool",
        action="store_true",
        help="only verify the case filter resolves to a non-empty set (no NPU run)",
    )
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="extra options forwarded to mssanitizer (e.g. --check-unused-memory=yes)",
    )
    return parser.parse_args()


def find_mssanitizer():
    roots = []
    env = os.environ.get("ASCEND_HOME_PATH")
    if env:
        roots.append(env)
    roots.extend(MSSANITIZER_ROOT_HINTS)
    for root in roots:
        cand = os.path.join(root, MSSANITIZER_REL)
        if os.path.isfile(cand):
            return cand
    return shutil.which("mssanitizer")


def build_group_cmd(test_bin, group, mssanitizer, skip_tool, extra_args):
    """Return the command line that runs one group in its own process."""
    run_dir = os.path.dirname(test_bin)
    if skip_tool:
        return [test_bin, run_dir], run_dir

    # --check-unused-memory is deliberately off by default: the wrapper uploads
    # whole per-batch strides (which include tail padding the kernel never
    # reads), so "unused memory" reports describe the test harness layout rather
    # than an operator defect. Out-of-bounds (the real risk for the strided
    # DataCopyPad geometry) and leaks stay on. extra_args opts back in.
    cmd = [
        mssanitizer,
        "--tool=memcheck",
        "--leak-check=yes",
        "--padding=64",
    ]
    cmd.extend(extra_args)
    cmd.extend([test_bin, run_dir])
    return cmd, run_dir


def classify_group(exit_code, output):
    """Return (group_ok, ran_any, result_dict) for one finished group."""
    # mssanitizer prints "ERROR: ..." / "SUMMARY: ..." lines on a finding, and a
    # clean run ends with "No error detected." style output.
    sanitizer_error = ("ERROR: " in output) or ("SUMMARY: " in output)
    clean_marker = ("No error detected" in output) or ("no error detected" in output)
    # gtest prints one "[       OK ] ..." line per executed instance.
    ran_any = ("[ RUN " in output) or ("[ OK " in output) or ("[perfreg] " in output)

    group_ok = (exit_code == 0) and (not sanitizer_error) and ran_any
    result = {
        "exit_code": exit_code,
        "sanitizer_error": sanitizer_error,
        "ran_any_test": ran_any,
        "clean_marker_seen": clean_marker,
    }
    return group_ok, ran_any, result


def log_group_failure(output):
    """Print the tail of a failed group's log, where the report lands."""
    tail = "\n".join(output.strip().splitlines()[-40:])
    if tail:
        log.info("-" * 60)
        log.info("%s", tail)
        log.info("-" * 60)


def run_groups(test_bin, groups, mssanitizer, skip_tool, extra_args):
    """Run every requested group. Returns (results, started_groups, ok)."""
    results = {}
    ok = True
    started_groups = set()

    log.info("=" * 60)
    log.info("cgemv_batched memcheck")
    log.info("=" * 60)
    for group in groups:
        hints = GROUPS[group]
        cmd, run_dir = build_group_cmd(test_bin, group, mssanitizer, skip_tool, extra_args)
        # The suite's custom main turns this into a gtest filter.
        env = dict(os.environ)
        env["BLAS_TEST_CASES"] = "%s:%s" % (common.OP_NAME, ",".join(hints))
        exit_code, output = common.run_captured(cmd, run_dir, env)

        group_ok, ran_any, result = classify_group(exit_code, output)
        results[group] = result
        # "started" tracks whether this group actually executed test cases, so
        # the vacuous-gate check stays meaningful even when the group is dirty:
        # a group that ran cases but reported a finding still proves the kernel
        # was launched.
        if ran_any:
            started_groups.add(group)

        status = "PASS" if group_ok else "FAIL"
        log.info(
            "  [%s] %-9s rc=%d sanitizer_error=%s ran=%s",
            status,
            group,
            exit_code,
            result["sanitizer_error"],
            ran_any,
        )
        if not group_ok:
            ok = False
            log_group_failure(output)

    return results, started_groups, ok


def check_launch_covered(started_groups):
    """Require at least one kernel-launching group to have actually run cases."""
    launch_covered = bool(started_groups & set(REQUIRED_LAUNCH_GROUPS))
    if not launch_covered:
        log.info(
            "[FAIL] no kernel-launching group ran (need one of: %s); the gate would be vacuous",
            ", ".join(REQUIRED_LAUNCH_GROUPS),
        )
    return launch_covered


def write_summary(path, results, launch_covered):
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"op": common.OP_NAME, "groups": results, "launch_covered": launch_covered},
            fh,
            indent=2,
        )


def main():
    args = parse_args()

    test_bin = common.resolve_test_bin_or_exit(args.test_bin)
    if test_bin is None:
        return 2

    mssanitizer = None
    if not args.skip_tool:
        mssanitizer = args.mssanitizer or find_mssanitizer()
        if mssanitizer is None:
            log.info("[FAIL] mssanitizer not found; set ASCEND_HOME_PATH or pass --mssanitizer")
            return 2

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        log.info("[FAIL] unknown group(s): %s (known: %s)", ", ".join(unknown), ", ".join(GROUPS))
        return 2

    results, started_groups, ok = run_groups(test_bin, groups, mssanitizer, args.skip_tool, args.extra_args)

    launch_covered = check_launch_covered(started_groups)
    if not launch_covered:
        ok = False

    log.info("[PASS] cgemv_batched memcheck" if ok else "[FAIL] cgemv_batched memcheck")

    write_summary(args.json, results, launch_covered)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
