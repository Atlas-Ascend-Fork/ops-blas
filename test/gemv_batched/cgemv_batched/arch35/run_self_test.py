#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Self-check gate for the cgemv_batched arch35 test suite.

Runs the whole CSV case set and verifies that every case was actually executed.
A silently truncated CSV (dropped rows, a case that failed to construct) still
makes the test binary exit 0, so the executed case count has to be checked
against the CSV here, in the gate, rather than trusted from a green run.

Usage:
    python3 run_self_test.py [--test-bin PATH] [--csv PATH] [--json OUT]
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cgemv_batched_gate_common as common
from cgemv_batched_gate_common import log

# Perf cases are the TC_PF_* group; the rest are precision / negative cases.
PERF_PREFIX = "TC_PF"

# gtest prints one "[       OK ] Suite/Test.CsvDriven/<case_name> (N ms)" line per
# executed parameterised instance: the instance name (the CSV case_name) follows
# the LAST slash. The perf report line carries the same name.
OK_RE = re.compile(r"^\[\s*OK\s*\]\s+\S+/([^/\s]+)\s+\(", re.M)
FAILED_RE = re.compile(r"^\[\s*FAILED\s*\]\s+\S+/([^/\s]+)", re.M)
PERFREG_RE = re.compile(r"\[perfreg\]\s+(\S+)")


def parse_args():
    parser = argparse.ArgumentParser(description="cgemv_batched arch35 self-check gate")
    parser.add_argument(
        "--test-bin",
        default=None,
        help="path to cgemv_batched_test; default searches the build tree",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(common.here(), "cgemv_batched_test.csv"),
        help="path to cgemv_batched_test.csv",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="write a JSON summary to this path",
    )
    return parser.parse_args()


def load_csv_cases(csv_path):
    """Return (all_case_names, perf_case_names) from the CSV data rows."""
    names = []
    perf = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("case_name") or "").strip()
            if not name:
                continue
            names.append(name)
            if name.startswith(PERF_PREFIX):
                perf.append(name)
    return names, perf


def parse_executed(output):
    """Return (executed_case_names, failed_case_names) from a gtest log."""
    executed = set(OK_RE.findall(output))
    executed.update(PERFREG_RE.findall(output))
    return executed, FAILED_RE.findall(output)


def build_summary(csv_names, csv_perf, executed, failed, exit_code):
    """Assemble the reported counts and the machine-readable summary dict."""
    missing = [n for n in csv_names if n not in executed]
    missing_perf = [n for n in csv_perf if n not in executed]
    executed_perf = [n for n in executed if n.startswith(PERF_PREFIX)]
    summary = {
        "op": common.OP_NAME,
        "csv_cases": len(csv_names),
        "executed_cases": len(executed),
        "csv_perf_cases": len(csv_perf),
        "executed_perf_cases": len(executed_perf),
        "failed_cases": sorted(set(failed)),
        "missing_cases": missing[:50],
        "test_exit_code": exit_code,
    }
    return summary, missing, missing_perf


def report_counts(summary, missing, missing_perf):
    log.info("=" * 60)
    log.info("cgemv_batched self-check")
    log.info("=" * 60)
    log.info("  CSV cases            : %d", summary["csv_cases"])
    log.info("  executed cases       : %d", summary["executed_cases"])
    log.info("  CSV perf cases       : %d", summary["csv_perf_cases"])
    log.info("  executed perf cases  : %d", summary["executed_perf_cases"])
    log.info("  failed cases         : %d", len(summary["failed_cases"]))
    log.info("  missing cases        : %d", len(missing))
    if missing:
        log.info("  first missing        : %s", ", ".join(missing[:10]))
    if missing_perf:
        log.info("  missing perf cases   : %s", ", ".join(missing_perf[:10]))


def main():
    args = parse_args()

    test_bin = common.resolve_test_bin_or_exit(args.test_bin)
    if test_bin is None:
        return 2

    csv_names, csv_perf = load_csv_cases(args.csv)
    if not csv_names:
        log.info("[FAIL] no cases loaded from %s", args.csv)
        return 2

    run_dir = os.path.dirname(test_bin)
    exit_code, output = common.run_captured([test_bin, run_dir], run_dir)

    executed, failed = parse_executed(output)
    summary, missing, missing_perf = build_summary(csv_names, csv_perf, executed, failed, exit_code)

    report_counts(summary, missing, missing_perf)

    ok = True
    if exit_code != 0:
        log.info("[FAIL] test binary exited with code %d", exit_code)
        ok = False
    if missing:
        log.info("[FAIL] %d CSV case(s) were not executed", len(missing))
        ok = False
    if missing_perf:
        log.info("[FAIL] %d perf case(s) were not executed", len(missing_perf))
        ok = False

    log.info("[PASS] cgemv_batched self-check" if ok else "[FAIL] cgemv_batched self-check")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
