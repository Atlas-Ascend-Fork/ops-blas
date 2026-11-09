#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Self-test gate for the aclblasCrot GTest.

Validates the CSV composition (total case count, performance case count, unique names), runs
the whole suite on the device, verifies that every CSV case was executed and that nothing
failed, then writes a JSON summary. Exits non-zero on any mismatch so the gate cannot pass
silently when the CSV is truncated or rewritten with different line endings.

Requires the CANN environment to be sourced first (for example
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`), otherwise the test binary cannot
load libplatform.so and the run exits with code 127.

Usage
  python3 test/rot/crot/run_self_test.py --repo <repo-root> --device 0
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from test_runner_utils import find_binary, parse_gtest_output, repo_root, report, run_binary, write_summary

EXPECTED_CSV_CASES = 1206
EXPECTED_PERF_CASES = 200
EXPECTED_FIXED_TESTS = 7
PERF_RE = re.compile(r"^\[PERF\] n=(\d+) kernel_best_us=([0-9.]+) kernel_avg_us=([0-9.]+)")
LOGGER = logging.getLogger("crot_self_test")


def parse_args():
    parser = argparse.ArgumentParser(description="aclblasCrot self-test gate")
    parser.add_argument("--repo", default=None, help="ops-blas repo root")
    parser.add_argument("--device", type=int, default=None, help="ASCEND_DEVICE_ID")
    parser.add_argument("--timeout", type=int, default=7200, help="timeout in seconds")
    parser.add_argument("--output", default="self_test_summary.json", help="summary JSON path")
    return parser.parse_args()


def csv_stats(csv_path):
    text = csv_path.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.strip()][1:]
    names = [row.split(",")[0] for row in rows]
    perf = sum(1 for name in names if name.startswith("TC_PF_"))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    return len(rows), perf, duplicates


def check_csv(csv_path):
    total, perf, duplicates = csv_stats(csv_path)
    errors = []
    if total != EXPECTED_CSV_CASES:
        errors.append("csv case count %d != expected %d" % (total, EXPECTED_CSV_CASES))
    if perf != EXPECTED_PERF_CASES:
        errors.append("perf case count %d != expected %d" % (perf, EXPECTED_PERF_CASES))
    if duplicates:
        errors.append("duplicate case names: %s" % ", ".join(duplicates))
    return {"csv_cases": total, "perf_cases": perf, "duplicate_case_names": duplicates}, errors


def parse_perf(text):
    rows = []
    for line in text.splitlines():
        match = PERF_RE.match(line)
        if match:
            rows.append({"n": int(match.group(1)), "best_us": float(match.group(2)),
                         "avg_us": float(match.group(3))})
    return rows


def check_run(proc, parsed, perf_rows):
    csv_executed = sum(1 for name in parsed["executed"] if "CsvDriven" in name)
    errors = []
    if proc.returncode != 0:
        errors.append("gtest exited with %d" % proc.returncode)
    if parsed["failed"]:
        errors.append("%d gtest failures" % parsed["failed"])
    if csv_executed != EXPECTED_CSV_CASES:
        errors.append("executed csv cases %d != expected %d" % (csv_executed, EXPECTED_CSV_CASES))
    passed = parsed["passed"]
    if passed is None or passed < EXPECTED_CSV_CASES + EXPECTED_FIXED_TESTS:
        errors.append("gtest passed count %s is below %d" % (passed, EXPECTED_CSV_CASES + EXPECTED_FIXED_TESTS))
    if not perf_rows:
        errors.append("performance gate produced no measurement")
    return csv_executed, errors


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    repo = repo_root(args.repo)
    binary = find_binary(repo)
    csv_path = Path(__file__).resolve().parent / "arch35" / "crot_test.csv"
    summary = {"binary": str(binary), "csv": str(csv_path)}

    if binary is None:
        LOGGER.error("test binary not found, build it first")
        return 2
    if not csv_path.is_file():
        LOGGER.error("crot_test.csv not found")
        return 2

    statistics, errors = check_csv(csv_path)
    summary.update(statistics)
    if errors:
        summary["errors"] = errors
        write_summary(args.output, summary, LOGGER)
        report(LOGGER, summary)
        return 3

    env = dict(os.environ)
    if args.device is not None:
        env["ASCEND_DEVICE_ID"] = str(args.device)
    proc = run_binary([str(binary)], repo, env, args.timeout)
    parsed = parse_gtest_output(proc.stdout)
    perf_rows = parse_perf(proc.stdout)
    csv_executed, errors = check_run(proc, parsed, perf_rows)

    summary.update({
        "return_code": proc.returncode,
        "executed_total": len(parsed["executed"]),
        "executed_csv": csv_executed,
        "gtest_passed": parsed["passed"],
        "gtest_failed": parsed["failed"],
        "perf": perf_rows,
        "errors": errors,
    })
    write_summary(args.output, summary, LOGGER)
    report(LOGGER, summary)
    return 0 if not errors else 4


if __name__ == "__main__":
    sys.exit(main())
