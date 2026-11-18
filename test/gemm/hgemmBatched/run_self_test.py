#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate aclblasHgemmBatched CSV completeness and run every accuracy case.

The CSV must hold 1227 unique rows (1002 accuracy-success + 200 TC_PF performance
shapes + 25 negative/boundary).  Every accuracy row is executed as a CsvDriven
GTest case, together with the NullHandle and CsvCaseCount contract tests; the run
fails on any failure, skip, timeout or executed-name/count mismatch.  A
self_test_summary.json is written next to the run artifacts.
"""
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

DIRECTORY = Path(__file__).resolve().parent
CSV_NAME = "arch35/hgemmBatched_test.csv"
EXPECTED_ROWS = 1227
EXPECTED_ACCURACY_SUCCESS = 1002
EXPECTED_PERF = 200
EXPECTED_NEGATIVE = 25
CONTRACT_TESTS = ("HgemmBatchedCsvTest.CsvCaseCount", "HgemmBatchedTest.NullHandle")
SUITE = "HgemmBatched"


def read_cases():
    path = DIRECTORY / CSV_NAME
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    names = [row.get("case_name", "") for row in rows]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"{CSV_NAME}: expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if len(set(names)) != EXPECTED_ROWS:
        raise ValueError(f"{CSV_NAME}: duplicate case names present")
    if not all(re.fullmatch(r"[A-Za-z0-9_]+", name) for name in names):
        raise ValueError(f"{CSV_NAME}: empty or unsafe case name")
    perf = [row for row in rows if row["case_name"].startswith("TC_PF_")]
    negative = [row for row in rows if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS"]
    accuracy_success = []
    for row in rows:
        if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS":
            continue
        if row["case_name"].startswith("TC_PF_"):
            continue
        accuracy_success.append(row)
    if len(perf) != EXPECTED_PERF:
        raise ValueError(f"{CSV_NAME}: expected {EXPECTED_PERF} TC_PF rows, got {len(perf)}")
    if len(negative) != EXPECTED_NEGATIVE:
        raise ValueError(f"{CSV_NAME}: expected {EXPECTED_NEGATIVE} negative rows, got {len(negative)}")
    if len(accuracy_success) != EXPECTED_ACCURACY_SUCCESS:
        raise ValueError(
            f"{CSV_NAME}: expected {EXPECTED_ACCURACY_SUCCESS} accuracy-success rows, got {len(accuracy_success)}"
        )
    return path, rows


def expected_tests(rows):
    names = [f"{SUITE}/{SUITE}Test.CsvDriven/{row['case_name']}" for row in rows]
    return names + list(CONTRACT_TESTS)


def test_completed_cleanly(test):
    if test.get("status") != "run":
        return False
    if test.get("result") != "completed":
        return False
    if test.find("failure") is not None:
        return False
    return test.find("skipped") is None


def validate_xml(path, expected):
    root = ET.parse(path).getroot()
    tests = root.findall(".//testcase")
    actual = [test.get("classname", "") + "." + test.get("name", "") for test in tests]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Executed tests do not match expected CSV+contract names/count")
    for test in tests:
        if not test_completed_cleanly(test):
            raise ValueError(f"Failed, skipped or incomplete test: {test.get('name')}")
    if int(root.get("tests", "-1")) != len(expected):
        raise ValueError("GTest total count differs from CSV plus contract tests")
    if any(int(root.get(key, "-1")) != 0 for key in ("failures", "errors", "disabled")):
        raise ValueError("GTest reported failures, errors or disabled tests")
    return len(tests)


def execution_env():
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("GTEST_") or key.startswith("HGEMM_BATCHED_"):
            env.pop(key, None)
    return env


def execute(command, env, log, timeout):
    with log.open("x", encoding="utf-8") as output:
        try:
            completed = subprocess.run(
                command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False
            )
            return completed.returncode
        except subprocess.TimeoutExpired:
            output.write("\nRUNNER_TIMEOUT\n")
            return 124
        except OSError as error:
            output.write(f"\nRUNNER_LAUNCH_ERROR: {error}\n")
            return 127


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default = DIRECTORY.parents[2] / "build/test/gemm/hgemmBatched/hgemmBatched_test"
    parser.add_argument("--binary", type=Path, default=default)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=43200)
    parser.add_argument("--dry-run", action="store_true", help="Validate CSV and print the plan; no NPU call")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args()
    summary = {"passed": False, "expected_rows": EXPECTED_ROWS}
    try:
        csv_path, rows = read_cases()
        expected = expected_tests(rows)
        summary.update(
            csv=str(csv_path),
            csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            expected_tests=len(expected),
            accuracy_success=EXPECTED_ACCURACY_SUCCESS,
            performance=EXPECTED_PERF,
            negative=EXPECTED_NEGATIVE,
        )
        if args.dry_run:
            summary.update(status="planned", npu_calls=0)
            logging.info("%s", json.dumps(summary, indent=2))
            return 0
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        if not args.binary.is_file():
            raise FileNotFoundError(args.binary)
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
        args.output.chmod(0o750)
        xml = args.output / "results.xml"
        log = args.output / "run.log"
        command = [
            str(args.binary.resolve()),
            "--gtest_filter=*CsvDriven/*:" + ":".join(CONTRACT_TESTS),
            "--gtest_break_on_failure=0",
            "--gtest_output=xml:" + str(xml),
        ]
        env = execution_env()
        env["BLAS_DEVICE_ID"] = str(args.device)
        summary["command"] = command
        summary["returncode"] = execute(command, env, log, args.timeout)
        summary["executed_tests"] = validate_xml(xml, expected)
        if summary["returncode"] != 0:
            raise ValueError("Test process returned nonzero")
        summary["passed"] = True
    except (ValueError, OSError, ET.ParseError) as error:
        summary["first_failure"] = str(error)
    if not args.dry_run:
        (args.output / "self_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
