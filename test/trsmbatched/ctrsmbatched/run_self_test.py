#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check the arch22 official CSV, registered GTests and executed test results."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OFFICIAL = "Official/CtrsmBatchedCsvTest.CsvDriven/"
COUNT_TEST = "CtrsmBatchedCsvCoverage.CsvCaseCount"


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/test/trsmbatched/ctrsmbatched/ctrsmbatched_test")
    parser.add_argument("--csv", type=Path, default=HERE / "arch22/ctrsmbatched_test.csv")
    parser.add_argument("--cases", nargs="+", help="explicit official case IDs; omitted means the default cohort")
    parser.add_argument("--device", type=int, default=0, help="single visible device; logical device is always 0")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for logs, XML and summary")
    parser.add_argument("--timeout", type=int, default=3600)
    return parser


def device_environment(device):
    env = os.environ.copy()
    for key in ("GTEST_FILTER", "GTEST_OUTPUT", "GTEST_REPEAT", "GTEST_TOTAL_SHARDS", "GTEST_SHARD_INDEX"):
        env.pop(key, None)
    env.update(ASCEND_RT_VISIBLE_DEVICES=str(device), BLAS_DEVICE_ID="0", GTEST_COLOR="no")
    return env


def read_csv_cases(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "case_name" not in reader.fieldnames:
            raise ValueError("CSV must contain a case_name header")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV contains a malformed row")
    names = [row["case_name"] for row in rows]
    performance = {name for name in names if name.startswith("TC_PF_")}
    if len(names) != 1200 or len(set(names)) != 1200:
        raise ValueError("expected 1200 unique official CSV cases")
    if performance != {f"TC_PF_{number}" for number in range(1001, 1201)}:
        raise ValueError("expected exactly TC_PF_1001..1200 and 1000 precision cases")
    return names


def run_command(command, env, output, timeout, cwd):
    with output.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def registered_tests(args, env, output, names):
    log = output / "list-tests.log"
    command = [str(args.binary.resolve()), "--gtest_list_tests", "--gtest_filter=" + OFFICIAL + "*"]
    if run_command(command, env, log, args.timeout, output) != 0:
        raise ValueError("GTest listing failed; see list-tests.log")
    registered, suite = [], ""
    for line in log.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if not line.startswith(" ") and text.endswith("."):
            suite = text
        elif line.startswith("  ") and (suite + text).startswith(OFFICIAL):
            registered.append(suite + text)
    expected = {OFFICIAL + name for name in names}
    if len(registered) != 1200 or set(registered) != expected:
        raise ValueError("registered official GTests do not match the 1200 CSV IDs")
    return registered


def select_tests(registered, cases):
    selected = [OFFICIAL + name for name in cases] if cases else registered
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(registered):
        raise ValueError("selected cases must be distinct official CSV IDs")
    return selected


def read_test_result(path, expected):
    root = ET.parse(path).getroot()
    nodes = list(root.iter("testcase"))
    names = [node.attrib["classname"] + "." + node.attrib["name"] for node in nodes]
    failures = max(int(root.get("failures", "0")) + int(root.get("errors", "0")),
                   sum(node.find("failure") is not None or node.find("error") is not None for node in nodes))
    skipped = max(int(root.get("disabled", "0")), sum(node.find("skipped") is not None
                  or node.get("status") != "run" or node.get("result") != "completed" for node in nodes))
    passed = (int(root.attrib["tests"]) == len(expected) == len(names) == len(set(names))
              and set(names) == set(expected) and failures == 0 and skipped == 0)
    return dict(tests_ran=len(names), expected_tests=len(expected), failed=failures, skipped=skipped,
                missing_tests=sorted(set(expected) - set(names)), unexpected_tests=sorted(set(names) - set(expected)),
                passed=passed)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argument_parser(__doc__).parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, mode=0o750)
    summary = dict(status="FAIL", scope="selected" if args.cases else "all_official", expected_csv_cases=1200)
    try:
        names = read_csv_cases(args.csv)
        summary.update(csv_cases=len(names), precision_cases=1000, performance_cases=200)
        env = device_environment(args.device)
        registered = registered_tests(args, env, output, names)
        selected = select_tests(registered, args.cases)
        expected = [COUNT_TEST] + selected
        xml = output / "gtest.xml"
        env["GTEST_OUTPUT"] = "xml:" + str(xml)
        command = [str(args.binary.resolve()), "--gtest_filter=" + ":".join(expected)]
        summary.update(registered_cases=len(registered), selected_cases=len(selected), command=command)
        summary["exit_code"] = run_command(command, env, output / "run.log", args.timeout, output)
        summary.update(read_test_result(xml, expected))
        summary["status"] = "PASS" if summary["passed"] and summary["exit_code"] == 0 else "FAIL"
    except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
    (output / "self_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
