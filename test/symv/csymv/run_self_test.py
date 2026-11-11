#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run CSYMV accuracy checks and audit CSV, baseline and executed GTest counts."""
import argparse
from collections import Counter
import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ARCH = Path(__file__).resolve().parent / "arch22"


def read_csv(name):
    with (ARCH / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def audit_csvs():
    rows = read_csv("csymv_test.csv")
    names = [row["case_name"] for row in rows]
    performance = [name for name in names if name.startswith("TC_PF_")]
    baselines = read_csv("gpu_baseline.csv")
    expected_ids = [f"csymv-{'base' if i < 5 else 'perf'}-{i:03d}" for i in range(200)]
    if len(names) != 1244 or len(set(names)) != 1244:
        raise ValueError("Expected 1244 unique CSV cases")
    if set(performance) != {f"TC_PF_{i}" for i in range(1001, 1201)}:
        raise ValueError("Expected all 200 performance CSV cases")
    if Counter(row["id"] for row in baselines) != Counter(expected_ids):
        raise ValueError("Expected 200 unique GPU baselines")
    return {"csv_cases": len(names), "accuracy_cases": len(names) - len(performance),
            "performance_cases": len(performance), "baseline_cases": len(baselines)}


def environment(device):
    env = os.environ.copy()
    # Explicit arguments own test selection, repetition and output.
    for key in list(env):
        if key.startswith("GTEST_"):
            env.pop(key)
    env["BLAS_DEVICE_ID"] = str(device)
    return env


def list_tests(binary, pattern, env):
    result = subprocess.run([str(binary), "--gtest_list_tests", f"--gtest_filter={pattern}"],
                            env=env, capture_output=True, text=True, timeout=120, check=True)
    tests, suite = [], ""
    for line in result.stdout.splitlines():
        value = line.split("#", 1)[0].strip()
        if value.endswith(".") and not line.startswith(" "):
            suite = value
        elif line.startswith("  ") and value and suite:
            tests.append(suite + value)
    if not tests or len(set(tests)) != len(tests):
        raise ValueError("Empty or duplicate GTest discovery")
    return tests


def check_mapping(tests, performance=False):
    names = [
        row["case_name"]
        for row in read_csv("csymv_test.csv")
        if performance or not row["case_name"].startswith("TC_PF_")
    ]
    actual = [name.rsplit("/", 1)[-1] for name in tests if ".CsvDriven/" in name]
    if Counter(actual) != Counter(names):
        raise ValueError("Registered GTests do not match CSV cases")


def check_xml(path, expected):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    actual = [case.attrib["classname"] + "." + case.attrib["name"] for case in cases]
    if Counter(actual) != Counter(expected) or int(root.get("tests", -1)) != len(expected):
        raise ValueError("Executed GTests do not match discovery")
    failures = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    incomplete = sum(case.get("status") != "run" or case.get("result") != "completed"
                     or case.find("skipped") is not None for case in cases)
    record = {"expected_tests": len(expected), "executed_tests": len(cases),
              "failures": failures, "incomplete_tests": incomplete}
    record["passed"] = not (failures or incomplete or any(int(root.get(key, -1)) != 0
                                                          for key in ("failures", "errors", "disabled")))
    return record


def run_suite(command, output, stem, env, timeout):
    with (output / f"{stem}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=int(os.environ.get("BLAS_DEVICE_ID", "0")))
    parser.add_argument("--include-performance", action="store_true")
    parser.add_argument("--timeout", type=int, default=14400)
    return parser.parse_args()


def main():
    args = parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "FAIL", "performance_included": args.include_performance}
    try:
        summary.update(audit_csvs())
        env = environment(args.device)
        if env.get("CSYMV_EXTERNAL_PROFILER") == "1" or "libmsopprof_injection.so" in env.get("LD_PRELOAD", ""):
            raise ValueError("Run self-test without an external profiler")
        pattern = "*" if args.include_performance else "*-*TC_PF_*"
        expected = list_tests(binary, pattern, env)
        check_mapping(expected, args.include_performance)
        xml = output / "self_test.xml"
        command = [str(binary), f"--gtest_filter={pattern}", f"--gtest_output=xml:{xml}"]
        summary["command"] = command
        xml.unlink(missing_ok=True)
        summary["returncode"] = run_suite(command, output, "self_test", env, args.timeout)
        summary.update(check_xml(xml, expected))
        if summary.get("returncode") != 0 or not summary.get("passed", False):
            raise ValueError("GTest failure or incomplete execution; see self_test.log")
        if args.include_performance:
            lines = (output / "self_test.log").read_text(encoding="utf-8").splitlines()
            perf = [
                dict(item.split("=", 1) for item in line.split() if "=" in item)
                for line in lines
                if line.startswith("[PERF] ")
            ]
            if Counter(row.get("case") for row in perf) != Counter(f"TC_PF_{i}" for i in range(1001, 1201)):
                raise ValueError("Missing or duplicate performance records")
            failed = [row["case"] for row in perf if row.get("verdict") != "PASS"]
            summary["performance_failures"] = failed
            if failed:
                raise ValueError("Performance thresholds failed")
        summary["status"] = "PASS"
    except Exception as error:
        summary["error"] = str(error)
    (output / "self_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
