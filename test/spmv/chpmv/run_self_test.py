#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""执行 Chpmv 完整测试，核对 CSV、注册名单、XML 和实虚部精度记录。"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXPECTED_TESTS = 2504


def read_cases():
    with (HERE / "arch35/chpmv_test.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    names = [row["case_name"] for row in rows]
    if len(names) != 1200 or len(set(names)) != 1200 or not all(names):
        raise ValueError("CSV must contain 1200 unique nonempty case names")
    if sum(name.startswith("TC_PF_") for name in names) != 200:
        raise ValueError("CSV must contain exactly 200 performance cases")
    return rows


def clean_environment():
    env = os.environ.copy()
    for key in ("BLAS_TEST_CASES", "GTEST_FILTER", "GTEST_TOTAL_SHARDS", "GTEST_SHARD_INDEX",
                "GTEST_REPEAT", "GTEST_ALSO_RUN_DISABLED_TESTS", "CHPMV_PROFILE_REPEAT"):
        env.pop(key, None)
    return env


def run_logged(command, log, env, timeout):
    with log.open("w", encoding="utf-8") as output:
        # 独立进程组，超时时连同工具启动的测试进程一起结束。
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                import signal
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise


def listed_tests(text):
    suite, names = "", []
    for line in text.splitlines():
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        if not line[0].isspace() and value.endswith("."):
            suite = value
        elif line[0].isspace() and suite:
            names.append(suite + value)
    if not names or len(names) != len(set(names)):
        raise ValueError("GTest registration is empty or contains duplicates")
    return names


def inventory(binary, output, env, timeout):
    log = output / "registered.txt"
    if run_logged([str(binary), "--gtest_list_tests"], log, env, timeout) != 0:
        raise ValueError("GTest registration failed")
    names = listed_tests(log.read_text(encoding="utf-8", errors="replace"))
    official = [name.rsplit("/", 1)[-1] for name in names if name.rsplit("/", 1)[-1].startswith("TC_")]
    if Counter(official) != Counter(row["case_name"] for row in read_cases()):
        raise ValueError("Registered official cases do not match the CSV")
    if len(names) != EXPECTED_TESTS:
        raise ValueError(f"Expected {EXPECTED_TESTS} tests, registered {len(names)}")
    return names


def component_record(name, properties):
    n = int(properties["n"])
    expected = int(properties["expected_status"])
    if int(properties["actual_status"]) != expected:
        raise ValueError("Unexpected API status: " + name)
    if expected != 0:
        return "expected_negative"
    if n == 0:
        if properties.get("comparison") != "legal_empty_output":
            raise ValueError("Missing legal empty-output record: " + name)
        return "legal_empty"
    if n < 0:
        raise ValueError("Negative n unexpectedly succeeded: " + name)
    for component in ("real", "imag"):
        finite = int(properties.get(component + "_finite", "-1"))
        special = int(properties.get(component + "_special", "-1"))
        if finite < 0 or special < 0 or finite + special != n:
            raise ValueError("Incomplete output comparison: " + name + " " + component)
        if properties.get(component + "_verified") != "true":
            raise ValueError("Component comparison failed: " + name + " " + component)
    return "compared_nonempty"


def executed_records(path, expected):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    names = [item.get("classname", "") + "." + item.get("name", "") for item in cases]
    if int(root.get("tests", "-1")) != len(cases) or Counter(names) != Counter(expected):
        raise ValueError("Executed test names do not exactly match the expected list")
    records = []
    for name, item in zip(names, cases):
        if item.get("status") != "run" or item.get("result") != "completed":
            raise ValueError("Unexecuted or skipped test: " + name)
        if any(item.find(tag) is not None for tag in ("failure", "error", "skipped")):
            raise ValueError("Failed or skipped test: " + name)
        properties = {p.get("name"): p.get("value") for p in item.findall("properties/property")}
        category = "fixed_test"
        if ".CsvDriven/" in name:
            category = component_record(name, properties)
        elif name.startswith("ChpmvComparator.") or "framework_selfcheck" in properties:
            category = "framework_self_check"
        records.append({"name": name, "status": "PASS", "category": category, "properties": properties})
    return records


def file_identity(paths):
    result = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        result[str(resolved)] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return result


def verify_all(binary, output, timeout, summary):
    env = clean_environment()
    inputs = [binary, HERE / "arch35/chpmv_test.csv", Path(__file__)]
    inputs += [Path(env[name]) for name in ("CHPMV_NETLIB_CBLAS", "CHPMV_NETLIB_BLAS")]
    before = file_identity(inputs)
    names = inventory(binary, output, env, timeout)
    summary.update(csv_cases=1200, performance_shapes=200, registered=len(names), inputs=before)
    command = [str(binary), "--gtest_filter=*", "--gtest_output=xml:" + str(output / "results.xml")]
    summary["exit_code"] = run_logged(command, output / "run.log", env, timeout)
    if summary["exit_code"] != 0:
        raise ValueError("Test process returned a nonzero exit code")
    records = executed_records(output / "results.xml", names)
    with (output / "precision.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if before != file_identity(inputs):
        raise ValueError("Test inputs changed during execution")
    providers = [line for line in (output / "run.log").read_text(errors="replace").splitlines()
                 if line.startswith("NETLIB_CBLAS=")]
    if not providers:
        raise ValueError("Netlib provider verification record is missing")
    summary.update(status="PASS", executed=len(records), categories=dict(Counter(r["category"] for r in records)),
                   netlib_providers=providers, failed=0, skipped=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o750)
    summary = {"status": "FAIL", "binary": str(args.binary.resolve()), "timeout_seconds": args.timeout}
    started = time.monotonic()
    try:
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive")
        verify_all(args.binary.resolve(strict=True), output, args.timeout, summary)
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
    summary["wall_seconds"] = time.monotonic() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    raise SystemExit(main())
